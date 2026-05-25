import os
import re
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

# ── Download models from Google Drive if not present ────────
import gdown

DRIVE_FILES = {
    'resnet50_crisis_connect.pth': '1QY40L6gDE6kYVv09ylAwh6b1JY0T5AaM',
    'mlp_classifier.pkl':          '11Mh-kmGgQqqo3H-stb-CEHC47gvT-Ucn',
    'tfidf_vectorizer.pkl':        '1Y6q_w9eGrdCIHxJ8iZfNCmCOP7OtH52I',
    'label_encoder.pkl':           '1CsUHzx51EBVuqUzIBkxUvilE3NOwqL_q',
    'fusion.py':                   '1dRXQjzi0wfONKObY8OT1aVO6bUEqTd5j',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_models():
    for fname, file_id in DRIVE_FILES.items():
        dest = os.path.join(BASE_DIR, fname)
        if not os.path.exists(dest):
            url = f'https://drive.google.com/uc?id={file_id}'
            st.info(f'Downloading {fname}...')
            gdown.download(url, dest, quiet=False)

download_models()

# ── Constants ────────────────────────────────────────────────
CLASSES    = ['earthquake', 'flood', 'fire', 'traffic_incident']
CNN_WEIGHT = 0.70
MLP_WEIGHT = 0.30
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_EMOJI = {
    'earthquake':       '🏚️',
    'flood':            '🌊',
    'fire':             '🔥',
    'traffic_incident': '🚗'
}

SEVERITY_COLOR = {
    'High':   '#FF4444',
    'Medium': '#FFA500',
    'Low':    '#44BB44'
}

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title='Crisis Connect',
    page_icon='🚨',
    layout='centered'
)

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    # CNN
    cnn = models.resnet50(pretrained=False)
    cnn.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(cnn.fc.in_features, 4)
    )
    ckpt = torch.load(
        os.path.join(BASE_DIR, 'resnet50_crisis_connect.pth'),
        map_location=device
    )
    cnn.load_state_dict(ckpt['model_state_dict'])
    cnn = cnn.to(device)
    cnn.eval()
    cnn_idx_to_class = {v: k for k, v in ckpt['class_to_idx'].items()}

    # MLP + tools
    mlp = joblib.load(os.path.join(BASE_DIR, 'mlp_classifier.pkl'))
    vec = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
    le  = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

    return cnn, cnn_idx_to_class, mlp, vec, le

# ── Preprocessing ────────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def clean_text(text):
    sw = set(stopwords.words('english')) - {
        'no','not','very','too','above','below','near','under','over'
    }
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    return ' '.join([t for t in text.split() if t not in sw and len(t) > 2])

def get_severity(confidence, predicted_class):
    HIGH_RISK = ['earthquake', 'flood']
    if confidence >= 0.75:
        return 'High' if predicted_class in HIGH_RISK else 'Medium'
    elif confidence >= 0.50:
        return 'Medium'
    return 'Low'

def predict(image, text, cnn, cnn_idx_to_class, mlp, vec, le):
    # CNN
    tensor = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_raw = torch.softmax(cnn(tensor), dim=1)[0].cpu().numpy()
    cnn_probs = np.zeros(len(CLASSES))
    for idx, cls in cnn_idx_to_class.items():
        if cls in CLASSES:
            cnn_probs[CLASSES.index(cls)] = cnn_raw[idx]

    # MLP
    mlp_raw = mlp.predict_proba(vec.transform([clean_text(text)]))[0]
    mlp_probs = np.zeros(len(CLASSES))
    for i, cls in enumerate(le.classes_):
        if cls in CLASSES:
            mlp_probs[CLASSES.index(cls)] = mlp_raw[i]

    # Fuse
    fused      = CNN_WEIGHT * cnn_probs + MLP_WEIGHT * mlp_probs
    pred_idx   = int(np.argmax(fused))
    pred_class = CLASSES[pred_idx]
    confidence = float(fused[pred_idx])

    return pred_class, confidence, get_severity(confidence, pred_class), cnn_probs, mlp_probs, fused

# ── UI ───────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#CC0000;'>🚨 Crisis Connect</h1>
    <p style='text-align:center; font-size:16px; color:#666;'>AI-Powered Disaster Incident Classifier</p>
    <p style='text-align:center; font-size:13px; color:#888;'>
    Upload a disaster image and describe what you see.<br>
    The AI will classify the incident and estimate severity.
    </p>
    <hr/>
""", unsafe_allow_html=True)

# Load models
with st.spinner('Loading AI models... please wait'):
    cnn, cnn_idx_to_class, mlp, vec, le = load_all_models()
st.success('Models loaded. Ready to classify.')

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader('📷 Upload Disaster Image')
    uploaded = st.file_uploader(
        'Upload a photo of the incident',
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption='Uploaded image', use_column_width=True)

with col2:
    st.subheader('📝 Describe the Incident')
    text_input = st.text_area(
        'Describe what you see (max 200 characters)',
        placeholder='e.g. Water rising fast on main road, cars stranded near bridge',
        max_chars=200,
        height=120
    )
    st.caption(f'{len(text_input)}/200 characters')

st.markdown('<br/>', unsafe_allow_html=True)

# Classify button
if st.button('🔍 Classify Incident', use_container_width=True, type='primary'):
    if uploaded is None:
        st.error('Please upload an image first.')
    elif len(text_input.strip()) < 5:
        st.error('Please enter a description of at least 5 characters.')
    else:
        with st.spinner('Analysing...'):
            pred_class, confidence, severity, cnn_probs, mlp_probs, fused = predict(
                image, text_input, cnn, cnn_idx_to_class, mlp, vec, le
            )

        st.markdown('<hr/>', unsafe_allow_html=True)
        st.subheader('🎯 Classification Result')

        sev_color = SEVERITY_COLOR[severity]
        emoji     = CLASS_EMOJI.get(pred_class, '⚠️')

        st.markdown(f"""
            <div style='background:#f8f8f8; border-radius:12px; padding:20px;
                        text-align:center; border: 2px solid {sev_color};'>
                <h2>{emoji} {pred_class.replace('_', ' ').title()}</h2>
                <p style='font-size:18px;'>Confidence: <b>{confidence*100:.1f}%</b></p>
                <p style='font-size:20px; color:{sev_color}; font-weight:bold;'>
                    Severity: {severity}
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<br/>', unsafe_allow_html=True)

        st.subheader('📊 Class Probabilities')
        tabs = st.tabs(['Fused (Final)', 'CNN Only', 'Text Only'])

        for tab, probs, label in zip(
            tabs,
            [fused, cnn_probs, mlp_probs],
            ['Fused (CNN 70% + Text 30%)', 'CNN Image Model', 'Text MLP Model']
        ):
            with tab:
                st.caption(label)
                for i, cls in enumerate(CLASSES):
                    e = CLASS_EMOJI.get(cls, '')
                    st.progress(
                        float(probs[i]),
                        text=f"{e} {cls.replace('_', ' ').title()}: {probs[i]*100:.1f}%"
                    )

        st.markdown('<br/>', unsafe_allow_html=True)
        st.info('Powered by ResNet-50 (CNN) + TF-IDF MLP (Text) — Crisis Connect AI Lab Spring 2026')

# Footer
st.markdown('<hr/>', unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; font-size:12px; color:#aaa;'>
    Crisis Connect | CSC-233 AI Lab | Spring 2026 | Beaconhouse National University
    </p>
""", unsafe_allow_html=True)
