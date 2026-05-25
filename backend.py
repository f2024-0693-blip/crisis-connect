"""
Crisis Connect — FastAPI Backend
CSC-233 Artificial Intelligence Lab | Spring 2026
Beaconhouse National University

Run locally:
    pip install fastapi uvicorn python-multipart
    uvicorn backend:app --reload
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import re
import io
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

app = FastAPI(
    title="Crisis Connect API",
    description="AI-powered disaster incident classification API",
    version="1.0.0"
)

# Allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ────────────────────────────────────────────────
CLASSES    = ['earthquake', 'flood', 'fire', 'traffic_incident']
CNN_WEIGHT = 0.70
MLP_WEIGHT = 0.30
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Response Model ───────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence:      float
    severity:        str
    cnn_probs:       dict
    mlp_probs:       dict
    fused_probs:     dict

# ── Load Models ──────────────────────────────────────────────
def load_models():
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

    mlp = joblib.load(os.path.join(BASE_DIR, 'mlp_classifier.pkl'))
    vec = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
    le  = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

    return cnn, cnn_idx_to_class, mlp, vec, le

# Load on startup
cnn, cnn_idx_to_class, mlp, vec, le = load_models()

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

def run_fusion(image: Image.Image, text: str):
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

    return {
        'predicted_class': pred_class,
        'confidence':      round(confidence, 4),
        'severity':        get_severity(confidence, pred_class),
        'cnn_probs':       {CLASSES[i]: round(float(cnn_probs[i]), 4) for i in range(len(CLASSES))},
        'mlp_probs':       {CLASSES[i]: round(float(mlp_probs[i]), 4) for i in range(len(CLASSES))},
        'fused_probs':     {CLASSES[i]: round(float(fused[i]), 4) for i in range(len(CLASSES))},
    }

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Crisis Connect API is running",
        "version": "1.0.0",
        "classes": CLASSES,
        "docs":    "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Disaster image (jpg/png)"),
    text:  str        = Form(..., description="Text description of the incident")
):
    """
    Classify a disaster incident from an image and text description.

    - **image**: Upload a JPG or PNG photo of the disaster scene
    - **text**: Short description of what you see (max 200 characters)

    Returns predicted class, confidence score, severity level,
    and probability breakdown from CNN, MLP, and fused model.
    """
    # Read image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # Run fusion
    result = run_fusion(img, text)
    return result

@app.post("/predict/text-only")
async def predict_text_only(text: str = Form(...)):
    """Classify based on text description only (MLP model)."""
    cleaned  = clean_text(text)
    vec_feat = vec.transform([cleaned])
    probs    = mlp.predict_proba(vec_feat)[0]
    pred_idx = int(np.argmax(probs))
    pred_cls = le.inverse_transform([pred_idx])[0]
    return {
        "predicted_class": pred_cls,
        "confidence":      round(float(probs[pred_idx]), 4),
        "all_probs":       {cls: round(float(probs[i]), 4) for i, cls in enumerate(le.classes_)}
    }
