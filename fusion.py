
import os, re, numpy as np, torch, joblib
from torchvision import models, transforms
from torch import nn
from PIL import Image
from nltk.corpus import stopwords

CLASSES     = ["earthquake", "flood", "fire", "traffic_incident"]
CNN_WEIGHT  = 0.70
MLP_WEIGHT  = 0.30
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(cnn_path, mlp_path, vec_path, enc_path):
    # CNN
    cnn = models.resnet50(pretrained=False)
    cnn.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(cnn.fc.in_features, 4))
    ckpt = torch.load(cnn_path, map_location=device)
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn = cnn.to(device)
    cnn.eval()
    cnn_class_to_idx = ckpt["class_to_idx"]
    cnn_idx_to_class = {v: k for k, v in cnn_class_to_idx.items()}
    # MLP
    mlp = joblib.load(mlp_path)
    vec = joblib.load(vec_path)
    le  = joblib.load(enc_path)
    return cnn, cnn_idx_to_class, mlp, vec, le

img_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def clean_text(text):
    import nltk
    nltk.download("stopwords", quiet=True)
    sw = set(stopwords.words("english")) - {"no","not","very","too","above","below","near","under","over"}
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join([t for t in text.split() if t not in sw and len(t) > 2])

def get_severity(confidence, predicted_class):
    HIGH_RISK = ["earthquake", "flood"]
    if confidence >= 0.75:
        return "High" if predicted_class in HIGH_RISK else "Medium"
    elif confidence >= 0.50:
        return "Medium"
    return "Low"

def fuse_predict(image, text, cnn, cnn_idx_to_class, mlp, vec, le):
    # CNN
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = Image.fromarray(image).convert("RGB")
    tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_probs_raw = torch.softmax(cnn(tensor), dim=1)[0].cpu().numpy()
    cnn_probs = np.zeros(len(CLASSES))
    for idx, cls in cnn_idx_to_class.items():
        if cls in CLASSES:
            cnn_probs[CLASSES.index(cls)] = cnn_probs_raw[idx]
    # MLP
    mlp_probs_raw = mlp.predict_proba(vec.transform([clean_text(text)]))[0]
    mlp_probs = np.zeros(len(CLASSES))
    for i, cls in enumerate(le.classes_):
        if cls in CLASSES:
            mlp_probs[CLASSES.index(cls)] = mlp_probs_raw[i]
    # Fuse
    fused      = CNN_WEIGHT * cnn_probs + MLP_WEIGHT * mlp_probs
    pred_idx   = int(np.argmax(fused))
    pred_class = CLASSES[pred_idx]
    confidence = float(fused[pred_idx])
    return {
        "predicted_class": pred_class,
        "confidence":      confidence,
        "severity":        get_severity(confidence, pred_class),
        "cnn_probs":       cnn_probs.tolist(),
        "mlp_probs":       mlp_probs.tolist(),
        "fused_probs":     fused.tolist()
    }
