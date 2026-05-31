# 🚨 Crisis Connect

> AI-powered disaster classification system that detects crisis type and severity from images and text descriptions.

---

## 🌐 Live Application

| Service | URL |
|---|---|
| Frontend (Streamlit App) | https://crisis-connect-vjmeydb54qdgg7yxm3xfes.streamlit.app/ |
| GitHub Repository | https://github.com/yourusername/crisis-connect |

---

## 📌 About

**Crisis Connect** is an AI-powered web application that classifies disaster situations from photos and text descriptions, providing instant crisis type detection and severity assessment. Built for emergency responders and disaster management agencies who need fast, automated triage of incoming crisis reports — especially in a Pakistani context where response speed is critical.

---

## ✨ Features

- 🔍 AI disaster type detection (4 conditions)
- 📊 Confidence score + severity level (Low / Medium / High)
- 🖼️ Image-based classification using deep learning (CNN)
- 📝 Text-based classification using NLP (TF-IDF + MLP)
- 🔗 Multimodal fusion of image + text for maximum accuracy
- 📱 Fully responsive Streamlit web application

---

## 🤖 AI Models

### Image Classifier (Phase 2)

| Property | Details |
|---|---|
| Architecture | ResNet-50 (pretrained ImageNet) |
| Overall Accuracy | **91.03%** |
| Classes | 4 disaster types |
| Training Images | ~1,776 |
| Val / Test Images | ~381 each |
| Framework | PyTorch |

### Text Classifier (Phase 3b)

| Property | Details |
|---|---|
| Architecture | TF-IDF + MLP |
| Overall Accuracy | **89.47%** (after audio data) |
| Features | 3000 TF-IDF features (unigrams + bigrams) |
| MLP Layers | 3000 → 512 → 256 → 4 |
| Framework | scikit-learn |

### Fusion Layer (Phase 4)

| Property | Details |
|---|---|
| Method | Weighted average (CNN 70% + MLP 30%) |
| **Final Accuracy** | **95.07%** |
| Improvement over CNN alone | +4.04 percentage points |

---

## 📊 Detected Conditions

| Class | CNN Accuracy |
|---|---|
| Earthquake | 93.4% |
| Flood | 95.4% |
| Fire | 95.8% |
| Traffic Incident | 95.9% |

---

## 🗂️ Dataset

| Stage | Images |
|---|---|
| 📁 Original Raw Dataset | 4,386 |
| 🧹 Cleaned Dataset | 2,537 |
| ✂️ Split Dataset (train/val/test) | 2,537 |
| 🤖 ResNet-50 Model Weights (.pth)  | [Download](https://drive.google.com/file/d/1QY40L6gDE6kYVv09ylAwh6b1JY0T5AaM/view?usp=sharing) |

### Dataset Split

| Split | Images | Percentage |
|---|---|---|
| Train | ~1,776 | 70% |
| Validation | ~381 | 15% |
| Test | ~381 | 15% |

### Dataset Classes

| Class | Images (after cleaning) |
|---|---|
| Earthquake | ~635 |
| Flood | ~635 |
| Fire | ~635 |
| Traffic Incident | ~632 |
| **TOTAL** | **2,537** |

---

## 🏗️ System Architecture

```
User → Streamlit Frontend → Fusion Layer → ResNet-50 (CNN) + TF-IDF MLP → Crisis Class + Severity
```

---

## 🛠️ Tech Stack

### Frontend
- Streamlit
- Python

### Backend
- FastAPI
- PyTorch
- scikit-learn

### Machine Learning
- PyTorch
- TorchVision
- ResNet-50 (transfer learning)
- TF-IDF Vectorizer
- MLP Classifier
- OpenAI Whisper (audio transcription)
- NumPy / Pandas

### Deployment
- Streamlit Cloud (frontend + app)
- Google Drive (model weights via gdown)

---

## 📁 Project Structure

```
crisis-connect/
│
├── notebooks/                          # Jupyter Notebooks (Google Colab)
│   ├── Phase1_Dataset.ipynb            # Data cleaning, dedup, augmentation, splitting
│   ├── Phase2_CNN.ipynb                # ResNet-50 training, evaluation, confusion matrix
│   ├── Phase3a_Audio.ipynb             # Whisper STT, Urdu translation, TF-IDF prep
│   ├── Phase3b_NLP.ipynb               # MLP training, evaluation, per-class report
│   ├── Phase4_Fusion.ipynb             # Weighted fusion, severity logic, final evaluation
│   └── Phase5_App.ipynb                # Streamlit deployment setup
│
├── app.py                              # Main Streamlit frontend application
├── backend.py                          # FastAPI REST API (/predict, /predict/text-only)
├── fusion.py                           # Reusable fusion module (CNN + MLP)
├── requirements.txt                    # Python dependencies
│
├── models/                             # Trained model weights
│   ├── resnet50_crisis_connect.pth     # ResNet-50 weights (~90MB, Google Drive)
│   ├── mlp_classifier.pkl              # Trained MLP model
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
│   └── label_encoder.pkl              # Class label encoder
│
├── datasets/                           # Dataset folders
│   └── crisis_connect_split/           # train / val / test split
│       ├── train/
│       ├── val/
│       └── test/
│
└── README.md                           # Project documentation
```

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Check API status |
| POST | /predict | Upload image + text → get crisis classification |
| POST | /predict/text-only | Text-only crisis classification |

---

## ⚙️ Run Locally

### Backend (FastAPI)

```bash
cd crisis-connect
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend:app --reload --port 8000
```

### Frontend (Streamlit)

```bash
streamlit run app.py
```

---

## 📈 Accuracy Summary

| Model | Accuracy |
|---|---|
| ResNet-50 CNN (Phase 2) | 91.03% |
| TF-IDF + MLP — before audio (Phase 3b) | 73.33% |
| TF-IDF + MLP — after audio (Phase 3b) | 89.47% |
| **Fusion Layer — Final (Phase 4)** | **95.07%** |

---

## 👥 Team

| Name | Student ID | Role |
|---|---|---|
| Maleeha Fatima | F2024-0693 | CNN Model (Phase 2) + Streamlit App (Phase 5) |
| Imman Aamir | F2024-0641 | Audio & Text Preprocessing (Phase 3a) |
| Ezzah Ali | F2024-0705 | NLP Text Classifier (Phase 3b) |
| Syed Hudair Shah Bukhari | F2024-0788 | Fusion Layer + Severity (Phase 4) |
| Abdulraheem Kashif | F2024-0667 | Dataset Preparation (Phase 1) |

---

## 🏫 Course Info

**CSC-233 — Artificial Intelligence Lab**
Beaconhouse National University | Spring 2026

