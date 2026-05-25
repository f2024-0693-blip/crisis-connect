# Crisis Connect — Dataset Information

## Overview

The Crisis Connect dataset consists of disaster images across 4 classes collected and processed for the AI Lab project.

## Classes

| Class | Description | Source |
|---|---|---|
| Earthquake | Building damage, rubble, cracked structures | AIDER + Kaggle |
| Flood | Submerged roads, flooded buildings, rising water | AIDER + Kaggle |
| Fire | Burning buildings, smoke, flames | AIDER + Kaggle |
| Traffic Incident | Road accidents, overturned vehicles, collisions | Kaggle |

## Dataset Statistics

| Split | Earthquake | Flood | Fire | Traffic Incident | Total |
|---|---|---|---|---|---|
| Train | 397 | 482 | 438 | 339 | 1656 |
| Val | 104 | 135 | 124 | 72 | 435 |
| Test | 122 | 130 | 120 | 74 | 446 |
| **Total** | **623** | **747** | **682** | **485** | **2537** |

## Sources

- **AIDER Dataset** — Aerial Image Dataset for Emergency Response
  - Source: https://zenodo.org/records/3888300
  - Classes used: Flood, Fire, Collapsed Building (mapped to Earthquake)

- **Kaggle Disaster Recognition Dataset**
  - Source: https://www.kaggle.com/datasets/mikolajbabula/disaster-recognition
  - Classes used: Flood, Earthquake, Wildfire (mapped to Fire), Traffic

## Preprocessing Steps

1. Corrupt image removal using PIL verify
2. Duplicate removal using perceptual hashing (threshold = 5)
3. Resize all images to 224x224 pixels (required for ResNet-50)
4. Class balancing — capped at equal distribution
5. Split 70% train / 15% val / 15% test using splitfolders

## Image Specifications

- Format: JPEG
- Size: 224 x 224 pixels
- Channels: RGB
- Normalization: ImageNet mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225]

## Text Descriptions

- 100 synthetic text descriptions (25 per class)
- Written in English
- Maximum 200 characters each
- Used for TF-IDF vectorisation and MLP classifier training

## Note on Dataset Size

The full processed dataset (2537 images) is too large to host on GitHub.
The dataset is available on Google Drive upon request.

Processing notebooks are available in this repository:
- `Phase1_Dataset.ipynb` — Full data collection, cleaning, and splitting pipeline
