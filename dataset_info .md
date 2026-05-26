# Crisis Connect — Dataset Information

## Overview

The Crisis Connect dataset consists of disaster images and text/audio descriptions across 4 classes.

## Classes

| Class | Description | Source |
|---|---|---|
| Earthquake | Building damage, rubble, cracked structures | AIDER + Kaggle |
| Flood | Submerged roads, flooded buildings, rising water | AIDER + Kaggle |
| Fire | Burning buildings, smoke, flames | AIDER + Kaggle |
| Traffic Incident | Road accidents, overturned vehicles, collisions | Kaggle |

## Image Dataset Statistics

| Split | Earthquake | Flood | Fire | Traffic Incident | Total |
|---|---|---|---|---|---|
| Train | 397 | 482 | 438 | 339 | 1656 |
| Val | 104 | 135 | 124 | 72 | 435 |
| Test | 122 | 130 | 120 | 74 | 446 |
| **Total** | **623** | **747** | **682** | **485** | **2537** |

## Text and Audio Dataset

| Source | Count | Language |
|---|---|---|
| Synthetic text descriptions | 100 | English |
| Audio recordings (Whisper transcribed) | 24 | Urdu / code-switched |
| **Total** | **124** | — |

## Sources

- **AIDER Dataset** — https://zenodo.org/records/3888300
- **Kaggle Disaster Recognition** — https://www.kaggle.com/datasets/mikolajbabula/disaster-recognition

## Preprocessing Steps

1. Corrupt image removal using PIL verify
2. Duplicate removal using perceptual hashing (threshold = 5)
3. Resize all images to 224x224 pixels
4. Class balancing
5. Split 70% train / 15% val / 15% test
6. Audio: Whisper transcription + Google Translate (Urdu → English)

## Image Specifications

- Format: JPEG
- Size: 224 x 224 pixels
- Channels: RGB
- Normalization: ImageNet mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225]
