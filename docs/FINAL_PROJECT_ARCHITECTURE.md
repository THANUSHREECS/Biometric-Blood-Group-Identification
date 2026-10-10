# FINAL PROJECT ARCHITECTURE
## Biometric Blood Group Identification System

**Document Type:** System Architecture Reference
**Project:** B.E. Final Year Major Project — Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Model Status:** FROZEN — Champion Ensemble V2 (85.41% locked test accuracy)

---

## 1. System Overview

The Biometric Blood Group Identification System is an end-to-end AI-based research prototype that classifies human ABO/Rh blood groups from optical fingerprint impressions.

```
Physical Fingerprint (Mantra MFS100) OR Uploaded Image
        ↓
  Image Acquisition Layer
        ↓
  Centralized Preprocessing Pipeline (src/preprocessing.py)
        ↓
  Champion Ensemble V2 Inference Engine (src/predictor.py)
        ↓
  ABO/Rh Analytical Decomposition
        ↓
  Confidence Threshold Evaluation (>= 60%)
        ↓
  SQLite Audit Record (src/database.py)
        ↓
  Clinical PDF Report (src/report_generator.py)
        ↓
  Web Dashboard Response (Flask REST API)
```

---

## 2. Complete Component Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER (Browser)                              │
│  templates/index.html — Clinical Dashboard                                   │
│  templates/history.html — Audit History View                                 │
│  static/js/app.js — UI Logic & API calls                                    │
│  static/css/style.css — Healthcare-themed responsive design                  │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ HTTP REST (port 5000)
┌─────────────────────────────────▼────────────────────────────────────────────┐
│                        FLASK BACKEND (app.py)                                │
│  GET  /                 — Dashboard HTML                                     │
│  GET  /history          — History HTML                                       │
│  GET  /api/status       — System health (model + scanner status)             │
│  POST /api/upload       — Image file upload → prediction                     │
│  POST /api/capture      — MFS100 single capture → prediction                 │
│  POST /api/capture/consensus — 3-scan LEFT THUMB consensus                  │
│  GET  /api/history      — Prediction audit records (JSON)                    │
│  GET  /api/report/<id>  — Clinical PDF download                              │
│  POST /api/scanner/mode — Toggle hardware/mock mode                          │
│  GET  /api/scanner/status — Scanner + bridge connectivity                    │
└──────┬──────────────────────┬──────────────────────┬──────────────────────┬──┘
       │                      │                      │                      │
┌──────▼──────┐  ┌───────────▼────────┐  ┌──────────▼────────┐  ┌─────────▼───┐
│ src/        │  │ src/predictor.py   │  │ src/database.py   │  │ src/report_ │
│ mfs100/     │  │ BloodGroupPredictor│  │ SQLite Audit DB   │  │ generator.py│
│ scanner.py  │  │                    │  │                   │  │ ReportLab   │
│ MFS100      │  │ Champion Ensemble  │  │ predictions table │  │ PDF engine  │
│ Adapter     │  │ V2 Inference       │  │ CRUD operations   │  │             │
└──────┬──────┘  └───────────┬────────┘  └──────────────────┘  └─────────────┘
       │                      │
┌──────▼──────┐  ┌────────────▼───────────────────────────────────────────┐
│ src/mfs100_ │  │ src/preprocessing.py                                    │
│ bridge/     │  │ Centralized Image Preprocessing Pipeline               │
│ bridge.py   │  │ (shared by training, upload, MFS100, mock)             │
│ 32-bit HTTP │  └────────────────────────────────────────────────────────┘
│ bridge      │
└─────────────┘
```

---

## 3. MFS100 Hardware Architecture

### 3.1 The Cross-Architecture Problem

The Mantra MFS100 driver DLL (`MFS100Dll.dll`) is a 32-bit (x86) Windows native library.
Python-based AI/ML environments run as 64-bit (x64) processes.

```
Windows 64-bit Python (AI/ML environment)
    ├── TensorFlow 2.21.0 (64-bit)
    ├── Flask (64-bit)
    └── CANNOT load 32-bit DLLs directly (WinError 193)
```

### 3.2 32-bit Bridge Solution

A dedicated lightweight 32-bit Python bridge process (`src/mfs100_bridge/bridge.py`) runs under a separate 32-bit Python installation and serves as an HTTP intermediary:

```
┌─────────────────────────────────┐       ┌──────────────────────────────────┐
│   64-bit Flask Backend          │       │   32-bit MFS100 Bridge           │
│   (Python 3.10.11 x64)         │       │   (C:\Python310-32\python.exe)   │
│                                 │       │                                   │
│   MFS100Scanner.capture() ────► │ HTTP  │ ◄─── ctypes.WinDLL(MFS100Dll)   │
│   (src/mfs100/scanner.py)      │ :5100 │       Physical MFS100 USB SDK    │
│                                 │◄───── │───►  scanner.capture()           │
│   Returns: image bytes +        │       │      quality, nfiq, image bytes  │
│   quality_meta (nfiq, score)    │       │                                   │
└─────────────────────────────────┘       └──────────────────────────────────┘
                                                        │
                                          ┌─────────────▼──────────────────┐
                                          │   Mantra MFS100 USB Hardware    │
                                          │   Optical platen: 316x354 px    │
                                          │   500 DPI optical sensor        │
                                          │   SDK: MFS100Dll.dll (32-bit)   │
                                          └────────────────────────────────┘
```

### 3.3 Bridge Startup

```bash
# Terminal 1: Start 32-bit bridge
C:\Python310-32\python.exe -m src.mfs100_bridge.bridge

# Terminal 2: Start 64-bit Flask backend
python app.py
```

### 3.4 Mock Mode

When `MFS100_MOCK=true` (default) or physical hardware is unavailable:
- `MFS100Scanner` operates in mock mode.
- Cycles through authentic fingerprint samples from `data/sample/`.
- Uses identical preprocessing, inference, and reporting paths.
- `finger_used` is set to `None` (not LEFT_THUMB) in mock captures.
- DB label: `MFS100 (Mock)` instead of `MFS100`.

---

## 4. Dataset Architecture

```
dataset_blood_group/          ← Raw source fingerprint dataset
data/
  raw/                        ← Unmodified raw images (6,000 total)
  processed/                  ← 5,985 cleaned images (15 duplicates quarantined)
  quarantine/
    duplicates/               ← 15 quarantined SHA-256 duplicate images
    quarantine_manifest.json  ← Full quarantine audit map
  sample/                     ← 8 reference images (one per class, for mock mode)
  captured/                   ← Runtime: MFS100 and upload captures stored here
  blood_group.db              ← SQLite prediction audit database
```

**Dataset splits (stratified, leakage-free):**

| Split | Images | Purpose |
|---|---|---|
| Training | 4,189 | Model training |
| Validation | 898 | Hyperparameter tuning and intermediate evaluation |
| Test (LOCKED) | 898 | Official one-time benchmark evaluation — NOT re-accessed |

---

## 5. Preprocessing Pipeline Architecture

### 5.1 Upload Mode (source="upload")

```
Input: File path / bytes / PIL
    ↓ load_image_to_pil()
    ↓ composite_alpha_on_white()    ← RGBA transparent borders → white (255,255,255)
    ↓ resize_to_target(224×224)     ← Lanczos resampling
    ↓ Per-architecture preprocessing:
       ResNet50: ImageEnhance.Contrast(×1.2) + resnet50.preprocess_input
       VGG16:    vgg16.preprocess_input
       ENB0:     efficientnet.preprocess_input
    ↓ Batch tensors → ensemble inference
```

### 5.2 MFS100 Hardware Mode (source="mfs100")

```
Input: Raw MFS100 capture (316×354, NFIQ quality check)
    ↓ load_image_to_pil()
    ↓ composite_alpha_on_white()
    ↓ crop_mfs100_foreground()      ← Hardware foreground crop (removes sensor margins)
       4 safeguards: blank guard, region check, size floor, margin presence check
    ↓ enhance_mfs100_ridge_contrast()
       CLAHE (clipLimit=2.5, 8×8) + percentile stretch + bilateral filter
    ↓ resize_to_target(224×224)
    ↓ Per-architecture preprocessing (same as upload above)
    ↓ Batch tensors → ensemble inference
```

---

## 6. Champion Ensemble V2 Inference Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Champion Ensemble V2                                  │
│              Locked Test Accuracy: 85.41% (767/898)                     │
│                                                                          │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┐  │
│  │ ResNet50 (Improved)  │  │      VGG16           │  │ EfficientNetB0│  │
│  │ 277 MB checkpoint    │  │ 164 MB checkpoint    │  │ 41 MB checkpt │  │
│  │ Weight: 0.58 (58%)  │  │ Weight: 0.38 (38%)  │  │ Weight: 0.04  │  │
│  │ 1.2x contrast        │  │ Native VGG16         │  │ Native ENB0   │  │
│  │ preprocessing        │  │ preprocessing        │  │ preprocessing │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └───────┬───────┘  │
│             │                         │                       │          │
│  P_resnet[8] × 0.58   +  P_vgg[8] × 0.38   +  P_eff[8] × 0.04         │
│             └─────────────────────────┴───────────────────────┘          │
│                                       ↓                                  │
│              Ensemble_probs[8] = weighted soft-vote average              │
│                                       ↓                                  │
│              predicted_class = argmax(Ensemble_probs)                   │
│              confidence = max(Ensemble_probs)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Post-inference decision layers:**

```
confidence >= 0.60 → status = NORMAL
confidence <  0.60 → status = LOW_CONFIDENCE (advisory + recapture recommendation)

ABO/Rh Decomposition (analytics only — does NOT change prediction):
  P(A)  = P(A+) + P(A-)
  P(B)  = P(B+) + P(B-)
  P(AB) = P(AB+) + P(AB-)
  P(O)  = P(O+) + P(O-)
  top_abo = argmax of above

Rh Uncertainty Advisory (does NOT change prediction):
  rh_margin < 0.20  → advisory flag
  confidence < 0.65 → advisory flag
  predicted_Rh == "-" → advisory flag (weak ridge-Rh correlation)
```

---

## 7. Database Architecture

**File:** `data/blood_group.db` (SQLite)

**Table: predictions**

| Column | Type | Description |
|---|---|---|
| id | INTEGER PRIMARY KEY | Auto-increment record ID |
| timestamp | TEXT | ISO 8601 datetime of prediction |
| input_type | TEXT | 'upload', 'MFS100', 'MFS100 Consensus - Left Thumb (3 Impressions)', 'MFS100 (Mock)' |
| image_path | TEXT | Filename of stored fingerprint image |
| predicted_group | TEXT | Predicted blood group (e.g., 'O+') |
| confidence | REAL | Float 0.0–1.0 |
| status | TEXT | 'NORMAL' or 'LOW_CONFIDENCE' |
| notes | TEXT | Model message / advisory text |
| finger_used | TEXT | 'LEFT_THUMB' for hardware; NULL for upload/mock |

**Migration safety:** `finger_used` was added via non-destructive `ALTER TABLE` — existing records retain `NULL`.

---

## 8. PDF Report Architecture

**Generator:** `src/report_generator.py` (ReportLab)
**Output:** `reports/blood_group_report_<id>_<timestamp>.pdf`

**Report structure:**
1. Header banner — project title
2. Medical disclaimer (mandatory red alert box)
3. Examination & audit details table (Report ID, timestamp, input source, confidence, model name, LEFT THUMB row if hardware)
4. Biometric impression + predicted blood group card (large 32pt display)
5. ABO group, Rh factor, confidence metadata
6. Rh Uncertainty Advisory (amber banner, conditional)
7. 8-class probability distribution table
8. Footer

---

## 9. Three Operational Modes

| Mode | Trigger | Source | finger_used | Preprocessing | PDF Label |
|---|---|---|---|---|---|
| Real MFS100 | scanner.mock=False | mfs100 | LEFT_THUMB | Crop + enhance + multi-arch | MFS100 \| LEFT THUMB |
| Mock Mode | MFS100_MOCK=true | upload | NULL | Multi-arch (no crop) | MFS100 (Mock Mode) |
| File Upload | POST /api/upload | upload | NULL | Multi-arch (no crop) | Upload |

---

## 10. Left Thumb Acquisition Protocol

**Standard finger:** LEFT THUMB (standardized for all physical MFS100 demonstrations)

**Important constraints:**
- LEFT_THUMB is acquisition metadata stored in `finger_used` column only.
- The AI model does NOT receive `finger_used` as an input.
- The prediction is NOT modified based on which finger is used.
- The scanner does NOT automatically verify finger identity.
- This is a convention-based protocol for demo reproducibility only.

**Reference:** `docs/MFS100_ACQUISITION_PROTOCOL.md`

---

## 11. Project Directory Structure

```
major project/
├── app.py                          ← Flask web server (main entry point)
├── requirements.txt                ← Python dependencies
├── README.md                       ← Project overview
├── models/
│   ├── resnet50_improved_blood_group.keras   ← ResNet50 (277 MB, FROZEN)
│   ├── vgg16_blood_group.keras               ← VGG16 (164 MB, FROZEN)
│   ├── efficientnetb0_blood_group.keras      ← EfficientNetB0 (41 MB, FROZEN)
│   └── class_names.json                      ← 8-class label mapping
├── outputs/
│   ├── ensemble_weights.json        ← LOCKED: ResNet50=58%, VGG16=38%, ENB0=4%
│   ├── FINAL_MODEL_EVALUATION.json  ← Official 85.41% benchmark record
│   ├── dataset_manifest.csv         ← Full 5,985-image split manifest
│   └── ab_plus_diagnostics/         ← AB+ generalization analysis outputs
├── src/
│   ├── config.py                    ← All paths and constants
│   ├── predictor.py                 ← Inference engine (Champion Ensemble V2)
│   ├── preprocessing.py             ← Centralized preprocessing pipeline
│   ├── database.py                  ← SQLite CRUD operations
│   ├── report_generator.py          ← PDF clinical report generator
│   ├── mfs100/scanner.py            ← MFS100 hardware + mock adapter
│   └── mfs100_bridge/bridge.py      ← 32-bit HTTP bridge process
├── templates/
│   ├── index.html                   ← Clinical dashboard UI
│   └── history.html                 ← Audit history page
├── static/
│   ├── css/style.css                ← Healthcare dashboard styles
│   └── js/app.js                    ← Frontend logic
├── tests/                           ← 56 automated tests (all passing)
├── data/
│   ├── blood_group.db               ← SQLite prediction audit database
│   ├── sample/                      ← 8 mock mode reference images
│   └── captured/                    ← Runtime fingerprint captures
├── reports/                         ← Generated PDF clinical reports
└── docs/                            ← Full project documentation suite
```

---

*This document is part of the B.E. Final Year Major Project — Biometric Blood Group Identification System.*
*Architecture frozen as of 2026-09-05. For academic demonstration purposes only.*
