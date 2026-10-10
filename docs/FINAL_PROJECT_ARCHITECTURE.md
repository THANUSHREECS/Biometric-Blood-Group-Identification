ï»¿# FINAL PROJECT ARCHITECTURE
## Biometric Blood Group Identification System

**Document Type:** System Architecture Reference
**Project:** B.E. Final Year Major Project â€” Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Model Status:** FROZEN â€” Champion Ensemble V2 (85.41% locked test accuracy)

---

## 1. System Overview

The Biometric Blood Group Identification System is an end-to-end AI-based research prototype that classifies human ABO/Rh blood groups from optical fingerprint impressions.

```
Physical Fingerprint (Mantra MFS100) OR Uploaded Image
        â†“
  Image Acquisition Layer
        â†“
  Centralized Preprocessing Pipeline (src/preprocessing.py)
        â†“
  Champion Ensemble V2 Inference Engine (src/predictor.py)
        â†“
  ABO/Rh Analytical Decomposition
        â†“
  Confidence Threshold Evaluation (>= 60%)
        â†“
  SQLite Audit Record (src/database.py)
        â†“
  Clinical PDF Report (src/report_generator.py)
        â†“
  Web Dashboard Response (Flask REST API)
```

---

## 2. Complete Component Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        FRONTEND LAYER (Browser)                              â”‚
â”‚  templates/index.html â€” Clinical Dashboard                                   â”‚
â”‚  templates/history.html â€” Audit History View                                 â”‚
â”‚  static/js/app.js â€” UI Logic & API calls                                    â”‚
â”‚  static/css/style.css â€” Healthcare-themed responsive design                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                  â”‚ HTTP REST (port 5000)
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                        FLASK BACKEND (app.py)                                â”‚
â”‚  GET  /                 â€” Dashboard HTML                                     â”‚
â”‚  GET  /history          â€” History HTML                                       â”‚
â”‚  GET  /api/status       â€” System health (model + scanner status)             â”‚
â”‚  POST /api/upload       â€” Image file upload â†’ prediction                     â”‚
â”‚  POST /api/capture      â€” MFS100 single capture â†’ prediction                 â”‚
â”‚  POST /api/capture/consensus â€” 3-scan LEFT THUMB consensus                  â”‚
â”‚  GET  /api/history      â€” Prediction audit records (JSON)                    â”‚
â”‚  GET  /api/report/<id>  â€” Clinical PDF download                              â”‚
â”‚  POST /api/scanner/mode â€” Toggle hardware/mock mode                          â”‚
â”‚  GET  /api/scanner/status â€” Scanner + bridge connectivity                    â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”˜
       â”‚                      â”‚                      â”‚                      â”‚
â”Œâ”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”
â”‚ src/        â”‚  â”‚ src/predictor.py   â”‚  â”‚ src/database.py   â”‚  â”‚ src/report_ â”‚
â”‚ mfs100/     â”‚  â”‚ BloodGroupPredictorâ”‚  â”‚ SQLite Audit DB   â”‚  â”‚ generator.pyâ”‚
â”‚ scanner.py  â”‚  â”‚                    â”‚  â”‚                   â”‚  â”‚ ReportLab   â”‚
â”‚ MFS100      â”‚  â”‚ Champion Ensemble  â”‚  â”‚ predictions table â”‚  â”‚ PDF engine  â”‚
â”‚ Adapter     â”‚  â”‚ V2 Inference       â”‚  â”‚ CRUD operations   â”‚  â”‚             â”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚                      â”‚
â”Œâ”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ src/mfs100_ â”‚  â”‚ src/preprocessing.py                                    â”‚
â”‚ bridge/     â”‚  â”‚ Centralized Image Preprocessing Pipeline               â”‚
â”‚ bridge.py   â”‚  â”‚ (shared by training, upload, MFS100, mock)             â”‚
â”‚ 32-bit HTTP â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚ bridge      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 3. MFS100 Hardware Architecture

### 3.1 The Cross-Architecture Problem

The Mantra MFS100 driver DLL (`MFS100Dll.dll`) is a 32-bit (x86) Windows native library.
Python-based AI/ML environments run as 64-bit (x64) processes.

```
Windows 64-bit Python (AI/ML environment)
    â”œâ”€â”€ TensorFlow 2.21.0 (64-bit)
    â”œâ”€â”€ Flask (64-bit)
    â””â”€â”€ CANNOT load 32-bit DLLs directly (WinError 193)
```

### 3.2 32-bit Bridge Solution

A dedicated lightweight 32-bit Python bridge process (`src/mfs100_bridge/bridge.py`) runs under a separate 32-bit Python installation and serves as an HTTP intermediary:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚   64-bit Flask Backend          â”‚       â”‚   32-bit MFS100 Bridge           â”‚
â”‚   (Python 3.10.11 x64)         â”‚       â”‚   (C:\Python310-32\python.exe)   â”‚
â”‚                                 â”‚       â”‚                                   â”‚
â”‚   MFS100Scanner.capture() â”€â”€â”€â”€â–º â”‚ HTTP  â”‚ â—„â”€â”€â”€ ctypes.WinDLL(MFS100Dll)   â”‚
â”‚   (src/mfs100/scanner.py)      â”‚ :5100 â”‚       Physical MFS100 USB SDK    â”‚
â”‚                                 â”‚â—„â”€â”€â”€â”€â”€ â”‚â”€â”€â”€â–º  scanner.capture()           â”‚
â”‚   Returns: image bytes +        â”‚       â”‚      quality, nfiq, image bytes  â”‚
â”‚   quality_meta (nfiq, score)    â”‚       â”‚                                   â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                                        â”‚
                                          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                                          â”‚   Mantra MFS100 USB Hardware    â”‚
                                          â”‚   Optical platen: 316x354 px    â”‚
                                          â”‚   500 DPI optical sensor        â”‚
                                          â”‚   SDK: MFS100Dll.dll (32-bit)   â”‚
                                          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
dataset_blood_group/          â† Raw source fingerprint dataset
data/
  raw/                        â† Unmodified raw images (6,000 total)
  processed/                  â† 5,985 cleaned images (15 duplicates quarantined)
  quarantine/
    duplicates/               â† 15 quarantined SHA-256 duplicate images
    quarantine_manifest.json  â† Full quarantine audit map
  sample/                     â† 8 reference images (one per class, for mock mode)
  captured/                   â† Runtime: MFS100 and upload captures stored here
  blood_group.db              â† SQLite prediction audit database
```

**Dataset splits (stratified, leakage-free):**

| Split | Images | Purpose |
|---|---|---|
| Training | 4,189 | Model training |
| Validation | 898 | Hyperparameter tuning and intermediate evaluation |
| Test (LOCKED) | 898 | Official one-time benchmark evaluation â€” NOT re-accessed |

---

## 5. Preprocessing Pipeline Architecture

### 5.1 Upload Mode (source="upload")

```
Input: File path / bytes / PIL
    â†“ load_image_to_pil()
    â†“ composite_alpha_on_white()    â† RGBA transparent borders â†’ white (255,255,255)
    â†“ resize_to_target(224Ã—224)     â† Lanczos resampling
    â†“ Per-architecture preprocessing:
       ResNet50: ImageEnhance.Contrast(Ã—1.2) + resnet50.preprocess_input
       VGG16:    vgg16.preprocess_input
       ENB0:     efficientnet.preprocess_input
    â†“ Batch tensors â†’ ensemble inference
```

### 5.2 MFS100 Hardware Mode (source="mfs100")

```
Input: Raw MFS100 capture (316Ã—354, NFIQ quality check)
    â†“ load_image_to_pil()
    â†“ composite_alpha_on_white()
    â†“ crop_mfs100_foreground()      â† Hardware foreground crop (removes sensor margins)
       4 safeguards: blank guard, region check, size floor, margin presence check
    â†“ enhance_mfs100_ridge_contrast()
       CLAHE (clipLimit=2.5, 8Ã—8) + percentile stretch + bilateral filter
    â†“ resize_to_target(224Ã—224)
    â†“ Per-architecture preprocessing (same as upload above)
    â†“ Batch tensors â†’ ensemble inference
```

---

## 6. Champion Ensemble V2 Inference Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    Champion Ensemble V2                                  â”‚
â”‚              Locked Test Accuracy: 85.41% (767/898)                     â”‚
â”‚                                                                          â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚ ResNet50 (Improved)  â”‚  â”‚      VGG16           â”‚  â”‚ EfficientNetB0â”‚  â”‚
â”‚  â”‚ 277 MB checkpoint    â”‚  â”‚ 164 MB checkpoint    â”‚  â”‚ 41 MB checkpt â”‚  â”‚
â”‚  â”‚ Weight: 0.58 (58%)  â”‚  â”‚ Weight: 0.38 (38%)  â”‚  â”‚ Weight: 0.04  â”‚  â”‚
â”‚  â”‚ 1.2x contrast        â”‚  â”‚ Native VGG16         â”‚  â”‚ Native ENB0   â”‚  â”‚
â”‚  â”‚ preprocessing        â”‚  â”‚ preprocessing        â”‚  â”‚ preprocessing â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚             â”‚                         â”‚                       â”‚          â”‚
â”‚  P_resnet[8] Ã— 0.58   +  P_vgg[8] Ã— 0.38   +  P_eff[8] Ã— 0.04         â”‚
â”‚             â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜          â”‚
â”‚                                       â†“                                  â”‚
â”‚              Ensemble_probs[8] = weighted soft-vote average              â”‚
â”‚                                       â†“                                  â”‚
â”‚              predicted_class = argmax(Ensemble_probs)                   â”‚
â”‚              confidence = max(Ensemble_probs)                           â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Post-inference decision layers:**

```
confidence >= 0.60 â†’ status = NORMAL
confidence <  0.60 â†’ status = LOW_CONFIDENCE (advisory + recapture recommendation)

ABO/Rh Decomposition (analytics only â€” does NOT change prediction):
  P(A)  = P(A+) + P(A-)
  P(B)  = P(B+) + P(B-)
  P(AB) = P(AB+) + P(AB-)
  P(O)  = P(O+) + P(O-)
  top_abo = argmax of above

Rh Uncertainty Advisory (does NOT change prediction):
  rh_margin < 0.20  â†’ advisory flag
  confidence < 0.65 â†’ advisory flag
  predicted_Rh == "-" â†’ advisory flag (weak ridge-Rh correlation)
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
| confidence | REAL | Float 0.0â€“1.0 |
| status | TEXT | 'NORMAL' or 'LOW_CONFIDENCE' |
| notes | TEXT | Model message / advisory text |
| finger_used | TEXT | 'LEFT_THUMB' for hardware; NULL for upload/mock |

**Migration safety:** `finger_used` was added via non-destructive `ALTER TABLE` â€” existing records retain `NULL`.

---

## 8. PDF Report Architecture

**Generator:** `src/report_generator.py` (ReportLab)
**Output:** `reports/blood_group_report_<id>_<timestamp>.pdf`

**Report structure:**
1. Header banner â€” project title
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
â”œâ”€â”€ app.py                          â† Flask web server (main entry point)
â”œâ”€â”€ requirements.txt                â† Python dependencies
â”œâ”€â”€ README.md                       â† Project overview
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ resnet50_improved_blood_group.keras   â† ResNet50 (277 MB, FROZEN)
â”‚   â”œâ”€â”€ vgg16_blood_group.keras               â† VGG16 (164 MB, FROZEN)
â”‚   â”œâ”€â”€ efficientnetb0_blood_group.keras      â† EfficientNetB0 (41 MB, FROZEN)
â”‚   â””â”€â”€ class_names.json                      â† 8-class label mapping
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ ensemble_weights.json        â† LOCKED: ResNet50=58%, VGG16=38%, ENB0=4%
â”‚   â”œâ”€â”€ FINAL_MODEL_EVALUATION.json  â† Official 85.41% benchmark record
â”‚   â”œâ”€â”€ dataset_manifest.csv         â† Full 5,985-image split manifest
â”‚   â””â”€â”€ ab_plus_diagnostics/         â† AB+ generalization analysis outputs
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py                    â† All paths and constants
â”‚   â”œâ”€â”€ predictor.py                 â† Inference engine (Champion Ensemble V2)
â”‚   â”œâ”€â”€ preprocessing.py             â† Centralized preprocessing pipeline
â”‚   â”œâ”€â”€ database.py                  â† SQLite CRUD operations
â”‚   â”œâ”€â”€ report_generator.py          â† PDF clinical report generator
â”‚   â”œâ”€â”€ mfs100/scanner.py            â† MFS100 hardware + mock adapter
â”‚   â””â”€â”€ mfs100_bridge/bridge.py      â† 32-bit HTTP bridge process
â”œâ”€â”€ templates/
â”‚   â”œâ”€â”€ index.html                   â† Clinical dashboard UI
â”‚   â””â”€â”€ history.html                 â† Audit history page
â”œâ”€â”€ static/
â”‚   â”œâ”€â”€ css/style.css                â† Healthcare dashboard styles
â”‚   â””â”€â”€ js/app.js                    â† Frontend logic
â”œâ”€â”€ tests/                           â† 56 automated tests (all passing)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ blood_group.db               â† SQLite prediction audit database
â”‚   â”œâ”€â”€ sample/                      â† 8 mock mode reference images
â”‚   â””â”€â”€ captured/                    â† Runtime fingerprint captures
â”œâ”€â”€ reports/                         â† Generated PDF clinical reports
â””â”€â”€ docs/                            â† Full project documentation suite
```

---

*This document is part of the B.E. Final Year Major Project â€” Biometric Blood Group Identification System.*
*Architecture frozen as of 2026-09-05. For academic demonstration purposes only.*
