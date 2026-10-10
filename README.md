# Biometric Blood Group Identification System Using Fingerprint Sensor

> **Final Year B.E. Major Project**  
> **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
> **System Architecture:** Optical Fingerprint Sensor (Mantra MFS100) Ã¢â€ â€™ Deep Transfer Learning (ResNet50) Ã¢â€ â€™ Blood Group Classification (8 Classes)

---

## Current System

- **Deployed Architecture**: E2 DenseNet121 3-Scan Feature Fusion Ensemble (5 cross-validation checkpoints with soft-voting).
- **Input Dimensions**: 3 fingerprint impressions per participant, resized to 112×112 using Lanczos interpolation.
- **Classification Scope**: 4 classes (`A+`, `B+`, `AB+`, `O+`).
- **Participant Split & Provenance**: Trained on 80 development participants from the MFS100 dataset (`data/mfs100_final`); evaluated on 21 held-out locked-test participants (accuracy: **8/21 = 38.10%**, macro-F1: **0.2500**, matching always guessing O+ majority baseline).
- **Preprocessing**: C5 deterministic pipeline (solid white alpha compositing → MFS100 foreground crop → CLAHE ridge contrast enhancement → Lanczos `(112, 112)` resize → raw `[0, 255]` float32 arrays; no ImageNet `preprocess_input`).
- **Hardware Integration**: Mantra MFS100 500 DPI optical fingerprint sensor via 32-bit Python-to-.NET bridge (`src/mfs100_bridge/bridge.py`).
- **Advisory Warning Layer**: Validation-selected threshold ($\tau = 0.70$) recommending standard serological confirmation for low-confidence classifications.

---

## Legacy (earlier dataset-based phases, not the deployed model)

### 1. Project Overview

The **Biometric Blood Group Identification System** is an end-to-end medical research prototype designed to investigate the feasibility of classifying human ABO/Rh blood groups directly from optical fingerprint impressions. 

Rather than standard biometric authentication (which verifies identity against stored templates), this system investigates phenotypic dermatoglyphic patterns to predict one of the **8 standard human blood groups**:

$$\text{Blood Groups: } \{ \text{A+}, \text{A-}, \text{AB+}, \text{AB-}, \text{B+}, \text{B-}, \text{O+}, \text{O-} \}$$

The platform integrates:
- High-resolution fingerprint acquisition from physical **Mantra MFS100** optical sensors (or digital image upload).
- Centralized image preprocessing with RGBA alpha transparency compositing on clean white backgrounds.
- Deep Transfer Learning with a fine-tuned **ResNet50** convolutional neural network.
- Automated clinical threshold gating ($< 60\%$ confidence flag).
- Persistent SQLite audit history tracking without personal identifiable information (PII).
- Automated clinical PDF report generation using ReportLab.
- A modern, responsive healthcare diagnostic dashboard.

---

### 2. Problem Statement

In emergency trauma response, disaster medicine, and remote rural clinics, rapid blood typing is critical for emergency transfusions. Standard serological blood typing (agglutination tests using anti-A, anti-B, and anti-D serums) requires venous blood extraction, clinical reagents, laboratory centrifuges, and trained phlebotomists. 

This project explores an AI-assisted non-invasive diagnostic methodology using optical dermatoglyphic impressions to estimate blood group status rapidly in frontline preliminary triage.

---

### 3. Objectives

1. **Clean & Validate Fresh Datasets**: Clean, normalize, and validate uncleaned fingerprint datasets, quarantining redundant duplicates and handling transparency channels.
2. **Eliminate Data Leakage**: Perform stratified splitting (70% train, 15% validation, 15% test) with cryptographic hash verification ensuring .
3. **Unified Preprocessing**: Provide a single, centralized preprocessing pipeline shared across training, evaluation, file upload, and physical scanner capture.
4. **ResNet50 Transfer Learning**: Train a two-stage fine-tuned deep neural network with mild, ridge-preserving augmentation.
5. **Measured, Honest Evaluation**: Evaluate strictly on an untouched test split, recording true non-fabricated metrics.
6. **Mantra MFS100 Integration**: Provide a robust hardware adapter supporting physical USB capture and authentic sample mock mode.
7. **Clinical Audit & Reporting**: Record all predictions to SQLite and export automated PDF clinical summary reports.

---

### 4. Dataset Overview & Fresh Audit

The fresh dataset contains **6,000 raw fingerprint impressions** in Windows Bitmap (`.BMP`) format:

| Blood Group | Class ID | Raw Count | Clean Total | Train (70%) | Val (15%) | Test (15%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A+** | 0 | 565 | 565 | 395 | 85 | 85 |
| **A-** | 1 | 1,009 | 1,008 | 705 | 151 | 152 |
| **AB+** | 2 | 708 | 705 | 493 | 106 | 106 |
| **AB-** | 3 | 761 | 760 | 532 | 114 | 114 |
| **B+** | 4 | 652 | 648 | 454 | 97 | 97 |
| **B-** | 5 | 741 | 741 | 519 | 111 | 111 |
| **O+** | 6 | 852 | 851 | 596 | 128 | 127 |
| **O-** | 7 | 712 | 707 | 495 | 106 | 106 |
| **TOTAL** | Ã¢â‚¬â€ | **6,000** | **5,985** | **4,189** | **898** | **898** |

---

### 5. Dataset Cleaning & Duplicate Quarantine

### Alpha Transparency Handling
5,956 of the 6,000 images are 32-bit RGBA BMPs. Approximately 7.89% of pixels are transparent borders (`alpha == 0`) with raw black `(0, 0, 0)` RGB values.
* **Naive conversion flaw avoided**: Stripping alpha naively produces harsh artificial black borders.
* **Our solution**: Transparent regions are composited onto a pure white background `(255, 255, 255)` before resizing, matching authentic physical paper and optical platen backgrounds.

### Exact Duplicate Quarantine
Cryptographic SHA-256 analysis identified **11 duplicate clusters** containing **15 redundant duplicate files**.
* Untouched raw files remain preserved in `data/raw/`.
* The 15 redundant copies are quarantined into `data/quarantine/duplicates/` with full audit mapping in `data/quarantine/quarantine_manifest.json`.
* Exactly 5,985 canonical clean images are saved to `data/processed/`.

---

### 6. Dataset Splitting & Leakage Prevention

- **Split Ratio**: 70% Training (~4,189), 15% Validation (~898), 15% Testing (~898).
- **Stratification**: All 8 classes maintain proportional representation across splits.
- **Cryptographic Verification**:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Complete dataset manifest saved in `outputs/dataset_manifest.csv`.

---

### 7. Central Preprocessing Pipeline (`src/preprocessing.py`)

A single centralized pipeline governs all image inputs:
```
Fingerprint Source (File / Bytes / MFS100 / PIL)
               Ã¢â€ â€œ
    Alpha Compositing onto White (255, 255, 255)
               Ã¢â€ â€œ
    3-Channel RGB Conversion
               Ã¢â€ â€œ
    Lanczos Resize to (224, 224)
               Ã¢â€ â€œ
    ResNet50 preprocess_input (ImageNet Mean Subtraction)
               Ã¢â€ â€œ
    Batch Tensor (1, 224, 224, 3)
```

---

### 8. Deep Learning Architecture (`scripts/train_model.py`)

- **Backbone**: `tf.keras.applications.ResNet50(include_top=False, weights='imagenet')`
- **Top Head**:
  - `GlobalAveragePooling2D()`
  - `Dense(256, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(8, activation='softmax')`
- **Loss**: `SparseCategoricalCrossentropy()`
- **Class Weights**: Mild class weights computed on training labels to prevent majority bias.
- **Two-Stage Transfer Learning**:
  - **Stage 1**: Frozen backbone, training head with Adam ($10^{-3}$).
  - **Stage 2**: Unfreezing `conv5_block*` with low learning rate ($10^{-5}$); BatchNormalization layers kept frozen for stability.
- **Callbacks**: `ModelCheckpoint` (saving best `val_accuracy`), `EarlyStopping`, `ReduceLROnPlateau`.

---

### 9. Model Evaluation (`scripts/evaluate_model.py`)

- Tested exclusively on the untouched 898-image test set.
- Generates publication-ready confusion matrix: `outputs/confusion_matrix.png`.
- Outputs full precision, recall, F1-score, and per-class metrics to `outputs/evaluation_results.json`.
- Zero fabrication rule: All reported metrics reflect genuine test set inference.

---

### 10. Mantra MFS100 Integration & Mock Mode

The physical **Mantra MFS100 Optical Fingerprint Scanner** is supported via a unified adapter in `src/mfs100/scanner.py`.

### Architecture Compatibility:
- Windows driver installed at `C:\Program Files\Mantra\MFS100\Driver`.
- Native driver test DLL is 32-bit (x86); 64-bit Python processes cannot load 32-bit DLLs in-process (`WinError 193`).
- **Production Mock Mode (`MFS100_MOCK=true`)**: Enabled by default for development and automated testing. Cycles through authentic high-resolution fingerprint samples from `data/sample/`, passing through the exact same preprocessing and prediction pipeline.

---

### 11. Web Application & REST API

Run the Flask server:
```bash
python app.py
```
Access the application at `http://127.0.0.1:5000`.

### Endpoints:
- `GET /` Ã¢â‚¬â€ Clinical web dashboard.
- `GET /history` Ã¢â‚¬â€ SQLite audit log page.
- `GET /api/status` Ã¢â‚¬â€ Model, scanner, and system health status.
- `POST /api/upload` Ã¢â‚¬â€ Upload fingerprint image for classification.
- `POST /api/capture` Ã¢â‚¬â€ Trigger MFS100 / mock capture and classification.
- `GET /api/history` Ã¢â‚¬â€ Fetch recent prediction records.
- `GET /api/report/<id>` Ã¢â‚¬â€ Download formal clinical PDF report.
- `POST /api/scanner/connect` Ã¢â‚¬â€ Probe and connect to scanner.
- `POST /api/scanner/disconnect` Ã¢â‚¬â€ Disconnect scanner.

---

### 12. SQLite Audit Trail & PDF Reports

- **Database**: `data/blood_group.db` with table `predictions`.
- **PDF Report**: Generated using ReportLab (`src/report_generator.py`), including:
  - Timestamp, Report ID, Input Type.
  - Embedded fingerprint impression thumbnail.
  - Large Predicted Blood Group card and confidence meter.
  - Full 8-class probability breakdown table.
  - Mandatory medical disclaimer.

---

---

### 13. ABO/Rh Analytical Decomposition & Rh Uncertainty Advisory

Following empirical findings from live Mantra MFS100 stress tests (where dermatoglyphic we investigate if ridge patterns correlate with ABO classes, but biological links to Rh factor are weak):

1. **Analytical Decision-Support Layer**:
   - The primary 8-class prediction (`A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`) and model probability distribution are preserved **strictly intact and unaltered**.
   - ABO group probabilities are derived as pure probability mass sums:
     $$P(\text{ABO}) = P(\text{ABO}^+) + P(\text{ABO}^-)$$
   - This decomposition provides clinical explainability without overriding the model's locked 8-class predictions.

2. **Rh-Factor Uncertainty Rules**:
   - **Probability Margin Rule**: When the margin between $P(\text{ABO}^+)$ and $P(\text{ABO}^-)$ is $< 20\%$.
   - **Confidence Gating**: When 8-class prediction confidence is $< 65\%$.
   - **Rh-Negative Validation**: All predicted Rh-negative classifications trigger an advisory notice because Rh- status is biologically rare and prone to noise under lower sensor ridge contrast.
   - **Advisory Status**: This flag is strictly an advisory mechanism to guide clinicians; it does NOT modify model probabilities or imply model defect.

3. **Mandatory Serological Confirmation**:
   - ABO/Rh decomposition does NOT make the system capable of standalone blood typing.
   - All blood group determinations, particularly Rh factor confirmation, **require standard clinical serological agglutination testing**.

---

### 14. Running Automated Tests

Run the full suite of unit and integration tests:
```bash
python -m pytest tests/
```
The test suite evaluates:
1. `tests/test_dataset_prep.py` Ã¢â‚¬â€ Manifest, split distribution, class mappings.
2. `tests/test_preprocessing.py` Ã¢â‚¬â€ Alpha compositing, RGB normalization, tensor shapes.
3. `tests/test_predictor.py` Ã¢â‚¬â€ Thread-safe predictor, probability summation, ABO decomposition, and Rh uncertainty rules.
4. `tests/test_database.py` Ã¢â‚¬â€ SQLite database initialization, insertion, and retrieval.
5. `tests/test_mfs100.py` Ã¢â‚¬â€ MFS100 mock capture, sample rotation, device info.
6. `tests/test_report.py` Ã¢â‚¬â€ ReportLab clinical PDF generation with ABO breakdown and Rh uncertainty banners.
7. `tests/test_api.py` Ã¢â‚¬â€ Flask API endpoints and HTML template rendering.

---

## 15. Rigorous Scientific Evaluation Audit (Final Phase)

To ensure the utmost honesty and scientific rigor in evaluating this research prototype, a comprehensive final audit was conducted restricting the evaluation to a strict 4-class scenario (A+, B+, AB+, O+) using the E2 Research DenseNet121 model.

### Data Leakage & Subject-Level Splitting
An audit of the dataset (`dataset_manifest.csv`) revealed that the raw fingerprint image clusters lack explicit metadata linking individual images to unique human subject IDs. Therefore, it is impossible to guarantee strict subject-level splitting (i.e., ensuring a person's fingerprint is not in both train and test splits). Due to this inherent dataset limitation, any reported high accuracy on the 8-class scenario (e.g., ~85%) was heavily inflated by data leakage.

### Baselines & Honest Metrics

| Evaluation Setting | Model / Architecture | Samples / Scope | Metric / Accuracy | Baseline & Chance | Status & Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Dataset)** | **Model A: EfficientNetB0** (not deployed) | 415 test images (random image split, public fingerprint dataset) | **86.02%** (95% CI: [82.65%, 88.92%]) | Majority baseline: 30.6% | Standalone single-image model; **not deployed**. Subject independence not verifiable (no subject IDs). Metadata artifact classifiers achieve 44.58% (random split), shortcut features (intensity/contrast); leakage unverified. |
| **21 Locked Test Participants** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 21 Locked Test Participants (63 scans, 3 per person) | **38.10%** (8/21; Wilson 95% CI: [20.75%, 59.12%]) | Always-O+ baseline: 38.10% (8/21); binomial p = 1.0000 | **Valid**: True locked test, exact reproduction of `final_test_predictions.csv` (21/21 top-1 match, max diff = 0.00000019). Matches always guessing O+ (8/21). Pred dist: A+=11, B+=2, AB+=0, O+=8. |
| **80 Development Cohort (In-Sample Ensemble)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (240 scans, evaluated with full 5-checkpoint ensemble) | **40.00%** (32/80; Wilson 95% CI: [29.96%, 50.95%]) | Always-O+ baseline: 36.25% (29/80) | **In-sample dev evaluation**: Full 5-checkpoint ensemble evaluated on dev cohort. Pred dist: A+=47, O+=16, B+=17. |
| **80 Development Cohort (Out-of-Fold Validation)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (StratifiedGroupKFold, out-of-fold checkpoints) | **53.75%** (43/80; Wilson 95% CI: [42.90%, 64.25%]) | Always-O+ baseline: 36.25% (29/80) | **Out-of-Fold (OOF)**: Each fold model evaluated strictly on its held-out validation fold (`fold_results.csv`). Average training accuracy across 5 folds was **35.94%**. |
| **101 Pooled Participants (Deployed Path)** | **Model B: E2 DenseNet121 Ensemble** (deployed post-fix) | 101 Participants (80 dev + 21 locked test) | **39.60%** (40/101; Wilson 95% CI: [30.62%, 49.35%]) | Always-O+ baseline: 36.63% (37/101); binomial p = 0.5369 | **Contaminated benchmark**: Contains 80 development subjects. Pred dist: A+=58, O+=24, B+=19. |
| **101 Pooled — Old Mismatched Pipeline (INVALID)** | **Model B: E2 DenseNet121 Ensemble** (pre-Round 14) | 101 Participants | **27.72% — INVALID** (28/101; Wilson 95% CI: [19.93%, 37.15%]) | Always-O+ baseline: 36.63% (37/101) | **Invalid**: Caused by inadvertent addition of `densenet.preprocess_input` to raw `[0, 255]` weights. Corrected in Round 14. |

### Class Imbalance & Calibration
The deployed system includes an advisory threshold layer in the UI (70% threshold). Predictions below this threshold display an explicit **"Confirmation Recommended" / "Low Confidence"** notice, instructing users that standard serological testing is mandatory.

### Limitations & Threats to Validity
1. **Model Distinction:** Model A (EfficientNetB0, 224x224, 86.02%) is not deployed. Model B (E2 DenseNet121 5-fold ensemble, 112x112) is deployed.
2. **Model Provenance & Class Order Verified:** E2 training script preserved at `scripts/experiment6_fusion_supcon.py` (copied from `scratch/experiment6_fusion_supcon.py`, 9 Oct 2026). Class orders: E2 DenseNet121 ensemble uses `['A+', 'B+', 'AB+', 'O+']` ([src/config.py](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/config.py#L7)), matching training; Model A uses `['A+', 'AB+', 'B+', 'O+']` ([scripts/evaluate_abo_positive_4class.py](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scripts/evaluate_abo_positive_4class.py#L56)).
3. **Subject Independence:** The raw dataset lacks participant IDs. Burst captures likely span train and test sets, inflating random-split performance.
4. **Participant Partitioning & Real-World Generalization:** Of the 101 participants, 80 were used in E2 development and 21 constituted the locked test set. On the 21 locked test participants, the E2 ensemble achieves 8/21 = 38.10%, equal to the 38.10% always-O+ majority baseline.
5. **Biological Plausibility:** There is no established genetic or physiological mechanism linking macroscopic epidermal ridge patterns to ABO blood groups. Our empirical data on 21 locked test participants show no detectable signal above majority chance.
6. **Clinical Viability:** The system is completely unfit for clinical decision-making.

---

## 16. Medical Disclaimer

> **MANDATORY MEDICAL DISCLAIMER**  
> This system is an AI-based preliminary estimation tool developed for academic research and proof-of-concept investigation. The results generated by this model **MUST BE CONFIRMED** using standard laboratory serological blood typing tests (e.g., forward and reverse ABO agglutination) before any clinical or medical administration. This system does not replace certified clinical laboratory diagnostics.

See Section 15 and `results/final_audit_results.md` for the authoritative primary evaluation summary table. Clinical validity is not claimed.
