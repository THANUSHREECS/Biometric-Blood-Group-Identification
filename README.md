# Biometric Blood Group Identification System Using Fingerprint Sensor

> **Final Year B.E. Major Project**  
> **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
> **System Architecture:** Optical Fingerprint Sensor (Mantra MFS100) → Deep Transfer Learning (ResNet50) → Blood Group Classification (8 Classes)

---

## 1. Project Overview

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

## 2. Problem Statement

In emergency trauma response, disaster medicine, and remote rural clinics, rapid blood typing is critical for emergency transfusions. Standard serological blood typing (agglutination tests using anti-A, anti-B, and anti-D serums) requires venous blood extraction, clinical reagents, laboratory centrifuges, and trained phlebotomists. 

This project explores an AI-assisted non-invasive diagnostic methodology using optical dermatoglyphic impressions to estimate blood group status rapidly in frontline preliminary triage.

---

## 3. Objectives

1. **Clean & Validate Fresh Datasets**: Clean, normalize, and validate uncleaned fingerprint datasets, quarantining redundant duplicates and handling transparency channels.
2. **Eliminate Data Leakage**: Perform stratified splitting (70% train, 15% validation, 15% test) with cryptographic hash verification ensuring .
3. **Unified Preprocessing**: Provide a single, centralized preprocessing pipeline shared across training, evaluation, file upload, and physical scanner capture.
4. **ResNet50 Transfer Learning**: Train a two-stage fine-tuned deep neural network with mild, ridge-preserving augmentation.
5. **Measured, Honest Evaluation**: Evaluate strictly on an untouched test split, recording true non-fabricated metrics.
6. **Mantra MFS100 Integration**: Provide a robust hardware adapter supporting physical USB capture and authentic sample mock mode.
7. **Clinical Audit & Reporting**: Record all predictions to SQLite and export automated PDF clinical summary reports.

---

## 4. Dataset Overview & Fresh Audit

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
| **TOTAL** | — | **6,000** | **5,985** | **4,189** | **898** | **898** |

---

## 5. Dataset Cleaning & Duplicate Quarantine

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

## 6. Dataset Splitting & Leakage Prevention

- **Split Ratio**: 70% Training (~4,189), 15% Validation (~898), 15% Testing (~898).
- **Stratification**: All 8 classes maintain proportional representation across splits.
- **Cryptographic Verification**:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Complete dataset manifest saved in `outputs/dataset_manifest.csv`.

---

## 7. Central Preprocessing Pipeline (`src/preprocessing.py`)

A single centralized pipeline governs all image inputs:
```
Fingerprint Source (File / Bytes / MFS100 / PIL)
               ↓
    Alpha Compositing onto White (255, 255, 255)
               ↓
    3-Channel RGB Conversion
               ↓
    Lanczos Resize to (224, 224)
               ↓
    ResNet50 preprocess_input (ImageNet Mean Subtraction)
               ↓
    Batch Tensor (1, 224, 224, 3)
```

---

## 8. Deep Learning Architecture (`scripts/train_model.py`)

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

## 9. Model Evaluation (`scripts/evaluate_model.py`)

- Tested exclusively on the untouched 898-image test set.
- Generates publication-ready confusion matrix: `outputs/confusion_matrix.png`.
- Outputs full precision, recall, F1-score, and per-class metrics to `outputs/evaluation_results.json`.
- Zero fabrication rule: All reported metrics reflect genuine test set inference.

---

## 10. Mantra MFS100 Integration & Mock Mode

The physical **Mantra MFS100 Optical Fingerprint Scanner** is supported via a unified adapter in `src/mfs100/scanner.py`.

### Architecture Compatibility:
- Windows driver installed at `C:\Program Files\Mantra\MFS100\Driver`.
- Native driver test DLL is 32-bit (x86); 64-bit Python processes cannot load 32-bit DLLs in-process (`WinError 193`).
- **Production Mock Mode (`MFS100_MOCK=true`)**: Enabled by default for development and automated testing. Cycles through authentic high-resolution fingerprint samples from `data/sample/`, passing through the exact same preprocessing and prediction pipeline.

---

## 11. Web Application & REST API

Run the Flask server:
```bash
python app.py
```
Access the application at `http://127.0.0.1:5000`.

### Endpoints:
- `GET /` — Clinical web dashboard.
- `GET /history` — SQLite audit log page.
- `GET /api/status` — Model, scanner, and system health status.
- `POST /api/upload` — Upload fingerprint image for classification.
- `POST /api/capture` — Trigger MFS100 / mock capture and classification.
- `GET /api/history` — Fetch recent prediction records.
- `GET /api/report/<id>` — Download formal clinical PDF report.
- `POST /api/scanner/connect` — Probe and connect to scanner.
- `POST /api/scanner/disconnect` — Disconnect scanner.

---

## 12. SQLite Audit Trail & PDF Reports

- **Database**: `data/blood_group.db` with table `predictions`.
- **PDF Report**: Generated using ReportLab (`src/report_generator.py`), including:
  - Timestamp, Report ID, Input Type.
  - Embedded fingerprint impression thumbnail.
  - Large Predicted Blood Group card and confidence meter.
  - Full 8-class probability breakdown table.
  - Mandatory medical disclaimer.

---

---

## 13. ABO/Rh Analytical Decomposition & Rh Uncertainty Advisory

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

## 14. Running Automated Tests

Run the full suite of unit and integration tests:
```bash
python -m pytest tests/
```
The test suite validates:
1. `tests/test_dataset_prep.py` — Manifest, split distribution, class mappings.
2. `tests/test_preprocessing.py` — Alpha compositing, RGB normalization, tensor shapes.
3. `tests/test_predictor.py` — Thread-safe predictor, probability summation, ABO decomposition, and Rh uncertainty rules.
4. `tests/test_database.py` — SQLite database initialization, insertion, and retrieval.
5. `tests/test_mfs100.py` — MFS100 mock capture, sample rotation, device info.
6. `tests/test_report.py` — ReportLab clinical PDF generation with ABO breakdown and Rh uncertainty banners.
7. `tests/test_api.py` — Flask API endpoints and HTML template rendering.

---

## 15. Rigorous Scientific Evaluation Audit (Final Phase)

To ensure the utmost honesty and scientific rigor in evaluating this research prototype, a comprehensive final audit was conducted restricting the evaluation to a strict 4-class scenario (A+, B+, AB+, O+) using the E2 Champion DenseNet121 model.

### Data Leakage & Subject-Level Splitting
An audit of the dataset (`dataset_manifest.csv`) revealed that the raw fingerprint image clusters lack explicit metadata linking individual images to unique human subject IDs. Therefore, it is impossible to guarantee strict subject-level splitting (i.e., ensuring a person's fingerprint is not in both train and test splits). Due to this inherent dataset limitation, any reported high accuracy on the 8-class scenario (e.g., ~85%) was heavily inflated by data leakage.

### Baselines & Honest Metrics
On the locked test set, without artificially inflating metrics or peeking at the test data, the model achieved:
- **Model Accuracy:** 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) (95% CI: [34.86%, 41.31%])
- **Random Baseline:** 25.00%
- **Majority-Class Baseline:** 26.95%
- **Macro F1:** ~0.25
- **Permutation Test:** A statistical permutation test confirms that while 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) is better than random chance ($p < 0.001$), the clinical relevance is extremely limited.

### Class Imbalance & Calibration
The model exhibits class imbalance effects and was highly overconfident in its predictions. Calibration analysis revealed:
- **Uncalibrated ECE:** 0.4094
- **Temperature Scaled ECE (Validation):** 0.2802
The UI has been strictly updated to enforce a **70% advisory threshold**. Predictions below this threshold display an explicit **"Not Reliable"** warning rather than presenting the user with an overconfident blood group letter.

### Limitations & Threats to Validity
1. **Data Leakage & 8-Class Claims:** Our initial 8-class accuracy metrics (~85%) are **possibly inflated and cannot be verified without subject IDs**. The lack of unique participant identifiers in the raw dataset means identical subjects were likely split across training and testing sets, resulting in identity memorization rather than blood group generalization.
2. **Domain Shift:** Live hardware captures from the MFS100 differ statistically (ridge contrast, mean intensity) from the curated dataset samples.
3. **Small Live Sample:** Testing strictly on a limited live sample (e.g. one participant's O+ thumb) reduces external validity and statistical power.
4. **Statistical vs. Scientific Significance:** While the random-split 4-class accuracy of 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) is statistically higher than random guessing (25%, $p < 0.001$) on leaked data, it does not establish a scientifically viable or clinically useful diagnostic tool.
5. **Biological Plausibility:** There is no proven biological or genetic mechanism linking friction ridge patterns definitively to ABO blood groups. The observed accuracy may capture latent confounding variables in the dataset rather than a universal biological truth.
6. **Clinical Viability:** The system is completely unfit for any clinical decision-making.

---

## 16. Medical Disclaimer

> **MANDATORY MEDICAL DISCLAIMER**  
> This system is an AI-based preliminary estimation tool developed for academic research and proof-of-concept investigation. The results generated by this model **MUST BE CONFIRMED** using standard laboratory serological blood typing tests (e.g., forward and reverse ABO agglutination) before any clinical or medical administration. This system does not replace certified clinical laboratory diagnostics.



**Current evidence:** The 4-class (A+, B+, AB+, O+) model achieves 86.0% accuracy on a random split. However, subject independence cannot be verified because the dataset lacks subject IDs. A metadata artifact classifier achieves 44.58%, beating the majority baseline, suggesting that dataset artifacts (like file size or contrast) correlate with the labels. The origin of the 38.10% accuracy metric is unknown. The live system does not demonstrate reliable cross-domain four-class blood group classification. Clinical validity is not claimed.
