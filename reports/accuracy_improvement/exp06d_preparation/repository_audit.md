# EXP-06D: Repository & Baseline Architecture Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-16  
**Auditor:** Senior ML Research Engineer & Data Integrity Auditor  
**Experiment Reference:** EXP-06D Preparation (Preprocessing & Multi-Scan Fusion Investigation)  
**Baseline Reference Configuration:** EXP-06C Config A (Linear Projection Head Reference)  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, code, data manifests, and inspection findings documented herein are academic research prototypes designed to evaluate biometric representation learning on optical fingerprint impressions. They **must not** be used for clinical diagnostic decision-making, patient transfusion matching, blood typing, triage, or healthcare delivery.

---

## 1. Overview & Audit Objectives

In accordance with Phase 1 instructions, a comprehensive audit of the active codebase, data structures, preprocessing implementations, multi-scan fusion mechanisms, and checkpoint protection manifests was conducted prior to designing **EXP-06D**.

This audit establishes the baseline facts regarding:
1. EXP-06C source code, configuration parameters, and reference results.
2. The exact implementation and execution order of the E2 preprocessing pipeline.
3. The exact three-scan input construction and feature fusion mechanism.
4. Input and output tensor shapes at every model boundary.
5. Native vs. Canonical class order mappings.
6. Participant ID handling and 5-fold cross-validation definitions.
7. Checkpoint locations and pre-existing cryptographic hash manifests.
8. Physical and metadata locations of quarantined cohorts (21 locked test subjects, 26 hospital subjects).
9. Architectural and experimental confounders.

---

## 2. Source Files Inspected

The following source files and manifests were directly read and audited in the repository:

1. **`experiments/exp06c/run_exp06c.py`**: The complete execution script for EXP-06C, detailing `E2ProjectionAblationModel`, `supervised_contrastive_loss`, training loops, StratifiedGroupKFold splitting, and checkpoint selection.
2. **`src/preprocessing.py`**: The centralized image preprocessing library, specifically `transform_c5()`, `crop_mfs100_foreground()`, `enhance_mfs100_ridge_contrast()`, `composite_alpha_on_white()`, and `resize_to_target()`.
3. **`reports/phase3_participant_benchmark/participant_split.csv`**: The primary dataset split manifest containing all 305 scan records across 101 participants.
4. **`reports/accuracy_improvement/exp06c/final_report.md`**: The audited report of EXP-06C establishing Config A reference metrics.
5. **`reports/accuracy_improvement/exp06c/metrics_comparison.csv`**: Tabular metric summary of EXP-06C Config A and Config B.
6. **`reports/accuracy_improvement/exp06c/statistical_analysis_results.json`**: Complete numerical records, contingency tables, bootstrap intervals, and SHA-256 hashes.
7. **`reports/accuracy_improvement/exp06c_verification/post_experiment_audit.md`**: Independent post-experiment audit confirming 0 discrepancies in EXP-06C.
8. **`experiments/exp05/run_exp05.py` & `reports/accuracy_improvement/exp05/final_report.md`**: The prior multi-scan fusion experiment.
9. **`scratch/exp06c_pre_preparation_hashes.json`**: Pre-computed SHA-256 hash registry for the 56 protected baseline checkpoints.
10. **`reports/accuracy_improvement/dataset_audit.md`**: Physical location and metadata documentation for the quarantined 26-participant hospital cohort.

---

## 3. Actual Preprocessing Steps & Execution Order

In the active reference configuration (EXP-06C Config A), preprocessing is executed in `experiments/exp06c/run_exp06c.py` line 380 via `arr, _, _ = transform_c5(str(img_p))`.

The implementation in `src/preprocessing.py` lines 247–260 defines the following sequential pipeline:

```
[Input Image (filepath / raw bytes)]
               │
               ▼
1. load_image_to_pil()
   - Loads image into PIL.Image format (RGB / RGBA / Grayscale).
               │
               ▼
2. composite_alpha_on_white()
   - Evaluates pil_img.mode.
   - If alpha channel exists (RGBA, LA, or transparency info), composites image
     onto solid white background (255, 255, 255).
   - Converts to 3-channel RGB. Prevents black platen border artifacts.
               │
               ▼
3. crop_mfs100_foreground()
   - Converts image to 8-bit grayscale.
   - Evaluates platen variance safeguard: if std < 5.0, aborts crop.
   - Computes row and column mean intensities.
   - Finds row/column indices where mean intensity < 238 (active platen threshold).
   - If active region exists and dimensions >= 80 px:
     Adds 6% padding (pad_x = 0.06 * w, pad_y = 0.06 * h).
     Crops pil_img to active bounding box.
               │
               ▼
4. enhance_mfs100_ridge_contrast()
   - Converts cropped image to 8-bit grayscale.
   - Applies OpenCV CLAHE (clipLimit=2.5, tileGridSize=(8, 8)).
   - Performs dynamic percentile contrast stretching:
     p_low = percentile(1.0), p_high = percentile(99.0).
     Stretches pixel intensities from [p_low, p_high] to [15, 245].
   - Applies bilateral filter (d=3, sigmaColor=25, sigmaSpace=25) to suppress optical platen noise.
   - Converts back to 3-channel RGB PIL.Image.
               │
               ▼
5. resize_to_target()
   - Resizes PIL.Image to target dimensions (112, 112) using Lanczos resampling.
               │
               ▼
6. Array Conversion
   - np.array(resized_pil, dtype=np.float32)
   - Outputs float32 tensor in range [0.0, 255.0].
   - CRITICAL NOTE: NO tf.keras.applications.densenet.preprocess_input is applied.
     The values remain raw [0.0, 255.0] float32.
```

---

## 4. Actual Three-Scan Input Construction & Fusion Mechanism

### 4.1 Input Construction
For each participant $p$, three fingerprint scans are loaded in `load_development_dataset()`:
1. Scans are filtered from `participant_split.csv` where `participant_id == p`.
2. Scans are sorted alphabetically by file path: `p_scans.sort_values("scan_path")`.
3. Exactly three scans per participant are required and asserted: `assert len(p_scans) == 3`.
4. The three preprocessed `(112, 112, 3)` arrays are stacked along axis 0:
   $$\mathbf{X}_p \in \mathbb{R}^{3 \times 112 \times 112 \times 3}$$
5. Across the $N = 80$ development participants, the full dataset tensor is:
   $$\mathbf{X} \in \mathbb{R}^{80 \times 3 \times 112 \times 112 \times 3}, \quad \mathbf{y} \in \mathbb{R}^{80}$$

### 4.2 Fusion Mechanism
In `build_exp06c_model()`:
1. **Input Layer:** `inputs = tf.keras.Input(shape=(3, 112, 112, 3))`
2. **Scan Slicing:**
   $$s_1 = \mathbf{X}[:, 0, :, :, :], \quad s_2 = \mathbf{X}[:, 1, :, :, :], \quad s_3 = \mathbf{X}[:, 2, :, :, :]$$
3. **Feature Extraction:**
   Each scan is passed through the ImageNet-pretrained DenseNet-121 backbone (`base.trainable = False`), followed by `GlobalAveragePooling2D()`:
   $$f_1 = \text{GAP}(\text{DenseNet121}(s_1)) \in \mathbb{R}^{B \times 1024}$$
   $$f_2 = \text{GAP}(\text{DenseNet121}(s_2)) \in \mathbb{R}^{B \times 1024}$$
   $$f_3 = \text{GAP}(\text{DenseNet121}(s_3)) \in \mathbb{R}^{B \times 1024}$$
4. **Fusion Operation (Concatenation):**
   $$f_{\text{fused}} = \text{Concatenate}([f_1, f_2, f_3]) \in \mathbb{R}^{B \times 3072}$$
5. **Head Branches:**
   - **Classification Head:**
     $$d_1 = \text{ReLU}(\mathbf{W}_{\text{dense}} f_{\text{fused}} + \mathbf{b}_{\text{dense}}), \quad \mathbf{W}_{\text{dense}} \in \mathbb{R}^{3072 \times 512}$$
     $$\hat{y}_{\text{clf}} = \text{Softmax}(\mathbf{W}_{\text{clf}} \cdot \text{Dropout}_{0.5}(d_1) + \mathbf{b}_{\text{clf}}), \quad \mathbf{W}_{\text{clf}} \in \mathbb{R}^{512 \times 4}$$
   - **Linear Projection Head:**
     $$z = \mathbf{W}_{\text{emb}} f_{\text{fused}} + \mathbf{b}_{\text{emb}}, \quad \mathbf{W}_{\text{emb}} \in \mathbb{R}^{3072 \times 128}$$
     Passed to Supervised Contrastive Loss with L2 normalization: $\tilde{z} = z / \|z\|_2$.

---

## 5. Model Input, Intermediate, and Output Shapes

| Layer / Stage | Layer Name | Tensor Type | Input Shape | Output Shape | Parameters |
|:---|:---|:---:|:---:|:---:|:---:|
| Model Input | `participant_scans` | InputLayer | — | `(None, 3, 112, 112, 3)` | 0 |
| Scan Slices | `s1, s2, s3` | Lambda Slices | `(None, 3, 112, 112, 3)` | $3 \times `(None, 112, 112, 3)`$ | 0 |
| Feature Trunk | `densenet121` | Frozen Backbone | `(None, 112, 112, 3)` | `(None, 3, 3, 1024)` | 7,037,504 (all frozen) |
| Spatial Pooling | `gap` | GlobalAveragePooling2D | `(None, 3, 3, 1024)` | $3 \times `(None, 1024)`$ | 0 |
| Feature Fusion | `fused_features` | Concatenate | $3 \times `(None, 1024)`$ | `(None, 3072)` | 0 |
| Classifier Hidden | `dense_512` | Dense (ReLU) | `(None, 3072)` | `(None, 512)` | $3072 \times 512 + 512 = 1,573,376$ |
| Regularization | `dropout_0.5` | Dropout (rate=0.5) | `(None, 512)` | `(None, 512)` | 0 |
| Classifier Output | `clf` | Dense (Softmax) | `(None, 512)` | `(None, 4)` | $512 \times 4 + 4 = 2,052$ |
| Metric Projection | `emb` | Dense (Linear) | `(None, 3072)` | `(None, 128)` | $3072 \times 128 + 128 = 393,344$ |
| **Total Model** | `E2_EXP06C_ConfigA` | Custom Functional | `(None, 3, 112, 112, 3)` | `[(None, 4), (None, 128)]` | **1,968,772 Trainable / 7,037,504 Frozen** |

---

## 6. Class Order at Each Model Boundary

A critical finding from historical audits is the existence of two distinct class orderings:

### 6.1 Native Model Order (Historical E2 Training Indexing)
- `NATIVE_CLASSES = ['A+', 'B+', 'AB+', 'O+']`
- `NATIVE_REV_MAP = {'A+': 0, 'B+': 1, 'AB+': 2, 'O+': 3}`
- The classification output layer (`Dense(4, activation='softmax')`) outputs probabilities in this native order:
  $$\hat{y}_{\text{native}} = [P(A+), \; P(B+), \; P(AB+), \; P(O+)]$$

### 6.2 Canonical Order (Audited Reporting Standard)
- `CANONICAL_CLASSES = ['A+', 'AB+', 'B+', 'O+']`
- `CANONICAL_REV_MAP = {'A+': 0, 'AB+': 1, 'B+': 2, 'O+': 3}`
- Permutation mapping between native and canonical:
  $$\text{Index}_{\text{canonical}} = [\text{native}[0], \; \text{native}[2], \; \text{native}[1], \; \text{native}[3]]$$
- In `run_exp06c.py` lines 572–573, canonical remapping is explicitly performed:
  ```python
  p_canon = np.array([p_nat[0], p_nat[2], p_nat[1], p_nat[3]], dtype=np.float64)
  p_canon = p_canon / p_canon.sum()
  ```
- All OOF prediction CSVs, confusion matrices, and final reports are strictly published in Canonical Order.

---

## 7. Participant ID Handling & Cross-Validation Folds

### 7.1 Participant Manifest (`participant_split.csv`)
- **Total Records:** 305 scan rows across 101 unique participants.
- **Split Distribution:**
  - `TRAIN`: 44 participants (132 scans)
  - `VALIDATION`: 36 participants (108 scans)
  - `TEST`: 21 participants (63 scans) — **STRICTLY QUARANTINED**
- **Approved Development Cohort:**
  - Defined as participants in `TRAIN` $\cup$ `VALIDATION`.
  - Exactly **80 unique participants** (240 scans).
  - Class distribution in development cohort:
    - `A+`: 19 participants (57 scans, $23.75\%$)
    - `AB+`: 7 participants (21 scans, $8.75\%$) — **RARE CLASS**
    - `B+`: 25 participants (75 scans, $31.25\%$)
    - `O+`: 29 participants (87 scans, $36.25\%$)

### 7.2 Five-Fold StratifiedGroupKFold Definition
- Cross-validation splits are constructed using:
  ```python
  sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  folds = list(sgkf.split(dummy_X, dev_y, dev_pids))
  ```
- **Grouping Guarantee:** Each participant's 3 scans are collocated. No participant appears in both training and validation within any fold.
- **Fold Allocation:**
  - Fold 1: 64 train participants, 16 validation participants (Val: 4 A+, 1 AB+, 5 B+, 6 O+)
  - Fold 2: 64 train participants, 16 validation participants (Val: 4 A+, 2 AB+, 5 B+, 5 O+)
  - Fold 3: 64 train participants, 16 validation participants (Val: 4 A+, 1 AB+, 5 B+, 6 O+)
  - Fold 4: 64 train participants, 16 validation participants (Val: 3 A+, 2 AB+, 5 B+, 6 O+)
  - Fold 5: 64 train participants, 16 validation participants (Val: 4 A+, 1 AB+, 5 B+, 6 O+)

---

## 8. Checkpoint Integrity & Cryptographic Registry

The project currently protects **66 total checkpoints**:
- **56 Pre-Existing Historical Checkpoints:**
  - Production model: `models/abo_positive_4class/best_model.keras`
  - Original E2 baseline: `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5`
  - EXP-03 Arm 1 baseline: `experiments/exp03/checkpoints/arm_1/fold_[1-5]_best.weights.h5`
  - EXP-05 Config A: `experiments/exp05/checkpoints/config_a/fold_[1-5]_best.weights.h5`
  - EXP-06A Configs A–D: `experiments/exp06a/checkpoints/config_[a-d]/fold_[1-5]_best.weights.h5` (20 files)
  - EXP-06B Arms A–D: `experiments/exp06b/checkpoints/arm_[a-d]/fold_[1-5]_best.weights.h5` (20 files)
- **10 EXP-06C Checkpoints:**
  - `experiments/exp06c/checkpoints/config_a/fold_[1-5]_best.weights.h5` (5 files)
  - `experiments/exp06c/checkpoints/config_b/fold_[1-5]_best.weights.h5` (5 files)

All 66 checkpoints were cryptographically verified during the EXP-06C post-experiment audit with zero modifications (0 bytes altered).

---

## 9. Quarantined Cohorts Documentation

1. **Locked Test Cohort (21 Participants, 63 Scans):**
   - **Location:** Rows where `split == 'TEST'` in `reports/phase3_participant_benchmark/participant_split.csv`.
   - **Participant IDs:** `P002, P0023, P0024, P0029, P003, P0035, P0039, P0041, P0042, P0047, P0048, P0054, P0071, P0073, P0075, P0082, P0083, P0088, P009, P0093, P0095`.
   - **Quarantine Rule:** Zero reads, zero evaluation, zero tuning.
2. **Hospital Cohort (26 Participants, 78 Scans):**
   - **Location:** Physically stored under `data/captured/` (and referenced in `reports/accuracy_improvement/dataset_audit.md`).
   - **Status:** Real-world clinical collection with laboratory-confirmed serological ground truth.
   - **Quarantine Rule:** Completely segregated from development pipelines. Never referenced in any training or validation script.

---

## 10. Audit Findings & Architectural Confounder Analysis

1. **Permutation Non-Invariance in Fusion:**
   - In current Config A, `fused = Concatenate([f1, f2, f3])`. The 3 scans are ordered alphabetically by scan filename.
   - If scans are presented in a different order (e.g., $s_3, s_1, s_2$), the fused vector undergoes a 1024-block permutation, producing a different classification output.
   - However, any structural change to fusion (such as Mean Pooling) reduces the fused dimension from 3072 to 1024, altering head parameter count from 1.97M to 0.92M (introducing a $-53.3\%$ parameter confound).
2. **Untested Complexities in Preprocessing:**
   - `transform_c5` couples platen cropping with an aggressive contrast enhancement function (`enhance_mfs100_ridge_contrast`) consisting of CLAHE (clipLimit=2.5), percentile stretching, and bilateral filtering.
   - Bilateral filtering smoothes high-frequency ridge boundaries. CLAHE can introduce platen gradient artifacts.
   - Preprocessing has never been systematically ablated under the frozen DenseNet-121 + SupCon benchmark.
   - Modifying preprocessing involves **zero architectural changes** to the model, ensuring **0% parameter confound** (exactly 1,968,772 trainable parameters in both configurations).

---

## 11. Sign-off

- **Source Code Inspected:** Complete.
- **Preprocessing Pipeline Traced:** Verified step-by-step.
- **Fusion Mechanism Formulated:** Verified mathematically.
- **Class Order Invariants Documented:** Native vs. Canonical mapped.
- **Quarantines Verified:** Test and Hospital cohorts documented and locked.

*Report compiled autonomously following strict repository inspection protocols.*
