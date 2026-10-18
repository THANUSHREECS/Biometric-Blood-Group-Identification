# Candidate Freeze Manifest: EXP-06C Config A
## Final Independent Evaluation Protocol Preparation

**Document Type:** Formal Architectural & Cryptographic Freeze Manifest  
**Candidate Identifier:** `EXP-06C Config A` (Supervised Contrastive Linear Reference)  
**Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Freeze Status:** **FROZEN & CRYPTOGRAPHICALLY SECURED**  
**Execution Guard:** `TRAINING_AUTHORIZED = False`  

---

> [!CAUTION]
> ### STRICT RESEARCH PROTOCOL — NON-CLINICAL DISCLAIMER
> The model frozen in this manifest is an academic research prototype designed to explore deep photometric fingerprint feature representations. It has not been validated for clinical diagnostics, transfusion medicine, emergency care, or blood banking. Standard laboratory serological typing is mandatory for all clinical applications.

---

## 1. Candidate Specification & Architecture Overview

The candidate frozen for independent evaluation is **EXP-06C Config A**, developed and cross-validated across the 80-participant development cohort.

### 1.1 Structural Architecture
- **Backbone Feature Extractor:** DenseNet-121 (`include_top=False`, ImageNet pretrained weights), strictly non-trainable (`trainable = False`). *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:108`]*
- **Multi-Scan Input:** 3 fingerprint impressions per participant (Left Thumb protocol). Input tensor shape: `(None, 3, 112, 112, 3)`. *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:104`]*
- **Pooling & Concatenation:** Each scan is independently passed through the frozen DenseNet-121 backbone followed by `GlobalAveragePooling2D`, generating a 1024-dimensional feature vector per scan. The three vectors are horizontally concatenated into a unified 3072-dimensional representation:
  $$\mathbf{f}_{\text{fused}} = [\mathbf{f}_1 \,\|\, \mathbf{f}_2 \,\|\, \mathbf{f}_3] \in \mathbb{R}^{3072}$$
  *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:112`]*
- **Classification Head Branch:**
  - `Dense(512, activation='relu', name='dense_512')`
  - `Dropout(0.5, name='dropout_0.5')`
  - `Dense(4, activation='softmax', name='clf')`
  *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:115-130`]*
- **Projection Head Branch:**
  - `Dense(128, activation=None, name='emb')` (Single linear layer projection head into $\mathbb{R}^{128}$ for auxiliary Supervised Contrastive loss).
  *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:133-138`]*

### 1.2 Parameter Inventory

| Sub-Module | Layer Name | Output Shape | Param Count | Trainability Status |
| :--- | :--- | :---: | :---: | :---: |
| **Backbone Tower** | DenseNet-121 (Shared across 3 scans) | `(None, 1024)` | 7,037,504 | **Strictly Frozen** ($0$ trainable) |
| **Scan Fusion** | `fused_features` (Concatenate) | `(None, 3072)` | 0 | Non-parametric |
| **Classifier Dense** | `dense_512` | `(None, 512)` | $3072 \times 512 + 512 = 1,573,376$ | Trained on Development Folds |
| **Classifier Dropout** | `dropout_0.5` | `(None, 512)` | 0 | Non-parametric (Inactive at inference) |
| **Classifier Softmax** | `clf` | `(None, 4)` | $512 \times 4 + 4 = 2,052$ | Trained on Development Folds |
| **Projection Layer** | `emb` | `(None, 128)` | $3072 \times 128 + 128 = 393,344$ | Trained on Development Folds |
| **Total Model Parameters** | — | — | **9,006,276** | — |
| **Total Trainable Head Parameters** | — | — | **1,968,772** | **Frozen Post-Training** |
| **Non-Trainable Parameters** | — | — | **7,037,504** | **Permanently Frozen** |

*[Evidence: Verified from source — model parameter calculation matched bit-for-bit with instantiated candidate in `experiments/final_evaluation/evaluate_frozen_candidate.py`]*

---

## 2. Cryptographic Checkpoint Ledger

The frozen candidate comprises 5 independent fold checkpoints trained under the 5-fold StratifiedGroupKFold cross-validation protocol on the 80 development participants. All 5 files exist and have been cryptographically hashed:

| Fold Index | File Path | File Size (Bytes) | Cryptographic SHA-256 Hash |
| :---: | :--- | :---: | :--- |
| **Fold 1** | `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5` | 52,899,216 | `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc` |
| **Fold 2** | `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5` | 52,899,472 | `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c` |
| **Fold 3** | `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5` | 52,899,472 | `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1` |
| **Fold 4** | `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5` | 52,899,472 | `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3` |
| **Fold 5** | `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5` | 52,899,472 | `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6` |

*[Evidence: Verified from source — SHA-256 computed directly from filesystem and confirmed during synthetic execution]*

---

## 3. Preprocessing & Input Specifications

- **Pipeline Version:** `transform_c5` (Round 14 Verified Protocol). *[Evidence: Verified from source — `src/preprocessing.py:118`]*
- **Transformation Steps:**
  1. *Alpha Composition:* Composite input on pure white background ($[255, 255, 255]$) if alpha channel is present.
  2. *Platen Cropping:* Bounding box crop isolating the Mantra MFS100 optical platen foreground ($[y_1:y_2, x_1:x_2]$).
  3. *CLAHE Ridge Enhancement:* Contrast Limited Adaptive Histogram Equalization with `clipLimit = 2.5`, `tileGridSize = (8, 8)`.
  4. *Dynamic Intensity Stretch:* 1st to 99th percentile contrast stretching.
  5. *Bilateral Filtering:* Edge-preserving smoothing with diameter $d = 3$, $\sigma_{\text{color}} = 25$, $\sigma_{\text{space}} = 25$.
  6. *Target Resizing:* Bilinear interpolation resizing to $112 \times 112$ pixels.
- **Data Type & Range:** Raw `float32` array with pixel values in range $[0.0, 255.0]$.
- **Normalization Policy:** **NO** `densenet.preprocess_input` (zero centering / variance scaling is omitted, matching training-time inputs).
- **Canonical Class Mapping:**
  ```python
  CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
  CLASS_TO_INDEX = {"A+": 0, "AB+": 1, "B+": 2, "O+": 3}
  ```
  *[Evidence: Verified from source — `experiments/exp06c/run_exp06c.py:42`]*

---

## 4. Inference & Ensemble Aggregation Procedure

For any participant $i$ presented for evaluation:
1. Exactly 3 scans ($s_{i,1}, s_{i,2}, s_{i,3}$) are loaded and processed through `transform_c5`, yielding three arrays of shape $(112, 112, 3)$.
2. The scans are stacked into a batch tensor $\mathbf{X}_i \in \mathbb{R}^{1 \times 3 \times 112 \times 112 \times 3}$.
3. Forward inference is executed across all 5 frozen checkpoint models ($k \in \{1, 2, 3, 4, 5\}$):
   $$\mathbf{p}_{i,k} = \operatorname{Softmax}(\text{Model}_k(\mathbf{X}_i)) \in [0, 1]^4$$
4. The ensemble consensus probability vector is computed via unweighted arithmetic soft voting:
   $$\mathbf{P}_{\text{ensemble}}(y=c \mid \mathbf{X}_i) = \frac{1}{5} \sum_{k=1}^5 \mathbf{p}_{i,k}(c)$$
5. The predicted class is assigned via argmax over the ensemble probability vector:
   $$\hat{y}_i = \operatorname{CANONICAL\_CLASSES}\left[\operatorname{argmax}_{c \in \{0,1,2,3\}} \mathbf{P}_{\text{ensemble}}(y=c \mid \mathbf{X}_i)\right]$$

*[Evidence: Verified by executed synthetic test — `experiments/final_evaluation/evaluate_frozen_candidate.py:112-132`]*

---

## 5. Development-Influenced Configuration Decisions (Adaptive Selection Bias Disclosure)

In adherence to scientific transparency, every configuration hyperparameter of EXP-06C Config A that was influenced by prior experimental evaluations on the 80 development participants is explicitly cataloged:

| Decision Component | Selected Value | Alternative Evaluated | Prior Experiment Source | Empirical Justification on Dev Cohort |
| :--- | :---: | :---: | :---: | :--- |
| **Backbone Trainability** | **Frozen** ($0$ trainable) | Partial Fine-Tuning | EXP-04 | Unfreezing DenseBlock 4 or 3+4 caused catastrophic representation collapse and overfitting on $N=80$. |
| **Fusion Mechanism** | **Concatenation** ($3072$-d) | Gated / Self-Attention | EXP-05 | Cross-scan attention overfitted ($45.0\%$ acc); concatenation remained equal or superior ($52.5\%$). |
| **Contrastive Temperature** | **$\tau = 0.05$** | $\tau \in \{0.07, 0.10, 0.15\}$ | EXP-06A | Sharper contrastive scaling achieved peak balanced accuracy ($55.18\%$) and peak AB+ recall ($71.43\%$). |
| **Contrastive Weight** | **$\lambda = 0.50$** | $\lambda \in \{0.05, 0.10, 0.20\}$ | EXP-06B | Heavier metric loss weight stabilized fold variance and maximized majority class sensitivity ($75.86\%$ on O+). |
| **Projection Architecture** | **Linear Head** (`3072 -> 128`) | 2-Layer Nonlinear MLP | EXP-06C | Additional MLP parameter capacity ($+63\%$) caused severe overfitting (accuracy dropped from $53.75\%$ to $46.25\%$). |
| **Data Augmentation** | **Zero Augmentation** | Affine / Zoom / Rot | EXP-02, EXP-03 | Synthetic zoom distorted epidermal ridge frequencies; native scans boosted AB+ recall from $14.29\%$ to $57.14\%$. |
| **Training Duration** | **100-Epoch Floor** | Unconstrained early stop | EXP-03, EXP-04 | Enforcing a 100-epoch floor prevented premature stopping on early noisy validation plateaus. |

*[Evidence: Verified from source — cross-referenced with `reports/accuracy_improvement/master_experiment_review.md`]*

> [!WARNING]
> Because these 7 architectural choices were optimized iteratively against the 80 development participants, the reported cross-validation metrics (Macro F1: $0.5487$, Accuracy: $53.75\%$) are subject to **adaptive model selection bias**. Performance on unseen cohorts is expected to regress toward the mean.

---

## 6. Freeze Certification

- **Checkpoints Verified:** 5 of 5 present, bit-for-bit identical to recorded hashes.
- **Architecture Locked:** `DenseNet121_Fusion_LinearHead`.
- **Training Guard:** Hardcoded `TRAINING_AUTHORIZED = False`.
- **Candidate Status:** **READY FOR INDEPENDENT EVALUATION PROTOCOL.**
