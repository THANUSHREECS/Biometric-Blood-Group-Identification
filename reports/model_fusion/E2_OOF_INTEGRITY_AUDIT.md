# E2 OUT-OF-FOLD INTEGRITY AUDIT REPORT

**Audit Date:** 2026-10-11  
**Target Architecture:** E2 (DenseNet-121 3-Scan Feature Fusion with Supervised Contrastive Loss)  
**Checkpoints Audited:**
- `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5`
- `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5`
- `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5`
- `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5`
- `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5`

**Target Cohort:** 80 Development Participants (240 Scans) from `reports/phase3_participant_benchmark/participant_split.csv`  
**Cross-Validation Protocol:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`  

---

## 1. Provenance & Cross-Validation Traceability

We traced the origin of the E2 model checkpoints through:
- `scripts/experiment6_fusion_supcon.py`
- `scripts/provenance/audit_e2_ensemble.py`
- `reports/model_fusion/all_models_participant_probs.npz`

### 1.1 Training & Validation Partition Verification
The 80 development participants were split into 5 stratified groups. Each participant belongs to exactly one held-out validation fold:

| Fold Index | Training Participants | Validation Participants | Participant Overlap | Checkpoint Verified | Fold Val Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 64 | 16 | **0 (Zero)** | `fold_1_best.weights.h5` | 43.75% (7/16) |
| **Fold 2** | 64 | 16 | **0 (Zero)** | `fold_2_best.weights.h5` | 50.00% (8/16) |
| **Fold 3** | 64 | 16 | **0 (Zero)** | `fold_3_best.weights.h5` | 50.00% (8/16) |
| **Fold 4** | 64 | 16 | **0 (Zero)** | `fold_4_best.weights.h5` | 56.25% (9/16) |
| **Fold 5** | 64 | 16 | **0 (Zero)** | `fold_5_best.weights.h5` | 68.75% (11/16) |
| **Total / Mean** | — | **80 Unique** | **0 Across All Folds** | **5 Checkpoints** | **53.75% (43/80)** |

### 1.2 Zero-Leakage Verdict
1. **Zero Participant Overlap:** In every fold, `set(train_participants) ∩ set(val_participants) == ∅`.
2. **True Out-Of-Fold Execution:** Every participant's prediction was generated exclusively by the model checkpoint trained on the remaining 4 folds. No participant was ever evaluated on a checkpoint that saw their fingerprints during gradient descent.
3. **No Test Set Access:** All 21 held-out test participants in `participant_split.csv` remained untouched throughout model training, checkpoint selection, and out-of-fold inference.

---

## 2. Recomputed E2 Development Metrics (80 Participants)

All metrics were recomputed directly from the out-of-fold prediction array and verified against the ground truth labels:

### 2.1 Cohort Class Distribution
- **A+ (Class 0):** 19 participants (23.75%)
- **B+ (Class 1):** 25 participants (31.25%)
- **AB+ (Class 2):** 7 participants (8.75%)
- **O+ (Class 3):** 29 participants (36.25%)
- **Total:** 80 participants (100.00%)

### 2.2 Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

$$\text{Class Ordering: } [0: \text{A+}, \; 1: \text{B+}, \; 2: \text{AB+}, \; 3: \text{O+}]$$

| True \ Pred | Pred A+ (0) | Pred B+ (1) | Pred AB+ (2) | Pred O+ (3) | Row Total (True) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True A+ (0)** | **12** | 3 | 0 | 4 | **19** |
| **True B+ (1)** | 8 | **14** | 0 | 3 | **25** |
| **True AB+ (2)** | 2 | 2 | **1** | 2 | **7** |
| **True O+ (3)** | 10 | 3 | 0 | **16** | **29** |
| **Col Total (Pred)** | **32** | **22** | **1** | **25** | **80** |

### 2.3 Performance Metrics & Consistency Verification
- **Total Correct (Diagonal):** $12 + 14 + 1 + 16 = \mathbf{43}$
- **Overall Out-Of-Fold Accuracy:** $\frac{43}{80} = \mathbf{0.5375}$ (**53.75%**)
- **Macro F1 Score:** $\mathbf{0.4772}$
- **Balanced Accuracy:** $\mathbf{0.4715}$

| Class | Support | Correct (Diag) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A+** | 19 | 12 | 0.3750 (12/32) | 0.6316 (12/19) | 0.4706 |
| **B+** | 25 | 14 | 0.6364 (14/22) | 0.5600 (14/25) | 0.5957 |
| **AB+** | 7 | 1 | 1.0000 (1/1) | 0.1429 (1/7) | 0.2500 |
| **O+** | 29 | 16 | 0.6400 (16/25) | 0.5517 (16/29) | 0.5926 |
| **Macro Average** | **80** | **43** | **0.6629** | **0.4715** | **0.4772** |

### 2.4 Consistency Checks
- $\sum_{i,j} C_{i,j} = 80 \equiv 80$ (**PASSED**).
- $\text{Trace}(C) / 80 = 43 / 80 = 0.5375 \equiv \text{Accuracy}$ (**PASSED**).
- Per-class recall equals diagonal divided by row total (**PASSED**).
- Macro F1 equals mean of per-class F1 scores ($0.4772$) (**PASSED**).
- Balanced accuracy equals mean of per-class recalls ($0.4715$) (**PASSED**).
- Exact match with saved probability matrix in `all_models_participant_probs.npz` (**CONFIRMED**).

---

## 3. Preprocessing, Class Order, & Architectural Pipeline Audit

1. **Preprocessing Pipeline (C5):**
   - Each raw BMP image is processed via: `composite_alpha_on_white` $\rightarrow$ `crop_mfs100_foreground` $\rightarrow$ `enhance_mfs100_ridge_contrast` $\rightarrow$ `resize_to_target((112, 112))`.
   - Output array is float32 with values in $[0, 255]$.
2. **Three-Scan Input Structure:**
   - E2 requires exactly 3 scans per participant: shape $(3, 112, 112, 3)$.
   - During training, data augmentation was applied with `training=False` during evaluation to avoid non-deterministic perturbation.
   - Multi-tower feature extraction routes each of the 3 scans through a shared frozen DenseNet-121 backbone, followed by global average pooling and feature concatenation (3072-dimensional representation).
3. **Class Order Consistency:**
   - Model classifier head: `Dense(4, activation="softmax", name="clf")`.
   - Target mapping: $0 \rightarrow \text{A+}, \; 1 \rightarrow \text{B+}, \; 2 \rightarrow \text{AB+}, \; 3 \rightarrow \text{O+}$.
   - Verified to match ground truth encoding exactly.

---

## 4. Audit Verdict

The reported **53.75% accuracy and 0.4772 Macro F1** for E2 on the development cohort are **scientifically valid, reproducible, and mathematically verified**. The predictions are strictly out-of-fold with zero participant leakage across all five folds.
