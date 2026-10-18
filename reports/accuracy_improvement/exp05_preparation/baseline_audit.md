# EXP-05 Preparation: Baseline & Architectural Provenance Audit

**Audit Date:** 2026-10-15T09:25:00+05:30  
**Scope:** Verification of Existing E2 Architecture, Training Dynamics, Preprocessing, Checkpoint Integrity, and Baseline Metrics Prior to EXP-05 Execution  
**Audit Finding:** **PASS — BASELINE RIGOROUSLY VERIFIED & PROTECTED**  

---

## 1. Executive Summary

This audit establishes the empirical, architectural, and mathematical baseline for **EXP-05: Controlled Three-Scan Feature Fusion**. EXP-05 investigates whether learning adaptive inter-scan representations (via Gated Feature Fusion or Cross-Scan Self-Attention) improves biometric blood group classification over the baseline concatenation approach, while keeping the pretrained DenseNet-121 backbone strictly frozen.

Before introducing any architectural modifications or scripts, this document verifies all prior experiment artifacts, source scripts, loss definitions, preprocessing pipelines, participant splits, and cryptographic checkpoint hashes.

---

## 2. Source Code & Pipeline Inspection

The following files and scripts were inspected directly on disk:
1. `scripts/experiment6_fusion_supcon.py`: Original E2 champion training script.
2. `experiments/exp04/run_exp04.py`: EXP-04 partial fine-tuning script (validated dual-optimizer, canonical class mapping, and I/O safety).
3. `experiments/exp03/run_exp03.py`: EXP-03 data augmentation comparison script.
4. `src/preprocessing.py`: Single source of truth for `transform_c5` preprocessing.
5. `reports/phase3_participant_benchmark/participant_split.csv`: Approved development split manifest.
6. `reports/accuracy_improvement/exp03/experiment_config.json` & `final_report.md`: EXP-03 verified results.
7. `reports/accuracy_improvement/exp04/experiment_config.json` & `final_report.md`: EXP-04 verified results.

---

## 3. Verified Baseline Architecture & Tensor Dimensions

### 3.1 Input & Feature Extraction
- **Input Tensor:** Shape `(batch_size, 3, 112, 112, 3)`, dtype `float32`, value scale `[0.0, 255.0]`.
  - Slice 0: Scan 1 `(batch_size, 112, 112, 3)`
  - Slice 1: Scan 2 `(batch_size, 112, 112, 3)`
  - Slice 2: Scan 3 `(batch_size, 112, 112, 3)`
- **Backbone Extractor:** `tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")`.
  - Backbone trainability: Strictly frozen (`base.trainable = False`).
  - Total backbone parameters: **7,037,504** non-trainable parameters.
- **Spatial Pooling:** `tf.keras.layers.GlobalAveragePooling2D(name="gap")` applied independently to each scan's feature map.
  - Scan 1 embedding ($F_1$): `(batch_size, 1024)`
  - Scan 2 embedding ($F_2$): `(batch_size, 1024)`
  - Scan 3 embedding ($F_3$): `(batch_size, 1024)`

### 3.2 Baseline Concatenation Neck (Config A Reference)
- **Concatenation Layer:** `tf.keras.layers.Concatenate(name="fused_features")([F_1, F_2, F_3])`.
  - Fused representation shape: `(batch_size, 3072)`.

### 3.3 Multi-Task Output Heads
- **Classification Head:**
  - `Dense(512, activation="relu", name="dense_512")`: Weight shape `(3072, 512)`, bias `(512,)` $\to$ **1,573,376 parameters**.
  - `Dropout(0.5, name="dropout_0.5")`.
  - `Dense(4, activation="softmax", name="clf")`: Weight shape `(512, 4)`, bias `(4,)` $\to$ **2,052 parameters**.
- **Supervised Contrastive Embedding Head:**
  - `Dense(128, activation=None, name="emb")`: Connected directly to `fused_features`.
  - Weight shape `(3072, 128)`, bias `(128,)` $\to$ **393,344 parameters**.
- **Total Trainable Head Parameters (Baseline):**
  $$1,573,376 + 2,052 + 393,344 = \mathbf{1,968,772}\text{ parameters}.$$

---

## 4. Preprocessing & Augmentation Protocol

- **Preprocessing Function:** `src.preprocessing.transform_c5`
  1. Alpha channel compositing onto clean white background (`rgb = alpha * img + (1 - alpha) * 255`).
  2. Foreground bounding box segmentation and cropping (`crop_mfs100_foreground`).
  3. Ridge contrast enhancement (`enhance_mfs100_ridge_contrast` via CLAHE clipLimit=2.5, tileGrid=(8,8), percentile stretch, and bilateral filter).
  4. High-quality bicubic resize to target shape `(112, 112)`.
  5. Scaled as raw `float32` in range `[0.0, 255.0]`. No zero-centering or ImageNet normalization (matching original E2 training).
- **Data Augmentation:** **None (Unaugmented)**. As established by EXP-03, synthetic geometric distortion degrades ridge frequency fidelity on the Mantra MFS100 optical sensor.

---

## 5. Loss Formulation & Sample Weighting Rules

### 5.1 Multi-Task Objective
$$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{SCCE}}(y, \hat{y}_{\text{clf}}) + 0.1 \times \mathcal{L}_{\text{SupCon}}(y, z_{\text{emb}})$$

1. **Classification Loss ($\mathcal{L}_{\text{SCCE}}$):** Sparse Categorical Crossentropy on softmax output $\hat{y}_{\text{clf}}$.
2. **Supervised Contrastive Loss ($\mathcal{L}_{\text{SupCon}}$):**
   - Temperature parameter: $\tau = 0.07$.
   - Embeddings $z_{\text{emb}} \in \mathbb{R}^{B \times 128}$ normalized via $\ell_2$-norm: $\tilde{z}_i = \frac{z_i}{\|z_i\|_2}$.
   - Self-similarity excluded from denominator via `tf.eye(batch_size)`.
   - Stability regularized with `reduce_max` subtraction.
   - Numerator and denominator protected against zero-division via $\epsilon = 10^{-12}$.

### 5.2 Sample Weighting Rules
- Inverse-class frequency weights computed per training fold:
  $$w_c = \frac{N_{\text{train}}}{4 \times N_c}$$
- Applied strictly to the classification loss branch via `sample_weight=[sw, np.ones_like(sw)]`.
- Contrastive loss branch receives unweighted unity weights (`ones_like(sw)`).
- Weights are multiplied exactly once inside Keras loss reduction.

---

## 6. Class Ordering & Canonical Mapping

- **Native E2 Model Order:** `["A+", "B+", "AB+", "O+"]`
  - Index 0: `A+`
  - Index 1: `B+`
  - Index 2: `AB+`
  - Index 3: `O+`
- **Canonical Project Order:** `["A+", "AB+", "B+", "O+"]`
  - Index 0: `A+`
  - Index 1: `AB+`
  - Index 2: `B+`
  - Index 3: `O+`
- **Mapping Transform:**
  $$P_{\text{canonical}} = [P_{\text{native}}[0], P_{\text{native}}[2], P_{\text{native}}[1], P_{\text{native}}[3]]$$
  Followed by sum-normalization ($P_{\text{canonical}} / \sum P_{\text{canonical}}$) to preserve exact probability distribution integrity.

---

## 7. Dataset Manifest & Quarantine Verification

- **Manifest Path:** `reports/phase3_participant_benchmark/participant_split.csv`
- **Development Cohort:** Exactly 80 participants (`split.isin(["TRAIN", "VALIDATION"])`), 240 scans (exactly 3 scans per participant).
  - Class distribution: O+: 29, B+: 25, A+: 19, AB+: 7.
- **Cross-Validation Split:** 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `participant_id`.
  - Exactly 64 participants (192 scans) in training per fold.
  - Exactly 16 participants (48 scans) in validation per fold.
  - Zero participant leakage across folds verified.
- **Strict Quarantine:**
  - Locked 21-participant test set (63 scans) strictly excluded.
  - Separate 26-participant hospital cohort (78 scans) strictly excluded.
  - Verification confirmed zero overlapping participant IDs.

---

## 8. Historical Baseline Performance Reference

| Experiment & Arm | Architecture Description | Accuracy | Balanced Accuracy | Macro F1 | AB+ Recall | Source Artifact |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **EXP-03 Arm 1** | E2 Concatenation (Unaugmented) | **53.75%** | **52.84%** | **0.5068** | 57.14% (4/7) | `reports/accuracy_improvement/exp03/final_report.md` |
| **EXP-04 Config A** | E2 Concatenation (Frozen Backbone) | **50.00%** | **47.86%** | **0.4781** | 42.86% (3/7) | `reports/accuracy_improvement/exp04/final_report.md` |
| **EXP-04 Config B** | E2 DenseBlock 4 Partial FT ($10^{-4}$) | 41.25% | 42.97% | 0.4032 | 57.14% (4/7) | `reports/accuracy_improvement/exp04/final_report.md` |
| **EXP-04 Config C** | E2 DenseBlocks 3+4 Deep FT ($5\times 10^{-5}$) | 47.50% | 39.04% | 0.3789 | 14.29% (1/7) | `reports/accuracy_improvement/exp04/final_report.md` |

*Note: EXP-05 Configuration A will re-run the concatenation baseline under the exact EXP-05 execution script to establish a contemporaneous, protocol-matched reference.*

---

## 9. Checkpoint Integrity Ledger (26 Pre-Existing Assets Verified)

All 26 pre-existing model weights and production assets were verified with bit-for-bit SHA-256 integrity:

| Checkpoint Identifier | File Path | File Size | Verified SHA-256 Hash |
|:---|:---|:---:|:---|
| **Orig E2 Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | 52,910,112 | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` |
| **Orig E2 Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | 52,910,112 | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` |
| **Orig E2 Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | 52,910,112 | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` |
| **Orig E2 Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | 52,910,112 | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` |
| **Orig E2 Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | 52,910,112 | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` |
| **EXP-03 Arm 1 Fold 1** | `experiments/exp03/checkpoints/arm1/fold_1_best.weights.h5` | 52,910,112 | `0d047b7a10153718b067e84a27a6ad2bc0486ccf1d35057a07da6e2e2d627ac5` |
| **EXP-03 Arm 1 Fold 2** | `experiments/exp03/checkpoints/arm1/fold_2_best.weights.h5` | 52,910,112 | `dd94e4c44887399d700ac2de195f513a5b1db73ac561f4ff8c2345d5156f7063` |
| **EXP-03 Arm 1 Fold 3** | `experiments/exp03/checkpoints/arm1/fold_3_best.weights.h5` | 52,910,112 | `f763f67068a4411dbd4a405e39248f2ead1b1b1fe228c4b1bca5da807005b3e3` |
| **EXP-03 Arm 1 Fold 4** | `experiments/exp03/checkpoints/arm1/fold_4_best.weights.h5` | 52,910,112 | `589f1e52275f68ed02329b1d4a7a9c16f406ed3868b8afeef9706ebace61b496` |
| **EXP-03 Arm 1 Fold 5** | `experiments/exp03/checkpoints/arm1/fold_5_best.weights.h5` | 52,910,112 | `848aea5498e469871d5fdcd136d849f971c9c3c485a4c66afb2505d3b0a46776` |
| **Production Model** | `models/abo_positive_4class/best_model.keras` | 61,023,607 | `7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9` |
| **EXP-04 Config A Fold 1** | `experiments/exp04/checkpoints/config_a/fold_1_best.weights.h5` | 60,779,864 | `5d2157aeee17cc7dbeb27260af5e6fce3ae8f0871bf8aa75fa8b6aee82687dcf` |
| **EXP-04 Config A Fold 2** | `experiments/exp04/checkpoints/config_a/fold_2_best.weights.h5` | 60,779,864 | `9c4388b6d506776df5d1c99a557bfa62a9391a56357c1ad56cd3dc4f07d732f9` |
| **EXP-04 Config A Fold 3** | `experiments/exp04/checkpoints/config_a/fold_3_best.weights.h5` | 60,779,864 | `8793741a9ed54abc91dfa1e7f3fa23a1797a16ed71c1801bf52abf60ba2e9a7d` |
| **EXP-04 Config A Fold 4** | `experiments/exp04/checkpoints/config_a/fold_4_best.weights.h5` | 60,779,864 | `4a24b9713129974f6e75b951d7fa91818021517e5dc8af8fb428d37d3bbf3abe` |
| **EXP-04 Config A Fold 5** | `experiments/exp04/checkpoints/config_a/fold_5_best.weights.h5` | 60,779,864 | `e0c8a4135a70b58ed31dde493d9daf3cd62c88fcd0c9dfc04eced334d100d40e` |
| **EXP-04 Config B Fold 1** | `experiments/exp04/checkpoints/config_b/fold_1_best.weights.h5` | 86,376,168 | `532332b72300223d59a04e26eb02ac6a456974431a7dc483cb1edfa993d1ac95` |
| **EXP-04 Config B Fold 2** | `experiments/exp04/checkpoints/config_b/fold_2_best.weights.h5` | 86,376,168 | `ff2cd8e0a7dee42447585e59d0145ec4ed5d15c9c59caa8d607c2c7cc47d258f` |
| **EXP-04 Config B Fold 3** | `experiments/exp04/checkpoints/config_b/fold_3_best.weights.h5` | 86,376,168 | `9967b27ba6b081314d027cc8436fba5f7b4e30528f47a15211ca44539e4bd16b` |
| **EXP-04 Config B Fold 4** | `experiments/exp04/checkpoints/config_b/fold_4_best.weights.h5` | 86,376,168 | `e2ae4b1c56091f43f80443878da30f4dd7479c360a5b410038122995c03e1b00` |
| **EXP-04 Config B Fold 5** | `experiments/exp04/checkpoints/config_b/fold_5_best.weights.h5` | 86,376,168 | `a07ceb2d0bf09cf33333b7eecc267a41443fca6c3de70a63ae5769d1837753ec` |
| **EXP-04 Config C Fold 1** | `experiments/exp04/checkpoints/config_c/fold_1_best.weights.h5` | 126,342,472 | `e9fc3b82c4cec6dc76783d8d5f40691346d6452bd50b729f9e50318cf1189fa0` |
| **EXP-04 Config C Fold 2** | `experiments/exp04/checkpoints/config_c/fold_2_best.weights.h5` | 126,342,472 | `c38f42a0b40037136e4b1519768eb8cb773b30fb9cf42eab7d353eddc0d8e691` |
| **EXP-04 Config C Fold 3** | `experiments/exp04/checkpoints/config_c/fold_3_best.weights.h5` | 126,342,472 | `b733d834faa95a24b989c8166a74c9a9e41b85a0b6bba17697c3cff999493b84` |
| **EXP-04 Config C Fold 4** | `experiments/exp04/checkpoints/config_c/fold_4_best.weights.h5` | 126,342,472 | `826edb9a3383cc6d9f4b1a9bbfdb6282642c0faa2c1190d09c39e208be52b87a` |
| **EXP-04 Config C Fold 5** | `experiments/exp04/checkpoints/config_c/fold_5_best.weights.h5` | 126,342,472 | `6d6f0a3418eb1273ef1bbceab0a3bab73e9a9dfe818da3977229a77978cb51ab` |

---

## 10. Dedicated Output Structure for EXP-05

To prevent overwriting existing assets, EXP-05 outputs will reside strictly in dedicated paths:
- Preparation Audits: `reports/accuracy_improvement/exp05_preparation/`
- Training Checkpoints: `experiments/exp05/checkpoints/{config_a,config_b,config_c}/`
- Experiment Reports & OOF CSVs: `reports/accuracy_improvement/exp05/`
- Synthetic Dry Run Tests: `scratch/exp05_synthetic/`
