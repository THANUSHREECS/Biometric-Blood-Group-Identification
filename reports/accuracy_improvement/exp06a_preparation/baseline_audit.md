# EXP-06A Baseline Audit: Reference Fidelity & Source Checkpoint Verification

**Document Version:** 1.0.0  
**Date of Audit:** 2026-10-15  
**Audit Scope:** Independent Verification of EXP-05 Configuration A as the Experimental Reference for EXP-06A  
**Auditor:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> **NON-CLINICAL ACADEMIC RESEARCH DISCLAIMER**  
> This project is an exploratory computer vision investigation evaluating metric learning and biometric feature representations. The models and metrics discussed herein are research prototypes and **must not** be used for medical diagnosis, blood transfusion decision-making, emergency care, or clinical laboratory blood grouping.

---

## 1. Objective of Baseline Audit

The objective of **EXP-06A** is to investigate whether altering the **supervised contrastive temperature** ($\tau \in \{0.05, 0.07, 0.10, 0.15\}$) within the multi-task loss objective improves four-class fingerprint classification performance.

Before conducting this ablation, this audit independently verifies:
1. The exact mathematical and empirical metrics of the designated reference architecture: **EXP-05 Configuration A (Concatenation Baseline)**.
2. The integrity and exact implementation of all architectural layers, loss formulations, sample-weighting conventions, and checkpoint selection policies in `experiments/exp05/run_exp05.py`.
3. The exact operational locus of the supervised contrastive loss (confirming whether it operates on individual scan embeddings or the fused participant embedding).
4. The immutability and cryptographic checksums of all pre-existing baseline checkpoints across the project.

---

## 2. Independent Verification of Reference Metrics

The reference metrics reported in the EXP-05 final documentation were independently recomputed directly from the raw out-of-fold predictions in `reports/accuracy_improvement/exp05/config_a_oof_predictions.csv` and cross-referenced with `reports/accuracy_improvement/exp05/metrics_comparison.csv`:

| Evaluation Metric | Reported Reference Value | Independently Recomputed Value | Discrepancy ($\Delta$) | Verification Status |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | **52.50%** (42/80) | **52.5000%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **Balanced Accuracy** | **51.84%** | **51.8356%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **Macro F1-Score (Primary)** | **0.5107** | **0.510734** | $\pm 0.0000$ | **VERIFIED EXACT** |
| **Macro Precision** | **50.49%** | **50.4902%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **Macro Recall** | **51.84%** | **51.8356%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **A+ Recall** ($N=19$) | **31.58%** (6/19) | **31.5789%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **AB+ Recall** ($N=7$) | **57.14%** (4/7) | **57.1429%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **B+ Recall** ($N=25$) | **60.00%** (15/25) | **60.0000%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **O+ Recall** ($N=29$) | **58.62%** (17/29) | **58.6207%** | $\pm 0.0000\%$ | **VERIFIED EXACT** |
| **Cross-Fold F1 Mean** | 0.4579 | 0.457870 | $\pm 0.0000$ | **VERIFIED EXACT** |
| **Cross-Fold F1 Std** | 0.1148 | 0.114771 | $\pm 0.0000$ | **VERIFIED EXACT** |

### Verified Reference Confusion Matrix (EXP-05 Config A)
```
Predicted ──►    A+      AB+      B+      O+    │ Total │ Recall
True Label ┌────────────────────────────────────┼───────┼─────────
    A+     │     6        1       6       6     │   19  │ 31.58%
    AB+    │     1        4       1       1     │    7  │ 57.14%
    B+     │     3        1      15       6     │   25  │ 60.00%
    O+     │     7        2       3      17     │   29  │ 58.62%
───────────┴────────────────────────────────────┴───────┴─────────
Total Pred │    17        8      25      30     │   80  │ Acc: 52.5%
```

---

## 3. Architecture & Loss Implementation Audit

Inspection of `experiments/exp05/run_exp05.py` confirms the following implementation details:

### 3.1 Frozen DenseNet-121 Backbone
- **Base Architecture:** `tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")`.
- **Trainability Status:** `base.trainable = False`.
- **Parameter Audit:**
  - Non-Trainable Backbone Parameters: **7,037,504**
  - Trainable Backbone Parameters: **0**
  - Trainable Batch Normalization Layers: **0** (All BN layers strictly in non-trainable mode).
- **Scan Inputs:** 3 inputs of shape $(112, 112, 3)$, preprocessed with aspect-preserving resize, CLAHE contrast enhancement, and zero geometric augmentation (`transform_c5`).

### 3.2 Concatenation Fusion Neck & Projection Head
- **Global Pooling:** `GlobalAveragePooling2D` applied to each scan output: $f_1, f_2, f_3 \in \mathbb{R}^{1024}$.
- **Fusion Layer:** `tf.keras.layers.Concatenate(name="fused_features")([f1, f2, f3])` producing fused vector $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{3072}$.
- **Classification Sub-Head:**
  - `Dense(512, activation="relu", name="dense_512")`
  - `Dropout(0.5, name="dropout_0.5")`
  - `Dense(4, activation="softmax", name="clf")`
- **Embedding Sub-Head:**
  - `Dense(128, activation=None, name="emb")` branched directly from `fused_features` ($\mathbf{z}_{\text{fused}} \in \mathbb{R}^{3072}$).
- **Trainable Parameter Audit:**
  - `dense_512`: $3072 \times 512 + 512 = 1,573,376$ params (kernel: `(3072, 512)`, bias: `(512,)`)
  - `clf`: $512 \times 4 + 4 = 2,052$ params (kernel: `(512, 4)`, bias: `(4,)`)
  - `emb`: $3072 \times 128 + 128 = 393,344$ params (kernel: `(3072, 128)`, bias: `(128,)`)
  - **Total Trainable Parameters:** $1,573,376 + 2,052 + 393,344 = \mathbf{1,968,772}$ across exactly **6 variables**.

### 3.3 Contrastive Loss Operational Locus
> [!IMPORTANT]
> **VERIFIED OPERATIONAL LOCUS:**  
> In `experiments/exp05/run_exp05.py`, the embedding sub-head is defined as `out_emb = emb(fused)` and computed as a single 128-dimensional vector per participant. The supervised contrastive loss function `supervised_contrastive_loss(y_emb, pred_emb, temperature=SUPCON_TEMP)` operates on **one fused participant embedding**, comparing participants across the mini-batch ($B=16$), **not** on individual scan embeddings.  
> 
> In EXP-06A, this exact formulation is strictly preserved. Only the scalar hyperparameter `temperature` ($\tau$) is varied.

### 3.4 Multi-Task Loss & Sample Weighting
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{CE}}(\hat{y}_{\text{clf}}, y; w_{\text{class}}) + \lambda_{\text{supcon}} \mathcal{L}_{\text{SupCon}}(\hat{z}_{\text{emb}}, y; \tau)$$
- $\lambda_{\text{supcon}} = 0.1$
- Reference temperature $\tau_{\text{ref}} = 0.07$
- Inverse class sample weighting:
  $$w_c = \frac{N_{\text{train}}}{4 \cdot N_c}$$
  applied strictly and exclusively to the categorical crossentropy loss $\mathcal{L}_{\text{CE}}$, while the contrastive loss $\mathcal{L}_{\text{SupCon}}$ remains unweighted.

---

## 4. Protected Source Checkpoint Ledger

All pre-existing project checkpoints were verified for immutability prior to beginning EXP-06A:

| Checkpoint Identifier | File Location | File Size | Verified SHA-256 Checksum | Integrity Status |
|:---|:---|:---:|:---:|:---:|
| **Production Model** | `models/abo_positive_4class/best_model.keras` | 32,642,427 | `7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9` | **100% MATCH** |
| **Original E2 Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | 52,910,112 | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` | **100% MATCH** |
| **Original E2 Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | 52,910,112 | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` | **100% MATCH** |
| **Original E2 Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | 52,910,112 | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` | **100% MATCH** |
| **Original E2 Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | 52,910,112 | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` | **100% MATCH** |
| **Original E2 Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | 52,910,112 | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` | **100% MATCH** |
| **EXP-03 Arm 1 Fold 1** | `experiments/exp03/checkpoints/arm1/fold_1_best.weights.h5` | 52,910,112 | `0d047b7a10153718b067e84a27a6ad2bc0486ccf1d35057a07da6e2e2d627ac5` | **100% MATCH** |
| **EXP-03 Arm 1 Fold 2** | `experiments/exp03/checkpoints/arm1/fold_2_best.weights.h5` | 52,910,112 | `dd94e4c44887399d700ac2de195f513a5b1db73ac561f4ff8c2345d5156f7063` | **100% MATCH** |
| **EXP-03 Arm 1 Fold 3** | `experiments/exp03/checkpoints/arm1/fold_3_best.weights.h5` | 52,910,112 | `f763f67068a4411dbd4a405e39248f2ead1b1b1fe228c4b1bca5da807005b3e3` | **100% MATCH** |
| **EXP-03 Arm 1 Fold 4** | `experiments/exp03/checkpoints/arm1/fold_4_best.weights.h5` | 52,910,112 | `589f1e52275f68ed02329b1d4a7a9c16f406ed3868b8afeef9706ebace61b496` | **100% MATCH** |
| **EXP-03 Arm 1 Fold 5** | `experiments/exp03/checkpoints/arm1/fold_5_best.weights.h5` | 52,910,112 | `848aea5498e469871d5fdcd136d849f971c9c3c485a4c66afb2505d3b0a46776` | **100% MATCH** |
| **EXP-05 Config A Fold 1** | `experiments/exp05/checkpoints/config_a/fold_1_best.weights.h5` | 52,899,216 | `52bd4a65becdb6146a338f1a55fe73b6b4722ef4ba93d8a924b5d1a0081bd95a` | **100% MATCH** |
| **EXP-05 Config A Fold 2** | `experiments/exp05/checkpoints/config_a/fold_2_best.weights.h5` | 52,899,216 | `a3935aca244e471a98166684be75d7ba00736c05a59212eca43db4ddb2ccdf36` | **100% MATCH** |
| **EXP-05 Config A Fold 3** | `experiments/exp05/checkpoints/config_a/fold_3_best.weights.h5` | 52,899,216 | `ba355773dd0b23bcde90300b29e4fb4e7cc0dbd2d99d9f115cf84ab15bc4a6b8` | **100% MATCH** |
| **EXP-05 Config A Fold 4** | `experiments/exp05/checkpoints/config_a/fold_4_best.weights.h5` | 52,899,216 | `0a2f7a859f3b956564a04211945ee92c7cea9437e7822ecc37a0e376862e8cce` | **100% MATCH** |
| **EXP-05 Config A Fold 5** | `experiments/exp05/checkpoints/config_a/fold_5_best.weights.h5` | 52,899,216 | `d1404d789c04dfeb09b27564f67c78e7b3cb722a06766999d4efa7493fb268dc` | **100% MATCH** |
| **EXP-05 Config A OOF CSV** | `reports/accuracy_improvement/exp05/config_a_oof_predictions.csv` | 7,364 | `379aa03fb219076818a1a9e334df59eef7ff5068222a075e0108340d8692634e` | **100% MATCH** |

---

## 5. Audit Conclusion

1. **Reference Values Authenticated:** EXP-05 Configuration A metrics (Macro F1 = **0.5107**, Balanced Accuracy = **51.84%**, Accuracy = **52.50%**) are certified as exact mathematical baselines.
2. **Loss Semantics Preserved:** Supervised contrastive loss operates on the 128-d fused participant embedding, and sample weighting applies strictly to classification crossentropy.
3. **Cryptographic Protection Confirmed:** All 17 protected checkpoints and reference artifacts remain unaltered.
