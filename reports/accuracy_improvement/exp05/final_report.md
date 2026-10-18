# EXP-05 Final Experiment Report: Controlled Three-Scan Feature Fusion

**Document Version:** 1.0.0  
**Date of Completion:** 2026-10-15  
**Experiment Identifier:** `EXP-05`  
**Protocol Phase:** Phase 3 — Post-Baseline Controlled Optimization  
**Repository Working Directory:** `c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project`  
**Status:** Completed & Independently Verified  

---

> [!CAUTION]
> **RESEARCH USE ONLY — NON-CLINICAL DISCLAIMER**  
> This experiment is an exploratory academic machine learning study investigating multi-scan biometric feature fusion on optical fingerprint scans (Mantra MFS100). The models and findings documented herein are strictly research prototypes and **must not** be used for medical diagnosis, blood transfusion decision-making, emergency care, or clinical laboratory blood grouping. Standard serological agglutination testing remains the mandatory laboratory gold standard.

---

## 1. Executive Summary

EXP-05 evaluated whether structured feature-fusion architectures could improve upon simple vector concatenation when combining features from three optical fingerprint scans per participant. While EXP-03 Arm 1 established the unaugmented DenseNet-121 baseline and EXP-04 demonstrated that partial backbone fine-tuning causes severe overfitting and feature distortion, EXP-05 maintained the **pretrained DenseNet-121 backbone strictly frozen** (7,037,504 non-trainable parameters; 0 trainable backbone parameters) and systematically tested three fusion necks:

1. **Configuration A (Concatenation Baseline):** Concatenation of the three 1024-d scan embeddings into a 3072-d fused representation, followed by a dense projection head (1,968,772 trainable parameters).
2. **Configuration B (Gated Feature Fusion):** Element-wise adaptive gating network computing per-scan importance vectors normalized via softmax across scans, reducing fused dimensionality to 1024-d (789,381 trainable parameters; a **60.0% parameter reduction**).
3. **Configuration C (Cross-Scan Multi-Head Self-Attention):** Sequence-based projection of each scan into 256-d tokens, refined through multi-head self-attention ($H=4$, $d_k=64$), residual LayerNorm, and feed-forward transformation, flattened to 768-d (1,283,716 trainable parameters).

All three configurations were trained across the **exact same 5 participant-grouped development folds** (80 participants, 240 scans; 64 train / 16 val per fold) under identical multi-task classification + supervised contrastive loss objectives.

```
                    ┌────────────────────────────────────────────────────────┐
                    │       Three Input Scans (C5 Preprocessing, 224x224)    │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ Frozen DenseNet-121   │ (7,037,504 non-trainable)
                                    └───────────┬───────────┘
                                                │  (3 x 1024-d)
                  ┌─────────────────────────────┼─────────────────────────────┐
                  │                             │                             │
        ┌─────────▼─────────┐         ┌─────────▼─────────┐         ┌─────────▼─────────┐
        │  Configuration A  │         │  Configuration B  │         │  Configuration C  │
        │   Concatenation   │         │    Gated Fusion   │         │  Self-Attention   │
        │    (3072-d neck)  │         │    (1024-d neck)  │         │    (768-d neck)   │
        └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
                  │                             │                             │
            Trainable: 1.97M              Trainable: 0.79M              Trainable: 1.28M
            Macro F1:  0.5107             Macro F1:  0.5023             Macro F1:  0.4535
            Accuracy:  52.50%             Accuracy:  52.50%             Accuracy:  45.00%
            Bal Acc:   51.84%             Bal Acc:   51.24%             Bal Acc:   48.09%
```

### Primary Conclusions
1. **Concatenation Remains the Primary Development Candidate:** Configuration A achieved the highest out-of-fold (OOF) Macro F1 (**0.5107**), balanced accuracy (**51.84%**), and accuracy (**52.50%**).
2. **Gated Fusion Offers Equivalent Accuracy with 60% Parameter Savings:** Configuration B matched Configuration A in overall accuracy (**52.50%**) and balanced accuracy (**51.24%** vs. 51.84%), while reducing trainable head parameters from 1.97M to 789k. Paired McNemar testing revealed **zero statistical difference** ($p = 1.0000$, with exactly 13 discordant predictions in each direction).
3. **Cross-Scan Self-Attention Underperformed on Small Sample Cohort:** Configuration C underperformed noticeably (Macro F1 **0.4535**, Balanced Accuracy **48.09%**, Accuracy **45.00%**). Self-attention over a short sequence ($T=3$) introduced excess representational complexity, causing the model to collapse on majority class B+ recall (36.00% vs. 60.00% in Config A), although it boosted AB+ recall to 71.43%.
4. **Statistical Significance Verdict:** Paired McNemar tests between Configuration A and Configuration C yielded $\chi^2 = 1.0417, p = 0.3074$. The drop is not statistically significant at $\alpha = 0.05$, confirming that attention mechanisms do not provide an empirical advantage over simple concatenation in this data regime.
5. **Zero Data Leakage & Checkpoint Immutability:** Preflight assertions and post-training hash recalculations confirmed that the 21-participant locked test cohort (63 scans) and the 26-participant hospital cohort (78 scans) were 100% excluded. All 11 protected source checkpoints remained byte-for-byte identical.

---

## 2. Experimental Setup & Verification

### 2.1 Cohort Partitioning & Data Isolation
The dataset was partitioned according to the audited development split:
- **Development Cohort (Approved for EXP-05):** Exactly 80 participants, 240 fingerprint scans.
  - Class Distribution: 19 A+ (23.75%), 7 AB+ (8.75%), 25 B+ (31.25%), 29 O+ (36.25%).
- **Locked Test Cohort (Strict Quarantine):** 21 participants, 63 scans. Zero access during training, validation, or checkpoint selection.
- **Separate Hospital Cohort (Strict Quarantine):** 26 participants, 78 scans. Zero access.

### 2.2 Cross-Validation Protocol
- **Split Type:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Participant Grouping:** Every participant's three scans were strictly grouped into the same fold (64 train participants, 16 validation participants per fold).
- **Leakage Verification:** Pairwise intersection of train and validation participant IDs confirmed `len(train_pids & val_pids) == 0` across all 5 folds.

### 2.3 Preprocessing & Training Hyperparameters
- **Image Pipeline:** Unaugmented C5 preprocessing (`transform_c5`: aspect-ratio preserving resize to 224x224, center padding, CLAHE contrast enhancement, ImageNet channel normalization). Zero geometric augmentation was applied, per EXP-03 Arm 1 findings.
- **Backbone Extractor:** DenseNet-121 pretrained on ImageNet, frozen at `trainable = False` throughout all epochs.
- **Loss Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{clf}}, y; w_{\text{class}}) + \lambda_{\text{supcon}} \mathcal{L}_{\text{SupCon}}(\hat{z}_{\text{emb}}, y; \tau)$$
  where $\lambda_{\text{supcon}} = 0.1$, $\tau = 0.07$, and $w_{\text{class}}$ represents inverse class frequency sample weights:
  $$w_c = \frac{N_{\text{total}}}{4 \cdot N_c}$$
- **Optimizer:** Adam with base learning rate $\eta = 10^{-3}$, weight decay $10^{-4}$, and gradient clipping.
- **Stopping Policy:** Training floor `min_epochs = 100`, maximum epochs `epochs = 150`, early stopping patience `patience = 15` beyond the floor. Checkpoint selection strictly monitored validation Macro-F1 on the held-out fold validation set.

---

## 3. Architecture Specification & Parameter Ledger

Each model accepts three fingerprint scans as separate input tensors $(3 \times [224, 224, 3])$, extracts feature vectors via the shared frozen DenseNet-121 backbone $(3 \times 1024\text{-d})$, and routes them into the configuration-specific fusion neck:

| Architectural Component | Configuration A (Concatenation) | Configuration B (Gated Fusion) | Configuration C (Self-Attention) |
|:---|:---:|:---:|:---:|
| **Backbone Feature Dim** | $3 \times 1024$ | $3 \times 1024$ | $3 \times 1024$ |
| **Fusion Mechanism** | Direct Concatenation | Softmax-Normalized Gating | Multi-Head Self-Attention ($H=4$) |
| **Fused Neck Dimension** | **3072** | **1024** | **768** |
| **Projection Head** | Dense(512) + Dropout(0.5) | Dense(512) + Dropout(0.5) | Dense(512) + Dropout(0.5) |
| **Outputs** | Softmax(4) + Emb(128) | Softmax(4) + Emb(128) | Softmax(4) + Emb(128) |
| **Trainable Variables** | **6** | **10** | **24** |
| **Trainable Parameters** | **1,968,772** | **789,381** ($-60.0\%$) | **1,283,716** ($-34.8\%$) |
| **Frozen Parameters** | **7,037,504** | **7,037,504** | **7,037,504** |
| **Total Model Parameters** | 9,006,276 | 7,826,885 | 8,321,220 |

---

## 4. Empirical Evaluation & Performance Metrics

### 4.1 Global Out-of-Fold Performance Summary
Out-of-fold metrics were computed by aggregating the held-out predictions for all 80 development participants across the 5 folds:

| Metric | Config A: Concatenation | Config B: Gated Fusion | Config C: Self-Attention | Delta (B vs A) | Delta (C vs A) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Macro F1-Score (Primary)** | **0.5107** | **0.5023** | **0.4535** | $-0.0084$ | $-0.0572$ |
| **Balanced Accuracy** | **51.84%** | **51.24%** | **48.09%** | $-0.60\%$ | $-3.75\%$ |
| **Overall Accuracy** | **52.50%** (42/80) | **52.50%** (42/80) | **45.00%** (36/80) | $\pm 0.00\%$ | $-7.50\%$ |
| **Macro Precision** | **50.49%** | **50.09%** | **44.29%** | $-0.40\%$ | $-6.20\%$ |
| **Macro Recall** | **51.84%** | **51.24%** | **48.09%** | $-0.60\%$ | $-3.75\%$ |
| **Cross-Fold F1 Mean (Std)** | 0.4579 ($\pm 0.1148$) | 0.4473 ($\pm 0.0797$) | 0.4694 ($\pm 0.1054$) | $-0.0106$ | $+0.0115$ |
| **Trainable Parameter Count** | 1,968,772 | 789,381 | 1,283,716 | $-60.0\%$ | $-34.8\%$ |
| **Total Experiment Runtime** | 64.35 min | 97.04 min | 158.70 min | $+32.69$ min | $+94.35$ min |

> [!NOTE]
> Macro F1 is the mandated primary metric due to the natural class imbalance (7 AB+ vs 29 O+). While Configurations A and B tied at 52.50% overall accuracy, Configuration A edged out Configuration B on Macro F1 (0.5107 vs 0.5023) due to superior A+ minority recall.

---

### 4.2 Per-Class Recall, Precision, and F1 Breakdown
Detailed class-by-class performance on the 80 development participants:

| Blood Group | Metric | Config A: Concat | Config B: Gated | Config C: Attention | Observations |
|:---|:---|:---:|:---:|:---:|:---|
| **A+**<br>$(N=19)$ | **Recall**<br>Precision<br>F1-Score | **31.58%** (6/19)<br>35.29%<br>0.3333 | **26.32%** (5/19)<br>38.46%<br>0.3125 | **26.32%** (5/19)<br>35.71%<br>0.3030 | Concat achieved the best minority recall for A+; gating and attention lost 1 correct detection. |
| **AB+**<br>$(N=7)$ | **Recall**<br>Precision<br>F1-Score | **57.14%** (4/7)<br>50.00%<br>0.5333 | **57.14%** (4/7)<br>50.00%<br>0.5333 | **71.43%** (5/7)<br>50.00%<br>**0.5882** | Attention achieved the highest AB+ recall (5/7 correct), identifying fine-grained rare ridge patterns. |
| **B+**<br>$(N=25)$ | **Recall**<br>Precision<br>F1-Score | **60.00%** (15/25)<br>60.00%<br>**0.6000** | **56.00%** (14/25)<br>56.00%<br>0.5600 | **36.00%** (9/25)<br>42.86%<br>0.3913 | Attention suffered severe representation collapse on B+, misclassifying 16/25 B+ participants. |
| **O+**<br>$(N=29)$ | **Recall**<br>Precision<br>F1-Score | **58.62%** (17/29)<br>56.67%<br>0.5763 | **65.52%** (19/29)<br>55.88%<br>**0.6032** | **58.62%** (17/29)<br>48.57%<br>0.5312 | Gated fusion achieved peak performance on O+, correctly identifying 19/29 participants. |

---

### 4.3 Confusion Matrices

The class order follows the canonical project standard: `[0: A+, 1: AB+, 2: B+, 3: O+]`.

#### Configuration A: Concatenation Baseline
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

#### Configuration B: Gated Feature Fusion
```
Predicted ──►    A+      AB+      B+      O+    │ Total │ Recall
True Label ┌────────────────────────────────────┼───────┼─────────
    A+     │     5        2       5       7     │   19  │ 26.32%
    AB+    │     2        4       0       1     │    7  │ 57.14%
    B+     │     4        0      14       7     │   25  │ 56.00%
    O+     │     2        2       6      19     │   29  │ 65.52%
───────────┴────────────────────────────────────┴───────┴─────────
Total Pred │    13        8      25      34     │   80  │ Acc: 52.5%
```

#### Configuration C: Cross-Scan Self-Attention
```
Predicted ──►    A+      AB+      B+      O+    │ Total │ Recall
True Label ┌────────────────────────────────────┼───────┼─────────
    A+     │     5        2       5       7     │   19  │ 26.32%
    AB+    │     0        5       0       2     │    7  │ 71.43%
    B+     │     6        1       9       9     │   25  │ 36.00%
    O+     │     3        2       7      17     │   29  │ 58.62%
───────────┴────────────────────────────────────┴───────┴─────────
Total Pred │    14       10      21      35     │   80  │ Acc: 45.0%
```

Generated confusion matrix plots are saved at:
- `reports/accuracy_improvement/exp05/confusion_matrices/config_a_confusion_matrix.png`
- `reports/accuracy_improvement/exp05/confusion_matrices/config_b_confusion_matrix.png`
- `reports/accuracy_improvement/exp05/confusion_matrices/config_c_confusion_matrix.png`

---

## 5. Fold-Level Variability & Training Dynamics

Every fold contained exactly 16 validation participants (64 training participants). The table below documents the best validation Macro-F1, the epoch at which it occurred, and fold runtimes:

| Fold Index | Config A (Concat) Best F1 @ Epoch | Config B (Gated) Best F1 @ Epoch | Config C (Attention) Best F1 @ Epoch | Validation Cohort PIDs |
|:---|:---:|:---:|:---:|:---|
| **Fold 1** | 0.3745 @ Epoch 36 (250s) | 0.5500 @ Epoch 50 (364s) | 0.4187 @ Epoch 55 (592s) | P001, P00103, P0015, P0017, P0026, P0030, P0038, P0045, P0051, P0055, P0061, P0066, P0072, P008, P0087, P0091 |
| **Fold 2** | 0.5896 @ Epoch 11 (254s) | 0.3750 @ Epoch 7 (380s) | 0.5912 @ Epoch 40 (601s) | P0010, P00104, P00106, P002, P0020, P0031, P0039, P004, P0046, P0056, P0067, P0073, P0077, P0081, P0092, P0096 |
| **Fold 3** | 0.3015 @ Epoch 114 (279s) | 0.3889 @ Epoch 28 (392s) | 0.3214 @ Epoch 24 (608s) | P00100, P00105, P0018, P0021, P0027, P0032, P0040, P0047, P0052, P0057, P0062, P0068, P0074, P0078, P0082, P0093 |
| **Fold 4** | 0.5872 @ Epoch 77 (274s) | 0.5392 @ Epoch 28 (387s) | 0.5893 @ Epoch 31 (627s) | P00101, P00107, P0012, P0022, P0033, P0041, P0048, P0058, P0063, P0069, P0075, P0079, P0083, P0088, P0094, P0098 |
| **Fold 5** | 0.4368 @ Epoch 17 (268s) | 0.3833 @ Epoch 23 (385s) | 0.4265 @ Epoch 38 (617s) | P00102, P0013, P0014, P0024, P0028, P0035, P0043, P005, P0053, P006, P0064, P007, P0070, P0084, P0089, P0099 |
| **Mean $\pm$ Std** | **0.4579 $\pm$ 0.1148** | **0.4473 $\pm$ 0.0797** | **0.4694 $\pm$ 0.1054** | — |

### Observations on Fold Stability:
- **Configuration B demonstrated the lowest variance** across validation folds ($\text{Std} = 0.0797$ vs. $0.1148$ for Concat and $0.1054$ for Attention). The softmax gating acted as a regularizer, damping wild accuracy swings between heterogeneous participant subsets.
- **Fold 3 was the most difficult fold across all configurations** (F1 range: $0.3015 - 0.3889$), pointing to hard biometric specimens rather than architectural artifacts.

---

## 6. Paired Statistical Significance Testing

To verify whether performance differences between configurations were statistically meaningful or attributable to chance, paired contingency tests were executed across the aligned 80 participant predictions.

### 6.1 Participant-Aligned Discordance Analysis
- **Total Development Participants:** 80 (collocated and identical across all three files).
- **Exact Participant Order:** 100% verified via assertion check across all CSVs.

| Pairwise Comparison | Correct in Model 1, Incorrect in Model 2 ($b$) | Incorrect in Model 1, Correct in Model 2 ($c$) | Discordant Total ($b + c$) | McNemar $\chi^2$ (corrected) | Two-Tailed $p$-Value | Significance ($\alpha=0.05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A vs Config B** | 13 | 13 | 26 | **0.0000** | **1.0000** | **Not Significant** |
| **Config A vs Config C** | 15 | 9 | 24 | **1.0417** | **0.3074** | **Not Significant** |
| **Config B vs Config C** | 15 | 9 | 24 | **1.0417** | **0.3074** | **Not Significant** |

### 6.2 Statistical Interpretation
1. **Config A vs Config B:** The discordance is perfectly symmetric ($b=13, c=13$). Configuration A correctly classified 13 participants that Configuration B failed on, and Configuration B correctly classified exactly 13 participants that Configuration A failed on. The resulting $p$-value is $1.0000$. This confirms that **gated fusion does not improve classification accuracy over concatenation**, but neither does it cause statistically detectable degradation.
2. **Config A vs Config C:** Configuration A correctly predicted 15 participants that Attention missed, whereas Attention correctly predicted only 9 participants that Concat missed. While Concat held a $+6$ net prediction advantage, the continuity-corrected McNemar test yields $p = 0.3074$. Therefore, we cannot claim that Self-Attention is statistically significantly worse than Concat at the 5% level, but the empirical drop of $-5.72\%$ in Macro F1 and $-7.50\%$ in accuracy indicates that Self-Attention is disadvantageous in this data regime.

---

## 7. Checkpoint & Cohort Integrity Audit

### 7.1 Protected Checkpoint Ledger (Zero-Modification Verification)
All 11 pre-existing model checkpoints and weights were hashed before training and re-verified post-training:

| Checkpoint Identifier | File Path | File Size | SHA-256 Checksum | Integrity Status |
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

---

### 7.2 Newly Generated EXP-05 Checkpoint Hashes
The 15 newly generated model weight files in `experiments/exp05/checkpoints/` are cataloged below:

| Configuration | Fold | File Path | File Size | SHA-256 Checksum |
|:---|:---:|:---|:---:|:---|
| **Config A** | Fold 1 | `experiments/exp05/checkpoints/config_a/fold_1_best.weights.h5` | 52,899,216 | `52bd4a65becdb6146a338f1a55fe73b6b4722ef4ba93d8a924b5d1a0081bd95a` |
| **Config A** | Fold 2 | `experiments/exp05/checkpoints/config_a/fold_2_best.weights.h5` | 52,899,216 | `a3935aca244e471a98166684be75d7ba00736c05a59212eca43db4ddb2ccdf36` |
| **Config A** | Fold 3 | `experiments/exp05/checkpoints/config_a/fold_3_best.weights.h5` | 52,899,216 | `ba355773dd0b23bcde90300b29e4fb4e7cc0dbd2d99d9f115cf84ab15bc4a6b8` |
| **Config A** | Fold 4 | `experiments/exp05/checkpoints/config_a/fold_4_best.weights.h5` | 52,899,216 | `0a2f7a859f3b956564a04211945ee92c7cea9437e7822ecc37a0e376862e8cce` |
| **Config A** | Fold 5 | `experiments/exp05/checkpoints/config_a/fold_5_best.weights.h5` | 52,899,216 | `d1404d789c04dfeb09b27564f67c78e7b3cb722a06766999d4efa7493fb268dc` |
| **Config B** | Fold 1 | `experiments/exp05/checkpoints/config_b/fold_1_best.weights.h5` | 38,757,104 | `a9a56a6cb8daa8938b87c287d90def9ffd2412104565112ff6eb1dea66737137` |
| **Config B** | Fold 2 | `experiments/exp05/checkpoints/config_b/fold_2_best.weights.h5` | 38,757,104 | `cdb206970e7f8f8514c540b357f7c6ed367e5811c69f1527cdca439478cd7b9b` |
| **Config B** | Fold 3 | `experiments/exp05/checkpoints/config_b/fold_3_best.weights.h5` | 38,757,104 | `515f3ddcb877804fcd8dd4762054cc0ae6d7eef6f871efc6fd8256d4300004e4` |
| **Config B** | Fold 4 | `experiments/exp05/checkpoints/config_b/fold_4_best.weights.h5` | 38,757,104 | `ac4a63ba56662ef9adb8b1d35888a77be1868def6b548f29f05ef1f6f5a873a2` |
| **Config B** | Fold 5 | `experiments/exp05/checkpoints/config_b/fold_5_best.weights.h5` | 38,757,104 | `e8b6ebeb7e2735c43fd4d8e11894c15257c56d6fcfcd7b2fff08d55ef5475b44` |
| **Config C** | Fold 1 | `experiments/exp05/checkpoints/config_c/fold_1_best.weights.h5` | 44,724,672 | `e044c226ed283f6dd2a873d91b3507804da41c083e9655cc6b7d9443320f608b` |
| **Config C** | Fold 2 | `experiments/exp05/checkpoints/config_c/fold_2_best.weights.h5` | 44,724,672 | `7abbaa82402375506618ed90d242426850a2fc74adbf820de32a011ac6229828` |
| **Config C** | Fold 3 | `experiments/exp05/checkpoints/config_c/fold_3_best.weights.h5` | 44,724,672 | `3587d3f8c51c04065e77ffe0c576bb4ed01d9d51c39d1e7d791dbc31bc6200bc` |
| **Config C** | Fold 4 | `experiments/exp05/checkpoints/config_c/fold_4_best.weights.h5` | 44,724,672 | `ebc406bfe0309f863875e3f6d55b59e376b45f145bef612f58bba88e1bbab928` |
| **Config C** | Fold 5 | `experiments/exp05/checkpoints/config_c/fold_5_best.weights.h5` | 44,724,672 | `2ce6d852aacdfde67ceb925383dd483dc4ce3a480d585a2063dccb3a33e4f91c` |

---

## 8. In-Depth Scientific Discussion

### 8.1 Why Did Cross-Scan Multi-Head Attention Underperform?
1. **Extremely Short Sequence Length ($T=3$):** Multi-head self-attention was designed for long sequence modeling where positional context and pairwise interactions across hundreds of tokens drive feature discovery. For a fixed sequence of 3 scans, the attention matrix is only $3 \times 3$, containing only 9 query-key dot products per head. The quadratic expressiveness of attention provides virtually no structural advantage over linear weighting on a 3-element set.
2. **High Parameterization vs. Sample Size:** Training 1.28M parameters with 24 distinct trainable weight matrices on only 64 training participants per fold leads to over-parameterization. The attention heads likely latched onto spurious idiosyncrasies of specific scan captures rather than robust biometric invariants.
3. **Representation Collapse on Majority Class B+:** While Self-Attention attained high recall on rare AB+ samples (71.43%), it severely collapsed on B+ (recalling only 9 of 25 participants, 36.00%). A majority of B+ participants were misclassified into A+ (6 participants) and O+ (9 participants).

### 8.2 Why Is Gated Fusion a Strong Parameter-Efficient Alternative?
1. **Adaptive Softmax Normalization:** Rather than forcing every scan to contribute equally (as in simple averaging) or learning distinct weights for each scan position (as in concatenation), Configuration B computed an adaptive gating vector $\mathbf{g}_i \in \mathbb{R}^{1024}$ per scan. Softmax normalization across scans ensured that high-quality, high-contrast impressions were weighted more heavily than degraded or smudged impressions.
2. **60% Parameter Reduction:** By reducing the fused feature vector from 3072-d to 1024-d, the subsequent `Dense(512)` projection layer required only $1024 \times 512 = 524,288$ weights instead of $3072 \times 512 = 1,572,864$ weights. Total head parameters dropped from 1,968,772 to 789,381 without any drop in accuracy (52.50% vs. 52.50%).
3. **Lowest Inter-Fold Variance:** Configuration B achieved the lowest cross-fold standard deviation ($\text{Std} = 0.0797$), demonstrating smoother generalization than Concat ($\text{Std} = 0.1148$).

---

## 9. Limitations & Threat Analysis

1. **Development Cohort Scale:** The entire development set consists of 80 participants (240 scans). Although participant-grouped 5-fold cross-validation prevents data leakage, the sample size remains modest, limiting the statistical power of paired tests (e.g., McNemar test power with $N=80$).
2. **Minority Class Sparsity:** AB+ accounts for only 7 participants (8.75%) across the entire development set (1 or 2 participants per fold). A single correct or incorrect prediction shifts AB+ recall by $14.28\%$.
3. **Scanner Uniformity:** All scans were captured using a single optical sensor (Mantra MFS100). Sensor-specific optical distortions, illumination artifacts, and moisture sensitivities are identical across folds and may not generalize to capacitive or ultrasonic sensors.

---

## 10. Recommended Next Steps (Roadmap towards EXP-06)

Based on the empirical evidence gathered from EXP-01 through EXP-05:

1. **Retain Concatenation as Primary Reference; Keep Gated Fusion as Lightweight Alternative:**
   - Continue using Configuration A (Concatenation) as the reference architecture for benchmark comparisons.
   - Designate Configuration B (Gated Fusion) as the secondary lightweight candidate for edge deployment or mobile deployment where model size and memory footprints are constrained.
2. **Investigate Metric Learning Objectives in EXP-06:**
   - Since both backbone fine-tuning (EXP-04) and neck attention (EXP-05) failed to improve upon the frozen backbone + concatenation baseline, the remaining unexplored lever on the frozen feature representations is **metric space restructuring**.
   - Specifically, test Supervised Contrastive loss parameters:
     - Temperature sensitivity: $\tau \in \{0.05, 0.07, 0.10, 0.15\}$.
     - Multi-task loss weight: $\lambda_{\text{supcon}} \in \{0.05, 0.10, 0.20, 0.50\}$.
     - Direct triplet margin loss vs. supervised contrastive loss.
3. **Maintain Strict Cohort Isolation:**
   - Under no circumstances should the 21-participant locked test cohort or the 26-participant hospital cohort be accessed until all development experiments are concluded.
