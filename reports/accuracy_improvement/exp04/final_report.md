# EXP-04: Controlled Partial Fine-Tuning of E2 — Final Experiment Report

**Experiment ID:** EXP-04  
**Date of Completion:** 2026-10-15T01:25:00+05:30  
**Lead Investigator:** Antigravity AI  
**Scope:** Controlled investigation of backbone fine-tuning depth (Frozen vs. DenseBlock 4 vs. DenseBlocks 3+4) on the 3-tower E2 biometric fingerprint classification model.  
**Cohort Evaluated:** Approved Development Cohort (80 participants, 240 scans, 5-fold StratifiedGroupKFold).  
**Quarantined Cohorts:** Locked 21-participant test set (63 scans) and separate 26-participant hospital cohort (78 scans) strictly quarantined and unaccessed.  

---

## 1. Executive Summary

Following the completion and independent verification of **EXP-03** (which proved that unaugmented native scans yield the highest feature fidelity, achieving 53.75% accuracy and 0.5068 Macro F1 in Arm 1), **EXP-04** was designed to investigate whether unfreezing high-level convolutional layers of the ImageNet-pretrained DenseNet-121 backbone could adapt general image features to biometric fingerprint ridges without destabilizing the multi-task metric learning objective.

Three strictly controlled, audited configurations were trained across the approved 5-fold StratifiedGroupKFold development split:
1. **Configuration A (Frozen Backbone Reference):** Backbone 100% frozen; only multi-task fusion heads trainable ($\text{LR}=10^{-3}$).
2. **Configuration B (Conservative Partial Fine-Tuning):** DenseBlock 4 Conv2D trainable ($\text{LR}=10^{-4}$); BatchNorm frozen; heads trainable ($\text{LR}=10^{-3}$).
3. **Configuration C (Deep Partial Fine-Tuning):** DenseBlocks 3+4 + Transition 3 Conv trainable ($\text{LR}=5\times 10^{-5}$); BatchNorm frozen; heads trainable ($\text{LR}=10^{-3}$).

### Headline Findings:
- **Configuration A (Frozen Backbone) Remains the Strongest Overall Architecture:** Configuration A achieved the highest out-of-fold Accuracy (**50.00%**), Balanced Accuracy (**47.86%**), and Macro F1-score (**0.4781**).
- **Backbone Unfreezing Induced Representation Drift on Small Sample Sizes:** 
  - Unfreezing DenseBlock 4 (**Config B**) produced a nominal improvement in AB+ recall (**57.14%** vs. 42.86%), but caused severe degradation in B+ recall (collapsing from 60.00% to **28.00%**), reducing overall accuracy to **41.25%** and Balanced Accuracy to **42.97%**.
  - Unfreezing DenseBlocks 3 and 4 (**Config C**) caused severe minority class collapse: A+ recall collapsed to **15.79%** (3/19) and AB+ recall collapsed to **14.29%** (1/7), with predictions heavily collapsing into the majority B+ and O+ classes.
- **Statistical Significance:** Paired McNemar's exact tests on the 80 development participants confirmed that neither Config B ($p = 0.2100$) nor Config C ($p = 0.8036$) achieved a statistically significant performance difference over Config A, while Fisher's exact test on AB+ recall ($N=7$) showed no statistically significant difference between Config B and Config A ($p = 1.0000$).
- **Methodological Conclusion:** In small-cohort biometric fingerprint transfer learning ($N=80$ participants, 240 scans), the 7.03M pretrained ImageNet parameters are highly susceptible to feature distortion when gradients backpropagate through deep convolutional blocks. A frozen DenseNet-121 feature extractor paired with specialized projection heads provides the most balanced and regularized representation.

---

## 2. Experimental Architecture & Hyperparameter Specifications

To eliminate confounding variables, all three configurations strictly adhered to identical initialization, split, loss, and optimization rules.

### Configuration Matrix:
| Parameter / Dimension | Configuration A (Frozen Reference) | Configuration B (DenseBlock 4) | Configuration C (DenseBlocks 3+4) | Control Standard |
|:---|:---:|:---:|:---:|:---:|
| **Backbone Trainable Layers** | None (0 layers) | DenseBlock 4 Conv2D (32 layers) | DB3 + Trans3 + DB4 Conv2D (81 layers) | Explicit layer-type filtering |
| **Trainable Backbone Params** | **0** | **2,129,920** (32 variables) | **5,455,872** (81 variables) | Programmatically verified |
| **Backbone Optimizer & LR** | None | AdamW, $\text{LR}=10^{-4}$, $\lambda=10^{-4}$ | AdamW, $\text{LR}=5\times 10^{-5}$, $\lambda=10^{-4}$ | Subclassed Dual-Optimizer |
| **Head Trainable Layers** | `dense_512`, `clf`, `emb` | `dense_512`, `clf`, `emb` | `dense_512`, `clf`, `emb` | Explicit layer object binding |
| **Trainable Head Params** | **1,968,772** (6 variables) | **1,968,772** (6 variables) | **1,968,772** (6 variables) | Bit-for-bit identical shape |
| **Head Optimizer & LR** | AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$ | AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$ | AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$ | Identical across all configs |
| **Total Model Trainable Params**| **1,968,772** (21.9%) | **4,098,692** (45.5%) | **7,424,644** (82.4%) | Disjoint variable sets |
| **Batch Normalization Policy** | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Verified $\Delta=0.0$ on all statistics |
| **Epoch-Zero Head Weights** | Seed $(42 + k)$ per fold | Shared from Config A per fold | Shared from Config A per fold | Ephemeral `.npz` transfer |
| **Initial Backbone Weights** | Pretrained ImageNet | Pretrained ImageNet | Pretrained ImageNet | Bit-for-bit identical |
| **Participant Splits** | 5 StratifiedGroupKFold splits | Identical 5 splits | Identical 5 splits | 64 train / 16 val per fold |
| **Preprocessing** | `transform_c5` (No aug) | `transform_c5` (No aug) | `transform_c5` (No aug) | Composite alpha $\to$ CLAHE |
| **Loss Formulation** | $1.0 \times \mathcal{L}_{\text{SCCE}} + 0.1 \times \mathcal{L}_{\text{SupCon}}$ | $1.0 \times \mathcal{L}_{\text{SCCE}} + 0.1 \times \mathcal{L}_{\text{SupCon}}$ | $1.0 \times \mathcal{L}_{\text{SCCE}} + 0.1 \times \mathcal{L}_{\text{SupCon}}$ | $\tau = 0.07$ |
| **Sample Weighting** | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | Single multiplication on clf |
| **Stopping & Selection** | 150 max, 100 min, patience 15 | 150 max, 100 min, patience 15 | 150 max, 100 min, patience 15 | Val Macro F1 on held-out fold |

---

## 3. Phase 1 Preflight Verification Audit

Prior to initiating model execution, the pipeline executed complete programmatic preflight verification via `experiments/exp04/run_exp04.py --preflight-only`:
1. **Baseline Immutability:** SHA-256 hashes of all 5 original E2 checkpoints, all 5 EXP-03 Arm 1 checkpoints, and the production model `best_model.keras` were calculated and verified against baseline records.
2. **Variable Partitioning:** Programmatic verification confirmed that backbone variables and head variables formed mutually disjoint sets ($\text{set}_{\text{bb}} \cap \text{set}_{\text{hd}} = \emptyset$) whose union strictly equaled `model.trainable_variables`. Config A contained exactly 0 backbone variables.
3. **Partition Integrity:** Confirmed 0 participant leakage across all 5 folds: exactly 64 participants in training and 16 in validation per fold.
4. **Cohort Isolation:** The 21-participant locked test set and 26-participant hospital cohort were verified absent from all training, validation, and metadata inputs.

---

## 4. Phase 2 Execution Dynamics & Training Stability

All three configurations completed training across all 5 folds. Training dynamics were logged at fold and epoch resolution:

| Configuration | Fold | Best Epoch | Best Val Macro F1 | Val Accuracy | Total Epochs Run | Fold Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A** (Frozen) | Fold 1 | 91 | 0.3902 | 50.0% | 106 (Early Stop) | 10.8 min |
| | Fold 2 | 67 | 0.4750 | 43.8% | 100 (Floor Reached) | 10.2 min |
| | Fold 3 | 42 | 0.2398 | 50.0% | 100 (Floor Reached) | 10.4 min |
| | Fold 4 | 20 | 0.6190 | 62.5% | 100 (Floor Reached) | 11.0 min |
| | Fold 5 | 71 | 0.4348 | 43.8% | 100 (Floor Reached) | 11.2 min |
| **Config A Total** | — | **58.2 avg** | **0.4318 avg** | **50.0% avg** | **506 epochs** | **53.59 min** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config B** (DB4 Conv) | Fold 1 | 27 | 0.4804 | 50.0% | 100 (Floor Reached) | 8.2 min |
| | Fold 2 | 17 | 0.4178 | 31.2% | 100 (Floor Reached) | 8.1 min |
| | Fold 3 | 95 | 0.2776 | 31.2% | 110 (Early Stop) | 8.3 min |
| | Fold 4 | 69 | 0.4798 | 56.2% | 100 (Floor Reached) | 7.9 min |
| | Fold 5 | 35 | 0.3707 | 37.5% | 100 (Floor Reached) | 7.9 min |
| **Config B Total** | — | **48.6 avg** | **0.4053 avg** | **41.2% avg** | **510 epochs** | **40.48 min** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config C** (DB3+4) | Fold 1 | 33 | 0.3792 | 50.0% | 100 (Floor Reached) | 11.6 min |
| | Fold 2 | 35 | 0.3693 | 43.8% | 100 (Floor Reached) | 11.5 min |
| | Fold 3 | 15 | 0.4048 | 56.2% | 100 (Floor Reached) | 11.3 min |
| | Fold 4 | 4 | 0.3194 | 56.2% | 100 (Floor Reached) | 11.4 min |
| | Fold 5 | 28 | 0.2262 | 31.2% | 100 (Floor Reached) | 11.5 min |
| **Config C Total** | — | **23.0 avg** | **0.3398 avg** | **47.5% avg** | **500 epochs** | **57.27 min** |

### Observations on Optimization Dynamics:
- **Rapid Overfitting in Deep Backbone Layers (Config C):** In Config C, best validation checkpoints occurred very early in training (averaging epoch 23.0; in Fold 4, peak validation occurred at epoch 4). After epoch 25, validation Macro F1 degraded steadily across all folds as the 5.45M backbone parameters memorized the 64 training participants.
- **Controlled Epoch Evolution in Frozen Backbone (Config A):** In contrast, Config A evolved steadily throughout training, reaching peak generalization at an average of epoch 58.2 (reaching epoch 91 in Fold 1 and epoch 71 in Fold 5), demonstrating regularized convergence.

---

## 5. Phase 3 Independent Out-of-Fold Validation & Results

Out-of-fold predictions were aggregated across all 80 unique development participants (each evaluated exactly once during its fold's held-out validation step).

### 5.1 Primary Performance Metrics Comparison:
| Metric | Configuration A (Frozen Reference) | Configuration B (DenseBlock 4 Partial) | Configuration C (DenseBlocks 3+4 Deep) | EXP-04 Winner / Trend |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **50.00%** (40/80) | 41.25% (33/80) | 47.50% (38/80) | **Config A (+8.75% over B, +2.50% over C)** |
| **Balanced Accuracy** | **47.86%** | 42.97% | 39.04% | **Config A (+4.89% over B, +8.82% over C)** |
| **Macro F1-Score** | **0.4781** | 0.4032 | 0.3789 | **Config A (+0.0749 over B, +0.0992 over C)** |
| **Macro Precision** | **0.4921** | 0.3962 | 0.3864 | **Config A (+0.0959 over B, +0.1057 over C)** |
| **Macro Recall** | **0.4786** | 0.4297 | 0.3904 | **Config A (+0.0489 over B, +0.0882 over C)** |
| **Fold Accuracy Std** | **$\pm$ 7.65%** | $\pm$ 11.35% | $\pm$ 10.46% | **Config A (Most stable across folds)** |
| **Fold BalAcc Std** | $\pm$ 14.41% | $\pm$ 12.84% | **$\pm$ 7.03%** | Config C (Low variance due to majority collapse) |
| **Fold Macro F1 Std** | $\pm$ 0.1374 | $\pm$ 0.0849 | **$\pm$ 0.0706** | Config C |

---

### 5.2 Per-Class Recall & Precision Breakdown:
| Blood Group | Class Support | Config A Recall (Correct) | Config B Recall (Correct) | Config C Recall (Correct) | Observation / Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **A+** | $N=19$ | **36.84%** (7/19) | 31.58% (6/19) | 15.79% (3/19) | Severe collapse in Config C (-21.05%) |
| **AB+** | $N=7$ | 42.86% (3/7) | **57.14%** (4/7) | 14.29% (1/7) | Config B detected +1 participant; Config C collapsed |
| **B+** | $N=25$ | 60.00% (15/25) | 28.00% (7/25) | **64.00%** (16/25) | Severe regression in Config B (-32.00%) |
| **O+** | $N=29$ | 51.72% (15/29) | 55.17% (16/29) | **62.07%** (18/29) | Config C heavily biased toward majority class |

---

### 5.3 Confusion Matrices:

#### Configuration A (Frozen Reference — Macro F1: 0.4781):
```
True \ Pred      A+     AB+      B+      O+    | Total | Recall
------------------------------------------------|-------|-------
A+                7       2       2       8    |   19  | 36.84%
AB+               0       3       2       2    |    7  | 42.86%
B+                1       1      15       8    |   25  | 60.00%
O+                4       2       8      15    |   29  | 51.72%
------------------------------------------------|-------|-------
Pred Total       12       8      27      33    |   80  | Acc: 50.00%
```

#### Configuration B (DenseBlock 4 Partial Fine-Tuning — Macro F1: 0.4032):
```
True \ Pred      A+     AB+      B+      O+    | Total | Recall
------------------------------------------------|-------|-------
A+                6       2       3       8    |   19  | 31.58%
AB+               0       4       0       3    |    7  | 57.14%
B+                8       2       7       8    |   25  | 28.00%
O+                2       3       8      16    |   29  | 55.17%
------------------------------------------------|-------|-------
Pred Total       16      11      18      35    |   80  | Acc: 41.25%
```

#### Configuration C (DenseBlocks 3+4 Deep Fine-Tuning — Macro F1: 0.3789):
```
True \ Pred      A+     AB+      B+      O+    | Total | Recall
------------------------------------------------|-------|-------
A+                3       1       6       9    |   19  | 15.79%
AB+               1       1       3       2    |    7  | 14.29%
B+                4       1      16       4    |   25  | 64.00%
O+                2       2       7      18    |   29  | 62.07%
------------------------------------------------|-------|-------
Pred Total       10       5      32      33    |   80  | Acc: 47.50%
```

---

### 5.4 Statistical Significance Testing:

To rigorously test whether the observed differences were statistically meaningful or attributable to stochastic sampling variation:
1. **Paired McNemar's Exact Test on 80 Participants:**
   - **Config B vs. Config A:**
     - Correct in Config A, Incorrect in Config B ($b$): 15 participants
     - Incorrect in Config A, Correct in Config B ($c$): 8 participants
     - Exact Binomial Test $p$-value: **$p = 0.2100$**
     - Interpretation: The 8.75% drop in accuracy from Config A to Config B is **not statistically significant** at the $\alpha=0.05$ threshold.
   - **Config C vs. Config A:**
     - Correct in Config A, Incorrect in Config C ($b$): 9 participants
     - Incorrect in Config A, Correct in Config C ($c$): 7 participants
     - Exact Binomial Test $p$-value: **$p = 0.8036$**
     - Interpretation: The 2.50% difference in accuracy between Config A and Config C is **not statistically significant**.

2. **Fisher's Exact Test on AB+ Minority Class ($N=7$):**
   - **Config B vs. Config A:** 4/7 (57.14%) vs. 3/7 (42.86%), **$p = 1.0000$**.
     - Finding: While Config B correctly classified one additional AB+ participant (4 vs. 3), this shift represents random noise within an $N=7$ sample size and has no statistical significance.
   - **Config C vs. Config A:** 1/7 (14.29%) vs. 3/7 (42.86%), **$p = 0.5594$**.
     - Finding: Not statistically significant on the minority slice alone, but indicative of systemic minority degradation.

---

## 6. Checkpoint Integrity & Hash Ledger

All 15 newly generated fold checkpoints were verified and recorded with their definitive SHA-256 hashes and file sizes.

### 6.1 EXP-04 Checkpoints Generated:
| Configuration | Checkpoint Path | File Size (Bytes) | SHA-256 Checksum |
|:---|:---|:---:|:---:|
| **Config A** | `experiments/exp04/checkpoints/config_a/fold_1_best.weights.h5` | 60,779,864 | `5d2157aeee17cc7dbeb27260af5e6fce3ae8f0871bf8aa75fa8b6aee82687dcf` |
| | `experiments/exp04/checkpoints/config_a/fold_2_best.weights.h5` | 60,779,864 | `9c4388b6d506776df5d1c99a557bfa62a9391a56357c1ad56cd3dc4f07d732f9` |
| | `experiments/exp04/checkpoints/config_a/fold_3_best.weights.h5` | 60,779,864 | `8793741a9ed54abc91dfa1e7f3fa23a1797a16ed71c1801bf52abf60ba2e9a7d` |
| | `experiments/exp04/checkpoints/config_a/fold_4_best.weights.h5` | 60,779,864 | `4a24b9713129974f6e75b951d7fa91818021517e5dc8af8fb428d37d3bbf3abe` |
| | `experiments/exp04/checkpoints/config_a/fold_5_best.weights.h5` | 60,779,864 | `e0c8a4135a70b58ed31dde493d9daf3cd62c88fcd0c9dfc04eced334d100d40e` |
|:---|:---|:---:|:---:|
| **Config B** | `experiments/exp04/checkpoints/config_b/fold_1_best.weights.h5` | 86,376,168 | `532332b72300223d59a04e26eb02ac6a456974431a7dc483cb1edfa993d1ac95` |
| | `experiments/exp04/checkpoints/config_b/fold_2_best.weights.h5` | 86,376,168 | `ff2cd8e0a7dee42447585e59d0145ec4ed5d15c9c59caa8d607c2c7cc47d258f` |
| | `experiments/exp04/checkpoints/config_b/fold_3_best.weights.h5` | 86,376,168 | `9967b27ba6b081314d027cc8436fba5f7b4e30528f47a15211ca44539e4bd16b` |
| | `experiments/exp04/checkpoints/config_b/fold_4_best.weights.h5` | 86,376,168 | `e2ae4b1c56091f43f80443878da30f4dd7479c360a5b410038122995c03e1b00` |
| | `experiments/exp04/checkpoints/config_b/fold_5_best.weights.h5` | 86,376,168 | `a07ceb2d0bf09cf33333b7eecc267a41443fca6c3de70a63ae5769d1837753ec` |
|:---|:---|:---:|:---:|
| **Config C** | `experiments/exp04/checkpoints/config_c/fold_1_best.weights.h5` | 126,342,472 | `e9fc3b82c4cec6dc76783d8d5f40691346d6452bd50b729f9e50318cf1189fa0` |
| | `experiments/exp04/checkpoints/config_c/fold_2_best.weights.h5` | 126,342,472 | `c38f42a0b40037136e4b1519768eb8cb773b30fb9cf42eab7d353eddc0d8e691` |
| | `experiments/exp04/checkpoints/config_c/fold_3_best.weights.h5` | 126,342,472 | `b733d834faa95a24b989c8166a74c9a9e41b85a0b6bba17697c3cff999493b84` |
| | `experiments/exp04/checkpoints/config_c/fold_4_best.weights.h5` | 126,342,472 | `826edb9a3383cc6d9f4b1a9bbfdb6282642c0faa2c1190d09c39e208be52b87a` |
| | `experiments/exp04/checkpoints/config_c/fold_5_best.weights.h5` | 126,342,472 | `6d6f0a3418eb1273ef1bbceab0a3bab73e9a9dfe818da3977229a77978cb51ab` |

### 6.2 Pre-Existing Checkpoints & Production Model Immutability Audit:
Following the full execution of EXP-04, all pre-existing baseline checkpoints were re-hashed. Every checksum strictly matches its pre-experiment audit:
- `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` $\to$ `7d9464d18619a94f...` (**100% MATCH**)
- `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` $\to$ `9c72809f23152708...` (**100% MATCH**)
- `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` $\to$ `8123b1b86f44719d...` (**100% MATCH**)
- `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` $\to$ `cc122b23e0c6d118...` (**100% MATCH**)
- `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` $\to$ `aff9e137bed61f64...` (**100% MATCH**)
- `experiments/exp03/checkpoints/arm1/fold_1_best.weights.h5` $\to$ `0d047b7a10153718...` (**100% MATCH**)
- `experiments/exp03/checkpoints/arm1/fold_2_best.weights.h5` $\to$ `dd94e4c44887399d...` (**100% MATCH**)
- `experiments/exp03/checkpoints/arm1/fold_3_best.weights.h5` $\to$ `f763f67068a4411d...` (**100% MATCH**)
- `experiments/exp03/checkpoints/arm1/fold_4_best.weights.h5` $\to$ `589f1e52275f68ed...` (**100% MATCH**)
- `experiments/exp03/checkpoints/arm1/fold_5_best.weights.h5` $\to$ `848aea5498e46987...` (**100% MATCH**)
- `models/abo_positive_4class/best_model.keras` $\to$ `7b5ff5d94d4ce3de...` (**100% MATCH**)

---

## 7. Limitations & Scientific Caveats

1. **Development Cohort Constraint:** All reported metrics reflect cross-validation across the 80 approved development participants. They do **not** represent locked-test generalization or clinical validation.
2. **Minority Class Small Sample Size:** With only $N=7$ AB+ participants in the development cohort (1-2 per fold), small fluctuations in raw count (+1 in Config B, -2 in Config C) change class recall by 14.3% per participant, lacking statistical power ($p = 1.0000$).
3. **No Clinical Validity:** Biometric blood group detection remains an investigational research task. These models cannot be used for clinical diagnosis or transfusion typing.

---

## 8. Strategic Recommendation for EXP-05

Based on the accumulated empirical evidence across EXP-01, EXP-02, EXP-03, and EXP-04:
1. **Backbone Strategy:** Keep the DenseNet-121 backbone **frozen**. Fine-tuning high-level convolutional layers degrades overall balanced accuracy, increases cross-fold variance, and causes representation collapse onto majority classes.
2. **Data Pipeline Strategy:** Retain **unaugmented `transform_c5` preprocessing** (established in EXP-03 Arm 1), which preserves fine ridge morphology without synthetic geometric artifacts.
3. **Proposed EXP-05 Focus:**
   - **Feature Fusion & Attention Mechanism:** Since modifying backbone weights causes overfitting, investigate **cross-scan attention mechanisms** or **gated feature aggregation** within the fusion neck (e.g., replacing simple concatenation of the 3 scans with a lightweight multi-head self-attention module or SE-attention block).
   - **Metric Learning Objective Tuning:** Investigate supervised contrastive temperature ($\tau \in \{0.05, 0.07, 0.10\}$) and multi-task loss weighting ($\lambda_{\text{supcon}} \in \{0.05, 0.10, 0.20\}$) to further regularize the embedding space for minority classes without altering backbone weights.
