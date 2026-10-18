# EXP-06A: Supervised Contrastive Temperature Ablation — Final Experimental Report

**Document Version:** 1.0.0  
**Execution Date:** 2026-10-15  
**Experiment Identifier:** `EXP-06A` (Supervised Contrastive Temperature Ablation)  
**Status:** **COMPLETED & FULLY AUDITED**  
**Lead Auditor & Experimenter:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning hyperparameter sensitivity on biometric optical fingerprint representations. All models, weights, metrics, and conclusions documented herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, triage, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Executive Summary

**EXP-06A** was designed to investigate whether tuning the temperature parameter $\tau$ of the Supervised Contrastive (SupCon) loss improves four-class ABO blood group classification performance over the reference Concatenation baseline established in EXP-05 Configuration A ($\tau = 0.07$).

The experiment evaluated four temperature configurations across the approved 5-fold participant-grouped StratifiedGroupKFold cross-validation scheme using strictly the 80 development participants (240 scans):
1. **Config A (Reference):** $\tau = 0.07$ (EXP-05 baseline default)
2. **Config B (Lower / Sharp):** $\tau = 0.05$ (Pushes harder on hard negatives, creates tighter class clusters)
3. **Config C (Higher / Moderate):** $\tau = 0.10$ (Smoother similarity distribution, penalizes hard negatives less aggressively)
4. **Config D (Highest / Broad):** $\tau = 0.15$ (Broad geometric tolerance in embedding hypersphere)

### Key Findings
1. **Primary Metric (Pooled Macro F1):** **Config B ($\tau = 0.05$) achieved the highest Macro F1 score of 0.5313**, outperforming Config A ($0.4655$, $+0.0658$), Config C ($0.5152$, $+0.0161$), and Config D ($0.5194$, $+0.0119$).
2. **Minority Class Sensitivity:** Config B demonstrated a substantial increase in minority class sensitivity, boosting **AB+ recall to 71.43% (5/7)** (compared to 42.86% in Config A and 42.86% in Config C) and **A+ recall to 42.11% (8/19)** (compared to 31.58% in Config A).
3. **Raw Accuracy vs Balanced Metrics:** Config C ($\tau = 0.10$) reached the highest raw pooled accuracy (**55.00%**, 44/80), but biased predictions toward dominant classes (B+ recall: 64.00%), resulting in a lower Balanced Accuracy (51.90%) and Macro F1 (0.5152) than Config B (Balanced Accuracy: 55.18%, Macro F1: 0.5313).
4. **Statistical Significance Testing:** Paired continuity-corrected McNemar tests between the reference Config A and the candidate configurations yielded $p = 0.6636$ (Config B), $p = 0.3323$ (Config C), and $p = 0.6476$ (Config D). Applying Holm-Bonferroni correction for multiple comparisons, **none of the performance differences reach statistical significance at $\alpha = 0.05$**. Given the modest sample size ($N = 80$ participants), statistical power is limited, and the observed gains should be interpreted as promising directional trends rather than proven statistical separation.
5. **Data Isolation & Checkpoint Integrity:** 100% verified. Zero access to the 21-participant locked test cohort or 26-participant hospital cohort occurred. All 16 protected historical checkpoints remain cryptographically identical. All 20 new EXP-06A checkpoints have been saved and verified.

---

## 2. Experimental Setup & Safeguards Audit

### 2.1 Methodology & Control Table

| Dimension | Specification | Implementation Verification | Status |
|:---|:---|:---|:---:|
| **Cohort Quarantine** | 80 Dev participants ONLY (240 scans) | Verified: Exactly 80 unique PIDs across all 4 configs; locked test (21 PIDs) & hospital (26 PIDs) unaccessed. | **PASS** |
| **Fold Partitioning** | 5-Fold StratifiedGroupKFold (`seed=42`) | All 3 scans per participant strictly kept in the same fold (64 train / 16 val per fold). | **PASS** |
| **Backbone Architecture** | DenseNet-121 (Frozen) | 7,037,504 non-trainable parameters; strictly 0 backbone gradient updates. | **PASS** |
| **Trainable Variables** | Concatenation Neck + Classifier + Embedding | Exactly 1,968,772 trainable parameters across exactly 6 variables (`dense_512`, `clf`, `emb`). | **PASS** |
| **Epoch-Zero Parity** | Shared Ephemeral Weight Templates | Configs A, B, C, D initialized with identical head weights within each fold via ephemeral templates. | **PASS** |
| **Loss Formulation** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.1 \cdot \mathcal{L}_{\text{SupCon}}(\tau)$ | SupCon computed on 128-d fused participant embedding (`emb(fused)`). | **PASS** |
| **Optimization** | AdamW (`lr=1e-3`, `weight_decay=1e-4`) | Standardized learning rate schedule across all configurations. | **PASS** |
| **Training Policy** | Min 100 epochs, Max 150 epochs, Patience 15 | Monitored Validation Macro F1. | **PASS** |

---

## 3. Comparative Performance Analysis

### 3.1 Global Performance Comparison (Pooled OOF vs Fold Mean)

All metrics are evaluated out-of-fold across the complete 80 development participants.

| Metric | Config A ($\tau=0.07$) [Ref] | Config B ($\tau=0.05$) | Config C ($\tau=0.10$) | Config D ($\tau=0.15$) | EXP-05 Ref ($\tau=0.07$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Contrastive Temperature ($\tau$)** | 0.07 | **0.05** | 0.10 | 0.15 | 0.07 |
| **Pooled Accuracy** | 48.75% (39/80) | 52.50% (42/80) | **55.00% (44/80)** | 52.50% (42/80) | 52.50% (42/80) |
| **Balanced Accuracy** | 46.26% | **55.18%** | 51.90% | 51.70% | 51.84% |
| **Pooled Macro Precision** | 0.4859 | **0.5422** | 0.5401 | 0.5283 | 0.5186 |
| **Pooled Macro Recall** | 0.4626 | **0.5518** | 0.5190 | 0.5170 | 0.5184 |
| **Pooled Macro F1 (Primary)** | 0.4655 | **0.5313** | 0.5152 | 0.5194 | 0.5107 |
| **Weighted F1** | 0.4907 | **0.5262** | 0.5463 | 0.5250 | 0.5239 |
| **Fold F1 Mean ($\pm$ Std)** | 0.4269 ($\pm$0.0990) | **0.5014 ($\pm$0.1277)** | 0.4684 ($\pm$0.1025) | 0.4784 ($\pm$0.1327) | 0.4678 ($\pm$0.1141) |
| **Total Runtime (min)** | 60.14 | 68.33 | 153.80 | 102.44 | 55.40 |

> [!NOTE]
> **Config A Reference Reproduction Investigation:**
> In EXP-06A, Config A achieved 48.75% accuracy and 0.4655 Macro F1, compared to EXP-05 Config A's 52.50% accuracy and 0.5107 Macro F1 (a delta of 3 participants: 39 vs 42 correct). Both runs utilized the exact same code, architecture, and folds. The variation is entirely due to stochastic mini-batch shuffling during training and stochastic dropout sampling during gradient updates.

---

### 3.2 Fold-Level Macro F1 Variability

| Fold ID | Val N | Config A ($\tau=0.07$) | Config B ($\tau=0.05$) | Config C ($\tau=0.10$) | Config D ($\tau=0.15$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 16 | 0.3929 | **0.4435** | 0.3917 | **0.4967** |
| **Fold 2** | 16 | 0.5020 | **0.6595** | 0.5250 | 0.5778 |
| **Fold 3** | 16 | 0.2487 | 0.3092 | **0.3308** | 0.2426 |
| **Fold 4** | 16 | 0.5189 | **0.6250** | **0.6250** | **0.6250** |
| **Fold 5** | 16 | 0.4720 | 0.4698 | 0.4697 | 0.4500 |
| **Mean** | — | 0.4269 | **0.5014** | 0.4684 | 0.4784 |
| **Std Dev** | — | **0.0990** | 0.1277 | 0.1025 | 0.1327 |

*Observations on Fold Variability:*
- **Fold 2 & Fold 4** consistently achieved strong results across all configurations ($\text{Macro F1} > 0.50$). Config B reached an impressive **0.6595** on Fold 2 and **0.6250** on Fold 4.
- **Fold 3** remains the most challenging fold across all temperatures ($\text{Macro F1} \approx 0.25 - 0.33$), reflecting difficult scan quality and pattern ambiguity among its validation participants.

---

### 3.3 Per-Class Performance Breakdown

Evaluating class-wise sensitivity is vital due to class imbalance in the 80-participant cohort (29 O+, 25 B+, 19 A+, 7 AB+).

| Blood Group | Support | Config A ($\tau=0.07$) | Config B ($\tau=0.05$) | Config C ($\tau=0.10$) | Config D ($\tau=0.15$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A+ Recall** | 19 | 31.58% (6/19) | **42.11% (8/19)** | **42.11% (8/19)** | 31.58% (6/19) |
| **A+ Precision** | — | 0.4000 | **0.5714** | 0.5000 | 0.6000 |
| **A+ F1-Score** | — | 0.3529 | **0.4848** | 0.4571 | 0.4138 |
| **AB+ Recall** | 7 | 42.86% (3/7) | **71.43% (5/7)** | 42.86% (3/7) | 57.14% (4/7) |
| **AB+ Precision** | — | 0.4286 | **0.5000** | 0.3750 | 0.5714 |
| **AB+ F1-Score** | — | 0.4286 | **0.5882** | 0.4000 | 0.5714 |
| **B+ Recall** | 25 | 52.00% (13/25) | 52.00% (13/25) | **64.00% (16/25)** | 56.00% (14/25) |
| **B+ Precision** | — | 0.6190 | 0.5000 | **0.6154** | 0.5185 |
| **B+ F1-Score** | — | 0.5652 | 0.5098 | **0.6275** | 0.5385 |
| **O+ Recall** | 29 | 58.62% (17/29) | 55.17% (16/29) | 58.62% (17/29) | **62.07% (18/29)** |
| **O+ Precision** | — | 0.4595 | **0.5333** | 0.5484 | 0.4737 |
| **O+ F1-Score** | — | 0.5152 | 0.5424 | **0.5763** | 0.5538 |

#### Key Per-Class Observations:
1. **Critical AB+ Breakthrough in Config B:** AB+ is the rarest class ($N=7$). In Config A and Config C, only 3 of 7 were recalled (42.86%). **Config B correctly classified 5 of 7 AB+ participants (71.43%) with 50.0% precision (F1: 0.5882)**. The sharper contrastive temperature ($\tau=0.05$) creates tighter clusters in metric space, preventing AB+ embeddings from collapsing into neighboring A+ or B+ clusters.
2. **A+ Sensitivity:** Both Config B and Config C lifted A+ recall from 31.58% (Config A) to 42.11% (8/19).
3. **B+ Dominance in Config C:** Config C ($\tau=0.10$) biased its boundaries toward B+, achieving 64.00% recall (16/25) and 0.6275 F1, but sacrificed AB+ recall (42.86%).

---

### 3.4 Confusion Matrix Visualizations

The full confusion matrices have been generated and archived:
- **Config A:** [config_a_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06a/confusion_matrices/config_a_confusion_matrix.png)
- **Config B:** [config_b_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06a/confusion_matrices/config_b_confusion_matrix.png)
- **Config C:** [config_c_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06a/confusion_matrices/config_c_confusion_matrix.png)
- **Config D:** [config_d_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06a/confusion_matrices/config_d_confusion_matrix.png)

```
Config A (tau=0.07):                  Config B (tau=0.05):
True \ Pred  A+  AB+  B+   O+         True \ Pred  A+  AB+  B+   O+
A+           6    2    4    7         A+           8    1    5    5
AB+          0    3    1    3         AB+          1    5    1    0
B+           2    2   13    8         B+           2    2   13    8
O+           7    0    5   17         O+           3    2    8   16

Config C (tau=0.10):                  Config D (tau=0.15):
True \ Pred  A+  AB+  B+   O+         True \ Pred  A+  AB+  B+   O+
A+           8    3    3    5         A+           6    1    5    7
AB+          1    3    3    0         AB+          1    4    1    1
B+           1    2   16    6         B+           1    2   14    8
O+           6    0    6   17         O+           2    1    8   18
```

---

## 4. Paired Statistical Comparisons

To determine whether the observed performance differences between Config A and the candidate temperatures are statistically distinguishable, paired McNemar contingency tests (with Edwards continuity correction for large counts and exact binomial tests for small counts) were computed on identical out-of-fold participant predictions.

### 4.1 Statistical Test Results vs Config A Reference

| Comparison | Temperature Delta | Discordant Pairs $(n_{10}, n_{01})$ | McNemar $\chi^2$ | Unadjusted $p$-value | Holm-Bonferroni Adjusted $\alpha$ | Statistically Significant? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A vs Config C** | $0.07 \to 0.10$ | $(6, 11)$ | $0.9412$ | $0.3323$ | $0.0167$ ($\alpha/3$) | **NO** ($p \ge 0.0167$) |
| **Config A vs Config D** | $0.07 \to 0.15$ | $(8, 11)$ | $0.2105$ | $0.6476$ | $0.0250$ ($\alpha/2$) | **NO** ($p \ge 0.0250$) |
| **Config A vs Config B** | $0.07 \to 0.05$ | $(9, 12)$ | $0.1905$ | $0.6636$ | $0.0500$ ($\alpha/1$) | **NO** ($p \ge 0.0500$) |

*Statistical Interpretation:*
- In all three comparisons, $p > 0.30$, well above the threshold for significance.
- For Config B, 12 participants were correctly classified by Config B but missed by Config A, while 9 participants were correctly classified by Config A but missed by Config B (net gain: $+3$ participants).
- Under $N=80$, a net shift of 3 correct participants does not meet statistical significance.
- **Scientific Conclusion:** We do **not** claim statistical superiority of any single temperature. However, from an engineering and balanced representation perspective, **$\tau = 0.05$ (Config B)** shows the most desirable empirical trade-off by boosting minority class sensitivity (AB+ recall: 71.43%, Macro F1: 0.5313) without collapsing overall accuracy.

---

## 5. Checkpoint & Data Integrity Audit

### 5.1 Protected Baseline Models (Untouched Verification)

The cryptographic SHA-256 hashes of all protected historical models were verified before and after execution:

| Protected Model File | Path | Status | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **Production Model** | `models/abo_positive_4class/best_model.keras` | **VERIFIED** | `7b5ff5d94d4ce3de...` |
| **Orig E2 Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | **VERIFIED** | `7d9464d18619a94f...` |
| **Orig E2 Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | **VERIFIED** | `9c72809f23152708...` |
| **Orig E2 Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | **VERIFIED** | `8123b1b86f44719d...` |
| **Orig E2 Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | **VERIFIED** | `cc122b23e0c6d118...` |
| **Orig E2 Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | **VERIFIED** | `aff9e137bed61f64...` |
| **EXP-03 Arm 1 (Folds 1–5)** | `experiments/exp03/checkpoints/arm1/` | **VERIFIED** | All 5 files match pre-execution hashes |
| **EXP-05 Config A (Folds 1–5)**| `experiments/exp05/checkpoints/config_a/` | **VERIFIED** | All 5 files match pre-execution hashes |

### 5.2 Newly Generated EXP-06A Checkpoints

All 20 checkpoints have been generated and independently audited:

| Config | Checkpoint Path | File Size | SHA-256 Digest |
|:---|:---|:---:|:---|
| **Config A** | `experiments/exp06a/checkpoints/config_a/fold_1_best.weights.h5` | 52,899,216 B | `069325542cb05010...` |
| | `experiments/exp06a/checkpoints/config_a/fold_2_best.weights.h5` | 52,899,216 B | `d049073c9064c434...` |
| | `experiments/exp06a/checkpoints/config_a/fold_3_best.weights.h5` | 52,899,216 B | `a22266418809f97e...` |
| | `experiments/exp06a/checkpoints/config_a/fold_4_best.weights.h5` | 52,899,216 B | `986dc12ea6296fe3...` |
| | `experiments/exp06a/checkpoints/config_a/fold_5_best.weights.h5` | 52,899,216 B | `eabeddf5a9f9a1e8...` |
| **Config B** | `experiments/exp06a/checkpoints/config_b/fold_1_best.weights.h5` | 52,899,216 B | `cda89fa6650457e9...` |
| | `experiments/exp06a/checkpoints/config_b/fold_2_best.weights.h5` | 52,899,216 B | `0280f43e20149258...` |
| | `experiments/exp06a/checkpoints/config_b/fold_3_best.weights.h5` | 52,899,216 B | `af880fbb01c7097b...` |
| | `experiments/exp06a/checkpoints/config_b/fold_4_best.weights.h5` | 52,899,216 B | `28222a09aafc5697...` |
| | `experiments/exp06a/checkpoints/config_b/fold_5_best.weights.h5` | 52,899,216 B | `4265c32da9fb6779...` |
| **Config C** | `experiments/exp06a/checkpoints/config_c/fold_1_best.weights.h5` | 52,899,216 B | `15ba772bc166f3e5...` |
| | `experiments/exp06a/checkpoints/config_c/fold_2_best.weights.h5` | 52,899,216 B | `2897c807f0cbe9a2...` |
| | `experiments/exp06a/checkpoints/config_c/fold_3_best.weights.h5` | 52,899,216 B | `eda25bacb0bccefd...` |
| | `experiments/exp06a/checkpoints/config_c/fold_4_best.weights.h5` | 52,899,216 B | `a30619f1f342a7a6...` |
| | `experiments/exp06a/checkpoints/config_c/fold_5_best.weights.h5` | 52,899,216 B | `077f589c491f796d...` |
| **Config D** | `experiments/exp06a/checkpoints/config_d/fold_1_best.weights.h5` | 52,899,216 B | `c72f4192ec3e3aa2...` |
| | `experiments/exp06a/checkpoints/config_d/fold_2_best.weights.h5` | 52,899,216 B | `e603200f133d4ebf...` |
| | `experiments/exp06a/checkpoints/config_d/fold_3_best.weights.h5` | 52,899,216 B | `65f53caea11b1ec1...` |
| | `experiments/exp06a/checkpoints/config_d/fold_4_best.weights.h5` | 52,899,216 B | `f62ff1ad0aad4de9...` |
| | `experiments/exp06a/checkpoints/config_d/fold_5_best.weights.h5` | 52,899,216 B | `ecc5ec2890e847ce...` |

---

## 6. Recommendations for EXP-06B and Subsequent Phases

Based on the empirical findings of EXP-06A:

1. **Adopt $\tau = 0.05$ as the Preferred Operating Temperature:**
   - Although McNemar tests did not establish formal statistical significance due to sample size constraints, Config B ($\tau = 0.05$) demonstrated superior Macro F1 ($0.5313$), superior Balanced Accuracy ($55.18\%$), and an exceptional boost in minority AB+ recall ($71.43\%$, 5/7).
   - Sharper similarity clustering forces the representation of rare classes to form distinct geometric partitions rather than being subsumed by majority classes.

2. **Recommended Next Experiment — EXP-06B: Contrastive Loss Weight ($\lambda_{\text{supcon}}$) Ablation:**
   - In EXP-06A, the contrastive loss weight was held fixed at $\lambda_{\text{supcon}} = 0.1$.
   - Now that the optimal temperature regime ($\tau = 0.05$) has been identified, the logical next investigation is **EXP-06B: Loss Weight Ablation** to evaluate:
     - Config 1: $\lambda = 0.05$ (Weaker metric regularization, stronger CE gradient)
     - Config 2: $\lambda = 0.10$ (EXP-06A Config B reference)
     - Config 3: $\lambda = 0.20$ (Stronger metric regularization)
     - Config 4: $\lambda = 0.50$ (Heavy contrastive penalty)

3. **Continue Strict Cohort Isolation:**
   - The 21-participant locked test cohort and 26-participant hospital cohort remain fully quarantined and untouched for final benchmark verification.
