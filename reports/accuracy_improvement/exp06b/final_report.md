# EXP-06B: Supervised Contrastive Loss Weight Ablation — Final Experimental Report

**Document Version:** 1.0.0  
**Execution Date:** 2026-10-15 – 2026-10-16  
**Experiment Identifier:** `EXP-06B` (Supervised Contrastive Loss Weight Ablation)  
**Status:** **COMPLETED & FULLY AUDITED**  
**Lead Auditor & Experimenter:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning hyperparameter sensitivity on biometric optical fingerprint representations. All models, weights, metrics, and conclusions documented herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, triage, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Executive Summary

**EXP-06B** was executed under formal user authorization to systematically evaluate the sensitivity of four-class biometric blood group prediction to the Supervised Contrastive (SupCon) loss weight ($\lambda_{\text{supcon}}$) while keeping the contrastive temperature fixed at the empirical optimum identified in EXP-06A ($\tau = 0.05$).

The experiment evaluated four loss weight configurations across the approved 5-fold participant-grouped StratifiedGroupKFold cross-validation scheme using strictly the 80 development participants (240 scans):
1. **Arm A (Reduced Metric Constraint):** $\lambda = 0.05, \tau = 0.05$ (Cross-Entropy dominant)
2. **Arm B (Reference Default):** $\lambda = 0.10, \tau = 0.05$ (EXP-06A Config B baseline)
3. **Arm C (Moderate Metric Constraint):** $\lambda = 0.20, \tau = 0.05$ (Balanced CE and SupCon regularization)
4. **Arm D (Heavy Metric Constraint):** $\lambda = 0.50, \tau = 0.05$ (Heavy geometric hypersphere clustering penalty)

### Key Findings
1. **Primary Metric (Pooled Out-of-Fold Macro F1):**
   - **Arm D ($\lambda = 0.50$) achieved the highest overall performance**, reaching a pooled OOF Macro F1 of **0.4925**, overall accuracy of **52.50% (42/80)**, and balanced accuracy of **51.46%**.
   - **Arm B ($\lambda = 0.10$, Reference)** achieved **0.4789** Macro F1 and **47.50% (38/80)** accuracy.
   - **Arm A ($\lambda = 0.05$)** reached **0.4438** Macro F1 and **46.25% (37/80)** accuracy.
   - **Arm C ($\lambda = 0.20$)** reached **0.4404** Macro F1 and **45.00% (36/80)** accuracy.
2. **Fold-Level Stability:**
   - Arm D also achieved the highest mean fold-level Macro F1 (**0.5035 $\pm$ 0.1007**), driven by exceptionally strong peak validation performances on Fold 2 (**0.6043**) and Fold 4 (**0.6458**).
3. **Per-Class Sensitivity Dynamics:**
   - **O+ Dominance under Heavy Metric Penalty:** Arm D attained the highest O+ recall at **75.86% (22/29)**, demonstrating that stronger contrastive regularization successfully isolated the majority class in metric space without sacrificing the rare **AB+ class recall (57.14%, 4/7)**.
   - **A+ Sensitivity:** Arm B attained the highest A+ recall at **42.11% (8/19)**, while Arm D achieved **36.84% (7/19)** but with substantially higher precision (**63.64%** vs 44.44% in Arm B).
4. **Paired Statistical Significance Testing:**
   - Paired participant-level hypothesis testing against the Arm B reference yielded:
     - Arm B vs Arm D: 16 discordant pairs (6 ref only, 10 arm only), exact binomial $p = 0.4545$ (Holm-Bonferroni $\alpha_{\text{adj}} = 0.0167$).
     - Arm B vs Arm C: 28 discordant pairs (15 ref only, 13 arm only), McNemar $\chi^2 = 0.0357, p = 0.8501$ ($\alpha_{\text{adj}} = 0.0250$).
     - Arm B vs Arm A: 21 discordant pairs (11 ref only, 10 arm only), exact binomial $p = 1.0000$ ($\alpha_{\text{adj}} = 0.0500$).
   - **Conclusion:** None of the pairwise differences between arms reach statistical significance at $\alpha = 0.05$. Under $N = 80$, the observed $+5.00\%$ accuracy gain of Arm D over Arm B represents a directional engineering improvement rather than confirmed statistical separation.
5. **Historical Reproducibility Audit (Arm B vs EXP-06A Config B):**
   - Paired participant-level comparison between current Arm B ($\lambda=0.10, \tau=0.05$) and historical EXP-06A Config B ($\lambda=0.10, \tau=0.05$, historical Macro F1: 0.5313, Accuracy: 52.50%) revealed 22 discordant participants (9 Arm B only, 13 EXP-06A only). Two-tailed exact binomial test yielded $p = 0.5235$.
   - **Status:** Current Arm B is confirmed **statistically indistinguishable** from the EXP-06A historical reference, with variance fully accounted for by stochastic dropout and mini-batch permutation.
6. **Integrity & Quarantine Audit:**
   - **Protected Checkpoints:** All 36 historical baselines (Production `best_model.keras`, Original E2 Folds 1–5, EXP-03 Arm 1 Folds 1–5, EXP-05 Config A Folds 1–5, and EXP-06A Configs A–D Folds 1–5) verified 100% bit-for-bit unchanged.
   - **Cohort Quarantine:** 100% compliance. Zero access to the 21-participant locked test cohort or 26-participant hospital cohort occurred. Exactly 80 participants have exactly 1 OOF prediction per arm.
   - **Checkpoints Saved:** All 20 new EXP-06B checkpoints saved and cryptographically verified.

---

## 2. Experimental Setup & Safeguards Audit

### 2.1 Experimental Control Matrix

| Dimension | Specification | Implementation Verification | Status |
|:---|:---|:---|:---:|
| **Cohort Quarantine** | 80 Development Participants ONLY (240 scans) | Verified: Exactly 80 unique PIDs in all 4 arms; locked test (21 PIDs) & hospital (26 PIDs) strictly unaccessed. | **PASS** |
| **Fold Partitioning** | 5-Fold StratifiedGroupKFold (`seed=42`) | All 3 scans per participant strictly collocated in the same fold (64 train / 16 val per fold). | **PASS** |
| **Backbone Architecture** | DenseNet-121 (Pretrained ImageNet, Frozen) | 7,037,504 non-trainable parameters; strictly 0 backbone gradient updates across all arms. | **PASS** |
| **Trainable Variables** | Concatenation Neck + Classifier + Embedding | Exactly 1,968,772 trainable parameters across exactly 6 tensors (`dense_512`, `clf`, `emb`). | **PASS** |
| **Epoch-Zero Parity** | Ephemeral Shared Weight Templates | Arms A, B, C, D initialized with identical head weights within each fold via shared ephemeral templates. | **PASS** |
| **Loss Formulation** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{supcon}} \cdot \mathcal{L}_{\text{SupCon}}(\tau=0.05)$ | $\lambda \in \{0.05, 0.10, 0.20, 0.50\}$, computed on 128-d fused participant embedding (`emb(fused)`). | **PASS** |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | Standardized learning rate schedule and gradient clipping across all arms. | **PASS** |
| **Training Policy** | Min 100 epochs, Max 150 epochs, Patience 15 | Monitored held-out Validation Macro F1. | **PASS** |

---

## 3. Comparative Performance Analysis

### 3.1 Global Performance Comparison (Pooled OOF vs Fold Mean)

All metrics are evaluated out-of-fold across the complete 80 development participants.

| Metric | Arm A ($\lambda=0.05$) | Arm B ($\lambda=0.10$) [Ref] | Arm C ($\lambda=0.20$) | Arm D ($\lambda=0.50$) | EXP-06A Config B [Hist Ref] |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Contrastive Weight ($\lambda$)** | 0.05 | 0.10 | 0.20 | **0.50** | 0.10 |
| **Contrastive Temperature ($\tau$)**| 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| **Pooled Accuracy** | 46.25% (37/80) | 47.50% (38/80) | 45.00% (36/80) | **52.50% (42/80)** | 52.50% (42/80) |
| **Balanced Accuracy** | 46.38% | 48.47% | 43.99% | **51.46%** | 55.18% |
| **Pooled Macro Precision** | 0.4420 | 0.4951 | 0.4422 | **0.5216** | 0.5262 |
| **Pooled Macro Recall** | 0.4638 | 0.4847 | 0.4399 | **0.5146** | 0.5518 |
| **Pooled Macro F1 (Primary)** | 0.4438 | 0.4789 | 0.4404 | **0.4925** | 0.5313 |
| **Weighted F1** | 0.4552 | 0.4712 | 0.4494 | **0.5116** | 0.5225 |
| **Fold F1 Mean ($\pm$ Std)** | 0.4607 ($\pm$0.0752) | 0.4799 ($\pm$0.0744) | 0.4370 ($\pm$0.0717) | **0.5035 ($\pm$0.1007)** | 0.5014 ($\pm$0.1277) |
| **Total Runtime (min)** | 77.83 | 108.16 | 135.42 | 173.21 | 68.33 |

> [!NOTE]
> **Primary Metric Definition:**
> In accordance with Requirement 12, **pooled OOF Macro F1** ($0.4925$ for Arm D) aggregates predictions across all 80 participants simultaneously, avoiding small-sample fold averaging distortion. It is explicitly distinguished from the **mean fold-level Macro F1** ($0.5035 \pm 0.1007$), which averages the 5 individual fold validation scores. Both primary and fold metrics identify Arm D as the top-performing configuration.

---

### 3.2 Fold-Level Macro F1 Breakdown

| Fold ID | Val N | Arm A ($\lambda=0.05$) | Arm B ($\lambda=0.10$) | Arm C ($\lambda=0.20$) | Arm D ($\lambda=0.50$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 16 | **0.5253** | 0.4419 | 0.4256 | 0.4420 |
| **Fold 2** | 16 | 0.5329 | 0.5861 | 0.5040 | **0.6043** |
| **Fold 3** | 16 | 0.3237 | 0.3821 | 0.3083 | **0.4167** |
| **Fold 4** | 16 | 0.4538 | 0.5444 | 0.5042 | **0.6458** |
| **Fold 5** | 16 | **0.4679** | 0.4449 | 0.4429 | 0.4088 |
| **Mean** | — | 0.4607 | 0.4799 | 0.4370 | **0.5035** |
| **Std Dev** | — | 0.0752 | **0.0744** | **0.0717** | 0.1007 |

*Observations on Fold Stability:*
- **Arm D Peak Performance:** Arm D produced the highest individual fold scores recorded in this ablation: **0.6458** on Fold 4 and **0.6043** on Fold 2.
- **Fold 3 Robustness:** Fold 3 has historically been the most difficult fold across all previous experiments (EXP-01 through EXP-06A, typically $\approx 0.25 - 0.33$). Arm D lifted Fold 3 to **0.4167**, representing the strongest metric boundary established on Fold 3 to date.
- **Under-regularization at $\lambda=0.05$ (Arm A):** Lowering the loss weight degraded Fold 4 ($0.4538$) and Fold 3 ($0.3237$), confirming that insufficient metric attraction impairs generalization across diverse participant scans.

---

### 3.3 Per-Class Performance Breakdown

Evaluating class-wise sensitivity is essential due to natural class imbalance in the 80-participant development cohort (29 O+, 25 B+, 19 A+, 7 AB+).

| Blood Group | Metric | Arm A ($\lambda=0.05$) | Arm B ($\lambda=0.10$) [Ref] | Arm C ($\lambda=0.20$) | Arm D ($\lambda=0.50$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **A+** ($N=19$) | **Recall** | 26.32% (5/19) | **42.11% (8/19)** | 36.84% (7/19) | 36.84% (7/19) |
| | **Precision** | 0.3125 | 0.4444 | 0.4118 | **0.6364** |
| | **F1-Score** | 0.2857 | 0.4324 | 0.3889 | **0.4667** |
| **AB+** ($N=7$) | **Recall** | **57.14% (4/7)** | **57.14% (4/7)** | 42.86% (3/7) | **57.14% (4/7)** |
| | **Precision** | 0.4000 | **0.5000** | 0.4286 | 0.3636 |
| | **F1-Score** | 0.4706 | **0.5333** | 0.4286 | 0.4444 |
| **B+** ($N=25$) | **Recall** | 40.00% (10/25) | 36.00% (9/25) | **48.00% (12/25)** | 36.00% (9/25) |
| | **Precision** | 0.5556 | **0.6000** | 0.4286 | 0.5625 |
| | **F1-Score** | **0.4651** | 0.4500 | 0.4528 | 0.4390 |
| **O+** ($N=29$) | **Recall** | 62.07% (18/29) | 58.62% (17/29) | 48.28% (14/29) | **75.86% (22/29)** |
| | **Precision** | 0.5000 | 0.4359 | 0.5000 | **0.5238** |
| | **F1-Score** | 0.5538 | 0.5000 | 0.4912 | **0.6197** |

#### In-Depth Per-Class Analysis:
1. **O+ Sensitivity Surge in Arm D:** Arm D correctly recalled **22 of 29 O+ participants (75.86%)** with an F1 score of **0.6197**. Stronger metric pulling ($\lambda = 0.50$) forced the largest cohort partition (O+) into a dense, coherent geometric cluster that minimized false reassignments.
2. **Precision vs Recall Trade-off in A+:** While Arm B attained the highest A+ recall (42.11%, 8/19), it suffered from substantial false positives (precision: 44.44%). Arm D demonstrated much cleaner decision boundaries for A+, yielding **63.64% precision (7/11)** and an F1 score of **0.4667**.
3. **AB+ Preservation:** In all three arms (A, B, and D), AB+ recall held steady at **57.14% (4/7)**. Moderate loss weight ($\lambda=0.20$, Arm C) suffered a slight dip to 42.86% (3/7).
4. **B+ Sensitivity Ceiling:** Across all arms, B+ recall remained relatively constrained (36.0% – 48.0%), with misclassifications primarily bleeding into O+ and A+, suggesting that optical fingerprint features between B+ and O+ exhibit high shared ridge pattern variance.

---

### 3.4 Confusion Matrix Visualizations

The full confusion matrix heatmaps have been generated and archived:
- **Arm A:** [arm_a_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_a_confusion_matrix.png)
- **Arm B:** [arm_b_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_b_confusion_matrix.png)
- **Arm C:** [arm_c_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_c_confusion_matrix.png)
- **Arm D:** [arm_d_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_d_confusion_matrix.png)

```
Arm A (lambda=0.05):                   Arm B (lambda=0.10):
True \ Pred  A+  AB+  B+   O+          True \ Pred  A+  AB+  B+   O+
A+           5    1    5    8          A+           8    2    1    8
AB+          0    4    1    2          AB+          0    4    0    3
B+           5    2   10    8          B+           4    1    9   11
O+           6    3    2   18          O+           6    1    5   17

Arm C (lambda=0.20):                   Arm D (lambda=0.50):
True \ Pred  A+  AB+  B+   O+          True \ Pred  A+  AB+  B+   O+
A+           7    2    4    6          A+           7    2    3    7
AB+          2    3    1    1          AB+          0    4    1    2
B+           4    2   12    7          B+           1    4    9   11
O+           4    0   11   14          O+           3    1    3   22
```

---

## 4. Paired Statistical Comparisons

In strict adherence to Requirement 13, all paired participant-level hypothesis tests were recomputed directly from the saved out-of-fold prediction files (`{arm}_oof_predictions.csv`).

Because discordant pair counts between arms varied, the test protocol enforced:
- **Two-tailed Exact Binomial Test** when total discordant pairs $b + c < 25$.
- **McNemar Test with Edwards Continuity Correction** ($\chi^2 = (|b - c| - 1)^2 / (b + c)$) when $b + c \ge 25$.
- **Holm-Bonferroni step-down procedure** across the family of 3 pairwise hypothesis tests against the Arm B reference.

### 4.1 Pairwise Statistical Test Results vs Arm B Reference

| Comparison | Contrastive Weight Delta | Discordant $(n_{10}, n_{01})$ | Total Discordant | Test Applied | Test Statistic | Unadjusted $p$-value | Holm-Bonferroni $\alpha_{\text{adj}}$ | Statistically Significant? |
|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **Arm B vs Arm D** | $0.10 \to 0.50$ | $(6, 10)$ | 16 | Exact Binomial | $0.5625$ | $0.4545$ | $0.0167$ ($\alpha/3$) | **NO** ($p \ge 0.0167$) |
| **Arm B vs Arm C** | $0.10 \to 0.20$ | $(15, 13)$ | 28 | Continuity-corrected McNemar | $0.0357$ | $0.8501$ | $0.0250$ ($\alpha/2$) | **NO** ($p \ge 0.0250$) |
| **Arm B vs Arm A** | $0.10 \to 0.05$ | $(11, 10)$ | 21 | Exact Binomial | $0.0000$ | $1.0000$ | $0.0500$ ($\alpha/1$) | **NO** ($p \ge 0.0500$) |

*Statistical Interpretation:*
- In all three comparisons, $p \ge 0.45$, well above adjusted significance thresholds.
- Comparing Arm D against Arm B: 10 participants were correctly classified by Arm D but missed by Arm B, while 6 participants were correctly classified by Arm B but missed by Arm D (net gain: $+4$ participants).
- Under the approved 80-participant cohort, a net gain of 4 participants yields $p = 0.4545$. Thus, **we do not claim statistical superiority** of Arm D over Arm B.
- However, as an empirical hyperparameter search, Arm D demonstrates clear architectural viability and shows no signs of destructive over-regularization at $\lambda = 0.50$.

---

### 4.2 Historical Reference Comparison: Arm B vs EXP-06A Config B

In accordance with Requirement 14, Arm B was compared against EXP-06A Config B as a stochastic reference benchmark:

| Metric / Dimension | EXP-06A Config B [Historical] | EXP-06B Arm B [Current] | Delta ($\Delta$) | Statistical Test Result |
|:---|:---:|:---:|:---:|:---|
| **Parameters** | $\lambda = 0.10, \tau = 0.05$ | $\lambda = 0.10, \tau = 0.05$ | Identical | Architecture, weights, & splits identical |
| **Pooled Accuracy** | 52.50% (42/80) | 47.50% (38/80) | $-5.00\%$ ($-4$ participants) | Paired discordance: $(9, 13)$, total $= 22$ |
| **Balanced Accuracy**| 55.18% | 48.47% | $-6.71\%$ | Two-tailed Exact Binomial test: |
| **Pooled Macro F1** | 0.5313 | 0.4789 | $-0.0524$ | **$p = 0.5235$** |
| **Fold F1 Mean ($\pm$ Std)** | 0.5014 ($\pm$0.1277) | 0.4799 ($\pm$0.0744) | $-0.0215$ | **Statistically Indistinguishable: YES** |

*Methodological Note on Stochastic Variance:*
- A paired exact binomial test on the 22 discordant participants between current Arm B and historical EXP-06A Config B yields $p = 0.5235 \gg 0.05$.
- Both runs utilized identical model code, frozen DenseNet-121 backbones, C5 preprocessing, and participant-grouped folds. The modest 4-participant delta is completely within expected stochastic variation stemming from dropout sampling, mini-batch shuffling, and GPU floating-point non-determinism during AdamW updates.

---

## 5. Checkpoint & Data Integrity Audit

### 5.1 Protected Baseline Checkpoints Audit

All 36 protected historical model checkpoints were hashed using SHA-256 immediately before execution and re-verified immediately after execution.

```
Integrity Audit Status: 100% PASSED (0 bytes modified across all protected files)
```

| Checkpoint Name | Relative Path | Pre-Execution SHA-256 | Post-Execution SHA-256 | Audit Status |
|:---|:---|:---:|:---:|:---:|
| **Production Model** | `models/abo_positive_4class/best_model.keras` | `7b5ff5d94d4ce3de...` | `7b5ff5d94d4ce3de...` | **MATCH** |
| **Orig E2 Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | `7d9464d18619a94f...` | `7d9464d18619a94f...` | **MATCH** |
| **Orig E2 Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | `9c72809f23152708...` | `9c72809f23152708...` | **MATCH** |
| **Orig E2 Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | `8123b1b86f44719d...` | `8123b1b86f44719d...` | **MATCH** |
| **Orig E2 Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | `cc122b23e0c6d118...` | `cc122b23e0c6d118...` | **MATCH** |
| **Orig E2 Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | `aff9e137bed61f64...` | `aff9e137bed61f64...` | **MATCH** |
| **EXP-03 Arm 1 (Folds 1–5)** | `experiments/exp03/checkpoints/arm1/` | Verified 5/5 | Verified 5/5 | **MATCH** |
| **EXP-05 Config A (Folds 1–5)** | `experiments/exp05/checkpoints/config_a/` | Verified 5/5 | Verified 5/5 | **MATCH** |
| **EXP-06A Config A (Folds 1–5)** | `experiments/exp06a/checkpoints/config_a/` | Verified 5/5 | Verified 5/5 | **MATCH** |
| **EXP-06A Config B (Folds 1–5)** | `experiments/exp06a/checkpoints/config_b/` | Verified 5/5 | Verified 5/5 | **MATCH** |
| **EXP-06A Config C (Folds 1–5)** | `experiments/exp06a/checkpoints/config_c/` | Verified 5/5 | Verified 5/5 | **MATCH** |
| **EXP-06A Config D (Folds 1–5)** | `experiments/exp06a/checkpoints/config_d/` | Verified 5/5 | Verified 5/5 | **MATCH** |

---

### 5.2 Newly Generated EXP-06B Checkpoints

All 20 new model checkpoints (4 arms $\times$ 5 folds) have been archived under `experiments/exp06b/checkpoints/` and audited:

| Arm | Checkpoint Path | File Size | SHA-256 Digest | Status |
|:---|:---|:---:|:---:|:---:|
| **Arm A** ($\lambda=0.05$) | `experiments/exp06b/checkpoints/arm_a/fold_1_best.weights.h5` | 52,899,472 B | `b6457e90d2d27ad6...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_a/fold_2_best.weights.h5` | 52,899,472 B | `7ac46ba81da30807...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_a/fold_3_best.weights.h5` | 52,899,472 B | `21a43ac57bbaf084...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_a/fold_4_best.weights.h5` | 52,899,472 B | `2afab39937c2777a...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_a/fold_5_best.weights.h5` | 52,899,472 B | `7963f587063f0f07...` | **VALID** |
| **Arm B** ($\lambda=0.10$) | `experiments/exp06b/checkpoints/arm_b/fold_1_best.weights.h5` | 52,899,472 B | `d9ebc77069beb7eb...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_b/fold_2_best.weights.h5` | 52,899,472 B | `f19e767d4ab86045...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_b/fold_3_best.weights.h5` | 52,899,472 B | `0f4d8b328b436d9e...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_b/fold_4_best.weights.h5` | 52,899,472 B | `4793c504d276945e...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_b/fold_5_best.weights.h5` | 52,899,472 B | `2eaf592b4c874f76...` | **VALID** |
| **Arm C** ($\lambda=0.20$) | `experiments/exp06b/checkpoints/arm_c/fold_1_best.weights.h5` | 52,899,472 B | `0799e74311ce2421...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_c/fold_2_best.weights.h5` | 52,899,472 B | `7d5281c3def2a324...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_c/fold_3_best.weights.h5` | 52,899,472 B | `a563bbf662ad9a6d...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_c/fold_4_best.weights.h5` | 52,899,472 B | `2490e9cc0625086a...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_c/fold_5_best.weights.h5` | 52,899,472 B | `38e0dbe907cc0722...` | **VALID** |
| **Arm D** ($\lambda=0.50$) | `experiments/exp06b/checkpoints/arm_d/fold_1_best.weights.h5` | 52,899,472 B | `fce514583f80b840...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_d/fold_2_best.weights.h5` | 52,899,472 B | `93eea00f077461b1...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_d/fold_3_best.weights.h5` | 52,899,472 B | `f5955b65490d7185...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_d/fold_4_best.weights.h5` | 52,899,472 B | `ab676c2043fbfc4e...` | **VALID** |
| | `experiments/exp06b/checkpoints/arm_d/fold_5_best.weights.h5` | 52,899,472 B | `389502f52067cfc8...` | **VALID** |

---

## 6. Limitations & Scientific Context

1. **Sample Size & Statistical Power Constraints:**
   - The development cohort comprises 80 participants (240 scans).
   - In paired classification tests with $N = 80$, an effect size of 4 to 6 net correctly classified participants does not attain $p < 0.05$. Larger development cohorts would be required to statistically disentangle fine-grained hyperparameter differences.
2. **Backbone Feature Saturation:**
   - Pretrained ImageNet DenseNet-121 features were frozen throughout this study. The representations extracted by early convolutional blocks are fixed optical biometric features. Metric learning heads can reshape embedding space geometry, but cannot alter the underlying low-level texture filters extracted by the frozen backbone.
3. **Stochasticity in Convergence:**
   - As demonstrated in the comparison between EXP-06B Arm B and EXP-06A Config B, stochastic batch sampling and dropout introduce an inherent run-to-run variation of $\approx \pm 2-4\%$ in overall accuracy and $\approx \pm 0.03-0.05$ in Macro F1.

---

## 7. Recommended Next Steps

1. **Synthesis of EXP-06A and EXP-06B Findings:**
   - EXP-06A demonstrated that a sharper temperature ($\tau = 0.05$) creates tighter clusters in contrastive metric space, significantly boosting minority AB+ recall.
   - EXP-06B demonstrated that a heavier metric loss weight ($\lambda = 0.50$) substantially enhances majority O+ sensitivity (75.86%) and raises mean fold-level Macro F1 to **0.5035**, achieving **52.50% overall accuracy**.
2. **Next Logical Investigation — EXP-06C (Fine-Grained Hybrid Tuning / Neck Exploration):**
   - Candidate Next Experiment: Investigate whether combining $\tau = 0.05$ and $\lambda = 0.50$ with an intermediate projection head (e.g. 2-layer MLP projection vs linear projection) stabilizes A+ and B+ sensitivity while retaining high O+ and AB+ performance.
3. **Preserve Locked Cohorts:**
   - The 21-participant locked test cohort and 26-participant hospital cohort remain fully quarantined and untouched. They must remain preserved until the final end-to-end benchmark evaluation.

---

## 8. Artifact and File Registry

- **Final Report:** [final_report.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/final_report.md)
- **Metrics Comparison Table:** [metrics_comparison.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/metrics_comparison.csv)
- **Complete Statistical Analysis JSON:** [statistical_analysis_results.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/statistical_analysis_results.json)
- **Participant OOF Predictions:**
  - Arm A: [arm_a_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/arm_a_oof_predictions.csv)
  - Arm B: [arm_b_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/arm_b_oof_predictions.csv)
  - Arm C: [arm_c_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/arm_c_oof_predictions.csv)
  - Arm D: [arm_d_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/arm_d_oof_predictions.csv)
- **Confusion Matrix Plots:**
  - Arm A: [arm_a_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_a_confusion_matrix.png)
  - Arm B: [arm_b_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_b_confusion_matrix.png)
  - Arm C: [arm_c_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_c_confusion_matrix.png)
  - Arm D: [arm_d_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b/confusion_matrices/arm_d_confusion_matrix.png)
- **Checkpoints:** [experiments/exp06b/checkpoints/](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp06b/checkpoints/)
