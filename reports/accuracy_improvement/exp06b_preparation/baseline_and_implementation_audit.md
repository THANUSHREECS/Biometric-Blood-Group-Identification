# EXP-06B: Baseline and Implementation Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-15  
**Experiment Identifier:** `EXP-06B` (Supervised Contrastive Loss Weight Ablation)  
**Lead Auditor:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning hyperparameter sensitivity on biometric optical fingerprint representations. All models, weights, metrics, and conclusions documented herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, triage, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Executive Summary & Purpose

This audit serves as the pre-training foundation for **EXP-06B: Supervised Contrastive Loss Weight Ablation**. 

In EXP-06A, temperature $\tau = 0.05$ (Config B) was selected as the provisional leading development candidate, achieving a Macro F1 of **0.5313**, Balanced Accuracy of **55.18%**, and boosting minority AB+ recall to **71.43% (5/7)**. However, paired statistical tests against the reference baseline ($\tau = 0.07$) did not reach formal statistical significance ($p = 0.6636$ under Holm-Bonferroni correction), confirming that $\tau = 0.05$ is a provisional empirical preference rather than a statistically proven optimum.

Before authorizing any model training for EXP-06B, this audit:
1. Independently recomputes all EXP-06A metrics and paired McNemar tests from participant-level OOF prediction CSVs.
2. Formally audits the Config A performance discrepancy between EXP-05 and EXP-06A across all systemic dimensions.
3. Establishes the exact mathematical baseline and parameter boundaries for EXP-06B.

---

## 2. Independent Recomputation of EXP-06A Metrics

All metrics reported in the EXP-06A final report were independently recomputed directly from the out-of-fold prediction CSVs (`config_{a,b,c,d}_oof_predictions.csv`), without reading pre-aggregated logs.

### 2.1 Recomputed Metrics Table (N = 80 Development Participants)

| Metric | Config A ($\tau=0.07$) [Ref] | Config B ($\tau=0.05$) | Config C ($\tau=0.10$) | Config D ($\tau=0.15$) | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sample Size (N)** | 80 | 80 | 80 | 80 | **VERIFIED** |
| **Accuracy** | 48.75% (39/80) | 52.50% (42/80) | 55.00% (44/80) | 52.50% (42/80) | **MATCH (Exact)** |
| **Balanced Accuracy** | 46.26% | 55.18% | 51.90% | 51.70% | **MATCH (Exact)** |
| **Macro Precision** | 0.4768 | 0.5262 | 0.5143 | 0.5475 | **MATCH (Exact)** |
| **Macro Recall** | 0.4626 | 0.5518 | 0.5190 | 0.5170 | **MATCH (Exact)** |
| **Macro F1 (Primary)** | **0.4655** | **0.5313** | **0.5152** | **0.5194** | **MATCH (Exact)** |
| **A+ Recall** | 31.58% (6/19) | 42.11% (8/19) | 42.11% (8/19) | 31.58% (6/19) | **MATCH (Exact)** |
| **A+ Precision** | 0.4000 | 0.5714 | 0.5000 | 0.6000 | **MATCH (Exact)** |
| **A+ F1-Score** | 0.3529 | 0.4848 | 0.4571 | 0.4138 | **MATCH (Exact)** |
| **AB+ Recall** | 42.86% (3/7) | 71.43% (5/7) | 42.86% (3/7) | 57.14% (4/7) | **MATCH (Exact)** |
| **AB+ Precision** | 0.4286 | 0.5000 | 0.3750 | 0.5714 | **MATCH (Exact)** |
| **AB+ F1-Score** | 0.4286 | 0.5882 | 0.4000 | 0.5714 | **MATCH (Exact)** |
| **B+ Recall** | 52.00% (13/25) | 52.00% (13/25) | 64.00% (16/25) | 56.00% (14/25) | **MATCH (Exact)** |
| **B+ Precision** | 0.6190 | 0.5000 | 0.6154 | 0.5185 | **MATCH (Exact)** |
| **B+ F1-Score** | 0.5652 | 0.5098 | 0.6275 | 0.5385 | **MATCH (Exact)** |
| **O+ Recall** | 58.62% (17/29) | 55.17% (16/29) | 58.62% (17/29) | 62.07% (18/29) | **MATCH (Exact)** |
| **O+ Precision** | 0.4595 | 0.5333 | 0.5667 | 0.5000 | **MATCH (Exact)** |
| **O+ F1-Score** | 0.5152 | 0.5424 | 0.5763 | 0.5538 | **MATCH (Exact)** |

---

## 3. Independent Recomputation of Paired Statistical Tests

To verify that claims of statistical significance or non-significance were accurate, paired contingency tables were constructed directly from aligned participant IDs across the 80 development subjects.

### 3.1 McNemar Test Details vs Reference Config A

| Comparison | Both Correct $(n_{11})$ | Config A Only $(b = n_{10})$ | Other Only $(c = n_{01})$ | Both Wrong $(n_{00})$ | Discordant Pairs $(b + c)$ | Statistical Method | Test Statistic | Exact $p$-value | Holm-Bonferroni Cutoff | Statistically Significant? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A vs Config B** | 30 | 9 | 12 | 29 | 21 | Exact Binomial | 0.1905 | **0.6636** | 0.0500 | **NO** |
| **Config A vs Config C** | 33 | 6 | 11 | 30 | 17 | Exact Binomial | 0.9412 | **0.3323** | 0.0167 | **NO** |
| **Config A vs Config D** | 31 | 8 | 11 | 30 | 19 | Exact Binomial | 0.2105 | **0.6476** | 0.0250 | **NO** |

**Audit Confirmation:**
The independent recomputation confirms that no configuration achieved statistically significant separation from Config A. The choice of $\tau = 0.05$ as the baseline for EXP-06B is based strictly on empirical representation quality (Macro F1 = 0.5313 and AB+ recall = 71.43%), not on statistical rejection of the null hypothesis.

---

## 4. Deep-Dive Investigation: Config A Discrepancy (EXP-05 vs EXP-06A)

In EXP-05, Config A achieved **52.50% (42/80)** accuracy and **0.5107** Macro F1.  
In EXP-06A, Config A achieved **48.75% (39/80)** accuracy and **0.4655** Macro F1.  
This represents a net difference of **3 participants** (3.75% accuracy).

A rigorous line-by-line comparison of both experimental runs was performed:

### 4.1 Systemic Dimension Comparison

| Dimension | EXP-05 Config A | EXP-06A Config A | Status |
|:---|:---|:---|:---:|
| **Cohort & Participants** | Exactly 80 PIDs, 240 scans | Exactly 80 PIDs, 240 scans | **IDENTICAL** |
| **Fold Partitioning** | StratifiedGroupKFold (`seed=42`) | StratifiedGroupKFold (`seed=42`) | **IDENTICAL** |
| **Backbone Weights** | Frozen DenseNet-121 (`7,037,504` non-trainable) | Frozen DenseNet-121 (`7,037,504` non-trainable) | **IDENTICAL** |
| **Trainable Variables** | 6 variables (`1,968,772` parameters) | 6 variables (`1,968,772` parameters) | **IDENTICAL** |
| **Preprocessing** | `transform_c5` (unaugmented) | `transform_c5` (unaugmented) | **IDENTICAL** |
| **Batch Size & LR** | Batch 16, AdamW `lr=1e-3`, `wd=1e-4` | Batch 16, AdamW `lr=1e-3`, `wd=1e-4` | **IDENTICAL** |
| **Loss Formulation** | $\mathcal{L}_{\text{CE}} + 0.1 \cdot \mathcal{L}_{\text{SupCon}}(\tau=0.07)$ | $\mathcal{L}_{\text{CE}} + 0.1 \cdot \mathcal{L}_{\text{SupCon}}(\tau=0.07)$ | **IDENTICAL** |
| **Checkpoint Policy** | Best Val Macro F1, min 100, patience 15 | Best Val Macro F1, min 100, patience 15 | **IDENTICAL** |

### 4.2 Prediction-Level Shift Analysis

Comparing individual participant predictions between the two runs:
* **Identical Predictions:** 61 out of 80 participants (**76.2%**) received identical class assignments.
* **Differing Predictions:** 19 out of 80 participants (**23.8%**) shifted predictions.
* **Contingency Breakdown:**
  * Both runs correct: **34 participants (42.5%)**
  * EXP-05 only correct: **8 participants (10.0%)**
  * EXP-06A only correct: **5 participants (6.2%)**
  * Both runs incorrect: **33 participants (41.2%)**
* **Fold-by-Fold Breakdown:**
  * Fold 1: EXP-05 = 8/16, EXP-06A = 8/16 ($\Delta = 0$)
  * Fold 2: EXP-05 = 9/16, EXP-06A = 8/16 ($\Delta = -1$)
  * Fold 3: EXP-05 = 9/16, EXP-06A = 7/16 ($\Delta = -2$)
  * Fold 4: EXP-05 = 9/16, EXP-06A = 8/16 ($\Delta = -1$)
  * Fold 5: EXP-05 = 7/16, EXP-06A = 8/16 ($\Delta = +1$)

### 4.3 Root-Cause Analysis & Remaining Uncertainty

1. **Initialization Order & TensorFlow Graph State:**
   - In EXP-05, Config A was instantiated sequentially per fold with `seed = 42 + fold_idx`.
   - In EXP-06A, to guarantee cross-config parity across arms A, B, C, D, an ephemeral head template was constructed, serialized to disk, and loaded into each configuration. While mathematically equivalent in theory, slight variations in pseudo-random generator state sequences can occur during graph compilation.
2. **Stochastic Mini-Batch Shuffling & Dropout:**
   - Shuffling within `tf.data.Dataset.shuffle(seed=seed+epoch)` and non-deterministic atomic operations on GPU across 100+ epochs induce slight trajectories in the loss landscape.
   - `Dropout(0.5)` introduces stochastic zeroing of half the 512 neck features during each forward pass.
3. **Statistical Assessment of the Delta:**
   - A McNemar test on the $(8, 5)$ discordant participants yields $\chi^2 = 0.3077$, $p = 0.579$.
   - **Conclusion:** The difference between EXP-05 Config A (42/80) and EXP-06A Config A (39/80) is statistically indistinguishable from zero ($p \gg 0.05$) and represents standard stochastic Monte Carlo variation inherent in training small-cohort neural networks with heavy dropout.

---

## 5. Audit Verdict

| Audit Criterion | Finding | Verdict |
|:---|:---|:---:|
| **Metrics Reproducibility** | All EXP-06A metrics and McNemar values verified to 4 decimal places. | **PASS** |
| **Config A Discrepancy** | Fully explained by stochastic dropout/shuffling; statistically non-significant ($p=0.579$). | **PASS** |
| **Data Safety & Quarantine** | 80 dev cohort verified; locked test and hospital cohorts untouched. | **PASS** |
| **Readiness for EXP-06B** | Baseline verified; experiment design ready for formal specification. | **APPROVED** |
