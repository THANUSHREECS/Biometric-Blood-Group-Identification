# EXP-06C: Statistical Analysis Plan & Bootstrap Protocol

**Document Version:** 1.0.0  
**Preparation Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Lead Statistician & ML Auditor:** Senior ML Research Scientist & Verification Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All statistical methods, metric formulations, and inference plans documented herein are academic research protocols designed to evaluate algorithm performance on optical biometric datasets. They **must not** be used for clinical decision-making, patient diagnostic validation, blood typing, or transfusion matching.

---

## 1. Overview & Primary Evaluation Standard

In accordance with mandatory research requirements, **Pooled Out-of-Fold (OOF) Macro F1** across all 80 development participants is established as the **primary decision metric** for EXP-06C.

To avoid the small-sample averaging distortions inherent in fold-level unweighted means, the primary metric evaluates the concatenated out-of-fold prediction vector across the complete 80-participant development cohort:

$$\text{Macro F1}_{\text{pooled}} = \frac{1}{4} \sum_{c \in \{A+, AB+, B+, O+\}} F1_c$$

Where $F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$, computed globally on all 80 held-out participant predictions.

### Formal Metric Distinction
- **Primary Metric:** Pooled OOF Macro F1 (single aggregate value per configuration across all 80 participants).
- **Secondary Diagnostic Metric:** Mean fold-level Macro F1 ($\mu_{\text{fold}} \pm \sigma_{\text{fold}}$ across the 5 validation folds).
- Both metrics will be explicitly reported and clearly distinguished in all tables and text.

---

## 2. Statistical Testing Protocols

Two complementary statistical testing procedures will evaluate the empirical difference between **Config A (Linear Reference)** and **Config B (Nonlinear MLP)**:

1. **Paired Classification Correctness:** McNemar's Test / Two-Tailed Exact Binomial Test.
2. **Participant-Level Paired Bootstrap Confidence Interval:** For the paired difference in pooled Macro F1 ($\Delta \text{Macro F1}$).

---

### 2.1 Paired Classification Correctness Test (McNemar / Exact Binomial)

To determine whether the discrete classification correctness between Config A and Config B differs significantly, a paired $2 \times 2$ contingency table is constructed across the 80 development participants:

| Participant Outcome | Config B Correct ($Y_B = 1$) | Config B Incorrect ($Y_B = 0$) | Marginal Total |
|:---|:---:|:---:|:---:|
| **Config A Correct ($Y_A = 1$)** | $n_{11}$ (Both Correct) | $n_{10}$ (Config A only) | $n_{11} + n_{10}$ |
| **Config A Incorrect ($Y_A = 0$)** | $n_{01}$ (Config B only) | $n_{00}$ (Both Incorrect) | $n_{01} + n_{00}$ |
| **Marginal Total** | $n_{11} + n_{01}$ | $n_{10} + n_{00}$ | $N = 80$ |

#### Decision Rules:
- Let discordant pairs be $b = n_{10}$ and $c = n_{01}$. Total discordant pairs $D = b + c$.
- **Small-Sample Regime ($D < 25$):**  
  Execute a **Two-Tailed Exact Binomial Test** under null hypothesis $H_0: p = 0.5$:
  $$p\text{-value} = 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{D}{k} (0.5)^D$$
- **Large-Sample Regime ($D \ge 25$):**  
  Execute **McNemar's Test with Edwards Continuity Correction**:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{with } \text{df} = 1$$
  $$p\text{-value} = P(\chi^2_1 \ge \chi^2_{\text{calc}})$$

---

### 2.2 Participant-Level Paired Bootstrap Protocol

While McNemar's test assesses binary 0/1 correctness, Macro F1 is a harmonic mean of precision and recall that depends nonlinearly on the entire confusion matrix. Therefore, a **participant-level paired non-parametric bootstrap** will be executed to quantify uncertainty around the Macro F1 difference:

$$\Delta \text{Macro F1} = \text{Macro F1}(\text{Config B}) - \text{Macro F1}(\text{Config A})$$

#### Bootstrap Specification:
1. **Resampling Unit:** Strictly the **individual participant** ($N = 80$).
   - *Critical Constraint:* Resampling must **never** be performed on individual scans. Because all 3 scans of a participant are aggregated into a single out-of-fold prediction, resampling participants maintains complete intra-participant scan collocation and prevents cluster-correlation distortion.
2. **Number of Replicates:** $B = 2,000$ bootstrap iterations.
3. **Random Seed:** Formally locked to `seed = 42` for exact reproducibility.
4. **Resampling Algorithm:**
   For $k = 1, \dots, B$:
   a. Draw an index sample $\mathbf{i}_k$ of size $N = 80$ with replacement from $\{1, \dots, 80\}$.
   b. Extract resampled ground truth $\mathbf{y}^* = \mathbf{y}[\mathbf{i}_k]$, Config A predictions $\hat{\mathbf{y}}_A^* = \hat{\mathbf{y}}_A[\mathbf{i}_k]$, and Config B predictions $\hat{\mathbf{y}}_B^* = \hat{\mathbf{y}}_B[\mathbf{i}_k]$.
   c. Compute $\text{Macro F1}_A^{(k)}$ and $\text{Macro F1}_B^{(k)}$ on the resampled vectors.
   d. Compute paired difference: $\Delta^{(k)} = \text{Macro F1}_B^{(k)} - \text{Macro F1}_A^{(k)}$.
5. **Confidence Interval Estimation:**
   - 95% Percentile Bootstrap Confidence Interval:
     $$\text{CI}_{95\%} = \left[ \Delta_{(0.025 \cdot B)}, \; \Delta_{(0.975 \cdot B)} \right]$$
6. **Inference Rule:**
   - If the 95% bootstrap confidence interval includes zero ($0.0 \in \text{CI}_{95\%}$), the Macro F1 difference **cannot** be concluded to be statistically distinct from zero at $\alpha = 0.05$.
   - The empirical two-tailed bootstrap $p$-value will be reported alongside the interval.

---

## 3. Multiple-Comparison & Reproducibility Analysis

### 3.1 Multiple-Comparison Correction
If secondary exploratory comparisons are conducted (e.g., comparing Config A to historical EXP-06B Arm D, or evaluating individual fold subsets), the **Holm-Bonferroni step-down procedure** will be applied across the family of $M$ hypothesis tests to strictly control Family-Wise Error Rate (FWER) at $\alpha = 0.05$:

$$\alpha_{(i)} = \frac{0.05}{M - i + 1}, \quad \text{for ordered } p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$$

### 3.2 Historical Reference Benchmark (Config A vs EXP-06B Arm D)
Config A will also be compared against historical EXP-06B Arm D ($0.4925$ Macro F1, $52.50\%$ accuracy):
- Concordance and discordant pairs between Config A and EXP-06B Arm D will be audited.
- Status will be evaluated to verify that Config A remains within expected stochastic variation ($p \ge 0.05$).

---

## 4. Scientific Restraint Guidelines

In accordance with Requirement 16:
- **No Claims of Superiority Based on Accuracy Alone:** Raw accuracy is easily inflated by majority class bias (O+ and B+). A configuration will not be declared superior unless supported by both balanced accuracy and Macro F1.
- **Statistical Significance Requirement:** No claim of statistical superiority will be made unless the paired statistical test achieves $p < 0.05$ and the 95% bootstrap confidence interval excludes zero.
- **Non-Clinical Boundaries:** Under no circumstances will any finding be interpreted as proving clinical validity or suitability for human blood group identification.
