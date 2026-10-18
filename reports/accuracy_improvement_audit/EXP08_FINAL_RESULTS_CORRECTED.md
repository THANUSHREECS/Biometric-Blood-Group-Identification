# EXP-08-DERMATOGLYPHIC-MINUTIAE: Corrected Final Scientific Results & Evaluation Report

**Document ID:** `EXP08_FINAL_RESULTS_CORRECTED.md`  
**Original Reference:** `reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md` (Preserved Unmodified)  
**Correction Date:** 2026-10-18  
**Experiment Name:** EXP-08-DERMATOGLYPHIC-MINUTIAE (Handcrafted Biometric Feature Benchmark)  
**Project:** Biometric Blood Group Detection Using Fingerprint Images  
**Hardware Platform:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI, Native $316 \times 354$ Pixels)  
**Authorized Cohort:** 80 Development Participants (240 Raw Impressions, 3 scans per participant)  
**Target Classes:** 4-Class ABO Positive (`A+`, `AB+`, `B+`, `O+`)  
**Class Distribution:** A+: 19 (23.75%), AB+: 7 (8.75%), B+: 25 (31.25%), O+: 29 (36.25%)  
**Evaluation Protocol:** 5-Fold Participant-Level StratifiedGroupKFold with In-Fold Imputation and Feature Scaling  
**Audit Status:** AUDITED & RECONCILED AGAINST AUTHORITATIVE JSON METRICS  

---

## Document Purpose & Erratum Notice

This document is the **official corrected version** of the EXP-08 evaluation report. It reconciles the two known legacy discrepancies identified during the independent consistency audit:
1. **Permutation Test Provenance:** Clarifies the distinction between the legacy fixed-prediction shuffle ($p = 0.3250$) and the authoritative **full-pipeline participant permutation test** ($N=1000$ iterations with in-fold refitting), which established $p = \mathbf{0.2537}$ for Random Forest and $p = \mathbf{0.6963}$ for Logistic Regression.
2. **Majority Baseline Definition:** Clarifies the difference between the constant cohort-majority baseline ($36.25\%$, 29/80) and the fold-local `DummyClassifier` baseline ($28.75\%$, 23/80).

The original report [EXP08_FINAL_RESULTS.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md) and all underlying prediction artifacts remain preserved untouched.

---

## 1. Executive Summary

EXP-08 investigated whether biologically interpretable, handcrafted dermatoglyphic features (primary pattern classification, standardized sub-core ridge density, crossing-number minutiae, and core-to-delta distances) extract predictive signal for 4-class ABO blood-group classification on optical fingerprint scans.

Feature extraction succeeded on **100% of the 240 scans** without interpolation, cropping, or scale distortion. All 3 impressions per participant were collapsed into **exactly one feature row**, producing an $(80, 28)$ dataset with 24 numeric features evaluated under strict participant-level 5-fold cross-validation.

### Primary Experimental Findings:

1. **Failure to Beat Majority Base Rate:**
   - The empirical majority class base rate in Dev-80 is **$36.25\%$** (29/80 participants are O+).
   - A Constant Majority Baseline predicting O+ for all participants achieves **$36.25\%$** (29/80).
   - Random Forest achieved an overall accuracy of **$35.00\%$** (28/80, 95% CI: $[25.5\%, 45.9\%]$), falling **below the constant majority class base rate**.
   - L2 Logistic Regression achieved **$27.50\%$** (22/80, 95% CI: $[18.9\%, 38.1\%]$).
2. **Nonsignificance on Primary Metric (Balanced Accuracy):**
   - The primary metric, **Balanced Accuracy**, was **$27.06\%$** for Random Forest and **$21.15\%$** for Logistic Regression (theoretical chance = $25.0\%$).
   - In the authoritative 1,000-run full-pipeline permutation test (refitting scaling, imputation, and classifier per fold on shuffled labels):
     - **Random Forest:** Observed = $27.06\%$, Null Mean = $23.62\%$, Null Max = $43.20\%$, **$p = \mathbf{0.2537}$** ($254/1001$).
     - **Logistic Regression:** Observed = $21.15\%$, Null Mean = $24.30\%$, Null Max = $45.96\%$, **$p = \mathbf{0.6963}$** ($697/1001$).
3. **Complete Failure on Minority Class (AB+):**
   - Both Logistic Regression and Random Forest achieved **$0.0\%$ recall on AB+** ($0/7$ participants correctly identified).
4. **Standardized Scientific Conclusion:**
   - *Under the evaluated datasets, fingerprint representations, and participant-level validation procedures, the experiments did not establish reliable, generalizable blood-group classification performance.*

---

## 2. Experimental Benchmark Comparison (Reconciled Table)

| Model Architecture | Out-of-Fold Accuracy (95% Wilson CI) | Primary Metric: Balanced Accuracy | Full-Pipeline Permutation $p$-value | Macro Precision | Macro Recall | Macro F1 | Cohen's Kappa | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Majority Baseline (All O+)** | 36.25% [26.57%, 47.19%] | 25.00% | — | 0.0906 | 0.2500 | 0.1330 | 0.0000 | **VERIFIED** |
| **In-Fold Dummy Baseline (Fold Majority)** | 28.75% [19.99%, 39.46%] | 20.10% | $p = 0.9760$ | 0.1133 | 0.2010 | 0.1373 | -0.1004 | **VERIFIED (Fold-Local)** |
| **L2 Logistic Regression ($C=1.0$)** | 27.50% [18.92%, 38.14%] | 21.15% | **$p = \mathbf{0.6963}$** | 0.2311 | 0.2115 | 0.2193 | -0.0030 | **VERIFIED** |
| **Random Forest ($n=100, d=4$)** | **35.00% [25.45%, 45.92%]** | **27.06%** | **$p = \mathbf{0.2537}$** | **0.2533** | **0.2706** | **0.2603** | **+0.0571** | **VERIFIED** |

*Note on In-Fold Dummy Baseline:* In fold 2, holdout of O+ samples caused B+ to become the most frequent training-fold class (23 B+ vs 21 O+), causing fold-2 predictions to be B+ and yielding 23/80 (28.75%) correct predictions overall. The global constant majority baseline predicting O+ for all participants is 29/80 (36.25%).

---

## 3. Detailed Model Evaluations & Confusion Matrices

All confusion matrices and per-class metrics are reported in canonical class order: `["A+", "AB+", "B+", "O+"]`.

### A. Constant Majority Baseline (Predicts O+ for all 80 participants)
- **Correct Predictions:** 29 / 80 ($36.25\%$)
- **Balanced Accuracy:** $25.00\%$
- **Macro F1:** $0.1330$
- **Cohen's Kappa:** $0.0000$

```
Actual \ Predicted    A+    AB+    B+    O+    Total
A+                     0      0     0    19       19
AB+                    0      0     0     7        7
B+                     0      0     0    25       25
O+                     0      0     0    29       29
Total                  0      0     0    80       80
```

---

### B. L2-Regularized Logistic Regression
- **Correct Predictions:** 22 / 80 ($27.50\%$)
- **Balanced Accuracy:** $21.15\%$ (Full-pipeline permutation $p = 0.6963$)
- **Macro F1:** $0.2193$ ($p = 0.6254$)
- **Cohen's Kappa:** $-0.0030$

```
Actual \ Predicted    A+    AB+    B+    O+    Total
A+                     3      3     5     8       19
AB+                    2      0     2     3        7
B+                     8      6     6     5       25
O+                     8      3     5    13       29
Total                 21     12    18    29       80
```

#### Per-Class Performance:
| Blood Group Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **A+** | 14.29% | 15.79% (3/19) | 0.1500 | 19 |
| **AB+** | 0.00% | 0.00% (0/7) | 0.0000 | 7 |
| **B+** | 33.33% | 24.00% (6/25) | 0.2791 | 25 |
| **O+** | 44.83% | 44.83% (13/29) | 0.4483 | 29 |

---

### C. Random Forest Classifier
- **Correct Predictions:** 28 / 80 ($35.00\%$)
- **Balanced Accuracy:** $27.06\%$ (Full-pipeline permutation $p = 0.2537$)
- **Macro F1:** $0.2603$ ($p = 0.2817$)
- **Cohen's Kappa:** $+0.0571$

```
Actual \ Predicted    A+    AB+    B+    O+    Total
A+                     4      1     5     9       19
AB+                    1      0     4     2        7
B+                     6      2     8     9       25
O+                     6      0     7    16       29
Total                 17      3    24    36       80
```

#### Per-Class Performance:
| Blood Group Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **A+** | 23.53% | 21.05% (4/19) | 0.2222 | 19 |
| **AB+** | 0.00% | 0.00% (0/7) | 0.0000 | 7 |
| **B+** | 33.33% | 32.00% (8/25) | 0.3265 | 25 |
| **O+** | 44.44% | 55.17% (16/29) | 0.4923 | 29 |

---

## 4. Full-Pipeline Permutation Test Results ($B=1000$)

Authoritative results loaded directly from [exp08_full_pipeline_permutation_metrics.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/permutation_test/exp08_full_pipeline_permutation_metrics.json):

| Metric / Parameter | Random Forest | Logistic Regression |
| :--- | :---: | :---: |
| **Total Permutations ($B$)** | 1,000 | 1,000 |
| **Observed Accuracy** | $35.00\%$ | $27.50\%$ |
| **Observed Balanced Accuracy** | $27.06\%$ | $21.15\%$ |
| **Null Balanced Accuracy Mean $\pm$ Std** | $23.62\% \pm 5.45\%$ | $24.30\% \pm 5.88\%$ |
| **Null Balanced Accuracy 95th Percentile** | $33.09\%$ | $34.46\%$ |
| **Null Balanced Accuracy Maximum** | **$43.20\%$** | **$45.96\%$** |
| **Permutations $\ge$ Observed Balanced Acc** | 253 / 1000 | 696 / 1000 |
| **Empirical $p$-value (Balanced Acc)** | **$p = \mathbf{0.2537}$** | **$p = \mathbf{0.6963}$** |
| **Permutations $\ge$ Observed Macro F1** | 281 / 1000 | 625 / 1000 |
| **Empirical $p$-value (Macro F1)** | **$p = \mathbf{0.2817}$** | **$p = \mathbf{0.6254}$** |

---

## 5. Mandatory Research & Clinical Disclaimer

1. **Research Prototype Only:** This system is an investigational computational prototype developed on a small single-center development cohort ($N=80$).
2. **Zero Clinical Validity:** The models are not clinically validated and must never be used for blood group determination, clinical diagnosis, transfusion decisions, or therapeutic interventions.
3. **Serological Verification Notice:** Development cohort labels were self-reported by student participants and have not been validated by clinical hemagglutination records.
4. **Statistical Boundary:** A non-significant result ($p = 0.2537$) demonstrates that optical fingerprint features on this dataset do not support predictive blood-group determination, but cannot prove that every conceivable biological association is absent across wider populations.
