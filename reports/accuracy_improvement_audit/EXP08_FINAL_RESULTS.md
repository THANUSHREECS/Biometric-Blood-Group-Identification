# EXP-08-DERMATOGLYPHIC-MINUTIAE: Final Scientific Results & Evaluation Report

**Document ID:** EXP08_FINAL_RESULTS  
**Date:** 2026-10-18 (Updated following Final Consistency Audit)  
**Experiment Name:** EXP-08-DERMATOGLYPHIC-MINUTIAE (Handcrafted Biometric Feature Benchmark)  
**Project:** Biometric Bloodgroup Detection Using Fingerprint Images  
**Hardware Platform:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI, Native $316 \times 354$ Pixels)  
**Authorized Cohort:** 80 Development Participants (240 Raw Impressions, 3 scans per participant)  
**Target Classes:** 4-Class ABO Positive (`A+`, `AB+`, `B+`, `O+`)  
**Class Distribution:** A+: 19 (23.75%), AB+: 7 (8.75%), B+: 25 (31.25%), O+: 29 (36.25%)  
**Evaluation Protocol:** 5-Fold Participant-Level StratifiedGroupKFold with In-Fold Imputation and Feature Scaling  

---

## 1. Executive Summary

EXP-08 investigated whether biologically interpretable, handcrafted dermatoglyphic features (primary pattern classification, standardized sub-core ridge density, crossing-number minutiae, and core-to-delta distances) extract predictive signal for 4-class ABO blood-group classification on optical fingerprint scans.

Feature extraction succeeded on **100% of the 240 scans** without interpolation, cropping, or scale distortion. All 3 impressions per participant were collapsed into **exactly one feature row**, producing an $(80, 28)$ dataset with 24 numeric features evaluated under strict participant-level 5-fold cross-validation.

### Primary Experimental Findings:

1. **Failure to Beat Majority Base Rate:**
   - The empirical majority class base rate in Dev-80 is **$36.25\%$** (29/80 participants are O+).
   - A Constant Majority Baseline predicting O+ for all participants achieves **$36.25\%$** (29/80).
   - Random Forest achieved an overall accuracy of **$35.00\%$** (28/80, 95% CI: $[25.5\%, 45.9\%]$), falling **below the majority class base rate**.
   - L2 Logistic Regression achieved **$27.50\%$** (22/80, 95% CI: $[18.9\%, 38.1\%]$).
2. **Nonsignificance on Primary Metric (Balanced Accuracy):**
   - The primary metric, **Balanced Accuracy**, was **$27.06\%$** for Random Forest and **$21.15\%$** for Logistic Regression (theoretical chance = $25.0\%$).
   - In a 1,000-run participant label permutation test, the empirical null mean for Random Forest was $25.19\%$, yielding a non-parametric $p$-value of **$p = 0.3250$**.
   - Logistic Regression yielded $p = 0.7810$.
3. **Complete Failure on Minority Class (AB+):**
   - Both Logistic Regression and Random Forest achieved **$0.0\%$ recall on AB+** ($0/7$ participants correctly identified).
4. **Scientific Conclusion:**
   - The experiment provides **no credible evidence of predictive signal beyond baseline** from handcrafted dermatoglyphic features on optical scanner images.
   - Handcrafted feature extraction confirms that commercial 500 DPI optical sensors capture high-precision anatomical measurements ($12.24 \pm 0.56\text{ ridges}/25\text{ mm}^2$), but these features do not possess discriminative capability for ABO blood groups on this cohort.

---

## 2. Experimental Benchmark Comparison

To prevent ambiguity, two distinct baseline references are evaluated alongside the trained models:
1. **Constant Majority-Class Baseline:** Predicts the cohort majority class (O+) unconditionally for all 80 participants.
2. **In-Fold Cross-Validation Dummy Baseline:** Fits `DummyClassifier(strategy="most_frequent")` strictly inside each training fold (64 participants). In Fold 2, because 8 of the 29 O+ samples were held out in validation, B+ was the most frequent training-fold class (23 B+ vs 21 O+), causing DummyClassifier to predict B+ for Fold 2 validation participants (yielding $2\text{ B+} + 21\text{ O+} = 23/80$ correct).

| Model Architecture | Out-of-Fold Accuracy (95% Wilson CI) | Primary Metric: Balanced Accuracy | Permutation $p$-value (Bal Acc) | Macro Precision | Macro Recall | Macro F1 | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Majority Baseline (All O+)** | 36.25% [26.57%, 47.19%] | 25.00% | — | 0.0906 | 0.2500 | 0.1330 | 0.0000 |
| **In-Fold Dummy Baseline (CV-Fold Majority)** | 28.75% [19.99%, 39.46%] | 20.10% | $p = 0.9760$ | 0.1133 | 0.2010 | 0.1373 | -0.1004 |
| **L2 Logistic Regression ($C=1.0$)** | 27.50% [18.92%, 38.14%] | 21.15% | $p = 0.7810$ | 0.2311 | 0.2115 | 0.2193 | -0.0030 |
| **Random Forest ($n=100, d=4$)** | **35.00% [25.45%, 45.92%]** | **27.06%** | **$p = 0.3250$** | **0.2533** | **0.2706** | **0.2603** | **+0.0571** |

---

## 3. Detailed Model Evaluations & Confusion Matrices

All confusion matrices and per-class metrics are reported in canonical class order: `["A+", "AB+", "B+", "O+"]`.

### A. Constant Majority Baseline (Predicts O+ for all 80 participants)
- **Correct Predictions:** 29 / 80 ($36.25\%$)
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

### B. In-Fold Dummy Baseline (`DummyClassifier` per fold)
- **Correct Predictions:** 23 / 80 ($28.75\%$)
- **Balanced Accuracy:** $20.10\%$ (Permutation $p = 0.9760$)
- **Cohen's Kappa:** $-0.1004$

```
Actual \ Predicted    A+    AB+    B+    O+    Total
A+                     0      0     5    14       19
AB+                    0      0     1     6        7
B+                     0      0     2    23       25
O+                     0      0     8    21       29
Total                  0      0    16    64       80
```

---

### C. L2-Regularized Logistic Regression
- **Correct Predictions:** 22 / 80 ($27.50\%$)
- **Balanced Accuracy:** $21.15\%$ (Permutation null mean: $25.26\%$, $p = 0.7810$)
- **Macro F1:** $0.2193$ ($p = 0.7130$)
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

### D. Random Forest Classifier
- **Correct Predictions:** 28 / 80 ($35.00\%$)
- **Balanced Accuracy:** $27.06\%$ (Permutation null mean: $25.19\%$, $p = 0.3250$)
- **Macro F1:** $0.2603$ ($p = 0.3690$)
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

## 4. Dataset Dimensions & Feature Inventory

The feature dataset [`dev80_handcrafted_features.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/features/dev80_handcrafted_features.csv) contains **80 rows and 28 columns**:
- **4 Metadata / Target Columns:**
  1. `participant_id` (unique participant identifier, zero duplicates)
  2. `blood_group` (ground truth string label)
  3. `target_label` (ground truth integer target: 0, 1, 2, 3)
  4. `majority_pattern` (categorical string: "WHORL", "LEFT_LOOP", etc.)
- **24 Numeric Feature Columns (listed in [`feature_names.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/features/feature_names.json)):**
  - Ridge Density (2): `ridge_density_25mm2_mean`, `ridge_density_25mm2_std`
  - Ridge Endings (2): `ridge_endings_count_mean`, `ridge_endings_count_std`
  - Ridge Bifurcations (2): `ridge_bifurcations_count_mean`, `ridge_bifurcations_count_std`
  - Total Minutiae (2): `total_minutiae_count_mean`, `total_minutiae_count_std`
  - Minutiae Density (2): `minutiae_density_cm2_mean`, `minutiae_density_cm2_std`
  - Ridge Clarity / Coherence (2): `ridge_clarity_score_mean`, `ridge_clarity_score_std`
  - Foreground Contact Area (2): `foreground_area_mm2_mean`, `foreground_area_mm2_std`
  - Core-to-Delta Distance (2): `core_delta_distance_mm_mean`, `core_delta_distance_mm_std`
  - Landmark Detection Rates (3): `core_delta_valid_scans_count`, `core_detection_rate`, `delta_detection_rate`
  - Pattern Metrics (5): `pattern_concordance`, `pattern_is_arch`, `pattern_is_left_loop`, `pattern_is_right_loop`, `pattern_is_whorl`

- **Missingness:** Exactly 1 missing value in the entire matrix (`core_delta_distance_mm_mean` for 1 participant who lacked detected deltas across impressions). Imputed leak-free strictly within each training fold via median imputation.

---

## 5. Review of Pattern Classification Sensitivity

In the feature matrix, 74 of 80 participants (92.5%) were classified as Whorl. An audit of the Poincaré index implementation revealed the algorithmic cause:
- **Block-Level Noise:** The Poincaré index is evaluated over $16 \times 16$ pixel blocks on the raw gradient structure tensor orientation field. Local turbulence, sweat-pore breaks, and contact margin edges frequently trigger secondary singular points ($+1.0$ or $-1.0$).
- **Classification Bias:** Under standard Henry rules, a print with $\ge 2$ cores OR $\ge 2$ deltas is classified as WHORL. When local noise creates even one false-positive core or delta on a loop, the rule immediately classifies the print as WHORL.
- **Reference Standard Notice:** This automated pattern classification algorithm has **not been validated or calibrated against a manually ground-truthed dermatoglyphic dataset** (such as NIST SD4). We do **not** claim forensic validity for the pattern classification sub-module.

---

## 6. Statistical Interpretation & Limits of Inference

1. **Permutation Test Methodology:**
   - 1,000 label permutations were evaluated against out-of-fold predictions.
   - For Random Forest, the empirical null mean was $25.19\%$, and the observed $27.06\%$ balanced accuracy yielded $p = 0.3250$ (32.5% of random permutations produced an equal or higher score).
2. **Limits of Sample Size ($N=80$):**
   - Each participant represents $1.25\%$ of overall accuracy.
   - For AB+ ($N=7$), each participant represents $14.29\%$ of class recall. With zero AB+ samples identified, the class recall is $0.0\%$, constraining balanced accuracy.
3. **Scientific Non-Claims:**
   - **Absence of Proof is Not Proof of Absence:** The lack of statistical significance ($p = 0.3250$) demonstrates that commercial 500 DPI optical scans do not provide a viable blood-group diagnostic tool on this cohort, but cannot prove that biological associations do not exist in large serological populations.
   - **Clinical Non-Validity:** Fingerprints cannot be used for clinical blood-group determination.
