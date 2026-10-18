# EXP-08: Final Results Consistency & Reconciliation Audit

**Document ID:** EXP08_FINAL_CONSISTENCY_AUDIT  
**Date:** 2026-10-18  
**Experiment:** EXP-08-DERMATOGLYPHIC-MINUTIAE  
**Scope:** Forensic cross-check of participant-level ground truth labels, out-of-fold predictions, feature matrix structure, model evaluation metrics, and statistical null distributions.  

---

## 1. Executive Summary & Discrepancy Reconciliation

A thorough consistency audit was conducted across all artifacts produced by EXP-08. No models were retrained, no ground truth labels were altered, and no quarantined data cohorts were accessed.

### Summary of Audit Findings & Resolutions:

| Audit Item | Reported in Initial Summary | Verified Empirical Reality | Root Cause & Resolution |
| :--- | :---: | :---: | :--- |
| **Majority Baseline Correct Counts** | 23 / 80 ($28.75\%$) | **29 / 80 ($36.25\%$)** for Constant Baseline; **23 / 80 ($28.75\%$)** for In-Fold CV Dummy | In `train.py`, `DummyClassifier` was evaluated within 5-fold CV. In Fold 2, B+ was the training majority ($23 > 21$), so it predicted B+ for all Fold 2 samples (getting 2 correct), while predicting O+ in other folds (getting 21 correct). Both baselines are now explicitly reported and distinguished. |
| **Cohort Class Distribution** | A+: 19, AB+: 7, B+: 25, O+: 29 | **A+: 19, AB+: 7, B+: 25, O+: 29** | **100% verified**. Matches authoritative `reports/phase3_participant_benchmark/participant_split.csv` exactly. |
| **Dataset Dimensions** | 80x28 reported vs 80x24 numeric | **80 rows, 28 columns total (24 numeric features)** | Reconciled: 28 total columns = 4 metadata/label columns (`participant_id`, `blood_group`, `target_label`, `majority_pattern`) + 24 numeric features in `feature_names.json`. |
| **Model Metrics (LogReg & RF)** | Acc: 27.50% & 35.00%; Bal Acc: 21.15% & 27.06% | **Exact match to 4th decimal place** | Recomputed from `exp08_participant_oof_predictions.csv`. All metrics, CIs, and confusion matrices verified identical. |
| **Pattern Classification Bias** | 74 / 80 Whorl (92.5%) | **Poincaré block sensitivity on optical scans** | Documented: Optical noise triggers secondary singular points; Henry rule ($\ge 2$ cores or deltas $\implies$ Whorl) creates structural bias toward Whorl. Validated that forensic accuracy is not claimed. |
| **Permutation Null Distribution** | 1,000 permutations ($p=0.3250$) | **$p = 0.3250$ verified** | Non-parametric label permutation test verified against fixed OOF predictions. Null mean was $25.19\%$. |

---

## 2. Step 1: Verification of Majority-Class Baseline

### A. Authoritative Ground Truth Verification
The class distribution was verified directly from `reports/phase3_participant_benchmark/participant_split.csv` restricted to `split.isin(["TRAIN", "VALIDATION"])`:
- Total participants: **80**
- Total scans: **240** (exactly 3 per participant)
- **Class Counts:**
  - `A+`: **19** ($23.75\%$)
  - `AB+`: **7** ($8.75\%$)
  - `B+`: **25** ($31.25\%$)
  - `O+`: **29** ($36.25\%$)

### B. Investigation of 23/80 vs 29/80 Correct Predictions
Why did the initial model evaluation report 23 correct predictions for the majority baseline while the O+ support is 29?
- **Root Cause:** In `train.py`, the majority baseline was implemented using scikit-learn's `DummyClassifier(strategy="most_frequent")` evaluated inside the 5-fold cross-validation loop.
- **Fold-by-Fold Majority Shift:**
  - In Fold 2, the validation fold happened to draw 8 O+ participants and only 2 B+ participants.
  - As a result, the training fold for Fold 2 (the remaining 64 participants) contained:
    $$\text{O+} = 29 - 8 = 21, \quad \text{B+} = 25 - 2 = 23, \quad \text{A+} = 19 - 5 = 14, \quad \text{AB+} = 7 - 1 = 6$$
  - Because $23\text{ (B+)} > 21\text{ (O+)}$, `DummyClassifier` identified **B+ as the training majority** in Fold 2, predicting B+ for all 16 validation participants.
  - Since only 2 validation participants in Fold 2 were actually B+, it obtained 2 correct predictions in Fold 2.
  - In Folds 1, 3, 4, and 5, the training set majority was O+, obtaining 6, 6, 7, and 2 correct O+ predictions, respectively ($21$ total).
  - Sum of correct predictions: $2\text{ (B+ in Fold 2)} + 21\text{ (O+ in Folds 1, 3, 4, 5)} = \mathbf{23/80}$ ($28.75\%$).

### C. Recalculation of Both Baseline References:

#### 1. Constant Majority Baseline (Unconditionally Predicts O+ for all 80 participants):
- Correct Predictions: **29 / 80** ($36.25\%$)
- 95% Wilson Confidence Interval: $[26.57\%, 47.19\%]$
- Balanced Accuracy: $\frac{0.0 + 0.0 + 0.0 + 1.0}{4} = \mathbf{25.00\%}$
- Macro Precision: $\frac{0.0 + 0.0 + 0.0 + 29/80}{4} = \frac{0.3625}{4} = \mathbf{0.0906}$
- Macro Recall: $\frac{0.0 + 0.0 + 0.0 + 1.0}{4} = \mathbf{0.2500}$
- Macro F1: $\frac{0.0 + 0.0 + 0.0 + 0.5321}{4} = \mathbf{0.1330}$
- Cohen's Kappa: $\mathbf{0.0000}$
- **Canonical Confusion Matrix:**
  ```
  Actual \ Predicted    A+    AB+    B+    O+    Total
  A+                     0      0     0    19       19
  AB+                    0      0     0     7        7
  B+                     0      0     0    25       25
  O+                     0      0     0    29       29
  Total                  0      0     0    80       80
  ```

#### 2. In-Fold Cross-Validation Dummy Baseline (`DummyClassifier` per fold):
- Correct Predictions: **23 / 80** ($28.75\%$)
- 95% Wilson Confidence Interval: $[19.99\%, 39.46\%]$
- Balanced Accuracy: $\frac{0.0 + 0.0 + 0.08 + 0.7241}{4} = \mathbf{20.10\%}$
- Macro Precision: $\mathbf{0.1133}$
- Macro Recall: $\mathbf{0.2010}$
- Macro F1: $\mathbf{0.1373}$
- Cohen's Kappa: $\mathbf{-0.1004}$
- **Canonical Confusion Matrix:**
  ```
  Actual \ Predicted    A+    AB+    B+    O+    Total
  A+                     0      0     5    14       19
  AB+                    0      0     1     6        7
  B+                     0      0     2    23       25
  O+                     0      0     8    21       29
  Total                  0      0    16    64       80
  ```

---

## 3. Step 2: Verification of the Feature Matrix

An audit of [`dev80_handcrafted_features.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/features/dev80_handcrafted_features.csv) and [`feature_names.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/features/feature_names.json) confirms:
- **Total Rows:** Exactly 80 rows.
- **Participant ID Column:** `participant_id` (zero duplicate values; 80 unique IDs).
- **Label / Target Columns:** `blood_group` (string) and `target_label` (integer).
- **Categorical Feature Column:** `majority_pattern` (string).
- **Total Columns in CSV:** **28 columns**.
- **Explanation of 80x28 vs 80x24:**
  $$28\text{ total columns} - 4\text{ non-feature/metadata columns} = \mathbf{24\text{ numeric features}}$$
  The 24 numeric feature names in `feature_names.json` match the 24 input features fed into the scikit-learn models.
- **Missing Values:** Exactly 1 missing value in the entire $(80, 28)$ dataset:
  - `core_delta_distance_mm_mean`: 1 missing value (participant without detected deltas across all 3 impressions).
  - All other 27 columns: **0 missing values**.
  - Missingness was handled strictly inside each training fold using `SimpleImputer(strategy="median")`.

---

## 4. Step 3: Verification of Model Metrics

Metrics were recomputed independently from the 80 participant-level out-of-fold predictions in [`exp08_participant_oof_predictions.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/results/exp08_participant_oof_predictions.csv):

| Metric | L2 Logistic Regression | Random Forest Classifier | Verification Check |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 27.50% (22/80) | 35.00% (28/80) | **EXACT MATCH** |
| **95% Wilson CI** | [18.92%, 38.14%] | [25.45%, 45.92%] | **EXACT MATCH** |
| **Balanced Accuracy** | 21.15% | 27.06% | **EXACT MATCH** |
| **Macro Precision** | 0.2311 | 0.2533 | **EXACT MATCH** |
| **Macro Recall** | 0.2115 | 0.2706 | **EXACT MATCH** |
| **Macro F1** | 0.2193 | 0.2603 | **EXACT MATCH** |
| **Cohen's Kappa** | -0.0030 | +0.0571 | **EXACT MATCH** |
| **A+ Recall (Support=19)** | 15.79% (3/19) | 21.05% (4/19) | **EXACT MATCH** |
| **AB+ Recall (Support=7)** | 0.00% (0/7) | 0.00% (0/7) | **EXACT MATCH** |
| **B+ Recall (Support=25)** | 24.00% (6/25) | 32.00% (8/25) | **EXACT MATCH** |
| **O+ Recall (Support=29)** | 44.83% (13/29) | 55.17% (16/29) | **EXACT MATCH** |

Both models use the identical canonical class order `["A+", "AB+", "B+", "O+"]` across ground truth labels, predicted labels, and confusion matrices.

---

## 5. Step 4: Review of Pattern Classification Sensitivity

An audit of the pattern classification results in `dev80_handcrafted_features.csv` (74 Whorls, 3 Left Loops, 2 Right Loops, 1 Arch) was conducted:
1. **Algorithmic Sensitivity:**
   - Orientation field angles are computed in $16 \times 16$ pixel blocks.
   - The Poincaré Index integrates wrapped angle differences along block boundaries.
   - On optical sensor images, sweat pores along ridge crests and platen contact margin turbulence cause local block orientations to fluctuate, producing secondary singular points.
   - In a detailed audit of 30 sample scans, core counts ranged from 1 to 9 (median: 3) and delta counts ranged from 0 to 9 (median: 2).
2. **Classification Rule Bias:**
   - Under standard Henry classification rules, any fingerprint with $\ge 2$ cores OR $\ge 2$ deltas is classified as WHORL.
   - Because raw optical noise frequently produces at least one false-positive core or delta, prints that are anatomically loops frequently trigger the WHORL classification condition.
3. **Reference Standard Notice:**
   - The automated pattern classification implementation has **not been validated or calibrated against an external reference standard** (such as manually annotated NIST SD4 cards).
   - Forensic validity for the automated pattern classifier is explicitly **not claimed**.

---

## 6. Step 5: Review of Statistical Interpretation & Limits of Inference

1. **Permutation Test Verification:**
   - Tested under 1,000 participant label shuffles against the fixed out-of-fold predictions.
   - Random Forest Balanced Accuracy: observed $27.06\%$, null mean $25.19\%$, $p = 0.3250$.
   - The test demonstrates that $32.5\%$ of random label permutations produce equal or higher balanced accuracy than the trained Random Forest model.
2. **Limits of Small-Sample Inference ($N=80$):**
   - Each participant represents $1.25\%$ of total accuracy.
   - For AB+ ($N=7$), each participant represents $14.29\%$ of class recall.
   - With 0 AB+ participants identified, AB+ recall is $0.0\%$, creating an unweighted penalty on balanced accuracy.
3. **Scientific Non-Claims:**
   - **Nonsignificance $\neq$ Proof of Absence:** A nonsignificant result ($p = 0.3250$) indicates that on this dataset with commercial 500 DPI optical scans, handcrafted dermatoglyphic features provide no detectable blood-group predictive signal. It cannot prove that biological associations do not exist in large-scale serological cohorts.
   - **Clinical Non-Validity:** Commercial fingerprint images cannot be used for clinical blood-group determination.

---

## 7. Deliverables & Reconciliation Log

1. **Audit Document:** [`EXP08_FINAL_CONSISTENCY_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FINAL_CONSISTENCY_AUDIT.md) (This document).
2. **Reconciled Results Report:** [`EXP08_FINAL_RESULTS.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md) (Updated to include both baseline definitions and dataset dimensional breakdowns).
3. **Reconciled Machine-Readable Metrics:** [`exp08_evaluation_metrics.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/results/exp08_evaluation_metrics.json) (Updated with `constant_majority_baseline` and `infold_dummy_baseline`).
4. **Source Artifact Integrity:**
   - Original labels in `participant_split.csv` preserved.
   - Out-of-fold predictions in `exp08_participant_oof_predictions.csv` preserved.
   - Extracted features in `dev80_handcrafted_features.csv` preserved.
   - Locked test cohort and hospital cohort were 100% untouched.
