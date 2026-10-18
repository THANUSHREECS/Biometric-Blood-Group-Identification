# Proposed Scientific Corrections: Phase 3 Final Evaluation Report

**Target Document:** `reports/phase3_final_evaluation/FINAL_TEST_REPORT.md`  
**Document Status:** Reference Erratum and Methodological Corrections Document  
**Constraint Compliance:** `FINAL_TEST_REPORT.md` remains strictly unmodified in accordance with repository audit immutability rules. All corrections are documented here.  
**Date:** October 17, 2026  

---

## 1. Audit of Development Validation Claims

### 1.1 Origin of "0.42 Macro F1 during Validation"
In `reports/phase3_final_evaluation/FINAL_TEST_REPORT.md` (Section 6.1, line 45), the report states:
> *"The extensive development improvements (reaching 53% accuracy and 0.42 Macro F1 during validation) did not translate to the locked test set."*

- **Documented Source:** `reports/phase3_fusion_metric_learning/EXPERIMENT6_REPORT.md:5`:
  ```markdown
  | Method | Val Macro F1 | Val Acc | Train F1 | Gap | O+ Share |
  |---|---|---|---|---|---|
  | E2 | 0.4240 | 0.5375 | 0.2744 | -0.1496 | 0.3125 |
  ```
- **Finding:** The figure `0.42` directly reflects the **unweighted arithmetic mean of validation fold Macro F1 scores (0.4240)** achieved by candidate E2 during Phase 3 Experiment 6 (Three-Scan Feature Fusion + Supervised Contrastive Metric Learning). Similarly, `53%` reflects the mean validation accuracy of `0.5375` (53.75%).

### 1.2 Documented E2 Out-of-Fold (OOF) Macro F1
- **Documented Source File:** `reports/final/eval_101_raw_results.json` (`metrics_dev_oof`)  
- **Also Documented in:** `reports/final/E2_101_PARTICIPANT_EVALUATION.md` (Section 1, line 32)
- **Documented Metrics:**
  - **OOF Accuracy:** `0.5375` (43 / 80 correct)
  - **OOF Balanced Accuracy:** `0.4715` (`0.4714583333333333`)
  - **OOF Macro F1:** **`0.4772`** (`0.47723137718444353`)
  - **OOF Macro Precision:** `0.5284` (`0.5283838383838384`)
  - **OOF Macro Recall:** `0.4715` (`0.4714583333333333`)
- **Methodological Distinction:**
  The `0.4240` reported in `EXPERIMENT6_REPORT.md` is the mean across fold-level classification reports during development iterations. When predictions from all 5 held-out folds are concatenated to form the global Out-of-Fold (OOF) confusion matrix across all 80 development participants, the true global E2 OOF Macro F1 is **`0.4772`**.

---

## 2. Itemized Proposed Corrections to `FINAL_TEST_REPORT.md`

The following proposed textual and interpretation corrections should be applied to future revisions of `FINAL_TEST_REPORT.md`:

### Correction 1: Baseline Definition (Majority Class vs Random Guess)
- **Target Section:** Section 6.4, line 54
- **Existing Text:**
  > *"The 38% accuracy is essentially indistinguishable from a naive random guess distributed around the class priors."*
- **Proposed Correction:**
  > *"The 38.10% accuracy (8/21) equals the majority-class baseline accuracy (8/21); not the same behavior. E2's 38.10% accuracy came from 3/5 A+ and 5/8 O+ correct with 11 A+ predictions (as detailed in Section 4's per-class performance table), rather than predicting O+ for all participants. While it does not exceed the numerical accuracy of a trivial majority baseline (8/21 = 38.10%), it should be compared to the majority-class baseline rather than a uniform 4-class random guess (25.00%)."*
- **Rationale:** In a 4-class cohort of 21 participants (8 O+, 6 B+, 5 A+, 2 AB+), a constant majority classifier achieves 8/21 = 38.10%. E2 matched this numerical rate by predicting 11 A+, 8 O+, and 2 B+, correctly identifying 3/5 A+ and 5/8 O+; it was not produced by predicting O+ for everyone. Calling 38% a "naive random guess distributed around class priors" mischaracterizes both the majority baseline and the model's actual predictions.

---

### Correction 2: Strength of Empirical Evidence vs Sample Size
- **Target Section:** Section 6.3, line 51
- **Existing Text:**
  > *"This experiment provides extremely strong empirical evidence against the hypothesis that fingerprint ridge patterns contain a biologically deterministic, generalizable macroscopic signature for the ABO blood group that can be resolved by a standard MFS100 scanner."*
- **Proposed Correction:**
  > *"This experiment provides no evidence of generalization; the locked test cohort is too small (N=21 participants) to statistically establish or rule out the existence of a biological signature (the 95% bootstrap confidence interval spans [19.05%, 57.14%])."*
- **Rationale:** A sample of 21 participants yields a $95\%$ bootstrap confidence interval spanning $19\%$ to $57\%$. Because a true accuracy of $50\%$ falls well within this interval, the small test set lacks statistical power to provide "extremely strong evidence" against biological plausibility; it simply fails to demonstrate generalization.

---

### Correction 3: Selection Optimism vs "Mathematical Artifact"
- **Target Section:** Section 6.3, line 51
- **Existing Text:**
  > *"Any signal detected during development was likely a mathematical artifact of the specific cross-validation splits, rather than a universal biological biomarker."*
- **Proposed Correction:**
  > *"The substantial drop between cross-validation performance (53.75% accuracy, 0.4772 OOF macro F1) and locked test performance (38.10% accuracy, 0.2500 macro F1) is consistent with selection optimism arising from iterative architectural choices and checkpoint selection rules across small sample sizes (N=80)."*
- **Rationale:** The performance drop between CV and held-out test is the classic hallmark of selection optimism (optimism bias from repeatedly selecting champion models on small validation partitions), rather than an unexplained "mathematical artifact."

---

### Correction 4: Test Cohort Terminology ("Unseen" vs "Held-Out, Previously Scored")
- **Target Sections:** Section 5 (line 34), Section 6.1 (line 45), Section 6.2 (line 48)
- **Existing Text:**
  - Line 34: *"Both models were evaluated on the exact same 21 unseen participants."*
  - Line 45: *"failed to generalize to strictly unseen participants."*
  - Line 48: *"subtle, purported blood-group textures of the unseen B/AB individuals"*
- **Proposed Correction:**
  - Replace instances of *"unseen participants"* with *"held-out, previously scored participants"*.
- **Rationale:** As documented in `reports/project_report/locked_cohort_usage.csv`, the 21 participants in the locked test cohort were held out from gradient updates, but had already been evaluated and scored during the Day 27 zero-shot benchmark (October 7) and the Phase 3 architecture benchmark (October 7). Describing them as "strictly unseen" ignores their prior scoring history in the repository.

---

### Correction 5: Bootstrap Sample Proportions ("Many" vs "Some")
- **Target Section:** Section 3 ("UNCERTAINTY"), line 23
- **Existing Text:**
  > *"Furthermore, AB+ had only 2 representatives, meaning many bootstrap samples contained zero AB+ cases, making Macro F1 highly unstable and statistically precarious."*
- **Proposed Correction:**
  > *"Furthermore, AB+ had only 2 representatives, meaning some bootstrap samples contained zero AB+ cases (the exact probability that a bootstrap resample of size 21 drawn with replacement contains zero AB+ cases is $(19/21)^{21} \approx 0.12224 \implies 12.22\%$), contributing to the instability of Macro F1."*
- **Rationale:** With 2 AB+ participants out of 21, the exact binomial probability of drawing 0 AB+ cases in a resample of size $N=21$ is $(1 - 2/21)^{21} = (19/21)^{21} = 0.1222419954... \approx 12.22\%$. In mathematical terms, ~12.2% is properly described as "some" bootstrap samples, rather than "many."

---

## 3. Summary of Discrepancies and Clarifications

| Report Item | Stated in `FINAL_TEST_REPORT.md` | Conflicting artifact value | Audit Source File & Line |
|---|---|---|---|
| **Validation Macro F1** | `0.42` | `0.4240` (mean fold F1) / `0.4772` (global OOF F1) | `EXPERIMENT6_REPORT.md:5` / `eval_101_raw_results.json` (`metrics_dev_oof`) |
| **Baseline Accuracy** | 38% "naive random guess" | 38.10% (8/21) majority-class baseline | `FINAL_TEST_REPORT.md:31` |
| **Day 27 Baseline Acc** | `38.09%` (0.3125 Macro F1) | `33.33%` (7/21) (0.1875 Macro F1) | `final_zero_shot_participant_predictions.csv` |
| **Evidence Strength** | "Extremely strong empirical evidence" | "No evidence of generalization; test too small" | Sample size analysis ($N=21$, CI $[19\%, 57\%]$) |
| **CV-Test Drop Mechanism** | "Mathematical artifact" | "Consistent with selection optimism" | Methodological audit |
| **Test Cohort Status** | "Strictly unseen" | "Held-out, previously scored" | `locked_cohort_usage.csv` |
| **Zero-AB+ Bootstraps** | "Many" | "Some" ($(19/21)^{21} \approx 12.22\%$) | Exact probability: $(19/21)^{21} = 0.12224$ |

---

## 4. Inventory Row Reconciliation ($127 = 124 + 3$)

### 4.1 Decomposition of Evaluated Metric Rows
The full results inventory (`reports/project_report/results_inventory.csv`) contains **145 rows**, partitioned strictly into two mutually exclusive files:
1. `reports/project_report/model_results.csv`: **124 rows**
2. `reports/project_report/audit_diagnostics.csv`: **21 rows**
   - Identity: $145 = 124 + 21$.

Across the entire inventory, exactly **127 rows** contain numeric accuracy and classification performance metrics ($127 = 124 + 3$):
- **124 rows** in `model_results.csv` all report numeric performance metrics (accuracy, macro F1, etc.).
- **3 rows** in `audit_diagnostics.csv` report numeric performance metrics from audit permutation tests and selection null evaluations.
- The remaining **18 rows** in `audit_diagnostics.csv` represent non-performance diagnostic tests (2 perceptual hash reference comparisons and 16 graph percolation / cluster component analyses) where classification accuracy is `"not applicable"`:
  - **Perceptual Hash Comparisons (2 rows):**
    1. `AUDIT-EXP1-HASH-TRAIN-REF`
    2. `AUDIT-EXP1-HASH-TRAINVAL-REF`
  - **Percolation / Cluster Analysis on EfficientNet-B0 Embeddings (11 rows):**
    3. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.985`
    4. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.98`
    5. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.975`
    6. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.97`
    7. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.965`
    8. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.96`
    9. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.955`
    10. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.95`
    11. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.92`
    12. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.9`
    13. `AUDIT-EXP1-PERCOLATION-EFFB0-COS-0.85`
  - **Percolation / Cluster Analysis on DenseNet-121 Embeddings (5 rows):**
    14. `AUDIT-EXP1-PERCOLATION-DENSE121-COS-0.98`
    15. `AUDIT-EXP1-PERCOLATION-DENSE121-COS-0.95`
    16. `AUDIT-EXP1-PERCOLATION-DENSE121-COS-0.92`
    17. `AUDIT-EXP1-PERCOLATION-DENSE121-COS-0.9`
    18. `AUDIT-EXP1-PERCOLATION-DENSE121-COS-0.85`

### 4.2 Listing of the 3 Numeric Diagnostic Rows in `audit_diagnostics.csv`
The 3 numeric diagnostic rows in `audit_diagnostics.csv` are identified as follows:

1. **`AUDIT-EXP2-SIGNAL-DEV80`**
   - **Model Architecture:** `DenseNet121_Fusion_LinearHead`
   - **Dataset / Split:** MFS100 Development Cohort (Dev 80 participants)
   - **Accuracy:** `0.5375` (43 / 80 correct)
   - **Macro F1:** `0.5487`
   - **Source:** `experiments/leakage_audit/exp2_signal_tests_results.json` (`dev_80_participants.observed_metrics`)
   - **Description:** Observed cross-validation baseline metrics re-calculated during the leakage audit permutation signal test.

2. **`AUDIT-EXP2-SIGNAL-TEST21`**
   - **Model Architecture:** `DenseNet121_Fusion_LinearHead`
   - **Dataset / Split:** MFS100 Locked Test Cohort (21 participants)
   - **Accuracy:** `0.3333` (7 / 21 correct)
   - **Macro F1:** `0.2083`
   - **Source:** `experiments/leakage_audit/exp2_signal_tests_results.json` (`primary_test_21_participants.observed_metrics`)
   - **Description:** Observed locked test performance of EXP-06C Config A re-analyzed under the leakage audit permutation test.

3. **`AUDIT-PHASE2-SELECTION-INCLUSIVE-NULL`**
   - **Model Architecture:** `DenseNet121_Fusion_LinearHead (Permuted Labels)`
   - **Dataset / Split:** MFS100 Development Cohort (80 participants across 100 null iterations)
   - **Accuracy:** `0.4677`
   - **Macro F1:** `0.4277`
   - **Source:** `experiments/leakage_audit/selection_inclusive_null_summary.json` (`summary_metrics.mean_val_acc`, `mean_val_macro_f1`)
   - **Description:** Empirical null distribution mean achieved when applying the identical model selection rule to completely permuted labels.

### 4.3 Separation of Audit Experiments in `model_results.csv`
Note that `model_results.csv` (124 rows) includes both original development/evaluation models and **48 classifier models evaluated during the leakage audit**. 

These 48 audit rows are characterized by the following ID prefixes and counts:
- **`AUDIT-EXP2-METADATA-*` (6 rows):** Metadata baseline models (`AUDIT-EXP2-METADATA-01` through `06`).
- **`AUDIT-PHASE2-RULE-*` (12 rows):** Phase 2 model selection rule evaluations across Rule I, Rule A, and Rule B (evaluating seeds 42, 43, 44, and multi-seed means).
- **`AUDIT-PHASE3-ARM*` (8 rows):** Phase 3 purged cross-validation experiments evaluating Arm 1 (fixed epochs) and Arm 2 (inner validation) across purge fractions 0%, 10%, 25%, and 50%.
- **`AUDIT-EXP3-TRIVIAL-*` (4 rows):** Baseline property classifiers using Logistic Regression and Random Forest under true vs. shuffled labels.
- **`AUDIT-EXP3-ABL-*` (16 rows):** Image feature group ablation models across Groups I, II, III, and IV under Logistic Regression and Random Forest with true vs. shuffled labels.
- **`AUDIT-EXP3-EFFNET-*` (2 rows):** EfficientNet-B0 evaluated with and without concatenated trivial image features (`AUDIT-EXP3-EFFNET-WITHOUT-TRIVIAL` and `AUDIT-EXP3-EFFNET-WITH-TRIVIAL`).

These audit models are cleanly separated from standard development experiments by the **`group`** column:
- `Leakage Audit Diagnostics & Experiments` (48 rows)
- `EXP-03 to EXP-06D Hyperparameter Ablations` (23 rows)
- `MFS100 Individual Models` (18 rows)
- `E2 Variants & Ensembles` (12 rows)
- `Phase 3 Architecture Benchmark (Locked Cohort)` (7 rows)
- `Original-Dataset Models` (7 rows)
- `Original-Dataset Ensembles` (3 rows)
- `Day 27 Source-Domain Benchmarks` (2 rows)
- `Locked Test Final Evaluations` (2 rows)
- `Day 27 Locked Cohort Zero-Shot` (2 rows)


