# Independent EXP-06D Post-Experiment Verification Audit

**Project:** Biometric Blood Group Detection / HemoPrint AI  
**Experiment Audited:** EXP-06D — Controlled Preprocessing Ablation (Enhanced C5 vs. Raw Platen C5)  
**Audit Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Scope:** Prediction Integrity, Metrics Recomputation, Statistical Verification, Scientific Claim Validity, Checkpoint & Cohort Safety  
**Audited Artifacts:**
- `reports/accuracy_improvement/exp06d/final_report.md`
- `reports/accuracy_improvement/exp06d/metrics_comparison.csv`
- `reports/accuracy_improvement/exp06d/statistical_analysis_results.json`
- `reports/accuracy_improvement/exp06d/experiment_config.json`
- `reports/accuracy_improvement/exp06d/config_a_oof_predictions.csv`
- `reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv`

---

## 1. Executive Summary & Verification Matrix

All independent verification tasks passed with 100% reproducibility and bit-for-bit numerical consistency. No data leakage, cohort contamination, checkpoint corruption, or unhedged scientific claims were identified.

### Verification Register Table

| Requirement | Recomputed Result | Reported Result | Status | Evidence |
| :--- | :--- | :--- | :---: | :--- |
| **80 Unique Dev Participants** | Exactly 80 unique IDs in both Config A and Config B | 80 participants | **PASS** | `len(unique(participant_id)) == 80` in both OOF CSVs |
| **Participant ID Matching** | 100% identical participant IDs in identical row order | Identical IDs across configs | **PASS** | `(df_a['participant_id'] == df_b['participant_id']).all() == True` |
| **Ground-Truth Label Consistency** | 100% identical true blood group labels for all 80 rows | Identical ground-truth | **PASS** | `(df_a['true_label'] == df_b['true_label']).all() == True` |
| **Single OOF Prediction per Participant** | Exactly 1 prediction per participant per config | Exactly 1 prediction per participant | **PASS** | `df.groupby('participant_id').size().max() == 1` |
| **Probability Vector Properties** | All probabilities finite, $\ge 0.0$, and sum to $1.0 \pm 10^{-6}$ | Normalized probability vectors | **PASS** | Finite check pass, non-negative pass, max absolute sum diff $< 10^{-6}$ |
| **Canonical Class Order** | `[A+, AB+, B+, O+]` (indices 0, 1, 2, 3) | `[A+, AB+, B+, O+]` | **PASS** | Column alignment: `prob_A+`, `prob_AB+`, `prob_B+`, `prob_O+` |
| **Argmax Consistency** | `predicted_label == argmax(probs)` for 100% of participants | Predicted labels match argmax | **PASS** | Recomputed argmax matches `predicted_label` for all 80 rows in both CSVs |
| **Config A OOF Accuracy** | **46.25%** (37 / 80) | 46.25% (37 / 80) | **PASS** | Exact match with final report and metrics comparison CSV |
| **Config A Balanced Accuracy** | **45.86%** | 45.86% | **PASS** | Recomputed macro recall = 45.86% |
| **Config A Macro Precision** | **0.4507** | 0.4507 | **PASS** | Exact match ($0.450731$) |
| **Config A Macro Recall** | **0.4586** | 0.4586 | **PASS** | Exact match ($0.458627$) |
| **Config A Macro F1 (Pooled OOF)** | **0.4456** | 0.4456 | **PASS** | Exact match ($0.445582$) |
| **Config A Fold F1 Mean (Std)** | **0.4317 $\pm$ 0.1044** | 0.4317 ($\pm 0.1044$) | **PASS** | Fold F1s: `[0.3919, 0.4697, 0.2951, 0.6088, 0.3929]` |
| **Config B OOF Accuracy** | **52.50%** (42 / 80) | 52.50% (42 / 80) | **PASS** | Exact match with final report and metrics comparison CSV |
| **Config B Balanced Accuracy** | **50.07%** | 50.07% | **PASS** | Recomputed macro recall = 50.07% |
| **Config B Macro Precision** | **0.5220** | 0.5220 | **PASS** | Exact match ($0.521996$) |
| **Config B Macro Recall** | **0.5007** | 0.5007 | **PASS** | Exact match ($0.500735$) |
| **Config B Macro F1 (Pooled OOF)** | **0.5078** | 0.5078 | **PASS** | Exact match ($0.507759$) |
| **Config B Fold F1 Mean (Std)** | **0.4771 $\pm$ 0.1099** | 0.4771 ($\pm 0.1099$) | **PASS** | Fold F1s: `[0.4911, 0.5764, 0.3321, 0.6131, 0.3730]` |
| **Paired Contingency Counts** | $n_{11}=28, n_{10}=9, n_{01}=14, n_{00}=29$ | $n_{11}=28, n_{10}=9, n_{01}=14, n_{00}=29$ | **PASS** | Discordant total: 23 ($9 + 14$). Concordant: 57 ($28 + 29$) |
| **Exact 2-Sided Binomial p-value** | **$p = 0.404873$** (rounded: 0.4049) | $p = 0.4049$ | **PASS** | `scipy.stats.binomtest(k=14, n=23, p=0.5, alternative='two-sided')` |
| **Paired Bootstrap 95% CI** | **`[-0.0669, 0.1912]`** (mean $+0.0618$) | `[-0.0669, 0.1912]` | **PASS** | 2000 resamples, seed=42, participant-level resampling |
| **Statistical Equivalence vs Non-Sig** | Non-significance formally distinguished from equivalence | Documented failure to reject $H_0$; no equivalence claim | **PASS** | Section 1, 6 of final report explicitly distinguish failure to reject from equivalence |
| **Scientific Claims Check** | Zero unsupported claims of statistical significance, proof of feature destruction, or clinical validity | Fully hedged as empirical development hypothesis | **PASS** | Mandatory disclaimers present; investigational scope maintained |
| **Checkpoint Hash Ledger (66 models)** | 66 of 66 hashes match `scratch/exp06d_pre_preparation_hashes.json` | 66 protected checkpoints | **PASS** | 0 corrupted, 0 missing, 0 altered historical weights |
| **Cohort Quarantine Integrity** | 0 access events to locked test (21) or hospital (26) cohorts | Quarantined | **PASS** | Zero test/hospital IDs present in OOF data; quarantine verified |
| **Execution Guard Status** | `TRAINING_AUTHORIZED = False` | Restored fail-closed | **PASS** | Verified in `experiments/exp06d/run_exp06d.py:73` |

*Note: All verifications were executed and recomputed. No item was marked NOT RUN.*

---

## 2. Task 1: Prediction Integrity Audit

An independent parsing of `reports/accuracy_improvement/exp06d/config_a_oof_predictions.csv` and `reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv` was conducted:

1. **Cohort Validation:**
   - Both CSV files contain exactly 80 rows corresponding to the 80 approved development participants.
   - Zero participants from the locked 21-participant test cohort (`P081`–`P101`) or the 26-participant hospital cohort are present.
   - Row-by-row participant ID order is identical between Config A and Config B.
2. **Ground-Truth Integrity:**
   - The true class distribution across the 80 participants is:
     - `A+`: 19 participants (23.75%)
     - `AB+`: 7 participants (8.75%)
     - `B+`: 25 participants (31.25%)
     - `O+`: 29 participants (36.25%)
   - True labels match between Config A and Config B with 100% agreement.
3. **Probability Vector Validation:**
   - Minimum probability: $0.000000$ (non-negative)
   - Maximum probability: $0.999981$ (finite, within $[0, 1]$)
   - Sum of probabilities per row: $\sum_{c} P(y=c|x) = 1.000000 \pm 10^{-6}$ for all 80 rows across both configurations.
4. **Argmax & Canonical Ordering:**
   - Canonical class mapping: `0: A+`, `1: AB+`, `2: B+`, `3: O+`.
   - In both files, `predicted_label` exactly equals `['A+', 'AB+', 'B+', 'O+'][argmax([prob_A+, prob_AB+, prob_B+, prob_O+])]` across all 80 rows.

---

## 3. Task 2: Independent Metrics Recomputation

Metrics were calculated from first principles directly from the raw out-of-fold prediction files and compared against reported values:

### 3.1 Overall Performance Comparison

| Metric | Config A Recomputed | Config A Reported | Config B Recomputed | Config B Reported | $\Delta$ (B - A) Recomputed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **46.25%** (37/80) | 46.25% | **52.50%** (42/80) | 52.50% | **+6.25%** (+5 participants) |
| **Balanced Accuracy** | **45.86%** | 45.86% | **50.07%** | 50.07% | **+4.21%** |
| **Macro Precision** | **0.4507** | 0.4507 | **0.5220** | 0.5220 | **+0.0713** |
| **Macro Recall** | **0.4586** | 0.4586 | **0.5007** | 0.5007 | **+0.0421** |
| **Macro F1 (Pooled OOF)** | **0.4456** | 0.4456 | **0.5078** | 0.5078 | **+0.0622** |
| **Fold F1 Mean** | **0.4317** | 0.4317 | **0.4771** | 0.4771 | **+0.0454** |
| **Fold F1 Std** | **0.1044** | 0.1044 | **0.1099** | 0.1099 | **+0.0055** |

### 3.2 Per-Class Metrics Recomputation

#### Config A: Enhanced C5 Preprocessing (Reference)
- **A+** ($N=19$): Precision = 44.44% (8/18), Recall = 42.11% (8/19), F1 = 0.4324
- **AB+** ($N=7$): Precision = 37.50% (3/8), Recall = 42.86% (3/7), F1 = 0.4000
- **B+** ($N=25$): Precision = 45.71% (16/35), Recall = 64.00% (16/25), F1 = 0.5333
- **O+** ($N=29$): Precision = 52.63% (10/19), Recall = 34.48% (10/29), F1 = 0.4167

#### Config B: Raw Platen C5 Preprocessing (Candidate)
- **A+** ($N=19$): Precision = 47.37% (9/19), Recall = 47.37% (9/19), F1 = 0.4737
- **AB+** ($N=7$): Precision = 50.00% (3/6), Recall = 42.86% (3/7), F1 = 0.4615
- **B+** ($N=25$): Precision = 60.00% (12/20), Recall = 48.00% (12/25), F1 = 0.5333
- **O+** ($N=29$): Precision = 51.43% (18/35), Recall = 62.07% (18/29), F1 = 0.5625

### 3.3 Confusion Matrix Recomputation

Recomputed confusion matrices strictly match the canonical class order `[A+, AB+, B+, O+]` and the generated heatmap figures:

```
Config A Recomputed Confusion Matrix:
Rows = Ground Truth, Columns = Predicted [A+, AB+, B+, O+]
[[ 8,  1,  5,  5],    <- True A+  (19)
 [ 1,  3,  2,  1],    <- True AB+ (7)
 [ 4,  2, 16,  3],    <- True B+  (25)
 [ 5,  2, 12, 10]]    <- True O+  (29)

Config B Recomputed Confusion Matrix:
Rows = Ground Truth, Columns = Predicted [A+, AB+, B+, O+]
[[ 9,  1,  2,  7],    <- True A+  (19)
 [ 1,  3,  2,  1],    <- True AB+ (7)
 [ 3,  1, 12,  9],    <- True B+  (25)
 [ 6,  1,  4, 18]]    <- True O+  (29)
```

Both matrices correspond 100% with `reports/accuracy_improvement/exp06d/confusion_matrices/config_a_confusion_matrix.png` and `config_b_confusion_matrix.png`.

---

## 4. Task 3: Statistical Verification & Equivalence Distinction

### 4.1 Paired Classification Correctness Contingency Table

The paired correctness of predictions was computed across all 80 participants:

| Participant Outcome | Config B Correct | Config B Incorrect | Total Config A |
| :--- | :---: | :---: | :---: |
| **Config A Correct** | **$n_{11} = 28$** | **$n_{10} = 9$** | **37** (46.25%) |
| **Config A Incorrect** | **$n_{01} = 14$** | **$n_{00} = 29$** | **43** (53.75%) |
| **Total Config B** | **42** (52.50%) | **38** (47.50%) | **$N = 80$** |

- Concordant participants: $n_{11} + n_{00} = 28 + 29 = 57$ (71.25%)
- Discordant participants: $n_{10} + n_{01} = 9 + 14 = 23$ (28.75%)

### 4.2 Exact Binomial Test

The two-sided exact binomial test on the discordant pairs ($n_{10}=9, n_{01}=14, n_{\text{discordant}}=23, p_0=0.5$):
$$\text{Recomputed } p\text{-value} = 2 \times \sum_{k=0}^{9} \binom{23}{k} 0.5^{23} = 0.404873$$
- **Reported value:** $0.4049$
- **Verification status:** **PASS (Exact Match)**

### 4.3 Paired Bootstrap Confidence Interval

Using the documented methodology ($B = 2,000$ paired participant-level resamples with replacement, NumPy random seed = 42):
- **Recomputed Bootstrap Mean $\Delta \text{Macro F1}$:** $+0.0618$
- **Recomputed 95% Confidence Interval (2.5th to 97.5th percentiles):** `[-0.0669, 0.1912]`
- **Reported values:** Mean $+0.0618$, 95% CI `[-0.0669, 0.1912]`
- **Verification status:** **PASS (Bit-for-bit Exact Match)**

### 4.4 Crucial Methodological Distinction: Non-Significance vs. Equivalence

> [!CAUTION]
> **Absence of Evidence is Not Evidence of Absence:**  
> A p-value of $p = 0.4049$ ($> 0.05$) indicates that the null hypothesis ($H_0: \pi_{\text{Config A}} = \pi_{\text{Config B}}$) **cannot be rejected** at the $\alpha = 0.05$ level.
> 
> This non-significant result **MUST NOT** be interpreted as evidence that Config A and Config B are equivalent. On a sample size of $N=80$ development participants with only 23 discordant pairs, statistical power is low ($\approx 26\%$ to detect a true $+6.25\%$ difference). The wide 95% confidence interval `[-0.0669, 0.1912]` encompasses both a clinically non-trivial loss of $-6.7\%$ and a substantial gain of $+19.1\%$. Establishing equivalence would require a formalized Two One-Sided Tests (TOST) procedure with an agreed equivalence margin ($[-\delta, +\delta]$), which the current data cannot support.

---

## 5. Task 4: Scientific Claims Audit

The text of `reports/accuracy_improvement/exp06d/final_report.md` was subjected to a line-by-line compliance audit:

1. **Claim of Statistical Significance:**
   - **Requirement:** Do NOT claim statistical significance.
   - **Audit Finding:** COMPLIANT. The report states: *"the exact binomial test ($p=0.4049$) and bootstrap 95% CI spanning zero (`[-0.0669, 0.1912]`) indicate that the null hypothesis of equal participant-level classification performance cannot be formally rejected at the $\alpha = 0.05$ significance level on the $N=80$ sample."* (Lines 45, 192, 223).
2. **Claim that Preprocessing Damaged Features:**
   - **Requirement:** Do NOT claim that preprocessing has been proven to damage fingerprint features.
   - **Audit Finding:** COMPLIANT. Section 4 explicitly frames feature degradation as an *"experimental hypothesis"* and *"empirical results support this explanation"* rather than an asserted mathematical proof or general biological law.
3. **Claim of Clinical Validity:**
   - **Requirement:** Do NOT claim clinical validity.
   - **Audit Finding:** COMPLIANT. Prominent alert disclaimer is placed at the head of the report (Lines 11–13): *"The metrics reported in this document are 5-fold participant-grouped cross-validation results on the approved 80-participant development cohort ($N=80$). They do NOT represent independent clinical diagnostic performance estimates. Fingerprint biometric classification remains an investigational methodology and cannot replace standard laboratory serological blood-group typing."*
4. **Description of Cross-Validation as Independent Test Performance:**
   - **Requirement:** Do NOT describe CV results as independent test performance.
   - **Audit Finding:** COMPLIANT. All metrics are explicitly qualified throughout as *"pooled OOF"* and *"5-fold participant-grouped cross-validation on the development cohort ($N=80$)"*.
5. **Definitive Claim of Config B Superiority:**
   - **Requirement:** Do NOT conclude that Config B is definitively superior.
   - **Audit Finding:** COMPLIANT. The report concludes that Config B achieved *"superior empirical results"* on this sample, while reiterating that the difference *"does not achieve statistical significance on $N=80$"* and represents an exploratory developmental baseline rather than a finalized production verdict.

---

## 6. Task 5: Checkpoint and Cohort Safety Audit

1. **Cryptographic Checkpoint Ledger Verification:**
   - All 66 historical and production model checkpoints listed in `scratch/exp06d_pre_preparation_hashes.json` were audited using SHA-256:
     - Production E2 (5 folds): Verified bit-for-bit unchanged.
     - EXP-03 (15 models): Verified bit-for-bit unchanged.
     - EXP-05 (20 models): Verified bit-for-bit unchanged.
     - EXP-06A (6 models): Verified bit-for-bit unchanged.
     - EXP-06B (10 models): Verified bit-for-bit unchanged.
     - EXP-06C (10 models): Verified bit-for-bit unchanged.
   - **Corrupted / Missing Checkpoints:** **0**
2. **Quarantine of Protected Cohorts:**
   - Locked 21-participant test cohort: Zero files accessed, zero IDs included in training or evaluation partitions.
   - External 26-participant hospital cohort: Zero files accessed, zero IDs included.
3. **Fail-Closed Execution Guard:**
   - Inspection of `experiments/exp06d/run_exp06d.py:73` confirms:
     ```python
     TRAINING_AUTHORIZED = False
     ```
   - Execution guard is safely armed in fail-closed state.

---

## 7. Audit Conclusion & Stop Confirmation

- **EXP-06D Status:** **VERIFIED & AUDITED**
- **Recommendation:** Raw Platen Preprocessing (Config B) serves as a scientifically sound, non-destructive, and computationally efficient candidate for future exploratory research, showing an empirical $+6.25\%$ accuracy improvement over Enhanced C5 without adding model complexity. However, because statistical significance is not achieved on the $N=80$ sample ($p=0.4049$), neither configuration can be declared definitively superior.
- **Action Taken:** Audit is complete. No further experiments, training runs, or cohort evaluations are initiated. Execution has halted.
