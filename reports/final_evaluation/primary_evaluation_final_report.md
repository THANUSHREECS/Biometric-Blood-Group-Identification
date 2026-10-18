# Definitive Primary Evaluation Report: EXP-06C Config A
**Project:** Biometric Blood Group Detection / HemoPrint AI  
**Evaluation Target:** Primary Locked Test Cohort ($N = 21$ participants, 63 scans)  
**Evaluator Implementation:** `experiments/final_evaluation/evaluate_frozen_candidate.py`  
**Execution Timestamp:** 2026-10-16T16:38:44Z  
**Protocol Reference:** `reports/final_evaluation/preregistered_evaluation_protocol.md`  
**Candidate Manifest:** `reports/final_evaluation/candidate_freeze_manifest.md`  

---

## 1. Executive Summary & Core Scientific Findings

Under explicit authorization, the pre-registered primary evaluation of the frozen **EXP-06C Config A** candidate (`DenseNet121_Fusion_LinearHead`) was executed exactly once on the primary locked 21-participant test cohort (`data/mfs100_final`, split = `TEST`).

### 1.1 Observed Metrics Summary

| Metric | Pre-Registered Threshold | Historical Phase 3 Baseline (Model B) | EXP-06C Config A Primary Test Result | Bootstrap 95% CI ($B=2,000$) | Pre-Registered Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary: Macro F1** | $> 0.3500$ & $> \text{Phase 3}$ | **0.2500** | **0.2083** | **[0.0749, 0.3481]** | **FAIL** |
| **Accuracy** | — | **38.10%** (8 / 21) | **33.33%** (7 / 21) | [14.29%, 52.38%] | Underperformed baseline |
| **Balanced Accuracy** | — | **30.62%** | **31.25%** | [12.50%, 50.00%] | Parity with baseline |
| **Majority (Always-O+)** | Benchmark | 38.10% (8 / 21) | 38.10% (8 / 21) | — | Underperformed majority rule |
| **Random Guessing** | Benchmark | 25.00% | 25.00% | — | Above chance, below majority |

### 1.2 Core Scientific Conclusion

> [!CAUTION]
> ### CLINICAL VALIDITY HAS NOT BEEN ESTABLISHED
> **EXP-06C Config A failed to achieve the pre-registered primary performance threshold.**  
> - The observed **Macro F1 of 0.2083** falls below the pre-registered minimum benchmark ($0.3500$) and is lower than the historical Phase 3 Model B baseline ($0.2500$). The 95% bootstrap confidence interval $[0.0749, 0.3481]$ excludes the pre-registered target of $0.3500$.
> - The observed **Accuracy of 33.33%** (7 of 21 participants correctly classified) underperforms naive majority-class assignment (Always-O+ = $38.10\%$).
> - Paired comparison on discordant participant pairs against the Phase 3 baseline yields an exact two-sided binomial $p = 1.0000$ ($n_{\text{discordant}} = 7$), confirming no statistically significant difference from the historical baseline.
> - **The biometric blood group detection candidate cannot be deployed in any clinical, diagnostic, screening, or operational setting.** Cross-validation gains observed during development cross-validation ($N=80$, OOF Macro F1 $= 0.5487$) did not generalize to the held-out test cohort.

---

## 2. Cryptographic Checkpoint & Environment Verification

### 2.1 Candidate Checkpoint Ledger (Pre- and Post-Evaluation)

In accordance with Section 5 of the protocol, all five checkpoint files were cryptographically verified using SHA-256 before inference, and re-verified bit-for-bit immediately following evaluation completion. Zero bytes were modified.

| Fold | Checkpoint Relative Path | File Size (Bytes) | SHA-256 Hash | Pre-Check | Post-Check | Status |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| **Fold 1** | `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5` | 52,899,216 | `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc` | MATCH | MATCH | **UNTOUCHED** |
| **Fold 2** | `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5` | 52,899,472 | `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c` | MATCH | MATCH | **UNTOUCHED** |
| **Fold 3** | `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5` | 52,899,472 | `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1` | MATCH | MATCH | **UNTOUCHED** |
| **Fold 4** | `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5` | 52,899,472 | `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3` | MATCH | MATCH | **UNTOUCHED** |
| **Fold 5** | `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5` | 52,899,472 | `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6` | MATCH | MATCH | **UNTOUCHED** |

### 2.2 Execution Guards & Parameter Count
- `TRAINING_AUTHORIZED = False` remained active continuously throughout execution.
- `EVALUATION_AUTHORIZED` was temporarily enabled for the single authorized run and immediately reset to `False` upon termination.
- Checkpoint models were instantiated with `training=False` and all layers frozen:
  - **Trainable Parameters during evaluation:** `0` (Expected `0`).
  - **Non-trainable Parameters:** `7,043,908` per fold model.

---

## 3. Evaluation Protocol & Execution Details

### 3.1 Input Assembly and Preprocessing
- **Cohort Dataset:** `data/mfs100_final/raw/`, split = `TEST` from `reports/phase3_participant_benchmark/participant_split.csv`.
- **Participant Verification:** Exactly 21 participants, each with exactly 3 raw BMP scans (`scan_01.bmp`, `scan_02.bmp`, `scan_03.bmp`), yielding 63 total scans.
- **Preprocessing:** Each raw scan was processed via `src.preprocessing.transform_c5` (alpha compositing $\rightarrow$ platen crop $\rightarrow$ CLAHE $\rightarrow$ percentile stretch $\rightarrow$ bilateral filter $\rightarrow$ Lanczos resize $112 \times 112 \times 3$, float32 $[0, 255]$).
- **Ensemble Inference:** Each participant's 3-scan tensor $(1, 3, 112, 112, 3)$ was evaluated through all five fold checkpoints. Softmax probability distributions were averaged using equal-weight soft voting:
  $$\mathbf{P}_{\text{ensemble}}(y=c) = \frac{1}{5} \sum_{k=1}^5 \operatorname{Softmax}_c(\text{Model}_k(\mathbf{X}))$$
- **Class Decision:** Strict $\operatorname{argmax}$ over canonical classes:
  $$\text{CANONICAL\_CLASSES} = [\text{"A+"}, \text{"AB+"}, \text{"B+"}, \text{"O+"}]$$
  No threshold tuning, temperature adjustment, or post-hoc calibration was applied.

### 3.2 Cohort Isolation Verification
- The separate 26-participant hospital cohort (`data/captured/`) was **strictly quarantined and never accessed**.
- The assertion `assert "captured" not in scan_file` was verified on all 63 processed scan paths.

---

## 4. Comprehensive Evaluation Metrics

### 4.1 Primary & Secondary Endpoints

| Metric | Point Estimate | 95% Bootstrap CI (2,000 Resamples) | Bootstrap Mean |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score (Primary)** | **0.2083** | **[0.0749, 0.3481]** | 0.1959 |
| **Overall Accuracy** | **33.33%** (7 / 21) | **[14.29%, 52.38%]** | 33.27% |
| **Balanced Accuracy** | **31.25%** | **[12.50%, 50.00%]** | 30.45% |
| **Macro Precision** | **0.1562** | — | — |
| **Macro Recall** | **0.3125** | — | — |

### 4.2 Per-Class Performance Breakdown

| Class | True Support ($N_c$) | Model Predictions | True Positives | False Positives | False Negatives | Precision | Recall | Class F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 5 (23.8%) | 0 | 0 | 0 | 5 | 0.0000 | 0.0000 | **0.0000** |
| **AB+** | 2 (9.5%) | 4 | 1 | 3 | 1 | 0.2500 | 0.5000 | **0.3333** |
| **B+** | 6 (28.6%) | 1 | 0 | 1 | 6 | 0.0000 | 0.0000 | **0.0000** |
| **O+** | 8 (38.1%) | 16 | 6 | 10 | 2 | 0.3750 | 0.7500 | **0.5000** |
| **Macro Avg** | **21** | **21** | **7** | **14** | **14** | **0.1562** | **0.3125** | **0.2083** |

### 4.3 Confusion Matrix

Rows represent ground-truth labels; columns represent ensemble-predicted labels. Canonical class ordering: `[A+, AB+, B+, O+]`.

```
                    PREDICTED
               A+    AB+     B+     O+    Total
         A+ [   0      1      0      4 ]     5
TRUE    AB+ [   0      1      0      1 ]     2
         B+ [   0      1      0      5 ]     6
         O+ [   0      1      1      6 ]     8
      Total     0      4      1     16    N=21
```

#### Detailed Error Analysis
- **Severe Majority-Class Bias:** The ensemble predicted `O+` for 16 out of 21 participants (76.2% of all test predictions), despite `O+` comprising only 38.1% of the cohort.
- **Complete Inability to Detect A+ and B+:**
  - `A+` sensitivity: **0.0%** (0 / 5). 4 participants misclassified as `O+`, 1 misclassified as `AB+`.
  - `B+` sensitivity: **0.0%** (0 / 6). 5 participants misclassified as `O+`, 1 misclassified as `AB+`.
- **AB+ Over-prediction:** AB+ was predicted 4 times with only 1 true positive (`P0082`) and 3 false positives (`P0039` [O+], `P0083` [A+], `P0093` [B+]).

---

## 5. Statistical Hypothesis Testing vs. Historical Baseline

### 5.1 Paired Participant-Level Comparison

A paired comparison of individual participant classifications was conducted between the historical **Phase 3 Model B** baseline and **EXP-06C Config A** across all 21 test participants:

| Participant ID | True Label | Phase 3 Model B Prediction | EXP-06C Config A Prediction | Phase 3 Correct? | EXP-06C Correct? | Concordance |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P002** | O+ | A+ | O+ | Incorrect | **Correct** | Discordant (EXP-06C+) |
| **P0023** | A+ | B+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0024** | B+ | A+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0029** | O+ | O+ | O+ | **Correct** | **Correct** | Concordant Correct |
| **P003** | O+ | O+ | O+ | **Correct** | **Correct** | Concordant Correct |
| **P0035** | A+ | A+ | O+ | **Correct** | Incorrect | Discordant (Phase 3+) |
| **P0039** | O+ | A+ | AB+ | Incorrect | Incorrect | Concordant Error |
| **P0041** | O+ | O+ | O+ | **Correct** | **Correct** | Concordant Correct |
| **P0042** | O+ | O+ | O+ | **Correct** | **Correct** | Concordant Correct |
| **P0047** | B+ | A+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0048** | B+ | A+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0054** | A+ | O+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0071** | A+ | A+ | O+ | **Correct** | Incorrect | Discordant (Phase 3+) |
| **P0073** | O+ | A+ | O+ | Incorrect | **Correct** | Discordant (EXP-06C+) |
| **P0075** | O+ | O+ | B+ | **Correct** | Incorrect | Discordant (Phase 3+) |
| **P0082** | AB+ | B+ | AB+ | Incorrect | **Correct** | Discordant (EXP-06C+) |
| **P0083** | A+ | A+ | AB+ | **Correct** | Incorrect | Discordant (Phase 3+) |
| **P0088** | B+ | O+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P009** | AB+ | O+ | O+ | Incorrect | Incorrect | Concordant Error |
| **P0093** | B+ | A+ | AB+ | Incorrect | Incorrect | Concordant Error |
| **P0095** | B+ | A+ | O+ | Incorrect | Incorrect | Concordant Error |

### 5.2 Contingency Table & Exact Binomial Test

```
                              EXP-06C Config A
                            Correct     Incorrect     Total
Phase 3 Model B  Correct        4           4           8
                Incorrect       3          10          13
                  Total         7          14          21
```

- **Both Models Correct:** 4 participants (`P0029`, `P003`, `P0041`, `P0042` — all O+).
- **Phase 3 Model B Only Correct:** 4 participants (`P0035` [A+], `P0071` [A+], `P0075` [O+], `P0083` [A+]).
- **EXP-06C Config A Only Correct:** 3 participants (`P002` [O+], `P0073` [O+], `P0082` [AB+]).
- **Both Models Incorrect:** 10 participants.
- **Discordant Pairs ($n$):** $4 + 3 = 7$.
- **Two-Sided Exact Binomial Test ($H_0: \pi = 0.5$):**
  $$\text{Successes} = 3, \quad n = 7, \quad p\text{-value} = 1.0000$$

**Interpretation:** There is no statistically significant performance difference between EXP-06C Config A and the Phase 3 baseline ($p = 1.0000$). The model changes in EXP-06C did not produce an empirical improvement on unseen participants.

---

## 6. Execution Errors and Protocol Deviations

- **Execution Errors:** None. The evaluator completed all 21 inferences, probability calculations, metric computations, and 2,000 bootstrap resamples cleanly with exit code 0.
- **Deviations:** Zero protocol deviations occurred.
  - The model architecture, preprocessing pipeline, canonical class ordering, ensemble soft-voting method, and test cohort remained strictly identical to pre-registered specifications.
  - No post-evaluation retraining, fine-tuning, or threshold selection was attempted.

---

## 7. Methodological and Clinical Limitations

### 7.1 Sample Size and Statistical Power ($N = 21$)
- The primary test cohort contains only 21 participants (63 scans).
- Each individual participant represents $\frac{1}{21} \approx 4.76\%$ of overall accuracy.
- Rare classes (such as AB+, $N=2$) have extreme granularity: correctly identifying or missing a single participant shifts class recall by $\pm 50.0\%$.
- The 95% bootstrap confidence intervals are consequently broad: Macro F1 $[0.0749, 0.3481]$ and Accuracy $[14.29\%, 52.38\%]$.

### 7.2 Total Absence of Verified Laboratory Serology Labels
- Ground-truth blood group labels in the MFS100 dataset were collected via **unverified participant self-reporting** during biometric enrollment.
- No hemagglutination tube testing, column agglutination card tests, or laboratory serological validation was performed to establish gold-standard phenotypic ground truth.
- As a consequence, label noise or erroneous participant self-reports cannot be ruled out.

### 7.3 Prior Test Cohort Exposure History
- As disclosed in `cohort_provenance_review.md`, this 21-participant cohort was historically evaluated once during Phase 3 baseline development.
- Although the cohort remained strictly quarantined and unaccessed throughout the subsequent ten-experiment improvement campaign (EXP-01 through EXP-06D), it cannot be claimed as a completely virgin dataset within the total multi-year project history.

### 7.4 Generalization Gap from Development to Test
- During 5-fold cross-validation on the 80 development participants, EXP-06C Config A achieved an out-of-fold Macro F1 of **0.5487** and an Accuracy of **53.75%**.
- On the held-out test cohort ($N=21$), performance dropped precipitously to **Macro F1 = 0.2083** and **Accuracy = 33.33%**.
- This discrepancy indicates that deep feature representations learned on the development cohort captured sample-specific biometric correlations rather than true, generalizable phenotypic blood group signals.

---

## 8. Definite Audit File Manifest

The primary evaluation execution generated the following persistent audit records:

1. **Prediction File:**
   [`reports/final_evaluation/primary_evaluation_predictions.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/primary_evaluation_predictions.csv)  
   Contains 21 rows of participant IDs, ground-truth labels, predicted labels, and ensemble class probabilities.
2. **Metrics & Bootstrap JSON:**
   [`reports/final_evaluation/primary_evaluation_results.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/primary_evaluation_results.json)  
   Contains complete serialized metric results, confusion matrix, per-class metrics, and 2,000-resample bootstrap distributions.
3. **Task Log Record:**
   [`C:\Users\thanu\.gemini\antigravity-ide\brain\b76080da-a0bf-4741-9da1-3d1249514eab\.system_generated\tasks\task-1553.log`](file:///C:/Users/thanu/.gemini/antigravity-ide/brain/b76080da-a0bf-4741-9da1-3d1249514eab/.system_generated/tasks/task-1553.log)  
   Complete stdout/stderr execution log from process task 1553.

---

## 9. Final Regulatory & Scientific Attestation

```
================================================================================
FINAL SCIENTIFIC ATTESTATION: CLINICAL NON-VALIDITY
================================================================================
Evaluation Candidate: EXP-06C Config A (DenseNet121_Fusion_LinearHead)
Primary Test Cohort:  MFS100 Held-Out Test Set (N = 21 participants, 63 scans)
Primary Metric:       Macro F1 = 0.2083 [95% CI: 0.0749 - 0.3481]
Target Status:        FAILED (Pre-registered threshold > 0.3500 not achieved)
Accuracy:             33.33% (7 / 21) [Below Always-O+ baseline of 38.10%]
Baseline Comparison:  No significant difference vs. Phase 3 Baseline (p = 1.0000)

CLINICAL DECLARATION:
The evaluated optical biometric blood group detection system DOES NOT possess
clinical validity, diagnostic accuracy, or screening utility. The hypothesis
that optical fingerprint scans can reliably determine ABO/Rh blood groups via
deep convolutional neural networks is NOT supported by empirical test results.
================================================================================
```
