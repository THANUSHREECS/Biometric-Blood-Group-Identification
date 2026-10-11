# HemoPrint AI: Final Audit Results Table
*(Updated Round 20 — branch `fix-preprocessing-mismatch`)*

## 1. Primary Evaluation Summary Table

| Evaluation Setting | Model / Architecture | Samples / Scope | Metric / Accuracy | Baseline & Chance | Status & Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Dataset)** | **Model A: EfficientNetB0** (not deployed) | 415 test images (random image split, public fingerprint dataset) | **86.02%** (95% CI: [82.65%, 88.92%]) | Majority baseline: 30.6% | Standalone single-image model; **not deployed**. Subject independence not verifiable (no subject IDs). Metadata artifact classifiers achieve 44.58% (random split), shortcut features (intensity/contrast); leakage unverified. |
| **21 Locked Test Participants** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 21 Locked Test Participants (63 scans, 3 per person) | **38.10%** (8/21; Wilson 95% CI: [20.75%, 59.12%]) | Always-O+ baseline: 38.10% (8/21); binomial p = 1.0000 | **Valid**: True locked test, exact reproduction of `final_test_predictions.csv` (21/21 top-1 match, max diff = 0.00000019). Matches always guessing O+ (8/21). Pred dist: A+=11, B+=2, AB+=0, O+=8. |
| **80 Development Cohort (In-Sample Ensemble)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (240 scans, evaluated with full 5-checkpoint ensemble) | **40.00%** (32/80; Wilson 95% CI: [29.96%, 50.95%]) | Always-O+ baseline: 36.25% (29/80) | **In-sample dev evaluation**: Full 5-checkpoint ensemble evaluated on dev cohort. Pred dist: A+=47, O+=16, B+=17. |
| **80 Development Cohort (Out-of-Fold Validation)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (StratifiedGroupKFold, out-of-fold checkpoints) | **53.75%** (43/80; Wilson 95% CI: [42.90%, 64.25%]) | Always-O+ baseline: 36.25% (29/80) | **Out-of-Fold (OOF)**: Each fold model evaluated strictly on its held-out validation fold (`fold_results.csv`). Average training accuracy across 5 folds was **35.94%**. |
| **101 Pooled Participants (Deployed Path)** | **Model B: E2 DenseNet121 Ensemble** (deployed post-fix) | 101 Participants (80 dev + 21 locked test) | **39.60%** (40/101; Wilson 95% CI: [30.62%, 49.35%]) | Always-O+ baseline: 36.63% (37/101); binomial p = 0.5369 | **Contaminated benchmark**: Contains 80 development subjects. Pred dist: A+=58, O+=24, B+=19. |
| **101 Pooled — Old Mismatched Pipeline (INVALID)** | **Model B: E2 DenseNet121 Ensemble** (pre-Round 14) | 101 Participants | **27.72% — INVALID** (28/101; Wilson 95% CI: [19.93%, 37.15%]) | Always-O+ baseline: 36.63% (37/101) | **Invalid**: Caused by inadvertent addition of `densenet.preprocess_input` to raw `[0, 255]` weights. Corrected in Round 14. |

---

## 2. Three-Pipeline Sensitivity Analysis (21 Locked Test Participants)

Pipelines evaluated:
- **P1 (Experiment 6 Loader)**: Verbatim `scratch/experiment6_fusion_supcon.py` / `scratch/final_evaluation_e2.py` (composite + crop + CLAHE + LANCZOS 112x112 + raw `[0, 255]` float32).
- **P2 (Current Deployed `transform_c5`)**: `src/preprocessing.py:transform_c5` post-fix (identical subroutine calls, raw `[0, 255]` float32).
- **P3 (Old Deployed Pipeline)**: P2 + `densenet.preprocess_input` (pre-Round 14 bug).
- **P_naive (Ablation: Naive Bilinear)**: Direct PIL RGB Bilinear resize to 112x112, raw `[0, 255]` (no alpha composite, crop, or CLAHE).

| Pipeline | Agreement with Saved CSV (`final_test_predictions.csv`) | Max Abs Probability Diff vs Saved CSV | Test Accuracy (21 subjects) | Wilson 95% CI | Predicted Distribution | Per-Class Recall (A+, B+, AB+, O+) |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **P1 (Experiment 6)** | **21 / 21 (100.0%)** | **0.00000019** | **38.10% (8/21)** | [20.75%, 59.12%] | A+: 11, B+: 2, O+: 8 | A+: 3/5 (60%), B+: 0/6 (0%), AB+: 0/2 (0%), O+: 5/8 (62.5%) |
| **P2 (Deployed `transform_c5`)** | **21 / 21 (100.0%)** | **0.00000019** | **38.10% (8/21)** | [20.75%, 59.12%] | A+: 11, B+: 2, O+: 8 | A+: 3/5 (60%), B+: 0/6 (0%), AB+: 0/2 (0%), O+: 5/8 (62.5%) |
| **P3 (Old Pre-Round 14)** | 9 / 21 (42.9%) | 0.73301940 | 23.81% (5/21) | [10.63%, 45.09%] | A+: 18, B+: 3 | A+: 4/5 (80%), B+: 1/6 (16.7%), AB+: 0/2 (0%), O+: 0/8 (0%) |
| **P_naive (Bilinear Ablation)** | 17 / 21 (81.0%) | 0.37943891 | 38.10% (8/21) | [20.75%, 59.12%] | A+: 8, B+: 4, O+: 9 | A+: 3/5 (60%), B+: 1/6 (16.7%), AB+: 0/2 (0%), O+: 4/8 (50%) |

**Decision & Deployment Verification**:
Only P1 and P2 reproduce `final_test_predictions.csv` with 21/21 top-1 match and <$10^{-6}$ numerical tolerance. Because P2 is pixel-identical to P1, the deployed pipeline (`src/preprocessing.py:transform_c5`) is retained unchanged.

---

## 3. Development In-Sample vs. Out-of-Fold vs. Training Accuracy

From the 5-fold Stratified Group K-Fold training run documented in `reports/phase3_fusion_metric_learning/fold_results.csv`:

| Fold | Training Accuracy | Out-of-Fold Validation Accuracy | Out-of-Fold Validation Macro F1 |
| :---: | :---: | :---: | :---: |
| Fold 1 | 40.63% (26/64) | 43.75% (7/16) | 0.2902 |
| Fold 2 | 29.69% (19/64) | 50.00% (8/16) | 0.4167 |
| Fold 3 | 34.38% (22/64) | 50.00% (8/16) | 0.3818 |
| Fold 4 | 39.06% (25/64) | 56.25% (9/16) | 0.6253 |
| Fold 5 | 35.94% (23/64) | 68.75% (11/16) | 0.4059 |
| **Average / Total** | **35.94% (115/320)** | **53.75% (43/80)** | **0.4240** |

- **In-Sample 5-Fold Ensemble on Dev Cohort**: **40.00%** (32/80 correct).
- **Out-of-Fold Validation on Dev Cohort**: **53.75%** (43/80 correct).
- **Mean Training Accuracy across 5 Folds**: **35.94%** (~35.9%).

---

## 4. Key Provenance and Determinism Findings

1. **Determinism at Inference**:
   Predictions are fully deterministic. `data_augmentation` layers (`RandomRotation`, `RandomTranslation`, `RandomZoom`, `RandomContrast`) and `Dropout` layers are called with `training=False`, making them identity transforms at inference. Repeating evaluation on identical inputs yields max probability difference = `0.0000000000`. No random seed or stochastic augmentation effects exist at test time.

2. **Frozen Config vs. Code Parity**:
   `reports/phase3_final_evaluation/frozen_model_config.json` describes the preprocessing sequence:
   `"composite_alpha_on_white -> crop_mfs100_foreground -> enhance_mfs100_ridge_contrast -> resize (112,112) Lanczos"`.
   Reading `scratch/experiment6_fusion_supcon.py` lines 45–51 shows `transform_c5` calls `composite_alpha_on_white`, `crop_mfs100_foreground`, `enhance_mfs100_ridge_contrast`, and `resize_to_target` (Lanczos). Reading `src/preprocessing.py:transform_c5` (post Round 14 fix, commit `bce0e7a`) shows the same four calls. The frozen config description matches both. *(Round 19 qualification: `c1aac08` is the first commit of `src/preprocessing.py` in the repository's git history (`git log --follow` shows only two commits: `c1aac08` and `bce0e7a`). `git show c1aac08` shows the file was introduced already containing `densenet.preprocess_input`. This means git records when tracking began, not necessarily when the code was originally written. The statement "bug introduced in commit c1aac08" should be read as: `densenet.preprocess_input` was present in the file from its first tracked version; there is no earlier commit showing it being added.)*

---

## 5. Statistical Power, OOF Checkpoint Selection, and Saved CSV Equivalence

1. **Saved CSV = Deployed Path**:
   The saved locked test predictions (`reports/phase3_final_evaluation/final_test_predictions.csv`) record: `A+: 11, B+: 2, AB+: 0, O+: 8` (8/21 = 38.10% accuracy). This is identical to the current deployed path (`src/preprocessing.py:transform_c5`), matching 21/21 top-1 predictions with a max absolute probability difference of `0.00000019`. Prior speculative explanations referencing "4/1/0/16" or "random seed drift" were conclusively refuted: model inference is 100% deterministic, and the saved CSV directly reflects the production pipeline.

2. **Out-of-Fold (OOF) 53.75% Checkpoint Selection Bias**:
   In `scratch/experiment6_fusion_supcon.py`, each fold's checkpoint (`fold_N_best.weights.h5`) was selected using `F1Checkpoint(monitor='val_macro_f1', mode='max')`. Selecting model weights at the empirical peak of validation macro-F1 creates an inherent optimistic bias. When the full 5-fold ensemble is evaluated jointly on the development cohort without per-fold peak tuning, accuracy drops to 40.00%, training accuracy across the 5 folds averages only 35.94%, and locked test accuracy drops to 38.10%.

3. **Statistical Power Limitation of N = 21 Locked Test Set**:
   The locked test cohort consists of only 21 independent participants (Always-O+ baseline = 38.10%, 8/21). A sample size of $N = 21$ is statistically underpowered and cannot detect genuine accuracy differences below approximately 60% (at $\alpha = 0.05, 1-\beta = 0.80$, distinguishing 50% from 38% would require $N > 130$ independent subjects). The observed score of 8/21 (38.10%) yields a binomial $p = 1.0000$, exactly matching the majority class baseline.

---

## 6. Round 18 Matched Permutation Test (101 Pooled Participants, 5-Fold CV)

**Design**: Observed statistic = mean of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` CV accuracy/balanced-accuracy on the 101 pooled participants. Null distribution = same 5-fold CV applied to 200 participant-level label shuffles. Both observed and null use *identical* folds and *identical* MLP head (NumPy FastNumPyHead: 3072→Dense512/ReLU/Dropout0.5→Dense4/Softmax, Adam lr=1e-3, 20 epochs). This corrects the prior Round 17 test which used mismatched statistics.

> [!WARNING]
> All p-values from Round 17 and earlier permutation tests are **invalid: mismatched statistic** (observed used a single-run score; null used a different evaluation mode).

**Variant A — Raw [0, 255] float32 (matching E2 training)** *(200 shuffles completed in 1897.78s)*:

| Metric | Value |
| :--- | :--- |
| Observed 5-Fold CV Accuracy | **39.67%** |
| Observed 5-Fold CV Balanced Accuracy | **32.18%** |
| Always-O+ Baseline | 36.63% |
| Null Accuracy Mean ± SD | 30.13% ± 4.91% |
| Null Accuracy 95th Percentile | 37.62% |
| Null Accuracy Max | 43.48% |
| **Empirical p-value (Accuracy)** | **0.0348** |
| Null Balanced Accuracy Mean ± SD | 25.90% ± 4.38% |
| Null Balanced Accuracy 95th Percentile | 33.22% |
| Null Balanced Accuracy Max | 41.27% |
| **Empirical p-value (Balanced Accuracy)** | **0.0896** |

**Variant B — DenseNet `preprocess_input` (mismatched pipeline)**: Observed 5-fold CV accuracy = 39.76%, balanced accuracy = 31.20%. Null distribution not completed within 15-minute time limit per Round 19 instruction. p-values for Variant B: **not reported**; Variant B incomplete.

**Interpretation**:
- single seed-42 partition: accuracy 39.7%, balanced accuracy 32.2%; the 10-repeat mean from `round17_fast_clean_estimate.py` is 35.4% +/- 3.4% (balanced 29.7%); always-O+ = 36.6%;
- null 95th percentile of accuracy is 37.6%, close to always-O+, so the accuracy p-value (0.035) mostly reflects class imbalance; balanced-accuracy p = 0.09; Variant B incomplete;
- conclusion: "not distinguishable from chance; weak effects not excluded".
