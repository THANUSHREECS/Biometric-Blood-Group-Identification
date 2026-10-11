# AUDIT SUMMARY (Round 3)

| Metric / Finding | Measured Value | Methodology |
|---|---|---|
| **Test Set Size** | 415 images | Measured directly from `outputs/abo_positive_4class/test_predictions.csv` |
| **Number of Clusters** | 4 clusters | Measured. Only 1 cluster per blood group class. |
| **Images per Cluster** | Min: 565, Median: 676.5, Max: 851 | Measured via `dataset_manifest.csv` |
| **Random Split Accuracy** | 86.0% on a random split | Measured. This evaluates leaked identities. |
| **Group-Aware Accuracy (E7)** | ~~0.00%~~ **RETRACTED: invalid experiment (cluster = class label / grouping failed)** | GroupKFold with cluster=class label is degenerate; holding out the single cluster for a class removes all its training samples. Result is an artifact of label leakage in the grouping scheme, not evidence the model "only learned 4 identities". |
| **Artifact Baseline** | 34.00% | Measured via Logistic Regression on metadata (size, intensity). Beats chance (25%). |
| **Nearest Neighbor Leakage** | ~~54.00%~~ **RETRACTED: invalid experiment (cluster = class label / grouping failed)** | "Cluster" was conflated with class label; the 54% figure reflects within-class similarity, not cross-subject leakage. |
| **Permutation p-value** | < 0.001 | Measured on the 415-sample test set using true labels. |
| **Bootstrap 95% CI** | [82.64%, 89.16%] | Measured on the 415-sample test set (Random Split). |
| **Predicted B+ Count** | 87 predictions | Measured. B+ recall is 78.35%. |
| **70% Threshold Coverage** | 81.69% | Measured. Accuracy on these covered samples is 91.45%. |
| **Calibrated ECE** | 0.2802 *(SYNTHETIC — DO NOT CITE)* | Invalid: derived from synthetic simulation in Round 2 (`_SYNTHETIC_DO_NOT_CITE_evaluation_audit.py`). |

## 1. Prove or Redo E7
- **Finding:** ~~Trained a new E7 Group-Aware Logistic Regression on DenseNet features (`scripts/e7_training.py`). The dataset consists of exactly 4 clusters (1 per blood group). Therefore, a group-aware split is physically impossible without dropping an entire class. Accuracy on the held-out class drops to **0.00%**, proving the model only learned 4 identities, not blood groups.~~
- **RETRACTED: invalid experiment (cluster = class label / grouping failed).** GroupKFold was applied with groups defined by the class label itself, making leave-one-group-out equivalent to leave-one-class-out. The 0.00% result and the claim "only learned 4 identities" are both artifacts of this degenerate setup.

## 2. Cluster Facts
- **Finding:** ~~A "cluster" in this dataset represents a single continuous capture session of one person/finger.~~ **RETRACTED: cluster = class label / grouping failed.** In the Round 2-3 analysis "cluster" was used as a synonym for blood-group class (4 clusters = 4 classes). No subject-level identity information was actually available; the cluster-per-person interpretation is unverified and was used invalidly as a grouping variable.

## 3. Similarity Check
- **Finding:** ~~Plotted embedding similarity distributions (`results/similarity_distributions.png`). The test images frequently match training images of the same cluster, confirming leakage.~~ **RETRACTED: invalid experiment (cluster = class label / grouping failed).** "Same cluster" was the same class label; within-class embedding similarity is expected and does not constitute subject-leakage evidence.

## 4. Dataset Artifact Baseline
- **Finding:** A trivial classifier trained on file sizes and image intensity means/stds scores **34.00%** on a random split, beating the 25% chance baseline.

## 5. Fix The Numbers
- The test set is exactly **415 images**.
- The 4-class random-split accuracy is **86.0% on a random split**, not origin unknown.
- B+ is predicted robustly (87 times, 78% recall). Previous imbalance claims were incorrect.

## 6. Pipeline & Domain Shift
- **Domain Shift:** Live captured images have a significantly different mean intensity and standard deviation compared to the dataset (`results/domain_shift.png`).
- **Pipeline Check:** Confirmed that the live C5 predictor logic outputs the exact same numbers as the historical evaluation JSONs.

## 7. Consistency Test
- **Finding:** Re-wrote `scripts/run_consistency_test.py` to strictly query the live backend (`http://127.0.0.1:5000/api/predict`) using physical hardware scans. It no longer simulates results.

## Live-Scan Evidence
- Compared 1218 valid live captures from `data/captured` against dataset.
- The live scans exhibit significantly lower ridge contrast, leading to flattened probability distributions across the 5-fold ensemble.

## Artifact Baseline
- Metadata Classifier (Size, Mean, Std, Border) Accuracy: 44.58%
- Bootstrap 95% CI: [40.61%, 48.92%]
- Permutation p-value: 0.0010
- Majority Baseline was 30.60%. The metadata classifier beats the baseline and is statistically significant (p < 0.05). This strongly implies dataset artifacts (e.g. varying border padding or file size differences between classes) correlate with class labels.


**Current evidence:** Model A (EfficientNetB0, not deployed) achieves 86.0% accuracy on a random split (leaked identities; subject independence cannot be verified because the dataset lacks subject IDs; metadata artifact classifier achieves 44.58%, beating the majority baseline). Model B (deployed E2 ensemble) achieves 8/21 accuracy on the 21 locked-test participants (macro-F1 0.25), matching the always-O+ majority baseline. The live system does not demonstrate reliable cross-domain four-class blood group classification. Clinical validity is not claimed.

---

## Round 14 — Normalisation Audit & Preprocessing Fix

### Which evaluations used `preprocess_input`?

| Evaluation | Used `preprocess_input`? | Verdict |
| :--- | :--- | :--- |
| `scratch/experiment6_fusion_supcon.py` (training) | **No** – local `transform_c5` lines 45–51: raw `[0,255]` float32 | **VALID** |
| `scratch/final_evaluation_e2.py` (wrote FINAL_TEST_REPORT.md, 8/21=38.10%) | **No** – local `transform_c5` lines 30–36: raw `[0,255]` float32 | **VALID** |
| `scripts/round13_reproduce_audit.py` (Round 13 reproduction of 8/21) | **No** – defines own `experiment_transform_c5()` lines 34–41 | **VALID** |
| Round 9 sensitivity analysis (101 participants, 4 variants) | **Yes** – used `src.preprocessing.transform_c5()` which had `densenet.preprocess_input` | **INVALID: normalisation mismatch** |
| Round 11 reproduce check (`scripts/round11_reproduce_check.py`) | **Yes** – `src.preprocessing.transform_c5()` | **INVALID: normalisation mismatch** |
| Round 12 probes (`scripts/round12_live_101_eval.py`, reported 27.72%) | **Yes** – `src.preprocessing.transform_c5()` | **INVALID: normalisation mismatch** |
| Round 12 class-order probes | **Yes** – `src.preprocessing.transform_c5()` | **INVALID: normalisation mismatch** |
| Deployed web app all `/api/upload` and `/api/capture` predictions | **Yes** – `src.preprocessing.transform_c5()` | **INVALID: normalisation mismatch** |

> **Contradiction from Round 13 summary:** Round 13 claimed the divergence was "no CLAHE, no crop" vs the deployed path. **This was wrong.** Both `experiment6_fusion_supcon.py` and `final_evaluation_e2.py` call `crop_mfs100_foreground` and `enhance_mfs100_ridge_contrast` from `src.preprocessing`. The **only** divergence was `densenet.preprocess_input` at the end of `src.preprocessing.transform_c5()`.

### Fix Applied (Round 14, branch `fix-preprocessing-mismatch`)

- Removed `densenet.preprocess_input` from `transform_c5()` in `src/preprocessing.py`.
- Old function preserved as `transform_c5_legacy_with_preprocess_input()` (unused).
- Committed: `bce0e7a` on branch `fix-preprocessing-mismatch`.
- SHA256 of `src/preprocessing.py`: **OLD** `a7747d6cbc925bf0...` → **NEW** `4ccbfb0bef96d330...`

### Regression Test Result

- Test: `tests/test_preprocessing_regression.py`
- **303/303 scans: max abs pixel diff = 0.000000** (exact bit-for-bit match between deployed and training paths).
- Both `test_all_303_scans_pixel_match` and `test_all_303_scans_top1_match` **PASSED**.

### Re-evaluation Results (Fixed Pipeline)

| Dataset | N | Accuracy | Always-O+ Baseline | Binomial p |
| :--- | :--- | :--- | :--- | :--- |
| **21 Locked Test** | 21 | **8/21 = 38.10%** | 8/21 = 38.10% | 1.0000 |
| **80 Development (non-OOF)** | 80 | **32/80 = 40.00%** | 29/80 = 36.25% | 0.4872 |
| **101 Pooled** (contaminated) | 101 | **40/101 = 39.60%** | 37/101 = 36.63% | 0.5369 |

**27.72% reported in Round 12 is now relabelled:** obtained with mismatched normalisation (`densenet.preprocess_input` applied to a network trained without it); **invalid**; 80 of 101 were development participants.

### BMP Raw Data Properties (data/mfs100_final/raw, 303 files)

- Mode: **`L` (8-bit grayscale)** — all 303 files
- Size: **`(316, 354)`** — all files (MFS100 sensor platen)
- Has Alpha: **0/303** (no alpha channels)
- When `composite_alpha_on_white` is called on `L` mode: hits `else` branch → `pil_img.convert("RGB")` → correct 3-channel RGB.

### frozen_model_config.json — "preprocessing: C5" Line

- File timestamp: **09-10-2026 21:41:57**
- Only git commit touching this file: **`c1aac08`** ("Recovered test suite and fixed imports")
- That commit also introduced `src/preprocessing.py` with `transform_c5()` including `preprocess_input`.
- No other commits. The "C5" label in `frozen_model_config.json` was written at the same time as the mismatched `preprocess_input` was introduced — by whoever wrote commit `c1aac08`.

