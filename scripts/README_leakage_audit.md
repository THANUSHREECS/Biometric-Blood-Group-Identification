# Quantitative Leakage Audit: Scripts & Reproduction Index

**Project:** Fingerprint-Based Blood Group Classification (A+, AB+, B+, O+)  
**Audit Status:** Complete & Frozen  
**Audit Date:** October 17, 2026  
**Auditor:** ML Research Engineer (Independent Evaluation Protocol)  
**Output Directories:** `experiments/leakage_audit/`, `reports/leakage_audit/`  
**Freeze Manifest:** `reports/leakage_audit/audit_freeze_manifest.sha256`  

---

## 1. Overview & Scope

This directory contains standalone, reproducible scripts developed for the independent audit of evaluation leakage, selection optimism, and shortcut learning. The audit evaluated two primary vulnerabilities:
1. **Selection Optimism on MFS100 Development Cohort (80 participants):** Evaluating held-out validation fold checkpoint selection against inner validation splits and fixed-epoch procedures, with a 100-repeat selection-inclusive permutation null.
2. **Evaluation Protocol on Original Image-Level Dataset (2,769 positive images):** Auditing near-duplicate percolation, executing a 60-run similarity-purged cross-validation experiment, extracting 42 trivial image properties (dimensions, file size, border pixels, intensity histograms), and assessing shortcut learning.

### Governance Rules Upheld
- **Locked Test Cohort Integrity:** `data/mfs100_final/` contains all 101 participants, of which 80 comprise the development set and 21 constitute the locked test cohort. The audit was strictly read-only on the locked 21-participant test cohort (zero inference or parameter tuning was conducted in this audit; only previously saved predictions were evaluated).
- **Historical Immutability:** No historical file was modified or overwritten by this audit. Pre-existing uncommitted modifications in `app.py`, `src/predictor.py`, `static/js/app.js`, `templates/index.html`, and `scripts/evaluate_model.py` belong to earlier web application work and remain outside the freeze. All new code and outputs reside strictly in `scripts/`, `experiments/leakage_audit/`, and `reports/leakage_audit/`.
- **Pre-Declared Gates & Decision Rules:** Each experiment's decision rule was declared before it ran, but the audit as a whole was sequential and exploratory in response to empirical findings.
- **Run-to-Run Variation:** Reruns will not reproduce results bitwise identically due to non-deterministic GPU/CPU floating-point operations and dropout/initialization differences across seeds (run-to-run variation was observed across seeds, e.g., Seed 42 vs. 43 vs. 44).

---

## 2. Technical Findings & Data Alignments

### 2.1 BMP Header Bit Depth & Alpha Channel Verification
Direct header parsing across `data/raw/` and `data/processed/` confirms:
- **`data/raw/`:** Standard images ($96 \times 103$) have a bit depth of 32 bits per pixel (`biBitCount` = 32, compression `BI_BITFIELDS`, file size 39,690 bytes) containing an explicit binary alpha channel. Exactly 17 non-standard size images ($241 \times 298$, 215,806 bytes) are 24-bit RGB without alpha.
- **`data/processed/`:** All 2,769 images have a bit depth of 24 bits per pixel (`biBitCount` = 24, compression `BI_RGB`, file size 29,718 bytes).
- **Narrowing of Earlier RGBA Statements:** Images in `data/processed/` are standard 24-bit RGB without an alpha channel. The condition `has_alpha = 0` and `alpha_mean = 255.0` for processed images is true **by construction**, because `data/processed/` was produced by compositing raw cutouts onto a solid white background $(255, 255, 255)$ and saving as 24-bit RGB.
- **Group (ii) Implication:** Because the alpha channel is absent and invariant across all processed images, feature group (ii) ("Alpha + Border Statistics") evaluated in Section 6 is effectively **border-only**.
- **Detailed Addendum:** For exact per-class alpha statistics in `data/raw/` and the 100.00% bitwise pixel match with `data/processed/`, refer to [`reports/leakage_audit/raw_bmp_alpha_channel_addendum.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/leakage_audit/raw_bmp_alpha_channel_addendum.md).

### 2.2 Filename Alignment & Class Match Rate
- The cached raw EfficientNet-B0 features (`original_dataset_raw_effb0_features.npy`, $2,769 \times 1,280$) follow the row ordering of `exp1_positive_images_clusters.csv` (manifest order).
- The trivial feature dataset (`exp3_trivial_image_properties_features.csv`, $2,769 \times 45$) was extracted by scanning class directories.
- Merging on image filename (`fname == filename`) yields an **exact class concordance check**:
  - Total aligned images: 2,769 / 2,769.
  - Class match rate: **1.0000 (100.0% match, 0 mismatches)**.
  - This verified alignment is enforced in `leakage_audit_trivial_ablation_and_concat.py`.

---

## 3. Reproduction Workflow (Sequential Execution & Runtimes)

To reproduce the complete audit from scratch in a clean environment, execute the following commands in exact sequential dependency order:

```bash
# Step 1: Experiment 1 - Exact SHA-256 and perceptual hash duplicate audit against Train (~20s measured)
python scripts/leakage_audit_exp1_duplicates_and_hashes.py

# Step 1b: Experiment 1 - Compute & serialize perceptual hash metrics against Train+Val (~25s measured)
python scripts/compute_trainval_reference_hashes.py

# Step 2: Experiment 1 - Extract EfficientNet-B0 embeddings & percolation clustering (~2.5 min measured)
python scripts/leakage_audit_exp1_embeddings_and_clustering.py

# Step 3: Experiment 1 - Finer threshold scan (0.985 to 0.850) on EfficientNet-B0 embeddings (~8s measured)
python scripts/compute_percolation_finer_scan.py

# Step 4: Experiment 2 - Read-only signal-existence permutation & bootstrap tests (~2 min measured)
python scripts/leakage_audit_exp2_signal_tests.py

# Step 5: Phase 2 - Cache frozen DenseNet-121 features for 80 MFS100 dev participants (~1.5 min measured)
python scripts/leakage_audit_cache_dev_features.py

# Step 6: Phase 2 - Evaluate competing selection rules on dev participants across 3 seeds (~8.5 min measured)
python scripts/leakage_audit_phase2_experiments.py

# Step 7: Phase 2 - Run 100-repeat selection-inclusive permutation null (~89 min measured)
python scripts/leakage_audit_phase2_selection_null.py

# Step 8: Phase 3 - Extract raw un-normalized GAP EfficientNet-B0 features for 2,769 images (initial task extraction: 212.4s [~3.5 min measured]; verification rerun: 58.3s [~1.0 min measured])
python scripts/extract_original_dataset_raw_effb0_features.py

# Step 9: Phase 3 - Extract DenseNet-121 features & compute similarity distribution (~3 min measured)
python scripts/leakage_audit_phase3_densenet_similarity.py

# Step 10: Phase 3 - Execute 60-run similarity-purged cross-validation experiment (~57 min measured)
python scripts/leakage_audit_phase3_purged_cv.py

# Step 11: Experiment 3 - Extract 42 trivial image properties (dimensions, border, histograms) (~1.5 min measured)
python scripts/leakage_audit_trivial_image_properties.py

# Step 12: Experiment 3 - Run feature-group ablations and EfficientNet-B0 concat experiment (~5.5 min measured)
python scripts/leakage_audit_trivial_ablation_and_concat.py

# Step 13: Visualization - Generate high-resolution audit figures (Figure 1, Figure 2, Figure 3) (~13s, approximate)
python scripts/generate_leakage_audit_figures.py

# Step 14: Verification - Recompute sample SDs (ddof=1) and empirical p-values directly (~2s, approximate)
python scripts/compute_audit_exact_numbers.py

# Step 15: Concordance - Verify that 81 report markdown numbers match JSON/CSV artifacts (~2s, approximate)
python scripts/verify_leakage_audit_report_concordance.py

# Step 16: Manifest Verification - Verify all 55 artifacts against SHA-256 manifest (~2s, approximate)
python scripts/verify_audit_manifest.py
```
*Note on Runtimes: Durations for Steps 1–12 are derived directly from task log creation and completion timestamps (`.system_generated/tasks/`). Step 8 is reported with both its initial task log duration (212.4s, ~3.5 min) and its verification rerun duration (58.3s, ~1.0 min). Steps 13–16 ran synchronously in interactive terminal sessions and are marked as approximate.*

---

## 4. Scripts Directory Index

### Primary Audit Scripts
| Script | Inputs | Primary Outputs |
| :--- | :--- | :--- |
| `leakage_audit_exp1_duplicates_and_hashes.py` | `outputs/dataset_manifest.csv`<br>`data/processed/` | `experiments/leakage_audit/exp1_duplicates_and_perceptual_hashes.json`<br>`experiments/leakage_audit/exp1_positive_images_hashes.csv` |
| `compute_trainval_reference_hashes.py` | `exp1_positive_images_hashes.csv` | `experiments/leakage_audit/exp1_perceptual_hashes_trainval_reference.json`<br>*(Provenance copy of scratch script)* |
| `leakage_audit_exp1_embeddings_and_clustering.py` | `outputs/dataset_manifest.csv`<br>`data/processed/` | `experiments/leakage_audit/exp1_clustering_summary.json`<br>`experiments/leakage_audit/exp1_positive_images_clusters.csv`<br>`experiments/leakage_audit/exp1_efficientnetb0_embeddings.npy` |
| `compute_percolation_finer_scan.py` | `exp1_efficientnetb0_embeddings.npy` | `experiments/leakage_audit/exp1_percolation_finer_scan.json` |
| `leakage_audit_exp2_signal_tests.py` | `reports/.../config_a_oof_predictions.csv`<br>`reports/.../primary_evaluation_predictions.csv` | `experiments/leakage_audit/exp2_signal_tests_results.json`<br>`experiments/leakage_audit/exp2_permutation_distributions.csv` |
| `leakage_audit_cache_dev_features.py` | `data/mfs100_final/metadata/participants.csv`<br>`data/mfs100_final/raw/` | `experiments/leakage_audit/dev_densenet121_features.npz` |
| `leakage_audit_phase2_experiments.py` | `dev_densenet121_features.npz` | `experiments/leakage_audit/phase2_selection_rules_comparison.json`<br>`experiments/leakage_audit/seed42_rerun_predictions.csv` |
| `leakage_audit_phase2_selection_null.py` | `dev_densenet121_features.npz` | `experiments/leakage_audit/selection_inclusive_null_results.csv`<br>`experiments/leakage_audit/selection_inclusive_null_summary.json` |
| `extract_original_dataset_raw_effb0_features.py` | `outputs/dataset_manifest.csv`<br>`data/processed/` | `experiments/leakage_audit/original_dataset_raw_effb0_features.npy`<br>*(Transcription of original inline command)* |
| `leakage_audit_phase3_densenet_similarity.py` | `outputs/dataset_manifest.csv`<br>`data/processed/` | `experiments/leakage_audit/original_dataset_densenet121_features.npz`<br>`experiments/leakage_audit/phase3_densenet121_similarity_distribution.json` |
| `leakage_audit_phase3_purged_cv.py` | `original_dataset_raw_effb0_features.npy`<br>`original_dataset_densenet121_features.npz` | `experiments/leakage_audit/phase3_purged_cv_results.csv`<br>`experiments/leakage_audit/phase3_purged_cv_summary.json` |
| `leakage_audit_trivial_image_properties.py` | `data/processed/`<br>`data/raw/` | `experiments/leakage_audit/exp3_trivial_image_properties_features.csv`<br>`experiments/leakage_audit/exp3_trivial_image_properties_results.json` |
| `leakage_audit_trivial_ablation_and_concat.py` | `exp3_trivial_image_properties_features.csv`<br>`exp1_positive_images_clusters.csv`<br>`original_dataset_raw_effb0_features.npy` | `experiments/leakage_audit/exp3_trivial_ablation_results.json`<br>`experiments/leakage_audit/exp3_effnet_concat_results.json` |
| `generate_leakage_audit_figures.py` | Audit CSV/JSON outputs | `reports/leakage_audit/figures/figure1_null_distribution_phase2.png`<br>`reports/leakage_audit/figures/figure2_purge_vs_random_phase3.png`<br>`reports/leakage_audit/figures/figure3_similarity_distribution_phase3.png` |
| `compute_audit_exact_numbers.py` | Audit CSV/JSON outputs | Recomputes sample SDs ($\text{ddof}=1$), empirical $p$-values, and summary metrics directly from raw data arrays for mathematical verification. |
| `verify_leakage_audit_report_concordance.py` | `leakage_audit_report.md`<br>Audit CSV/JSON outputs | Programmatically verifies 81 quantitative claims across Sections 3–6 of the markdown report against serialized JSON/CSV artifacts (asserts 81/81 matches, 0 mismatches; exits non-zero on error). |
| `verify_audit_manifest.py` | `audit_freeze_manifest.sha256`<br>Audit artifacts | Reads the freeze manifest, computes SHA-256 hashes of working-tree bytes for every file, prints mismatches, and exits non-zero on any error. |

### Provenance Copies (Historical / Exploratory Scripts)
The following scripts in `scripts/` are copies with provenance headers of original scratch scripts, preserved for full audit provenance (scratch originals are retained):
- `scripts/audit_phase1_readonly.py`: Copy with provenance header of `scratch/audit_phase1_readonly.py` (produces `phase1_readonly_results.json`).
- `scripts/audit_mfs100_metadata_correlations.py`: Copy with provenance header of `scratch/audit_mfs100_metadata_correlations.py` (metadata enrollment order vs. class labels).
- `scripts/audit_collection_date.py`: Copy with provenance header of `scratch/audit_collection_date.py` (collection date and operational metadata checks).
- `scripts/compare_seed42_historical.py`: Copy with provenance header of `scratch/compare_seed42_historical.py` (fold-level comparison of Seed 42 rerun vs. historical OOF).
- `scripts/extract_seed42_participant_preds.py`: Copy with provenance header of `scratch/extract_seed42_participant_preds.py` (extracts participant-level predictions for Seed 42).
- `scripts/audit_search_05085_traceability.py`: Transcription of the repository-wide search command across all `.py`, `.md`, `.json`, `.csv`, `.log`, and `.ipynb` files, verifying that "0.5085" is not generated by any code.

---

## 5. Freezing Declaration

The quantitative leakage audit is officially **frozen**. All experiments, data artifacts in `experiments/leakage_audit/`, scripts in `scripts/`, and reports in `reports/leakage_audit/` are complete, cross-verified, and immutable. A complete SHA-256 manifest of all artifacts is recorded in [`reports/leakage_audit/audit_freeze_manifest.sha256`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/leakage_audit/audit_freeze_manifest.sha256).

### Working-Tree Bytes & Line-Ending Handling
All audit paths (`experiments/leakage_audit/**`, `reports/leakage_audit/**`, `scripts/README_leakage_audit.md`, and all audit scripts) are declared with `-text` in `.gitattributes`. Marking files with `-text` instructs Git to treat them as verbatim binary byte streams, preventing any automatic CRLF/LF line-ending conversion on commit or checkout across all platforms and environments regardless of local `core.autocrlf` configuration. The SHA-256 manifest records the exact bytes of each file on disk, guaranteed invariant across clones.
