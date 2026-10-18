# Repository Inventory & Artifact Verification

**Audit Date:** 2026-10-17  
**Scope:** Forensic inventory of all datasets, models, code scripts, results files, and scientific audits across the Biometric Bloodgroup Detection project.

---

## 1. Executive Summary of Artifact Integrity

Across the codebase, 145 discrete result entries, 17 locked-cohort evaluation runs, multiple model checkpoints, and multiple cohorts were inventoried. The repository exhibits a sharp partition between:
1. **Historical legacy experiments (Pre-October 2026):** High reported scores (up to 86.02%) evaluated without participant-level separation on an unverified 6,000-image dataset.
2. **Controlled development experiments (EXP-01 to EXP-06D):** Three-scan feature fusion using DenseNet-121 on the primary MFS100 101-participant cohort, reporting 46%–54% validation accuracy, but contaminated by checkpoint-selection leakage.
3. **Locked-cohort evaluation and scientific audits:** Definitive evidence that locked-cohort test accuracy is 33.33% (7/21 correct, Macro F1 0.2083), below the 38.10% (8/21) majority-class baseline.

---

## 2. File-by-File Inventory Matrix

| File Path | Contents & Purpose | Exists | Reproducible | Claims Supported | Discrepancies & Limitations |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `reports/final_evaluation/post_evaluation_scientific_audit.md` | Comprehensive 5-part forensic scientific audit (Audits A–E). | Yes | Yes | Unifies dataset leakage, serology unreliability, and failure on locked cohort. | Definitive ground truth audit document. |
| `reports/final_evaluation/final_scientific_assessment.md` | Final evaluation report and biological feasibility critique. | Yes | Yes | Confirms 33.33% locked-cohort performance; establishes non-feasibility of clinical claims. | None. |
| `reports/final_evaluation/primary_evaluation_final_report.md` | Primary test report for EXP-06C Config A on 21-participant locked cohort. | Yes | Yes | Reports 33.33% (7/21) accuracy, 0.2083 macro F1, balanced acc 31.25%. | Notes that Day 27 baseline had achieved 38.10% (8/21). |
| `reports/final_evaluation/primary_evaluation_predictions.csv` | Raw scan-level and participant-level model predictions for the 21 test participants. | Yes | Yes | Ground truth predictions: 7 correct out of 21. | Verified against checkpoint outputs. |
| `reports/final_evaluation/primary_evaluation_results.json` | Machine-readable metrics for final locked evaluation. | Yes | Yes | Exact test metrics: accuracy 0.3333, macro_f1 0.2083, cohen_kappa 0.0870. | Permutation p-values: 0.5641 (accuracy), 0.3712 (macro F1). |
| `reports/leakage_audit/leakage_audit_report.md` | Phase 1–3 leakage investigation: selection optimism, similarity purging, and shortcut features. | Yes | Yes | Proves checkpoint selection leakage explains high validation scores (shuffled labels reach 46.77%). | Uncovers 42 image-level shortcut features in legacy dataset. |
| `reports/project_report/results_inventory.csv` | 145-row inventory of every reported metric across all project phases. | Yes | Yes | Complete census of historical and audit figures. | Reconciled into 124 model results and 21 audit diagnostics. |
| `reports/project_report/model_results.csv` | 124 verified model training and evaluation runs. | Yes | Yes | Tracks performance progression from Custom CNN to EXP-06D. | Discrepancy noted between Day-27 reported (38.09%) and zero-shot table (33.33%). |
| `reports/project_report/audit_diagnostics.csv` | 21 diagnostic runs (permutation tests, hash checks, percolation analysis). | Yes | Yes | Validates null distributions and dataset duplicate checks. | Explicitly separates diagnostics from predictive models. |
| `reports/project_report/report_corrections.md` | Reconciliations and errata for historical project reports. | Yes | Yes | Corrects statistical claims, clarifies baseline behaviors, documents lineage. | Resolves earlier ambiguous claims regarding majority voting and E2 behavior. |
| `reports/project_report/audit_erratum_locked_cohort.md` | Forensic audit of locked-cohort exposure (17 prediction artifacts across 10 evaluation dates). | Yes | Yes | Proves the 21-participant cohort was repeatedly probed before final model selection. | Disproves claims that locked test set was completely untouched. |
| `data/processed/mfs100_split_manifest.json` | Participant-level split for primary MFS100 cohort: 64 train, 16 val, 21 test. | Yes | Yes | Establishes participant-level firewall for MFS100 experiments. | 100% self-reported labels; zero laboratory verification. |
| `data/hospital_verified/manifest.json` | Inventory of 26 hospital participants (78 scans), 25 serology verified. | Yes | Quarantined | Independent cohort with serological ground truth. | Quarantined: cross-cohort participant identity overlap with MFS100 is unverified. |
| `experiments/exp06c/run_exp06c.py` | Training script for EXP-06C Config A (linear head) and Config B (MLP head). | Yes | Yes | Defines three-scan DenseNet-121 fusion architecture. | `MacroF1Checkpoint` selects best weights on outer held-out fold (selection leakage). |
| `experiments/exp06c/checkpoints/config_a/` | Saved weights for Config A across 5 cross-validation folds. | Yes | Yes | Re-evaluable weights; inputs shape (1, 3, 112, 112, 3). | Trainable params: 1,968,772; frozen backbone: 7,037,504. |
| `experiments/exp06d/run_exp06d.py` | Preprocessing ablation script (Enhanced C5 vs Raw C5). | Yes | Yes | Measures impact of CLAHE and bilateral filtering. | Removal of CLAHE/bilateral improved OOF macro F1 (+0.0622), proving filtering was destructive. |
| `models/abo_positive_4class/best_model.keras` | Historical single-image EfficientNet-B0 checkpoint on legacy dataset. | Yes | Yes | Achieved historical 86.02% 4-class accuracy. | Zero-shot on MFS100 locked test: 19.05% to 28.57% (below chance). |

---

## 3. Discrepancy & Conflict Analysis

1. **Day-27 Baseline Score Conflict:**
   - In `reports/final_evaluation/primary_evaluation_final_report.md`, Day-27 baseline is cited as 38.09% (8/21).
   - In zero-shot prediction logs, it appears as 33.33% (7/21, Macro F1 0.1875).
   - *Resolution:* 38.09% represents the participant-level majority voting output (8/21), whereas 33.33% represents raw single-scan inference without participant aggregation.
2. **EfficientNet-B0 Locked-Test Discrepancy:**
   - Architecture benchmark table reports 0.1905 (4/21, Macro F1 0.1250).
   - `locked_test_metrics.json` records 0.2857 (6/21, Macro F1 0.1587).
   - *Resolution:* Caused by differing preprocessing pipelines (raw resize vs C5 enhancement) applied during distinct benchmark runs.
3. **Checkpoint Path Ambiguity:**
   - Historical documentation alternately claimed that `models/abo_positive_4class/best_model.keras` belonged to ResNet-50 and EfficientNet-B0.
   - *Resolution:* Inspection of the Keras config confirmed the file contains an EfficientNet-B0 architecture trained on the legacy 4-class dataset.
