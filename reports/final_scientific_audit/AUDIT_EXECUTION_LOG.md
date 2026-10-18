# Audit Execution Log: Final Reproducibility & Scientific Evidence Audit

**Project:** Biometric Blood Group Detection  
**Audit Execution Date:** 2026-10-18  
**Auditor:** Autonomous Verification & Meta-Audit Subsystem  
**Execution Guard:** STRICT READ-ONLY AUDIT (Zero model training, zero parameter tuning, zero cohort quarantine violations)

---

## 1. Commands Executed

| Timestamp (Local) | Command Line | Working Directory | Exit Code | Purpose / Action |
| :--- | :--- | :--- | :---: | :--- |
| `2026-10-18 11:55:36` | `powershell -Command "New-Item -ItemType Directory -Force -Path 'reports/final_scientific_audit'"` | Project Root | `0` | Create final audit report directory. |
| `2026-10-18 11:55:50` | `python -c "import os ... find .csv, .json, .py in experiments"` | Project Root | `0` | Inventory experiment artifacts across `experiments/`. |
| `2026-10-18 11:56:50` | `python scratch/audit_exp10_details.py` | Project Root | `0` | Audit EXP-10 scan quality, consistency, and prediction flips. |
| `2026-10-18 11:57:23` | `python -c "import os ... find *pred*.csv, *oof*.csv"` | Project Root | `0` | Inventory prediction CSV artifacts across workspace. |
| `2026-10-18 11:58:18` | `python -c "import numpy ... check exp07 npy shape"` | Project Root | `0` | Verify EXP-07 out-of-fold prediction probability files. |
| `2026-10-18 11:59:33` | `python scratch/build_audit_data.py` | Project Root | `0` | Preliminary verification loop across 13 experimental configs. |
| `2026-10-18 12:00:40` | `python reports/final_scientific_audit/audit_existing_results.py` | Project Root | `0` | Execute master audit script and generate reconciliation CSVs. |

---

## 2. Artifacts Inspected

### Cohort & Split Files:
- `reports/phase3_participant_benchmark/participant_split.csv` (80 Dev, 21 Test, 240 Dev scans)

### Prediction Artifacts:
- `reports/accuracy_improvement/exp01/e2_focal_oof_predictions.csv`
- `reports/accuracy_improvement/exp02/e2_augmented_oof_predictions.csv`
- `reports/accuracy_improvement/exp03/arm0_oof_predictions.csv`, `arm1_oof_predictions.csv`, `arm2_oof_predictions.csv`
- `reports/accuracy_improvement/exp04/config_a_oof_predictions.csv`, `config_b_oof_predictions.csv`, `config_c_oof_predictions.csv`
- `reports/accuracy_improvement/exp05/config_a_oof_predictions.csv`, `config_b_oof_predictions.csv`, `config_c_oof_predictions.csv`
- `reports/accuracy_improvement/exp06a/config_b_oof_predictions.csv`
- `reports/accuracy_improvement/exp06b/arm_d_oof_predictions.csv`
- `reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv`, `config_b_oof_predictions.csv`
- `reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv`
- `experiments/exp07_resolution_fidelity/checkpoints/arm_a_112/oof_predictions.npy`
- `experiments/exp07_resolution_fidelity/checkpoints/arm_b_224/oof_predictions.npy`
- `experiments/exp08_dermatoglyphic_minutiae/results/exp08_participant_oof_predictions.csv`
- `experiments/exp09_orientation_tensor/exp09_participant_oof_predictions.csv`
- `experiments/exp10_multi_scan_quality/exp10_participant_oof_predictions.csv`
- `experiments/exp10_multi_scan_quality/exp10_scan_quality_audit.csv`

### Evaluation Metrics JSON Artifacts:
- `experiments/exp07_resolution_fidelity/evaluation_results.json`
- `experiments/exp08_dermatoglyphic_minutiae/results/exp08_evaluation_metrics.json`
- `experiments/exp08_dermatoglyphic_minutiae/permutation_test/exp08_full_pipeline_permutation_metrics.json`
- `experiments/exp09_orientation_tensor/exp09_evaluation_metrics.json`
- `experiments/exp10_multi_scan_quality/exp10_evaluation_metrics.json`

### Reports & Reviews:
- `reports/accuracy_improvement/master_experiment_review.md`
- `reports/accuracy_improvement_audit/EXP07_FINAL_RESULTS.md`
- `reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md`
- `reports/accuracy_improvement_audit/EXP08_FINAL_CONSISTENCY_AUDIT.md`
- `reports/accuracy_improvement_audit/NEW_ACCURACY_IMPROVEMENT_INVESTIGATION.md`
- `reports/accuracy_improvement_audit/EXP10_MULTI_SCAN_QUALITY_REPORT.md`

---

## 3. Checks Passed

1. **Cohort Isolation & Quarantine:**
   - 80 Development participants verified ($N=80$, exactly 240 scans).
   - 21 Locked Test participants confirmed quarantined ($0$ access).
   - 26 Hospital participants confirmed quarantined ($0$ access).
   - 898 historical application logs confirmed unreferenced in evaluation.
2. **Participant Uniqueness & Zero Cross-Fold Contamination:**
   - Every audited prediction CSV contained exactly 80 unique participant IDs.
   - Zero duplicated participant rows.
   - All 3 scans of every participant resided strictly within the same outer validation fold.
3. **Metric Recomputation Concordance:**
   - 71 out of 75 evaluated performance metrics ($94.7\%$) reproduced the reported figures exactly within $\pm 0.005$ tolerance.
4. **Permutation Test Integrity (EXP-08):**
   - 1000 full-pipeline permutations refitting StandardScaler, imputer, and Random Forest verified.
   - Finite-sample correction formula $(r+1)/(B+1)$ confirmed ($254/1001 = 0.2537$).
5. **Quality Metric Bounds (EXP-10):**
   - All 240 scans have finite, non-NaN quality scores $Q \in [0.0339, 0.7416]$.
   - Foreground coverage, ridge coherence, ridge contrast, and unusable proportions are strictly bounded within mathematical domains.

---

## 4. Checks Failed / Inconsistencies Documented

1. **EXP-10 Ridge Condition B Markdown Transcription Error (ISSUE-03):**
   - Markdown report stated: `"only a single O+ prediction flipped"` with matrix `[[5, 1, 6, 7], ...]`.
   - Saved predictions and metrics JSON show that **two** predictions flipped (`P0012` A+ to O+, `P007` B+ to O+), resulting in matrix `[[4, 1, 6, 8], ...]`, A+ recall $21.1\%$, and O+ recall $37.9\%$.
2. **EXP-08 Baseline & Logistic Regression Mapping Mismatch (ISSUE-02):**
   - Legacy `train.py` computed fold-local majority (yielding 28.75% acc instead of global 36.25%) and inverted native B+/AB+ labels for Logistic Regression.
   - Corrected in `EXP08_FINAL_RESULTS.md` and reconciled in audit.
3. **Historical Deep CNN Scanner Border Confounder (ISSUE-01):**
   - Platen border luminosity correlated with collection batches ($F=5.996, p=0.0006$). CNN apparent development accuracies were driven by background illumination artifacts rather than friction ridge biology.
4. **Minority Class AB+ Complete Blindness (ISSUE-04):**
   - Every single late-stage participant-level model (EXP-07 through EXP-10) failed to detect any AB+ participants ($0.0\%$ recall).

---

## 5. Audit Limitations

1. **Ground Truth Quality:** Development cohort blood group labels are self-reported by student participants, with no clinical laboratory verification records.
2. **Permutation Test Resolution in EXP-09:** EXP-09 executed 100 permutations, providing an empirical p-value resolution of $\approx 0.01$.
3. **Quarantine Adherence:** In accordance with strict audit rules, locked test and hospital cohorts were not accessed or evaluated.
