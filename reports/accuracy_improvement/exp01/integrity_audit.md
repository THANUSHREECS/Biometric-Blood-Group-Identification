# EXP-01 Phase 10: Integrity Audit Report

**Experiment:** EXP-01 — Class-Balanced Focal Loss for E2  
**Date:** 2026-10-14T09:57:05.557563  
**Scope:** Verification of Training Isolation, Manifest Integrity, and Checkpoint Preservation  

---

## 1. Audit Verification Matrix

| # | Integrity Check | Status | Verification Details |
|---|---|:---:|---|
| **1** | **Participant Count & Uniqueness** | **PASS** | Exactly 80 rows in `e2_focal_oof_predictions.csv`, matching the 80 development participants with 0 duplicates. |
| **2** | **Ground-Truth Label Fidelity** | **PASS** | All 80 participant labels match `participant_split.csv` exactly. |
| **3** | **Fold Isolation & OOF Integrity** | **PASS** | In every fold, `set(train_participants)` and `set(val_participants)` have an empty intersection. All 80 predictions are genuine out-of-fold inferences. |
| **4** | **Probability Vector Sanity** | **PASS** | All values are finite, non-negative ($p \ge 0.0$), and sum to $1.000000 \pm 10^{-4}$. |
| **5** | **Class Order Alignment** | **PASS** | Canonical remapping verified: column 0 = A+, 1 = AB+, 2 = B+, 3 = O+. Argmax matches `predicted_label`. |
| **6** | **Independent Metric Recalculation** | **PASS** | Metrics computed from CSV match `metrics.json` to within $10^{-6}$. |
| **7** | **Original Checkpoint Preservation** | **PASS** | All five original E2 checkpoints in `models/phase3_fusion_metric_learning/E2/` have identical SHA-256 hashes before and after training. |
| **8** | **Zero Locked-Test Contamination** | **PASS** | All 21 locked test participants remained untouched and were never evaluated or accessed. |
| **9** | **Zero 26-Cohort Contamination** | **PASS** | The separate 26-cohort was completely excluded from training, validation, and model selection. |

---

## 2. Original E2 Checkpoint Hash Verification

| Checkpoint | Pre-Experiment SHA-256 | Post-Experiment SHA-256 | Status |
|:---|:---:|:---:|:---:|
| **Fold 1** | `7d9464d18619a94f...` | `7d9464d18619a94f...` | **UNCHANGED** |
| **Fold 2** | `9c72809f23152708...` | `9c72809f23152708...` | **UNCHANGED** |
| **Fold 3** | `8123b1b86f44719d...` | `8123b1b86f44719d...` | **UNCHANGED** |
| **Fold 4** | `cc122b23e0c6d118...` | `cc122b23e0c6d118...` | **UNCHANGED** |
| **Fold 5** | `aff9e137bed61f64...` | `aff9e137bed61f64...` | **UNCHANGED** |

**Conclusion:** All Phase 10 integrity checks have **PASSED**.
