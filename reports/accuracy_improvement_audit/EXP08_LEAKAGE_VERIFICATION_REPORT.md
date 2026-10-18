# EXP-08: Cross-Validation Leakage & Protocol Verification Report

**Document ID:** EXP08_LEAKAGE_VERIFICATION_REPORT  
**Date:** 2026-10-18  
**Experiment:** EXP-08-DERMATOGLYPHIC-MINUTIAE  
**Protocol:** 5-Fold StratifiedGroupKFold on Dev-80 Cohort  

---

## 1. Executive Summary

This report verifies that EXP-08 adheres to zero-leakage cross-validation protocols. All multi-scan aggregation, training-validation partitions, preprocessing transformations, and evaluation metrics were audited for cross-fold contamination, cohort leakage, and feature snooping.

| Leakage Dimension | Protocol Specification | Verified Empirical Behavior | Status |
| :--- | :--- | :--- | :---: |
| **Instance Duplication** | 1 row per participant | 240 scans aggregated to exactly 80 participant rows | **PASSED** |
| **Cohort Quarantine** | 0 test/hospital participants | Zero overlap with 21 locked test or 26 hospital participants | **PASSED** |
| **Participant Grouping** | Grouped by `participant_id` | All 3 scans remain together in the same fold | **PASSED** |
| **Fold Partition** | Mutually exclusive splits | Each fold contains exactly 16 distinct validation participants | **PASSED** |
| **Transformation Fitting** | Strict in-fold fitting | `SimpleImputer` and `StandardScaler` fit strictly on `X_train` | **PASSED** |
| **Hyperparameter Selection** | Predefined parameters | No tuning performed on validation splits | **PASSED** |
| **Out-of-Fold Coverage** | 100% complete partition | Exactly 80 out-of-fold predictions (1 per participant) | **PASSED** |

---

## 2. Participant Grouping & Fold Partitioning

Cross-validation was executed using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `participant_id`.

### Empirical Fold Distribution:

| Fold Index | Train Participants | Val Participants | Val Participant IDs Sample | Val Blood Group Counts (A+ / B+ / AB+ / O+) |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 64 | 16 | P001, P0010, P00100, P00101, ... | 4 / 5 / 1 / 6 |
| **Fold 2** | 64 | 16 | P00103, P00107, P00108, P00114, ... | 5 / 2 / 1 / 8 |
| **Fold 3** | 64 | 16 | P00104, P00105, P00110, P00111, ... | 2 / 7 / 1 / 6 |
| **Fold 4** | 64 | 16 | P00102, P00109, P00112, P00113, ... | 4 / 4 / 1 / 7 |
| **Fold 5** | 64 | 16 | P00106, P00115, P00116, P00117, ... | 4 / 7 / 3 / 2 |
| **Total** | — | **80** | **Complete Partition** | **19 / 25 / 7 / 29** |

### Assertions Executed & Verified:
```python
# For each fold:
assert len(val_pids.intersection(train_pids)) == 0  # 0 overlap between train and val
assert len(val_pids.intersection(other_val_pids)) == 0  # 0 overlap between validation folds
assert len(df_oof) == 80  # Exactly 1 prediction per participant
```

---

## 3. Preprocessing & Imputation Leakage Safeguards

Learned transformations were encapsulated within scikit-learn `Pipeline` objects:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", classifier)
])
```

- **Median Imputation:** Medians were computed strictly on the 64 training fold participants (`X_train`) and transformed onto validation fold participants (`X_val`).
- **Feature Scaling:** StandardScaler means and variances were computed strictly on `X_train`.
- **Zero Validation Snooping:** At no point did validation fold features or labels influence imputer medians or scaling parameters.

---

## 4. Cohort Isolation & Quarantine Verification

The 80-participant development cohort was validated against external data manifests:
1. **Locked Test Cohort ($N=21$ Participants, 63 Scans):**
   - Disjoint intersection assertion: `len(set(dev_pids).intersection(test_pids)) == 0`.
   - Verified: **0 locked test participants accessed or processed.**
2. **Hospital Cohort ($N=26$ Participants, 78 Scans):**
   - Path assertion: `"hospital" not in scan_path.lower()`.
   - Verified: **0 hospital scans accessed or processed.**

---

## 5. Canonical Class Remapping Integrity

- Native class ordering: `[0: "A+", 1: "B+", 2: "AB+", 3: "O+"]` (historical directory index).
- Canonical reporting order: `[0: "A+", 1: "AB+", 2: "B+", 3: "O+"]`.
- Remapping vector: `canon_map = [0, 2, 1, 3]`.
- Both native and canonical confusion matrices are computed and stored.
