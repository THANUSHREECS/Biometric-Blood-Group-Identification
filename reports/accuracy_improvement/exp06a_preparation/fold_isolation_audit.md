# EXP-06A Fold Isolation & Cohort Quarantine Audit

**Document Version:** 1.0.0  
**Date:** 2026-10-15  
**Audit Scope:** Verification of Cohort Boundaries, Zero-Leakage Grouped Folds, and Data Quarantine for EXP-06A  
**Auditor:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> **RESEARCH USE ONLY — NON-CLINICAL DISCLAIMER**  
> This document pertains to an academic machine learning research investigation into fingerprint biometric feature representations. The data, splits, and models are intended strictly for scientific evaluation and **must not** be used for clinical diagnosis, patient triage, or medical blood group typing.

---

## 1. Cohort Boundary Verification

The project database is partitioned into three strictly segregated cohorts. Prior to authoring the EXP-06A training scripts, programmatic assertions verified the boundaries of each cohort against `reports/phase3_participant_benchmark/participant_split.csv`:

| Cohort Name | Approved Status for EXP-06A | Participant Count | Scan Count | Access Policy | Audit Verification |
|:---|:---:|:---:|:---:|:---|:---:|
| **Development Cohort** | **APPROVED** | **80** | **240** | Used exclusively for 5-fold cross-validation. | **VERIFIED (80 PIDs)** |
| **Locked Test Cohort** | **STRICT QUARANTINE** | **21** | **63** | **Zero access.** Never loaded, trained, or evaluated. | **VERIFIED (0 Access)** |
| **Separate Hospital Cohort** | **STRICT QUARANTINE** | **26** | **78** | **Zero access.** Kept fully isolated in separate storage. | **VERIFIED (0 Access)** |

### Cryptographic Assertion Check:
```python
test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
dev_pids = set(dev_df["participant_id"].unique())
assert len(test_pids) == 21
assert len(dev_pids) == 80
assert len(test_pids & dev_pids) == 0  # PASSED: Zero overlap between dev and test
```

---

## 2. Five-Fold StratifiedGroupKFold Structure

To guarantee that models are evaluated on completely unseen participants during cross-validation, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` was applied using `participant_id` as the grouping factor.

### 2.1 Collocation & Leakage Audit
- **Three-Scan Collocation:** Every participant has exactly 3 scans. In all 5 folds, all 3 scans of a given participant reside strictly in either the training set or the validation set.
- **Zero Participant Leakage:** In all 5 folds, the intersection of training participant IDs and validation participant IDs is strictly empty ($\text{train\_pids} \cap \text{val\_pids} = \emptyset$).
- **Fold Size Balance:** Every fold contains exactly **64 training participants (192 scans)** and **16 validation participants (48 scans)**.

---

## 3. Detailed Fold Composition

The table below catalogs the exact participant assignments, sample counts, and class balance across the 5 development folds:

| Fold Index | Train Participants | Val Participants | Validation Participant IDs | Val Class Distribution |
|:---:|:---:|:---:|:---|:---|
| **Fold 1** | 64 (192 scans) | 16 (48 scans) | `P001`, `P00103`, `P0015`, `P0017`, `P0026`, `P0030`, `P0038`, `P0045`, `P0051`, `P0055`, `P0061`, `P0066`, `P0072`, `P008`, `P0087`, `P0091` | 4 A+, 1 AB+, 5 B+, 6 O+ |
| **Fold 2** | 64 (192 scans) | 16 (48 scans) | `P0010`, `P00104`, `P00106`, `P002`, `P0020`, `P0031`, `P0039`, `P004`, `P0046`, `P0056`, `P0067`, `P0073`, `P0077`, `P0081`, `P0092`, `P0096` | 4 A+, 2 AB+, 5 B+, 5 O+ |
| **Fold 3** | 64 (192 scans) | 16 (48 scans) | `P00100`, `P00105`, `P0018`, `P0021`, `P0027`, `P0032`, `P0040`, `P0047`, `P0052`, `P0057`, `P0062`, `P0068`, `P0074`, `P0078`, `P0082`, `P0093` | 4 A+, 1 AB+, 5 B+, 6 O+ |
| **Fold 4** | 64 (192 scans) | 16 (48 scans) | `P00101`, `P00107`, `P0012`, `P0022`, `P0033`, `P0041`, `P0048`, `P0058`, `P0063`, `P0069`, `P0075`, `P0079`, `P0083`, `P0088`, `P0094`, `P0098` | 3 A+, 2 AB+, 5 B+, 6 O+ |
| **Fold 5** | 64 (192 scans) | 16 (48 scans) | `P00102`, `P0013`, `P0014`, `P0024`, `P0028`, `P0035`, `P0043`, `P005`, `P0053`, `P006`, `P0064`, `P007`, `P0070`, `P0084`, `P0089`, `P0099` | 4 A+, 1 AB+, 5 B+, 6 O+ |
| **Total** | — | **80 (Unique)** | — | **19 A+, 7 AB+, 25 B+, 29 O+** |

---

## 4. Class Weighting Integrity per Fold

Inverse class frequency weights are computed per fold strictly from that fold's 64 training participants:

$$w_{c, f} = \frac{N_{\text{train}, f}}{4 \cdot N_{c, f}}$$

Because participant splits are slightly stratified, the training class counts and resultant weights remain stable across folds:
- **A+ (15 train):** $w_{\text{A+}} \approx 1.067$
- **AB+ (5–6 train):** $w_{\text{AB+}} \approx 2.667 - 3.200$
- **B+ (20 train):** $w_{\text{B+}} = 0.800$
- **O+ (23–24 train):** $w_{\text{O+}} \approx 0.667 - 0.696$

Sample weights are passed as a sample weight tensor to the custom `train_step` and applied strictly to the categorical crossentropy classification loss. The supervised contrastive loss remains unweighted.

---

## 5. Audit Conclusion

1. **Cohort Quarantines 100% Intact:** The locked 21-participant test set and 26-participant hospital set are completely isolated.
2. **Zero Leakage:** Participant grouping ensures no identity overlap between training and validation splits.
3. **Reproducibility Certified:** The split is identical to that used in EXP-03, EXP-04, and EXP-05.
