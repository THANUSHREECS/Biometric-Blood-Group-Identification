# EXP-04: Fold Isolation & Cohort Quarantine Audit

**Audit Date:** 2026-10-14T18:21:00+05:30  
**Scope:** Verification of Cohort Partitioning, Grouped Folds, Zero-Leakage, and Quarantine Enforcement  
**Audit Finding:** **PASS — 100% ISOLATION VERIFIED**  

---

## 1. Executive Summary

This audit verifies that EXP-04 will utilize the **exact same participant-grouped folds** as EXP-03, guaranteeing complete participant isolation and strict quarantine of external evaluation cohorts.

---

## 2. Cohort Separation & Quarantine Verification

Using `reports/phase3_participant_benchmark/participant_split.csv`:

| Cohort | Assigned Split | Participant Count | Scan Count | Status in EXP-04 |
|:---|:---:|:---:|:---:|:---:|
| **Development Cohort** | `TRAIN` / `VALIDATION` | **80** | **240** (3 per subject) | **ACTIVE** for 5-fold CV |
| **Locked Test Set** | `TEST` | **21** | **63** (3 per subject) | **STRICTLY QUARANTINED** |
| **Hospital Cohort** | External / Separate | **26** | **78** (3 per subject) | **STRICTLY QUARANTINED** |

### Quarantine Assertions:
1. $\text{set}(dev\_pids) \cap \text{set}(test\_pids) = \emptyset$ (Zero participant leakage).
2. All 21 locked test subjects are completely excluded from training, validation, checkpoint selection, and hyperparameter tuning.
3. The separate 26-participant cohort is not referenced or accessed.

---

## 3. Five-Fold Stratified Group Cross-Validation Verification

The cross-validation generator is:
```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

### Partition Properties:
- **Participant Grouping:** All three scans belonging to a participant ($s_1, s_2, s_3$) are grouped by `participant_id`. No participant is ever split across training and validation.
- **Partition Sizes:** In every fold, exactly 64 participants are assigned to training (192 scans) and 16 participants are assigned to validation (48 scans).
- **Leakage Check:** $\text{set}(train\_pids) \cap \text{set}(val\_pids) = \emptyset$ verified across all five folds.

### Per-Fold Participant & Class Distribution:

| Fold | Split | Participant Count | A+ Count | AB+ Count | B+ Count | O+ Count | Overlap Check |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | Train<br>Val | 64<br>16 | 15<br>4 | 6<br>1 | 20<br>5 | 23<br>6 | 0 Overlap (**PASS**) |
| **Fold 2** | Train<br>Val | 64<br>16 | 14<br>5 | 6<br>1 | 23<br>2 | 21<br>8 | 0 Overlap (**PASS**) |
| **Fold 3** | Train<br>Val | 64<br>16 | 17<br>2 | 6<br>1 | 18<br>7 | 23<br>6 | 0 Overlap (**PASS**) |
| **Fold 4** | Train<br>Val | 64<br>16 | 15<br>4 | 6<br>1 | 21<br>4 | 22<br>7 | 0 Overlap (**PASS**) |
| **Fold 5** | Train<br>Val | 64<br>16 | 15<br>4 | 4<br>3 | 18<br>7 | 27<br>2 | 0 Overlap (**PASS**) |

### Key Observations:
1. **Class Representation:** Every training partition contains at least 4 participants of every class, ensuring that inverse-frequency class weights $w_i = \frac{N_{\text{train}}}{4 \cdot N_c}$ never encounter division by zero.
2. **Minority AB+ Distribution:** The 7 AB+ development participants are distributed with 1 subject in Folds 1–4 validation, and 3 subjects in Fold 5 validation.
3. **Out-of-Fold Aggregation:** Because each participant appears in the validation set of exactly one fold, aggregating held-out fold predictions generates exactly 80 true out-of-fold inferences per configuration.

---

## 4. Audit Verdict

Participant grouping, fold balance, cross-validation isolation, and quarantine enforcement are verified.
- **Audit Verdict:** **PASS**.
