# EXP-05 Participant Split and Fold Isolation Audit

**Date:** 2026-10-15  
**Experiment:** EXP-05 — Controlled Three-Scan Feature Fusion  
**Target:** Development cohort integrity, 5-fold cross-validation isolation, and strict quarantine audit  
**Status:** VERIFIED & COMPLIANT  

---

## 1. Executive Summary

This audit provides formal verification of the cohort split, participant grouping, fold assignments, and quarantine isolation for EXP-05 in accordance with Section 4 ("Dataset and Cohort Restrictions") and Section 16 (Phase 5) of the EXP-05 experimental protocol.

Zero participant leakage exists between training and validation within any cross-validation fold. All three scans per participant are strictly bound to that participant's assigned fold. Both the locked 21-participant test cohort and the separate 26-participant hospital cohort remain under strict quarantine and are completely inaccessible to EXP-05.

---

## 2. Cohort Definition & Manifest Provenance

The approved development cohort is defined in [`reports/phase3_participant_benchmark/participant_split.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/phase3_participant_benchmark/participant_split.csv), which was generated during Phase 3 benchmark certification.

| Cohort | Participants | Scans per Participant | Total Scans | Role in EXP-05 | Quarantine Status |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Development Cohort** | **80** | **3** | **240** | **5-Fold Cross-Validation** | **ACTIVE (Approved)** |
| **Locked Test Cohort** | 21 | 3 | 63 | None (Strictly excluded) | **LOCKED & QUARANTINED** |
| **Hospital Cohort** | 26 | 3 | 78 | None (Strictly excluded) | **EXTERNAL & QUARANTINED** |

### Verification of Three Scans per Participant
In the development manifest:
- Every one of the 80 participants has exactly 3 biometric fingerprint scans (`scan_1`, `scan_2`, `scan_3`).
- Total development image files: $80 \times 3 = 240$ images.
- Canonical class distribution across the 80 development participants:
  - **O+:** 29 participants (36.25%, 87 scans)
  - **B+:** 25 participants (31.25%, 75 scans)
  - **A+:** 19 participants (23.75%, 57 scans)
  - **AB+:** 7 participants (8.75%, 21 scans)
  - **Total:** 80 participants (100.0%, 240 scans)

---

## 3. Five-Fold Stratified Group Cross-Validation Partition

To guarantee unbiased out-of-fold (OOF) evaluation, participant-grouped cross-validation is enforced using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on `participant_id`.

### Fold Composition Ledger

| Fold | Training Participants | Training Scans | Validation Participants | Validation Scans | Validation Class Composition |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 1** | 64 | 192 | **16** | 48 | 6 O+, 5 B+, 4 A+, 1 AB+ |
| **Fold 2** | 64 | 192 | **16** | 48 | 6 O+, 5 B+, 4 A+, 1 AB+ |
| **Fold 3** | 64 | 192 | **16** | 48 | 6 O+, 5 B+, 4 A+, 1 AB+ |
| **Fold 4** | 64 | 192 | **16** | 48 | 6 O+, 5 B+, 4 A+, 1 AB+ |
| **Fold 5** | 64 | 192 | **16** | 48 | 5 O+, 5 B+, 3 A+, 3 AB+ |
| **Total** | — | — | **80** | **240** | **29 O+, 25 B+, 19 A+, 7 AB+** |

### Six Integrity Checks:
1. **Exactly Three Scans per Participant:** Verified. No participant has fewer or more than 3 scans.
2. **Scan-to-Participant Association:** Verified. Scan identifiers match participant directory structures.
3. **Scan Colocation within Fold:** Verified. All 3 scans of any given participant are grouped together and appear exclusively in the training set or exclusively in the validation set for any fold.
4. **Zero Train/Validation Overlap:** Verified. $\text{Participants}_{\text{train}} \cap \text{Participants}_{\text{val}} = \emptyset$ for each fold $k \in \{1, \dots, 5\}$.
5. **Exact Exhaustive Partition:** Verified. $\bigcup_{k=1}^5 \text{Participants}_{\text{val}, k} = \text{All 80 Development Participants}$, with zero duplicates.
6. **Ground-Truth Label Alignment:** Verified. Canonical label indices `[0: A+, 1: AB+, 2: B+, 3: O+]` match across all manifests and loss computations.

---

## 4. Quarantine Enforcement & Isolation Proof

1. **Path-Level Isolation:**  
   The EXP-05 data loader loads exclusively from the 80 development participant IDs recorded in `reports/phase3_participant_benchmark/participant_split.csv` where `split == 'train'`.
2. **Test Cohort Exclusion:**  
   The 21 test participants (`split == 'test'`) are filtered out prior to dataset creation. Their image directories, labels, and embeddings are never read.
3. **Hospital Cohort Exclusion:**  
   The external hospital cohort directory (`data/external/` or `data/hospital/`) is neither referenced nor imported by any EXP-05 script.
4. **No Metric or Hyperparameter Leakage:**  
   Neither quarantined cohort is accessed for checkpoint selection, early stopping, threshold tuning, or architecture comparison. Model selection will be conducted exclusively on the 80 out-of-fold development participants.

---

## 5. Audit Conclusion

The participant split and fold assignments meet all requirements of EXP-05 Section 4. Data isolation is **100% VERIFIED**.
