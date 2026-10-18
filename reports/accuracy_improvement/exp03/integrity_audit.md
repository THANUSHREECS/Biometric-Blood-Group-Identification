# EXP-03 Post-Training Integrity Audit Report

**Date:** 2026-10-14T17:48:15.392262  
**Scope:** Verification of Cohort Quarantines, Original Checkpoint Immutability, and Prediction Fidelity  

---

## 1. Safety & Quarantine Checks

| Check | Requirement | Result | Status |
|:---|---|---|:---:|
| **Original Checkpoints Preserved** | SHA-256 hashes must match baseline | Checkpoints identical | **PASS** |
| **Locked Test Quarantine** | 21 test participants never loaded/evaluated | 0 test participants accessed | **PASS** |
| **26-Hospital Cohort Quarantine** | 26 hospital participants never loaded/evaluated | 0 cohort participants accessed | **PASS** |
| **Development Cohort Isolation** | Exactly 80 development participants | 80 participants evaluated per arm | **PASS** |
| **Zero Participant Leakage** | train_pids intersection val_pids == empty set | 0 overlap across all 5 folds | **PASS** |
| **Canonical Class Remapping** | Output order matches [A+, AB+, B+, O+] | Verified remapping function applied | **PASS** |

---

## 2. Original Checkpoint Hashes
- Fold 1: `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` (MATCH)
- Fold 2: `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` (MATCH)
- Fold 3: `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` (MATCH)
- Fold 4: `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` (MATCH)
- Fold 5: `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` (MATCH)
