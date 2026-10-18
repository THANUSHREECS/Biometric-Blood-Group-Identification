# EXP-06D: Data & Checkpoint Safety Verification

**Document Version:** 1.0.0  
**Preparation Date:** 2026-10-16  
**Auditor:** Senior ML Verification Specialist & Independent Data Auditor  
**Experiment Reference:** EXP-06D (Preprocessing & Multi-Scan Representation Investigation)  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, configurations, data manifests, and safety checklists documented herein are academic research protocols designed to study representation learning on optical biometric fingerprints. They **must not** be used for clinical decision-making, patient transfusion matching, blood typing, triage, or clinical care.

---

## 1. Safety Objectives & Scope

This document establishes the mandatory data isolation, cohort quarantine, and checkpoint protection protocols for **EXP-06D**. It verifies that:
1. The development cohort is strictly confined to the 80 approved participants with zero inter-fold leakage.
2. The 21-participant locked test cohort and 26-participant hospital cohort are completely quarantined with zero access.
3. All 66 pre-existing model checkpoints are cataloged with cryptographic SHA-256 hashes and protected against modification.
4. A dedicated, isolated directory structure is allocated for EXP-06D.

---

## 2. Cohort Isolation & Quarantine Audit

### 2.1 Development Cohort Verification ($N = 80$)
- **Source Manifest:** `reports/phase3_participant_benchmark/participant_split.csv`
- **Eligibility Criterion:** Rows where `split == 'TRAIN'` or `split == 'VALIDATION'`.
- **Participant Count:** Exactly **80 unique participants** (240 scans; 3 scans per participant).
- **Class Distribution:**
  - `A+`: 19 participants (57 scans, $23.75\%$)
  - `AB+`: 7 participants (21 scans, $8.75\%$) — **Rare Class Safeguard**
  - `B+`: 25 participants (75 scans, $31.25\%$)
  - `O+`: 29 participants (87 scans, $36.25\%$)
- **Participant Grouping & Collocation:**
  - All three scans belonging to participant $p$ are grouped together under `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  - Under no circumstances will scans from the same participant be split between training and validation within any fold.
  - Inter-fold leakage check: $\text{Train}_{\text{Fold } k} \cap \text{Val}_{\text{Fold } k} = \emptyset$ for all $k \in \{1, 2, 3, 4, 5\}$.
- **Canonical Class Ordering:** Strictly enforced as `['A+', 'AB+', 'B+', 'O+']`.

---

### 2.2 Locked Test Cohort Quarantine ($N = 21$)
- **Source Manifest:** Rows where `split == 'TEST'` in `reports/phase3_participant_benchmark/participant_split.csv`.
- **Participant Count:** Exactly **21 unique participants** (63 scans).
- **Participant IDs:**
  `P002, P0023, P0024, P0029, P003, P0035, P0039, P0041, P0042, P0047, P0048, P0054, P0071, P0073, P0075, P0082, P0083, P0088, P009, P0093, P0095`.
- **Strict Quarantine Rules:**
  1. No script in EXP-06D may load, read, or parse these 21 participant IDs.
  2. No test metrics will be computed during development, training, tuning, or model selection.
  3. No hyperparameter or threshold adjustment may be influenced by test cohort performance.
  4. Preflight data loading assertions must explicitly reject any scan originating from these 21 participants.

---

### 2.3 Hospital Cohort Quarantine ($N = 26$)
- **Physical Location:** Segregated under `data/captured/` (and documented in `reports/accuracy_improvement/dataset_audit.md`).
- **Participant Count:** Exactly **26 unique participants** (78 scans).
- **Cohort Nature:** Live optical captures collected at an external hospital facility with physical serological ground truth.
- **Strict Quarantine Rules:**
  1. The hospital dataset directory is strictly excluded from all training, validation, and evaluation paths.
  2. The hospital cohort will not be used for model selection, thresholding, or exploratory testing.
  3. Preflight scripts must verify zero path resolution into `data/captured/`.

---

## 3. Checkpoint Protection & Cryptographic Hash Registry

Before any EXP-06D code is prepared, all **66 existing model checkpoints** across the project were verified and registered in `scratch/exp06d_pre_preparation_hashes.json`.

```
========================================================================================
CHECKPOINT PROTECTION REGISTRY (66 TOTAL MODELS PROTECTED)
========================================================================================
Checkpoint Group                   Directory Path                                   Count
----------------------------------------------------------------------------------------
Production Baseline Model          models/abo_positive_4class/best_model.keras        1
Original E2 Baseline Models        models/phase3_fusion_metric_learning/E2/           5
EXP-03 Arm 1 Baseline Models       experiments/exp03/checkpoints/arm1/                5
EXP-05 Config A Models             experiments/exp05/checkpoints/config_a/            5
EXP-06A Configs A–D Models         experiments/exp06a/checkpoints/config_[a-d]/      20
EXP-06B Arms A–D Models            experiments/exp06b/checkpoints/arm_[a-d]/         20
EXP-06C Config A Models (Linear)   experiments/exp06c/checkpoints/config_a/           5
EXP-06C Config B Models (MLP)      experiments/exp06c/checkpoints/config_b/           5
----------------------------------------------------------------------------------------
Total Protected Models                                                               66
========================================================================================
```

### 3.1 Sample Checkpoint Hashes
- **Production Model (`best_model.keras`):**
  `7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9` (32,642,427 B)
- **Original E2 Fold 1 (`fold_1_best.weights.h5`):**
  `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` (52,910,112 B)
- **EXP-06C Config A Fold 1 (`fold_1_best.weights.h5`):**
  `ba4917ce70382e339d1b712b8fffa2b8344e29eb0e0fca5664166a5e1cfdd6f5` (52,899,216 B)
- **EXP-06C Config B Fold 1 (`fold_1_best.weights.h5`):**
  `e2d747abc0dc554f7620bc261c1626ce8309dfd707299a9a3ea427974ce24fe5` (67,852,944 B)

### 3.2 Immutability Guarantee
All 66 files are locked. Any future execution of EXP-06D must verify pre- and post-execution SHA-256 hashes against `scratch/exp06d_pre_preparation_hashes.json`. A single bit mismatch will trigger an immediate fail-closed abort.

---

## 4. Directory Structure & Storage Isolation

To eliminate any risk of overwriting historical models or contaminating existing evaluation reports, all EXP-06D files are strictly quarantined in dedicated directories:

```
major project/
├── experiments/
│   └── exp06d/                               # Isolated EXP-06D code & checkpoints
│       ├── run_exp06d.py                     # Execution skeleton (training disabled)
│       └── checkpoints/                      # Dedicated checkpoint directory
│           ├── config_a/                     # Config A weights (5 folds)
│           └── config_b/                     # Config B weights (5 folds)
│
└── reports/
    └── accuracy_improvement/
        ├── exp06d_preparation/              # Pre-training audit & safety package
        │   ├── repository_audit.md
        │   ├── experiment_protocol.md
        │   ├── data_and_checkpoint_safety.md
        │   ├── synthetic_test_report.md
        │   └── preparation_audit.md
        │
        └── exp06d/                          # Post-training reports (empty until authorized)
            ├── metrics_comparison.csv
            ├── experiment_config.json
            ├── statistical_analysis_results.json
            ├── final_report.md
            ├── config_a_oof_predictions.csv
            ├── config_b_oof_predictions.csv
            └── confusion_matrices/
```

Under no circumstances will any file be written to `experiments/exp06c/`, `experiments/exp06b/`, `models/phase3_fusion_metric_learning/`, or `models/abo_positive_4class/`.

---

## 5. Fail-Closed Training Guard Specification

In strict adherence to the authorization boundary:
1. `experiments/exp06d/run_exp06d.py` must contain a hardcoded, fail-closed guard:
   ```python
   TRAINING_AUTHORIZED = False  # MUST REMAIN FALSE DURING PREPARATION
   ```
2. If `run_exp06d.py` is invoked without the explicit CLI flag `--authorize-training`, or if `TRAINING_AUTHORIZED is False`, execution must abort immediately with a safety exit code and descriptive refusal message.
3. Import-Time Safety: Importing `run_exp06d.py` into test suites or verification scripts must NEVER trigger dataset loading, directory creation, or model training. All execution logic must reside strictly inside guarded functions or `if __name__ == '__main__':`.

---

## 6. Safety Verification Sign-off

- **Development Cohort Isolation:** Confirmed (80 participants, 240 scans).
- **Test Cohort Quarantine:** Confirmed (21 participants, 63 scans locked).
- **Hospital Cohort Quarantine:** Confirmed (26 participants, 78 scans locked).
- **Protected Checkpoints Cataloged:** Confirmed (66 models registered in `exp06d_pre_preparation_hashes.json`).
- **Fail-Closed Guard:** Specified and required.

*Safety document compiled autonomously following strict research compliance standards.*
