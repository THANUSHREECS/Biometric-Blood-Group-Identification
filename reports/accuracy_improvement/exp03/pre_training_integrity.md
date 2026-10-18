# EXP-03 Pre-Training Integrity Report

**Experiment:** EXP-03 — Controlled Three-Arm Data Augmentation Experiment for E2  
**Timestamp:** 2026-10-14T13:40:00  
**Status:** **PRE-TRAINING VERIFICATION PASSED**  

---

## 1. Safety & Quarantine Verification

| Integrity Requirement | Target Condition | Verification Detail | Status |
|:---|---|---|:---:|
| **Original Checkpoint Preservation** | SHA-256 hashes of original 5 E2 checkpoints | Pre-training hashes recorded and monitored | **PASS** |
| **Locked Test Quarantine** | 21 test participants strictly excluded | Test participant set completely quarantined | **PASS** |
| **Separate Cohort Quarantine** | 26 hospital participants strictly excluded | Not referenced or accessed | **PASS** |
| **Development Cohort Isolation** | Exactly 80 participants (240 scans) | Verified from participant_split.csv | **PASS** |
| **Participant Grouping** | All 3 scans of each participant in 1 fold | Verified via StratifiedGroupKFold | **PASS** |
| **Fold Sharing across Arms** | Identical folds for Arm 0, Arm 1, Arm 2 | Identical generator (seed=42) | **PASS** |

---

## 2. Checkpoint SHA-256 Pre-Training Baseline
- models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5: 7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d
- models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5: 9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede
- models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5: 8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3
- models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5: cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519
- models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5: aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d

---

## 3. Cohort Split Verification
- Total Development Participants: 80 (Expected: 80)
- Total Locked Test Participants: 21 (Expected: 21)
- Participant Overlap between Dev and Test: 0 (Expected: 0)

---

## 4. Training Parameter Lock
- EPOCHS = 150
- MIN_EPOCHS = 100
- PATIENCE = 15
- BATCH_SIZE = 16
- LEARNING_RATE = 1e-3
- WEIGHT_DECAY = 1e-4
- LAMBDA_SUPCON = 0.1
- SAMPLE_WEIGHTS = inverse-frequency N / (4 * N_c)
- METRIC_MONITORED = Validation Macro F1

Pre-training integrity is fully verified. Ready for execution upon confirmation.
