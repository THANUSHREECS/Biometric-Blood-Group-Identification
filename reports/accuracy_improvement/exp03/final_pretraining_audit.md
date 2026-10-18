# EXP-03: Final Pre-Training Code Audit & Verification Report

**Experiment:** EXP-03 — Controlled Three-Arm Data Augmentation Experiment for E2  
**Audit Timestamp:** 2026-10-14T13:48:00+05:30  
**Status:** **AUDIT PASSED — READY FOR AUTHORIZED EXECUTION**  
**Quarantine Enforcement:** Active (Locked 21-participant test set & 26-participant cohort completely quarantined)  
**Training Status:** **NO TRAINING PERFORMED**  

---

## 1. Audit Verification Matrix (PASS / FAIL Table)

| Step | Audit Dimension | Evaluated Requirement | Result | Evidence / Details |
|:---:|:---|---|:---:|---|
| **1** | **Script Completeness & Syntax** | `experiments/exp03/run_exp03.py` is syntactically valid, complete, and runnable | **PASS** | `py_compile` succeeded with code 0; CLI `--help` verified with full argument parsing. |
| **2** | **Model Architecture & Tensor Shapes** | Frozen DenseNet-121 3-scan feature fusion with GAP and Concat(3072) | **PASS** | Validated tensor shapes: Input `(None, 3, 112, 112, 3)`, Output `clf: (None, 4)`, `emb: (None, 128)` for all 3 arms. |
| **2** | **Supervised Contrastive Loss** | Positive/negative handling, zero positive pair behavior, gradient finiteness | **PASS** | L2-normalized embeddings, $\tau=0.07$; returns $0.0$ loss with finite gradients when no positive pairs exist; identical to `experiment6_fusion_supcon.py`. |
| **2** | **Sample Weighting Protocol** | Inverse-frequency weights applied only to `clf` branch; unweighted for `emb` | **PASS** | `sample_weight=[sw, np.ones_like(sw)]` matches original E2 baseline exactly. |
| **3** | **Development Cohort Isolation** | Exactly 80 participants, 3 scans each (240 scans total) | **PASS** | Verified from `participant_split.csv`; 240 valid preprocessed images loaded with zero missing files. |
| **3** | **Quarantine of External Cohorts** | 21 locked test subjects and 26-cohort excluded | **PASS** | Intersection between dev participants and locked test participants is exactly 0. Neither quarantined cohort is loaded or evaluated. |
| **3** | **Cross-Validation Integrity** | 5-Fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | **PASS** | Exactly 64 train and 16 validation participants per fold; 0 participant overlap across folds; each participant gets exactly 1 OOF prediction. |
| **3** | **Class Representation in Folds** | All 4 classes represented in training partitions | **PASS** | Every fold contains $N_c \ge 4$ for all classes (including minority AB+ which has 6 in Folds 1-4, 4 in Fold 5). |
| **4** | **Class Order & Probability Mapping** | Native `[A+, B+, AB+, O+]` remapped to Canonical `[A+, AB+, B+, O+]` | **PASS** | Automated unit tests verified all 4 one-hot vectors map to expected canonical labels. |
| **5** | **Three Experimental Arms** | Arm 0 (Original E2), Arm 1 (No Augmentation), Arm 2 (Conservative) | **PASS** | Arm 0: Rot 0.027, Trans 0.1, Zoom 0.1, Cont 0.1.<br>Arm 1: Identity pass-through.<br>Arm 2: Rot $\pm 0.02$, Trans $\pm 0.04$, Cont 0.05, Bright 0.05, Zoom 0.0. |
| **5** | **Augmentation Inactive in Eval** | Augmentation layers pass inputs unchanged when `training=False` | **PASS** | Max numerical difference between input and output during evaluation is $0.000000$. |
| **6** | **Original Checkpoint Preservation** | Hashes of original 5 E2 checkpoints remain untouched | **PASS** | All 5 SHA-256 hashes matched pre-training baselines with 0 bit alterations. |
| **7** | **Reproducibility & Seed Governance** | Fixed random seeds across Python, NumPy, TensorFlow, Keras layers | **PASS** | Global `SEED = 42` and per-fold deterministic seed offsets applied. |
| **8** | **Checkpoint Selection & Early Stopping** | Monitored by Validation Macro F1; `MIN_EPOCHS = 100` floor enforced | **PASS** | `MacroF1Checkpoint` saves weights only on macro F1 improvement; early stopping strictly blocked before epoch 100. |

---

## 2. Exact Problems Found and Repaired

Prior to this audit, `experiments/exp03/run_exp03.py` contained several critical defects resulting from legacy code merges:
1. **NameError on Model Construction:**  
   The script defined `build_E2_model(arm_id)` but called `build_E2_augmented()`, which was undefined and would have crashed immediately upon invocation.
2. **Missing Multi-Arm Execution Architecture:**  
   The script was hardcoded to execute only one training loop and write to a single un-versioned file set (`e2_augmented_oof_predictions.csv`), rather than managing the three distinct arms (`arm0`, `arm1`, `arm2`) required by the EXP-03 protocol.
3. **Legacy Focal Loss Residue:**  
   The script contained unused focal loss functions (`make_class_balanced_focal_loss`, `FOCAL_GAMMA = 2.0`), misleading docstrings, and console banners referencing "EXP-03: CLASS-BALANCED FOCAL LOSS", conflicting with the categorical cross-entropy requirement.
4. **Incorrect Checkpoint Directory:**  
   Checkpoints were targeted to `experiments/checkpoints/exp03/` rather than the required isolated directory `experiments/exp03/checkpoints/{arm_id}/`.
5. **Missing Dry-Run and CLI Controls:**  
   There was no facility to verify execution pipelines without launching 150-epoch CPU training.

### Actions Taken:
- Fully rewrote `experiments/exp03/run_exp03.py` with clean, modular architecture.
- Added explicit CLI argument parsing (`--arm {arm0,arm1,arm2,all}`, `--dry-run`).
- Implemented dedicated builders for Arm 0, Arm 1, and Arm 2.
- Verified that all output deliverables match project standards:
  - `reports/accuracy_improvement/exp03/arm0_oof_predictions.csv`
  - `reports/accuracy_improvement/exp03/arm1_oof_predictions.csv`
  - `reports/accuracy_improvement/exp03/arm2_oof_predictions.csv`
  - `reports/accuracy_improvement/exp03/metrics_comparison.csv`
  - `reports/accuracy_improvement/exp03/confusion_matrices/`
  - `reports/accuracy_improvement/exp03/integrity_audit.md`
  - `reports/accuracy_improvement/exp03/final_report.md`

---

## 3. Exact Files Modified or Created

1. **Modified / Rewritten:**
   - [`experiments/exp03/run_exp03.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp03/run_exp03.py): Repaired missing function calls, eliminated legacy focal loss code, implemented multi-arm execution and clean checkpointing.
2. **Created:**
   - [`scratch/test_pretraining_exp03.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scratch/test_pretraining_exp03.py): 8-step pre-training unit test suite.
   - [`reports/accuracy_improvement/exp03/final_pretraining_audit.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp03/final_pretraining_audit.md): This comprehensive audit report.
3. **Verified Unchanged (Zero Modifications):**
   - [`scripts/experiment6_fusion_supcon.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scripts/experiment6_fusion_supcon.py)
   - [`src/preprocessing.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/preprocessing.py)
   - All 5 production checkpoints in [`models/phase3_fusion_metric_learning/E2/`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/models/phase3_fusion_metric_learning/E2/)

---

## 4. Confirmation of Training Inaction

- **No training has been initiated or executed.**
- Only non-training syntax compilation, argument parsing, dummy forward passes (batch size 2), gradient checks, and hash verifications were run.
- Zero epochs have been trained on actual participant data.

---

## 5. Original Checkpoint Hash Baseline

The SHA-256 hashes of all five original E2 checkpoints were computed and verified against baseline records:

| Original Checkpoint | File Size | SHA-256 Checksum | Integrity Status |
|:---|:---:|:---:|:---:|
| `fold_1_best.weights.h5` | 52,910,112 bytes | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` | **INTACT** |
| `fold_2_best.weights.h5` | 52,910,112 bytes | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` | **INTACT** |
| `fold_3_best.weights.h5` | 52,910,112 bytes | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` | **INTACT** |
| `fold_4_best.weights.h5` | 52,910,112 bytes | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` | **INTACT** |
| `fold_5_best.weights.h5` | 52,910,112 bytes | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` | **INTACT** |

---

## 6. Remaining Methodological Limitations

1. **Minority Class Sample Size:**
   The development cohort contains only 7 AB+ participants (8.75%). Consequently, a single correct or incorrect prediction shifts AB+ recall by $\pm 14.29\%$. Performance changes in AB+ recall must be interpreted as qualitative directional evidence rather than high-powered statistical significance.
2. **CPU Floating-Point Round-Off:**
   TensorFlow is operating in CPU mode with oneDNN optimizations enabled. Minor floating-point round-off differences may occur across CPU thread executions; fixed random seeds minimize but cannot eliminate cross-platform micro-variance.
3. **Frozen Backbone Representation:**
   The DenseNet-121 backbone is frozen with ImageNet weights. The augmentation policies influence the features fed into the frozen backbone, but the convolutional kernels themselves do not adapt to altered ridge dynamics.

---

## 7. Command to Run EXP-03 (DO NOT EXECUTE UNTIL APPROVED)

When authorized, EXP-03 will be run sequentially across all three arms using:

```powershell
python experiments/exp03/run_exp03.py --arm all
```

Or individually per arm:
```powershell
python experiments/exp03/run_exp03.py --arm arm0
python experiments/exp03/run_exp03.py --arm arm1
python experiments/exp03/run_exp03.py --arm arm2
```

Estimated runtime per arm: ~30–35 minutes on 12-core CPU; total runtime for all 3 arms: ~1.5 to 1.75 hours.

---

**AUDIT CONCLUSION:**  
All pre-training verification checks have **PASSED**. The implementation is secure, correct, isolated, and strictly aligned with the experimental protocol.

**Awaiting user approval before starting training.**
