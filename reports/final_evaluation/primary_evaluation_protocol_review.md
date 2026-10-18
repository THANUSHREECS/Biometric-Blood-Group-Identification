# Final Protocol Review: Primary EXP-06C Config A Evaluation
## Biometric Blood Group Detection / HemoPrint AI

**Document Type:** Final Pre-Execution Protocol, Checkpoint, and Safety Verification Review  
**Candidate Model Under Review:** `EXP-06C Config A` (DenseNet-121 3-Scan Concat + Linear Projection Head)  
**Target Evaluation Cohort:** Primary Locked 21-Participant Test Cohort ($N=21$, 63 scans)  
**Review Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Authorization Guard:** `TRAINING_AUTHORIZED = False`, `EVALUATION_AUTHORIZED = False`  
**Execution Status:** **REVIEW COMPLETE — ZERO REAL-DATA EVALUATION EXECUTED**  

---

> [!IMPORTANT]
> ### PRE-EXECUTION AUDIT BOUNDARY
> In strict accordance with the evaluation governance directive:
> 1. No protected fingerprint image files were accessed, opened, or loaded.
> 2. No protected ground-truth labels were utilized.
> 3. No real-cohort predictions or metrics were computed.
> 4. `EVALUATION_AUTHORIZED` remains **False**.
> 5. `TRAINING_AUTHORIZED` remains **False**.

---

## 1. Executive Summary & Verification Matrix

This review independently inspected the five core pre-evaluation artifacts:
- `reports/final_evaluation/candidate_freeze_manifest.md`
- `reports/final_evaluation/cohort_provenance_review.md`
- `reports/final_evaluation/preregistered_evaluation_protocol.md`
- `reports/final_evaluation/safe_evaluation_implementation_report.md`
- `experiments/final_evaluation/evaluate_frozen_candidate.py`

All ten pre-execution requirements were verified against physical source code, cryptographic hashes, and pre-registered protocol definitions:

| Req # | Audit Requirement | Verification Finding | Compliance Status | Evidence Category & Source |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Checkpoints Match Freeze Manifest** | Exact 5 checkpoints and SHA-256 hashes match manifest bit-for-bit | **PASS** | Verified from source — `candidate_freeze_manifest.md` vs `evaluate_frozen_candidate.py:38-69` |
| **2** | **Methodological Validity of 5-Fold Ensemble** | 21 test participants were 100% held out from all 5 training folds; soft-voting ensemble is pre-registered | **PASS** | Verified from source — `participant_split.csv` & `preregistered_evaluation_protocol.md:58-75` |
| **3** | **Zero Participant Overlap in Training** | Evaluated participants (`split == 'TEST'`) were never loaded or trained in any fold checkpoint | **PASS** | Verified from source — `reports/phase3_participant_benchmark/participant_split.csv` |
| **4** | **Inference & 3-Scan Aggregation Parity** | `transform_c5` (raw float32 $[0, 255]$), shape $(1, 3, 112, 112, 3)$, 3072-d concat, and soft-voting match candidate | **PASS** | Verified by synthetic test — `safe_evaluation_implementation_report.md:46-77` |
| **5** | **Primary Cohort Label Limitation Disclosed** | Lack of physical laboratory serology sheets explicitly disclosed (marked NOT VERIFIED for clinical serology) | **PASS** | Verified from source — `cohort_provenance_review.md:58-65` |
| **6** | **Evaluator Cannot Train or Modify Weights** | `TRAINING_AUTHORIZED = False`, `model.trainable = False`, 0 trainable variables, no optimizer/gradient tape | **PASS** | Verified by synthetic test — `evaluate_frozen_candidate.py:28, 127` |
| **7** | **Evaluation Restricted to 21-Cohort** | `evaluate_primary_cohort()` filters strictly to `split == 'TEST'`, asserts $N=21$, 63 scans | **PASS** | Verified from source — `evaluate_frozen_candidate.py:214-222` |
| **8** | **Hospital Cohort Inaccessible** | Evaluator does not reference `data/captured/`; explicit assertion blocks any path containing `"captured"` | **PASS** | Verified from source — `evaluate_frozen_candidate.py:246` |
| **9** | **Pre-Inference Checkpoint Hash Verification** | `FrozenCandidateEnsemble.__init__()` executes SHA-256 verification before model instantiation (fail-closed) | **PASS** | Verified by synthetic test — `evaluate_frozen_candidate.py:84-110, 148` |
| **10** | **Metric & Protocol Alignment** | Primary metric (Macro F1) and secondary metrics (Accuracy, Balanced Acc, per-class, CM, bootstrap CI) match | **PASS** | Verified from source — `preregistered_evaluation_protocol.md:78-115` |

---

## 2. Detailed Requirement-by-Requirement Audit

### Requirement 1: Checkpoint Parity with Manifest
- **Manifest:** `reports/final_evaluation/candidate_freeze_manifest.md` defines 5 fold checkpoints for EXP-06C Config A.
- **Implementation:** `FROZEN_CHECKPOINT_LEDGER` in `experiments/final_evaluation/evaluate_frozen_candidate.py:38-69` specifies identical paths, file sizes, and SHA-256 hashes:
  - Fold 1: `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5` (`ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc`, 52,899,216 bytes)
  - Fold 2: `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5` (`615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c`, 52,899,472 bytes)
  - Fold 3: `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5` (`e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1`, 52,899,472 bytes)
  - Fold 4: `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5` (`58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3`, 52,899,472 bytes)
  - Fold 5: `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5` (`6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6`, 52,899,472 bytes)
- **Verdict:** **PASS.** Checkpoints are 100% identical.

### Requirement 2: Methodological Validity of the 5-Fold Ensemble
- **Cross-Validation Scope:** During the development phase, 5-fold cross-validation was used as a model assessment tool on the 80 development participants. In that phase, each fold model was trained on 64 participants and validated on 16 out-of-fold participants.
- **Test-Set Independence:** The 21 test participants were **never** included in any development fold. They were held out from all five training runs.
- **Ensemble Rationale:** For unseen test participants, all 5 fold models act as distinct, fully held-out estimators trained on slightly different 64-participant bootstrap subsamples. Combining their predictions via soft-voting probability averaging ($\frac{1}{5}\sum_{k=1}^5 \mathbf{P}_k$) reduces variance, mitigates fold-specific initialization noise, and represents the established production inference design.
- **Verdict:** **PASS.** The 5-fold ensemble is methodologically valid.

### Requirement 3: Participant Disjointness from Training Data
- **Manifest Cross-Check:** In `reports/phase3_participant_benchmark/participant_split.csv`:
  - Exactly 80 participants are marked `TRAIN` (64) or `VALIDATION` (16).
  - Exactly 21 participants are marked `TEST` (`P002`, `P0023`, `P0024`, `P0029`, `P003`, `P0035`, `P0039`, `P0041`, `P0042`, `P0047`, `P0048`, `P0054`, `P0071`, `P0073`, `P0075`, `P0082`, `P0083`, `P0088`, `P009`, `P0093`, `P0095`).
- **Participant Grouping:** All 3 scans belonging to any participant stay strictly within their designated split. Zero test participant scans were present in any training partition.
- **Verdict:** **PASS.** Evaluated participants were not used to train any checkpoint.

### Requirement 4: Inference & Preprocessing Parity
- **Preprocessing:** Scans are transformed using `src.preprocessing.transform_c5`:
  - Alpha composite $\rightarrow$ platen crop $\rightarrow$ CLAHE ($clip=2.5$) $\rightarrow$ percentile stretch ($1-99\%$) $\rightarrow$ bilateral filter ($d=3, \sigma=25$) $\rightarrow$ resize $112\times 112$.
  - Data type: Raw `float32` array in $[0.0, 255.0]$ without zero-centering or variance scaling (`densenet.preprocess_input` is omitted).
- **Tensor Stacking:** 3 scans are stacked into shape `(1, 3, 112, 112, 3)`.
- **Classification Branch:** Fused feature vector (3072-d) $\rightarrow$ Dense(512, relu) $\rightarrow$ Dropout(0.5, inactive during inference) $\rightarrow$ Dense(4, softmax).
- **Soft Voting:** Probabilities across the 5 folds are averaged, normalized to $1.0$, and mapped to canonical classes `['A+', 'AB+', 'B+', 'O+']` via argmax.
- **Verdict:** **PASS.** Inference pipeline matches the frozen candidate definition.

### Requirement 5: Primary Cohort Label Limitation Disclosure
- **Documentation Status:** Section 2.2 of `reports/final_evaluation/cohort_provenance_review.md` and Section 1.1 of `reports/final_evaluation/preregistered_evaluation_protocol.md` explicitly disclose:
  - Ground-truth labels for the 21-participant cohort were obtained via donor self-reports / health camp records.
  - Physical laboratory serological test sheets are absent from the repository.
  - Serological status is classified as **NOT VERIFIED**.
  - The cohort legitimately tests internal held-out subject generalization, but **cannot support claims of clinical serological validity**.
- **Verdict:** **PASS.** Label limitations are fully and prominently disclosed.

### Requirement 6: Evaluator Immutability & Anti-Training Safeguards
- **Code Audit:** In `experiments/final_evaluation/evaluate_frozen_candidate.py`:
  - `TRAINING_AUTHORIZED = False` is hardcoded at line 28.
  - Every instantiated model has `model.trainable = False` explicitly set.
  - No `tf.GradientTape`, optimizer, learning rate schedule, loss function, `model.compile()`, or `model.fit()` exists anywhere in the script.
  - The executed synthetic test proved that the instantiated ensemble possesses exactly **0 trainable variables**.
- **Verdict:** **PASS.** The evaluator cannot train, fine-tune, or modify weights.

### Requirement 7: Restriction to the 21-Participant Cohort
- **Cohort Filter:** In `evaluate_primary_cohort()`:
  ```python
  df = pd.read_csv(os.path.join(base_dir, manifest_path))
  test_df = df[df["split"] == "TEST"].copy()
  participants = sorted(test_df["participant_id"].unique())
  assert len(participants) == 21, f"Expected exactly 21 participants, got {len(participants)}"
  assert len(test_df) == 63, f"Expected exactly 63 scans, got {len(test_df)}"
  ```
- **Scope Isolation:** The evaluation strictly processes the 21 participants in `split == 'TEST'`.
- **Verdict:** **PASS.** Evaluation is restricted strictly to the 21-participant test cohort.

### Requirement 8: Hospital Cohort Inaccessibility
- **Directory Isolation:** The evaluator does not read `data/captured/` or `data/mfs100_ground_truth.csv`.
- **Active Path Guard:** Line 246 contains an explicit runtime assertion:
  ```python
  assert "captured" not in scan_file, "Security violation: hospital cohort path detected!"
  ```
  If any scan path references the hospital cohort directory, inference halts immediately with an assertion error.
- **Verdict:** **PASS.** The hospital cohort remains completely inaccessible.

### Requirement 9: Pre-Inference Cryptographic Hash Verification
- **Verification Trigger:** `FrozenCandidateEnsemble.__init__()` executes `verify_checkpoint_integrity(base_dir)` before instantiating or loading any model.
- **Fail-Closed Behavior:** If any checkpoint is missing or its SHA-256 hash fails to match `FROZEN_CHECKPOINT_LEDGER`, it raises `RuntimeError("Checkpoint integrity verification FAILED. Aborting.")`.
- **Verdict:** **PASS.** Checkpoint hashes are verified prior to any inference.

### Requirement 10: Metric & Statistical Protocol Parity
- **Primary Metric:** Macro F1 (macro-averaged across `A+, AB+, B+, O+`).
- **Secondary Metrics:** Overall Accuracy, Balanced Accuracy, per-class Precision/Recall/F1/Support, Confusion Matrix in canonical order `[A+, AB+, B+, O+]`.
- **Uncertainty Quantification:** Non-parametric bootstrap ($B=2,000$ resamples with replacement, `seed=42`) computing 95% confidence intervals.
- **Baseline Comparison:** Two-sided exact binomial test on discordant participant pairs against historical Phase 3 baseline (`accuracy = 0.380952, macro_f1 = 0.250000`).
- **Verdict:** **PASS.** Metrics and statistical methods match the pre-registered protocol.

---

## 3. Findings & Certification

All 10 verification items passed without discrepancy. No code modifications or protocol corrections are required.

### Compliance Affirmations:
1. **NO protected fingerprint images were loaded, opened, or viewed.**
2. **NO protected ground-truth labels were used for model selection or calibration.**
3. **NO real-cohort predictions or metrics were generated.**
4. **`EVALUATION_AUTHORIZED = False` remains active and enforced.**
5. **`TRAINING_AUTHORIZED = False` remains active and enforced.**

---

*Protocol Review is complete. The primary evaluation framework is certified ready for execution upon explicit user authorization.*
