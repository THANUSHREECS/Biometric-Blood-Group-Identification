# Comprehensive Scientific & Implementation Review: EXP-07-RES-FIDELITY

**Document ID:** EXP07_IMPLEMENTATION_REVIEW  
**Review Date:** 2026-10-17  
**Reviewer:** Senior Machine Learning Research Engineer & Biometric Audit Specialist  
**Experiment Identifier:** EXP-07-RES-FIDELITY  
**Current Execution Status:** Implementation & Synthetic Validation **COMPLETE** — Training on Real Data **LOCKED** (`TRAINING_AUTHORIZED = False`)

---

## Executive Summary & Recommendation

| Audit Category | Evaluation Status | Key Finding / Risk Level |
| :--- | :---: | :--- |
| **1. Data Isolation & Quarantine** | **VERIFIED** | Dev-80 cohort (80 participants, 240 scans) strictly isolated. Locked test ($N=21$) and hospital ($N=26$) cohorts 100% quarantined with programmatic fail-closed assertion guards. |
| **2. Experiment Fairness & Parity** | **VERIFIED** | Zero parameter confound: both Arm A ($112 \times 112$) and Arm B ($224 \times 224$) share identical 1,968,772 trainable parameters, identical frozen DenseNet-121 backbones, identical seeds, and identical 5-fold splits. |
| **3. Leakage Prevention** | **VERIFIED** | Outer-fold checkpoint selection completely abolished. Fixed-duration training (80 epochs) evaluates final weights with zero outer-fold feedback. Nested inner-validation confines selection strictly to inner folds. |
| **4. Implementation Correctness** | **CORRECTED & VERIFIED** | Class remapping between Native `["A+", "B+", "AB+", "O+"]` and Canonical `["A+", "AB+", "B+", "O+"]` verified. Preprocessing respects the Round 14 fix (raw float32 $[0, 255]$, avoiding activation collapse). |
| **5. Statistical Rigor & Power** | **VERIFIED** | Predefined primary metric: Out-of-Fold Balanced Accuracy. Paired McNemar and paired permutation tests implemented. Statistical power limitations on $N=80$ documented. |
| **6. Synthetic Validation** | **PASSED** | 100% synthetic pipeline tests passed on dummy tensors. Zero real participant scans loaded for training. `TRAINING_AUTHORIZED` remains `False`. |
| **FINAL RECOMMENDATION** | **CONDITIONAL GO** | **Codebase and experimental design are fully verified and safe for training once explicit authorization is granted by the user.** |

---

## 1. Data Isolation & Cohort Quarantine Audit

### Verified Protocol & Safeguards
1. **Authorized Development Cohort:**
   - Only the 80 participants designated as `TRAIN` (64) and `VALIDATION` (16) in `reports/phase3_participant_benchmark/participant_split.csv` are loaded.
   - Total scans: exactly 240 (3 scans per participant).
   - Label consistency: 100% uniform blood groups across all 3 scans of every participant.
2. **Locked Test Cohort Quarantine ($N=21$ Participants, 63 Scans):**
   - The 21 test participants (`P0023`, `P0035`, etc.) were evaluated 17 times historically and are treated as an exposed holdout.
   - Programmatic assertion in `load_development_cohort()`:
     ```python
     assert len(dev_pids.intersection(test_pids)) == 0, "DATA LEAKAGE: Overlap with locked test set!"
     ```
   - Verified that zero test participant scans or labels can enter the EXP-07 data pipeline.
3. **Quarantined Hospital Cohort ($N=26$ Participants, 78 Scans):**
   - Located under `data/hospital_verified/`. Contains 25 serology-confirmed records.
   - Programmatic assertion in `preprocess_scan_fidelity()`:
     ```python
     assert "hospital" not in str(img_p).lower(), "QUARANTINE BREACH: Hospital scan in dev loader!"
     ```
   - Zero hospital scans are present in the Dev-80 manifest.
4. **Participant-Level Grouping:**
   - Evaluated using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly on `participant_id`.
   - All 3 impressions from any participant reside within the same fold.
   - Across all 5 folds: $\text{Train} = 64\text{ participants}$, $\text{Outer-Val} = 16\text{ participants}$. Zero participant overlap between train and validation.
5. **Fail-Safe Behavior:**
   - If any split manifest alteration, missing scan, or participant overlap occurs, the script immediately raises `AssertionError` and terminates without executing training.

---

## 2. Experiment Fairness & Structural Parity

The experiment isolates **spatial input resolution** while holding all other architectural, optimization, and dataset variables strictly invariant:

### Detailed Condition Comparison Matrix
| Architectural Component | Arm A: Baseline Resolution | Arm B: Fidelity Resolution | Parity Assessment |
| :--- | :---: | :---: | :---: |
| **Input Spatial Dimensions** | $112 \times 112 \times 3$ | $224 \times 224 \times 3$ | **Target Variable** |
| **Human Ridge Sampling Rate** | $\approx 2.67\text{ px/cycle}$ (Aliasing) | $\approx 5.33\text{ px/cycle}$ (Nyquist-compliant) | **Target Variable** |
| **Backbone Architecture** | DenseNet-121 (Pretrained ImageNet) | DenseNet-121 (Pretrained ImageNet) | Identical |
| **Backbone Trainability** | Strictly Frozen (`trainable = False`) | Strictly Frozen (`trainable = False`) | Identical |
| **Frozen Backbone Parameters** | $7,037,504$ | $7,037,504$ | Exact Match |
| **Per-Scan Pooling** | GlobalAveragePooling2D ($1024$-d) | GlobalAveragePooling2D ($1024$-d) | Identical |
| **Multi-Scan Feature Fusion** | Concatenation ($3 \times 1024 = 3072$-d) | Concatenation ($3 \times 1024 = 3072$-d) | Identical |
| **Classification Head** | Dense(512, relu) $\to$ Drop(0.5) $\to$ Dense(4) | Dense(512, relu) $\to$ Drop(0.5) $\to$ Dense(4) | Identical |
| **Projection Head** | Dense(128, activation=None) | Dense(128, activation=None) | Identical |
| **Trainable Head Parameters** | **1,968,772 across 6 tensors** | **1,968,772 across 6 tensors** | **0% Parameter Confound** |
| **Head Weight Initialization** | Glorot Uniform (Seed: $42 + \text{fold}$) | Glorot Uniform (Seed: $42 + \text{fold}$) | Identical Initial State |
| **Loss Function** | $1.0 \cdot \text{CCE} + 0.50 \cdot \text{SupCon}(\tau=0.05)$ | $1.0 \cdot \text{CCE} + 0.50 \cdot \text{SupCon}(\tau=0.05)$ | Identical |
| **Optimizer & Schedule** | Adam ($\text{lr}=10^{-3}$, decay $=10^{-4}$, CosineDecay) | Adam ($\text{lr}=10^{-3}$, decay $=10^{-4}$, CosineDecay) | Identical |
| **Training Duration** | Exactly 80 Epochs (Batch Size 16) | Exactly 80 Epochs (Batch Size 16) | Identical |
| **Cross-Validation Folds** | Exact same 5 folds on Dev-80 | Exact same 5 folds on Dev-80 | Identical Partition |

### Verification of DenseNet-121 Preprocessing & Input Resizing
- **Historical Context:** In early project phases (Round 14), applying `densenet.preprocess_input` shifted activations and caused $\approx 85\%$ collapse into class A+ on the locked test set.
- **EXP-07 Preprocessing:** Uses the proven EXP-06D "Raw C5" pipeline:
  1. Alpha channel composited onto solid white background $(255, 255, 255)$.
  2. MFS100 platen foreground crop (`crop_mfs100_foreground`) with 6% safety padding.
  3. Lanczos interpolation resizing to target dimension ($112 \times 112$ or $224 \times 224$).
  4. Raw float32 pixel values in $[0, 255]$ without destructive CLAHE, contrast stretching, or bilateral smoothing.
- **Scientific Boundary:** We explicitly make **no claim** that $224 \times 224$ will achieve higher accuracy. The hypothesis must be tested empirically under the predefined protocol.

---

## 3. Checkpoint-Selection Leakage Prevention

### Root Cause of Past Flaws
Historical experiments (EXP-03 to EXP-06C) tracked `val_macro_f1` directly on the outer 16-participant validation fold. Across 100 training epochs, random fluctuations routinely spiked above 50% purely by chance, capturing noise (averaging 46.77% on pure permuted labels).

### Safeguards Implemented in EXP-07
1. **Regime 1: Fixed-Duration Training (Primary Protocol):**
   - Outer validation participants are **never provided** to `model.fit()`.
   - Zero callbacks monitor outer-fold metrics.
   - The model trains for exactly **80 epochs**.
   - Model weights at the end of Epoch 80 are evaluated strictly once on the held-out fold.
   - *Outer Selection Leakage:* **Mathematically 0.00%.**
2. **Regime 2: Nested Inner-Validation (Secondary Diagnostic Protocol):**
   - The 64 training participants in each outer fold are split into **48 inner-training participants** and **16 inner-validation participants**.
   - Early stopping or checkpoint selection is monitored strictly on the 16 inner-validation participants.
   - The outer 16 participants remain completely unobserved until the selected checkpoint is frozen.
   - *Outer Selection Leakage:* **Zero.**
3. **Prohibition of Test Peeking:**
   - Hyperparameters (learning rate, weight decay, epochs) are fixed in `config.py` prior to training and will not be adjusted after observing validation results.

---

## 4. Implementation Correctness & Bug Audit

During the line-by-line code review, four implementation issues were detected and corrected:

### Issue Log & Corrections

| Issue ID | File & Function | Severity | Description | Correction Applied |
| :--- | :--- | :---: | :--- | :--- |
| **ISSUE-01** | `evaluate.py`<br>`compute_comprehensive_metrics` | **High** | A refactoring edit inadvertently deleted the computation of `majority_class_idx`, causing a runtime `NameError` during evaluation. | Restored `majority_class_idx` and `majority_baseline` calculation before building the results dictionary. |
| **ISSUE-02** | `train.py`<br>`train_condition` | **Medium** | In `nested_inner` mode, `best_inner_weights` was initialized to `None`. If inner F1 never exceeded -1.0, `set_weights(None)` would fail. | Initialized `best_inner_weights = model.get_weights()` at epoch 0 prior to the inner training loop. |
| **ISSUE-03** | `train.py`<br>`train_condition` | **Low** | Optimizer used constant learning rate ($10^{-3}$) rather than the cosine annealing schedule specified in the proposal. | Implemented `tf.keras.optimizers.schedules.CosineDecay` over 80 epochs with warm restart floor ($10^{-6}$). |
| **ISSUE-04** | `evaluate.py`<br>`compute_comprehensive_metrics` | **Medium** | Native class ordering is `["A+", "B+", "AB+", "O+"]` (indices 0, 1, 2, 3), whereas canonical historical reports use `["A+", "AB+", "B+", "O+"]`. Evaluating without explicit remapping could swap B+ and AB+ in downstream reports. | Added explicit canonical remapping (`canon_map = [0, 2, 1, 3]`) exporting both native and canonical confusion matrices and class labels. |

---

## 5. Statistical Evaluation Plan & Power Analysis

### Predefined Primary Metric
- **Out-of-Fold Balanced Accuracy:** Defined as the unweighted mean of recall across all four classes:
  $$\text{Balanced Accuracy} = \frac{1}{4} \sum_{c=1}^{4} \text{Recall}_c$$
  Prevents performance inflation from majority class guessing (class O+ represents 35.0% of Dev-80).

### Secondary Metrics
- Participant-Level Accuracy ($\frac{k}{80}$) with 95% Wilson Score Binomial Confidence Interval.
- Macro F1-Score (unweighted mean of per-class F1).
- Multi-class Cohen's Kappa ($\kappa$).
- Per-class precision, recall, and confusion matrix.

### Paired Hypothesis Testing on the Same 80 Participants
Because Arm A ($112 \times 112$) and Arm B ($224 \times 224$) are evaluated on the exact same 80 out-of-fold participants:
1. **Paired McNemar Exact Test:** Evaluates discordance between paired binary predictions:
   $$b = \text{Arm A correct, Arm B incorrect}, \quad c = \text{Arm A incorrect, Arm B correct}$$
   Two-sided exact binomial test under $H_0: P(b) = P(c) = 0.5$.
2. **Paired Permutation Test ($10,000$ Resamples):** Randomly swaps participant prediction vectors with probability 0.5 to determine whether $\Delta_{\text{bal\_acc}} = \text{BalAcc}_B - \text{BalAcc}_A$ exceeds permutation noise ($\alpha = 0.05$).
3. **Participant-Level Label Permutation Null Distribution ($1,000$ Resamples):** Shuffles ground truth labels across the 80 participants to calculate non-parametric $p$-values against chance.

### Statistical Power & Sample Size Limitations ($N=80$)
With $N=80$ participants, statistical power is constrained:
- A difference of 4 participants represents $\Delta_{\text{acc}} = \pm 5.00\%$.
- If Arm A achieves 36.25% (29/80) and Arm B achieves 43.75% (35/80), the raw difference is +7.5%. However, if discordant pairs are 11 vs 5, McNemar's exact $p$-value is $p = 0.210$ (not statistically significant at $\alpha = 0.05$).
- For Arm B to achieve statistically significant superiority over Arm A ($\alpha < 0.05$), discordant pairs must reach approximately 15 vs 3 ($p = 0.0075$) or 16 vs 4 ($p = 0.0118$), requiring a substantial performance separation.
- **Mandatory Reporting Rule:** We will not declare higher resolution "proven" based on a marginal accuracy gain unless supported by the paired permutation test ($p < 0.05$) and non-zero recall across all four classes.

---

## 6. Verification Tests Executed & Empirical Results

All pre-training validation suites were executed on the system without accessing real participant images:

```
[Test Suite 1: Participant Split & Safety Firewalls]
Command: python experiments/exp07_resolution_fidelity/split_validator.py
Result:  PASSED
Details:
  - Total participants: 101 | Scans: 303 (3 per participant).
  - Dev-80 cohort: 80 participants (240 scans) strictly isolated.
  - Locked test set: 21 participants (63 scans) strictly quarantined (overlap = 0).
  - Hospital cohort: 26 participants (78 scans) strictly quarantined (0 references).
  - 5-Fold Stratified Group Split: exactly 64 train / 16 val per fold; zero leakage.

[Test Suite 2: Synthetic Pipeline & Architecture Graph]
Command: python experiments/exp07_resolution_fidelity/synthetic_test.py
Result:  PASSED
Details:
  - Arm A (112x112): Frozen backbone = 7,037,504 params | Trainable head = 1,968,772 params.
  - Arm B (224x224): Frozen backbone = 7,037,504 params | Trainable head = 1,968,772 params.
  - Parameter Parity: Verified exact match (0% parameter confound).
  - Forward inference: Verified shapes (clf: [8, 4], emb: [8, 128]).
  - Backward gradients: 2 synthetic epochs passed without NaN loss or tensor errors.
  - Security Guard: Verified PermissionError raised when TRAINING_AUTHORIZED = False.

[Test Suite 3: Evaluation Engine & Statistical Tests]
Command: python experiments/exp07_resolution_fidelity/evaluate.py --synthetic-demo
Result:  PASSED
Details:
  - Predefined primary metric (Balanced Accuracy) computed cleanly.
  - Wilson 95% score confidence intervals verified.
  - Paired McNemar exact test and 10,000-resample paired permutation test verified.
  - Canonical and native confusion matrices verified.
```

---

## 7. Remaining Scientific Risks & Limitations

1. **Self-Reported Label Noise:** Dev-80 ground truth labels are self-reported without laboratory serological confirmation. Known literature error rates for self-reported blood types range from 10% to 25%.
2. **Single Sensor Domain:** All scans originate from a single optical sensor (Mantra MFS100). Sensor-specific optical aberrations or illumination gradients cannot be factored out.
3. **Absence of Pristine External Blind Set:** The 21-participant locked cohort was exposed across 10 evaluation dates, and the hospital cohort remains quarantined. This experiment provides an internal controlled development comparison, not an external clinical validation.

---

## 8. Final Recommendation: CONDITIONAL GO

The implementation review is complete. All code, split integrity assertions, architecture parameter parities, leakage guards, and statistical testing routines are **correct, robust, and verified**.

### Prerequisite for Real Training:
To initiate training on the Dev-80 cohort, the user must explicitly authorize execution. Once authorized, the following procedure is prepared:
1. Toggle `TRAINING_AUTHORIZED = True` in [`config.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp07_resolution_fidelity/config.py#L35).
2. Execute the verified training pipeline:
   ```bash
   python experiments/exp07_resolution_fidelity/train.py --condition all --regime fixed_duration
   ```
3. Run the evaluation and statistical significance pipeline:
   ```bash
   python experiments/exp07_resolution_fidelity/evaluate.py
   ```

**Current State: TRAINING_AUTHORIZED remains `False`. No real-data training has been executed. Awaiting user authorization.**
