# EXP-07-RES-FIDELITY: Resolution Scaling & Spatial Fidelity Benchmark

## 1. Overview & Scientific Objective
EXP-07-RES-FIDELITY is a controlled, single-factor comparative benchmark investigating whether **optical sensor downsampling and spatial aliasing** caused the failure of deep learning models on the Mantra MFS100 dataset.

- **Arm A:** $112 \times 112$ input resolution (Historical baseline spatial scale, ridge period $\approx 2.67\text{ px}$).
- **Arm B:** $224 \times 224$ input resolution (Fidelity spatial scale, ridge period $\approx 5.33\text{ px}$).

Both arms use the exact same three-scan DenseNet-121 multi-instance architecture with **1,968,772 trainable head parameters** (0% parameter confound), strictly frozen ImageNet backbone (**7,037,504 non-trainable parameters**), identical loss functions ($\text{CCE} + 0.50 \cdot \text{SupCon}$), and identical participant folds.

---

## 2. Methodology & Leakage-Free Design

### A. Elimination of Checkpoint-Selection Leakage
In historical experiments (EXP-03 to EXP-06C), models monitored validation Macro F1 directly on the outer 16-participant validation fold, allowing stochastic head fluctuations to inflate development scores up to 53.75% (and 46.77% on pure noise).

In EXP-07, this leakage is completely eradicated:
1. **Regime 1 (Primary - Fixed Duration):** Folds train for exactly **80 epochs**. Outer validation data is **never passed** to the training loop. No callbacks monitor outer-fold metrics. The final epoch weights are evaluated strictly once on the held-out participants.
2. **Regime 2 (Secondary - Nested Inner Validation):** The 64 training participants in each outer fold are split into 48 inner-train and 16 inner-validation participants. Model checkpointing is monitored **only** on the inner validation set. The selected checkpoint is evaluated once on the outer fold.

### B. Minimal Preprocessing Pipeline
Both arms eliminate CLAHE and bilateral filtering (which EXP-06D proved destroyed fine minutiae):
1. Composite alpha transparency onto solid white background.
2. Foreground platen crop (`crop_mfs100_foreground`).
3. High-quality Lanczos resize to target resolution ($112 \times 112$ or $224 \times 224$).
4. Minimal raw float32 array in $[0, 255]$.

---

## 3. Strict Safety Firewalls & Quarantines

1. **Exposed Test Cohort Quarantine:** The 21-participant locked cohort (`split == 'TEST'`) has been evaluated 17 times historically. It is **strictly excluded** from EXP-07. Automated assertions abort execution if any test participant ID appears in the data pipeline.
2. **Hospital Cohort Quarantine:** The 26-participant hospital dataset is **strictly excluded** pending cross-cohort participant identity auditing.
3. **Fail-Closed Training Guard:** `TRAINING_AUTHORIZED` is set to `False` in `config.py`. The training script raises `PermissionError` and halts if real participant training is attempted without explicit authorization.

---

## 4. Pipeline Scripts & Usage

| Script | Purpose | Status |
| :--- | :--- | :---: |
| `config.py` | Experiment hyperparameters, paths, and fail-closed security guard. | Verified |
| `split_validator.py` | Validates participant firewalls, 5-fold stratification, and cohort isolation. | Passed |
| `synthetic_test.py` | Validates model graph, parameter parity, forward/backward passes on dummy tensors. | Passed |
| `train.py` | Cross-validation training engine (locked by fail-closed guard). | Verified |
| `evaluate.py` | Statistical hypothesis testing: Wilson CI, paired McNemar, paired permutations. | Passed |
| `manifest.json` | Machine-readable experiment manifest, checksums, and validation log. | Verified |

### Validation Commands (Safe to Run - Zero Real Data Accessed)
```bash
# Validate participant-level split firewalls
python experiments/exp07_resolution_fidelity/split_validator.py

# Validate synthetic model graph and gradient steps
python experiments/exp07_resolution_fidelity/synthetic_test.py

# Verify statistical evaluation functions on synthetic demo
python experiments/exp07_resolution_fidelity/evaluate.py --synthetic-demo
```

### Proposed Execution Commands (DO NOT RUN WITHOUT AUTHORIZATION)
*These commands require `TRAINING_AUTHORIZED = True` in `config.py` and explicit user authorization:*
```bash
# Train Arm A and Arm B under Fixed-Duration Regime
python experiments/exp07_resolution_fidelity/train.py --condition all --regime fixed_duration

# Evaluate and perform paired statistical significance testing
python experiments/exp07_resolution_fidelity/evaluate.py
```
