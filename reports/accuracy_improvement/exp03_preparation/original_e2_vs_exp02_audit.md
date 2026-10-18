# EXP-03 Preparation: Original E2 vs. EXP-02 Code-Level Audit Report

**Audit Timestamp:** 2026-10-14T13:15:00
**Scope:** Forensic Code-Level Verification of EXP-02 vs. Original E2 Training & Evaluation Pipeline
**Target Files Inspected:**
- scripts/experiment6_fusion_supcon.py (Authoritative provenance of baseline E2)
- experiments/exp02_e2_augmentation.py (EXP-02 implementation)
- experiments/exp01_class_balanced_e2.py (Inheritance tracing for focal loss & weights)
- reports/model_fusion/all_models_participant_probs.npz (Verified baseline OOF predictions)
- reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv (Baseline OOF export)
- reports/accuracy_improvement/exp02/e2_augmented_oof_predictions.csv (EXP-02 OOF predictions)

---

## 1. Executive Summary & Audit Verdict

**Core Audit Finding:**
EXP-02 was **NOT an isolated test of data augmentation**. It was a **severely confounded experiment** characterized by three major hidden discrepancies:
1. **Severe Under-Training:** The original E2 model was trained for at least 100 epochs (MIN_EPOCHS = 100, EPOCHS = 150). In EXP-02, early stopping was permitted after only 15 epochs (MIN_EPOCHS = 15, EPOCHS = 60, PATIENCE = 12). Consequently, all folds in EXP-02 terminated after only 15 to 27 epochs—less than one-fourth the training duration of the baseline.
2. **Baseline Augmentation Misconception:** The baseline E2 model was not an unaugmented baseline. Original E2 already included an aggressive data augmentation block inside build_fusion_base: rotation (~9.7 deg), translation (10%), zoom (10%), and contrast (10%). EXP-02 did not compare unaugmented vs. augmented; it tested conservative augmentation (no zoom, smaller shifts) against aggressive augmentation under drastically truncated epochs.
3. **Class Weighting Provenance:** Variable alpha_weights was syntactically inherited from EXP-01 focal loss script, but mathematically and functionally it evaluated to N / (4 * N_c), which was passed as sample_weight to model.fit(). This exactly matches the inverse-frequency sample weights cw used in the original E2 training script. Standard cross-entropy was correctly restored, but the sample-weighting scheme was active in both.

**Conclusion:**
The reported performance drop in EXP-02 (accuracy 53.75% -> 47.50%, macro F1 0.4772 -> 0.4632) **cannot be scientifically attributed to conservative data augmentation**. The drop was heavily confounded by premature early stopping and severe under-training.

---

## 2. Forensic Tracing: alpha_weights & sample_weight

A central question was whether inverse-frequency class weighting inherited from EXP-01 was erroneously active in EXP-02.

### Original E2 (scripts/experiment6_fusion_supcon.py lines 209-238):
- train_counts = pd.Series(y_train).value_counts().to_dict()
- cw = {i: total_train / (4 * train_counts.get(i, 1.0)) for i in range(4)}
- sw = np.array([cw[y] for y in y_train])
- model.fit(X_train, [y_train, y_train], sample_weight=[sw, np.ones_like(sw)])

### EXP-02 (experiments/exp02_e2_augmentation.py lines 270-298):
- train_class_counts = {i: int(sum(y_train_f == i)) for i in range(4)}
- alpha_weights = [n_train / (4.0 * train_class_counts[i]) for i in range(4)]
- sw = np.array([alpha_weights[y] for y in y_train_f], dtype=np.float32)
- model.fit(X_train_f, [y_train_f, y_train_f], sample_weight=[sw, np.ones_like(sw)])

### Audit Finding on Class Weighting:
- **Was focal loss active?** **NO.** Categorical focal loss was completely removed; standard sparse_categorical_crossentropy was restored.
- **Were class weights active?** **YES.** In both scripts, sample weights w_i = N / (4 * N_c) were passed to the classification branch.
- **Did EXP-02 deviate from original E2 regarding sample weights?** **NO.** Although EXP-02 used the variable identifier alpha_weights (inherited from EXP-01), the mathematical formula, numerical values, and application via sample_weight=[sw, np.ones_like(sw)] were 100% identical to the original E2 baseline.

---

## 3. Confounding Differences Analysis

### Difference 1: Epoch Limits & Minimum Training Floor
| Parameter | Original E2 (experiment6_fusion_supcon.py) | EXP-02 (exp02_e2_augmentation.py) | Confounding Effect |
|:---|:---:|:---:|:---|
| EPOCHS | **150** | **60** | Max training horizon reduced by 60%. |
| MIN_EPOCHS | **100** | **15** | Training floor reduced by 85%. |
| PATIENCE | **15** | **12** | Early stopping allowed much earlier. |

In the original E2 implementation, early stopping had a strict gate:
if (epoch + 1) >= MIN_EPOCHS and self.wait >= PATIENCE: self.model.stop_training = True

Because MIN_EPOCHS = 100, the original E2 was **guaranteed to train for at least 100 epochs**, enabling the DenseNet-121 feature fusion head to stabilize its dense representations. In EXP-02, MIN_EPOCHS = 15 allowed the callback to stop training as early as epoch 24 (Fold 5 stopped at epoch 24 with best epoch 12).

### Difference 2: The Baseline Augmentation Reality
In the original E2 baseline (scripts/experiment6_fusion_supcon.py lines 80-85):
Original E2 used **aggressive data augmentation**:
- Rotation: +/- 0.027 * 360 = +/- 9.72 deg
- Translation: +/- 10% horizontal & vertical (~11.2 pixels)
- Zoom: +/- 10% scale expansion/contraction (distorting ridge-to-valley frequency)
- Contrast: +/- 10% contrast scaling

In EXP-02, the augmentation was modified to:
- Rotation: +/- 0.02 * 360 = +/- 7.2 deg
- Translation: +/- 4% (~4.5 pixels)
- Zoom: **EXCLUDED (0.0%)**
- Contrast: +/- 5%
- Brightness: +/- 5%

Therefore, EXP-02 was actually testing **conservative augmentation without zoom vs. aggressive augmentation with zoom**, while simultaneously suffering from severe under-training.

---

## 4. 16-Dimension Line-by-Line Comparison

| # | Dimension | Original E2 Baseline | EXP-02 Setting | Different? | Affects Result? |
|---|---|---|---|:---:|:---:|
| 1 | Architecture | 3-scan DenseNet-121 GAP Concat(3072) Dense(512) Drop(0.5) Dense(4) + Emb(128) | Identical structure and layer definitions | No | No |
| 2 | Initial Weights | ImageNet pretrained, backbone frozen | ImageNet pretrained, backbone frozen | No | No |
| 3 | Classification Loss | sparse_categorical_crossentropy | sparse_categorical_crossentropy | No | No |
| 4 | SupCon Loss | SupCon with temp=0.07, L2 norm | SupCon with temp=0.07, L2 norm | No | No |
| 5 | Loss Weights | [1.0, 0.1] | [1.0, 0.1] | No | No |
| 6 | Class Weighting | Inverse frequency N / (4 N_c) via sample_weight | Inverse frequency N / (4 N_c) via sample_weight | No | No |
| 7 | Optimizer | AdamW(lr=1e-3, wd=1e-4) | AdamW(lr=1e-3, wd=1e-4) | No | No |
| 8 | Learning Rate | Fixed 1e-3 | Fixed 1e-3 | No | No |
| 9 | Epoch Limits | **EPOCHS=150, MIN_EPOCHS=100** | **EPOCHS=60, MIN_EPOCHS=15** | **YES** | **YES (MAJOR)** |
| 10 | Early Stopping | **Patience=15, floor=100 epochs** | **Patience=12, floor=15 epochs** | **YES** | **YES (MAJOR)** |
| 11 | Preprocessing | transform_c5 (112x112, float32 [0, 255]) | transform_c5 (112x112, float32 [0, 255]) | No | No |
| 12 | Augmentation | **Aggressive (Rot 0.027, Trans 0.1, Zoom 0.1, Cont 0.1)** | **Conservative (Rot 0.02, Trans 0.04, No Zoom, Cont 0.05, Brt 0.05)** | **YES** | **YES (INTENDED)** |
| 13 | Random Seeds | Seed 42 for NumPy & TF; layers unseeded | Seed 42 for NumPy & TF; layers seeded with 42 | Minor | Negligible |
| 14 | Fold Splits | StratifiedGroupKFold(n_splits=5, seed=42) | StratifiedGroupKFold(n_splits=5, seed=42) | No | No |
| 15 | Aggregation | 1 OOF prediction per participant (16/fold) | 1 OOF prediction per participant (16/fold) | No | No |
| 16 | Class Mapping | Native [A+, B+, AB+, O+] -> Canonical [A+, AB+, B+, O+] | Native [A+, B+, AB+, O+] -> Canonical [A+, AB+, B+, O+] | No | No |

---

## 5. Audit Conclusion

The experiment EXP-02 must be formally reclassified as **CONFOUNDED**.
The hypothesis that conservative data augmentation hurts performance is scientifically unsupported by EXP-02 because the model was trained for only 15-27 epochs instead of the baseline 100-150 epochs.
