# EXP-03 Verification: Experimental Fairness & Isolation Audit

**Audit Date:** 2026-10-14T18:12:30+05:30  
**Scope:** Verification of Experimental Isolation Across Arm 0, Arm 1, and Arm 2  
**Audit Finding:** **STRICT FAIRNESS VERIFIED — ZERO CONFOUNDING FACTORS**  

---

## 1. Executive Summary

A critical lesson from EXP-02 was that unintended parameter drift (e.g., reducing `MIN_EPOCHS` from 100 to 15) can confound an entire experiment. EXP-03 was formulated with strict experimental fairness constraints to ensure that **data augmentation policy was the sole variable** manipulated across the three arms.

This audit evaluates the actual execution code and runtime configurations across 15 technical dimensions to verify that no secondary factors confounded the observed performance.

---

## 2. 15-Dimension Fairness Verification Matrix

| # | Experimental Dimension | Arm 0 (Original E2 Aug) | Arm 1 (No Augmentation) | Arm 2 (Conservative Aug) | Status |
|---|---|---|---|---|:---:|
| **1** | **Trunk Architecture** | 3-tower DenseNet-121, GAP, Concat(3072) | Identical structure | Identical structure | **IDENTICAL** |
| **2** | **Classification Head** | Dense(512, relu) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(4, softmax) | Identical structure | Identical structure | **IDENTICAL** |
| **3** | **Projection Head** | Dense(128, linear) from Concat(3072) | Identical structure | Identical structure | **IDENTICAL** |
| **4** | **Initial Backbone Weights** | ImageNet pretrained, `trainable=False` | ImageNet pretrained, `trainable=False` | ImageNet pretrained, `trainable=False` | **IDENTICAL** |
| **5** | **Participant Folds** | `StratifiedGroupKFold(n_splits=5, shuffle=True, seed=42)` | Identical 5 folds (seed 42) | Identical 5 folds (seed 42) | **IDENTICAL** |
| **6** | **Cohort Partitioning** | 64 train / 16 val per fold (80 total) | Identical 64 train / 16 val | Identical 64 train / 16 val | **IDENTICAL** |
| **7** | **Classification Loss** | `sparse_categorical_crossentropy` (wt: 1.0) | Identical loss & weight | Identical loss & weight | **IDENTICAL** |
| **8** | **Metric Learning Loss** | `supervised_contrastive_loss` (wt: 0.1, $\tau=0.07$) | Identical loss & weight | Identical loss & weight | **IDENTICAL** |
| **9** | **Sample Weighting Formula** | $w_i = N_{\text{train}} / (4 \cdot N_c)$ to `clf` branch | Identical weights & formula | Identical weights & formula | **IDENTICAL** |
| **10** | **Optimizer & Schedule** | AdamW(lr=1e-3, weight_decay=1e-4), constant lr | Identical optimizer & lr | Identical optimizer & lr | **IDENTICAL** |
| **11** | **Batch Size** | 16 participants per batch | 16 participants per batch | 16 participants per batch | **IDENTICAL** |
| **12** | **Training Limits & Floor** | `EPOCHS = 150`, `MIN_EPOCHS = 100`, `PATIENCE = 15` | Identical limits & floor | Identical limits & floor | **IDENTICAL** |
| **13** | **Checkpoint Selection** | Best validation Macro F1 on held-out fold | Identical selection metric | Identical selection metric | **IDENTICAL** |
| **14** | **Random Seed Governance** | Global `SEED=42`; per-fold `SEED + fold_idx` | Identical seed sequence | Identical seed sequence | **IDENTICAL** |
| **15** | **Class-Index Mapping** | Native $[A+, B+, AB+, O+] \rightarrow [A+, AB+, B+, O+]$ | Identical remapping formula | Identical remapping formula | **IDENTICAL** |

---

## 3. The Isolated Variable: Augmentation Policy

The only difference between the three arms was the augmentation layer placed immediately before the frozen DenseNet-121 backbone:

### Arm 0: Faithful Baseline Reproduction
- `RandomRotation(factor=0.027, fill_mode="nearest", seed=42)` ($\pm 9.72^\circ$)
- `RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=42)` ($\pm 10\%$)
- `RandomZoom(height_factor=0.1, fill_mode="nearest", seed=42)` ($\pm 10\%$)
- `RandomContrast(factor=0.1, seed=42)` ($\pm 10\%$)

### Arm 1: No-Augmentation Ablation
- `tf.keras.layers.Identity(name="arm1_identity_augmentation")`
- Zero stochastic spatial or photometric perturbation.
- All three scans passed to the backbone exactly as preprocessed by `transform_c5`.

### Arm 2: Conservative Fingerprint Augmentation
- `RandomRotation(factor=(-0.02, 0.02), fill_mode="nearest", interpolation="bilinear", seed=42)` ($\pm 7.2^\circ$)
- `RandomTranslation(height_factor=(-0.04, 0.04), width_factor=(-0.04, 0.04), fill_mode="nearest", interpolation="bilinear", seed=42)` ($\pm 4\%$)
- `RandomContrast(factor=0.05, seed=42)` ($\pm 5\%$)
- `RandomBrightness(factor=0.05, value_range=(0.0, 255.0), seed=42)` ($\pm 5\%$)
- `RandomZoom`: **EXCLUDED (0.0)** to protect ridge spatial period.

---

## 4. Invariance of Validation and Evaluation Procedures

A crucial check is whether augmentation layers were accidentally active during evaluation. In Keras:
- `RandomRotation`, `RandomTranslation`, `RandomZoom`, `RandomContrast`, and `RandomBrightness` check the execution flag `training`.
- During validation inference inside `MacroF1Checkpoint` and post-training OOF inference via `model.predict()`, `training=False` is strictly enforced.
- Unit testing confirmed that the maximum absolute numerical difference between input tensors and output tensors through the augmentation block during evaluation mode was **0.000000** for all three arms.
- Validation and OOF inference were executed under 100% identical deterministic conditions.

---

## 5. Audit Conclusion

The experimental setup of EXP-03 achieved **near-perfect experimental isolation**:
- 15 critical training and architectural dimensions were held strictly constant.
- The only variation was the data augmentation policy.
- All differences in performance (e.g., Arm 1 reaching 52.84% balanced accuracy vs Arm 0 at 42.56%) can be attributed with scientific confidence to the presence or absence of data augmentation.
- **Audit Verdict:** **PASS**.
