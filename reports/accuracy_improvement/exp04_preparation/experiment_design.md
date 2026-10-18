# EXP-04: Controlled Partial Fine-Tuning Experiment Design

**Experiment:** EXP-04 — Controlled Partial Fine-Tuning of DenseNet-121 Feature Fusion  
**Date:** 2026-10-14T18:20:00+05:30  
**Experimental Reference:** EXP-03 Arm 1 (E2 No-Augmentation Architecture)  
**Target Architecture:** DenseNet-121 3-Scan Feature Fusion with Multi-Task Supervised Contrastive Loss  
**Development Cohort:** Exactly 80 Development Participants (240 scans) from `participant_split.csv`  
**Safety & Quarantine Enforcement:** STRICT (Zero access to 21 locked test set or 26-cohort; production checkpoints untouched)  

---

## 1. Executive Summary & Scientific Motivation

In EXP-03, **Arm 1 (E2 without data augmentation)** proved to be the project's leading development candidate, reaching:
- **Accuracy:** 53.75%
- **Balanced Accuracy:** 52.84% (+5.69% over original baseline)
- **Macro F1-Score:** 0.5068 (+0.0296 over original baseline)
- **Minority Class AB+ Recall:** 57.14% (4/7) vs. 14.29% (1/7) in baseline E2
- **Minority Class B+ Recall:** 64.00% (16/25) vs. 56.00% in baseline E2

In both the baseline E2 and all EXP-03 arms, the DenseNet-121 convolutional backbone was **completely frozen** (`trainable = False`), carrying ImageNet pretrained weights. ImageNet features are optimized for macroscopic natural objects (animals, vehicles, textures) rather than microscopic epidermal ridge flow, core/delta minutiae patterns, and dermatoglyphic friction ridge geometry.

### Objective of EXP-04:
Investigate whether **controlled partial fine-tuning of top-level convolutional layers** enables DenseNet-121 to specialize its high-level feature representations to biometric fingerprint ridge structure without overfitting or catastrophic forgetting.

---

## 2. Experimental Configurations to Compare

To ensure rigorous experimental isolation, EXP-04 will evaluate three configurations across the exact same five participant-grouped folds:

### Configuration A: Frozen Backbone Reference (EXP-03 Arm 1 Replication)
- **Backbone Status:** Completely frozen (`base.trainable = False`).
- **Trainable Components:** Dense heads only (`Dense(512)`, `Dense(4, softmax)`, `Dense(128, emb)`).
- **Trainable Parameters:** 1,968,772.
- **Learning Rate:** $1 \times 10^{-3}$ (AdamW, weight decay $1 \times 10^{-4}$).
- **Augmentation:** Identity pass-through (No augmentation).
- **Purpose:** Direct control benchmark to isolate the effect of fine-tuning.

### Configuration B: Conservative Partial Fine-Tuning (DenseBlock 4 Only)
- **Backbone Status:** Partially unfrozen.
- **Unfrozen Layers:** DenseBlock 4 Conv2D layers (`conv5_block1_1_conv` through `conv5_block16_2_conv`, Layers 313–426).
- **Frozen Layers:** Layers 0–312 (Conv1, DenseBlock 1, Transition 1, DenseBlock 2, Transition 2, DenseBlock 3, Transition 3) remain frozen (`trainable = False`).
- **BatchNormalization Policy:** **FROZEN IN INFERENCE MODE** (`bn_layer.trainable = False`). Pretrained ImageNet running mean and variance are locked to prevent small-batch ($B=16$) statistical noise from corrupting feature distributions.
- **Trainable Parameters:** 4,098,692 (1,968,772 heads + 2,129,920 Conv2D in DenseBlock 4).
- **Learning Rate Strategy:** Substantially reduced learning rate:
  - $\eta = 1 \times 10^{-4}$ (10x smaller than head initialization rate) to prevent destroying pretrained convolutional filters.
- **Purpose:** Primary experimental hypothesis — specialized top-level ridge extraction with minimal overfitting risk.

### Configuration C: Deep Partial Fine-Tuning (DenseBlock 3 + DenseBlock 4)
- **Backbone Status:** Deeper fine-tuning.
- **Unfrozen Layers:** DenseBlock 3 + Transition 3 + DenseBlock 4 (Layers 144–426).
- **Frozen Layers:** Layers 0–143 (Conv1 through Transition 2) remain frozen.
- **BatchNormalization Policy:** All BatchNorm layers strictly frozen.
- **Trainable Parameters:** ~6.6 million.
- **Learning Rate Strategy:** $\eta = 5 \times 10^{-5}$ (20x smaller).
- **Purpose:** Boundary exploration to test whether deeper representation adaptation outperforms top-block fine-tuning or induces overfitting given $N=64$ training subjects.

---

## 3. Strict Experimental Invariants (Held Constant Across All Arms)

The following parameters and pipelines are **strictly locked and identical** across Configurations A, B, and C:
1. **Model Trunk:** 3-scan feature fusion architecture with GlobalAveragePooling2D and Concat(3072).
2. **Input Preprocessing:** `transform_c5` (alpha compositing $\rightarrow$ foreground platen crop $\rightarrow$ CLAHE bilateral enhancement $\rightarrow$ resize to $112 \times 112$). Value scale: raw float32 $[0.0, 255.0]$.
3. **Augmentation Policy:** None (Identity pass-through).
4. **Cohort & Splits:** Exactly 80 development participants; 5-Fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Exactly 64 train and 16 validation participants per fold.
5. **Loss Functions:** Sparse Categorical Cross-Entropy (weight 1.0) + Supervised Contrastive Loss (weight 0.1, $\tau=0.07$).
6. **Class Weighting:** Inverse-frequency sample weights $w_i = \frac{N_{\text{train}}}{4 \cdot N_c}$ passed via `sample_weight=[sw, np.ones_like(sw)]`.
7. **Training Epoch Limits:** `EPOCHS = 150`, `MIN_EPOCHS = 100`, `PATIENCE = 15`.
8. **Checkpoint Selection Rule:** Peak Validation Macro F1 on the held-out development fold.
9. **Class Ordering:** Native model output `[A+, B+, AB+, O+]` remapped to Canonical `[A+, AB+, B+, O+]`.
10. **Batch Size:** $B = 16$.

---

## 4. Expected Benefits vs. Technical Risks

| Aspect | Expected Benefit | Identified Risk | Mitigation Strategy |
|:---|---|---|---|
| **Ridge Specialization** | DenseNet features adapt from natural textures to biometric ridge flow and core/delta curvature. | Risk of destroying pretrained feature representations. | Use 10x lower learning rate ($\eta = 10^{-4}$) with AdamW weight decay ($10^{-4}$). |
| **Batch Normalization** | Preserves activation scales across deep dense connections. | Small batch size ($B=16$, only 4 batches/epoch) causes noisy mini-batch statistics, destabilizing representations. | **Explicitly freeze all BatchNormalization layers** (`trainable=False`), keeping running mean/var locked. |
| **Overfitting on Small $N$** | Higher capacity could improve validation metrics. | Model has 4.1M parameters vs. 64 training participants (192 scans). | Enforce 50% Dropout on dense head, strict validation-monitored early stopping, and `MIN_EPOCHS=100` floor. |

---

## 5. Acceptance & Candidate Selection Criteria

For a fine-tuning configuration to supersede EXP-03 Arm 1 as the new development champion, it must meet **all three criteria** on 5-fold development out-of-fold cross-validation:
1. **Macro F1:** $\text{Macro F1} \ge 0.5068$ (must match or exceed EXP-03 Arm 1).
2. **Balanced Accuracy:** $\text{Balanced Accuracy} \ge 52.84\%$.
3. **Minority Class Health:** AB+ recall must remain $\ge 50.0\%$ (at least 4 out of 7 correct).
4. **Full Integrity Pass:** All 80 predictions valid, zero leakage, original checkpoints unchanged.
