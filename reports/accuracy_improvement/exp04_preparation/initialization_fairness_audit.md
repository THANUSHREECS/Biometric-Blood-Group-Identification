# EXP-04: Initialization Fairness & Confounder Isolation Audit

**Audit Date:** 2026-10-14T21:36:00+05:30  
**Target:** Rigorous Verification of Initial Backbone and Head Weights Across Configurations A, B, and C  
**Audit Finding:** **PASS — STRICT INITIALIZATION EQUIVALENCE PROTOCOL LOCKED**  

---

## 1. Executive Summary & Core Mandate

To ensure that performance variations observed across EXP-04 are solely caused by the degree of backbone fine-tuning, **all three configurations (A, B, and C) must begin training from identical initial parameter states within each fold**.

### Critical Prohibition:
- **Zero Checkpoint Transfer:** Under no circumstances will Configuration B or Configuration C be initialized ("warm-started") from trained EXP-03 Arm 1 checkpoints.
- Initializing Config B or C from a trained checkpoint while Config A starts from scratch would confer 300 effective training epochs, pre-converged classification heads, and biased metric embeddings to the fine-tuned arms. This would completely invalidate the causal comparison.
- **Both backbone and heads must originate from the exact same initial state for all three configurations.**

---

## 2. Complete Initialization Protocol per Fold

Within each development fold $k \in \{1, 2, 3, 4, 5\}$:

1. **Random Seed State:**
   - Global seed: `SEED = 42`.
   - Per-fold offset: `tf.random.set_seed(SEED + fold_idx)` and `np.random.seed(SEED + fold_idx)`.
   - Clears existing Keras session state via `tf.keras.backend.clear_session()`.

2. **Backbone Weights:**
   - Initialized strictly from Keras official ImageNet pretrained weights (`tf.keras.applications.DenseNet121(weights="imagenet")`).
   - All BatchNormalization layers maintain their pre-calibrated ImageNet population moving statistics ($\hat{\mu}, \hat{\sigma}^2$) and affine parameters ($\gamma, \beta$).

3. **Head Weights:**
   - `dense_512` (`Dense(512, activation='relu')`): Glorot uniform kernel initialization, zeros bias.
   - `clf` (`Dense(4, activation='softmax')`): Glorot uniform kernel initialization, zeros bias.
   - `emb` (`Dense(128, activation=None)`): Glorot uniform kernel initialization, zeros bias.

4. **Programmatic Bit-for-Bit Head Weight Sharing:**
   To guarantee that initial head weights do not diverge due to layer instantiation ordering across configurations:
   - At the beginning of Fold $k$, an initial reference model instance is constructed under `SEED + fold_idx`.
   - Its initial weights are saved to a temporary file: `scratch/init_weights_fold_{k}.weights.h5`.
   - When Config A, Config B, and Config C are instantiated for Fold $k$, each loads this exact initial weight state before applying its respective layer trainability and optimizer policies.
   - This ensures **100.000% bit-for-bit identical starting weights** across all three configurations for both backbone and heads.

---

## 3. The Isolated Experimental Variable

To preserve scientific rigor, exactly ONE independent variable is varied across the three configurations:

| Dimension | Configuration A (Reference) | Configuration B (Conservative FT) | Configuration C (Deep FT) | Invariance Status |
|:---|:---:|:---:|:---:|:---:|
| **Initial Backbone Weights** | ImageNet Pretrained | ImageNet Pretrained | ImageNet Pretrained | **IDENTICAL** |
| **Initial Head Weights** | Seeded Glorot / He (`SEED + fold_idx`) | Seeded Glorot / He (`SEED + fold_idx`) | Seeded Glorot / He (`SEED + fold_idx`) | **IDENTICAL** |
| **Participant Folds** | 5 StratifiedGroupKFold splits (64/16) | 5 StratifiedGroupKFold splits (64/16) | 5 StratifiedGroupKFold splits (64/16) | **IDENTICAL** |
| **Input Preprocessing** | `transform_c5` (112x112, $[0, 255]$) | `transform_c5` (112x112, $[0, 255]$) | `transform_c5` (112x112, $[0, 255]$) | **IDENTICAL** |
| **Data Augmentation** | None (Identity pass-through) | None (Identity pass-through) | None (Identity pass-through) | **IDENTICAL** |
| **Batch Size & Shuffle** | 16, deterministic shuffle | 16, deterministic shuffle | 16, deterministic shuffle | **IDENTICAL** |
| **Loss Formulation** | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | **IDENTICAL** |
| **Sample Weighting** | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | Inverse-frequency $w_i = \frac{N}{4 N_c}$ | **IDENTICAL** |
| **Head Optimizer & LR** | AdamW, $\text{LR} = 10^{-3}$, $\lambda=10^{-4}$ | AdamW, $\text{LR} = 10^{-3}$, $\lambda=10^{-4}$ | AdamW, $\text{LR} = 10^{-3}$, $\lambda=10^{-4}$ | **IDENTICAL** |
| **Batch Normalization** | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | **IDENTICAL** |
| **Stopping & Selection** | 150 ep, 100 min, Val Macro F1 | 150 ep, 100 min, Val Macro F1 | 150 ep, 100 min, Val Macro F1 | **IDENTICAL** |
| **Backbone Trainability** | **FROZEN (0 layers)** | **DenseBlock 4 Conv2D (32 layers)**| **DenseBlock 3+4 + Trans 3 (81 layers)** | **CONTROLLED VARIABLE** |
| **Backbone Optimizer & LR** | N/A (Frozen) | **AdamW, $\text{LR} = 10^{-4}$, $\lambda=10^{-4}$** | **AdamW, $\text{LR} = 5\times 10^{-5}$, $\lambda=10^{-4}$** | **CONTROLLED VARIABLE** |

---

## 4. Preservation and Isolation of Prior Checkpoints

- **EXP-03 Checkpoints:** Preserved in `experiments/exp03/checkpoints/` and verified with pre-audit SHA-256 hashes.
- **Original E2 Checkpoints:** Preserved in `models/phase3_fusion_metric_learning/E2/`.
- **Production Checkpoint:** Preserved in `models/abo_positive_4class/best_model.keras`.
- **EXP-04 Destination:** All new checkpoints will be written to isolated subdirectories:
  - `experiments/exp04/checkpoints/config_a/`
  - `experiments/exp04/checkpoints/config_b/`
  - `experiments/exp04/checkpoints/config_c/`
- No write operations or in-place reads of prior checkpoints will occur during EXP-04 execution.

---

## 5. Audit Verdict

The initialization protocol guarantees strict parameter equivalence at epoch zero across all three configurations. The experimental design completely eliminates initialization bias, warm-start leakage, and head-learning-rate confounding.

- **Audit Verdict:** **PASS (INITIALIZATION FAIRNESS FORMALIZED)**.
