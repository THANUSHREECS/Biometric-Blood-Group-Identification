# EXP-04: Final Implementation Safety Gate Audit & Go / No-Go Recommendation

**Experiment:** EXP-04 — Controlled Partial Fine-Tuning of E2  
**Audit Timestamp:** 2026-10-14T21:56:00+05:30  
**Current Status:** **RECOMMENDATION: GO (ALL 5 SAFETY GATES CLEARED — PENDING USER AUTHORIZATION)**  
**Training Status:** **ZERO TRAINING PERFORMED**  
**Script Creation Status:** **NOT CREATED YET** (`experiments/exp04/run_exp04.py` deliberately withheld)  
**Quarantine Status:** **ACTIVE & LOCKED** (Zero access to 21-participant locked test set or 26-cohort)  
**Checkpoint Safety:** **VERIFIED 100% UNTOUCHED** (All baseline SHA-256 hashes confirmed intact)  

---

## 1. Executive Summary

This report concludes the final implementation-level safety gate audit for EXP-04. Prior to creating the training script or launching full model training, five rigorous programmatic verification checks were conducted on synthetic batches in Keras 3.12.4 and TensorFlow 2.21. All checks passed with zero errors.

---

## 2. Definitive Safety Gate Results

### Gate 1: Dual-Optimizer Variable Partitioning
- **Verification Method:** Evaluated across explicit layer objects in `E2DualOptimizerModel` for Configurations A, B, and C.
- **Finding:**
  - Config A: Exactly 0 backbone trainable variables, 6 head variables (1,968,772 params).
  - Config B: Exactly 32 Conv2D kernel variables (2,129,920 params), 6 head variables.
  - Config C: Exactly 81 Conv2D kernel variables (5,455,872 params), 6 head variables.
  - Variable Group Overlap: **0 variables** ($\text{Backbone} \cap \text{Heads} = \emptyset$).
  - Variable Group Completeness: **100%** ($\text{Backbone} \cup \text{Heads} = \text{Trainable}$).
  - DenseNet-121 Architecture Fact: DenseNet Conv2D layers use `use_bias=False`; each Conv2D layer contributes exactly 1 variable (`kernel`), explaining the 32 and 81 variable counts.
- **Status:** **PASS**.

### Gate 2: Gradient Finiteness & Update Dynamics
- **Verification Method:** Forward/backward pass on synthetic batch ($B=4, 3\text{ scans}, 112\times 112\times 3$).
- **Finding:**
  - Total Loss, SCCE Loss, and SupCon Loss are strictly finite across all configs.
  - Gradients are finite for every intended trainable parameter.
  - 100% of head variables updated by `opt_heads` ($\Delta > 0$).
  - 100% of target Conv2D variables updated by `opt_backbone` in Configs B and C ($\Delta > 0$).
  - Frozen backbone layers (Stem Conv1) remained **bit-for-bit identical ($\Delta = 0.000000$)**.
  - Batch Normalization moving statistics ($\hat{\mu}, \hat{\sigma}^2$) in Stem and Block 4 remained **bit-for-bit identical ($\Delta = 0.000000$)**.
  - Single-optimizer update rule strictly satisfied.
- **Status:** **PASS**.

### Gate 3: Loss Formulation & Sample-Weight Single Multiplication
- **Verification Method:** Mathematical inspection and synthetic gradient check.
- **Finding:**
  - Multi-task loss weights: $1.0 \times \text{SCCE} + 0.1 \times \text{SupCon}$ ($\tau=0.07$) matching EXP-03.
  - Inverse-frequency sample weights $w_i = \frac{N}{4 N_c}$ applied directly to sample-wise cross-entropy.
  - Single-multiplication guarantee: Because loss is evaluated directly in `train_step`, Keras compile wrappers cannot duplicate sample weighting.
  - Labels and 128-dim embeddings correctly routed to Supervised Contrastive Loss.
  - Canonical class-order remapping verified ($\sum p_{\text{canonical}} = 1.0$).
- **Status:** **PASS**.

### Gate 4: Shared Initialization Across Configs A, B, and C
- **Verification Method:** Ephemeral array serialization (`scratch/test_init_fold_1_heads.npz`) and restore test.
- **Finding:**
  - Initial head layer weights saved as explicit layer arrays via `layer.get_weights()`.
  - Config A, Config B, and Config C restored to the exact same initial parameter state.
  - Maximum weight difference across all 610 model variables (backbone + heads) at Epoch 0: **$\Delta = 0.000000$ (Bit-for-bit identical)**.
  - Zero checkpoint transfer: Config B and C do NOT inherit weights from trained EXP-03 checkpoints.
- **Status:** **PASS**.

### Gate 5: Checkpoint Selection & Evaluation Compatibility
- **Verification Method:** Callback execution and model weight save/reload test.
- **Finding:**
  - `MacroF1Checkpoint` callback executes cleanly during `model.fit()`.
  - Validation predictions output proper shape `(N_val, 4)` summing to $1.0$.
  - Best-epoch weights saved to `.weights.h5` and reloaded with **$\Delta = 0.000000$**.
  - Out-of-fold aggregation pipeline 100% compatible.
- **Status:** **PASS**.

---

## 3. The Approved Experimental Matrix

| Dimension | Configuration A (Reference) | Configuration B (Conservative FT) | Configuration C (Deep FT) | Invariance Rule |
|:---|:---:|:---:|:---:|:---:|
| **Backbone Trainability** | **FROZEN (0 layers)** | **DenseBlock 4 Conv2D (32 layers)** | **DenseBlock 3+4 + Trans 3 (81 layers)** | **Independent Variable** |
| **Backbone Optimizer & LR** | N/A (Frozen) | **AdamW, $\text{LR}=10^{-4}$, $\lambda=10^{-4}$** | **AdamW, $\text{LR}=5\times 10^{-5}$, $\lambda=10^{-4}$** | **Controlled Variable** |
| **Head Trainable Layers** | `dense_512`, `clf`, `emb` | `dense_512`, `clf`, `emb` | `dense_512`, `clf`, `emb` | **IDENTICAL** |
| **Head Optimizer & LR** | **AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$** | **AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$** | **AdamW, $\text{LR}=10^{-3}$, $\lambda=10^{-4}$** | **IDENTICAL** |
| **Total Trainable Params** | **1,968,772** (21.9%) | **4,098,692** (45.5%) | **7,424,644** (82.4%) | Explicitly Counted |
| **Batch Normalization** | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | **FROZEN** |
| **Epoch-Zero Weights** | ImageNet + Fold Seed $k$ | Bit-for-bit match to Config A | Bit-for-bit match to Config A | **FAIRNESS LOCKED** |
| **Participant Splits** | 5 folds (64 train / 16 val) | Same 5 folds | Same 5 folds | **FROZEN** |
| **Preprocessing & Aug** | `transform_c5` (No aug) | `transform_c5` (No aug) | `transform_c5` (No aug) | **FROZEN** |
| **Loss Formulation** | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | 1.0 $\times$ SCCE + 0.1 $\times$ SupCon | **FROZEN** |
| **Sample Weighting** | Inverse-class freq $w_i = \frac{N}{4 N_c}$| Inverse-class freq $w_i = \frac{N}{4 N_c}$| Inverse-class freq $w_i = \frac{N}{4 N_c}$| **FROZEN** |
| **Canonical Class Order** | Index 0: A+, 1: AB+, 2: B+, 3: O+ | Index 0: A+, 1: AB+, 2: B+, 3: O+ | Index 0: A+, 1: AB+, 2: B+, 3: O+ | **FROZEN** |
| **Stopping & Selection** | 150 ep, 100 min, Val Macro F1 | 150 ep, 100 min, Val Macro F1 | 150 ep, 100 min, Val Macro F1 | **FROZEN** |
| **Quarantined Cohorts** | Test (21) & Cohort (26) EXCLUDED | Test (21) & Cohort (26) EXCLUDED | Test (21) & Cohort (26) EXCLUDED | **QUARANTINED** |

---

## 4. Final Recommendation & Go / No-Go Determination

# **STATUS: AUTHORIZED, EXECUTED, AND COMPLETED**

All methodological questions, implementation safety gates, and preflight checks were verified. User authorization was granted to proceed with script creation and model execution.

**EXECUTION SUMMARY:**
- **Training Script:** [`experiments/exp04/run_exp04.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp04/run_exp04.py) created and executed.
- **Phase 1 Preflight:** 100% Passed.
- **Phase 2 Execution:** All 3 configurations (A, B, C) executed across all 5 participant-grouped folds.
- **Phase 3 Validation:** 80 unique OOF development participants per configuration verified, finite probabilities, aligned canonical classes, and independent statistical significance tests completed.
- **Baseline Checkpoints:** All original E2 checkpoints, EXP-03 checkpoints, and production model verified 100% unaltered.
- **Comprehensive Report:** Documented in [`reports/accuracy_improvement/exp04/final_report.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp04/final_report.md).
