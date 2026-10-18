# EXP-04: Pre-Training Go / No-Go Decision Report

**Experiment:** EXP-04 — Controlled Partial Fine-Tuning of E2  
**Audit Date:** 2026-10-14T18:22:30+05:30  
**Status:** **GO — PENDING USER APPROVAL (AUDIT PASSED)**  
**Training Status:** **ZERO TRAINING PERFORMED**  
**Quarantine Status:** ACTIVE (Zero access to locked 21-participant test set or 26-cohort)  

---

## 1. Pre-Training Verification Matrix

| # | Verification Dimension | Target Requirement | Audit Finding | Status |
|:---:|:---|---|---|:---:|
| **1** | **Participant Isolation** | Zero participant overlap between train and val in every fold | Verified across all 5 folds | **PASS** |
| **2** | **OOF Prediction Completeness** | Exactly 80 OOF predictions per configuration | 80 unique participant IDs verified | **PASS** |
| **3** | **Locked Test Quarantine** | 21 test participants strictly quarantined | 0 test participants accessed | **PASS** |
| **4** | **Hospital Cohort Quarantine** | 26 hospital participants strictly quarantined | 0 cohort participants accessed | **PASS** |
| **5** | **Production Checkpoint Safety** | Hashes of 5 original E2 checkpoints unchanged | All 5 SHA-256 hashes matched | **PASS** |
| **6** | **EXP-03 Reference Checkpoint Safety** | Hashes of 5 EXP-03 Arm 1 checkpoints locked | All 5 SHA-256 hashes baselined | **PASS** |
| **7** | **Directory Isolation** | Dedicated script and checkpoint paths | `experiments/exp04/` and `reports/.../exp04/` | **PASS** |
| **8** | **Class Ordering & Remapping** | Canonical mapping $[A+, AB+, B+, O+]$ | Verified automated remapping | **PASS** |
| **9** | **Layer Trainability Precision** | DenseBlock 4 Conv2D unfrozen; BatchNorm frozen | Exactly 32 Conv2D layers unfrozen | **PASS** |
| **10** | **Batch Normalization Safety** | Running stats locked to prevent small-batch corruption | All BN layers forced non-trainable | **PASS** |
| **11** | **Learning Rate Scaling** | Lower fine-tuning rate than head rate | $\eta = 10^{-4}$ (10x smaller than head) | **PASS** |
| **12** | **Reference Configuration Fidelity** | Matches EXP-03 Arm 1 champion identically | Architecture, losses, preprocessing match | **PASS** |

---

## 2. Summary of Proposed Configurations

1. **Config A (Reference):** Frozen DenseNet-121 backbone (`base.trainable = False`), no augmentation, head learning rate $\eta = 10^{-3}$. (Replicates EXP-03 Arm 1).
2. **Config B (Conservative Partial Fine-Tuning):**
   - Unfreeze DenseBlock 4 Conv2D layers (`conv5_block1_1_conv` to `conv5_block16_2_conv`, Layers 313–426).
   - Strictly freeze all BatchNormalization layers in DenseNet-121.
   - Fine-tuning learning rate $\eta = 10^{-4}$ (10x smaller).
   - 4,098,692 trainable parameters.
3. **Config C (Deep Fine-Tuning - Optional Boundary Test):**
   - Unfreeze DenseBlock 3 + Transition 3 + DenseBlock 4 Conv2D layers (Layers 144–426).
   - Strictly freeze all BatchNormalization layers.
   - Fine-tuning learning rate $\eta = 5 \times 10^{-5}$ (20x smaller).
   - ~6.6 million trainable parameters.

---

## 3. Risks & Expected Benefits Summary

- **Expected Benefit:** DenseNet-121 top-level feature maps specialize to biometric ridge curvature, core/delta flow, and dermatoglyphic friction structures, potentially improving class separation beyond ImageNet natural texture representations.
- **Identified Risks:**
  1. *Overfitting:* Small training sample size ($N=64$ participants per fold) vs. 4.1M parameters. Mitigated by freezing earlier blocks (Stem through DenseBlock 3), high dropout (0.5), and early stopping.
  2. *Batch Normalization Destabilization:* Noisy batch statistics on $B=16$ corrupting ImageNet moving averages. Mitigated by strictly locking all BatchNormalization layers to inference mode (`trainable = False`).
  3. *Weight Shock:* High initial gradients destroying pretrained filters. Mitigated by 10x smaller learning rate ($\eta = 10^{-4}$) and AdamW weight decay ($10^{-4}$).

---

## 4. Formal Decision

# **DECISION: GO (PENDING USER APPROVAL)**

All pre-training safety, isolation, architecture, and protocol checks have **PASSED**.
**No training has started. Awaiting user authorization before creating execution code or launching model training.**
