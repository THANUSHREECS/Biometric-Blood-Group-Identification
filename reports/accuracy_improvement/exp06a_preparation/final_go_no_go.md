# EXP-06A Final Pre-Training Review & GO/NO-GO Recommendation

**Document Version:** 1.0.0  
**Date of Decision:** 2026-10-15  
**Experiment Identifier:** `EXP-06A` (Supervised Contrastive Temperature Ablation)  
**Recommendation Status:** **GO (CONDITIONAL ON EXPLICIT USER AUTHORIZATION)**  
**Auditor:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> **RESEARCH USE ONLY — NON-CLINICAL DISCLAIMER**  
> This experiment investigates deep learning hyperparameter sensitivity on optical biometric representations. The models and metrics discussed herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Executive Summary & Review Purpose

This document presents the final pre-training review for **EXP-06A: Supervised Contrastive Temperature Ablation**. The experiment systematically investigates whether tuning the contrastive loss temperature hyperparameter ($\tau \in \{0.05, 0.07, 0.10, 0.15\}$) can improve four-class blood group classification accuracy and Macro F1 over the reference Concatenation baseline established in EXP-05 Configuration A.

No real-data training has been executed. All preparations were conducted under strict quarantine rules and evaluated via synthetic dummy tensors.

---

## 2. Pre-Training Audit Summary

| Audit Domain | Requirement | Audit Findings & Evidence | Status |
|:---|:---|:---|:---:|
| **Reference Fidelity** | Authenticate EXP-05 Config A metrics from raw OOF predictions. | Accuracy: **52.50%**, Balanced Accuracy: **51.84%**, Macro F1: **0.5107** independently confirmed to 6 decimal places. | **PASS** |
| **Loss Operational Locus** | Verify whether contrastive loss operates on scans or fused embedding. | Confirmed: Operates on the single 128-d **fused participant embedding** (`emb(fused)`), comparing subjects across the batch. | **PASS** |
| **Architectural Parity** | Verify all 4 configs share identical architecture and parameter count. | All 4 configs use Concatenation baseline: exactly **1,968,772 trainable parameters** across **6 variables**. | **PASS** |
| **Backbone Invariance** | Verify DenseNet-121 backbone is strictly frozen across all configurations. | DenseNet-121 `trainable=False`, **7,037,504 non-trainable parameters**, 0 trainable backbone parameters, 0 trainable BN layers. | **PASS** |
| **Cohort Quarantine** | Verify 0 access to locked test and hospital cohorts. | 21 test participants (63 scans) and 26 hospital participants (78 scans) quarantined with 0 access. Exactly 80 dev subjects used. | **PASS** |
| **Fold Collocation** | Verify all 3 scans of each participant remain collocated. | StratifiedGroupKFold (5 splits, seed 42) ensures 64 train / 16 val per fold; $\text{train\_pids} \cap \text{val\_pids} = \emptyset$. | **PASS** |
| **Initialization Parity** | Enforce equivalent epoch-zero weights across configurations within folds. | Ephemeral template weights generated per fold ensure all 4 configs start from identical points in parameter space. | **PASS** |
| **Synthetic Safety Gates** | Pass all 12 implementation safety checks on dummy data. | Loss finiteness, gradient finiteness, temperature monotonicity, and checkpoint save/restore confirmed 100%. | **PASS** |
| **Checkpoint Immutability** | Cryptographic verification of all existing source checkpoints. | All 17 protected checkpoints and reference artifacts matched baseline SHA-256 hashes with 100% fidelity. | **PASS** |

---

## 3. Risk Assessment & Mitigation Matrix

| Potential Risk | Severity | Empirical Mitigation Strategy |
|:---|:---:|:---|
| **Gradient Instability at Low $\tau$ ($\tau=0.05$)** | Medium | Contrastive loss uses numerical stabilization (`sim - tf.stop_gradient(sim_max)`). Gradient norm clipping and AdamW weight decay prevent exploding updates. Synthetic test confirmed finite gradients. |
| **Oversmoothing at High $\tau$ ($\tau=0.15$)** | Low | Multi-task loss retains full-weight classification loss ($\mathcal{L}_{\text{CE}}$ with inverse-frequency weighting), ensuring cross-entropy guides class separation even if contrastive gradients soften. |
| **Resource & Compute Overhead** | Low | 4 configurations $\times$ 5 folds = 20 folds total. With frozen backbone and fast concatenation head, each fold runs in $\sim 4.5$ minutes ($\sim 22.5$ min per configuration; total experiment runtime estimated at $\sim 90 - 110$ minutes). |
| **Evaluation Bias from Accuracy Metric** | High | Macro F1 is enforced as the primary selection criterion, supported by balanced accuracy and minority class recall (A+, AB+). No model will be selected on raw accuracy alone. |
| **Statistical False Positives** | High | Pairwise continuity-corrected McNemar tests will be conducted across aligned 80-participant predictions. No claim of statistical significance will be made unless supported by $p < 0.05$. |

---

## 4. Execution Plan (Post-Authorization)

Upon receiving user authorization, execution will proceed strictly according to the audited design:

1. **Training Script:** `experiments/exp06a/run_exp06a.py`
2. **Execution Command:**
   ```bash
   python experiments/exp06a/run_exp06a.py --config all
   ```
3. **Checkpoint Directory Hierarchy:**
   - `experiments/exp06a/checkpoints/config_a/` ($\tau = 0.07$, Reference)
   - `experiments/exp06a/checkpoints/config_b/` ($\tau = 0.05$, Lower Temp)
   - `experiments/exp06a/checkpoints/config_c/` ($\tau = 0.10$, Higher Temp)
   - `experiments/exp06a/checkpoints/config_d/` ($\tau = 0.15$, Highest Temp)
4. **Report Directory Hierarchy:**
   - `reports/accuracy_improvement/exp06a/`
     - `config_a_oof_predictions.csv`
     - `config_b_oof_predictions.csv`
     - `config_c_oof_predictions.csv`
     - `config_d_oof_predictions.csv`
     - `metrics_comparison.csv`
     - `experiment_config.json`
     - `final_report.md`
     - `confusion_matrices/`

---

## 5. Final Recommendation

### **RECOMMENDATION: GO (READY FOR TRAINING AUTHORIZATION)**

All pre-training safety gates, theoretical designs, fold isolations, and synthetic safety checks are complete and verified. The experiment is ready for execution pending explicit user authorization.
