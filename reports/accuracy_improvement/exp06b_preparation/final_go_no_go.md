# EXP-06B: Final Pre-Training Review & GO/NO-GO Recommendation

**Document Version:** 1.0.0  
**Date of Decision:** 2026-10-15  
**Experiment Identifier:** `EXP-06B` (Supervised Contrastive Loss Weight Ablation)  
**Recommendation Status:** **GO (CONDITIONAL ON EXPLICIT USER AUTHORIZATION)**  
**Lead Auditor:** Senior ML Researcher & Data Integrity Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning hyperparameter sensitivity on biometric optical fingerprint representations. All models, weights, metrics, and conclusions documented herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, triage, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Executive Summary & Review Purpose

This document presents the final pre-training review for **EXP-06B: Supervised Contrastive Loss Weight Ablation**. 

The experiment systematically investigates whether tuning the contrastive loss weight ($\lambda_{\text{supcon}} \in \{0.05, 0.10, 0.20, 0.50\}$) with fixed contrastive temperature $\tau = 0.05$ improves participant-level out-of-fold Macro F1, balanced accuracy, and per-class recall over the reference Concatenation baseline established in EXP-06A Config B ($\lambda = 0.10, \tau = 0.05$).

**Stop Condition Compliance:** No real-data training has been initiated. All preparations were executed strictly under quarantine protocols and validated via synthetic tensors.

---

## 2. Pre-Training Audit Summary

| Audit Domain | Requirement | Audit Findings & Evidence | Status |
|:---|:---|:---|:---:|
| **Baseline Recomputation** | Recompute EXP-06A metrics & McNemar tests directly from OOF CSVs. | 100% exact numerical match across all 4 configs and 3 statistical tests. | **PASS** |
| **Config A Discrepancy** | Formally investigate difference between EXP-05 and EXP-06A Config A. | Net difference of 3 participants ($42 \to 39$, $\Delta = 3.75\%$) confirmed statistically indistinguishable from zero ($p=0.579$), attributable to stochastic dropout and batch shuffling. | **PASS** |
| **Controlled Design** | Fix $\tau = 0.05$, frozen DenseNet-121, isolate $\lambda_{\text{supcon}} \in \{0.05, 0.10, 0.20, 0.50\}$. | Full formal mathematical specification established in [controlled_experiment_design.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06b_preparation/controlled_experiment_design.md). | **PASS** |
| **Fold Isolation & Quarantine** | 80 Dev participants ONLY; zero access to 21 locked test or 26 hospital participants. | Complete set intersection audit: Dev $\cap$ Test = $\emptyset$ (0 overlap). Scans per participant 100% collocated. | **PASS** |
| **Initialization Parity** | Identical head weights at Step 0 across all 4 arms within each fold. | Ephemeral weight template protocol specified and verified via synthetic save/load test. | **PASS** |
| **Synthetic Safety Gates** | 12-gate implementation verification on synthetic tensors. | All 12 gates passed with 100% compliance via `scratch/exp06b_safety_gate_test.py`. | **PASS** |

---

## 3. Risk Analysis & Mitigation Matrix

| Potential Risk | Severity | Mitigation Strategy Enforced |
|:---|:---:|:---|
| **Over-regularization at $\lambda=0.50$** | High | Heavy metric regularization could distort class boundaries. Model checkpoints are strictly selected by Validation Macro F1 with a minimum 100-epoch convergence window. |
| **Under-regularization at $\lambda=0.05$** | Medium | Lighter contrastive regularization could cause minority classes (AB+, A+) to collapse. Automated StratifiedGroupKFold class weighting ($w_c$) prevents majority bias during CE backpropagation. |
| **Cohort Contamination** | Critical | Quarantines are hardcoded. Split assertions verify exactly 80 development participants and reject any test or hospital IDs before training starts. |
| **Orphaned Scratch Weights** | Low | Script automatically deletes all `scratch/exp06b_init_fold_*.weights.h5` files once all 4 arms have completed execution. |

---

## 4. Final Recommendation & Decision Gate

The experimental architecture, data quarantine, initialization parity protocol, and loss mechanics are **fully verified and ready for execution**.

In strict accordance with the mandatory stop condition:
- **No real-data training has been launched.**
- **No protected checkpoints have been altered.**
- **The experiment is in an IDLE state awaiting explicit user authorization.**

### Recommendation: **GO (AWAITING USER AUTHORIZATION)**

When authorized by the user, the execution will proceed using:
```bash
python experiments/exp06b/run_exp06b.py --arm all
```
Across the four audited arms:
- **Arm A:** $\lambda = 0.05$
- **Arm B (Reference):** $\lambda = 0.10$
- **Arm C:** $\lambda = 0.20$
- **Arm D:** $\lambda = 0.50$
