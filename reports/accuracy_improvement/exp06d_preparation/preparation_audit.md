# EXP-06D: Independent Preparation & Pre-Training Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-16  
**Auditor:** Senior ML Verification Specialist & Independent Data Auditor  
**Experiment Identifier:** `EXP-06D` (Preprocessing & Multi-Scan Representation Investigation)  
**Audit Verdict:** **APPROVED FOR REVIEW — ZERO TRAINING EXECUTED (ALL GATES PASSED)**  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT PREPARATION AUDIT NOTICE
> This document audits preparation materials and synthetic test executions only. In strict compliance with user instructions, **real-data training has NOT been initiated**. The training authorization flag remains disabled (`TRAINING_AUTHORIZED = False`).

---

## 1. Executive Summary

An independent pre-experiment audit of **EXP-06D** was conducted to verify that:
1. The proposed protocol strictly isolates **exactly one principal factor** (Ridge Contrast Enhancement & Bilateral Filtering).
2. The model architecture, parameter count, backbone weights, optimizer, loss function, and participant splits remain 100% invariant compared to the EXP-06C Config A linear reference.
3. No real-data training, model fine-tuning, or checkpoint replacement occurred.
4. Quarantined cohorts (21 locked test subjects, 26 hospital subjects) were neither loaded nor accessed.
5. All 66 pre-existing model checkpoints across the project remain cryptographically bit-for-bit identical to their historical states.
6. The implementation skeleton enforces a fail-closed guard preventing execution without explicit user authorization.

---

## 2. Comprehensive Requirements Verification Matrix

| Requirement | Audited Specification | Verification Method | Result | Evidence Path | Outstanding Risk |
|:---|:---|:---|:---:|:---|:---:|
| **1. Single Principal Factor** | Change ONLY one factor: Ridge Enhancement (`transform_c5` vs. `transform_c5_raw`). | Automated AST and config spec inspection between Config A and Config B. | **PASS** | `reports/accuracy_improvement/exp06d_preparation/experiment_protocol.md` | None (0% parameter confound; exactly 1,968,772 params in both). |
| **2. Zero Real-Data Training** | Do not start real-data training or model fitting. | File system check on `experiments/exp06d/checkpoints/` (directory empty/non-existent). | **PASS** | `experiments/exp06d/checkpoints/` | None (Zero training steps executed). |
| **3. Locked Test Cohort Quarantine** | 21 locked test participants strictly quarantined (0 access). | Set intersection check between script paths and `split == 'TEST'` participant IDs. | **PASS** | `reports/accuracy_improvement/exp06d_preparation/data_and_checkpoint_safety.md` | None (Quarantine hardcoded in preflight assertions). |
| **4. Hospital Cohort Quarantine** | 26 hospital participants strictly quarantined (0 access). | Path audit verifying `data/captured/` is unreferenced in `run_exp06d.py`. | **PASS** | `reports/accuracy_improvement/exp06d_preparation/data_and_checkpoint_safety.md` | None (Completely segregated). |
| **5. Protected Checkpoint Invariance** | All 66 historical checkpoints must remain 100% unchanged. | SHA-256 hash recalculation against `exp06d_pre_preparation_hashes.json`. | **PASS** | `scratch/exp06d_pre_preparation_hashes.json` | None (All 66 models verified 100% bit-for-bit identical). |
| **6. Production Model Integrity** | Production `best_model.keras` preserved without modification. | SHA-256 hash check (`7b5ff5d94d4c...`). | **PASS** | `models/abo_positive_4class/best_model.keras` | None (0 bytes modified). |
| **7. Active Synthetic Assertions** | Unit tests must actively execute assertions (no dummy prints). | Code review and execution audit of `scratch/exp06d_synthetic_tests.py`. | **PASS** | `reports/accuracy_improvement/exp06d_preparation/synthetic_test_report.md` | None (15/15 tests executed and passed). |
| **8. Import-Time Safety** | Importing `run_exp06d` must NOT trigger training or side effects. | CLI probe: `python -c "import experiments.exp06d.run_exp06d..."`. | **PASS** | `experiments/exp06d/run_exp06d.py` | None (Clean import, exits with status 0). |
| **9. Disabled Authorization Flag** | `TRAINING_AUTHORIZED` must remain `False`. | Direct AST inspection of `run_exp06d.py` line 67. | **PASS** | `experiments/exp06d/run_exp06d.py` | None (`TRAINING_AUTHORIZED = False` hardcoded). |

---

## 3. Methodological Rigor & Confound Assessment

### 3.1 Purity of Isolated Factor
In EXP-06C, the comparison between Config A and Config B introduced a $+63.27\%$ parameter confound ($1.97\text{M}$ vs. $3.21\text{M}$ trainable parameters) because the projection head architecture was expanded.

In **EXP-06D**, this confound is completely eliminated:
- **Config A (Reference):** Enhanced C5 preprocessing $\rightarrow$ Frozen DenseNet-121 $\rightarrow$ 3072 Concatenation $\rightarrow$ Dense(512, relu) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(4, softmax) AND Linear Dense(128, None).
- **Config B (Candidate):** Raw Platen C5 preprocessing $\rightarrow$ Frozen DenseNet-121 $\rightarrow$ 3072 Concatenation $\rightarrow$ Dense(512, relu) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(4, softmax) AND Linear Dense(128, None).
- **Parameter Difference:** Exactly **0 trainable parameters** ($\Delta = 0$). Both models possess exactly $1,968,772$ trainable parameters across 6 weight/bias tensors.
- **Architectural Graph:** 100% identical. The sole difference is whether `enhance_mfs100_ridge_contrast` (CLAHE + contrast stretching + bilateral filter) is applied before resizing to $112 \times 112$.

### 3.2 Preprocessing Execution Logic
- **Enhanced C5 (Config A):** White composite $\rightarrow$ Platen Crop $\rightarrow$ CLAHE (clip=2.5) + dynamic contrast stretching ($p_1 \to 15, p_{99} \to 245$) + bilateral filter ($d=3, \sigma=25$) $\rightarrow$ Resize to $112 \times 112$.
- **Raw Platen C5 (Config B):** White composite $\rightarrow$ Platen Crop $\rightarrow$ Resize to $112 \times 112$ (raw optical platen grayscale converted to RGB, strictly preserving natural ridge gradients).

---

## 4. Checkpoint Protection Audit

All 66 protected checkpoints were cataloged with SHA-256 hashes in `scratch/exp06d_pre_preparation_hashes.json`:
- **1 Production Checkpoint:** `models/abo_positive_4class/best_model.keras`
- **5 Original E2 Baseline Checkpoints:** `models/phase3_fusion_metric_learning/E2/`
- **5 EXP-03 Arm 1 Baseline Checkpoints:** `experiments/exp03/checkpoints/arm1/`
- **5 EXP-05 Config A Checkpoints:** `experiments/exp05/checkpoints/config_a/`
- **20 EXP-06A Checkpoints:** `experiments/exp06a/checkpoints/config_[a-d]/`
- **20 EXP-06B Checkpoints:** `experiments/exp06b/checkpoints/arm_[a-d]/`
- **5 EXP-06C Config A Checkpoints:** `experiments/exp06c/checkpoints/config_a/`
- **5 EXP-06C Config B Checkpoints:** `experiments/exp06c/checkpoints/config_b/`

**Audit Verification:** 100% of the 66 checkpoints exist and match their expected hashes. Zero modifications have occurred.

---

## 5. Outstanding Risks & Audit Findings

1. **Rare Class Representation ($AB+, N=7$):**
   - The development cohort contains only 7 AB+ participants. Any evaluation of Config B must account for wide binomial confidence intervals on this minority class.
2. **Small Development Sample ($N = 80$):**
   - Statistical power is bounded by the 80 development participants. Pairwise differences of $|\Delta \text{Macro F1}| < 0.08$ may not reach statistical significance at $\alpha = 0.05$.
3. **No Clinical Generalization:**
   - As emphasized in the non-clinical disclaimer, all findings remain developmental and exploratory. The locked test cohort (21 participants) and hospital cohort (26 participants) remain locked for final independent evaluation.

---

## 6. Audit Verdict & Sign-off

- **Preparation Package:** Complete, fully documented, and verified.
- **Fail-Closed Training Guard:** Verified active (`TRAINING_AUTHORIZED = False`).
- **Real-Data Training:** **ZERO TRAINING EXECUTED.**
- **Recommendation:** EXP-06D preparation is fully audited and in compliance. Awaiting user authorization before any real-data training is scheduled.

*Audit completed autonomously following strict pairs-programming verification standards.*
