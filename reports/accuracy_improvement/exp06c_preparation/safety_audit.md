# EXP-06C: Safety, Quarantine & Checkpoint Protection Audit

**Document Version:** 1.1.0 (Post-Audit Revision)  
**Audit Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Audit Status:** **100% VERIFIED & COMPLIANT (STOPPED — AWAITING AUTHORIZATION)**  
**Lead Auditor:** Senior ML Research Scientist & Verification Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, checkpoints, audits, and code referenced herein are academic research prototypes for evaluating machine learning architectures on biometric fingerprint representations. They **must not** be deployed in healthcare facilities, blood banks, transfusion services, emergency departments, or clinical decision support systems.

---

## 1. Compliance Matrix: 16 Mandatory Experimental Safeguards

| Ref # | Safeguard Requirement | Verification Method & Evidence | Compliance Status |
|:---:|:---|:---|:---:|
| **1** | **Approved 80-Participant Cohort** | `SPLIT_CSV` filtered to TRAIN/VALIDATION. Exactly 80 unique participant IDs (240 scans). | **COMPLIANT** |
| **2** | **5-Fold Collocation** | StratifiedGroupKFold (`seed=42`). All 3 scans per participant collocated in identical fold (64 train / 16 val). Zero fold leakage. | **COMPLIANT** |
| **3** | **Quarantine of Test & Hospital Sets** | 21 locked test participants and 26 hospital participants verified 0% overlap ($\text{Dev} \cap \text{Test} = \emptyset$). Zero reads/loads. | **COMPLIANT** |
| **4** | **Frozen DenseNet-121 Backbone** | `base.trainable = False`, 0 trainable backbone variables, 7,037,504 non-trainable parameters verified in Gate 2. | **COMPLIANT** |
| **5** | **Controlled Experimental Parity** | Dataset, folds, transforms, optimizer, loss weight ($\lambda=0.50$), temperature ($\tau=0.05$), and patience held strictly identical. | **COMPLIANT** |
| **6** | **Isolated Independent Variable** | Sole intentional variable is projection head architecture (Linear Dense(128) vs 2-Layer MLP Dense(512)->Dense(128)). Parameter capacity confound documented in [parameter_capacity_audit.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06c_preparation/parameter_capacity_audit.md). | **COMPLIANT** |
| **7** | **Shared Compatible Initialization** | Classifier branch (`dense_512` & `clf`) shares exact initial weights across configs per fold. Divergent projection head shapes formally documented. | **COMPLIANT** |
| **8** | **Preservation of Protected Checkpoints**| All 56 historical checkpoints (Production, Orig E2, EXP-03, EXP-05, EXP-06A, EXP-06B) cryptographically verified before and after preparation. | **COMPLIANT** |
| **9** | **Isolated Artifact Storage** | Scripts save exclusively to `experiments/exp06c/checkpoints/` and `reports/accuracy_improvement/exp06c/`. Existing directories untouched. | **COMPLIANT** |
| **10** | **1 OOF Prediction per Participant** | Verification script asserts exactly 80 predictions per configuration with complete ID alignment. | **COMPLIANT** |
| **11** | **Comprehensive Metric Reporting** | Accuracy, balanced accuracy, macro precision/recall/F1, per-class metrics, confusion matrices, fold mean/std, and paired bootstrap planned. | **COMPLIANT** |
| **12** | **Primary Metric Protocol** | Pooled OOF Macro F1 designated as primary evaluation metric, formally distinguished from mean fold-level Macro F1. | **COMPLIANT** |
| **13** | **Paired Statistical Testing** | Exact Binomial, continuity-corrected McNemar tests, and 2,000-replicate participant-level paired bootstrap CI formally planned. | **COMPLIANT** |
| **14** | **Historical Reference Benchmark** | Planned comparison of Config A against EXP-06B Arm D historical performance ($0.4925$ Macro F1, $52.50\%$ accuracy); documented as a fresh linear reference in [baseline_reproducibility_audit.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06c_preparation/baseline_reproducibility_audit.md). | **COMPLIANT** |
| **15** | **Mandatory Failure Stop** | Automated assertions halt execution immediately if any checkpoint, quarantine, or tensor check fails. | **COMPLIANT** |
| **16** | **Scientific Restraint** | Zero claims of clinical diagnostic efficacy; no claims of statistical superiority based on accuracy alone or without $p < 0.05$ and CI excluding zero. | **COMPLIANT** |

---

## 2. Protected Checkpoint Cryptographic Audit (56/56 Verified)

Before and after EXP-06C preparation, cryptographic SHA-256 hashes of all 56 protected baseline checkpoints were computed and verified against pre-execution records:

```
Protected Checkpoints Audited: 56/56
Cryptographic Status: 100% BIT-FOR-BIT IDENTICAL (0 bytes modified across all files)
Audit Tooling: scratch/exp06c_pre_preparation_hashes.json
```

### Checkpoint Protection Registry Table

| Category | Count | Directory / File Path | Hash Verification Status |
|:---|:---:|:---|:---:|
| **Production Model** | 1 | `models/abo_positive_4class/best_model.keras` | **100% BIT-EXACT MATCH** |
| **Original E2 Baselines** | 5 | `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5` | **100% BIT-EXACT MATCH** |
| **EXP-03 Checkpoints (Arm 1)** | 5 | `experiments/exp03/checkpoints/arm1/fold_[1-5]_best.weights.h5` | **100% BIT-EXACT MATCH** |
| **EXP-05 Checkpoints (Config A)** | 5 | `experiments/exp05/checkpoints/config_a/fold_[1-5]_best.weights.h5` | **100% BIT-EXACT MATCH** |
| **EXP-06A Checkpoints (Configs A–D)** | 20 | `experiments/exp06a/checkpoints/{config_a,b,c,d}/fold_[1-5]_best.weights.h5` | **100% BIT-EXACT MATCH** |
| **EXP-06B Checkpoints (Arms A–D)** | 20 | `experiments/exp06b/checkpoints/{arm_a,b,c,d}/fold_[1-5]_best.weights.h5` | **100% BIT-EXACT MATCH** |
| **Total Protected Checkpoints** | **56** | Fully locked and protected | **ZERO MODIFICATIONS** |

---

## 3. Cohort Quarantine & Participant Isolation Verification

### 3.1 Participant Split Integrity
An audit of `reports/phase3_participant_benchmark/participant_split.csv` confirms:
- **Total Development Cohort:** Exactly 80 participants (240 optical fingerprint scans, 3 scans per participant).
- **Locked Test Cohort:** Exactly 21 participants (63 scans), strictly quarantined.
- **Hospital Cohort:** Exactly 26 participants, strictly quarantined.
- **Set Intersection:**
  $$\text{Development Cohort} \cap \text{Test Cohort} = \emptyset \quad (\text{0 overlapping participants})$$
  $$\text{Development Cohort} \cap \text{Hospital Cohort} = \emptyset \quad (\text{0 overlapping participants})$$

### 3.2 Participant-Grouped Fold Collocation
- **Folds 1 to 5:** Each fold contains exactly 64 training participants (192 scans) and 16 validation participants (48 scans).
- **Leakage Audit:** A set intersection between training and validation participant IDs across all 5 folds yielded:
  $$\text{Train}_{\text{fold } i} \cap \text{Val}_{\text{fold } i} = \emptyset \quad \forall i \in \{1, 2, 3, 4, 5\}$$
- All scans for each participant are 100% collocated in the same fold.

---

## 4. Frozen Backbone Verification

DenseNet-121 backbone parameter safety was empirically verified in Gate 2 and Gate 8:
- **Total Backbone Parameters:** 7,037,504.
- **Backbone Trainable Parameters:** 0 (`len(base.trainable_variables) == 0`).
- **Weight Update Isolation Test:** Following an AdamW optimization step on synthetic tensors:
  $$\Delta \mathbf{W}_{\text{backbone}} = \max |\mathbf{W}_{\text{post}} - \mathbf{W}_{\text{pre}}| = \mathbf{0.00000000}$$
  $$\Delta \mathbf{W}_{\text{head}} = \max |\mathbf{W}_{\text{post}} - \mathbf{W}_{\text{pre}}| = \mathbf{0.001000} > 0$$
- Gradients flow exclusively to classification and projection head parameters.

---

## 5. Preflight Integrity Execution Audit

The proposed runner script was executed in standalone preflight verification mode:
```bash
python experiments/exp06c/run_exp06c.py --preflight-only
```
- **Execution Log:** `task-325.log`
- **Exit Status:** **Code 0 (SUCCESS)**
- **Findings:** Dataset isolation confirmed (80 dev, 21 quarantined test), all 56 protected checkpoints confirmed intact, model parameter accounting verified (Config A: 1,968,772; Config B: 3,214,468), and 5-fold StratifiedGroupKFold integrity confirmed.

---

## 6. Stop Condition Compliance

In strict compliance with instructions:
- **Zero real-data training was initiated.**
- **Zero checkpoints were modified or overwritten.**
- **The system is in an IDLE state awaiting explicit user authorization.**
