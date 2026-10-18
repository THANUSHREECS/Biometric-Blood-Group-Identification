# EXP-06C: Projection Head Architecture Ablation — Experiment Plan

**Document Version:** 1.0.0  
**Preparation Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Status:** **PREPARED & VERIFIED (AWAITING USER AUTHORIZATION)**  
**Lead Researcher:** Senior ML Research Scientist & Biometrics Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning architectural representations on biometric optical fingerprint features. All models, weights, metrics, and documentation are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, blood typing, patient triage, or healthcare decisions. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Scientific Rationale & Experiment Objective

In metric learning architectures (such as SimCLR and Supervised Contrastive Learning), the **projection head** maps high-dimensional latent representations from the backbone/fusion neck into a compact hypersphere metric space where contrastive loss is computed.

During the previous investigations:
- **EXP-06A** demonstrated that sharpening the contrastive temperature to $\tau = 0.05$ creates tighter hypersphere clusters and dramatically elevates minority class sensitivity (boosting AB+ recall to $71.43\%$).
- **EXP-06B** established that increasing the contrastive loss weight to $\lambda_{\text{supcon}} = 0.50$ significantly enhances majority class clustering (O+ recall: $75.86\%$) and achieves the highest overall accuracy ($52.50\%$) and pooled Macro F1 ($0.4925$).

However, throughout EXP-05, EXP-06A, and EXP-06B, the projection head remained a **single linear projection layer** ($\mathbf{W}_{\text{proj}} \in \mathbb{R}^{3072 \times 128}$). In modern self-supervised and supervised contrastive literature (Chen et al., 2020; Khosla et al., 2020), a **nonlinear multi-layer perceptron (MLP)** projection head is standard because:
1. An intermediate nonlinear activation (e.g. ReLU) discards task-irrelevant nuisance transformations (e.g., sensor noise, pressure variations, local ridge distortions) before the metric loss is computed.
2. It decouples the representation space used by the linear classification head ($\mathbf{z}_{\text{clf}} \in \mathbb{R}^{512}$) from the metric embedding space ($\mathbf{z}_{\text{emb}} \in \mathbb{R}^{128}$), allowing the feature representations to retain rich discriminatory information rather than forcing the neck representation to be strictly linear with respect to contrastive similarity.

### Core Objective
To evaluate whether replacing the single-layer linear projection head with a **two-layer nonlinear MLP projection head** (`Dense(512, relu) -> Dense(128, None)`) improves participant-level classification accuracy, balanced accuracy, and pooled out-of-fold Macro F1 compared with the verified linear baseline under optimal loss parameters ($\tau = 0.05, \lambda_{\text{supcon}} = 0.50$).

---

## 2. Experimental Configurations

| Dimension | Config A (Linear Reference) | Config B (Nonlinear MLP) | Control Mechanism |
|:---|:---|:---|:---:|
| **Baseline Provenance** | EXP-06B Arm D (Leading Candidate) | New Architectural Candidate | Direct Head-to-Head Comparison |
| **Fusion Neck** | 3-scan Concatenation ($\mathbb{R}^{3072}$) | 3-scan Concatenation ($\mathbb{R}^{3072}$) | Identical |
| **Classifier Branch** | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | Identical |
| **Projection Head** | **Linear:** `Dense(128, activation=None)` | **2-Layer MLP:** `Dense(512, relu) -> Dense(128, None)` | **Independent Variable** |
| **Final Embedding Dim** | $128$ | $128$ | Identical ($d=128$) |
| **SupCon Temperature ($\tau$)** | $0.05$ | $0.05$ | Identical |
| **SupCon Loss Weight ($\lambda$)**| $0.50$ | $0.50$ | Identical |
| **Trainable Parameters** | **1,968,772** (6 variables) | **3,214,468** (8 variables) | $+1,245,696$ parameters ($+63.3\%$) |
| **Frozen Backbone Params** | **7,037,504** (DenseNet-121) | **7,037,504** (DenseNet-121) | Identical (Frozen) |

---

## 3. Strict Experimental Controls & Quarantine Protocols

1. **Approved Cohort Isolation:**
   - Exclusively utilize the approved **80 development participants** (240 scans).
   - The **21-participant locked test cohort** (63 scans) and **26-participant hospital cohort** are strictly quarantined and inaccessible. Zero scans from these cohorts will be loaded or read.
2. **Participant-Grouped Folds:**
   - Use the fixed 5-fold `StratifiedGroupKFold` split (`seed=42`).
   - Every scan from a given participant must remain strictly within the same fold (64 train / 16 validation participants per fold). Zero participant leakage across folds.
3. **Backbone Freeze Guarantee:**
   - DenseNet-121 backbone remains strictly frozen across all folds and configurations (`base.trainable = False`). Zero backbone gradients or weight updates.
4. **Classifier Head Initialization Parity:**
   - Within each fold $f \in \{1..5\}$, the classification branch layers (`dense_512` and `clf`) will share identical epoch-zero initialization weights between Config A and Config B via shared ephemeral JSON templates.
   - The projection heads differ in parameter shape and will be initialized using deterministic random seeds ($S_f + 2$ for Config A; $S_f + 2$ and $S_f + 3$ for Config B).
5. **Optimization & Training Policy:**
   - Optimizer: AdamW (`learning_rate=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`).
   - Loss: Sparse Categorical Cross-Entropy with balanced class weights + $\lambda \cdot \mathcal{L}_{\text{SupCon}}(\tau=0.05)$.
   - Training Floor: Minimum 100 epochs, maximum 150 epochs, patience 15 epochs on Validation Macro F1.

---

## 4. Evaluation & Statistical Hypothesis Testing Plan

1. **Primary Evaluation Metric:**
   - **Pooled Out-of-Fold (OOF) Macro F1** aggregated across all 80 development participants.
   - Secondary metrics: Pooled Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, per-class Precision/Recall/F1/Support, and confusion matrices.
   - Metric Distinction: Explicitly report pooled OOF metrics separately from mean fold-level metrics ($\text{Mean} \pm \text{Std}$).
2. **Paired Statistical Comparison:**
   - Pairwise participant-level hypothesis testing between Config A and Config B:
     - Exact Binomial Test if discordant pairs $b + c < 25$.
     - McNemar Test with Edwards Continuity Correction if discordant pairs $b + c \ge 25$.
   - Significance threshold: $\alpha = 0.05$.
3. **Reporting Standards:**
   - Do not claim statistical superiority without mathematical evidence ($p < \alpha$).
   - Recompute all statistical tests directly from the saved OOF prediction CSV files.

---

## 5. Decision Gate & Stop Condition

> [!IMPORTANT]
> **MANDATORY STOP CONDITION:**
> This experiment is currently in a **PREPARATION-ONLY** state.
> No real-data training will be launched without explicit authorization from the user.
> When authorization is granted, execution will proceed via:
> ```bash
> python experiments/exp06c/run_exp06c.py --config all
> ```
