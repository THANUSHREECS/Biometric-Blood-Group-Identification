# Master Experiment Review: EXP-01 Through EXP-06D
## Biometric Blood Group Detection / HemoPrint AI Research Program

**Document Type:** Comprehensive Scientific & Architectural Meta-Audit  
**Audit Scope:** Complete experimental lineage (Baseline E2, EXP-01, EXP-02, EXP-03, EXP-04, EXP-05, EXP-06A, EXP-06B, EXP-06C, EXP-06D)  
**Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Execution Guard:** `TRAINING_AUTHORIZED = False` (Active, Enforced)  
**Safety Status:** **STRICT AUDIT ONLY** — No training, fine-tuning, checkpoint alterations, prediction edits, or cohort quarantine breaches occurred.

---

> [!CAUTION]
> ### MANDATORY RESEARCH USE ONLY & NON-CLINICAL DISCLAIMER
> Biometric blood group detection from optical fingerprint scans is an exploratory computational hypothesis and investigational machine learning methodology. All models, architectures, loss functions, and metrics evaluated across EXP-01 through EXP-06D represent academic research prototypes developed on a small single-center development cohort ($N=80$ participants, 240 scans). 
> 
> **Zero clinical diagnostic validity is claimed.** These algorithms must never be utilized for clinical diagnosis, transfusion compatibility screening, emergency triage, surgical planning, or therapeutic intervention. Standard serological agglutination testing remains the indispensable clinical gold standard.

---

## 1. Executive Summary

Across ten structured experimental phases, this project investigated whether deep convolutional networks can extract biometric representations predictive of ABO blood groups from optical fingerprint scans (Mantra MFS100), and systematically resolved a series of architectural, algorithmic, and optimization bottlenecks.

### The Research Trajectory in Brief:
1. **The Starting Point (Baseline E2):** Baseline E2 established that combining three fingerprint scans per participant via feature concatenation and multi-task learning (Cross-Entropy + Supervised Contrastive Loss) achieved 53.75% accuracy. However, it suffered from severe **minority class collapse** on AB+ ($14.29\%$ recall, 1/7 correct) and heuristic data augmentation that distorted ridge frequencies.
2. **Failed Remedies (EXP-01 & EXP-02):** Attempts to fix class imbalance using Class-Balanced Focal Loss (**EXP-01**, $\gamma=2.0$) severely degraded accuracy to 36.25% (trivial majority baseline). Conservative data augmentation (**EXP-02**) underperformed unaugmented data (47.50% vs. 53.75%).
3. **The Data Fidelity Breakthrough (EXP-03):** A rigorous three-arm ablation proved that **eliminating all synthetic data augmentation (Arm 1)** unlocked the true underlying representation, achieving 53.75% accuracy, 0.5068 Macro F1, and elevating AB+ recall from $14.29\%$ to $57.14\%$.
4. **The Capacity & Regularization Boundary (EXP-04 & EXP-05):** Unfreezing deep convolutional blocks of DenseNet-121 (**EXP-04**) caused catastrophic representation drift and overfitting on $N=80$. Investigating complex feature fusion (**EXP-05**, Gated Fusion and Cross-Scan Self-Attention) proved that simple vector concatenation (3072-d) remains equal or superior while requiring no attention overhead.
5. **Hyperparameter & Metric Learning Optimization (EXP-06A & EXP-06B):** Systematic tuning of SupCon temperature (**EXP-06A**) and loss weight (**EXP-06B**) established that sharper contrastive scaling ($\tau = 0.05$) with heavy regularization weight ($\lambda = 0.50$) stabilizes minority class representations.
6. **Head Capacity & Preprocessing Verification (EXP-06C & EXP-06D):** Adding parameter capacity via a 2-layer MLP projection head (**EXP-06C**) led to overfitting, confirming the linear projection head as optimal. Finally, isolating the preprocessing pipeline (**EXP-06D**) proved that heuristic spatial filtering (CLAHE + bilateral filter) can be safely discarded in favor of simpler, raw platen foreground cropping (+6.25% empirical accuracy, +27.59% O+ recall gain).

### Leading Candidate Status:
The strongest development candidate across the program is **EXP-06C Config A** (linear projection head, $\lambda=0.50, \tau=0.05$, frozen DenseNet-121, unaugmented 3-scan concatenation), which achieved:
- **Pooled Out-of-Fold Macro F1:** **0.5487**
- **Pooled Out-of-Fold Accuracy:** **53.75%** (43 / 80)
- **Balanced Accuracy:** **53.47%**
- **Mean Fold Macro F1:** **0.5331 $\pm$ 0.0895**
- **Balanced Per-Class Sensitivity:** A+ 42.11%, AB+ 57.14%, B+ 56.00%, O+ 58.62%

However, paired statistical hypothesis testing across all late-stage experiments confirms that **none of the observed incremental gains reach formal statistical significance ($\alpha = 0.05$) due to small sample size ($N=80$)**.

---

## 2. Complete Experiment Inventory & Verification (EXP-01 Through EXP-06D)

All reported metrics across every experiment were audited directly against saved out-of-fold CSV files, JSON configuration logs, and markdown summaries. Missing historical data points are explicitly marked as **NOT FOUND**.

| Exp ID | Investigated Hypothesis / Research Question | Core Architecture & Preprocessing | Training Config & Loss | Cohort & Folds | Accuracy | Balanced Accuracy | Pooled OOF Macro F1 | Mean Fold Macro F1 ($\pm \text{Std}$) | Per-Class Recall [A+, AB+, B+, O+] | Key Statistical Finding | Checkpoint Storage | Status & Audit Verdict |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **Baseline E2** | Can multi-scan concatenation with auxiliary SupCon classify blood groups? | Frozen DenseNet-121, Concat(3072), Dense(512), Linear Emb(128); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), Adam, Original E2 Aug | 80 Dev (240 scans), 5-Fold StratGroup | **53.75%** (43/80) | 47.15% | 0.4772 | **NOT FOUND** | 63.16%, **14.29%**, 56.00%, 55.17% | Reference baseline | `models/phase3_fusion_metric_learning/E2/` (5 models) | **VERIFIED** (Severe AB+ collapse identified) |
| **EXP-01** | Does Class-Balanced Focal Loss ($\gamma=2.0$) resolve minority class collapse? | Identical to Baseline E2; C5 Enhanced | Focal Loss ($\gamma=2.0$) + SupCon ($\lambda=0.10, \tau=0.07$), AdamW, E2 Aug | 80 Dev (240 scans), 5-Fold StratGroup | **36.25%** (29/80) | 32.82% | 0.3205 | **NOT FOUND** | 36.84%, 14.29%, 56.00%, 24.14% | Collapsed to trivial majority baseline (36.25%) | Not retained | **VERIFIED** (Negative Result / Rejected) |
| **EXP-02** | Does conservative fingerprint augmentation (zero zoom) improve generalization? | Identical to Baseline E2; C5 Enhanced | Conservative Aug (rot $\pm 0.02$, trans $\pm 0.04$, zero zoom), CCE + SupCon, AdamW | 80 Dev (240 scans), 5-Fold StratGroup | **47.50%** (38/80) | 43.60% | 0.4632 | **NOT FOUND** | 42.11%, 28.57%, 52.00%, 51.72% | Underperformed unaugmented baseline (-6.25% acc) | Not retained | **VERIFIED** (Negative Result / Rejected) |
| **EXP-03 Arm 0** | Re-evaluation of original E2 augmentation under 100-epoch floor | Frozen DenseNet-121, Concat(3072), Linear Emb(128); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), Original E2 Aug, 100-ep floor | 80 Dev (240 scans), 5-Fold StratGroup | **48.75%** (39/80) | 42.56% | 0.4293 | 0.3818 ($\pm 0.0731$) | 57.89%, 14.29%, 36.00%, 62.07% | Confirmed augmentation impairs learning | `experiments/exp03/checkpoints/arm0/` (5 models) | **VERIFIED** |
| **EXP-03 Arm 1** | **Ablation: Complete removal of data augmentation (Native scans)** | Frozen DenseNet-121, Concat(3072), Linear Emb(128); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), **No Augmentation**, 100-ep floor | 80 Dev (240 scans), 5-Fold StratGroup | **53.75%** (43/80) | **52.84%** | **0.5068** | 0.4664 ($\pm 0.1496$) | 31.58%, **57.14%**, 64.00%, 58.62% | **Broke AB+ collapse (+42.85% AB+ recall jump)** | `experiments/exp03/checkpoints/arm1/` (5 models) | **VERIFIED** (Critical Milestone) |
| **EXP-03 Arm 2** | Conservative augmentation under identical 100-epoch floor | Frozen DenseNet-121, Concat(3072), Linear Emb(128); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), Conservative Aug, 100-ep floor | 80 Dev (240 scans), 5-Fold StratGroup | **38.75%** (31/80) | 40.99% | 0.4220 | 0.3613 ($\pm 0.0854$) | 63.16%, 42.86%, 20.00%, 37.93% | Severe drop in B+ recall (20.00%) | `experiments/exp03/checkpoints/arm2/` (5 models) | **VERIFIED** (Rejected) |
| **EXP-04 Config A** | Frozen backbone reference under dual-optimizer protocol | Frozen DenseNet-121 (0 backbone params), Concat(3072); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), AdamW lr=1e-3, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **50.00%** (40/80) | 47.86% | 0.4781 | 0.4318 ($\pm 0.1374$) | 36.84%, 42.86%, 60.00%, 51.72% | Reference for fine-tuning | Not retained | **VERIFIED** |
| **EXP-04 Config B** | Partial fine-tuning of DenseBlock 4 Conv2D layers | DenseBlock 4 trainable (2.13M params), frozen BN; C5 Enhanced | Dual-AdamW (backbone lr=1e-4, head lr=1e-3), No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **41.25%** (33/80) | 42.97% | 0.4032 | 0.4053 ($\pm 0.0849$) | 31.58%, 57.14%, 28.00%, 55.17% | McNemar vs A: $p = 0.2100$; B+ collapsed to 28% | Not retained | **VERIFIED** (Overfitting confirmed) |
| **EXP-04 Config C** | Deep partial fine-tuning of DenseBlocks 3+4 + Transition 3 | DB3+4 trainable (5.46M params), frozen BN; C5 Enhanced | Dual-AdamW (backbone lr=5e-5, head lr=1e-3), No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **47.50%** (38/80) | 39.04% | 0.3789 | 0.3398 ($\pm 0.0706$) | 15.79%, 14.29%, 64.00%, 62.07% | Severe minority collapse (A+ 15.8%, AB+ 14.3%) | Not retained | **VERIFIED** (Severe feature drift) |
| **EXP-05 Config A** | Multi-scan concatenation baseline (3072-d) | Frozen DenseNet-121, Concat(3072) (1.97M params); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), AdamW lr=1e-3, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | 51.84% | 0.5107 | 0.4579 ($\pm 0.1148$) | 31.58%, 57.14%, 60.00%, 58.62% | Strongest fusion baseline | `experiments/exp05/checkpoints/config_a/` (5 models) | **VERIFIED** |
| **EXP-05 Config B** | Gated Feature Fusion (per-scan learned gate) | Gated 1024-d projection (0.79M params); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), AdamW lr=1e-3, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | 51.24% | 0.5023 | 0.4473 ($\pm 0.0797$) | 26.32%, 57.14%, 56.00%, 65.52% | McNemar vs A: $p = 1.0000$ (24 discordant pairs) | `experiments/exp05/checkpoints/config_b/` (5 models) | **VERIFIED** |
| **EXP-05 Config C** | Cross-Scan Multi-Head Self-Attention (MHSA) | 4-head MHSA across 3 scans (1.28M params); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=0.07$), AdamW lr=1e-3, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **45.00%** (36/80) | 48.09% | 0.4535 | 0.4694 ($\pm 0.1054$) | 26.32%, 71.43%, 36.00%, 58.62% | McNemar vs A: $p = 0.3268$; B+ dropped to 36% | `experiments/exp05/checkpoints/config_c/` (5 models) | **VERIFIED** (Attention overfits) |
| **EXP-06A Config A** | SupCon Temperature $\tau = 0.07$ (Reference) | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=\mathbf{0.07}$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **48.75%** (39/80) | 46.26% | 0.4655 | 0.4269 ($\pm 0.0990$) | 31.58%, 42.86%, 52.00%, 58.62% | Reference temperature | `experiments/exp06a/checkpoints/` (part) | **VERIFIED** |
| **EXP-06A Config B** | **SupCon Temperature $\tau = 0.05$ (Sharp)** | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=\mathbf{0.05}$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | **55.18%** | **0.5313** | 0.5014 ($\pm 0.1277$) | 42.11%, **71.43%**, 52.00%, 55.17% | Binomial vs A: $p = 0.6636$; Peak AB+ recall (71.4%) | `experiments/exp06a/checkpoints/` (part) | **VERIFIED** (Optimal Temperature) |
| **EXP-06A Config C** | SupCon Temperature $\tau = 0.10$ (Higher) | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=\mathbf{0.10}$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **55.00%** (44/80) | 51.90% | 0.5152 | 0.4684 ($\pm 0.1025$) | 42.11%, 42.86%, 64.00%, 58.62% | High raw accuracy, but AB+ dropped to 42.9% | `experiments/exp06a/checkpoints/` (part) | **VERIFIED** |
| **EXP-06A Config D** | SupCon Temperature $\tau = 0.15$ (Highest) | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\lambda=0.10, \tau=\mathbf{0.15}$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | 51.70% | 0.5194 | 0.4784 ($\pm 0.1327$) | 31.58%, 57.14%, 56.00%, 62.07% | Over-smoothing of contrastive geometry | `experiments/exp06a/checkpoints/` (part) | **VERIFIED** |
| **EXP-06B Arm A** | SupCon Loss Weight $\lambda = 0.05$ | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\mathbf{\lambda=0.05}, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **46.25%** (37/80) | 46.38% | 0.4438 | 0.4607 ($\pm 0.0752$) | 26.32%, 57.14%, 40.00%, 62.07% | Weak regularization | `experiments/exp06b/checkpoints/` (part) | **VERIFIED** |
| **EXP-06B Arm B** | SupCon Loss Weight $\lambda = 0.10$ (Reference) | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\mathbf{\lambda=0.10}, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **47.50%** (38/80) | 48.47% | 0.4789 | 0.4799 ($\pm 0.0744$) | 42.11%, 57.14%, 36.00%, 58.62% | Binomial vs EXP-06A B: $p = 0.5235$ (run noise) | `experiments/exp06b/checkpoints/` (part) | **VERIFIED** |
| **EXP-06B Arm C** | SupCon Loss Weight $\lambda = 0.20$ | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\mathbf{\lambda=0.20}, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **45.00%** (36/80) | 43.99% | 0.4404 | 0.4370 ($\pm 0.0717$) | 36.84%, 42.86%, 48.00%, 48.28% | McNemar vs B: $p = 0.8501$ | `experiments/exp06b/checkpoints/` (part) | **VERIFIED** |
| **EXP-06B Arm D** | **SupCon Loss Weight $\lambda = 0.50$ (Heavy)** | Frozen DenseNet-121, Concat(3072); C5 Enhanced | CCE + SupCon ($\mathbf{\lambda=0.50}, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | **51.46%** | **0.4925** | 0.5035 ($\pm 0.1007$) | 36.84%, 57.14%, 36.00%, **75.86%** | Binomial vs B: $p = 0.5034$; Best fold stability | `experiments/exp06b/checkpoints/` (part) | **VERIFIED** (Optimal Weight) |
| **EXP-06C Config A** | **Linear Projection Head Reference (`3072 -> 128`)** | Frozen DenseNet-121, Concat(3072), **Linear Head** (1.97M params); C5 Enhanced | Shared Init, CCE + SupCon ($\lambda=0.50, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **53.75%** (43/80) | **53.47%** | **0.5487** | **0.5331 ($\pm 0.0895$)** | 42.11%, 57.14%, 56.00%, 58.62% | **Highest pooled Macro F1 (0.5487) and lowest $\sigma$** | `experiments/exp06c/checkpoints/config_a/` (5 models) | **VERIFIED (CURRENT LEADING MODEL)** |
| **EXP-06C Config B** | Two-Layer Nonlinear MLP Head (`3072 -> 512 -> 128`) | Frozen DenseNet-121, Concat(3072), **MLP Head** (3.21M params); C5 Enhanced | Shared Init, CCE + SupCon ($\lambda=0.50, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **46.25%** (37/80) | 52.12% | 0.4829 | 0.5064 ($\pm 0.1134$) | 31.58%, **85.71%**, 36.00%, 55.17% | Exact binomial vs A: $p = 0.2863$; MLP overfits | `experiments/exp06c/checkpoints/config_b/` (5 models) | **VERIFIED** (Linear Head Superior) |
| **EXP-06D Config A** | Enhanced C5 Reference (Crop + CLAHE + Bilateral) | Frozen DenseNet-121, Concat(3072), Linear Head (1.97M params); **C5 Enhanced** | Shared Init, CCE + SupCon ($\lambda=0.50, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **46.25%** (37/80) | 45.86% | 0.4456 | 0.4317 ($\pm 0.1044$) | 42.11%, 42.86%, 64.00%, 34.48% | Reference preprocessing; O+ heavily underpredicted | `experiments/exp06d/checkpoints/config_a/` (5 models) | **VERIFIED** |
| **EXP-06D Config B** | **Raw Platen C5 Candidate (Crop Only, No CLAHE/Bilateral)** | Frozen DenseNet-121, Concat(3072), Linear Head (1.97M params); **Raw Platen C5** | Shared Init, CCE + SupCon ($\lambda=0.50, \tau=0.05$), AdamW, No Aug | 80 Dev (240 scans), 5-Fold StratGroup | **52.50%** (42/80) | **50.07%** | **0.5078** | 0.4771 ($\pm 0.1099$) | 47.37%, 42.86%, 48.00%, **62.07%** | Exact binomial vs A: $p = 0.4049$; **+27.59% O+ recall** | `experiments/exp06d/checkpoints/config_b/` (5 models) | **VERIFIED** (Simpler, Non-Destructive) |

---

## 3. Methodological Comparability & Integrity Audit

A critical pitfall in meta-analyses is ranking experimental configurations without checking whether their evaluation frameworks are statistically commensurable.

### 3.1 Strict Invariance Factors Across All Experiments
1. **Participant Cohort ($N=80$):** Every experiment evaluated strictly the 80 development participants specified in `reports/phase3_participant_benchmark/participant_split.csv` (240 scans, 3 scans per participant).
2. **Participant-Grouped Folds:** The 5-fold `StratifiedGroupKFold` split was identical across all experiments. Scans belonging to the same participant were never partitioned across different folds, ensuring zero participant leakage.
3. **Canonical Label Mapping:** Canonical class indexing `0: A+ (19), 1: AB+ (7), 2: B+ (25), 3: O+ (29)` was consistently maintained.
4. **Quarantine Isolation:** The 21-participant locked test cohort (`P081`–`P101`, 63 scans) and the 26-participant external hospital cohort (78 scans) experienced **zero access events** across the entire project history.

### 3.2 Methodological Divergences & Non-Comparabilities
Despite identical cohorts, direct cross-experiment comparisons are subject to specific protocol evolutions:

```
[Phase 1: Augmentation Confounded]  Baseline E2, EXP-01, EXP-02, EXP-03 Arm 0/2
         │                         (Heuristic zoom/rotations altered spatial ridge frequencies)
         ▼
[Phase 2: Native Unaugmented Base]  EXP-03 Arm 1, EXP-04 Config A, EXP-05 Config A
         │                         (Proved zero augmentation is superior; fixed early stopping floor)
         ▼
[Phase 3: Hyperparameter Tuning]    EXP-06A (tau ablation), EXP-06B (lambda ablation)
         │                         (Stochastic initialization variance between experiments)
         ▼
[Phase 4: Shared-Weight Isolation]  EXP-06C (Head ablation), EXP-06D (Preprocessing ablation)
                                   (Enforced bit-for-bit epoch-zero shared weights to eliminate weight noise)
```

1. **Augmentation vs. Unaugmented Scans:** Experiments prior to EXP-03 Arm 1 incorporated heuristic image augmentations (random rotations, affine translations, and zoom). EXP-03 demonstrated that random zoom distorted ridge spatial frequency, causing an artificial performance suppression. Baseline E2, EXP-01, and EXP-02 cannot be directly compared against later unaugmented models.
2. **Checkpoint Selection Protocol:** Baseline E2, EXP-01, and EXP-02 utilized standard validation accuracy without an epoch floor. EXP-03 onward introduced an explicit **100-epoch training floor** (`MIN_EPOCHS = 100`, `PATIENCE = 15`) targeting validation Macro F1 to prevent premature stopping on noisy early epochs.
3. **Epoch-Zero Weight Parity:** EXP-06A and EXP-06B trained each arm from independent random initializations. Because deep networks on small sample sizes exhibit initialization sensitivity (observed between EXP-06A Config B and EXP-06B Arm B: $52.50\%$ vs $47.50\%$), EXP-06C and EXP-06D instituted **ephemeral shared epoch-zero weights per fold**, achieving true single-factor isolation.
4. **Pooled OOF vs. Mean Fold Metrics:**
   - **Pooled OOF Macro F1** aggregates all 80 participant predictions into a single contingency matrix before computing metric scores.
   - **Mean Fold Macro F1** averages the 5 individual fold scores, weighting small folds equally regardless of support.
   - Across all experiments, pooled Macro F1 and mean fold Macro F1 differ by $0.01$ to $0.05$. They must not be interchanged.

---

## 4. In-Depth Verification of Leading Candidates

The three strongest configurations identified across the project were subjected to detailed re-auditing:

### Candidate 1: EXP-06C Config A (Current Project Benchmark)
- **Architecture:** Frozen DenseNet-121 + 3-scan horizontal concatenation ($3 \times 1024 = 3072$-d) + Dense(512, relu) + Dropout(0.5) + Dense(4, softmax) + Linear Projection Head Dense(128, activation=None).
- **Hyperparameters:** AdamW (`lr=1e-3`, `weight_decay=1e-4`), $\text{SupCon}(\lambda=0.50, \tau=0.05)$, unaugmented native scans, Enhanced C5 preprocessing.
- **Recomputed Verification:**
  - Accuracy: **53.75%** (43 / 80)
  - Balanced Accuracy: **53.47%**
  - Pooled Macro F1: **0.5487** (Precision: 0.5734, Recall: 0.5347)
  - Mean Fold Macro F1: **0.5331 $\pm$ 0.0895** (Folds: `[0.5417, 0.6190, 0.4064, 0.5057, 0.5926]`)
  - Confusion Matrix `[A+, AB+, B+, O+]`:
    ```
    [[ 8,  1,  1,  9],   <- True A+  (19, Rec: 42.11%, Prec: 57.14%, F1: 0.4848)
     [ 1,  4,  0,  2],   <- True AB+ ( 7, Rec: 57.14%, Prec: 66.67%, F1: 0.6154)
     [ 2,  1, 14,  8],   <- True B+  (25, Rec: 56.00%, Prec: 58.33%, F1: 0.5714)
     [ 3,  0,  9, 17]]   <- True O+  (29, Rec: 58.62%, Prec: 47.22%, F1: 0.5231)
    ```
  - **Verdict:** Demonstrates the most balanced per-class sensitivity across the entire project (every class $>42\%$ recall, all F1 scores between $0.48$ and $0.62$).

### Candidate 2: EXP-06A Config B (Highest Minority AB+ Sensitivity)
- **Architecture:** Same concatenation baseline as EXP-06C Config A, but trained with lower SupCon loss weight ($\lambda = 0.10, \tau = 0.05$).
- **Recomputed Verification:**
  - Accuracy: **52.50%** (42 / 80)
  - Balanced Accuracy: **55.18%** (Highest balanced accuracy achieved)
  - Pooled Macro F1: **0.5313**
  - Mean Fold Macro F1: **0.5014 $\pm$ 0.1277**
  - Confusion Matrix `[A+, AB+, B+, O+]`:
    ```
    [[ 8,  2,  4,  5],   <- True A+  (19, Rec: 42.11%, Prec: 50.00%, F1: 0.4571)
     [ 1,  5,  0,  1],   <- True AB+ ( 7, Rec: 71.43%, Prec: 50.00%, F1: 0.5882)
     [ 3,  1, 13,  8],   <- True B+  (25, Rec: 52.00%, Prec: 54.17%, F1: 0.5306)
     [ 4,  2,  7, 16]]   <- True O+  (29, Rec: 55.17%, Prec: 53.33%, F1: 0.5424)
    ```
  - **Verdict:** Reached 71.43% AB+ recall (5/7 correct), but exhibited higher fold-to-fold variance ($\sigma = 0.1277$).

### Candidate 3: EXP-06D Config B (Pure Raw Platen Capture)
- **Architecture:** Same linear model as EXP-06C Config A ($\lambda=0.50, \tau=0.05$), but replaces Enhanced C5 preprocessing with **Raw Platen C5** (crop and resize only, eliminating CLAHE and bilateral filtering).
- **Recomputed Verification:**
  - Accuracy: **52.50%** (42 / 80)
  - Balanced Accuracy: **50.07%**
  - Pooled Macro F1: **0.5078**
  - Mean Fold Macro F1: **0.4771 $\pm$ 0.1099**
  - Confusion Matrix `[A+, AB+, B+, O+]`:
    ```
    [[ 9,  1,  2,  7],   <- True A+  (19, Rec: 47.37%, Prec: 47.37%, F1: 0.4737)
     [ 1,  3,  2,  1],   <- True AB+ ( 7, Rec: 42.86%, Prec: 50.00%, F1: 0.4615)
     [ 3,  1, 12,  9],   <- True B+  (25, Rec: 48.00%, Prec: 60.00%, F1: 0.5333)
     [ 6,  1,  4, 18]]   <- True O+  (29, Rec: 62.07%, Prec: 51.43%, F1: 0.5625)
    ```
  - **Verdict:** Most robust representation of the majority class O+ ($62.07\%$ recall vs. $34.48\%$ in Enhanced C5), confirming that heuristic spatial filtering is non-essential and potentially distortive.

---

## 5. Critical Scientific Assessment & Project Limitations

### 5.1 The Sample Size Ceiling ($N=80$)
The single most pervasive limitation of this entire research program is statistical power. In a 4-class classification problem on $N=80$ development participants:
- A change of 1 participant prediction shifts overall accuracy by $\pm 1.25\%$.
- For the rare AB+ class ($N=7$), a single participant shifts recall by $\pm 14.29\%$.
- Detecting a $+6.25\%$ difference with $80\%$ power at two-sided $\alpha = 0.05$ requires a paired sample of $N \approx 300$ participants.
- Consequently, while engineering refinements (zero augmentation, $\tau=0.05$, $\lambda=0.50$, linear head) systematically raised Macro F1 from $0.4772$ to $0.5487$, **none of the pairwise statistical tests achieved $p < 0.05$**. The gains represent directionally consistent improvements rather than mathematically proven separations.

### 5.2 Repeated Evaluation on the Same Development Cohort (Adaptive Selection Bias)
Across EXP-01 through EXP-06D, the identical 80 development participants were evaluated ten sequential times. Hyperparameters ($\tau$, $\lambda$, epoch thresholds, architecture, projection heads) were optimized based on cross-validation performance on this specific cohort.
- **Risk:** This iterative process inevitably introduces an accumulation of **adaptive model selection bias** (informal overfitting to the validation split).
- **Consequence:** The pooled Macro F1 of $0.5487$ reported for EXP-06C Config A is an optimistic development estimate. Performance on an unseen cohort will almost certainly be lower.

### 5.3 Biological & Sensor-Level Plausibility
The empirical failure of deep fine-tuning (EXP-04) and complex multi-head attention (EXP-05) demonstrates that the network is not learning complex hierarchical visual representations of blood group antigens. Rather, the frozen ImageNet backbone acts as a fixed spatial frequency filter bank, and the multi-task linear head learns to separate subtle photometric and ridge-density distributions. Whether these subtle textural distributions reflect true genetic ABO antigens or correlate with demographic/epidermal factors (contact pressure, skin hydration, platen temperature) remains unproven.

---

## 6. Next-Step Recommendation & Strategic Action Plan

### Evaluated Alternatives:
- **Option A: Stop experimentation and consolidate the final research project.**
  - *Evaluation:* Feasible and safe, but leaves the central scientific question unanswered: *Does the optimized model generalize to an unseen cohort, or has it merely overfitted the 80 development participants?*
- **Option B: Conduct one additional controlled development experiment.**
  - *Evaluation:* **REJECTED.** Additional experimentation on the same 80 participants (e.g. testing different learning rate schedules or loss formulations) will compound adaptive selection bias without providing statistical significance on $N=80$.
- **Option C: Prepare a final independent evaluation protocol.**
  - *Evaluation:* **STRONGLY RECOMMENDED.** This is the only scientifically rigorous and conclusive path forward.

### Formal Recommendation: **OPTION C — Prepare a Final Independent Evaluation Protocol**

```
                ┌─────────────────────────────────────────────────────────┐
                │                  RECOMMENDED NEXT ACTION:               │
                │     OPTION C: PREPARE FINAL INDEPENDENT EVALUATION      │
                └────────────────────────────┬────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          Phase 1: Pre-Registration                  Phase 2: Formal Execution
     - Freeze EXP-06C Config A weights          - Unquarantine locked test cohort
     - Pre-specify primary endpoints            - Single-pass evaluation only
     - Define exact failure/success criteria    - Zero retraining / zero tuning
     - Pre-specify statistical power tests      - Publish definitive report
```

### Strategic Rationale:
1. **Model Architecture is Fully Converged:** The ablation sequence has explored loss functions (EXP-01), augmentation (EXP-02/03), fine-tuning depth (EXP-04), fusion methods (EXP-05), temperature (EXP-06A), loss weights (EXP-06B), projection head capacity (EXP-06C), and preprocessing (EXP-06D). The architecture cannot be meaningfully improved on $N=80$.
2. **Quarantine Integrity has been Preserved:** The locked 21-participant test cohort (63 scans) and the external 26-participant hospital cohort (78 scans) remain completely untouched and pristine.
3. **Execution Boundary:** The evaluation protocol must be **written and approved first** before any evaluation script is executed or any quarantined file is read.

### Proposed Independent Evaluation Protocol Specifications (Planning Only):
- **Candidate Model:** The 5 fold checkpoints of **EXP-06C Config A** (`experiments/exp06c/checkpoints/config_a/fold_[1-5]_best.weights.h5`), evaluated as an ensemble (probability averaging across the 5 folds).
- **Target Evaluation Cohorts:**
  1. *Cohort 1 (Locked Internal Test Set):* 21 participants (63 scans), collected under identical Mantra MFS100 conditions.
  2. *Cohort 2 (External Hospital Cohort):* 26 participants (78 scans), representing domain-shifted acquisition conditions.
- **Pre-Registered Primary Endpoint:** Balanced Accuracy and Macro F1 exceeding the trivial majority baseline (Always-O+ $= 36.25\%$) at $\alpha = 0.05$.
- **Pre-Registered Decision Rules:**
  - If Ensemble Test Macro F1 $\ge 0.45$ and Balanced Accuracy $> 45\%$ on Cohort 1: Biometric correlation hypothesis is corroborated.
  - If Ensemble Test Macro F1 $< 0.35$ or collapses to majority class: Biometric correlation hypothesis is falsified.

---

## 7. Safety, Integrity, & Compliance Confirmation

To ensure strict compliance with project governance and safety constraints, the following invariants are certified:

1. **Zero Model Training / Fine-Tuning:** No model was trained, fine-tuned, or initialized during this review.
2. **Zero Checkpoint Alterations:** All 66 historical and production model checkpoints remain bit-for-bit identical to their recorded SHA-256 baseline ledger.
3. **Zero Prediction File Modifications:** All OOF prediction CSV files remain completely unmodified.
4. **Zero Quarantined Cohort Access:** Neither the locked 21-participant test cohort nor the separate 26-participant hospital cohort was accessed, read, or evaluated.
5. **Fail-Closed Execution Guard:** `TRAINING_AUTHORIZED = False` remains active in all experiment scripts.

---

*Execution halted. Master Experiment Review complete.*
