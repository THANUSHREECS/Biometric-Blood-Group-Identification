# EXP-06D: Experiment Protocol & Statistical Design

**Document Version:** 1.0.0  
**Preparation Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06D`  
**Investigation Theme:** Preprocessing & Multi-Scan Representation Investigation  
**Reference Benchmark:** EXP-06C Config A (Linear Projection Head, $\lambda = 0.50, \tau = 0.05$)  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, configurations, and evaluation metrics described in this protocol are exploratory academic research designs for investigating deep learning representations on optical biometric fingerprint data. They **must not** be used for clinical blood typing, transfusion medicine, patient diagnostics, triage, or healthcare applications. Standard serological agglutination testing is mandatory for medical use.

---

## 1. Factor Selection Rationale: Preprocessing vs. Multi-Scan Fusion

Before finalizing the experimental design, the repository implementation and historical trajectory were audited to determine whether **Input Preprocessing (Factor A)** or **Three-Scan Fusion (Factor B)** is the more defensible, isolated, and scientifically impactful factor for EXP-06D.

### Comparative Evaluation

| Evaluation Criteria | Factor A: Input Preprocessing | Factor B: Three-Scan Fusion | Evaluation Verdict |
|:---|:---|:---|:---:|
| **Prior Systematic Investigation** | **NEVER ablated** under the Phase 3 participant-level benchmark. `transform_c5` has been treated as a fixed legacy constant. | **Extensively ablated in EXP-05** across Concatenation, Gated Fusion, and Self-Attention. Attention underperformed (F1: 0.4535); Gated was identical ($p=1.0000$). | **Factor A Preferred** (unstudied domain) |
| **Parameter Capacity Confound** | **Zero Confound (0%).** Both configurations maintain exactly **1,968,772 trainable parameters** across 6 weight/bias tensors. | **Severe Confound (-53.3%).** Changing fusion from concatenation (3072-d) to mean/gated pooling (1024-d) reduces head parameters from 1.97M to 0.92M. | **Factor A Strongly Preferred** (pure isolation) |
| **Mechanistic Uncertainty** | `transform_c5` applies aggressive CLAHE ($2.5$), percentile stretching, and bilateral filtering ($d=3$). Bilateral filtering smoothes high-frequency ridge minutiae, potentially distorting ImageNet-pretrained convolutional filters. | Fusion across 3 scans is currently order-dependent (concatenation), but EXP-05 proved that learning dynamic weights on $N=80$ leads to overfitting or neutral impact. | **Factor A Preferred** (biological/optical fidelity) |
| **Pipeline Feasibility** | Single modular function swap in data loader (`transform_c5` vs. `transform_c5_raw`). Model architecture and training loop remain identical. | Requires rewriting model functional graph, neck construction, and epoch-zero weight initialization routines. | **Factor A Preferred** (cleanest isolated diff) |

### Formal Selection
**Factor A (Input Preprocessing)** is selected as the sole principal factor for EXP-06D. Specifically, the ablation isolates the effect of **Ridge Contrast Enhancement & Bilateral Filtering** versus **Raw Cropped Platen Impressions**.

---

## 2. Research Question & Hypothesis

### 2.1 Research Question
Does the legacy ridge enhancement pipeline (`enhance_mfs100_ridge_contrast` combining CLAHE, dynamic percentile contrast stretching, and bilateral filtering) improve participant-level blood group classification compared with raw, unenhanced platen-cropped optical fingerprint impressions, when evaluated under the verified EXP-06C Config A linear projection SupCon architecture?

### 2.2 Hypothesis
- **Null Hypothesis ($H_0$):** There is no significant difference in pooled out-of-fold Macro F1 between models trained on CLAHE/bilateral-enhanced fingerprint scans versus raw platen-cropped fingerprint scans ($\Delta \text{Macro F1} = 0$, $p \ge 0.05$).
- **Alternative Hypothesis ($H_1$):** Raw platen-cropped scans preserve authentic optical ridge-frequency gradients and fine minutiae boundaries that are otherwise attenuated by bilateral filtering, resulting in higher pooled out-of-fold Macro F1 without introducing artificial platen background contrast artifacts.

---

## 3. Experimental Configurations

| Experimental Dimension | Config A (Reference: Enhanced C5) | Config B (Candidate: Unenhanced Raw C5) | Controlled Status |
|:---|:---|:---|:---:|
| **Experiment ID** | `config_a` | `config_b` | A/B Pair |
| **Alpha Compositing** | White background (`255, 255, 255`) | White background (`255, 255, 255`) | **IDENTICAL** |
| **Platen Cropping** | Active platen crop with 6% padding | Active platen crop with 6% padding | **IDENTICAL** |
| **Ridge Enhancement** | **CLAHE (clip=2.5) + Stretch + Bilateral Filter** | **NONE (Raw optical ridge intensities)** | **PRINCIPAL FACTOR** |
| **Input Resolution** | $112 \times 112 \times 3$ (Lanczos resampling) | $112 \times 112 \times 3$ (Lanczos resampling) | **IDENTICAL** |
| **Pixel Value Range** | Raw $[0.0, 255.0]$ float32 (No `preprocess_input`) | Raw $[0.0, 255.0]$ float32 (No `preprocess_input`) | **IDENTICAL** |
| **Feature Backbone** | Pretrained DenseNet-121 (strictly frozen) | Pretrained DenseNet-121 (strictly frozen) | **IDENTICAL** |
| **Non-Trainable Params** | 7,037,504 (backbone frozen) | 7,037,504 (backbone frozen) | **IDENTICAL** |
| **Multi-Scan Fusion** | 3-scan Concatenation ($3 \times 1024 = 3072$-d) | 3-scan Concatenation ($3 \times 1024 = 3072$-d) | **IDENTICAL** |
| **Classification Head** | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | **IDENTICAL** |
| **Projection Head** | Linear: `Dense(128, activation=None)` | Linear: `Dense(128, activation=None)` | **IDENTICAL** |
| **Trainable Parameters** | **1,968,772** across 6 tensors | **1,968,772** across 6 tensors | **EXACT MATCH (0% Confound)** |
| **Initial Shared Weights** | Identical Step-0 classifier head weights per fold | Identical Step-0 classifier head weights per fold | **IDENTICAL** |
| **Loss Function** | $1.0 \times \text{CCE} + 0.50 \times \text{SupCon}(\tau=0.05)$ | $1.0 \times \text{CCE} + 0.50 \times \text{SupCon}(\tau=0.05)$ | **IDENTICAL** |
| **Optimizer & LR** | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | **IDENTICAL** |
| **Training Policy** | Min 100 epochs, Max 150 epochs, Patience 15 | Min 100 epochs, Max 150 epochs, Patience 15 | **IDENTICAL** |
| **Batch Size & Seed** | Batch size = 16, Global Random Seed = 42 | Batch size = 16, Global Random Seed = 42 | **IDENTICAL** |

---

## 4. Controlled Variables & Invariant Safeguards

To ensure that any observed difference in performance is caused strictly by the ridge enhancement operation, the following variables are strictly locked:
1. **Identical Datasets & Manifest:** Both configurations will train and validate on the identical 80 development participants from `reports/phase3_participant_benchmark/participant_split.csv`.
2. **Identical 5 Folds:** StratifiedGroupKFold (`n_splits=5, shuffle=True, random_state=42`) preserves exact participant groupings.
3. **Identical Initial Classifier Weights:** Epoch-zero shared weights for `dense_512` and `clf` layers will be transferred across configurations within each fold.
4. **Strict Backbone Freezing:** DenseNet-121 will have `trainable = False` throughout all epochs.
5. **Class Weighting:** Identical inverse-frequency weights applied to the classification loss: $w_c = \frac{N_{\text{train}}}{4 \times N_c}$.

---

## 5. Participant Eligibility & Cohort Quarantines

### 5.1 Development Cohort Eligibility Rules
- Must be designated as `TRAIN` or `VALIDATION` in `reports/phase3_participant_benchmark/participant_split.csv`.
- Exactly 80 participants (240 scans).
- Exactly 3 scans per participant must be present and readable.
- Ground truth must belong to the approved 4 ABO-positive classes: `A+`, `AB+`, `B+`, `O+`.

### 5.2 Quarantine Enforcements
- **Locked Test Cohort (21 Participants, 63 Scans):** Rows where `split == 'TEST'` are strictly quarantined. Under no circumstances will any scan, label, or participant ID from this cohort be loaded, processed, or evaluated.
- **Hospital Cohort (26 Participants, 78 Scans):** Stored under `data/captured/`. Strictly quarantined. Zero access permitted.

---

## 6. Evaluation Metrics & Decision Standard

### 6.1 Primary Decision Metric
**Pooled Out-of-Fold (OOF) Macro F1** across all 80 development participants:
$$\text{Macro F1}_{\text{pooled}} = \frac{1}{4} \sum_{c \in \{A+, AB+, B+, O+\}} F1_c$$
Where each $F1_c$ is calculated on the concatenated out-of-fold predictions of the 80 participants.

### 6.2 Secondary Diagnostic Metrics
1. Overall Classification Accuracy ($N_{\text{correct}} / 80$).
2. Balanced Accuracy (unweighted mean of recalls across the 4 classes).
3. Macro Precision and Macro Recall.
4. Per-Class Precision, Recall, F1-Score, and Support for each blood group.
5. $4 \times 4$ Out-of-Fold Confusion Matrix in Canonical Order `[A+, AB+, B+, O+]`.
6. Mean and Standard Deviation of Fold-Level Macro F1 ($\mu_{\text{fold}} \pm \sigma_{\text{fold}}$).
7. Total execution runtime (minutes) per configuration and per fold.

---

## 7. Statistical Testing & Paired Bootstrap Protocol

### 7.1 Paired Classification Correctness Test
A $2 \times 2$ contingency table of participant-level binary correctness ($Y_A, Y_B \in \{0, 1\}$) will be constructed:
- Let discordant pairs be $b = n_{10}$ (Config A correct only) and $c = n_{01}$ (Config B correct only).
- If total discordant pairs $D = b + c < 25$:
  Execute **Two-Tailed Exact Binomial Test** ($H_0: p = 0.5$):
  $$p = 2 \cdot \sum_{k=0}^{\min(b, c)} \binom{D}{k} (0.5)^D$$
- If $D \ge 25$:
  Execute **McNemar's Test with Edwards Continuity Correction**:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$

### 7.2 Participant-Level Paired Non-Parametric Bootstrap
1. **Resampling Unit:** Strictly the **individual participant** ($N = 80$, resampling all 3 scans collocated).
2. **Replicates:** $B = 2,000$ bootstrap iterations.
3. **Random Seed:** Locked to `seed = 42`.
4. **Omitted Class Handling:** Resamples that omit rare classes (e.g., AB+, $N=7$) must be evaluated deterministically using `precision_recall_fscore_support` with `labels=['A+', 'AB+', 'B+', 'O+']` and `zero_division=0`.
5. **Confidence Interval:** 95% Percentile Bootstrap Confidence Interval:
   $$\text{CI}_{95\%} = \left[ \Delta_{(0.025 \cdot B)}, \; \Delta_{(0.975 \cdot B)} \right]$$
6. **Significance Standard:** If $0.0 \in \text{CI}_{95\%}$ or $p \ge 0.05$, the difference cannot be concluded to be statistically significant.

---

## 8. Small Sample Limitations & Rejection Conditions

### 8.1 Small Sample Limitations
- The development cohort contains only $N = 80$ participants.
- The **AB+ class contains only 7 participants ($8.75\%$)**. A shift of just 1 participant alters AB+ recall by $14.29\%$.
- With $N = 80$, the statistical power to detect small effect sizes ($|\Delta \text{Macro F1}| < 0.08$) is limited, as reflected in the width of the bootstrap confidence intervals.

### 8.2 Experiment Rejection Conditions
The candidate configuration (Config B) must be **rejected** (and Config A retained) if:
1. Pooled OOF Macro F1 of Config B is lower than Config A ($\text{Macro F1}_B < \text{Macro F1}_A$).
2. The 95% bootstrap confidence interval includes zero, indicating no verifiable statistical separation.
3. Config B causes catastrophic collapse on any individual blood group (e.g., Recall $< 20\%$ on A+, B+, or O+).
4. Any data leakage or checkpoint corruption occurs during execution.

---

## 9. Estimated Computational Cost

Based on the verified execution logs of EXP-06C:
- Config A (100–120 epochs per fold) required **121.30 minutes (~2.02 hours)** on the CPU workstation.
- Config B (with identical parameter count and 112x112 input resolution) is projected to take **115–125 minutes (~2.0 hours)**.
- Total projected runtime for EXP-06D: **~4.0 to 4.5 hours**.

---

## 10. Audit Sign-off

- **Single Factor Verified:** Ridge enhancement vs. raw platen crop.
- **Confounders Eliminated:** Model architecture, parameter count, input resolution, optimizer, loss, and seeds are 100% invariant.
- **Safety Boundaries Locked:** Zero test or hospital access permitted.

*Protocol compiled autonomously following rigorous scientific standards.*
