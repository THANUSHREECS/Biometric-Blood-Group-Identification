# EXP-06B: Controlled Experiment Design Specification

**Document Version:** 1.0.0  
**Design Date:** 2026-10-15  
**Experiment Identifier:** `EXP-06B` (Supervised Contrastive Loss Weight Ablation)  
**Lead Researcher:** Senior ML Researcher & Computer Vision Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> This experiment investigates deep learning hyperparameter sensitivity on biometric optical fingerprint representations. All models, weights, metrics, and conclusions documented herein are exploratory academic research prototypes and **must not** be used for clinical diagnostics, transfusion medicine, triage, patient care, or medical blood group determination. Laboratory serological testing is mandatory for all clinical use.

---

## 1. Scientific Objective & Hypothesis

### 1.1 Objective
The primary objective of **EXP-06B** is to systematically isolate and quantify the effect of the **supervised contrastive loss weight** ($\lambda_{\text{supcon}}$) on participant-level four-class ABO blood group classification performance (Macro F1, Balanced Accuracy, and per-class recall).

### 1.2 Context & Provisional Reference
In EXP-06A, temperature $\tau = 0.05$ demonstrated superior class separation and minority class sensitivity (AB+ recall: 71.43%, Macro F1: 0.5313). Throughout EXP-06A, the loss weight was held fixed at the historical default: $\lambda_{\text{supcon}} = 0.10$.

In EXP-06B, we fix the contrastive temperature at $\tau = 0.05$ across all configurations and systematically ablate $\lambda_{\text{supcon}} \in \{0.05, 0.10, 0.20, 0.50\}$.

### 1.3 Scientific Hypotheses
1. **Under-Regularization Hypothesis ($\lambda = 0.05$):** Reducing contrastive weight allows cross-entropy to dominate training, which may improve majority class accuracy (O+, B+) but risk collapsing rare class representations (AB+, A+) into majority clusters.
2. **Balanced Metric Hypothesis ($\lambda = 0.10$):** Serves as the calibrated reference baseline from EXP-06A Config B.
3. **Enhanced Metric Regularization Hypothesis ($\lambda = 0.20$):** Doubling contrastive pressure will penalize inter-class overlap more aggressively, potentially refining decision boundaries between morphologically ambiguous classes without destabilizing classification gradients.
4. **Over-Constrained Clustering Hypothesis ($\lambda = 0.50$):** Over-weighting metric loss may prioritize clustering compactness at the expense of cross-entropy discrimination, degrading posterior probability calibration and raw accuracy.

---

## 2. Mathematical Loss Formulation

The multi-task objective minimized by the model during training is:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda_{\text{supcon}} \cdot \mathcal{L}_{\text{SupCon}}(z, y; \tau = 0.05)$$

Where:
- $\mathcal{L}_{\text{CE}}$ is the sample-weighted categorical cross-entropy loss operating on the 4-class softmax predictions:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \sum_{c=1}^4 y_{i,c} \log(\hat{y}_{i,c})$$
  with class weights $w_c = \frac{N_{\text{total}}}{4 \cdot N_c}$ recomputed strictly per training fold.
- $\mathcal{L}_{\text{SupCon}}$ is the Supervised Contrastive loss operating on the normalized 128-dimensional participant-level fused embeddings $z_i \in \mathbb{R}^{128}$ ($z_i = \text{emb}(h_{\text{fused}})$) with fixed temperature $\tau = 0.05$:
  $$\mathcal{L}_{\text{SupCon}} = - \sum_{i=1}^B \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left(\frac{z_i \cdot z_p}{\tau \cdot \|z_i\| \|z_p\|}\right)}{\sum_{a \in A(i)} \exp\left(\frac{z_i \cdot z_a}{\tau \cdot \|z_i\| \|z_a\|}\right)}$$
  Where $A(i) = \{1, \dots, B\} \setminus \{i\}$ and $P(i) = \{p \in A(i) : y_p = y_i\}$.
- $\lambda_{\text{supcon}}$ is the experimental independent variable.

---

## 3. Experimental Arms & Control Variables

### 3.1 Arm Definitions

| Arm Identifier | Contrastive Weight ($\lambda_{\text{supcon}}$) | Contrastive Temp ($\tau$) | Theoretical Role / Mechanism |
|:---|:---:|:---:|:---|
| **Arm A** | **0.05** | 0.05 | Reduced metric constraint; cross-entropy dominant. Tests if lighter regularization prevents over-clustering. |
| **Arm B** [Ref] | **0.10** | 0.05 | Reference candidate from EXP-06A Config B ($\tau=0.05, \lambda=0.10$). Direct continuity baseline. |
| **Arm C** | **0.20** | 0.05 | Moderate metric constraint. Increases penalty for inter-class embedding overlap by 2x. |
| **Arm D** | **0.50** | 0.05 | Heavy metric constraint. Tests high-density cluster repulsion on frozen DenseNet features. |

### 3.2 Strictly Controlled Variables (Identical Across All Arms)

| Experimental Dimension | Specification across Arms A, B, C, D | Verification Mechanism |
|:---|:---|:---:|
| **Backbone Architecture** | DenseNet-121 (Frozen, pretrained on ImageNet) | 7,037,504 non-trainable parameters; `trainable = False` |
| **Feature Fusion Neck** | Concatenation of 3 scans: $3 \times 1024 = 3072$-d | `tf.keras.layers.Concatenate(axis=-1)` |
| **Classification Head** | Dense(512, relu) + Dropout(0.5) + Dense(4, softmax) | Verified variable shapes & parameter counts |
| **Embedding Head** | Dense(128, linear, no activation) from `fused` | Produces single 128-d vector per participant |
| **Total Trainable Parameters** | Exactly **1,968,772** across exactly **6 variables** | Programmatic variable inspection at preflight |
| **Contrastive Temperature** | Fixed strictly at $\tau = 0.05$ | Hardcoded constant across all arms |
| **Cohort Definition** | 80 Development participants ONLY (240 scans) | Quarantined from locked test (21) & hospital (26) cohorts |
| **Validation Scheme** | 5-Fold StratifiedGroupKFold (`seed=42`) | All 3 scans of each participant strictly collocated |
| **Preprocessing** | `transform_c5` (5-stage optical standardization) | Unaugmented optical biometric preprocessing |
| **Data Augmentation** | Strictly None (unaugmented) | Exact parity with EXP-05 Config A & EXP-06A |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | Consistent optimization dynamics |
| **Batch Size** | 16 participants per mini-batch (48 scans) | Consistent mini-batch gradient updates |
| **Epoch Limits** | Minimum 100 epochs, Maximum 150 epochs | Guaranteed convergence window |
| **Early Stopping** | Patience 15 epochs on Validation Macro F1 | Active only after epoch 100 |
| **Initial Head Weights** | Identical epoch-zero weights per fold across arms | Ephemeral template serialized and loaded at Step 0 |

---

## 4. Evaluation Protocol & Decision Rules

1. **Primary Metric:** Pooled out-of-fold (OOF) Macro F1 across the complete 80 development participants.
2. **Secondary Metrics:** Balanced accuracy, minority class recall (AB+, A+), raw accuracy, and fold-level Macro F1 standard deviation.
3. **Statistical Testing:**
   - Paired participant-level McNemar tests against Arm B (Reference $\lambda=0.10$).
   - Multiple comparison correction using the Holm-Bonferroni step-down procedure ($\alpha = 0.05$).
4. **Model Selection Criteria:**
   - The leading candidate must demonstrate a superior Macro F1 over Arm B ($> 0.5313$) without catastrophic collapse of any individual class recall.
   - Claims of superiority will be reported strictly as empirical observations unless accompanied by formal statistical significance ($p < \alpha_{\text{adj}}$).
