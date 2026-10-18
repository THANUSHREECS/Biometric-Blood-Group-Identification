# EXP-06A Experimental Design: Supervised Contrastive Temperature Ablation

**Document Version:** 1.0.0  
**Date:** 2026-10-15  
**Experiment Identifier:** `EXP-06A`  
**Focus:** Supervised Contrastive Hyperparameter Optimization ($\tau$)  
**Reference Architecture:** EXP-05 Configuration A (Concatenation Fusion)  

---

> [!CAUTION]
> **RESEARCH USE ONLY — NON-CLINICAL DISCLAIMER**  
> This document details an academic machine learning experiment on biometric feature representations. The models, techniques, and metrics described are exploratory research prototypes and **must not** be used for clinical blood typing, medical decision-making, or emergency transfusions. Standard laboratory serological agglutination testing is mandatory for medical blood grouping.

---

## 1. Scientific Motivation & Theoretical Background

In Phase 3, **EXP-03** established that unaugmented C5 preprocessing yields superior generalization compared with geometric augmentations, and **EXP-04** proved that unfreezing DenseNet-121 convolutional layers causes severe feature distortion and representational collapse. **EXP-05** systematically evaluated feature fusion architectures (Concatenation, Gated Fusion, Cross-Scan Self-Attention) with a frozen backbone, demonstrating that simple Concatenation remains the leading development candidate (Macro F1 = 0.5107, Accuracy = 52.50%).

Having fixed both the data pipeline and the neck architecture, **EXP-06A** targets the multi-task metric learning objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{clf}}, y; w_{\text{class}}) + \lambda_{\text{supcon}} \mathcal{L}_{\text{SupCon}}(\hat{z}_{\text{emb}}, y; \tau)$$

The supervised contrastive loss (Khosla et al., 2020) maps the fused 128-dimensional participant representations onto a unit hypersphere ($\|\hat{z}_{\text{emb}}\|_2 = 1$):

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left(\frac{\hat{z}_i \cdot \hat{z}_p}{\tau}\right)}{\sum_{a \in A(i)} \exp\left(\frac{\hat{z}_i \cdot \hat{z}_a}{\tau}\right)}$$

### Role of Temperature ($\tau$):
1. **Hard Negative Mining ($\tau \to 0$):** As temperature decreases, the similarity softmax becomes sharper. Gradients are dominated by the most difficult negative pairs (e.g., distinguishing A+ from AB+ or B+ from O+). However, setting $\tau$ too low can destabilize gradient updates and cause over-clustering that penalizes natural biometric variability across different scans of the same participant.
2. **Smooth Representation Learning ($\tau \to \infty$):** As temperature increases, similarity probabilities become softer and more uniform. This tolerates sensor noise and optical distortion across different scan impressions, but risks blurring the geometric boundaries between minority and majority classes.
3. **Reference Standard ($\tau = 0.07$):** While $\tau = 0.07$ is the standard empirical default in general computer vision literature, its optimality for low-sample biometric fingerprint classification ($N=80$) has never been empirically evaluated.

---

## 2. Experimental Configurations

EXP-06A evaluates four configurations varying **only** the contrastive temperature $\tau$:

| Configuration ID | Configuration Name | Contrastive Temperature ($\tau$) | Theoretical Dynamic | Contrastive Weight ($\lambda$) | Trainable Parameters |
|:---|:---|:---:|:---|:---:|:---:|
| **Config A (Ref)** | **Reference Temperature** | $\tau = \mathbf{0.07}$ | Audited default from EXP-03, EXP-04, and EXP-05. | $0.1$ | 1,968,772 |
| **Config B** | **Lower Temperature** | $\tau = \mathbf{0.05}$ | Aggressive hard-negative discrimination; sharper hyperspherical clusters. | $0.1$ | 1,968,772 |
| **Config C** | **Higher Temperature** | $\tau = \mathbf{0.10}$ | Smoother similarity distribution; increased tolerance to ridge distortion. | $0.1$ | 1,968,772 |
| **Config D** | **Highest Temperature** | $\tau = \mathbf{0.15}$ | Highly regularized metric space; relaxed clustering penalty. | $0.1$ | 1,968,772 |

---

## 3. Strict Control Variables (Zero Confounding Policy)

To ensure that any observed performance delta is strictly attributable to the temperature hyperparameter $\tau$, all other pipeline variables are held constant:

1. **Backbone Architecture & State:** Pretrained ImageNet DenseNet-121, strictly frozen at `trainable = False` across all configurations (7,037,504 non-trainable parameters; 0 trainable backbone parameters).
2. **Fusion Architecture:** Concatenation baseline (concatenating three 1024-d vectors into a 3072-d representation).
3. **Head Architecture:**
   - Classification branch: `Dense(512, relu)` $\to$ `Dropout(0.5)` $\to$ `Dense(4, softmax)`.
   - Embedding branch: `Dense(128, linear)`.
4. **Parameter Equality:** Exactly **1,968,772 trainable parameters** across exactly **6 trainable variables** in every configuration.
5. **Preprocessing:** Aspect-preserving resize to $112 \times 112$, center padding, CLAHE contrast enhancement, ImageNet normalization (`transform_c5`). Zero geometric or photometric augmentation.
6. **Cross-Validation Splits:** The exact same 5 participant-grouped folds (64 train / 16 val per fold; 0 participant leakage).
7. **Loss Weighting:** Classification loss weighted by inverse class frequency $w_c = \frac{N_{\text{train}}}{4 \cdot N_c}$; contrastive loss weight $\lambda_{\text{supcon}} = 0.1$ across all configs.
8. **Optimizer & Schedule:** AdamW optimizer with base learning rate $\eta = 10^{-3}$, weight decay $10^{-4}$, and gradient clipping.
9. **Stopping Policy:** Training floor `min_epochs = 100`, maximum epochs `epochs = 150`, early stopping patience `patience = 15` after the floor.
10. **Model Selection:** Validation Macro-F1 on held-out fold validation set.

---

## 4. Evaluation Framework & Decision Policy

### 4.1 Primary Selection Metric
- **Macro F1-Score (Out-of-Fold):** Because the 80-participant development cohort is naturally imbalanced (19 A+, 7 AB+, 25 B+, 29 O+), Macro F1 is the mandated primary metric.
- A configuration will **not** be selected based on overall accuracy alone.

### 4.2 Supporting Metrics
- **Balanced Accuracy:** Average recall across all 4 classes.
- **Minority Class Recalls:** Specifically A+ recall (reference: 31.58%) and AB+ recall (reference: 57.14%).
- **Fold-to-Fold Stability:** Standard deviation of validation Macro F1 across the 5 folds.

### 4.3 Statistical Significance Protocol
- Predictions will be generated for all 80 development participants in an aligned out-of-fold manner.
- Pairwise comparisons (Config A vs Config B, Config A vs Config C, Config A vs Config D) will be subjected to continuity-corrected McNemar tests on discordant pairs.
- No claim of superiority will be made without supporting statistical and effect-size evidence.
