# EXP-06C: Configuration Comparison & Architectural Specification

**Document Version:** 1.0.0  
**Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Lead Researcher:** Senior ML Research Scientist & Biometrics Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, configurations, and architectural descriptions herein are research prototypes designed to study machine learning representation sensitivity on optical biometric features. They **must not** be used for clinical blood grouping, medical diagnostics, patient transfusion matching, or triage.

---

## 1. Architectural Overview & Component Comparison

EXP-06C tests the hypothesis that a **two-layer nonlinear MLP projection head** provides better feature regularization for four-class blood group classification than a **single-layer linear projection head** when training under the optimal contrastive hyperparameter regime established in EXP-06B ($\tau = 0.05, \lambda_{\text{supcon}} = 0.50$).

```
[3 Scans: (3, 112, 112, 3)]
        │
        ▼
[Frozen DenseNet-121 + GAP] ──► [f1, f2, f3 (1024-d each)]
        │
        ▼
[Concatenate Neck: fused (3072-d)]
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[Classification Branch]                  [Projection Head Branch]
(Identical across Config A & B)          (Independent Variable)
Dense(512, ReLU)                         Config A: Dense(128, Linear)
Dropout(0.5)                             Config B: Dense(512, ReLU) -> Dense(128, Linear)
Dense(4, Softmax)                                │
        │                                        ▼
        ▼                                [L2 Normalization]
[Weighted Cross-Entropy Loss]                    │
        │                                        ▼
        └─────────────────┬──────────────────────┘ [SupCon Loss (tau=0.05)]
                          ▼
             Total Loss = L_CE + 0.50 * L_SupCon
```

---

## 2. Layer-by-Layer Parameter & Tensor Accounting

### 2.1 Common Pretrained & Frozen Layers

Both configurations utilize identical frozen feature extraction:
- **Input Tensor:** `(Batch_Size, 3, 112, 112, 3)`, float32 normalized to $[0, 1]$.
- **Backbone:** ImageNet-pretrained `DenseNet121` (`include_top=False`), strictly non-trainable (`base.trainable = False`).
- **Backbone Parameters:** Exactly **7,037,504 non-trainable parameters** across 364 weight tensors. Zero backbone updates.
- **Pooling & Fusion:** `GlobalAveragePooling2D` applied to each scan $\to$ `(B, 1024)`. Concatenation yields fused tensor $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{B \times 3072}$.

---

### 2.2 Shared Classification Branch

Both configurations share an exact, bit-compatible classification head architecture:

| Layer Name | Layer Type | Input Dim | Output Dim | Activation | Kernel Shape | Bias Shape | Trainable Parameters |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `dense_512` | Dense | 3072 | 512 | ReLU | `(3072, 512)` | `(512,)` | $1,572,864 + 512 = \mathbf{1,573,376}$ |
| `dropout_0.5` | Dropout | 512 | 512 | None ($p=0.5$) | None | None | $0$ |
| `clf` | Dense | 512 | 4 | Softmax | `(512, 4)` | `(4,)` | $2,048 + 4 = \mathbf{2,052}$ |
| **Total Classifier** | — | — | — | — | — | — | **1,575,428 parameters (4 tensors)** |

---

### 2.3 Projection Head Branch Comparison

The projection head is the **sole isolated independent variable** in EXP-06C:

#### Configuration A: Linear Reference Head
- Architecture: Single linear projection layer directly mapping the 3072-dimensional fused representation to 128 embedding dimensions without intermediate nonlinearity.
- Provenance: Verified leading candidate from EXP-06B Arm D.

| Layer Name | Layer Type | Input Dim | Output Dim | Activation | Kernel Shape | Bias Shape | Trainable Parameters |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `emb` | Dense | 3072 | 128 | Linear (None) | `(3072, 128)` | `(128,)` | $393,216 + 128 = \mathbf{393,344}$ |
| **Total Projection** | — | — | — | — | — | — | **393,344 parameters (2 tensors)** |

#### Configuration B: Two-Layer Nonlinear MLP Projection Head
- Architecture: Two-layer MLP comprising an intermediate expansion dense layer with ReLU activation, followed by a final linear projection layer mapping to the identical 128-dimensional embedding space.
- Rationale: Follows standard contrastive representation learning paradigms (SimCLR, Khosla et al.) to discard high-frequency sample-specific noise while preserving rich invariant structural clusters.

| Layer Name | Layer Type | Input Dim | Output Dim | Activation | Kernel Shape | Bias Shape | Trainable Parameters |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `proj_mlp_dense1`| Dense | 3072 | 512 | ReLU | `(3072, 512)` | `(512,)` | $1,572,864 + 512 = \mathbf{1,573,376}$ |
| `proj_mlp_dense2`| Dense | 512 | 128 | Linear (None) | `(512, 128)` | `(128,)` | $65,536 + 128 = \mathbf{65,664}$ |
| **Total Projection** | — | — | — | — | — | — | **1,639,040 parameters (4 tensors)** |

---

### 2.4 Complete Parameter Summary

| Metric / Dimension | Config A (Linear Reference) | Config B (Nonlinear MLP) | Delta ($\Delta$) | Status |
|:---|:---:|:---:|:---:|:---:|
| **Trainable Variables** | **6 tensors** | **8 tensors** | $+2$ tensors | Controlled |
| **Trainable Parameters** | **1,968,772** | **3,214,468** | $+1,245,696$ ($+63.3\%$) | Controlled |
| **Frozen Backbone Parameters** | **7,037,504** | **7,037,504** | $0$ | Identical |
| **Total Model Parameters** | **9,006,276** | **10,251,972** | $+1,245,696$ ($+13.8\%$) | Controlled |
| **Final Embedding Dimension** | **128** | **128** | $0$ | Identical |
| **Classifier Output Dimension** | **4** | **4** | $0$ | Identical |

---

## 3. Mathematical Documentation of Initialization Parity & Differences

In strict compliance with experimental standards:
> **"Document any unavoidable initialization differences. Do not claim that architectures have identical initialization when their parameter shapes differ."**

### 3.1 Shared Initialization (Exact Parity Guaranteed)
The classification branch layers are geometrically and functionally identical between Config A and Config B:
1. `dense_512` kernel $\mathbf{W}_{\text{d512}} \in \mathbb{R}^{3072 \times 512}$ and bias $\mathbf{b}_{\text{d512}} \in \mathbb{R}^{512}$
2. `clf` kernel $\mathbf{W}_{\text{clf}} \in \mathbb{R}^{512 \times 4}$ and bias $\mathbf{b}_{\text{clf}} \in \mathbb{R}^{4}$

**Parity Protocol:**  
Within each fold $f \in \{1..5\}$, an ephemeral JSON weight template is instantiated using seed $S_f = \text{SEED} + f$ (for `dense_512`) and $S_f + 1$ (for `clf`). These exact floating-point arrays are loaded into both Config A and Config B before training commences.
- **Empirical Status:** 100% bit-for-bit parity verified in Gate 9 (`dense_512` and `clf` initial weights match with 0.000000 maximum absolute difference).

### 3.2 Unavoidable Initialization Differences (Formally Documented)
The projection heads differ fundamentally in topological structure and parameter shapes:
- **Config A:** Has a single projection matrix $\mathbf{W}_{\text{emb}} \in \mathbb{R}^{3072 \times 128}$ (393,216 weights).
- **Config B:** Has two projection matrices: $\mathbf{W}_{\text{mlp1}} \in \mathbb{R}^{3072 \times 512}$ (1,572,864 weights) and $\mathbf{W}_{\text{mlp2}} \in \mathbb{R}^{512 \times 128}$ (65,536 weights).

Because the matrix dimensions are non-isomorphic ($3072 \times 128 \neq 3072 \times 512$), **these weights cannot be shared directly**. 
- Config A projection head is initialized with Glorot Uniform under seed $S_f + 2$.
- Config B projection head is initialized with Glorot Uniform under seeds $S_f + 2$ (layer 1) and $S_f + 3$ (layer 2).
- **Explicit Statement:** We do **not** claim identical initialization for the projection heads. The initialization divergence is strictly dictated by the dimensional requirements of the architectural hypothesis being tested.

---

## 4. Controlled Training & Evaluation Settings

All training hyperparameters, loss calculations, and evaluation routines are locked:

| Hyperparameter | Value | Description |
|:---|:---:|:---|
| **Contrastive Temperature ($\tau$)** | $0.05$ | Fixed empirical optimum from EXP-06A |
| **Contrastive Loss Weight ($\lambda$)** | $0.50$ | Fixed empirical optimum from EXP-06B |
| **Batch Size** | $16$ | Fixed across both configurations |
| **Optimizer** | AdamW | `learning_rate=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0` |
| **Learning Rate Schedule** | Constant with AdamW | Identical decay mechanics |
| **Training Policy** | Min 100, Max 150 Epochs | Monitored Validation Macro F1, patience = 15 |
| **Class Weighting Scheme** | $w_c = N_{\text{train}} / (4 \cdot N_c)$ | Dynamic inverse-frequency weighting per fold |
| **Development Cohort** | Approved 80 Participants | 240 scans, zero test or hospital access |
| **Cross-Validation Scheme** | 5-Fold StratifiedGroupKFold | Fixed `seed=42`, strictly collocated scans |
