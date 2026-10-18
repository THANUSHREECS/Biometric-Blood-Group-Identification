# EXP-06C: Parameter Capacity & Architectural Confound Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Lead Auditor:** Senior ML Research Scientist & Verification Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All architectural evaluations, parameter counts, and capacity audits documented herein are academic research prototypes designed to study neural network representation mechanics on optical biometric features. They **must not** be used for clinical diagnostics, transfusion medicine, blood typing, patient triage, or healthcare decisions.

---

## 1. Executive Summary & Purpose

This audit documents the exact parameter counts recomputed directly from the instantiated model graphs of **Config A (Linear Reference)** and **Config B (Nonlinear MLP)**.

Crucially, this document addresses the **capacity confound** introduced by adding an intermediate hidden layer to the projection head, formally noting that any performance divergence between Config A and Config B cannot be attributed strictly to nonlinearity alone.

---

## 2. Direct Model Parameter Recomputation

Both model graphs were instantiated programmatically via `experiments/exp06c/run_exp06c.py` and audited tensor-by-tensor:

### 2.1 Configuration A: Linear Projection Reference

```
Model: "E2_EXP06C_ConfigA_Linear"
Backbone: DenseNet-121 (Frozen, non-trainable)
```

| Layer / Variable Name | Weight Tensor Shape | Data Type | Parameter Count | Trainable? |
|:---|:---:|:---:|:---:|:---:|
| `base_model` (DenseNet-121) | 364 weight tensors | float32 | **7,037,504** | **NO (Frozen)** |
| `dense_512/kernel:0` | `(3072, 512)` | float32 | $1,572,864$ | YES |
| `dense_512/bias:0` | `(512,)` | float32 | $512$ | YES |
| `clf/kernel:0` | `(512, 4)` | float32 | $2,048$ | YES |
| `clf/bias:0` | `(4,)` | float32 | $4$ | YES |
| `emb/kernel:0` | `(3072, 128)` | float32 | $393,216$ | YES |
| `emb/bias:0` | `(128,)` | float32 | $128$ | YES |
| **Trainable Subtotal** | **6 variables** | — | **1,968,772** | — |
| **Non-Trainable Subtotal** | **364 variables** | — | **7,037,504** | — |
| **Total Parameters (Config A)** | **370 variables** | — | **9,006,276** | — |

---

### 2.2 Configuration B: Two-Layer Nonlinear MLP Projection Head

```
Model: "E2_EXP06C_ConfigB_NonlinearMLP"
Backbone: DenseNet-121 (Frozen, non-trainable)
```

| Layer / Variable Name | Weight Tensor Shape | Data Type | Parameter Count | Trainable? |
|:---|:---:|:---:|:---:|:---:|
| `base_model` (DenseNet-121) | 364 weight tensors | float32 | **7,037,504** | **NO (Frozen)** |
| `dense_512/kernel:0` | `(3072, 512)` | float32 | $1,572,864$ | YES |
| `dense_512/bias:0` | `(512,)` | float32 | $512$ | YES |
| `clf/kernel:0` | `(512, 4)` | float32 | $2,048$ | YES |
| `clf/bias:0` | `(4,)` | float32 | $4$ | YES |
| `proj_mlp_dense1/kernel:0` | `(3072, 512)` | float32 | $1,572,864$ | YES |
| `proj_mlp_dense1/bias:0` | `(512,)` | float32 | $512$ | YES |
| `proj_mlp_dense2/kernel:0` | `(512, 128)` | float32 | $65,536$ | YES |
| `proj_mlp_dense2/bias:0` | `(128,)` | float32 | $128$ | YES |
| **Trainable Subtotal** | **8 variables** | — | **3,214,468** | — |
| **Non-Trainable Subtotal** | **364 variables** | — | **7,037,504** | — |
| **Total Parameters (Config B)** | **372 variables** | — | **10,251,972** | — |

---

## 3. Parameter Capacity Delta Analysis

| Dimension | Config A (Linear Reference) | Config B (Nonlinear MLP) | Absolute Delta ($\Delta$) | Relative Increase |
|:---|:---:|:---:|:---:|:---:|
| **Trainable Head Variables** | 6 variables | 8 variables | $+2$ variables | $+33.3\%$ |
| **Classification Head Params** | 1,575,428 | 1,575,428 | $0$ | $0.0\%$ (Identical) |
| **Projection Head Params** | 393,344 | 1,639,040 | $+1,245,696$ | $+316.7\%$ |
| **Total Trainable Parameters** | **1,968,772** | **3,214,468** | **+1,245,696** | **+63.27%** |
| **Total Model Parameters** | 9,006,276 | 10,251,972 | $+1,245,696$ | $+13.83\%$ |

---

## 4. The Parameter Capacity Confound & Attribution Limits

In accordance with scientific integrity guidelines:
> **"Explicitly document that Config B has more trainable parameters than Config A and that this limits attribution of any improvement solely to nonlinear projection."**

### 4.1 Theoretical Mechanism vs Capacity Expansion
The conceptual motivation for Config B is based on contrastive learning literature (Chen et al., 2020), which posits that an intermediate nonlinear activation (ReLU) allows the model to map task-irrelevant optical noise nonlinearly into the contrastive embedding space without distorting the linear features needed by the classifier.

However, adding an intermediate hidden layer with 512 units inevitably introduces **1,245,696 additional trainable parameters** into the projection head:
1. **Projection Head Capacity Jump:** The projection head capacity increases from $393\text{K}$ to $1.64\text{M}$ weights (a **$4.16\times$ expansion** in projection representation capacity).
2. **Total Trainable Head Capacity Jump:** Total trainable parameters expand by **$+63.27\%$** ($1.97\text{M} \to 3.21\text{M}$).

### 4.2 Formal Attribution Limitation Statement
Because Config B simultaneously alters **two distinct factors**:
1. **Architectural Nonlinearity:** Introduction of the ReLU activation function between `fused` and `out_emb`.
2. **Parameter Capacity:** An increase of $+1,245,696$ trainable weights in the projection pathway.

**Scientific Limitation:**  
If Config B outperforms Config A in empirical evaluation, **it will be impossible to attribute the performance improvement solely to the nonlinear projection mechanism**. The observed gain could be driven by:
- The geometric properties of nonlinear hypersphere clustering, OR
- The sheer representational capacity increase, which provides more gradient bandwidth during contrastive backpropagation, OR
- An interaction effect between both factors.

### 4.3 Why the Proposed Architecture is Retained Without Silent Modification
We do **not** silently shrink Config B (e.g. by reducing the hidden layer to a narrow bottleneck like `Dense(64)`) to match parameter counts, because:
1. Standard contrastive practice (SimCLR, Khosla et al.) explicitly uses hidden layer expansion/preservation ($\mathbb{R}^{3072} \to \mathbb{R}^{512} \to \mathbb{R}^{128}$) rather than extreme dimensional compression, which could induce severe feature bottlenecks.
2. The user's specification explicitly stated: *"Do not silently change the proposed architectures."*
3. By formally auditing and documenting this capacity confound upfront, the experiment remains scientifically sound, transparent, and appropriately qualified in its conclusions.

---

## 5. Audit Conclusion

- Trainable parameter counts are verified: Config A = **1,968,772** (6 variables); Config B = **3,214,468** (8 variables).
- Frozen backbone parameter counts are verified: **7,037,504** non-trainable parameters across both configurations.
- The $+63.27\%$ increase in trainable head capacity is formally documented as an unavoidable attribution confound when evaluating the nonlinear projection hypothesis.
