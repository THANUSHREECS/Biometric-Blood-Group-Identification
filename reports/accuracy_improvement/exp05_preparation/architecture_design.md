# EXP-05: Three-Scan Feature Fusion Architecture Design

**Audit Date:** 2026-10-15T09:28:00+05:30  
**Scope:** Architectural Specifications and Mathematical Formulations for Configurations A, B, and C in EXP-05  
**Audit Finding:** **PASS — ARCHITECTURE RIGOROUSLY SPECIFIED & AUDITED**  

---

## 1. Research Motivation & Theoretical Foundation

In biometric blood group detection using optical fingerprint captures (Mantra MFS100), the input for each participant consists of three separate fingerprint impressions ($S_1, S_2, S_3$). In the baseline E2 model:
- Each scan is independently processed by a pretrained DenseNet-121 backbone followed by Global Average Pooling, yielding three 1024-dimensional feature vectors: $F_1, F_2, F_3 \in \mathbb{R}^{1024}$.
- The baseline fuses these features by naive concatenation:
  $$F_{\text{fused}} = [F_1 \,\|\, F_2 \,\|\, F_3] \in \mathbb{R}^{3072}$$

### Theoretical Limitations of Concatenation:
1. **Equal and Static Weighting:** Concatenation implicitly assumes all three scans provide equally salient, distortion-free biometric features. In practice, optical finger placement varies in pressure, moisture, partial smudging, and core alignment. Concatenation lacks the capacity to dynamically downweight a corrupted or noisy scan.
2. **Ignored Cross-Scan Correlations:** Concatenation relies entirely on the post-fusion dense projection (`dense_512`) to discover non-linear inter-scan interactions. With a limited development cohort ($N=80$ participants), learning complex 3072-dimensional cross-scan relations without regularized attention or gating leads to overfitting.
3. **Parameter Footprint:** Projecting a 3072-D vector requires 1.57M parameters in `dense_512` alone, whereas an adaptive fusion neck can operate in a more compact, expressive subspace.

EXP-05 tests two distinct, lightweight fusion paradigms against the concatenation baseline while keeping the pretrained DenseNet-121 backbone strictly frozen:
- **Configuration B (Gated Feature Fusion):** Learns adaptive, input-dependent scalar importance weights $\alpha_1, \alpha_2, \alpha_3 \in [0, 1]$ satisfying $\sum_{i=1}^3 \alpha_i = 1$.
- **Configuration C (Cross-Scan Self-Attention):** Models pairwise and triplet semantic interactions among the three scan embeddings via a multi-head self-attention sequence operator.

---

## 2. Configuration Architectural Specifications

All three configurations share the identical feature extraction trunk:
$$\text{Input: } X \in \mathbb{R}^{B \times 3 \times 112 \times 112 \times 3} \implies S_1, S_2, S_3 \in \mathbb{R}^{B \times 112 \times 112 \times 3}$$
$$F_i = \text{GAP}(\text{DenseNet121}(S_i)) \in \mathbb{R}^{B \times 1024}, \quad \text{for } i \in \{1, 2, 3\}$$
*DenseNet-121 backbone parameters ($7,037,504$) and Batch Normalization moving statistics are strictly frozen across all configurations.*

```
                     ┌───────────────────────────────────────────────┐
                     │          Participant Input (3 Scans)          │
                     │             (B, 3, 112, 112, 3)               │
                     └───────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       ▼                     ▼                     ▼
                 ┌───────────┐         ┌───────────┐         ┌───────────┐
                 │  Scan 1   │         │  Scan 2   │         │  Scan 3   │
                 │(112,112,3)│         │(112,112,3)│         │(112,112,3)│
                 └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
                       │                     │                     │
                       ▼                     ▼                     ▼
                 ┌───────────┐         ┌───────────┐         ┌───────────┐
                 │DenseNet121│         │DenseNet121│         │DenseNet121│
                 │ (FROZEN)  │         │ (FROZEN)  │         │ (FROZEN)  │
                 └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
                       │                     │                     │
                       ▼                     ▼                     ▼
                 ┌───────────┐         ┌───────────┐         ┌───────────┐
                 │  GAP(F1)  │         │  GAP(F2)  │         │  GAP(F3)  │
                 │ (B, 1024) │         │ (B, 1024) │         │ (B, 1024) │
                 └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
                       │                     │                     │
                       └─────────────────────┼─────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
    │  CONFIG A: Neck │             │  CONFIG B: Neck │             │  CONFIG C: Neck │
    │  Concatenation  │             │  Gated Fusion   │             │ Self-Attention  │
    │   ([F1,F2,F3])  │             │ (Softmax Gate)  │             │(MHA + FFN + LN) │
    │    (B, 3072)    │             │    (B, 1024)    │             │    (B, 768)     │
    └────────┬────────┘             └────────┬────────┘             └────────┬────────┘
             │                               │                               │
             └───────────────────────────────┼───────────────────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             ┌───────────────────┐                       ┌───────────────────┐
             │Classification Head│                       │Contrastive Branch │
             │Dense(512)+Drop(0.5│                       │    Dense(128)     │
             │   +Dense(4,soft)  │                       │   (Metric Loss)   │
             └───────────────────┘                       └───────────────────┘
```

---

### 2.1 Configuration A: Concatenation Baseline

- **Neck Operation:**
  $$F_{\text{fused}} = \text{Concatenate}([F_1, F_2, F_3]) \in \mathbb{R}^{B \times 3072}$$
- **Classification Head:**
  $$h_1 = \text{ReLU}(W_{\text{dense}} F_{\text{fused}} + b_{\text{dense}}), \quad W_{\text{dense}} \in \mathbb{R}^{3072 \times 512}, \; b_{\text{dense}} \in \mathbb{R}^{512}$$
  $$h_{\text{drop}} = \text{Dropout}_{0.5}(h_1)$$
  $$\hat{y}_{\text{clf}} = \text{Softmax}(W_{\text{clf}} h_{\text{drop}} + b_{\text{clf}}), \quad W_{\text{clf}} \in \mathbb{R}^{512 \times 4}, \; b_{\text{clf}} \in \mathbb{R}^{4}$$
- **Embedding Head:**
  $$z_{\text{emb}} = W_{\text{emb}} F_{\text{fused}} + b_{\text{emb}}, \quad W_{\text{emb}} \in \mathbb{R}^{3072 \times 128}, \; b_{\text{emb}} \in \mathbb{R}^{128}$$
- **Trainable Parameters:**
  - `dense_512`: $3072 \times 512 + 512 = 1,573,376$
  - `clf`: $512 \times 4 + 4 = 2,052$
  - `emb`: $3072 \times 128 + 128 = 393,344$
  - Total: **1,968,772** (6 variables).

---

### 2.2 Configuration B: Gated Feature Fusion

- **Motivation:** Dynamically measure scan quality and ridge saliency by learning a participant-specific softmax gating score for each impression.
- **Layer Implementation (`GatedScanFusion`):**
  1. **Feature Stacking:**
     $$F_{\text{stacked}} = \text{Stack}([F_1, F_2, F_3], \text{axis}=1) \in \mathbb{R}^{B \times 3 \times 1024}$$
  2. **Shared Gating Projection:**
     A lightweight multi-layer perceptron processes each scan's 1024-D representation independently:
     $$g_i = \text{ReLU}(W_{g1} F_i + b_{g1}), \quad W_{g1} \in \mathbb{R}^{1024 \times 128}, \; b_{g1} \in \mathbb{R}^{128}$$
     $$s_i = W_{g2} g_i + b_{g2}, \quad W_{g2} \in \mathbb{R}^{128 \times 1}, \; b_{g2} \in \mathbb{R}^{1}$$
     Unnormalized logits: $s = [s_1, s_2, s_3] \in \mathbb{R}^{B \times 3}$.
  3. **Softmax Gate Normalization:**
     $$\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^3 \exp(s_j)}, \quad \text{for } i \in \{1, 2, 3\}$$
     *Guarantees: $\alpha_i \ge 0$, $\sum_{i=1}^3 \alpha_i = 1.0$, and gating is strictly computed from features without labels or target leakage.*
  4. **Weighted Fusion:**
     $$F_{\text{fused}} = \sum_{i=1}^3 \alpha_i F_i \in \mathbb{R}^{B \times 1024}$$
- **Downstream Heads:**
  - `dense_512`: $1024 \times 512 + 512 = 524,800$
  - `clf`: $512 \times 4 + 4 = 2,052$
  - `emb`: $1024 \times 128 + 128 = 131,200$
- **Neck Trainable Parameters:**
  - `gate_dense1`: $1024 \times 128 + 128 = 131,200$
  - `gate_logit`: $128 \times 1 + 1 = 129$
  - Neck Total: **131,329** (4 variables).
- **Total Trainable Parameters:**
  $$131,329 + 524,800 + 2,052 + 131,200 = \mathbf{789,381}\text{ parameters (10 variables)}.$$
  *(60.0% parameter reduction compared to Configuration A).*

---

### 2.3 Configuration C: Cross-Scan Self-Attention

- **Motivation:** Scans of different fingers from the same participant share an underlying systemic antigen phenotype. Multi-head self-attention computes query-key affinity matrices across scans to exchange complementary ridge frequency and orientation cues.
- **Layer Implementation (`CrossScanAttentionFusion`):**
  1. **Input Sequence Representation:**
     $$F_{\text{stacked}} = \text{Stack}([F_1, F_2, F_3], \text{axis}=1) \in \mathbb{R}^{B \times 3 \times 1024}$$
  2. **Subspace Projection:**
     Projects the high-dimensional backbone vector into a compact attention dimension ($d_{\text{model}} = 256$):
     $$H_0 = W_{\text{proj}} F_{\text{stacked}} + b_{\text{proj}}, \quad W_{\text{proj}} \in \mathbb{R}^{1024 \times 256}, \; b_{\text{proj}} \in \mathbb{R}^{256}$$
     $$H_0 \in \mathbb{R}^{B \times 3 \times 256}$$
  3. **Multi-Head Self-Attention ($h=4, d_k=64$):**
     $$\text{Head}_j = \text{Softmax}\left(\frac{Q_j K_j^T}{\sqrt{d_k}}\right) V_j, \quad Q_j, K_j, V_j \in \mathbb{R}^{B \times 3 \times 64}$$
     $$\text{MHA}(H_0) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_4) W_O$$
     $$\text{MHA}(H_0) \in \mathbb{R}^{B \times 3 \times 256}$$
  4. **Residual Connection & Layer Normalization:**
     $$H_1 = \text{LayerNorm}(H_0 + \text{MHA}(H_0)) \in \mathbb{R}^{B \times 3 \times 256}$$
  5. **Position-Wise Feed-Forward Network (FFN):**
     $$\text{FFN}(H_1) = W_2 \cdot \text{ReLU}(W_1 H_1 + b_1) + b_2$$
     $$W_1 \in \mathbb{R}^{256 \times 512}, \; b_1 \in \mathbb{R}^{512}; \quad W_2 \in \mathbb{R}^{512 \times 256}, \; b_2 \in \mathbb{R}^{256}$$
  6. **Second Residual Connection & Layer Normalization:**
     $$H_2 = \text{LayerNorm}(H_1 + \text{FFN}(H_1)) \in \mathbb{R}^{B \times 3 \times 256}$$
  7. **Aggregation:**
     Flattens the 3 contextualized scan tokens into a single sequence-aware fused representation:
     $$F_{\text{fused}} = \text{Flatten}(H_2) \in \mathbb{R}^{B \times 768}$$
- **Downstream Heads:**
  - `dense_512`: $768 \times 512 + 512 = 393,728$
  - `clf`: $512 \times 4 + 4 = 2,052$
  - `emb`: $768 \times 128 + 128 = 98,432$
- **Neck Trainable Parameters:**
  - `attn_input_proj`: $1024 \times 256 + 256 = 262,400$
  - `cross_scan_mha`: $3 \times (256 \times 256 + 256) + (256 \times 256 + 256) = 263,168$
  - `attn_ln1`: $2 \times 256 = 512$
  - `ffn_dense1`: $256 \times 512 + 512 = 131,584$
  - `ffn_dense2`: $512 \times 256 + 256 = 131,328$
  - `attn_ln2`: $2 \times 256 = 512$
  - Neck Total: **789,504** (16 variables).
- **Total Trainable Parameters:**
  $$789,504 + 393,728 + 2,052 + 98,432 = \mathbf{1,283,716}\text{ parameters (24 variables)}.$$
  *(34.8% parameter reduction compared to Configuration A).*

---

## 3. Comparative Architecture & Parameter Ledger

| Property | Configuration A (Concatenation) | Configuration B (Gated Fusion) | Configuration C (Cross-Scan MHA) |
|:---|:---:|:---:|:---:|
| **Fusion Operator** | `Concatenate(axis=-1)` | `GatedScanFusion` (Softmax Gate) | `CrossScanAttentionFusion` (MHA+FFN) |
| **Fusion Mechanism** | Static stacking | Adaptive scalar importance $\alpha_i$ | Dynamic pairwise token attention |
| **Fused Vector Dim ($F_{\text{fused}}$)** | **3072** | **1024** | **768** |
| **Neck Trainable Parameters** | **0** | **131,329** (4 variables) | **789,504** (16 variables) |
| **`dense_512` Parameters** | 1,573,376 | 524,800 | 393,728 |
| **`clf` Parameters** | 2,052 | 2,052 | 2,052 |
| **`emb` Parameters** | 393,344 | 131,200 | 98,432 |
| **Total Trainable Head Params** | 1,968,772 | 658,052 | 494,212 |
| **Total Trainable Model Params** | **1,968,772** (100.0%) | **789,381** (40.1% of Config A) | **1,283,716** (65.2% of Config A) |
| **Frozen Backbone Parameters** | 7,037,504 | 7,037,504 | 7,037,504 |
| **Trainable Variables Count** | **6** | **10** | **24** |
| **Batch Normalization Policy** | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) | Frozen ($\Delta\mu=0, \Delta\sigma^2=0$) |
| **Information Leakage Potential**| None | None (computed from features only) | None (computed within participant) |
| **Checkpoint Serialization** | Standard Keras `.weights.h5` | Standard Keras `.weights.h5` | Standard Keras `.weights.h5` |

---

## 4. Initialization & Weight Sharing Constraints

Because the three fusion mechanisms yield different fused representation dimensionalities (3072 for Config A, 1024 for Config B, 768 for Config C), the weight matrices of `dense_512` and `emb` have inherently incompatible input shapes:
- Config A `dense_512` kernel: shape `(3072, 512)`
- Config B `dense_512` kernel: shape `(1024, 512)`
- Config C `dense_512` kernel: shape `(768, 512)`

### Audited Initialization Fairness Rules:
1. **Pretrained Backbone Weights:** Pretrained ImageNet weights are strictly identical across all three configurations.
2. **Classification Output Layer (`clf`):** The output projection layer `clf` has shape `(512, 4)` and bias `(4,)`, which is **identical across all three configurations**. For each fold $k$, the initial weights of `clf` are generated under seed $(42 + k)$ and shared bit-for-bit across Configurations A, B, and C.
3. **Architecture-Specific Initializations:** Layers with incompatible shapes (`dense_512`, `emb`, `gate_dense1`, `attn_input_proj`, etc.) are initialized reproducibly using standard Glorot Uniform initialization seeded by `SEED + fold_idx`.
4. **No Warm-Start Transfer:** No configuration is initialized from a previously trained checkpoint from EXP-03 or EXP-04.
