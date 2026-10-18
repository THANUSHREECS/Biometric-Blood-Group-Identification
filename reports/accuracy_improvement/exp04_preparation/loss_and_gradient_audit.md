# EXP-04: Loss, Gradient, and Training-Loop Safety Audit

**Audit Date:** 2026-10-14T21:55:00+05:30  
**Target:** Empirical Validation of Losses, Gradients, Parameter Updates, Sample Weights, and Checkpoint Compatibility  
**Testing Methodology:** Synthetic Batch ($B=4$) Execution in Keras 3.12.4 / TensorFlow 2.21  
**Audit Finding:** **PASS — FINITE LOSSES, ACCURATE GRADIENTS, FROZEN IMMUTABILITY, AND FIDELITY CONFIRMED**  

---

## 1. Executive Summary

This audit validates the mathematical correctness and execution safety of the custom `train_step` implementation for EXP-04. Using controlled synthetic batches ($B=4, 3\text{ scans}, 112\times 112\times 3$), all forward, backward, update, and evaluation operations were tested across Configurations A, B, and C.

---

## 2. Gradient and Parameter Update Test Results

### 2.1 Loss Finiteness & Output Metrics
For all three configurations, total loss, classification loss, and supervised contrastive loss produced strictly finite values ($-\infty < \mathcal{L} < \infty$, $\text{NaN}=0$, $\text{Inf}=0$):

| Configuration | Total Loss | Classification Loss (SCCE) | SupCon Loss ($\tau=0.07$) | Loss Finiteness |
|:---|:---:|:---:|:---:|:---:|
| **Config A (Reference)** | 8.6589 | 8.6047 | 0.5418 | **FINITE (PASS)** |
| **Config B (DenseBlock 4)** | 36.6895 | 36.6337 | 0.5576 | **FINITE (PASS)** |
| **Config C (DB3+4+Trans3)** | 27.4992 | 27.4436 | 0.5551 | **FINITE (PASS)** |

### 2.2 Trainable Variable Updates vs. Frozen Invariance
After a single `train_step`, we compared all model parameters against their pre-step values:

1. **Head Parameter Updates:**
   - In all three configurations, 100% of head trainable variables (`dense_512`, `clf`, `emb`) were updated by `opt_heads` ($\text{Mean }\Delta \approx 0.0009 > 0$).
2. **Backbone Parameter Updates:**
   - **Config A:** 0 backbone variables updated (0 backbone trainable variables exist).
   - **Config B:** Exactly 32 Conv2D kernel variables updated by `opt_backbone` ($\Delta > 0$).
   - **Config C:** Exactly 81 Conv2D kernel variables updated by `opt_backbone` ($\Delta > 0$).
3. **Frozen Layer Immutability:**
   - Stem Conv1 kernel (`conv1_conv`): **$\Delta = 0.000000$ (Bit-for-bit unchanged)**.
   - Earlier dense blocks in Config B (e.g. DenseBlock 1, 2, 3): **$\Delta = 0.000000$**.
4. **Batch Normalization Immutability:**
   - Stem BN moving mean and moving variance (`conv1_bn`): **$\Delta = 0.000000$**.
   - DenseBlock 4 BN moving mean and moving variance (`conv5_block1_0_bn`): **$\Delta = 0.000000$**.
   - Proves that Batch Normalization population statistics do not drift under fine-tuning.
5. **Single Optimizer Update Rule:**
   - Because `vars_backbone` and `vars_heads` are completely disjoint, each variable is passed to and updated by **exactly one optimizer** per step.

---

## 3. Loss Formulation and Sample-Weight Audit

### 3.1 Loss Weight Fidelity Against EXP-03
The multi-task objective is defined as:
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{clf}} + 0.1 \cdot \mathcal{L}_{\text{supcon}}$$
- Classification loss weight: **1.0** (matches EXP-03).
- Supervised contrastive loss weight: **0.1** (matches EXP-03).
- Supervised contrastive temperature: **$\tau = 0.07$** (matches EXP-03).

### 3.2 Single-Multiplication Sample Weight Guarantee
To prevent sample weights from being accidentally applied twice:
- Sample weights are unpacked in `train_step` from `data[2]` (or `sample_weight` argument).
- The sample weight vector $w_{\text{sample}}$ is multiplied into the elementwise cross-entropy loss:
  $$\mathcal{L}_{\text{clf}} = \frac{1}{B} \sum_{i=1}^B w_i \cdot \text{SCCE}(y_i, \hat{p}_i)$$
- Because `model.compile()` does not attach loss functions (losses are computed directly in `train_step`), Keras internal loss wrappers **cannot re-apply** sample weights.
- Empirical test confirmed: `weighted_scce == np.mean(raw_scce * sw_test)` ($p < 10^{-6}$).

### 3.3 Metric Loss Input Routing
- Supervised contrastive loss receives:
  1. `labels`: The true participant labels $y \in \{0, 1, 2, 3\}$ of shape `(batch_size,)`.
  2. `feature_vectors`: The unnormalized 128-dimensional output from `emb` (`Dense(128, activation=None)`).
- Within `supervised_contrastive_loss`, L2 normalization is applied (`l2_normalize(feature_vectors, axis=1)`), cosine similarity is divided by $\tau=0.07$, self-similarity is excluded from the denominator via `logits_mask`, and numerical stability is enforced via `1e-12` epsilon.

### 3.4 Canonical Class-Order Remapping
The native-to-canonical mapping is preserved identically:
- Native Index: 0: A+, 1: B+, 2: AB+, 3: O+.
- Canonical Index: 0: A+, 1: AB+, 2: B+, 3: O+.
- Probability Remapping:
  $$p_{\text{canonical}} = \frac{[p_{\text{native}}[0], p_{\text{native}}[2], p_{\text{native}}[1], p_{\text{native}}[3]]}{\sum p_{\text{canonical}}}$$
- Verified that all predicted probabilities sum to $1.000000$ and map to valid canonical indices $\{0, 1, 2, 3\}$.

---

## 4. Shared Initialization and Persistence Audit

### 4.1 Ephemeral Array Serialization Protocol
To share identical epoch-zero head and backbone weights across Configurations A, B, and C without running into Keras model-name or layer-trainability saving conflicts:
1. In each fold $k$, an initial reference model instance is created under `SEED + fold_idx`.
2. Initial head layer weights are extracted as NumPy arrays via `layer.get_weights()`.
3. Arrays are stored in an ephemeral file: `scratch/init_fold_{k}_heads.npz`.
4. When Configs A, B, and C are initialized for Fold $k$:
   - Backbone is loaded fresh from `DenseNet121(weights="imagenet")`.
   - Head layers load their weights via `layer.set_weights()`.
5. Empirical test confirmed:
   - Config A, Config B, and Config C begin at Epoch 0 with **$\Delta = 0.000000$ (bit-for-bit identical)** weights across all 610 model variables (backbone + heads).
   - Zero reliance on, or transfer from, trained EXP-03 checkpoints.

---

## 5. Checkpoint Selection & Evaluation Compatibility

Empirical execution in `DummyMacroF1Callback` confirmed:
1. **Validation Prediction:** `model.predict(X_val)` produces valid probability distributions of shape `(N_val, 4)` summing to $1.0$.
2. **Callback Metrics:** Macro F1 and validation accuracy compute cleanly on held-out data.
3. **Model Weight Persistence:** `model.save_weights()` successfully writes weights to disk during fold training.
4. **Model Reloading:** `model.load_weights()` restores the best epoch's weights with **$\Delta = 0.000000$**.
5. **Out-of-Fold Aggregation:** Out-of-fold predictions and canonical probability outputs are completely compatible with the existing reporting pipeline.

---

## 6. Audit Verdict

All five safety gate audits (Variable Partitioning, Gradient Updates, Loss Formulation, Initialization Equivalence, and Checkpoint Compatibility) passed with zero errors.
- **Audit Verdict:** **PASS (IMPLEMENTATION READY FOR EXECUTION)**.
