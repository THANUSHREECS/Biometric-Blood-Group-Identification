# EXP-04: Learning-Rate Control & Parameter-Group Optimization Audit

**Audit Date:** 2026-10-14T21:35:00+05:30  
**Target:** Implementation Feasibility and Rigor of Differential Learning Rates Across Backbone and Head Parameters  
**Audit Finding:** **PASS — ARCHITECTURE VALIDATED & CONSTRAINTS FORMALIZED**  

---

## 1. Executive Summary & Problem Formulation

In partial fine-tuning experiments, applying an aggressive learning rate ($10^{-3}$) to pretrained convolutional weights can disrupt calibrated feature representations (catastrophic forgetting). Conversely, applying a conservative fine-tuning rate ($10^{-4}$ or $5\times 10^{-5}$) to freshly initialized classification and embedding heads can severely retard convergence, causing underfitting within a fixed epoch horizon ($150$ epochs).

To isolate the causal effect of backbone unfreezing, the preferred experimental design requires:
- **Configuration A (Reference):** Frozen backbone, Heads $\text{LR} = 10^{-3}$.
- **Configuration B (DenseBlock 4):** Backbone Conv2D $\text{LR} = 10^{-4}$, Heads $\text{LR} = 10^{-3}$.
- **Configuration C (DenseBlock 3 + Transition 3 + DenseBlock 4):** Backbone Conv2D $\text{LR} = 5\times 10^{-5}$, Heads $\text{LR} = 10^{-3}$.

This audit resolves whether and how the implementation genuinely applies these differential rates to the correct parameter groups without introducing confounding training loop artifacts.

---

## 2. Theoretical & Mathematical Constraints in Keras 3 / AdamW

### 2.1 The Gradient Scaling Fallacy in Adaptive Optimizers
In standard SGD with momentum, scaling the loss or multiplying layer gradients by a scalar $\alpha$ effectively scales the parameter update by $\alpha$. However, **this does not hold for Adam or AdamW**.

Adam parameter updates are governed by:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_{t-1}$$

If gradients $g_t$ are scaled by factor $\alpha$:
$$\hat{m}_t \propto \alpha g_t, \quad \hat{v}_t \propto \alpha^2 g_t^2 \implies \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \approx \frac{\alpha g_t}{\alpha \sqrt{g_t^2}} = \text{sgn}(g_t)$$
The update magnitude is **scale-invariant** with respect to gradient scaling. Therefore, merely multiplying loss weights or gradients cannot achieve differential learning rates in AdamW. Genuine differential learning rates require separate optimizer state updates or explicit learning rate scaling at the optimizer step level.

### 2.2 Native API Absence in Keras 3
Unlike PyTorch (which supports parameter group dictionaries `[{'params': bb, 'lr': 1e-4}, {'params': hd, 'lr': 1e-3}]`), TensorFlow / Keras 3 optimizers do not provide a native parameter-group dictionary interface in `tf.keras.optimizers.AdamW`. Furthermore, Keras 3 `BaseOptimizer` strictly validates variable consistency via `_check_variables_are_known()`, requiring that each optimizer instance only operates on its declared variable subset.

---

## 3. Evaluated Implementation Options

### Option 1: Dual-Optimizer Custom `train_step` Architecture (RECOMMENDED)
Wrap the functional E2 model in a subclassed `E2DualOptimizerModel(tf.keras.Model)` that overrides `train_step(data)`.

#### Mechanism:
1. Two separate `AdamW` instances are instantiated per fold:
   - `opt_backbone = AdamW(learning_rate=LR_BACKBONE, weight_decay=1e-4)`
   - `opt_heads = AdamW(learning_rate=1e-3, weight_decay=1e-4)`
2. In `train_step`, a single persistent `tf.GradientTape` records forward operations across all parameters.
3. Total multi-task loss is computed:
   $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{clf}}(y, \hat{y}_{\text{clf}}; w_{\text{sample}}) + 0.1 \cdot \mathcal{L}_{\text{supcon}}(y, \hat{z}_{\text{emb}}; \tau=0.07)$$
4. Trainable variables are partitioned into disjoint sets:
   - `vars_backbone = [v for v in self.trainable_variables if 'densenet121' in v.name or 'conv' in v.name]`
   - `vars_heads = [v for v in self.trainable_variables if any(h in v.name for h in ['dense_512', 'clf', 'emb'])]`
5. Gradients are computed and applied independently:
   - `grads_backbone = tape.gradient(total_loss, vars_backbone)`
   - `grads_heads = tape.gradient(total_loss, vars_heads)`
   - `self.opt_backbone.apply_gradients(zip(grads_backbone, vars_backbone))`
   - `self.opt_heads.apply_gradients(zip(grads_heads, vars_heads))`

#### Empirical Proof:
Tested in Keras 3.12.4 on Python 3.10 with the full multi-task loss, sample weights, and SupCon. Execution succeeded cleanly with zero shape mismatches, finite losses, and correct optimizer state updates.

---

### Option 2: Single Uniform Learning Rate (Alternative Baseline)
Maintain standard `model.compile(optimizer=AdamW(lr=UNIFORM_LR))` without custom `train_step`.
- Config A: Uniform LR = $10^{-3}$ (only heads are trainable).
- Config B: Uniform LR = $10^{-4}$ (heads and backbone both run at $10^{-4}$).
- Config C: Uniform LR = $5\times 10^{-5}$ (heads and backbone both run at $5\times 10^{-5}$).

#### Methodological Confounder of Option 2:
In Config B, the classification head receives a $10\times$ lower learning rate than in Config A. In Config C, it receives a $20\times$ lower rate. If Config B or C performs poorly, it is impossible to determine whether the degradation is due to backbone fine-tuning or to head underfitting. **Option 2 introduces a severe methodological confounder.**

---

### Option 3: Two-Stage Staged Fine-Tuning (Warm-Up Then Fine-Tune)
- Stage 1: Freeze backbone, train heads for $K$ epochs at $\text{LR}=10^{-3}$.
- Stage 2: Unfreeze backbone, train entire network at uniform $\text{LR}=10^{-4}$ for remaining epochs.

#### Methodological Confounder of Option 3:
Introduces multiple uncalibrated hyperparameters: the transition epoch $K$, whether optimizer momentum buffers are reset at transition, and whether the learning rate is stepped down abruptly. This confounds the comparison against the single-stage EXP-03 protocol.

---

## 4. End-to-End Compatibility Matrix for Option 1 (Dual-Optimizer)

| Component | EXP-03 Protocol | Option 1 Dual-Optimizer Compatibility | Status |
|:---|:---|:---|:---:|
| **Classification Loss** | Sparse categorical crossentropy (weight 1.0) | Formulated identically in `train_step` using `tf.keras.losses.sparse_categorical_crossentropy` | **COMPATIBLE** |
| **Metric Loss** | Supervised Contrastive Loss ($\tau=0.07$, weight 0.1) | Executed identically using existing audited `supervised_contrastive_loss` function | **COMPATIBLE** |
| **Sample Weighting** | Inverse-class frequency weights $w_i = \frac{N_{\text{train}}}{4 N_c}$ | Extracted from `data[2]` and multiplied into sample-wise classification loss | **COMPATIBLE** |
| **Gradient Updates** | Standard AdamW step with weight decay $10^{-4}$ | Disjoint gradient tape extraction and independent `apply_gradients` | **COMPATIBLE** |
| **Model Callbacks** | `MacroF1Checkpoint` monitoring validation Macro F1 | Callbacks receive standard `logs` dictionary (`clf_loss`, `supcon_loss`, `loss`) and evaluate validation fold using `model.predict()` | **COMPATIBLE** |
| **Weight Persistence** | `model.save_weights()` / `model.load_weights()` | Standard Keras functional/model weight saving is fully preserved | **COMPATIBLE** |
| **Configuration A Fidelity**| Single optimizer on heads at $10^{-3}$ | When `vars_backbone` is empty (Config A), `opt_backbone` performs zero updates, behaving identically to single-optimizer baseline | **COMPATIBLE** |

---

## 5. Approved Learning-Rate Specifications

| Configuration | Backbone Status | Backbone Trainable Layers | Backbone Learning Rate | Head Trainable Layers | Head Learning Rate | Optimizer Algorithm | Weight Decay |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A** | Frozen | 0 layers | N/A (Frozen) | `dense_512`, `clf`, `emb` | **1e-3** | AdamW | 1e-4 |
| **Config B** | Partial (DB4) | 32 Conv2D layers | **1e-4** | `dense_512`, `clf`, `emb` | **1e-3** | AdamW | 1e-4 |
| **Config C** | Deep (DB3+4+Trans3) | 81 Conv2D layers | **5e-5** | `dense_512`, `clf`, `emb` | **1e-3** | AdamW | 1e-4 |

---

## 6. Audit Verdict

Differential learning rate control via a Dual-Optimizer custom `train_step` architecture is mathematically sound, empirically verified in Keras 3, and completely compatible with all existing losses, sample weights, callbacks, and evaluation logic. It eliminates the head-learning-rate confounder and is formally designated as the approved mechanism for EXP-04.

- **Audit Verdict:** **APPROVED (OPTION 1 DESIGN LOCKED)**.
