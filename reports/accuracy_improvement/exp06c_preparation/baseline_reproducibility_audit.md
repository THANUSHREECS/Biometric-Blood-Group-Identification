# EXP-06C: Baseline Reproducibility Audit

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Lead Auditor:** Senior ML Research Scientist & Verification Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, configurations, audits, and reproducibility evaluations herein are academic research prototypes studying deep learning representation mechanics on biometric optical fingerprint representations. They **must not** be used for clinical blood grouping, medical diagnostics, patient transfusion matching, or triage.

---

## 1. Audit Scope & Purpose

This audit provides a rigorous, line-by-line inspection comparing the historical **EXP-06B Arm D** reference implementation (`experiments/exp06b/run_exp06b.py`) with the proposed **EXP-06C Config A** implementation (`experiments/exp06c/run_exp06c.py`).

The objective is to verify whether Config A constitutes an **exact numerical reproduction** or a **fresh linear reference experiment**, and to establish absolute methodological clarity before real-data training is authorized.

---

## 2. Line-by-Line Implementation Comparison Matrix

| Dimension | EXP-06B Arm D (Historical Reference) | EXP-06C Config A (Proposed Linear Reference) | Exact Code Match? | Methodological Impact |
|:---|:---|:---|:---:|:---|
| **Script Path** | `experiments/exp06b/run_exp06b.py` | `experiments/exp06c/run_exp06c.py` | Isolated | Standalone scripts in dedicated experiment directories. |
| **Cohort Definition** | 80 Development Participants (240 scans) | 80 Development Participants (240 scans) | **YES** | Identical `SPLIT_CSV` filtering (`TRAIN` & `VALIDATION`). |
| **Participant Ordering** | `sorted(dev_df["participant_id"].unique())` | `sorted(dev_df["participant_id"].unique())` | **YES** | Identical lexicographical participant sorting. |
| **Scan Preprocessing** | `transform_c5(str(img_p))` | `transform_c5(str(img_p))` | **YES** | Identical unaugmented C5 contrast/ridge preprocessing. |
| **Fold Partitioning** | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | **YES** | Identical 5-fold participant-grouped split (64 train / 16 val). |
| **Backbone Architecture** | DenseNet-121 (Pretrained ImageNet, Frozen) | DenseNet-121 (Pretrained ImageNet, Frozen) | **YES** | `base.trainable = False`, 7,037,504 non-trainable parameters. |
| **Backbone Feature Pooling** | `GlobalAveragePooling2D` per scan (1024-d each) | `GlobalAveragePooling2D` per scan (1024-d each) | **YES** | Fused concatenated dimension $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{3072}$. |
| **Classifier Head** | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | `Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)` | **YES** | Identical classifier layer hierarchy and names. |
| **Projection Head** | `Dense(128, activation=None, name="emb")` | `Dense(128, activation=None, name="emb")` | **YES** | Identical single linear projection matrix $\mathbf{W} \in \mathbb{R}^{3072 \times 128}$. |
| **Loss Parameters** | $\tau = 0.05, \lambda = 0.50$ | $\tau = 0.05, \lambda = 0.50$ | **YES** | Identical temperature and contrastive penalty. |
| **Loss Calculation** | `loss_clf + (lambda_supcon * loss_supcon)` | `loss_clf + (lambda_supcon * loss_supcon)` | **YES** | Identical multi-task scalar combination. |
| **Class Weighting** | $w_c = N_{\text{train}} / (4 \cdot N_c)$ | $w_c = N_{\text{train}} / (4 \cdot N_c)$ | **YES** | Identical dynamic balanced inverse-frequency weights per fold. |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | **YES** | Identical optimizer instantiation and gradient clipping. |
| **Training Policy** | Min 100, Max 150 epochs, patience 15 | Min 100, Max 150 epochs, patience 15 | **YES** | Identical evaluation floor and patience thresholds. |
| **Checkpoint Metric** | Validation Macro F1 on held-out fold data | Validation Macro F1 on held-out fold data | **YES** | `MacroF1Checkpoint` saves best validation weights. |
| **Process Execution Order** | **Executed 4th** (after Arms A, B, and C in single process) | **Executed 1st** (first configuration in fresh process) | **NO** | Global TF RNG state and CUDA context differ at runtime. |
| **Batch Shuffling Sequence** | Shuffled via global TF RNG after ~1,500 prior epochs | Shuffled via fresh global TF RNG from Step 0 | **NO** | Epoch-by-epoch mini-batch ordering will naturally vary. |

---

## 3. Empirical Verification of Initial Weights

To evaluate whether the Step-0 starting weights of Config A are identical to EXP-06B, both model builders were instantiated under identical seeds ($S_f = \text{SEED} + f = 43$ for Fold 1):

```python
m_b, _, h_b = build_exp06b_model(lambda_supcon=0.50, seed_head=43)
m_c, _, h_c = build_exp06c_model(head_type='linear', seed_head=43)
```

**Empirical Verification Results:**
- `dense_512` initial weights: $\max |\mathbf{W}_b - \mathbf{W}_c| = \mathbf{0.000000}$ (`np.allclose == True`)
- `clf` initial weights: $\max |\mathbf{W}_b - \mathbf{W}_c| = \mathbf{0.000000}$ (`np.allclose == True`)
- `emb` initial weights: $\max |\mathbf{W}_b - \mathbf{W}_c| = \mathbf{0.000000}$ (`np.allclose == True`)

**Finding:** At initialization time (Epoch 0, Step 0), the parameter values of Config A in EXP-06C are **100% mathematically identical** to the initial template generated in EXP-06B.

---

## 4. Formal Reproducibility Classification

### Determination: **Fresh Linear Reference Experiment** (NOT Exact Replay)

We formally classify Config A as a **fresh linear reference experiment** rather than an exact bit-for-bit reproduction of EXP-06B Arm D, for the following verified scientific reasons:

1. **Process Order Discrepancy:**
   - In EXP-06B, Arm D executed as the fourth sequential arm within a single continuous Python session lasting over 8 hours (preceded by Arm A: 78 min, Arm B: 108 min, Arm C: 135 min). Over 1,500 training epochs preceded Arm D, altering the internal pseudo-random state of TensorFlow, the CUDA context, and the memory allocator.
   - In EXP-06C, Config A will execute as the **first configuration in a fresh, isolated Python process**.
2. **Mini-Batch Shuffling Divergence:**
   - In Keras, `model.fit(..., shuffle=True)` draws batch permutations from the process's evolving global pseudo-random state. Because the process state in EXP-06C is at Epoch 0 rather than Epoch 1,500+, the mini-batch permutation sequence during training will follow a different stochastic trajectory.
3. **GPU Floating-Point Non-Determinism:**
   - Modern GPU convolution and gradient reductions involve atomic accumulation operations that exhibit minor floating-point non-determinism across runs.

### Methodological Advantage of a Fresh Linear Reference
Training Config A synchronously alongside Config B in EXP-06C is **methodologically superior** to reusing historical checkpoints:
- **Internal Validity:** Config A and Config B execute within the exact same process environment and share identical starting seeds, eliminating process-order confounding.
- **Fair Paired Evaluation:** Any observed performance difference between Config A and Config B can be attributed strictly to the architectural intervention (projection head), rather than cross-process drift.
- **Secondary Drift Tracking:** Comparing Config A against historical EXP-06B Arm D ($0.4925$ Macro F1, $52.50\%$ accuracy) provides an explicit check on baseline stability.

---

## 5. Audit Conclusion

- The implementation of Config A in `run_exp06c.py` matches EXP-06B Arm D in code structure, parameter count, loss definition, hyperparameters, and starting weight formulas.
- We explicitly **do not claim** that Config A will produce a bit-for-bit identical convergence trace to EXP-06B Arm D.
- Config A is properly documented as an independent, concurrent linear reference benchmark for EXP-06C.
