# EXP-05 Synthetic-Batch Implementation Safety Test Report

**Date:** 2026-10-15  
**Experiment:** EXP-05 — Controlled Three-Scan Feature Fusion  
**Target:** Pre-training implementation verification on synthetic batches (CPU/Deterministic)  
**Status:** PASS (100% compliant across all 15 safety checks)  
**Script:** [`scratch/exp05_synthetic/run_synthetic_safety_tests.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scratch/exp05_synthetic/run_synthetic_safety_tests.py)  
**Result Summary File:** [`scratch/exp05_synthetic/safety_test_results.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scratch/exp05_synthetic/safety_test_results.json)

---

## 1. Executive Summary

In accordance with Phase 4 and Section 10 ("Implementation Safety Tests") of the EXP-05 research specification, all three proposed architectures—Configuration A (Concatenation Baseline), Configuration B (Gated Feature Fusion), and Configuration C (Cross-Scan Multi-Head Self-Attention)—were subjected to rigorous synthetic batch testing prior to any real training.

The tests used strictly synthetic dummy tensors (`batch_size=8`, `scans=3`, `height=112`, `width=112`, `channels=3`) generated in `scratch/exp05_synthetic/`. Zero real development or quarantined cohort data was loaded.

All 15 core safety tests plus architecture-specific checks for Config B gating and Config C multi-head attention passed with zero failures, zero infinite/NaN values, zero weight drift on the frozen DenseNet-121 backbone, zero batch normalization drift, and exact prediction reproduction upon checkpoint reload (`max_diff = 0.00e+00`).

---

## 2. Safety Test Matrix & Verification Results

| Test # | Verification Criterion | Config A Result | Config B Result | Config C Result | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Model constructs successfully via Keras 3 Functional API | PASS | PASS | PASS | **PASS** |
| **2** | Input tensor shape matches 3-scan pipeline `(None, 3, 112, 112, 3)` | PASS | PASS | PASS | **PASS** |
| **3** | Accepts exactly 3 scans per participant; outputs `clf: (B, 4)`, `emb: (B, 128)` | PASS | PASS | PASS | **PASS** |
| **4** | Output class probabilities have shape `(B, 4)` and are strictly finite | PASS | PASS | PASS | **PASS** |
| **5** | Output class probabilities sum to 1.0 per participant ($tol = 10^{-5}$) | PASS | PASS | PASS | **PASS** |
| **6** | Classification loss, contrastive loss, and total loss are finite | PASS (46.1877) | PASS (25.0008) | PASS (4.9874) | **PASS** |
| **7** | All trainable gradients are finite (no NaN, Inf, or null gradients) | PASS | PASS | PASS | **PASS** |
| **8** | Intended fusion parameters receive active, nonzero gradients | PASS (N/A) | PASS ($\|\nabla\| > 0$) | PASS ($\|\nabla\| > 0$) | **PASS** |
| **9** | DenseNet-121 backbone parameters strictly unchanged after optimizer step | PASS ($\Delta = 0.0$) | PASS ($\Delta = 0.0$) | PASS ($\Delta = 0.0$) | **PASS** |
| **10** | Batch Normalization moving statistics strictly unchanged ($\mu, \sigma^2$) | PASS ($\Delta = 0.0$) | PASS ($\Delta = 0.0$) | PASS ($\Delta = 0.0$) | **PASS** |
| **11** | No trainable variable is omitted from optimizer updates | PASS | PASS | PASS | **PASS** |
| **12** | No variable is updated twice accidentally; partitions strictly disjoint | PASS (6 vars) | PASS (10 vars) | PASS (24 vars) | **PASS** |
| **13** | Checkpoints successfully saved to `.keras` and restored | PASS | PASS | PASS | **PASS** |
| **14** | Restored model reproduces predictions bit-for-bit ($\Delta = 0.00\times 10^0$) | PASS | PASS | PASS | **PASS** |
| **15** | Canonical class ordering mapping verified `[A+, AB+, B+, O+]` | PASS | PASS | PASS | **PASS** |

---

## 3. Configuration B (Gated Feature Fusion) Detailed Verification

The gated fusion mechanism was tested to ensure compliance with Section 6 requirements:
1. **Gate Weight Bounds:** For all synthetic participants, $\alpha_1, \alpha_2, \alpha_3 \in (0, 1)$. Minimum observed weight was $0.0016$, strictly non-negative.
2. **Partition of Unity:** For every sample $i \in \{1, \dots, 8\}$, $\sum_{k=1}^3 \alpha_{i, k} = 1.0 \pm 10^{-6}$.
   - Example observed participant gate distribution: `[0.50950, 0.48390, 0.00660]` (sum = 1.00000).
3. **Fused Representation Shape:** Fused feature tensor shape is strictly `(8, 1024)`.
4. **Independent Computation:** The gate network operates strictly on the 1024-d scan feature embeddings without accessing participant labels or external metrics.
5. **Gradient Propagation:** Gradients successfully flowed through the softmax normalization back into the `gate_dense1` and `gate_logit` projection weights.

---

## 4. Configuration C (Cross-Scan Self-Attention) Detailed Verification

The cross-scan multi-head self-attention module was tested for architectural integrity:
1. **Attention Sequence Tensor Shape:** Three 1024-d scan embeddings projected to 256-d and stacked into `(batch_size, 3, 256)`.
2. **Multi-Head Self-Attention:** Executed 4 attention heads ($d_k = 64$) with residual addition and Layer Normalization.
3. **Feed-Forward Network (FFN):** Position-wise FFN ($256 \to 512 \to 256$) with GELU activation, residual connection, and Layer Normalization.
4. **Sequence Aggregation:** Flattening the attended $3 \times 256$ representation produces a fused vector of strictly `(batch_size, 768)`.
5. **Trainable Variable Count:** Exactly 18 attention/neck variables (projection, MHA query/key/value/output kernels and biases, LN1 gamma/beta, FFN kernels and biases, LN2 gamma/beta). All 18 received non-zero finite gradients.
6. **Isolation:** Attention operation is strictly intra-participant (batch dimension preserved across sequence length of 3 scans); zero cross-participant token mingling.

---

## 5. Parameter Count Ledger

| Configuration | Total Parameters | Frozen Backbone Parameters | Trainable Parameters | Trainable Variables |
| :--- | :---: | :---: | :---: | :---: |
| **Config A (Concatenation)** | 9,006,276 | 7,037,504 | **1,968,772** | 6 |
| **Config B (Gated Fusion)** | 7,826,885 | 7,037,504 | **789,381** | 10 |
| **Config C (Self-Attention)** | 8,321,220 | 7,037,504 | **1,283,716** | 24 |

All parameter counts match theoretical architectural derivations exactly.

---

## 6. Conclusion

Phase 4 synthetic safety testing is **100% COMPLETE and PASSED**. The codebase contains no NaN instabilities, no gradient leaks into the frozen backbone, no BatchNorm corruption, and complete serialization fidelity.
