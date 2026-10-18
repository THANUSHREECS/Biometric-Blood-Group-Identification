# EXP-02 Baseline Verification Report

**Experiment:** EXP-02 — Controlled Data Augmentation for E2  
**Date:** 2026-10-14T12:54:31.251308  
**Scope:** Verification of Baseline Architecture, Checkpoints, Class Orders, and Performance  

---

## 1. Verified Baseline Metrics (N=80 Development Participants)
| Metric | Baseline Value | Description |
|:---|:---:|:---|
| **Accuracy** | **53.75%** (43/80) | Overall participant-level classification accuracy |
| **Balanced Accuracy** | **47.15%** | Macro-average recall across all 4 blood groups |
| **Macro F1-Score** | **0.4772** | Unweighted mean of per-class F1-scores |
| **Macro Precision** | **0.6628** | Unweighted mean of per-class precision |
| **Macro Recall** | **0.4715** | Equivalent to balanced accuracy |
| **Always-O+ Baseline** | **36.25%** (29/80) | Trivial majority class baseline |

### Per-Class Baseline Breakdown (Canonical Order: A+, AB+, B+, O+)
- **A+ (N=19):** Precision: 37.50% | Recall: 63.16% (12/19) | F1: 0.4706
- **AB+ (N=7):** Precision: 100.00% | Recall: 14.29% (1/7) | F1: 0.2500
- **B+ (N=25):** Precision: 63.64% | Recall: 56.00% (14/25) | F1: 0.5957
- **O+ (N=29):** Precision: 64.00% | Recall: 55.17% (16/29) | F1: 0.5926

---

## 2. Checkpoint SHA-256 Hashes (Pre-Experiment)
- `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5`: `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d`
- `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5`: `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede`
- `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5`: `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3`
- `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5`: `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519`
- `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5`: `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d`

---

## 3. Class Ordering & Remapping Protocol
- **Native E2 Output Order:** `["A+", "B+", "AB+", "O+"]` (indices 0, 1, 2, 3)
- **Canonical Evaluation Order:** `["A+", "AB+", "B+", "O+"]` (indices 0, 1, 2, 3)
- **Remapping Function:** `p_canonical = [p_native[0], p_native[2], p_native[1], p_native[3]]`
