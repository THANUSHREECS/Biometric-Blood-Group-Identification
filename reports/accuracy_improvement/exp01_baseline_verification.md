# EXP-01 Phase 1: Baseline Verification Report

**Experiment:** EXP-01 — Class-Balanced Focal Loss for E2  
**Date:** 2026-10-14  
**Authoritative Cohort:** 80 Development Participants (240 Scans)  
**Partitioning:** 64 TRAIN, 16 VALIDATION from `reports/phase3_participant_benchmark/participant_split.csv`  
**Evaluation Protocol:** 5-Fold Stratified Group Cross-Validation Out-Of-Fold (OOF)  
**Verification Status:** **100% REPRODUCIBLE & VERIFIED**  

---

## 1. Original E2 Checkpoint Hashes

All five original E2 checkpoints were verified before training:

| Checkpoint | File Path | Size (Bytes) | SHA-256 Hash |
|:---|:---|:---:|:---|
| **Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | 52,910,112 | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` |
| **Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | 52,910,112 | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` |
| **Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | 52,910,112 | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` |
| **Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | 52,910,112 | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` |
| **Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | 52,910,112 | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` |

---

## 2. Independent Recalculation vs. Historical Claims

All metrics were recalculated directly from the participant-level out-of-fold prediction array in `reports/model_fusion/all_models_participant_probs.npz`:

| Metric | Historical Claim | Independently Recalculated | Verification Status |
|:---|:---:|:---:|:---:|
| **Participants Evaluated** | 80 | 80 | **MATCH** |
| **Correct Predictions** | 43 / 80 | 43 / 80 | **MATCH** |
| **OOF Accuracy** | **53.75%** | **53.7500%** | **EXACT MATCH** |
| **OOF Macro F1** | **0.4772** | **0.4772** | **EXACT MATCH** |
| **OOF Balanced Accuracy** | **47.15%** | **47.1540%** | **EXACT MATCH** |
| **Macro Precision** | 0.6628 | 0.6628 | **EXACT MATCH** |
| **Macro Recall** | 0.4715 | 0.4715 | **EXACT MATCH** |
| **Always-O+ Baseline** | 36.25% (29/80) | 36.2500% (29/80) | **EXACT MATCH** |

---

## 3. Per-Class Performance Profile (Canonical Order: A+, AB+, B+, O+)

| Class | Support | Correct | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A+** | 19 | 12 | 0.3750 (12/32) | **63.16%** (12/19) | 0.4706 |
| **AB+** | 7 | 1 | 1.0000 (1/1) | **14.29%** (1/7) | 0.2500 |
| **B+** | 25 | 14 | 0.6364 (14/22) | **56.00%** (14/25) | 0.5957 |
| **O+** | 29 | 16 | 0.6400 (16/25) | **55.17%** (16/29) | 0.5926 |
| **Total / Macro** | **80** | **43** | **0.6628** | **0.4715** | **0.4772** |

### Confusion Matrix (Rows = True Class, Columns = Predicted Class)
*Canonical ordering: `[0: A+, 1: AB+, 2: B+, 3: O+]`*

$$\begin{pmatrix}
12 & 0 & 3 & 4 \\
2 & 1 & 2 & 2 \\
8 & 0 & 14 & 3 \\
10 & 0 & 3 & 16
\end{pmatrix}$$

---

## 4. Key Baseline Observations & Target Areas for EXP-01

1. **Extreme AB+ Vulnerability:** Recall is only **14.29%** ($1/7$ correct). 6 out of 7 AB+ participants were misclassified (2 as A+, 2 as B+, 2 as O+).
2. **False Positives in A+:** 20 false positives were assigned to A+ (10 from O+, 8 from B+, 2 from AB+), driving precision down to 37.5%.
3. **Success Criteria for EXP-01:**
   - Primary: Increase Macro F1 by at least $+0.05$ (target: $\ge 0.5272$).
   - Secondary: Increase AB+ recall from $14.29\%$ to $\ge 30\%$, and B+ recall from $56.0\%$ to $\ge 60\%$.
   - Constraint: Maintain overall accuracy above $45.0\%$.

---

## 5. Phase 1 Conclusion

The baseline is **100% verified and reproducible**. The development cohort is strictly isolated from the quarantined 21-participant locked test set and the separate 26-participant cohort.

We are authorized to proceed to **Phase 2 through Phase 7** in an isolated experimental directory.
