# EXP-04: Evaluation Protocol & Metric Standards

**Audit Date:** 2026-10-14T18:22:00+05:30  
**Scope:** Specification of Prediction Unit, Probability Aggregation, Class Mapping, Metric Definitions, and Statistical Evaluation  
**Audit Finding:** **PASS — PROTOCOL IDENTICAL TO EXP-03 STANDARDS**  

---

## 1. Executive Summary

To maintain strict comparability with EXP-03, baseline E2, and previous benchmark evaluations, EXP-04 adheres to the project's standardized participant-level evaluation protocol. No modifications to metric formulas, class mappings, or aggregation methods are introduced.

---

## 2. Frozen Experiment Protocol Checklist

The following 9 core experimental dimensions are strictly frozen and immutable across EXP-04:

| # | Protocol Component | Specification | Invariance Enforcement |
|:---|:---|:---|:---:|
| **1** | **Development Folds** | 5 participant-grouped folds (64 train / 16 val per fold) via `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` | **FROZEN** |
| **2** | **Preprocessing** | `transform_c5` (112x112, $[0.0, 255.0]$), identity pass-through (no augmentation) | **FROZEN** |
| **3** | **Canonical Class Order** | Index 0: A+, Index 1: AB+, Index 2: B+, Index 3: O+ | **FROZEN** |
| **4** | **Loss Functions** | Sparse categorical crossentropy (wt 1.0) + Supervised Contrastive Loss ($\tau=0.07$, wt 0.1) | **FROZEN** |
| **5** | **Sample Weighting** | Inverse-class frequency formula: $w_i = N_{\text{train}} / (4 N_c)$ | **FROZEN** |
| **6** | **Batch Normalization** | All BatchNorm layers locked in evaluation mode (`layer.trainable = False`), moving statistics frozen | **FROZEN** |
| **7** | **Checkpoint Selection** | Monitored on validation Macro-F1 (`MacroF1Checkpoint`) on held-out fold data | **FROZEN** |
| **8** | **Epoch Boundaries** | `EPOCHS = 150`, `MIN_EPOCHS = 100` floor, `PATIENCE = 15` early stopping | **FROZEN** |
| **9** | **Cohort Quarantine** | 21-participant locked test set and 26-participant hospital cohort strictly quarantined | **LOCKED & EXCLUDED** |

---

## 3. Unit of Prediction & Probability Aggregation

1. **Prediction Unit:** The evaluation unit is the **individual participant**, not individual fingerprint scans.
2. **Input Tensor:** For each participant, all three preprocessed fingerprint impressions ($s_1, s_2, s_3$) are stacked into a 5D batch tensor of shape `(batch_size, 3, 112, 112, 3)`.
3. **Simultaneous Processing:** The three scans are passed simultaneously through the three towers of the feature fusion model, yielding three 1024-dimensional feature vectors that are concatenated into a 3072-dimensional representation.
4. **Single Inference Output:** The classification head outputs a single probability vector per participant.
5. **Out-of-Fold Generation:** In each fold, predictions are generated strictly on the 16 held-out validation participants using the model loaded from `best_epoch`. Aggregating across the five folds produces exactly 80 out-of-fold predictions per configuration.

---

## 3. Class Ordering & Remapping Protocol

The model's native training indices follow the historical provenance of `experiment6_fusion_supcon.py`, while reporting follows the canonical project standard:

| Blood Group | Native Model Index | Canonical Reporting Index | Probability Mapping Formula |
|:---:|:---:|:---:|:---:|
| **A+** | Index 0 | Index 0 | $p_{\text{canonical}}[0] = p_{\text{native}}[0]$ |
| **AB+** | Index 2 | Index 1 | $p_{\text{canonical}}[1] = p_{\text{native}}[2]$ |
| **B+** | Index 1 | Index 2 | $p_{\text{canonical}}[2] = p_{\text{native}}[1]$ |
| **O+** | Index 3 | Index 3 | $p_{\text{canonical}}[3] = p_{\text{native}}[3]$ |

### Vector Remapping Implementation:
```python
p_canon = np.array([p_nat[0], p_nat[2], p_nat[1], p_nat[3]], dtype=np.float64)
p_canon = p_canon / p_canon.sum()  # Enforce exact unity sum
pred_idx = int(np.argmax(p_canon))
pred_label = CANONICAL_CLASSES[pred_idx]
```

---

## 4. Metric Definitions (Scikit-Learn Standards)

All metrics are computed on the 80 true out-of-fold predictions using scikit-learn:
- **Overall Accuracy:** $\frac{\sum_{i=1}^{80} \mathbb{I}(y_i = \hat{y}_i)}{80}$
- **Balanced Accuracy:** $\frac{1}{4} \sum_{c=1}^4 \text{Recall}_c$ (macro-average recall)
- **Macro Precision:** $\frac{1}{4} \sum_{c=1}^4 \text{Precision}_c$ (unweighted mean)
- **Macro F1-Score:** $\frac{1}{4} \sum_{c=1}^4 \text{F1}_c$ (unweighted mean)
- **Per-Class Metrics:** Precision, Recall, and F1 computed individually for A+, AB+, B+, and O+.
- **Confusion Matrix:** $4 \times 4$ contingency table indexed by `[A+, AB+, B+, O+]`.

---

## 5. Statistical Rigor Protocol

Given the small participant cohort ($N = 80$ total; $N = 7$ for AB+), numerical improvements must be qualified with appropriate statistical testing:
1. **Paired McNemar's Exact Binomial Test:** Evaluates discordant classification pairs between the fine-tuning candidate (Config B) and the reference model (Config A):
   $$p = 2 \sum_{k=0}^{\min(b, c)} \binom{b+c}{k} 0.5^{b+c}$$
   where $b$ is the number of subjects correct in Config B but wrong in Config A, and $c$ is the reverse.
2. **Fisher's Exact Test:** Evaluates whether differences in minority class recall (e.g. AB+ recall) reach statistical significance given $N=7$.
3. **Reporting Requirement:** If $p > 0.05$, the results will be reported as an affirmative development trend, explicitly acknowledging that statistical significance was not reached.

---

## 6. Locked Test Quarantine Mandate

- Under no circumstances will any EXP-04 configuration be evaluated on the 21-participant locked test set or the 26-participant hospital cohort during development.
- Model selection between Config A, Config B, and Config C will be determined **solely on the 80 development participants' out-of-fold cross-validation performance**.

---

## 7. Audit Verdict

Evaluation protocols, remapping functions, metric formulas, and statistical testing standards are verified and locked.
- **Audit Verdict:** **PASS**.
