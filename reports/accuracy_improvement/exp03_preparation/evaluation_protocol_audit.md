# EXP-03 Preparation: Evaluation Protocol Audit Report

**Audit Date:** 2026-10-14T13:20:00
**Objective:** Verify that baseline E2 and EXP-02 evaluation protocols are strictly comparable in participants, splits, prediction aggregation, class ordering, and metric definitions.

---

## 1. Participant Grouping & Leakage Verification

### Participant Cohort:
- Authoritative development split: reports/phase3_participant_benchmark/participant_split.csv
- Total development participants: Exactly 80 participants (64 TRAIN, 16 VALIDATION).
- Scans per participant: Exactly 3 scans (240 scans total).

### 5-Fold Stratified Group K-Fold Cross-Validation:
- Generator: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
- Participant grouping: Scans from the same participant were strictly kept together in the same fold.
- Zero-leakage check: set(train_participants) & set(val_participants) == empty set across all 5 folds.
- **Audit Verdict:** **PASS.** Participant assignment to folds was identical in both original E2 and EXP-02.

---

## 2. Prediction Unit & Aggregation Procedure

### Unit of Prediction:
- In both baseline E2 and EXP-02, the evaluation unit is the **individual participant** (not individual scans).
- The three scans belonging to a participant [s1, s2, s3] are stacked into shape (3, 112, 112, 3) and passed simultaneously through the 3-tower feature fusion network.
- Each participant receives exactly one prediction vector from their respective held-out fold checkpoint.

### Aggregation Across Folds:
- Baseline E2 OOF: 16 validation participants per fold x 5 folds = 80 out-of-fold participant predictions. Stored in reports/model_fusion/all_models_participant_probs.npz and exported in reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv.
- EXP-02 OOF: Exactly the same 16 validation participants per fold x 5 folds = 80 out-of-fold participant predictions. Exported in reports/accuracy_improvement/exp02/e2_augmented_oof_predictions.csv.
- **Audit Verdict:** **PASS.** Prediction units and aggregation procedures are 100% equivalent.

---

## 3. Class Ordering & Remapping Protocol

Both pipelines account for the disparity between native model output indices and canonical reporting indices:

| Class | Native E2 Model Output Index | Canonical Evaluation Index | Remapping Operation |
|:---:|:---:|:---:|:---:|
| **A+** | 0 | 0 | p_canonical[0] = p_native[0] |
| **B+** | 1 | 2 | p_canonical[2] = p_native[1] |
| **AB+** | 2 | 1 | p_canonical[1] = p_native[2] |
| **O+** | 3 | 3 | p_canonical[3] = p_native[3] |

Formula applied in both:
p_canonical = [p_native[0], p_native[2], p_native[1], p_native[3]]
predicted_label = CANONICAL_CLASSES[argmax(p_canonical)]

- Probability sanity check: In both baseline and EXP-02, all probabilities are non-negative, finite, and sum to 1.000000 +/- 1e-5.
- **Audit Verdict:** **PASS.** Class order remapping is verified correct in both pipelines.

---

## 4. Ground-Truth Fidelity & Metric Computation

### Ground-Truth Fidelity:
- All 80 participant labels in e2_augmented_oof_predictions.csv match participant_split.csv and e2_predictions.csv with zero discrepancies.
- True class distribution across the 80 participants:
  - O+: 29 (36.25%)
  - B+: 25 (31.25%)
  - A+: 19 (23.75%)
  - AB+: 7 (8.75%)

### Metric Definitions:
Both pipelines compute metrics using scikit-learn:
- **Overall Accuracy:** accuracy_score(y_true, y_pred)
- **Balanced Accuracy:** balanced_accuracy_score(y_true, y_pred) (macro recall)
- **Macro Precision, Recall, F1:** precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
- **Per-Class Metrics:** precision_recall_fscore_support(y_true, y_pred, labels=['A+', 'AB+', 'B+', 'O+'], average=None, zero_division=0)
- **Confusion Matrix:** confusion_matrix(y_true, y_pred, labels=['A+', 'AB+', 'B+', 'O+'])

- **Audit Verdict:** **PASS.** Evaluation metrics are computed with identical mathematical formulas.

---

## 5. Summary Conclusion

The evaluation protocols of baseline E2 and EXP-02 are **completely comparable and methodologically rigorous**.
The divergence in results was **not** caused by evaluation mismatch, fold leakage, label errors, or remapping bugs.
The divergence was caused entirely during the **training phase** by the severe epoch reduction (EPOCHS=60, MIN_EPOCHS=15 vs EPOCHS=150, MIN_EPOCHS=100).
