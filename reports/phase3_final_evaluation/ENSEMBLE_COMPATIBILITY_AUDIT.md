# ENSEMBLE COMPATIBILITY AUDIT

## 1. INSPECT THE CHECKPOINTS
- **File Exists and Loads:** PASS. All 5 `.weights.h5` files exist and loaded successfully into the memory graph.
- **Architecture Identical:** PASS. All 5 folds mapped identically to the `DenseNet121_Fusion_SupCon` graph.
- **Feature Extraction Identical:** PASS. Frozen DenseNet121 ImageNet base.
- **Feature Dimension Identical:** PASS. `(3072,)` concatenated features.
- **Class Order Identical:** PASS. Exact order: `["A+", "B+", "AB+", "O+"]` across all folds.
- **Preprocessing Identical:** PASS. C5 routine strictly applied.
- **Three-Scan Concatenation Order:** PASS. `scan1, scan2, scan3` order preserved natively in the graph.
- **Classification Head Structure:** PASS. Output shape `(None, 4)`.
- **Required Model Components:** PASS.
- **Test Data Contamination:** PASS. No test data was processed; StratifiedGroupKFold on Dev-80 strictly preserved.

## 2. REVIEW DEVELOPMENT RESULTS ONLY
- **Fold 1:** Val Macro F1: 0.2902, Val Acc: 0.4375, Pts: 16. Recalls -> A+: 0.25, B+: 0.20, AB+: 0.00, O+: 0.83
- **Fold 2:** Val Macro F1: 0.4167, Val Acc: 0.5000, Pts: 16. Recalls -> A+: 0.80, B+: 0.50, AB+: 0.00, O+: 0.38
- **Fold 3:** Val Macro F1: 0.3818, Val Acc: 0.5000, Pts: 16. Recalls -> A+: 1.00, B+: 0.29, AB+: 0.00, O+: 0.67
- **Fold 4:** Val Macro F1: 0.6253, Val Acc: 0.5625, Pts: 16. Recalls -> A+: 0.25, B+: 0.75, AB+: 1.00, O+: 0.57
- **Fold 5:** Val Macro F1: 0.4059, Val Acc: 0.6875, Pts: 16. Recalls -> A+: 1.00, B+: 1.00, AB+: 0.00, O+: 0.00

## 3. ENSEMBLE COMPATIBILITY
All 5 checkpoints perfectly share identical computational graphs, identical preprocessing paths, identical output layer scaling (softmax), and exactly matching class indices. **They are mathematically and practically compatible for a flat ensemble.**

## 4. DEFINE THE ENSEMBLE RULE
The ensemble uses a fixed, unweighted probability average across all 5 folds:
`P_ensemble(i, c) = (1/5) * sum(P_k(i, c))`
The final prediction is `argmax_c P_ensemble(i, c)`.

## OVERALL AUDIT STATUS
**PASS.** The ensemble protocol is valid, technically sound, and successfully frozen.
