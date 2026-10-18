# EXP-03 Preparation: Architectural Recommendation & Next Steps

**Date:** 2026-10-14T13:25:00
**Document Scope:** Synthesis of Findings from Forensic Audit of EXP-02 and Actionable Design for EXP-03
**Status:** PROPOSED EXPERIMENTAL ROADMAP (NO RETRAINING UNDERTAKEN)

---

## 1. Executive Summary & Audit Decision

The forensic code audit revealed that **EXP-02 was heavily confounded**:
- While EXP-02 correctly restored categorical cross-entropy loss and verified input dimensions, it drastically reduced training epochs from 100-150 down to 15-27 epochs.
- Furthermore, the original E2 champion was already trained with aggressive data augmentation (including 10% zoom, which distorts ridge frequency).
- Therefore, the hypothesis that conservative augmentation impairs E2 performance is **unsupported and invalid**. The performance drop was driven by under-training.

---

## 2. Distinction Between Verified Facts and Hypotheses

### A. Verified Facts (Ground Truth in Code & Artifacts)
1. **Training Floor Disparity:**
   - In scripts/experiment6_fusion_supcon.py, MIN_EPOCHS = 100 and EPOCHS = 150. Training was strictly prohibited from stopping before epoch 100.
   - In experiments/exp02_e2_augmentation.py, MIN_EPOCHS = 15 and EPOCHS = 60. Training stopped between epochs 15 and 27 on all folds.
2. **Baseline Augmentation Reality:**
   - The original E2 baseline was not unaugmented. It trained with RandomRotation(0.027), RandomTranslation(0.1, 0.1), RandomZoom(0.1), and RandomContrast(0.1).
3. **Class Weighting & Loss:**
   - Both original E2 and EXP-02 used standard sparse_categorical_crossentropy combined with supervised_contrastive_loss (weight 0.1).
   - Both applied per-fold inverse-frequency sample weights w_i = N / (4 * N_c) via sample_weight in model.fit(). The variable lpha_weights in EXP-02 was syntactically inherited from EXP-01, but its numerical calculation and role in sample_weight were identical to cw in original E2.
4. **Evaluation Equivalence:**
   - Both used the same 80 development participants, same 5-fold participant-grouped splits (zero leakage), same 3-scan input fusion, same canonical remapping, and same scikit-learn metric calculations.
5. **Data Quarantine:**
   - In both experiments, the 21-participant locked test set and the separate 26-participant hospital cohort were 100% quarantined and never accessed.

### B. Scientific Hypotheses
1. **Under-Training Hypothesis:**
   - The drop in accuracy from 53.75% to 47.50% in EXP-02 was caused by premature early stopping at epochs 15-27, preventing the feature fusion dense head from reaching convergence.
2. **Ridge Frequency Invariance Hypothesis:**
   - Removing RandomZoom is theoretically superior for biometric friction ridges because zoom distorts the physical spatial frequency of ridges. When trained for the full 100-150 epochs, conservative augmentation (rotation +/- 7.2 deg, translation +/- 4.5 px, contrast +/- 5%, no zoom) should equal or surpass the baseline without inducing synthetic ridge distortions.

---

## 3. Recommended Design for EXP-03: Multi-Arm Controlled Study

To eliminate confounding factors and isolate the exact causal effect of data augmentation, EXP-03 should be structured as a **controlled 3-arm experiment** using the verified 80 development participants under the identical 100-150 epoch training regime:

### Arm 0: Faithful Reproduction of Original E2 Baseline
- **Purpose:** Verify reproducibility of the reported 53.75% accuracy and 0.4772 macro F1 baseline from scratch.
- **Augmentation:** Original aggressive policy (Rotation 0.027, Translation 0.1, Zoom 0.1, Contrast 0.1).
- **Epochs:** EPOCHS = 150, MIN_EPOCHS = 100, PATIENCE = 15.
- **Loss & Weights:** Sparse CCE + SupCon (0.1), inverse-frequency sample weights.

### Arm 1: Unaugmented Baseline Ablation
- **Purpose:** Establish the true zero-augmentation baseline for E2.
- **Augmentation:** Completely disabled (Identity / no perturbation).
- **Epochs:** EPOCHS = 150, MIN_EPOCHS = 100, PATIENCE = 15.
- **Loss & Weights:** Identical to Arm 0.

### Arm 2: Conservative Fingerprint-Appropriate Augmentation
- **Purpose:** Isolate the true causal effect of fingerprint-appropriate augmentation.
- **Augmentation:** Conservative policy (Rotation +/- 7.2 deg, Translation +/- 4.5 px, Contrast +/- 5%, Brightness +/- 5%, Zoom EXCLUDED).
- **Epochs:** EPOCHS = 150, MIN_EPOCHS = 100, PATIENCE = 15.
- **Loss & Weights:** Identical to Arm 0 and Arm 1.

---

## 4. Acceptance & Decision Criteria for EXP-03

| Metric | Target vs. Faithful Arm 0 Baseline | Target vs. Original Checkpoint Baseline |
|:---|:---:|:---:|
| **Macro F1** | >= +0.03 | >= 0.5072 |
| **Balanced Accuracy** | Improvement (> 0.0%) | > 47.15% |
| **Minority (AB+) Recall** | Maintained or improved | > 14.29% |
| **Integrity Audit** | 100% Pass across all 3 arms | 100% Pass |

---

## 5. Immediate Action Plan

1. Do NOT retrain immediately; present this audit report and recommendation to the user.
2. Obtain confirmation on the 3-arm EXP-03 protocol.
3. Maintain strict quarantine on the locked test set (21) and separate cohort (26).
4. Preserve all original production checkpoints in models/phase3_fusion_metric_learning/E2/.
