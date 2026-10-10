ï»¿# AB+ Generalization Analysis â€” Champion Ensemble V2

**Document Type:** Research Prototype â€” Model Generalization Investigation
**Project:** Biometric Blood Group Identification System
**Degree:** B.E. Final Year Major Project â€” Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Status:** CLOSED â€” Strategy Selected: Option A (Accept current performance with documented limitation)

---

## 1. Background and Motivation

During real-hardware MFS100 testing conducted on 2026-09-05 under the standardized Left Thumb Acquisition Protocol, a subject with confirmed blood group **AB+** received the prediction **O-** (confidence: 41.6%), which was correctly flagged as **LOW CONFIDENCE** by the system threshold.

This triggered a formal validation-only investigation to:
1. Determine whether this failure was an isolated hardware/preprocessing artefact or a true model generalization weakness.
2. Quantify AB+ recall and confusion patterns on the full validation split (n=898 images).
3. Determine the primary responsible sub-model within Champion Ensemble V2.
4. Recommend a forward strategy suitable for an academic prototype.

> **IMPORTANT:** The locked 898-image test set was NOT evaluated during this investigation. All metrics reported here are from the **validation split** only.

---

## 2. Investigation Scope and Constraints

| Constraint | Status |
|---|---|
| Locked test set (898 images) accessed | NOT accessed |
| Champion Ensemble V2 weights modified | NOT modified |
| Heuristic correction rules added | NOT added |
| LEFT THUMB treated as a prediction rule | NOT implemented |
| Fabricated or inflated metrics | NOT done |
| Validation split used for analysis | YES |
| Root-cause diagnosis on real MFS100 case | YES |

Investigation script: scripts/investigate_ab_plus_generalization.py
Output artefacts: outputs/ab_plus_diagnostics/

---

## 3. Dataset Composition

### 3.1 Training Split

| Metric | Value |
|---|---|
| Total images | 4,189 |
| AB+ images | 493 |
| AB+ percentage | 11.77% |

| Class | Count |
|---|---|
| A+ | 395 |
| A- | 705 |
| AB+ | 493 |
| AB- | 532 |
| B+ | 454 |
| B- | 519 |
| O+ | 596 |
| O- | 495 |

### 3.2 Validation Split

| Metric | Value |
|---|---|
| Total images | 898 |
| AB+ images | 106 |
| AB+ percentage | 11.80% |

Finding: AB+ is well-represented in both splits at ~11.8%. Data imbalance is NOT the root cause.

---

## 4. Validation Performance â€” Champion Ensemble V2

### 4.1 Overall Accuracy

| Sub-Model | Validation Accuracy |
|---|---|
| ResNet50 (58% weight) | 80.96% |
| VGG16 (38% weight) | 78.62% |
| EfficientNetB0 (4% weight) | 70.60% |
| Ensemble (Combined) | 86.41% |

### 4.2 AB+ Specific Metrics

| Metric | Value |
|---|---|
| Total AB+ samples | 106 |
| Correctly classified | 92 |
| AB+ Recall (Ensemble) | 86.79% |
| Misclassified as O- | 7 (6.60%) |
| Misclassified as O+ | 2 (1.89%) |
| Misclassified as A+ | 3 (2.83%) |
| Misclassified as B+ | 2 (1.89%) |

### 4.3 AB+ Confusion Matrix (Validation)

| Predicted | A+ | A- | AB+ | AB- | B+ | B- | O+ | O- |
|---|---|---|---|---|---|---|---|---|
| Actual AB+ | 3 | 0 | 92 | 0 | 2 | 0 | 2 | 7 |

Primary confusion: AB+ predicted as O- (7 cases, 6.60%).

### 4.4 Per-Model AB+ Recall

| Sub-Model | AB+ Recall |
|---|---|
| ResNet50 (weight: 58%) | 85.85% |
| VGG16 (weight: 38%) | 68.87% |
| EfficientNetB0 (weight: 4%) | 70.75% |
| Ensemble (Combined) | 86.79% |

Key Observation: VGG16 has the lowest AB+ recall (68.87%) yet carries 38% ensemble weight.
EfficientNetB0 shows strongest AB-sensitivity despite its low 4% weight.

---

## 5. Confidence Analysis

| Prediction Type | Mean Confidence | Median Confidence |
|---|---|---|
| Correct AB+ predictions | 79.23% | 87.30% |
| Incorrect AB+ predictions | 63.94% | 68.98% |
| Fraction of wrong AB+ flagged LOW CONFIDENCE | 42.86% |

Finding: 43% of AB+ misclassifications were caught by the 60% confidence threshold.
The real-world case (41.6% confidence) was correctly flagged as LOW CONFIDENCE.

---

## 6. Root Cause Analysis

### 6.1 MFS100 Hardware / Preprocessing (RULED OUT)

Forensic diagnostic confirmed (see docs/AB_POSITIVE_MFS100_CASE_DIAGNOSTIC.md):
- MFS100 raw image quality: NFIQ 1, quality score 88/100 (excellent).
- No blur, saturation, or sensor artefact detected.
- Preprocessing pipeline produced a correct, clean input tensor.
- The same AB+ to O- failure pattern occurs in the validation dataset at 6.60% rate.

Conclusion: Preprocessing and hardware quality are NOT the root cause.

### 6.2 Sub-Model Analysis

- ResNet50 (58%) occasionally maps ambiguous AB+ ridge textures to O- feature space.
- VGG16 (38%) has poor AB+ recall (68.87%), contributing bias toward AB+ to O- failure.
- EfficientNetB0 (4%) is the strongest AB-discriminating model but its low weight limits impact.

### 6.3 Summary

The AB+ to O- confusion is an intrinsic model generalization issue arising from ResNet50/VGG16
feature representations, which share ridge-topology features across AB and O classes.
This is a known limitation of dermatoglyphic blood group classification.

---

## 7. Strategy Evaluation

### Option A â€” Accept Current Performance (SELECTED)
No model change. Document the limitation formally.
- 86.79% AB+ recall is academically defensible for a research prototype.
- Confidence threshold safety net flags ~43% of wrong AB+ predictions.
- Modifying the model without locked test set re-evaluation is academically unsound.
- Risk: LOW

### Option B â€” Reweight Ensemble Toward EfficientNetB0
Increase EfficientNetB0 weight from 4% to ~15-20%.
- Requires new ensemble weight search and full validation re-run.
- Cannot be validated without accessing the locked test set.
- Risk: MEDIUM. NOT recommended for this prototype phase.

### Option C â€” AB+ Targeted Training Experiment
Fine-tune with class-weighted cross-entropy loss emphasising AB+.
- Any metric improvement cannot be verified without a fresh held-out evaluation set.
- Risk: MEDIUM. NOT recommended without a fresh held-out set.

### Option D â€” Collect Real MFS100 AB+ Scans
Collect 50-100 real AB+ MFS100 scans and augment training set.
- Risk: LOW for data collection. HIGH long-term value.
- RECOMMENDED as future work.

---

## 8. Selected Strategy â€” Option A

Decision: Accept current Champion Ensemble V2 performance with formally documented limitation.

The system will:
1. Continue using Champion Ensemble V2 without weight changes.
2. Continue applying the 60% confidence threshold as a safety filter.
3. Formally document the AB+ to O- confusion pattern (this document).
4. Present the AB+ limitation transparently in the final project report and viva.
5. Reference Option D (real MFS100 AB+ data collection) as recommended future work.

---

## 9. Impact on Left Thumb Protocol

The AB+ misclassification does NOT invalidate the Left Thumb Acquisition Protocol:
- The Left Thumb setting is acquisition metadata only â€” it does not influence the prediction.
- The low-confidence flag (41.6%) was correctly raised, directing toward laboratory confirmation.
- The error is caused by model generalization, not by finger selection.
- The protocol is retained as-is per docs/MFS100_ACQUISITION_PROTOCOL.md.

---

## 10. Recommended Future Work

1. Increased EfficientNetB0 ensemble weight for AB-class predictions, validated against a fresh held-out set.
2. Collection of real MFS100 fingerprint data from AB+ donors for domain-specific training.
3. Ensemble architecture with class-conditional routing (dedicated AB-vs-O second-stage classifier).
4. Explainability (Grad-CAM) visualization of AB+ failure cases.

---

## 11. Outputs

| File | Description |
|---|---|
| outputs/ab_plus_diagnostics/ab_plus_confusion_summary.json | Full validation confusion and per-model recall metrics |
| outputs/ab_plus_diagnostics/ab_plus_sample_predictions.csv | Per-sample AB+ predictions with confidence values |
| outputs/ab_plus_diagnostics/confusion_matrix_val.csv | Full 8-class confusion matrix on validation split |
| docs/AB_POSITIVE_MFS100_CASE_DIAGNOSTIC.md | Forensic analysis of real-world MFS100 AB+ failure |

---

## 12. References

- docs/AB_POSITIVE_MFS100_CASE_DIAGNOSTIC.md
- docs/MFS100_ACQUISITION_PROTOCOL.md
- docs/FINAL_MODEL_RESULTS.md
- outputs/ensemble_weights.json
- scripts/investigate_ab_plus_generalization.py

---

*This document is part of the B.E. Final Year Major Project â€” Biometric Blood Group Identification System.*
*Strategy selected on 2026-09-05 under validation-only constraints. For academic purposes only.*
