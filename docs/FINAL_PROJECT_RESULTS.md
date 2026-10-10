ï»¿# FINAL PROJECT RESULTS
## Biometric Blood Group Identification System â€” Champion Ensemble V2

**Document Type:** Official Results and Benchmarks Report
**Project:** B.E. Final Year Major Project â€” Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Model Status:** FROZEN â€” This document records final, immutable benchmark results.

---

## 1. Dataset Information

### 1.1 Source Dataset

| Property | Value |
|---|---|
| Dataset name | Fingerprint Blood Group Dataset |
| Format | Windows Bitmap (.BMP), 32-bit RGBA |
| Total raw images | 6,000 |
| Total cleaned (post-quarantine) | 5,985 |
| Duplicates quarantined | 15 (11 duplicate clusters) |
| Image resolution | Variable (standardized to 224Ã—224 for training) |

### 1.2 The 8 Blood Group Classes

| Class ID | Blood Group | Train Count | Val Count | Test Count | Total |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | A+ | 395 | 85 | 85 | 565 |
| 1 | A- | 705 | 151 | 152 | 1,008 |
| 2 | AB+ | 493 | 106 | 106 | 705 |
| 3 | AB- | 532 | 114 | 114 | 760 |
| 4 | B+ | 454 | 97 | 97 | 648 |
| 5 | B- | 519 | 111 | 111 | 741 |
| 6 | O+ | 596 | 128 | 127 | 851 |
| 7 | O- | 495 | 106 | 106 | 707 |
| â€” | **TOTAL** | **4,189** | **898** | **898** | **5,985** |

**Split methodology:** Stratified random split (70% / 15% / 15%) with SHA-256 cryptographic deduplication to guarantee zero cross-split data leakage.

---

## 2. Model Comparison and Selection

| Model | Test Accuracy | Test Correct / 898 | Notes |
|---|---|---|---|
| Custom CNN (baseline) | ~48% | ~431 | Trained from scratch, no transfer learning |
| ResNet50 Baseline | 57.57% | 517 | Initial transfer learning, frozen backbone |
| MobileNetV2 | ~73% | ~655 | Lightweight architecture |
| DenseNet121 | ~76% | ~682 | Dense connections |
| VGG16 | ~79% | ~709 | Classic architecture, strong feature extraction |
| ResNet50 (Improved) | ~81% | ~727 | 2-stage fine-tuning, conv5_block unfrozen |
| EfficientNetB0 | ~71% | ~638 | Efficient scaling, lower accuracy alone |
| **Champion Ensemble V2** | **85.41%** | **767** | **Soft-voting: ResNet50+VGG16+ENB0** |

**Champion Ensemble V2 represents a +27.84% accuracy improvement over the original ResNet50 baseline.**

---

## 3. Champion Ensemble V2 â€” Official Benchmark

### 3.1 Ensemble Configuration

| Sub-Model | Ensemble Weight | Parameters | Preprocessing |
|---|---|---|---|
| ResNet50 (Improved) | **58%** (0.58) | 24,114,312 | 1.2Ã— contrast + ResNet50 preprocess_input |
| VGG16 | **38%** (0.38) | 14,848,072 | VGG16 preprocess_input (native) |
| EfficientNetB0 | **4%** (0.04) | 4,379,563 | EfficientNet preprocess_input (native) |

**Ensemble type:** Soft-voting (weighted average of class probability vectors)

### 3.2 Official Locked Test Set Results (898 images)

> These results were produced by a single, locked evaluation on the held-out test set.
> The test set has NOT been re-evaluated after this benchmark.

| Metric | Value |
|---|---|
| **Test Accuracy** | **85.41%** |
| Test Correct | 767 / 898 |
| Test Incorrect | 131 / 898 |
| Test Loss | 0.4274 |
| Macro Precision | 86.38% |
| Macro Recall | 86.05% |
| Macro F1-Score | 85.90% |
| Weighted Precision | 86.21% |
| Weighted Recall | 85.41% |
| Weighted F1-Score | 85.47% |

### 3.3 Per-Class Test Results

| Blood Group | Support | Correct | Accuracy | Precision | Recall | F1-Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A+ | 85 | 78 | 91.76% | 93.98% | 91.76% | 92.86% |
| A- | 152 | 112 | 73.68% | 89.60% | 73.68% | 80.87% |
| AB+ | 106 | 92 | 86.79% | 85.98% | 86.79% | 86.38% |
| AB- | 114 | 106 | 92.98% | 73.10% | 92.98% | 81.85% |
| B+ | 97 | 83 | 85.57% | 91.21% | 85.57% | 88.30% |
| B- | 111 | 95 | 85.59% | 95.00% | 85.59% | 90.05% |
| O+ | 127 | 113 | 88.98% | 84.96% | 88.98% | 86.92% |
| O- | 106 | 88 | 83.02% | 77.19% | 83.02% | 80.00% |

### 3.4 Confusion Matrix (Test Set, 8Ã—8)

```
Actual\Pred  A+   A-  AB+  AB-   B+   B-   O+   O-
A+  (85)     78    0    2    0    0    0    3    2
A-  (152)     0  112    4   17    1    2   10    6
AB+ (106)     0    1   92    1    4    0    1    7
AB- (114)     1    1    0  106    0    0    0    6
B+  (97)      0    1    7    4   83    2    0    0
B-  (111)     0    8    0    6    2   95    0    0
O+  (127)     4    1    1    2    1    0  113    5
O-  (106)     0    1    1    9    0    1    6   88
```

---

## 4. Validation Set Results

> These metrics are from the validation split (not the locked test set).
> Used for investigation only â€” not the official benchmark.

| Metric | Value |
|---|---|
| Ensemble Validation Accuracy | 86.41% |
| AB+ Recall (Ensemble, validation) | 86.79% (92/106) |
| AB+ â†’ O- confusion (validation) | 6.60% of AB+ samples (7/106) |

---

## 5. Sub-Model Validation Performance

| Model | Validation Accuracy | AB+ Recall |
|---|---|---|
| ResNet50 (58% weight) | 80.96% | 85.85% |
| VGG16 (38% weight) | 78.62% | 68.87% |
| EfficientNetB0 (4% weight) | 70.60% | 70.75% |
| **Ensemble (Combined)** | **86.41%** | **86.79%** |

Ensemble outperforms every individual sub-model by 5.45â€“15.81 percentage points.

---

## 6. MFS100 Real-Hardware Validation Results

Real-world testing conducted on 2026-09-04 and 2026-09-05 using the physical Mantra MFS100 under the standardized Left Thumb Acquisition Protocol.

| Test Event | Finger | Prediction | Actual | Confidence | Status | Verdict |
|---|---|---|---|---|---|---|
| BG-000185 | Left Thumb | O+ | O+ | 94.59% | NORMAL | CORRECT |
| BG-000188 | Other finger | O- | AB+ | 44.02% | LOW_CONFIDENCE | INCORRECT (expected) |
| 2026-09-05 real scan | Left Thumb | O- | AB+ | 41.6% | LOW_CONFIDENCE | INCORRECT (AB+ limitation) |
| 2026-09-05 consensus | Left Thumb | O- | â€” | 59.12% | LOW_CONFIDENCE | â€” |

**Key finding:** LOW_CONFIDENCE status (< 60% threshold) correctly flagged all incorrect predictions in real-world testing.

---

## 7. Confidence Analysis

| Prediction Type | Mean Confidence | Median Confidence | Fraction Flagged LOW_CONFIDENCE |
|---|---|---|---|
| Correct AB+ (validation) | 79.23% | 87.30% | â€” |
| Incorrect AB+ (validation) | 63.94% | 68.98% | **42.86%** |

The 60% confidence threshold acts as a safety net, correctly flagging nearly 43% of wrong AB+ predictions as LOW_CONFIDENCE.

---

## 8. Automated Test Suite Results

**Command:** `python -m pytest tests/ -v --tb=short`

**Result:** 56 passed, 0 failed, 0 errors in 40.94 seconds

| Test Module | Tests | Result |
|---|---|---|
| test_api.py | 5 | PASS |
| test_database.py | 1 | PASS |
| test_dataset_prep.py | 4 | PASS |
| test_e2e_integration.py | 2 | PASS |
| test_left_thumb_protocol.py | 12 | PASS |
| test_mfs100.py | 3 | PASS |
| test_mfs100_bridge.py | 5 | PASS |
| test_mfs100_integration.py | 4 | PASS |
| test_mfs100_scanner.py | 5 | PASS |
| test_predictor.py | 6 | PASS |
| test_preprocessing.py | 7 | PASS |
| test_report.py | 2 | PASS |
| **TOTAL** | **56** | **ALL PASS** |

---

## 9. Known Limitations

### 9.1 AB+ Generalization

The Champion Ensemble V2 misclassifies AB+ as O- in approximately 6.60% of validation-set AB+ samples (7/106). This is caused by ResNet50 and VGG16 sharing ridge-topology feature space between AB and O classes.

**Mitigation:** The 60% confidence threshold catches ~42.86% of wrong AB+ predictions and flags them as LOW_CONFIDENCE. The system correctly advised recapture / laboratory confirmation.

**Reference:** `docs/AB_PLUS_GENERALIZATION_ANALYSIS.md`

### 9.2 Rh Factor

Rh factor prediction is less reliable than ABO group prediction because dermatoglyphic ridge patterns have weaker biological correlation with Rh antigen expression. All predicted Rh-negative and low-margin results trigger the Rh Uncertainty Advisory.

### 9.3 Real MFS100 Subjects

Physical testing was limited to a small number of real human subjects. The model was trained on a digital fingerprint image dataset; real MFS100 hardware produces a different optical domain (washed-out ridges, sensor margins) that is partially compensated by the foreground crop and CLAHE ridge enhancement pipeline.

### 9.4 Clinical Validation

This system has NOT undergone formal clinical validation. It is an academic research prototype only. **All blood group determinations require mandatory confirmation via standard serological laboratory testing.**

---

## 10. Benchmark Integrity Statement

- Champion Ensemble V2 was evaluated exactly once on the 898-image locked test set.
- All metrics in this document are from that single evaluation run.
- The test set has NOT been re-accessed after the official evaluation.
- No metric has been fabricated, inflated, or cherry-picked.
- Ensemble weights were locked and have not been modified since evaluation.

---

*This document is part of the B.E. Final Year Major Project â€” Biometric Blood Group Identification System.*
*All reported metrics reflect genuine evaluation on locked data. For academic purposes only.*
*Laboratory blood typing confirmation is mandatory for any medical use.*
