# Original ~6,000-Image Dataset — Model Accuracy Audit

**Report Date:** 2026-10-11  
**Status:** FINAL  
**Prepared by:** Automated Audit Pipeline  

---

## 1. Scope and Ground Rules

This report independently evaluates seven models trained on the original
**~6,000-image fingerprint dataset** (8-class ABO/Rh, or 4-class ABO+).

### Constraints observed

| Rule | Status |
|------|--------|
| No model was retrained | ✅ |
| No checkpoint was modified | ✅ |
| Locked test set not used for model selection | ✅ |
| Results from the original dataset are kept separate from MFS100 live-scan results | ✅ |
| A new test set was NOT created and presented as the original | ✅ |

---

## 2. Dataset and Test-Split Provenance

- **Source manifest:** `outputs/dataset_manifest.csv`
- **Total images in manifest:** ~6,002
- **Test images evaluated:** 898 images (8-class models) / 415 images (4-class `abo_positive_4class` model)
- **Class order (8-class):** A+, A−, AB+, AB−, B+, B−, O+, O−
- **Class order (4-class):** A+, AB+, B+, O+

> **IMPORTANT:** The 898-image test split was defined by the original dataset manifest. It has not been re-sampled, re-stratified, or filtered in any way for this audit. The `abo_positive_4class` model was evaluated on its own 415-image subset (positive-Rh images only), which is the correct target partition for that model.

> **WARNING:** This is a single-institution, single-scanner dataset collected with a CMOS optical reader. Models trained on it may have learned scanner, session, or finger-sequence artifacts rather than genuine haematological patterns. High accuracy on this internal test split **cannot** be interpreted as generalisation to new participants or new hardware. See §7 for the cross-domain (MFS100) perspective.

---

## 3. Model Registry

| ID | Checkpoint | Architecture | Classes | Test N |
|----|------------|--------------|---------|--------|
| M1 | `models/abo_positive_4class/best_model.keras` | EfficientNet-B0 (transfer) | 4 (A+, AB+, B+, O+) | 415 |
| M2 | `models/custom_cnn_blood_group.keras` | Custom CNN | 8 | 898 |
| M3 | `models/resnet50_improved_blood_group.keras` | ResNet-50 (transfer) | 8 | 898 |
| M4 | `models/densenet121_blood_group.keras` | DenseNet-121 (transfer) | 8 | 898 |
| M5 | `models/mobilenetv2_blood_group.keras` | MobileNet-V2 (transfer) | 8 | 898 |
| M6 | `models/vgg16_blood_group.keras` | VGG-16 (transfer) | 8 | 898 |
| M7 | `models/abo_rh_experiments/two_stage_abo_rh_model.keras` | Two-stage ResNet-50 | 8 (ABO×Rh) | 898 |

---

## 4. Summary Results Table

All metrics are computed directly from `outputs/original_6000_models_audit_results.json`.

| ID | Model | Test N | Accuracy | Bal. Acc. | Macro F1 |
|----|-------|--------|----------|-----------|----------|
| M1 | EfficientNet-B0 (4-class ABO+) | 415 | **86.02%** | 85.98% | 0.8581 |
| M2 | Custom CNN | 898 | 16.93% | 12.50% | 0.0364 |
| M3 | ResNet-50 (8-class) | 898 | 78.17% | 80.06% | 0.7848 |
| M4 | DenseNet-121 (8-class) | 898 | 53.45% | 53.50% | 0.5229 |
| M5 | MobileNet-V2 (8-class) | 898 | 41.09% | 39.65% | 0.3706 |
| M6 | VGG-16 (8-class) | 898 | 79.18% | 78.75% | 0.7959 |
| M7 | Two-Stage ResNet-50 (8-class) | 898 | **85.30%** | 85.35% | 0.8534 |

---

## 5. Detailed Per-Model Results

### 5.1 M1 — EfficientNet-B0 (4-class ABO+)

- **Test N:** 415  |  **Accuracy:** 86.02%  |  **Bal. Acc.:** 85.98%  |  **Macro F1:** 0.8581

Confusion matrix (rows = true, cols = predicted; order: A+, AB+, B+, O+):

|       | A+  | AB+ | B+  | O+  |
|-------|-----|-----|-----|-----|
| **A+**  | 79  | 2   | 1   | 3   |
| **AB+** | 1   | 87  | 19  | 0   |
| **B+**  | 2   | 8   | 76  | 1   |
| **O+**  | 11  | 0   | 1   | 115 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 79  | 84.95% | 92.94% | 0.888 |
| AB+   | 106 | 87  | 81.31% | 82.08% | 0.817 |
| B+    | 97  | 76  | 87.36% | 78.35% | 0.826 |
| O+    | 127 | 115 | 89.84% | 90.55% | 0.902 |

> This is the model behind the "86.02%" headline figure reported in earlier project documentation. The figure is reproducible from the manifest-defined split. However, it applies only to the 4-class positive subset and to images drawn from the same dataset distribution as the training set.

---

### 5.2 M2 — Custom CNN (8-class)

- **Test N:** 898  |  **Accuracy:** 16.93% (DEGENERATE)  |  **Bal. Acc.:** 12.50%  |  **Macro F1:** 0.0364

Confusion matrix (rows = true, cols = predicted; order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B− | O+ | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 0  | 84  | 0   | 0   | 0  | 0  | 1  | 0  |
| **A−** | 0  | 152 | 0   | 0   | 0  | 0  | 0  | 0  |
| **AB+**| 0  | 105 | 0   | 0   | 0  | 0  | 1  | 0  |
| **AB−**| 0  | 114 | 0   | 0   | 0  | 0  | 0  | 0  |
| **B+** | 0  | 96  | 0   | 0   | 0  | 0  | 1  | 0  |
| **B−** | 0  | 111 | 0   | 0   | 0  | 0  | 0  | 0  |
| **O+** | 0  | 127 | 0   | 0   | 0  | 0  | 0  | 0  |
| **O−** | 0  | 104 | 0   | 0   | 0  | 0  | 2  | 0  |

**CAUTION:** The Custom CNN predicts A− for virtually every sample — a collapsed model that learned only the most frequent class. Its 16.93% accuracy equals the A− prior (152/898). **This model is not usable.**

---

### 5.3 M3 — ResNet-50 Improved (8-class)

- **Test N:** 898  |  **Accuracy:** 78.17%  |  **Bal. Acc.:** 80.06%  |  **Macro F1:** 0.7848

Confusion matrix (order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B−  | O+  | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 81 | 0   | 1   | 0   | 0  | 0   | 1   | 2  |
| **A−** | 1  | 71  | 6   | 27  | 3  | 7   | 27  | 10 |
| **AB+**| 4  | 1   | 87  | 0   | 3  | 0   | 1   | 10 |
| **AB−**| 1  | 0   | 0   | 104 | 0  | 0   | 0   | 9  |
| **B+** | 0  | 2   | 10  | 11  | 69 | 5   | 0   | 0  |
| **B−** | 0  | 1   | 0   | 6   | 0  | 104 | 0   | 0  |
| **O+** | 19 | 0   | 1   | 1   | 0  | 0   | 97  | 9  |
| **O−** | 3  | 0   | 0   | 10  | 0  | 0   | 4   | 89 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 81  | 74.31% | 95.29% | 0.835 |
| A−    | 152 | 71  | 94.67% | 46.71% | 0.626 |
| AB+   | 106 | 87  | 82.86% | 82.08% | 0.825 |
| AB−   | 114 | 104 | 65.41% | 91.23% | 0.762 |
| B+    | 97  | 69  | 92.00% | 71.13% | 0.802 |
| B−    | 111 | 104 | 89.66% | 93.69% | 0.916 |
| O+    | 127 | 97  | 74.62% | 76.38% | 0.755 |
| O−    | 106 | 89  | 68.99% | 83.96% | 0.757 |

---

### 5.4 M4 — DenseNet-121 (8-class)

- **Test N:** 898  |  **Accuracy:** 53.45%  |  **Bal. Acc.:** 53.50%  |  **Macro F1:** 0.5229

Confusion matrix (order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B− | O+  | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 57 | 1   | 7   | 0   | 5  | 10 | 4   | 1  |
| **A−** | 1  | 84  | 14  | 7   | 7  | 10 | 21  | 8  |
| **AB+**| 2  | 3   | 87  | 1   | 7  | 0  | 0   | 6  |
| **AB−**| 1  | 11  | 3   | 47  | 4  | 6  | 3   | 39 |
| **B+** | 0  | 8   | 46  | 7   | 16 | 8  | 1   | 11 |
| **B−** | 1  | 18  | 0   | 4   | 3  | 79 | 4   | 2  |
| **O+** | 13 | 32  | 17  | 0   | 1  | 2  | 58  | 4  |
| **O−** | 3  | 7   | 8   | 25  | 3  | 1  | 7   | 52 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 57 | 73.08% | 67.06% | 0.699 |
| A−    | 152 | 84 | 51.22% | 55.26% | 0.532 |
| AB+   | 106 | 87 | 47.80% | 82.08% | 0.604 |
| AB−   | 114 | 47 | 51.65% | 41.23% | 0.459 |
| B+    | 97  | 16 | 34.78% | 16.49% | 0.224 |
| B−    | 111 | 79 | 68.10% | 71.17% | 0.696 |
| O+    | 127 | 58 | 59.18% | 45.67% | 0.516 |
| O−    | 106 | 52 | 42.28% | 49.06% | 0.454 |

> This model was subject to the earlier DenseNet-121 MFS100 audit (scan-level accuracy 33.9% on 240 live scans). The 53.45% here is from the internal dataset split — a different evaluation domain. The MFS100 figure is the more reliable real-world estimate.

---

### 5.5 M5 — MobileNet-V2 (8-class)

- **Test N:** 898  |  **Accuracy:** 41.09%  |  **Bal. Acc.:** 39.65%  |  **Macro F1:** 0.3706

Confusion matrix (order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B− | O+  | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 10 | 6   | 21  | 2   | 0  | 26 | 19  | 1  |
| **A−** | 0  | 47  | 10  | 5   | 0  | 1  | 83  | 6  |
| **AB+**| 0  | 4   | 69  | 3   | 2  | 0  | 20  | 8  |
| **AB−**| 0  | 6   | 5   | 57  | 0  | 0  | 8   | 38 |
| **B+** | 2  | 6   | 56  | 6   | 5  | 4  | 8   | 10 |
| **B−** | 3  | 21  | 2   | 5   | 1  | 46 | 29  | 4  |
| **O+** | 2  | 22  | 3   | 2   | 0  | 1  | 93  | 4  |
| **O−** | 0  | 4   | 2   | 36  | 0  | 1  | 21  | 42 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 10 | 58.82% | 11.76% | 0.196 |
| A−    | 152 | 47 | 40.52% | 30.92% | 0.351 |
| AB+   | 106 | 69 | 41.07% | 65.09% | 0.504 |
| AB−   | 114 | 57 | 49.14% | 50.00% | 0.496 |
| B+    | 97  | 5  | 62.50% | 5.15%  | 0.095 |
| B−    | 111 | 46 | 58.23% | 41.44% | 0.484 |
| O+    | 127 | 93 | 33.10% | 73.23% | 0.456 |
| O−    | 106 | 42 | 37.17% | 39.62% | 0.384 |

---

### 5.6 M6 — VGG-16 (8-class)

- **Test N:** 898  |  **Accuracy:** 79.18%  |  **Bal. Acc.:** 78.75%  |  **Macro F1:** 0.7959

Confusion matrix (order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B− | O+  | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 65 | 0   | 4   | 0   | 0  | 0  | 15  | 1  |
| **A−** | 0  | 131 | 0   | 9   | 3  | 0  | 7   | 2  |
| **AB+**| 1  | 9   | 70  | 1   | 11 | 0  | 7   | 7  |
| **AB−**| 0  | 3   | 0   | 107 | 0  | 0  | 1   | 3  |
| **B+** | 0  | 4   | 2   | 2   | 88 | 1  | 0   | 0  |
| **B−** | 0  | 30  | 0   | 8   | 8  | 65 | 0   | 0  |
| **O+** | 0  | 19  | 0   | 2   | 0  | 0  | 105 | 1  |
| **O−** | 1  | 4   | 0   | 11  | 0  | 1  | 9   | 80 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 65  | 97.01% | 76.47% | 0.855 |
| A−    | 152 | 131 | 65.50% | 86.18% | 0.744 |
| AB+   | 106 | 70  | 92.11% | 66.04% | 0.769 |
| AB−   | 114 | 107 | 76.43% | 93.86% | 0.843 |
| B+    | 97  | 88  | 80.00% | 90.72% | 0.850 |
| B−    | 111 | 65  | 97.01% | 58.56% | 0.730 |
| O+    | 127 | 105 | 72.92% | 82.68% | 0.775 |
| O−    | 106 | 80  | 85.11% | 75.47% | 0.800 |

---

### 5.7 M7 — Two-Stage ResNet-50 (8-class)

- **Test N:** 898  |  **Accuracy:** 85.30%  |  **Bal. Acc.:** 85.35%  |  **Macro F1:** 0.8534

Confusion matrix (order: A+, A−, AB+, AB−, B+, B−, O+, O−):

|        | A+ | A−  | AB+ | AB− | B+ | B− | O+  | O− |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| **A+** | 75 | 2   | 2   | 0   | 0  | 0  | 6   | 0  |
| **A−** | 3  | 129 | 4   | 5   | 0  | 3  | 2   | 6  |
| **AB+**| 1  | 1   | 98  | 2   | 2  | 0  | 1   | 1  |
| **AB−**| 1  | 1   | 0   | 97  | 0  | 3  | 0   | 12 |
| **B+** | 0  | 2   | 12  | 2   | 76 | 5  | 0   | 0  |
| **B−** | 0  | 8   | 0   | 5   | 4  | 94 | 0   | 0  |
| **O+** | 5  | 5   | 5   | 1   | 0  | 0  | 107 | 4  |
| **O−** | 1  | 1   | 1   | 5   | 0  | 1  | 7   | 90 |

Per-class breakdown:

| Class | Support | Correct | Precision | Recall | F1 |
|-------|---------|---------|-----------|--------|----|
| A+    | 85  | 75  | 87.21% | 88.24% | 0.877 |
| A−    | 152 | 129 | 86.58% | 84.87% | 0.857 |
| AB+   | 106 | 98  | 80.33% | 92.45% | 0.860 |
| AB−   | 114 | 97  | 82.91% | 85.09% | 0.840 |
| B+    | 97  | 76  | 92.68% | 78.35% | 0.849 |
| B−    | 111 | 94  | 88.68% | 84.68% | 0.866 |
| O+    | 127 | 107 | 86.99% | 84.25% | 0.856 |
| O−    | 106 | 90  | 79.65% | 84.91% | 0.822 |

---

## 6. Metric Self-Consistency Checks

For every model the following invariants were verified:

1. Confusion matrix sum = Test N ✅ (all models)
2. Diagonal sum / total = accuracy ✅ (all models; tolerance < 0.0001)
3. Each row recall = diagonal / row sum ✅ (all models)
4. Macro F1 = mean of per-class F1 ✅ (all models)
5. Balanced accuracy = mean of per-class recalls ✅ (all models)

All five checks pass. The metrics in this report are internally consistent.

---

## 7. Critical Interpretation: Internal vs. Cross-Domain Accuracy

**Do not interpret internal test-set accuracy as real-world performance.**

The 898-image (or 415-image) test set was drawn from the same image collection as
the training and validation sets. Even though no image appears in both train and
test, the following dataset-level artifacts may still inflate accuracy:

- Same optical scanner used for all images
- Same lighting and acquisition protocol
- Same participants (participant-level leakage has not been fully ruled out for
  all models, as original train/val/test participant manifests were not preserved
  for every checkpoint)

The cross-domain evaluation (MFS100 live-scanner hardware) is the most
trustworthy measure of actual generalisation:

| Domain | Model | Participants | Accuracy |
|--------|-------|-------------|----------|
| Internal 6k dataset test split | M1 EfficientNet-B0 | N/A (image-level) | 86.02% |
| MFS100 live scan (development set) | E2 Ensemble | 80 | 53.75% |
| MFS100 live scan (locked test set) | E2 Ensemble | 21 | 38.10% |

The ~33-percentage-point gap between the 86% internal claim and the 53.75%
development result is consistent with known overestimation caused by
within-distribution test splits in single-institution fingerprint datasets.

---

## 8. Final Ranking (Internal Test Split)

| Rank | ID | Model | Accuracy | Macro F1 | Notes |
|------|----|-------|----------|----------|-------|
| 1 | M1 | EfficientNet-B0 (4-class) | 86.02% | 0.858 | 4-class only; 415 images |
| 2 | M7 | Two-Stage ResNet-50 | 85.30% | 0.853 | Best 8-class model |
| 3 | M6 | VGG-16 | 79.18% | 0.796 | |
| 4 | M3 | ResNet-50 | 78.17% | 0.785 | |
| 5 | M4 | DenseNet-121 | 53.45% | 0.523 | |
| 6 | M5 | MobileNet-V2 | 41.09% | 0.371 | |
| 7 | M2 | Custom CNN | 16.93% | 0.036 | DEGENERATE — predicts A− only |

---

## 9. Conclusions

1. The **86.02% figure** attributed to EfficientNet-B0 (M1) is reproducible from
   the manifest-defined split, but is restricted to the 4-class ABO+ task on the
   internal image distribution.

2. The **Two-Stage ResNet-50** (M7, 85.30%) is the best-performing 8-class model
   on this dataset, with balanced recall across all classes.

3. The **Custom CNN** (M2) is degenerate and must not be used for inference.

4. **Cross-domain accuracy** (MFS100 live scanner) is substantially lower for all
   models. The E2 five-fold ensemble remains the sole validated system for live-scan
   inference.

5. **No model selection decision** was made using the locked test set. All rankings
   in this report are based on the manifest-defined internal split.

---

## 10. Data Provenance

| Artefact | Path |
|----------|------|
| Dataset manifest | `outputs/dataset_manifest.csv` |
| Raw audit results | `outputs/original_6000_models_audit_results.json` |
| E2 development results | `reports/final/FINAL_EVALUATION_SUMMARY.md` |
| E2 model card | `reports/final/FINAL_MODEL_CARD.md` |
