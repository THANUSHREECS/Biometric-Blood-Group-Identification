# Model Comparison Report

**Generated**: 2026-09-06 11:50

> ⚠️ **Test set is LOCKED** — no test accuracy appears in this table.
> Model selection is based exclusively on validation performance.
> Only the final selected model will be evaluated on the test set.

## Comparison Table

| Model | Best Val Accuracy | Macro F1 | Training Time | Parameters | Notes |
|---|---|---|---|---|---|
| **resnet50_baseline** (locked) | 60.02% | 56.91% | — | — | Original 8-epoch baseline (test acc: 57.57%) |
| **resnet50 🏆** | 88.08% | 88.01% | 85.6 min | 24,114,312 | [full] Long‑epoch ResNet50 experiment |
| **efficientnetb0** | 87.42% | 87.58% | 48.4 min | 4,379,563 | [full] Long‑epoch EfficientNetB0 experiment |
| **ensemble_v2** | 86.41% | 86.63% | 2.0 min | 43,341,947 | [Phase F/G Champion] Optimized Ensemble V2 reaches 86.41% val acc, 0.8663 Macro F1 |
| **ensemble_3way** | 85.19% | 85.47% | 2.0 min | 43,341,947 | [Phase F] Initial 3-way soft-voting ensemble reaches 85.19% val acc |
| **ensemble_4way** | 83.96% | 84.15% | 2.5 min | 50,643,907 | [Phase G] Adding DenseNet121 hurts accuracy (-0.56%); excluded from final ensemble |
| **densenet121** | 82.52% | 82.76% | 308.9 min | 7,301,960 | [full] Long‑epoch DenseNet121 experiment |
| **vgg16** | 78.62% | 78.81% | 82.8 min | 14,848,072 | [pilot] best_epoch=5 |
| **mobilenetv2** | 43.10% | 38.62% | 7.5 min | 2,587,976 | [pilot] best_epoch=6 |
| **custom_cnn** | 16.82% | 3.62% | 2.0 min | 457,704 | [pilot] best_epoch=2 |

🏆 **Recommended Model**: `resnet50`
  - Best Val Accuracy: 79.73%
  - Val Macro F1: 79.93%

## Confusion Analysis (Validation Set)

See generated confusion matrix images in `outputs/`:

- `outputs/resnet50_confusion_matrix.png`
- `outputs/efficientnetb0_confusion_matrix.png`
- `outputs/ensemble_v2_confusion_matrix.png`
- `outputs/ensemble_3way_confusion_matrix.png`
- `outputs/ensemble_4way_confusion_matrix.png`
- `outputs/densenet121_confusion_matrix.png`
- `outputs/vgg16_confusion_matrix.png`
- `outputs/mobilenetv2_confusion_matrix.png`
- `outputs/custom_cnn_confusion_matrix.png`

## Per-Class Analysis

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| A+ | 0.768 | 0.894 | 0.826 | 85 |
| A- | 0.974 | 0.497 | 0.658 | 151 |
| AB+ | 0.833 | 0.802 | 0.817 | 106 |
| AB- | 0.755 | 0.921 | 0.830 | 114 |
| B+ | 0.923 | 0.742 | 0.823 | 97 |
| B- | 0.836 | 0.964 | 0.895 | 111 |
| O+ | 0.700 | 0.766 | 0.731 | 128 |
| O- | 0.726 | 0.925 | 0.813 | 106 |

## Scientific Note

> This project is a research/prototype system investigating whether fingerprint images can be used to classify blood group labels. Results are not medically validated.
> **All predictions must be confirmed using standard laboratory blood typing before medical use.**