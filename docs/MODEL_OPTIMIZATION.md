# Model Optimization Strategy

## Overview
This document records all hyperparameter decisions, fine-tuning strategies, and augmentation
choices for the blood group classification model comparison experiment.

## Architecture Selection Rationale

### Why These Architectures?
| Architecture | Rationale |
|---|---|
| **MobileNetV2** | Fewest parameters, fastest CPU training, supported on small-scale tasks |
| **EfficientNetB0** | State-of-art accuracy/parameter tradeoff, good generalization |
| **Custom CNN** | Baseline reference; trained on native 96Ã—96 resolution to avoid upsampling |
| **DenseNet121** | Feature reuse via dense connections; often effective on small datasets |
| **ResNet50 (improved)** | Same as baseline but with more epochs + deeper fine-tuning |
| **VGG16/19** | Larger models; included for completeness but likely CPU-limited |

### Priority Order (CPU-aware)
1. MobileNetV2 (fastest)
2. EfficientNetB0 (best accuracy/speed ratio)
3. Custom CNN (quick, no download needed)
4. DenseNet121
5. ResNet50 (improved)
6. VGG16
7. VGG19 (only if VGG16 looks promising)

---

## Preprocessing Design

### Central Preprocessing Pipeline
```
Image file (BMP/PNG/JPEG/RGBA)
  â†“ load_image_to_pil()
  â†“ composite_alpha_on_white()   # RGBA â†’ RGB on white background
  â†“ resize_to_target()           # Lanczos resampling
  â†“ model-specific preprocess_input()
```

### Architecture-Specific Preprocessing
| Model | Function | Description |
|---|---|---|
| custom_cnn | `x / 127.5 - 1.0` | [-1, 1] normalization |
| mobilenetv2 | `mobilenet_v2.preprocess_input` | [-1, 1] normalization |
| efficientnetb0 | `efficientnet.preprocess_input` | [0, 255] (no-op in newer TF) |
| densenet121 | `densenet.preprocess_input` | [0, 1] then ImageNet mean subtraction |
| resnet50 | `resnet50.preprocess_input` | BGR + ImageNet mean subtraction |
| vgg16 | `vgg16.preprocess_input` | BGR + ImageNet mean subtraction |
| vgg19 | `vgg19.preprocess_input` | BGR + ImageNet mean subtraction |

**Critical**: Each architecture uses its OWN `preprocess_input`. Cross-contamination
(e.g., using ResNet50 preprocessing for VGG16) would produce incorrect inputs.

---

## Data Augmentation

### Policy: Conservative (Fingerprint-Safe)
Fingerprint ridge structure must not be destroyed. Only mild spatial and
photometric augmentations are used.

| Augmentation | Value | Justification |
|---|---|---|
| RandomRotation | Â±3% (â‰ˆÂ±10.8Â°) | Minor scanner angle variation |
| RandomTranslation | 3% H + 3% W | Minor finger placement offset |
| RandomZoom | 3% | Minor scanner zoom variation |
| RandomBrightness | Â±5% | Scanner illumination variation |
| RandomContrast | Â±5% | Scanner sensitivity variation |
| Horizontal flip | **NO** | No scientific justification for fingerprints |
| Vertical flip | **NO** | No scientific justification for fingerprints |
| Heavy rotation | **NO** | Would destroy ridge direction information |
| Cutout/Dropout | **NO** | Removes critical ridge detail |

**Validation and test data are NEVER augmented.**

---

## Class Weighting Strategy

### Strategy B: Mild Dampened Balanced Weights (Default)
Formula: `weight = 0.6 * (balanced_weight - 1.0) + 1.0`

Class imbalance ratio is mild (max 1.78Ã— for A- vs A+), so mild dampening
prevents gradient spikes while still addressing imbalance.

| Class | Train Count | Balanced Weight | Mild Weight |
|---|---|---|---|
| A+  | 395 | 1.78 | 1.47 |
| A-  | 705 | 1.00 | 1.00 |
| AB+ | 493 | 1.43 | 1.26 |
| AB- | 532 | 1.33 | 1.20 |
| B+  | 454 | 1.55 | 1.33 |
| B-  | 519 | 1.36 | 1.22 |
| O+  | 596 | 1.18 | 1.11 |
| O-  | 495 | 1.42 | 1.25 |

*Note: A- is the most frequent class (weight=1.0). A+ is least frequent (highest weight).*

### Why NOT extreme class weights?
- A- has 705 training samples (most frequent) but 15% test recall
- This means the poor A- recall is NOT from data scarcity
- Extreme weights for minority classes would not fix the A- problem
- Root cause: insufficient training epochs, not class imbalance

---

## Fine-Tuning Strategy

### Two-Stage Protocol
**Stage 1**: Freeze backbone entirely. Train classification head only.
- High LR (1e-3): backbone weights safe from gradient interference
- 20 epochs max with EarlyStopping(patience=7)

**Stage 2**: Unfreeze upper backbone layers. Fine-tune carefully.
- Low LR (2e-5): prevents catastrophic forgetting of ImageNet features
- 20 epochs max with EarlyStopping(patience=7)
- BatchNormalization layers remain FROZEN (running statistics must be stable)

### Architecture-Specific Fine-Tuning Depth

| Architecture | Stage 2: Layers Unfrozen | Rationale |
|---|---|---|
| MobileNetV2 | Last 30 layers | Small dataset â†’ avoid overfitting |
| EfficientNetB0 | Last 50 layers | Compound scaling â†’ more relevant high-level features |
| DenseNet121 | Last dense block (~50 layers) | Dense connections need careful unfreezing |
| ResNet50 | conv4 + conv5 blocks | Improved over baseline (baseline only used conv5) |
| VGG16/19 | block4 + block5 | VGG benefits from unfreezing more blocks |

### Callbacks Configuration
```python
EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
ModelCheckpoint(monitor="val_accuracy", save_best_only=True)
ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
```

---

## Baseline Comparison

### Why the baseline failed:
1. **8 total epochs** â€” severe underfitting
2. **Only conv5 fine-tuned** â€” insufficient backbone adaptation
3. **Stage 2 only 3 epochs at 1e-5** â€” not enough time to improve
4. **EarlyStopping not used in Stage 1** â€” head not properly converged

### Improvements in this framework:
- 20 epochs per stage (with EarlyStopping patience=7)
- Deeper fine-tuning (conv4+conv5 for ResNet50)
- Lower Stage 2 LR (2e-5 vs 1e-5) for gentler adaptation
- ReduceLROnPlateau during both stages
- Per-run experiment logging

---

## Ensemble Optimization Strategy

To push model performance beyond individual model limits without overfitting, a multi-model soft-voting ensemble was implemented:
- **Constituent Models**:
  - `ResNet50 (Improved)` (deep residual feature extraction)
  - `VGG16` (deep sequential convolutional representations, high A- recall)
  - `EfficientNetB0` (compound-scaled depthwise separable features, fast & regularized)
- **Probability Fusion**: Soft-voting with weights $(w_1, w_2, w_3)$ on softmax outputs:
  $$P_{ensemble}(y = c \mid x) = \sum_{i=1}^3 w_i \cdot P_{model_i}(y = c \mid x), \quad \sum w_i = 1$$
- **Optimization Method**: Grid search over the 3D weight simplex (step = 0.05) evaluated exclusively on the validation set.
- **Optimal Weight Distribution**:
  - ResNet50 (Improved): $0.50$
  - VGG16: $0.35$
  - EfficientNetB0: $0.15$
- **Validation Outcome**:
  - Baseline Val Acc: 60.02%
  - Best Single Model (ResNet50 Improved): 79.73%
  - **Ensemble Val Acc: 85.19%** (+25.17% over baseline, +5.46% over best single model)
  - **Ensemble Val Macro F1: 85.47%** (+28.56% over baseline, +5.54% over best single model)

---

## Scientific Disclaimer
This project investigates whether fingerprint ridge patterns correlate with blood group
labels in this specific dataset. The relationship is not scientifically established.
All results are research-level and must not be used for clinical diagnosis without
laboratory confirmation.
