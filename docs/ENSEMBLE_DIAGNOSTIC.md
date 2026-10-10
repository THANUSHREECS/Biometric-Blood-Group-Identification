# Ensemble Diagnostic Report

**Generated**: 2026-09-04  
**Scope**: Validation Split Only (898 samples) — Test set strictly locked.

---

## 1. Executive Summary & Diagnostic Findings

### Q1: Which classes are still difficult?
- **`A-` (Rh-negative)** remains the most challenging class for single models, though the ensemble lifted recall to **75.5%** (from baseline 15.2%). On ResNet50 alone, `A-` recall is only **49.7%** (predicting it only 75 times out of 151).
- **`AB+`** recall is **78.3%** in the ensemble, with confusion into other `AB` or `A` subclasses.
- **`O+`** recall is **84.4%**, but precision is lowest at **74.5%**, receiving false positive classifications from other blood groups due to standard ridge feature dominance.

### Q2: Which models complement each other?
- **ResNet50 (Improved)** and **VGG16** exhibit extraordinary architectural complementarity:
  - Error correlation between ResNet50 and VGG16 is exceptionally low: **$r = 0.210$**.
  - **`A-` Recall**: VGG16 dominates at **88.1%**, while ResNet50 achieves only **49.7%**.
  - **`A+` Recall**: ResNet50 dominates at **89.4%**, while VGG16 achieves only **68.2%**.
  - **`B-` Recall**: ResNet50 dominates at **96.4%**, while VGG16 achieves only **72.1%**.
  - **`B+` Recall**: VGG16 dominates at **84.5%**, while ResNet50 achieves **74.2%**.
  - **`O-` Recall**: ResNet50 dominates at **92.5%**, while VGG16 achieves **76.4%**.
- **EfficientNetB0** serves as a regularizer ($r = 0.240$ with ResNet, $0.307$ with VGG16), stabilizing predictions and smoothing out overconfident misclassifications.

### Q3: Which model makes the most errors?
- **DenseNet121**: 378 errors out of 898 (error rate: **42.09%**, val accuracy: 57.91%).
- **EfficientNetB0**: 264 errors out of 898 (error rate: **29.40%**, val accuracy: 70.60%).
- **VGG16**: 192 errors out of 898 (error rate: **21.38%**, val accuracy: 78.62%).
- **ResNet50 (Improved)**: 182 errors out of 898 (error rate: **20.27%**, val accuracy: 79.73%).
- **Ensemble (ResNet + VGG + EffNet)**: **133 errors** (error rate: **14.81%**, val accuracy: 85.19%).
  - The ensemble fixed **77 errors** made by ResNet50 and **97 errors** made by VGG16.

### Q4: Which model is strongest for each blood group?

| Class | Support | Strongest Single Model | Best Single Recall | Ensemble Recall | Net Ensemble Impact |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **A+** | 85 | **ResNet50** | **89.4%** | 84.7% | High balance maintained |
| **A-** | 151 | **VGG16** | **88.1%** | 75.5% | ResNet weakness rescued |
| **AB+** | 106 | **ResNet50 / DenseNet** | **80.2%** | 78.3% | Consistent precision (>90%) |
| **AB-** | 114 | **ResNet50** | **92.1%** | **94.7%** | Ensemble beats best single |
| **B+** | 97 | **VGG16** | **84.5%** | **84.5%** | Matches best single |
| **B-** | 111 | **ResNet50** | **96.4%** | **96.4%** | Matches best single |
| **O+** | 128 | **VGG16** | **77.3%** | **84.4%** | Ensemble beats best single |
| **O-** | 106 | **ResNet50** | **92.5%** | 85.8% | High precision (81.2%) |

### Q5: Whether ensemble errors are correlated?
- Model error correlation matrix:
  ```
                  resnet50  vgg16  efficientnetb0  densenet121
  resnet50           1.000  0.210           0.240        0.187
  vgg16              0.210  1.000           0.307        0.194
  efficientnetb0     0.240  0.307           1.000        0.326
  densenet121        0.187  0.194           0.326        1.000
  ```
- The cross-model correlations are remarkably low ($0.187$ to $0.326$), confirming high diversity. When ResNet50 fails, VGG16 is correct in **79%** of those failure instances.

### Q6: Whether probability calibration is poor?
- **ResNet50**: ECE = $0.0309$, Brier score = $0.2850$. Exceptionally well calibrated.
- **VGG16**: ECE = $0.0655$, Brier score = $0.3166$. Slightly overconfident (mean confidence $84.4\%$ vs $78.6\%$ accuracy).
- **EfficientNetB0**: ECE = $0.0544$, Brier score = $0.3882$. Tends to be conservative (mean confidence $66.4\%$).
- **Ensemble**: Brier score drops significantly to **$0.2399$** (superior probability quality). However, soft probability averaging compresses confidence peaks, increasing ECE to $0.1005$. Temperature scaling ($T \approx 0.7 - 0.8$) can re-sharpen ensemble confidences.

### Q7: Whether class weighting is helping or hurting?
- The current **mild smoothing formula** ($w_{mild} = 0.6 \cdot (w_{balanced} - 1) + 1.0$) was instrumental in lifting `A-` recall from $15\%$ to $50\%$ on ResNet50 and $88\%$ on VGG16 without destroying precision on majority classes.
- Without class weights, gradient updates favor majority classes and collapse minority representations.
- Extreme inverse frequency weighting hurts overall accuracy by producing false positives on rare classes.

### Q8: Whether fine-tuning depth can be improved?
- In pilot training, ResNet50 unfreezing was restricted to `conv4` and `conv5` (last ~33 layers), and VGG16 to `block4` and `block5`.
- Extending fine-tuning to deeper intermediate layers with cosine annealing learning rate schedules can allow more fingerprint-specific spatial filter adaptation.

### Q9: Whether augmentation is too weak or too strong?
- The current conservative augmentation (rotation $\pm 3^\circ$, translation $3\%$, zoom $3\%$) successfully preserved ridge topology.
- However, contrast and ridge clarity variations exist across scanners and pressure conditions. Incorporating mild contrast-limited adaptive histogram equalization (CLAHE) or targeted brightness/contrast jittering will improve generalization without corrupting ridge flow.

### Q10: Whether current input resolution limits performance?
- Raw fingerprint images are $\sim 96 \times 103$ BMP upsampled to $224 \times 224$.
- Bilinear/bicubic interpolation to $224 \times 224$ produces slight smoothing of fine ridge bifurcations and delta points.
- Exploring higher resolutions (e.g. $256 \times 256$ or $288 \times 288$) with Lanczos-4 resampling can preserve high-frequency ridge minutiae.

---

## 2. Strategic Roadmap for Optimization (Phases B through G)
1. **Phase D & C**: Fingerprint-specific preprocessing (CLAHE & contrast enhancement) + input resolution benchmark on validation split.
2. **Phase B**: Train enhanced individual models with optimal preprocessing, deeper fine-tuning, and cosine-decay scheduling.
3. **Phase E & F**: Calibrated soft-voting ensemble search and threshold tuning on validation split only.
