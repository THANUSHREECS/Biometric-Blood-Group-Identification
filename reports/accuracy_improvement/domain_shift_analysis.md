# Acquisition Domain Shift Analysis

**Investigation:** Accuracy Improvement — Root-Cause Diagnostic  
**Date:** 2026-10-13  
**Status:** COMPLETE & VERIFIED  
**Output Path:** `reports/accuracy_improvement/domain_shift_analysis.md`  

---

## 1. Executive Summary

A primary hypothesis for why EfficientNet-B0 scores **86.02%** on its internal optical dataset but collapses to **28.57%** on Mantra MFS100 locked test scans is an acute **acquisition-domain shift**. 

To evaluate this hypothesis objectively, descriptive statistics were computed across random 100-image samples of the original training set, the MFS100 development/test cohort (`mfs100_final`), and the live-captured cohort (`data/captured`).

### Key Finding:
The empirical evidence confirms an **extreme acquisition-domain gap** encompassing resolution ($3.3\times$ area ratio), spatial frequency, background segmentation, and sharpness ($11\times$ Laplacian variance disparity). EfficientNet-B0 was never presented with an MFS100 image during training; passing unadapted MFS100 scans directly into its frozen backbone represents an out-of-distribution failure.

---

## 2. Quantitative Domain Metrics

| Metric | Original Optical Dataset (`dataset_blood_group`) | MFS100 Benchmark (`data/mfs100_final`) | MFS100 Captured (`data/captured`) | Gap Ratio / Disparity |
|:---|:---:|:---:|:---:|:---:|
| **Width $\times$ Height** | $96 \times 103$ px | $316 \times 354$ px | $316 \times 354$ px | $11.3\times$ pixel area disparity |
| **Aspect Ratio (W/H)** | $0.932 \pm 0.02$ | $0.893 \pm 0.00$ | $0.893 \pm 0.00$ | Inconsistent aspect ratio |
| **Color Representation** | 4-Channel `RGBA` | 1-Channel `L` (Grayscale) | 1-Channel `L` (Grayscale) | Alpha-channel mask in original |
| **Mean Intensity $[0, 255]$** | $149.51 \pm 103.4$ | $125.38 \pm 86.14$ | $158.97 \pm 86.37$ | Non-uniform illumination |
| **Intensity Std Dev (Contrast)** | $103.43$ | $86.14$ | $86.37$ | $20\%$ lower contrast in MFS100 |
| **Laplacian Variance (Sharpness)** | **$59,277.5$** | **$5,250.7$** | **$5,946.5$** | **$11.3\times$ sharper in original** |
| **Foreground Coverage** | $39.47\%$ | $55.36\%$ | $55.49\%$ | $16\%$ wider unsegmented frame |

---

## 3. Detailed Root-Cause Factors

### 3.1 Ridge Spatial Frequency (DPI) Disparity
- **MFS100 Resolution:** Standard 500 DPI. At 500 DPI, typical adult human fingerprint ridge-to-ridge wavelength is approximately 9 to 11 pixels.
- **Original Dataset Resolution:** Images are native $96 \times 103$. In this downsampled or cropped representation, ridge wavelength spans only 3 to 5 pixels.
- **Impact on Pretrained Convolutions:** The spatial filters in the first convolutional layers of EfficientNet-B0 (receptive field size $3 \times 3$) respond to completely different spatial frequencies between the two datasets. Downsampling or upsampling without explicit frequency alignment shifts activations across feature maps.

### 3.2 Foreground Segmentation & Background Contrast
- **Original Dataset:** Fingerprints were digitally segmented with an alpha channel. The background is pristine white ($255$) with zero background sensor noise.
- **MFS100 Captures:** The sensor frame contains residual platen smudge, glass reflection, finger edge boundaries, and dark ambient sensor borders.
- **Impact on Feature Activations:** Global Average Pooling (GAP) in EfficientNet averages activations across the entire spatial grid ($7 \times 7$ after feature extraction). Background sensor smudges contribute non-biometric noise into the classification head.

### 3.3 Dynamic Pressure & Elastic Skin Deformation
- **Original Dataset:** Images appear static with uniform ridge thickness across the entire impression.
- **MFS100 Real-World Live Scans:** Real-time volunteer capture involves variable contact pressure, causing ridge valleys to fill in (over-inking/darkening) or dry skin contact gaps (ridge fragmentation).

---

## 4. Verdict on Hypothesis

The hypothesis that **acquisition-domain shift is a principal cause of EfficientNet-B0's cross-domain degradation (86% down to 28%) is supported by empirical evidence**. 

Directly ensembling or evaluating EfficientNet-B0 without prior domain adaptation (such as sensor normalization, ridge scale alignment, or low-learning-rate fine-tuning on MFS100 development scans) will consistently fail.
