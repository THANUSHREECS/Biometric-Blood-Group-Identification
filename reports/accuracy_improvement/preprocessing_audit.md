# Preprocessing Implementation Audit Report

**Investigation:** Accuracy Improvement — Root-Cause Diagnostic  
**Date:** 2026-10-13  
**Status:** COMPLETE & VERIFIED  
**Output Path:** `reports/accuracy_improvement/preprocessing_audit.md`  

---

## 1. Executive Summary

This audit examined the end-to-end preprocessing pipelines across all training scripts, inference modules, and evaluation harnesses to identify discrepancies between training-time and inference-time transformations.

### Key Audit Findings:
1. **EfficientNet-B0 Preprocessing Gap:**  
   During training in `scripts/train_abo_positive_4class.py`, images were alpha-composited and resized using **Lanczos interpolation** to $224 \times 224$ as float32 in $[0, 255]$. In inference, standard **Bilinear interpolation** was applied. More critically, EfficientNet-B0 has never had a domain-appropriate foreground segmentation pipeline for raw MFS100 frames.
2. **E2 C5 Pipeline Integrity:**  
   The E2 `transform_c5` pipeline (`composite_alpha_on_white` $\rightarrow$ `crop_mfs100_foreground` $\rightarrow$ `enhance_mfs100_ridge_contrast` $\rightarrow$ `resize_to_target((112, 112))`) matches the exact training pipeline used in `scripts/experiment6_fusion_supcon.py`. The historical bug in Round 14 (which incorrectly applied `densenet.preprocess_input`) has been confirmed fixed.
3. **Severe Multi-Scan Spatial Downsampling in E2:**  
   E2 forces all MFS100 impressions ($316 \times 354$) down to **$112 \times 112$ pixels**. This extreme downsampling ($89.8\%$ area reduction) obliterates fine-grained minutiae (ridge bifurcations and endings), leaving the network to rely on low-frequency global smudge patterns.

---

## 2. Pipeline Comparison Matrix

| Preprocessing Step | EfficientNet-B0 (Training) | EfficientNet-B0 (MFS100 Inference) | E2 Fusion (`transform_c5`) |
|:---|:---:|:---:|:---:|
| **Input Source** | 4-channel `RGBA` $96 \times 103$ | 1-channel `L` $316 \times 354$ | 1-channel `L` $316 \times 354$ |
| **Alpha Compositing** | `composite_alpha_on_white` | PIL RGB conversion | `composite_alpha_on_white` |
| **Foreground Detection** | None (already masked) | **None (Full sensor frame)** | `crop_mfs100_foreground` (Otsu + BBox) |
| **Contrast Enhancement** | None | None | CLAHE (clip 2.5) + Percentile Stretch |
| **Denoising Filter** | None | None | Bilateral filter ($d=3, \sigma=25$) |
| **Resize Target** | $224 \times 224$ | $224 \times 224$ | **$112 \times 112$** |
| **Interpolation Method** | Lanczos (`Image.LANCZOS`) | Bilinear (`Image.BILINEAR`) | Lanczos (`Image.LANCZOS`) |
| **Value Range** | Float32 in $[0, 255]$ | Float32 in $[0, 255]$ | Float32 in $[0, 255]$ |
| **Input Tensor Shape** | $(1, 224, 224, 3)$ | $(1, 224, 224, 3)$ | $(1, 3, 112, 112, 3)$ |

---

## 3. Discrepancy Analysis

### 3.1 EfficientNet-B0 Ingestion of Uncropped MFS100 Frames
When EfficientNet-B0 is presented with an MFS100 scan:
- It receives the full $316 \times 354$ image without foreground cropping.
- The uncropped image contains extensive black platen borders and ambient glass reflection.
- When resized to $224 \times 224$, the aspect ratio shifts from $0.893$ to $1.000$, causing vertical ridge compression.
- Concurrently, because background pixels constitute ~45% of the frame, ImageNet feature extraction is dominated by border artifacts.

### 3.2 E2 Information Loss from 112x112 Downsampling
- Fingerprint ridge frequency standard at 500 DPI requires at least 8–10 pixels per ridge cycle for reliable minutiae extraction (ridge endings, bifurcations, sweat pores).
- Downsampling a $316 \times 354$ frame to $112 \times 112$ compresses the ridge wavelength to $\le 3$ pixels.
- At 3 pixels per period, Nyquist sampling limits are breached, inducing aliasing and destroying minutiae topology. E2 is essentially forced to act as a coarse macro-pattern classifier rather than a biometric feature analyzer.

---

## 4. Controlled Improvement Recommendations

1. **Exp A (Preprocessing):** Evaluate EfficientNet-B0 with E2's foreground cropping and CLAHE pipeline (`transform_c5` resized to $224 \times 224$ instead of uncropped input) on the 80 development participants.
2. **Exp B (Resolution Scaling):** Evaluate training an E2 variant at native $224 \times 224$ resolution instead of $112 \times 112$ to preserve ridge minutiae frequency.
