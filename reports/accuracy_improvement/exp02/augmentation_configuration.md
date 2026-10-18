# EXP-02: Controlled Augmentation Specification

**Experiment:** EXP-02 — Controlled Data Augmentation for E2  
**Date:** 2026-10-14T12:54:31.253159  
**Target Architecture:** E2 3-Scan Feature Fusion (Frozen DenseNet-121 Backbone)  

---

## 1. Biometric Fingerprint Domain Constraints
Fingerprint friction ridges are fine, high-frequency spatial structures with strict biometric constraints:
1. **Ridge-Valley Spatial Frequency:** Inter-ridge distance is anatomically constrained. Indiscriminate scaling (`RandomZoom`) alters spatial frequency, synthesizing non-anatomical ridge patterns.
2. **Platen Contact Dynamics:** When placing a finger on the Mantra MFS100 platen, realistic physical variations are limited to:
   - Small ergonomic rotation angles (+/- 5° to 8°).
   - Slight translational displacement (+/- 2 to 5 mm).
   - Minor LED illumination and skin moisture changes (+/- 5% contrast and brightness).

---

## 2. Transformation Pipeline Specification
The following augmentation pipeline is applied exclusively during training (`training=True`) and disabled during inference/validation (`training=False`):

| Transformation | Keras Layer | Parameters | Physical & Biometric Justification |
|:---|:---|:---|:---|
| **Rotation** | `RandomRotation` | `factor=(-0.02, 0.02)` (+/- 7.2°) | Simulates rotational variance upon touching the platen without severely rotating ridge flow. |
| **Translation** | `RandomTranslation` | `height_factor=(-0.04, 0.04)`, `width_factor=(-0.04, 0.04)` | Simulates +/- 4.5 pixel shifts on 112x112 image, keeping core/delta landmarks within frame. |
| **Contrast** | `RandomContrast` | `factor=0.05` (+/- 5%) | Accommodates sensor illumination drift without eroding CLAHE ridge separation. |
| **Brightness** | `RandomBrightness` | `factor=0.05`, `value_range=(0.0, 255.0)` | Simulates skin moisture / platen reflectance variation. |
| **Zoom** | *EXCLUDED* | None (`0.0`) | **Excluded** to prevent altering ridge spatial period. |
| **Noise** | *EXCLUDED* | None | **Excluded** because bilateral filtering in `transform_c5` already denoises sensor artifacts. |

---

## 3. Implementation Details
- **Random Seed:** 42 (applied to NumPy, TensorFlow, and all augmentation layers)
- **Fill Mode:** `nearest` (prevents artificial high-contrast boundary rings)
- **Interpolation:** `bilinear`
- **Application Target:** Individual scans $s_1, s_2, s_3$ prior to DenseNet-121 feature extraction.
