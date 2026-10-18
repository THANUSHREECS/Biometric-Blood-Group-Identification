# Addendum to Quantitative Audit: Alpha Channel Analysis in `data/raw` and Verification of Image Processing Transformation

**Project:** Fingerprint-Based Blood Group Classification (A+, AB+, B+, O+)  
**Document Type:** Read-Only Empirical Addendum  
**Date:** October 17, 2026  
**Auditor:** ML Research Engineer (Independent Audit)  
**Target Datasets:** `data/raw/` vs. `data/processed/` (2,769 positive-class images in `outputs/dataset_manifest.csv`)  

---

## 1. Overview & Mandate

This addendum records a read-only empirical examination of the 2,769 positive-class fingerprint images in `data/raw/` and evaluates the exact mathematical relationship between raw image representations and their counterparts in `data/processed/`. 

No historical or frozen audit files were modified. All evaluations adhere strictly to empirical reporting and "consistent with" framing.

---

## 2. Channel Representation & Dimensions in `data/raw`

Direct header and band inspection of all 2,769 positive-class images in `data/raw/` reveals two distinct image formats:

1. **Standard Cutout Format (2,752 images, 99.39% of dataset):**
   - Image dimensions: $96 \times 103$ pixels.
   - Header bit depth: 32 bits per pixel (`biBitCount` = 32, compression `BI_BITFIELDS`, file size 39,690 bytes).
   - Channels: 4 channels (RGBA format, comprising RGB color channels plus a 1-byte alpha channel).
2. **Non-Standard Unmasked Format (17 images, 0.61% of dataset):**
   - Image dimensions: $241 \times 298$ pixels.
   - Header bit depth: 24 bits per pixel (`biBitCount` = 24, compression `BI_RGB`, file size 215,806 bytes).
   - Channels: 3 channels (RGB format, no alpha channel present).

---

## 3. Alpha-Channel Statistics by Class in `data/raw`

For the 2,752 images containing an alpha channel in `data/raw/`, pixel values in the 4th channel were evaluated across each blood group class.

### Table 1: Per-Class Alpha Channel Descriptive Statistics in `data/raw`
*Evaluated on the 2,752 4-channel images in `data/raw/` matching `outputs/dataset_manifest.csv`.*

| Class | Image Count ($N$) | Mean Alpha $\pm$ SD | Min Alpha | Max Alpha | Transparent Pixels ($\alpha = 0$) | Opaque Pixels ($\alpha = 255$) | Anti-Aliased Pixels ($0 < \alpha < 255$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 563 | $234.88 \pm 0.00$ | 0 | 255 | $7.89\% \pm 0.00\%$ | $92.11\% \pm 0.00\%$ | $0.00\% \pm 0.00\%$ |
| **B+** | 644 | $234.88 \pm 0.00$ | 0 | 255 | $7.89\% \pm 0.00\%$ | $92.11\% \pm 0.00\%$ | $0.00\% \pm 0.00\%$ |
| **AB+** | 702 | $234.88 \pm 0.00$ | 0 | 255 | $7.89\% \pm 0.00\%$ | $92.11\% \pm 0.00\%$ | $0.00\% \pm 0.00\%$ |
| **O+** | 843 | $234.88 \pm 0.00$ | 0 | 255 | $7.89\% \pm 0.00\%$ | $92.11\% \pm 0.00\%$ | $0.00\% \pm 0.00\%$ |
| **Total / Overall** | **2,752** | **$234.88 \pm 0.00$** | **0** | **255** | **$7.89\% \pm 0.00\%$** | **$92.11\% \pm 0.00\%$** | **$0.00\% \pm 0.00\%$** |

### Empirical Observations on Alpha Distribution:
- **Strictly Binary Mask:** The alpha channel in `data/raw/` contains only two discrete values: 0 and 255. Exactly $0.00\%$ of pixels have intermediate values ($0 < \alpha < 255$), demonstrating that the mask is an unweighted binary segmentation boundary without anti-aliasing.
- **Identical Transparent Fraction:** Across all 2,752 standard images, the transparent fraction is identically $7.89\%$ (780 pixels out of 9,888 total pixels per image), and the interior opaque fraction is identically $92.11\%$ (9,108 pixels). Note: only the transparent fraction was evaluated across classes; pixel-by-pixel spatial mask identity was not compared across images.
- **Class Invariance of Transparent Fraction:** The transparent fraction does not vary across blood group classes; mean alpha ($234.88 \pm 0.00$), standard deviation ($68.74 \pm 0.00$), and transparent pixel fraction ($7.89\% \pm 0.00\%$) are identical across A+, B+, AB+, and O+.
- **Different Capture Format for Non-Standard Images:** Exactly 17 images deviate from the standard dimensions ($241 \times 298$ pixels). These 17 images have no alpha channel (standard 24-bit RGB), representing a different capture or export format from the remaining 2,752 images.

---

## 4. Pixel-Level Comparison: `data/processed` vs. `data/raw` Composited on White

To determine the exact transformation that produced `data/processed/`, each raw image was composited onto a solid white background $(255, 255, 255)$ using standard alpha blending:

$$C_{\text{out}} = \text{round}\left( C_{\text{raw}} \cdot \frac{\alpha}{255} + 255 \cdot \left(1 - \frac{\alpha}{255}\right) \right)$$

For the 17 RGB images without an alpha channel, pixels were compared directly without modification.

### Table 2: Pixel-Level Concordance between `data/processed` and `data/raw`
*Evaluated across all 2,769 positive-class images in `outputs/dataset_manifest.csv`.*

| Evaluation Metric | Observed Value | Concordance Interpretation |
| :--- | :---: | :--- |
| **Exact Bitwise Match (Composited on White)** | **2,769 / 2,769 (100.00%)** | Bit-for-bit identical across all pixels |
| **Exact Match with Raw RGB (Ignoring Alpha)** | 17 / 2,769 (0.61%) | Matches only for the 17 non-standard RGB images |
| **Maximum Absolute Pixel Difference ($\Delta_{\max}$)** | **0** | Zero discrepancy across all color channels |
| **Mean Absolute Pixel Difference ($\Delta_{\text{mean}}$)** | **0.000000** | Zero numerical difference |

---

## 5. Methodological Conclusions

1. **Deterministic Origin of `data/processed`:**
   The finding that all 2,769 images ($100.00\%$) in `data/processed/` match the raw images composited onto solid white with zero pixel difference ($\Delta = 0$) is **consistent with `data/processed/` having been produced by compositing the binary alpha cutouts of `data/raw/` onto a pure white background and serializing the output as standard 24-bit RGB**.
2. **Clarification of Trivial Feature Group (ii):**
   Because `data/processed/` contains 24-bit RGB images where the alpha channel was discarded during saving, `has_alpha = 0` and `alpha_mean = 255.0` are invariant constants across all 2,769 processed images **by construction**. Consequently, feature group (ii) ("Alpha + Border Statistics") evaluated in Section 6 of the primary audit report is effectively **border-only**.
3. **Implications for Shortcut Learning:**
   The class-invariance of the transparent fraction rules out an alpha channel or compositing imprint as the mechanism driving class separation. However, this finding **does not rule out class differences in RGB content** (such as overall image intensity histograms, brightness distribution, or border pixel tone), which remain predictive ($51.43\%$ balanced accuracy) under linear classifiers.
4. **Different Capture Format:**
   The 17 non-standard images ($241 \times 298$, 215,806 bytes) completely lack an alpha channel, indicating a different acquisition or export format from the rest of the dataset.

