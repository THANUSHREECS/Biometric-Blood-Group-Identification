# EXPERIMENT 9 FEATURE AUDIT

## STATUS: PASS

**1. Feature Families Implemented:**
- **H0 (Quality and Contrast):** Laplacian variance (sharpness) and RMS standard deviation (contrast).
- **H1 (Ridge Orientation):** Histogram of Oriented Gradients (HOG) extracted at 8 orientations.
- **H2 (Ridge Frequency):** 2D Fast Fourier Transform (FFT) magnitude spectrum statistics (mean, std, 90th percentile).
- **H3 (Local Texture):** Uniform Local Binary Patterns (LBP) histogram.
- **H4 (Combined Structural):** Orientation (H1) + Frequency (H2).
- **H5 (Combined All):** H0 + H1 + H2 + H3.

**2. Exclusions (Minutiae):**
- **Minutiae Extraction Omitted:** Extracting accurate ridge minutiae (bifurcations, endings) natively from optical captures via standard python libraries (like OpenCV or skimage) without heavy, specialized neuro-symbolic domain models is highly unreliable. Rather than feeding noisy, artifact-ridden minutiae points into the classifier, this feature family was excluded in favor of robust, global spatial frequency and texture orientations (HOG, FFT).

**3. Preprocessing:**
- Used the frozen C5 (composite + crop + contrast + resize) prior to feature extraction to ensure comparability with the CNN pipeline.

**4. Aggregation:**
- Features were successfully extracted per-scan and then mathematically averaged by participant (`np.mean`) into fixed-length descriptor vectors before classification.
