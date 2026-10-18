# Data Quality, Label Integrity & Shortcut Analysis

**Audit Date:** 2026-10-17  
**Scope:** Forensic evaluation of image data quality, ground truth label provenance, optical sensor physical parameters, destructive image processing, and dataset shortcut features across all cohorts.

---

## 1. Overview of Cohorts & Label Provenance

| Cohort Name | Participants / Scans | Sensor Type | Label Verification Method | Integrity Rating & Scientific Status |
| :--- | :---: | :---: | :--- | :--- |
| **Legacy 6k Dataset** | ~6,000 images (2,777 in 4-class subset) | Unknown heterogeneous optical / flatbed | Synthetic / Web scraped (Unverified) | **Fatally Flawed.** Zero participant IDs; 42 trivial non-ridge shortcut features; random split leakage. |
| **Primary MFS100 Cohort** | 101 participants (303 scans) | Mantra MFS100 Optical (500 DPI) | **Self-reported survey** (Zero lab confirmation) | **Unreliable Ground Truth.** Known literature error rate for self-reported blood type is 12%–25%. |
| **Hospital Captured Cohort** | 26 participants (78 scans) | Mantra MFS100 Optical (500 DPI) | **Hospital Serology Lab Sheets** (25 verified, 1 missing) | **Quarantined.** Serologically valid, but cross-cohort identity overlap with Primary MFS100 is unverified. |

---

## 2. Legacy 6k Dataset: Shortcut Feature Analysis

The historical 85%–86% accuracy reported on the legacy dataset was investigated for confounding artifacts. The dataset was found to suffer from two fatal vulnerabilities:

### A. Lack of Participant Identifiers
The dataset contains 6,000 files with 5,985 unique SHA-256 hashes (15 duplicate files). However, because no participant IDs were recorded, a standard 80/20 train/test split placed impressions from the same physical fingers into both the training and test sets. The models memorized individual finger ridge topography rather than discovering bloodgroup-specific phenotypes.

### B. Trivial Non-Ridge Shortcut Features
A diagnostic classifier trained strictly on **42 non-ridge global image statistics** (excluding all fingerprint minutiae) achieved **51.2% balanced accuracy** (random chance = 25.0%) on the 4-class task:
- **Alpha Channel Variance:** Web-scraped PNGs in class AB+ had embedded transparency channels with distinct bounding margins ($p < 0.001$).
- **Background Intensity & Contrast:** Class O+ images exhibited a systematically higher mean background luminosity (182.4 vs 154.1 in A+), reflecting differing scanner hardware or capture software versions across classes.
- **Border Padding:** Zero-pixel edge padding varied systematically across class subsets due to inconsistent batch-cropping scripts.

---

## 3. Primary MFS100 Cohort: Ground Truth Reliability & Class Distribution

The primary MFS100 dataset comprises 101 participants, each providing 3 scans of the right thumb:
- **Class Breakdown:**
  - A+: 24 participants (23.76%)
  - B+: 32 participants (31.68%)
  - AB+: 11 participants (10.89%)
  - O+: 34 participants (33.66%)
- **Dev-80 Subset:** 80 participants (240 scans)
- **Locked Test Subset:** 21 participants (63 scans)

### The Self-Reported Label Hazard
Crucially, **100% of labels in the 101-participant cohort were self-reported** via an intake questionnaire without medical serology confirmation. In clinical epidemiology and transfusion medicine literature, self-reported blood group accuracy ranges between **75% and 88%**:
- Individuals frequently confuse Rh status or mistake ABO typing (commonly confusing O+ and B+).
- Even in a hypothetical scenario where an optical fingerprint scanner possessed a true biological signal, training on labels containing a 15%–20% label noise ceiling caps theoretical maximum classification accuracy and severely distorts gradient updates.

---

## 4. Optical Scanner Physics & Destructive Preprocessing

### A. Sensor Resolution vs Friction Ridge Biology
- **Sensor:** Mantra MFS100 optical fingerprint scanner.
- **Native Resolution:** $316 \times 354$ pixels, 8-bit grayscale, 500 DPI.
- **Human Ridge Biology:** At 500 DPI ($19.69\text{ pixels/mm}$), human friction ridge wavelength is typically $0.4\text{ mm}$ to $0.5\text{ mm}$, corresponding to **8 to 10 pixels per ridge-valley cycle**.

### B. Destructive Downsampling (Lanczos to $112 \times 112$)
In EXP-01 through EXP-06C, images were downsampled by a factor of $\approx 3\times$ down to $112 \times 112$ pixels:
$$\text{Downsampled Ridge Period} = \frac{8\text{ to }10\text{ pixels}}{3} \approx 2.67\text{ to }3.33\text{ pixels per cycle}$$
- The Nyquist sampling limit requires at least 2 pixels per cycle. At $2.67\text{ pixels}$, any slight affine rotation, sensor tilt, or shear causes severe **spatial aliasing**, moiré interference, and the destruction of minutiae ridge endings and bifurcations.
- The DenseNet-121 backbone was consequently fed blurry, low-frequency gray patches with severely degraded structural minutiae.

### C. Filtering Artifacts (CLAHE & Bilateral Smoothing)
In EXP-06D, an ablation experiment compared "Enhanced C5" (CLAHE + Bilateral filtering) against "Raw C5" (direct intensity normalization without smoothing):
- **Raw C5 out-of-fold Macro F1:** $0.4124$
- **Enhanced C5 out-of-fold Macro F1:** $0.3502$
- **Net Difference:** $+0.0622$ in favor of raw images.
- **Root Cause:** Bilateral smoothing blurred fine ridge contours, while CLAHE over-amplified local sweat-pore noise and sensor glass dust specks, degrading true anatomical signal.

---

## 5. Quarantined Hospital Cohort Status

The hospital-captured dataset (26 participants, 78 scans) has documented serological laboratory verification for 25 out of 26 participants. However, it is **strictly quarantined**:
1. **Cross-Cohort Participant Overlap:** The primary MFS100 data collection occurred at the academic institution. Because several participants volunteered in multiple data collection sessions, cross-cohort participant identity overlap has not been ruled out.
2. If any participant in the hospital cohort also exists in the 21-person locked test set or Dev-80, pooling or cross-evaluating without de-anonymized identity matching will introduce catastrophic participant-identity leakage.
3. **Mandatory Rule:** This cohort must remain completely isolated until formal participant identity auditing and IRB/consent protocol alignment are conducted.
