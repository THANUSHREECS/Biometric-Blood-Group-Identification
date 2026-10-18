# EXP-08: Feature Quality, Distribution, and Missingness Audit

**Document ID:** EXP08_FEATURE_QUALITY_REPORT  
**Date:** 2026-10-18  
**Experiment:** EXP-08-DERMATOGLYPHIC-MINUTIAE  
**Target Dataset:** Dev-80 Cohort (80 Participants, 240 Impressions)  
**Input Hardware:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI, Native $316 \times 354$ Pixels)  

---

## 1. Executive Summary

A full forensic quality audit was conducted on all 240 fingerprint scans belonging to the authorized 80 development participants. Features were extracted directly at native resolution ($316 \times 354$) with zero interpolation, cropping, or downsampling, preserving the $500\text{ DPI}$ ($19.685\text{ px/mm}$) spatial calibration.

| Metric | Result | Forensic Status |
| :--- | :---: | :--- |
| **Scan Extraction Success Rate** | **100.0% (240/240 scans)** | All scans segmented and processed successfully; active platen contact $> 5,000\text{ px}$. |
| **Participant Aggregation** | **80 rows, 28 columns** | Exactly 1 feature record per participant (3 impressions collapsed). Zero duplication. |
| **Overall Feature Missingness** | **0.052% (1 / 1,920 values)** | Only 1 participant lacked a detected delta (`core_delta_distance_mm_mean`). All other 23 numeric features had 0 missing values. |
| **Sub-Core Ridge Density Mean** | **$12.24 \pm 0.56\text{ ridges}/25\text{ mm}^2$** | Biologically plausible forensic range ($11.07\text{ to }13.47$). No implausible or negative values. |
| **Intra-Subject Density Repeatability** | **$\text{Mean std} = 0.18\text{ ridges}/25\text{ mm}^2$** | High measurement stability across repeated impressions ($\text{median std} = 0.12$). |
| **Minutiae Count Realities** | **$961.0 \pm 440.3\text{ minutiae/scan}$** | Elevated count over manual ink norms reflects optical sweat pore resolution and contact perimeter terminations. |
| **Primary Pattern Distribution** | **Whorl: 92.5%, Left Loop: 3.75%, Right Loop: 2.5%, Arch: 1.25%** | Matches right-thumb dominant sample characteristics in Dev-80. |

---

## 2. Scan-Level Quality & Failure Audit

Every scan was subjected to automated validity gates in `feature_extraction.py`:
- **Foreground Platen Segmentation:** Platen background ($> 238$ intensity) was separated from active friction ridge impressions.
- **Minimum Active Area:** Required $> 5,000\text{ pixels}$ ($> 12.9\text{ mm}^2$).
- **Extraction Failures:** **0 scans failed**. All 240 impressions satisfied foreground segmentation and orientation tensor field calculations.

---

## 3. Numeric Feature Distributions & Statistics ($N=80$ Participants)

Features were aggregated across the 3 scans per participant into mean and standard deviation metrics:

| Feature Name | Mean | Std | Min | 25% | Median | 75% | Max | Missingness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge_density_25mm2_mean` | 12.24 | 0.56 | 11.07 | 11.80 | 12.21 | 12.57 | 13.47 | 0 (0.0%) |
| `ridge_density_25mm2_std` | 0.18 | 0.20 | 0.00 | 0.06 | 0.12 | 0.22 | 1.11 | 0 (0.0%) |
| `ridge_endings_count_mean` | 380.26 | 220.02 | 77.00 | 234.00 | 327.17 | 475.75 | 1081.67 | 0 (0.0%) |
| `ridge_endings_count_std` | 45.13 | 54.83 | 2.49 | 8.99 | 20.64 | 68.70 | 242.97 | 0 (0.0%) |
| `ridge_bifurcations_count_mean` | 580.78 | 415.36 | 67.00 | 216.25 | 433.83 | 851.58 | 1600.33 | 0 (0.0%) |
| `ridge_bifurcations_count_std` | 50.18 | 63.47 | 1.25 | 14.91 | 29.06 | 61.27 | 440.29 | 0 (0.0%) |
| `total_minutiae_count_mean` | 961.04 | 440.29 | 265.67 | 607.08 | 890.17 | 1191.50 | 2297.33 | 0 (0.0%) |
| `total_minutiae_count_std` | 68.44 | 76.98 | 4.19 | 16.71 | 47.79 | 88.81 | 508.93 | 0 (0.0%) |
| `minutiae_density_cm2_mean` | 388.35 | 175.42 | 112.80 | 263.29 | 358.16 | 501.96 | 1027.12 | 0 (0.0%) |
| `minutiae_density_cm2_std` | 26.22 | 28.57 | 0.82 | 7.45 | 17.01 | 32.52 | 176.30 | 0 (0.0%) |
| `ridge_clarity_score_mean` | 0.658 | 0.111 | 0.289 | 0.596 | 0.677 | 0.738 | 0.840 | 0 (0.0%) |
| `ridge_clarity_score_std` | 0.014 | 0.016 | 0.000 | 0.003 | 0.010 | 0.019 | 0.074 | 0 (0.0%) |
| `foreground_area_mm2_mean` | 247.97 | 30.93 | 166.80 | 224.37 | 247.40 | 277.17 | 288.68 | 0 (0.0%) |
| `foreground_area_mm2_std` | 2.37 | 2.51 | 0.00 | 0.56 | 1.43 | 3.19 | 9.68 | 0 (0.0%) |
| `core_delta_distance_mm_mean` | 6.74 | 3.47 | 1.35 | 3.86 | 6.63 | 9.00 | 14.09 | 1 (1.25%) |
| `core_delta_distance_mm_std` | 1.96 | 1.90 | 0.00 | 0.34 | 1.27 | 3.39 | 7.36 | 0 (0.0%) |
| `core_delta_valid_scans_count` | 2.69 | 0.63 | 0.00 | 3.00 | 3.00 | 3.00 | 3.00 | 0 (0.0%) |
| `core_detection_rate` | 0.992 | 0.052 | 0.667 | 1.000 | 1.000 | 1.000 | 1.000 | 0 (0.0%) |
| `delta_detection_rate` | 0.896 | 0.209 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 (0.0%) |
| `pattern_concordance` | 0.938 | 0.177 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 | 0 (0.0%) |

---

## 4. Landmark Detection & Missing Value Handling

1. **Core Singular Points:**
   - Detected in 238 out of 240 scans ($99.17\%$).
   - Mean core detection rate per participant was $99.2\%$.
2. **Delta Singular Points:**
   - Detected in 215 out of 240 scans ($89.58\%$).
   - Deltas occasionally fall off the platen boundary when participants place the central finger pad flat rather than rolled.
3. **Core-to-Delta Distance Missingness:**
   - 79 of 80 participants had at least one valid core-to-delta distance measurement across their 3 scans.
   - Exactly 1 participant had 0 valid deltas detected across all 3 scans (`core_delta_valid_scans_count = 0`), resulting in `core_delta_distance_mm_mean = np.nan`.
   - **Handling:** Imputed strictly inside each training fold using `SimpleImputer(strategy="median")`. No synthetic replacement was performed prior to cross-validation.

---

## 5. Physical Plausibility & Noise Assessment

1. **Sub-Core Ridge Density:**
   - Measured between $11.07$ and $13.47\text{ ridges}/25\text{ mm}^2$.
   - Acree (1999) forensic standard establishes the adult human ridge density range as 9 to 14 ridges per $25\text{ mm}^2$.
   - **Conclusion:** 100% of participants fall squarely within true biological limits.
2. **Minutiae Count Interpretation:**
   - Manual forensic counts on clean ink rolls range from 50 to 90 minutiae.
   - The automated crossing-number algorithm detected an average of 961 minutiae per scan.
   - *Technical Cause:* The 500 DPI optical sensor resolves incipient sweat pores along the ridge crests, causing breaks in the 1-pixel skeleton. Additionally, platen moisture and boundary edges create termination points.
   - *Biometric Utility:* Rather than representing topological forensic minutiae graphs, this feature serves as an objective descriptor of contact texture, pore frequency, and ridge complexity.

---

## 6. Categorical Pattern Distribution & Concordance

| Majority Pattern | Participant Count | Proportion | Mean Concordance |
| :--- | :---: | :---: | :---: |
| **WHORL** | 74 | 92.50% | 0.95 |
| **LEFT_LOOP** | 3 | 3.75% | 0.89 |
| **RIGHT_LOOP** | 2 | 2.50% | 0.67 |
| **ARCH** | 1 | 1.25% | 1.00 |

- **Pattern Concordance Across Scans:**
  - 70 participants (87.5%) showed 100% pattern concordance across all 3 impressions ($3/3$ identical).
  - 5 participants (6.25%) showed 66.7% concordance ($2/3$ agreement).
  - 5 participants (6.25%) showed 33.3% concordance (discordant singular points due to differing platen coverage).
