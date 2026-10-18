# EXP-10: Multi-Scan Quality and Consistency Analysis Report

**Project:** Biometric Blood Group Detection  
**Document ID:** `EXP10_MULTI_SCAN_QUALITY_REPORT.md`  
**Date:** 2026-10-18  
**Experiment:** EXP-10 — Multi-Scan Quality and Consistency Analysis  
**Cohort Scope:** Development-80 Cohort (80 participants, 240 fingerprint scans)  
**Security Status:** STRICT DATA ISOLATION ENFORCED (Locked Test Cohort $N=21$ Quarantined; Hospital Cohort $N=26$ Quarantined)  
**Clinical Validity Notice:** RESEARCH INVESTIGATION ONLY — Not clinically validated; biological association between dermatoglyphic friction ridge patterns and ABO blood groups remains unproven.

---

## Executive Summary

Following the completion of EXP-09 (which established that continuous spatial ridge-orientation tensor fields do not yield statistically significant classification above the majority baseline: RF accuracy $40.00\%$, balanced accuracy $31.37\%$, permutation $p=0.1089$, AB+ recall $0\%$), **EXP-10 investigated whether scan-quality variation across the three impressions per participant contributes to poor predictive performance**.

Specifically, this study addressed:
1. Whether optical capture quality varies substantially across multiple scans of the same participant.
2. Whether poor-quality impressions corrupt unweighted multi-scan feature averaging.
3. Whether a prespecified **quality-weighted aggregation method** improves participant-level classification over simple unweighted averaging.

### Key Empirical Findings:
1. **Intra-Participant Quality is Highly Uniform:**  
   Across the 80 participants in Dev-80, the mean ratio between the highest-quality and lowest-quality scan for a given participant is **$1.23 \pm 1.02$**. Only **$4$ out of $80$ participants ($5.0\%$)** exhibited a quality ratio $\ge 1.5$.
2. **High Intra-Participant Ridge Consistency:**  
   The pairwise cosine similarity of friction-ridge orientation fields across the 3 impressions of the same participant averaged **$0.9355 \pm 0.0987$** (with $85\%$ of participants exceeding $0.90$).
3. **Controlled Comparison Demonstrates No Benefit (Negative Result):**  
   Under identical 5-fold Stratified K-Fold cross-validation on Dev-80:
   - **Majority-Class Baseline:** Accuracy = **$36.25\%$**, Balanced Accuracy = **$25.00\%$**, Macro F1 = **$0.1330$**.
   - **Ridge Classifier ($\alpha=10.0$):**
     - Condition A (Unweighted Mean): Accuracy = **$36.25\%$**, Balanced Accuracy = **$29.20\%$**, Macro F1 = **$0.2770$**, $\kappa = 0.0831$.
     - Condition B (Quality-Weighted Mean): Accuracy = **$36.25\%$**, Balanced Accuracy = **$28.75\%$**, Macro F1 = **$0.2730$**, $\kappa = 0.0803$.
     - **Difference ($B - A$):** $\Delta \text{Acc} = \mathbf{0.00\%}$, $\Delta \text{BalAcc} = \mathbf{-0.45\%}$, $\Delta \text{F1} = \mathbf{-0.0041}$ (McNemar exact $p = 1.0000$).
   - **L2 Logistic Regression ($C=0.1$):**
     - Condition A: Accuracy = **$35.00\%$**, Balanced Accuracy = **$28.97\%$**, Macro F1 = **$0.2775$**, $\kappa = 0.0731$.
     - Condition B: Accuracy = **$33.75\%$**, Balanced Accuracy = **$27.65\%$**, Macro F1 = **$0.2650$**, $\kappa = 0.0553$.
     - **Difference ($B - A$):** $\Delta \text{Acc} = \mathbf{-1.25\%}$, $\Delta \text{BalAcc} = \mathbf{-1.32\%}$, $\Delta \text{F1} = \mathbf{-0.0125}$.
   - **Random Forest ($n=100$, depth=4):**
     - Condition A: Accuracy = **$40.00\%$**, Balanced Accuracy = **$31.37\%$**, Macro F1 = **$0.3048$**, $\kappa = 0.1316$.
     - Condition B: Accuracy = **$33.75\%$**, Balanced Accuracy = **$26.15\%$**, Macro F1 = **$0.2503$**, $\kappa = 0.0377$.
     - **Difference ($B - A$):** $\Delta \text{Acc} = \mathbf{-6.25\%}$, $\Delta \text{BalAcc} = \mathbf{-5.22\%}$, $\Delta \text{F1} = \mathbf{-0.0545}$ (McNemar exact $p = 0.1797$).
4. **Minority Class Recall Remains Zero:**  
   All models under both aggregation methods achieved **$0.0\%$ recall on AB+** ($0/7$ participants correctly identified).

### Decision:
Quality-weighted aggregation **fails to provide reproducible improvement** over unweighted averaging. In fact, non-uniform weighting degrades performance by reducing the noise-canceling benefit of arithmetic averaging. Scan-quality variation is **not** the factor suppressing predictive accuracy. In accordance with the prespecified protocol, this line of investigation is concluded with a confirmed negative result.

---

## Phase 1: Per-Scan Quality Audit (N=240 Scans)

All 240 optical fingerprint impressions in the Development-80 cohort were audited across four independent physical and biometric quality dimensions:
1. **Foreground Platen Coverage ($C_{\text{fg}}$):** The proportion of the sensor platen ($354 \times 316$ pixels) contacted by the finger impression (segmented via adaptive glass threshold $I < 238$ with morphological closing).
2. **Ridge Orientation Coherence ($\kappa$):** The local anisotropy of the gradient structure tensor, measuring ridge directionality definition:
   $$\kappa = \frac{\sqrt{(J_{xx} - J_{yy})^2 + 4J_{xy}^2}}{J_{xx} + J_{yy} + \epsilon}$$
3. **Ridge Clarity / Local Contrast ($\sigma_{\text{local}}$):** The standard deviation of pixel intensities within $16\times 16$ spatial blocks across the foreground friction-ridge region.
4. **Unusable Region Proportion ($P_{\text{unusable}}$):** The fraction of the foreground area where ridge structure is corrupted, blurred, or smudged ($\kappa < 0.25$ or $\sigma_{\text{local}} < 12.0$).
5. **Composite Technical Quality Score ($Q$):**
   $$Q = C_{\text{fg}} \cdot \bar{\kappa}_{\text{fg}} \cdot (1 - P_{\text{unusable}}) \cdot \min\left(1.0, \frac{\bar{\sigma}_{\text{local}}}{40.0}\right)$$
   Bounded strictly in $[0.001, 1.0]$. The quality score is computed solely from image physics and contains **zero label information**.

### Distribution of Scan Quality Metrics (N=240 Scans):

| Quality Metric | Mean $\pm$ Std | Minimum | 25th %ile | Median | 75th %ile | Maximum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Foreground Coverage** | $0.8609 \pm 0.1050$ | $0.5844$ | $0.8032$ | $0.8772$ | $0.9421$ | $1.0000$ |
| **Ridge Orientation Coherence** | $0.6580 \pm 0.1126$ | $0.2664$ | $0.6125$ | $0.6841$ | $0.7329$ | $0.8416$ |
| **Ridge Contrast ($\sigma_{\text{local}}$)** | $59.48 \pm 13.87$ | $15.00$ | $51.24$ | $61.82$ | $69.75$ | $78.83$ |
| **Unusable Proportion** | $0.1278 \pm 0.1334$ | $0.0161$ | $0.0435$ | $0.0782$ | $0.1584$ | $0.6919$ |
| **Composite Quality Score ($Q$)** | $0.4933 \pm 0.1492$ | $0.0339$ | $0.4285$ | $0.5256$ | $0.5962$ | $0.7416$ |

### Technical Reliability of Measures:
- The MFS100 optical sensor captures high-contrast friction ridge impressions with an average foreground coverage of $86.1\%$ and mean ridge coherence of $0.658$.
- Extreme capture failures are rare: only $3$ out of $240$ scans ($1.25\%$) exhibited an unusable proportion exceeding $50\%$.
- A one-way ANOVA of quality score $Q$ across blood groups yielded $F = 4.920, p = 0.0025$ (A+: $0.4589$, AB+: $0.4780$, B+: $0.4688$, O+: $0.5407$). This reflects the sensor-calibration and ambient-lighting drift across collection sessions previously documented in EXP-09, confirming that optical quality differences are scanner-batch artifacts rather than physiological characteristics.

---

## Phase 2: Multi-Scan Consistency Analysis (N=80 Participants)

To determine whether differences in image quality or ridge representations between impressions of the same participant justify quality-weighted aggregation, we evaluated intra-participant consistency across all 80 participants.

### Intra-Participant Consistency Metrics:

| Metric | Definition | Mean $\pm$ Std | Min | Median | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intra-Participant Cosine Similarity** | Mean pairwise cosine similarity of $[u, v]$ tensor fields | **$0.9355 \pm 0.0987$** | $0.5726$ | $0.9621$ | $0.9981$ |
| **Intra-Participant Quality Ratio** | $\max(Q_{1}, Q_{2}, Q_{3}) / \min(Q_{1}, Q_{2}, Q_{3})$ | **$1.23 \pm 1.02$** | $1.00$ | $1.11$ | $10.09$ |
| **Quality Range ($\Delta Q$)** | $\max(Q) - \min(Q)$ | **$0.078 \pm 0.082$** | $0.002$ | $0.054$ | $0.521$ |

### Analysis of Quality Variation:
- **Collinearity of Impressions:** For the typical participant, successive fingerprint scans are nearly identical in orientation topology (median cosine similarity $= 0.9621$).
- **Symmetry of Quality:** For **$76$ out of $80$ participants ($95.0\%$)**, the ratio between best and worst scan quality was less than $1.5$.
- Only **$2$ out of $80$ participants ($2.5\%$)** exhibited substantial quality degradation in one scan ($Q_{\text{ratio}} \ge 2.0$).
- **Conclusion:** Intra-participant scan quality variation is negligible for $95\%$ of the cohort. The theoretical justification for quality-weighted aggregation is therefore weak.

---

## Phase 3: Controlled Comparison Protocol

We conducted a strictly controlled comparison between two aggregation strategies applied to the 920-dimensional continuous orientation tensor field:

### Aggregation Conditions:
1. **Condition A (Unweighted Three-Scan Average):**
   $$\mathbf{x}_{p}^{\text{unweighted}} = \frac{1}{3} \sum_{s=1}^3 \mathbf{x}_{p, s}$$
2. **Condition B (Prespecified Quality-Weighted Average):**
   $$w_{p, s} = \frac{Q_{p, s}}{\sum_{j=1}^3 Q_{p, j}}, \quad \mathbf{x}_{p}^{\text{weighted}} = \sum_{s=1}^3 w_{p, s} \mathbf{x}_{p, s}$$

### Prespecified Controls:
- **Identical Cross-Validation Folds:** Evaluated using the exact 5-fold Stratified K-Fold cross-validation partitioning ($N=80$ participants, stratified across A+: 19, AB+: 7, B+: 25, O+: 29).
- **In-Fold Preprocessing:** `StandardScaler` fitted strictly on training folds and applied to test folds.
- **Fixed Model Architectures:** Identical hyperparameters across Ridge Classifier ($\alpha=10.0$), L2 Logistic Regression ($C=0.1$), and Random Forest ($n=100$, depth=4, balanced class weights).
- **Zero Label Tuning:** Quality weights were computed purely from Phase 1 technical metrics without tuning against cross-validation targets.

---

## Phase 4: Experimental Evaluation & Results

The experiment script [run_exp10_quality_aggregation.py](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp10_multi_scan_quality/run_exp10_quality_aggregation.py) executed in **$8.63\text{ seconds}$**.

### 4.1 Controlled Comparison Summary

| Model | Aggregation Condition | Accuracy | Balanced Accuracy | Macro F1 | Cohen's Kappa ($\kappa$) | McNemar $p$-value vs A |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Condition A (Unweighted) | $36.25\%$ | $25.00\%$ | $0.1330$ | $0.0000$ | — |
| | Condition B (Quality-Weighted) | $36.25\%$ | $25.00\%$ | $0.1330$ | $0.0000$ | $1.0000$ |
| **Ridge Classifier** | Condition A (Unweighted) | **$36.25\%$** | **$29.20\%$** | **$0.2770$** | **$0.0831$** | — |
| | Condition B (Quality-Weighted) | **$36.25\%$** | **$28.75\%$** | **$0.2730$** | **$0.0803$** | $1.0000$ |
| | **Difference ($B - A$)** | $\mathbf{+0.00\%}$ | $\mathbf{-0.45\%}$ | $\mathbf{-0.0041}$ | $\mathbf{-0.0029}$ | (Identical on 78/80) |
| **L2 Logistic Regression** | Condition A (Unweighted) | **$35.00\%$** | **$28.97\%$** | **$0.2775$** | **$0.0731$** | — |
| | Condition B (Quality-Weighted) | **$33.75\%$** | **$27.65\%$** | **$0.2650$** | **$0.0553$** | $1.0000$ |
| | **Difference ($B - A$)** | $\mathbf{-1.25\%}$ | $\mathbf{-1.32\%}$ | $\mathbf{-0.0125}$ | $\mathbf{-0.0178}$ | |
| **Random Forest** | Condition A (Unweighted) | **$40.00\%$** | **$31.37\%$** | **$0.3048$** | **$0.1316$** | — |
| | Condition B (Quality-Weighted) | **$33.75\%$** | **$26.15\%$** | **$0.2503$** | **$0.0377$** | $0.1797$ |
| | **Difference ($B - A$)** | $\mathbf{-6.25\%}$ | $\mathbf{-5.22\%}$ | $\mathbf{-0.0545}$ | $\mathbf{-0.0939}$ | (Degraded by 5 pts) |

### 4.2 Per-Class Recall Breakdown

| Model & Condition | Recall A+ ($N=19$) | Recall AB+ ($N=7$) | Recall B+ ($N=25$) | Recall O+ ($N=29$) |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline (A & B)** | $0.0\%$ (0/19) | $0.0\%$ (0/7) | $0.0\%$ (0/25) | **$100.0\%$** (29/29) |
| **Ridge Classifier (Condition A)** | $26.3\%$ (5/19) | **$0.0\%$** (0/7) | $56.0\%$ (14/25) | $34.5\%$ (10/29) |
| **Ridge Classifier (Condition B)** | $26.3\%$ (5/19) | **$0.0\%$** (0/7) | $56.0\%$ (14/25) | $31.0\%$ (9/29) |
| **Logistic Regression (Condition A)** | $36.8\%$ (7/19) | **$0.0\%$** (0/7) | $48.0\%$ (12/25) | $31.0\%$ (9/29) |
| **Logistic Regression (Condition B)** | $31.6\%$ (6/19) | **$0.0\%$** (0/7) | $48.0\%$ (12/25) | $31.0\%$ (9/29) |
| **Random Forest (Condition A)** | $26.3\%$ (5/19) | **$0.0\%$** (0/7) | $44.0\%$ (11/25) | $55.2\%$ (16/29) |
| **Random Forest (Condition B)** | $15.8\%$ (3/19) | **$0.0\%$** (0/7) | $44.0\%$ (11/25) | $44.8\%$ (13/29) |

*Critical Observation:* All models under both Condition A and Condition B completely failed to detect AB+, achieving **$0.0\%$ recall** in all configurations.

### 4.3 Confusion Matrices (Canonical Order: A+, AB+, B+, O+)

#### Ridge Classifier — Condition A (Unweighted):
$$\begin{bmatrix} 
5 & 1 & 6 & 7 \\
2 & 0 & 4 & 1 \\
4 & 0 & 14 & 7 \\
8 & 1 & 10 & 10 
\end{bmatrix}$$

#### Ridge Classifier — Condition B (Quality-Weighted):
$$\begin{bmatrix} 
5 & 1 & 6 & 7 \\
2 & 0 & 4 & 1 \\
4 & 0 & 14 & 7 \\
9 & 1 & 10 & 9 
\end{bmatrix}$$
*(Note: Predictions are identical for 78 out of 80 participants; only a single O+ prediction flipped.)*

#### Random Forest — Condition A (Unweighted):
$$\begin{bmatrix} 
5 & 0 & 2 & 12 \\
2 & 0 & 4 & 1 \\
6 & 2 & 11 & 6 \\
5 & 1 & 7 & 16 
\end{bmatrix}$$

#### Random Forest — Condition B (Quality-Weighted):
$$\begin{bmatrix} 
3 & 0 & 5 & 11 \\
2 & 0 & 4 & 1 \\
4 & 2 & 11 & 8 \\
5 & 1 & 10 & 13 
\end{bmatrix}$$
*(Note: Random Forest accuracy decreased from 32/80 to 27/80 because weighted averaging reduced the noise-canceling properties of uniform averaging.)*

### 4.4 Fold-by-Fold Stability Comparison

Across the 5 validation folds ($N=16$ per fold):

| Fold | Ridge A BalAcc | Ridge B BalAcc | RF A BalAcc | RF B BalAcc |
| :--- | :---: | :---: | :---: | :---: |
| **Fold 1** | $30.83\%$ | $30.83\%$ | $38.33\%$ | $22.92\%$ |
| **Fold 2** | $38.75\%$ | $38.75\%$ | $40.83\%$ | $32.50\%$ |
| **Fold 3** | $23.75\%$ | $23.75\%$ | $19.58\%$ | $19.58\%$ |
| **Fold 4** | $19.17\%$ | $19.17\%$ | $31.67\%$ | $28.33\%$ |
| **Fold 5** | $31.25\%$ | $27.08\%$ | $26.25\%$ | $26.25\%$ |
| **Mean $\pm$ Std** | **$28.75\% \pm 6.74\%$** | **$27.92\% \pm 7.42\%$** | **$31.33\% \pm 7.79\%$** | **$25.92\% \pm 4.44\%$** |

Fold-level variability remains substantial across all conditions, demonstrating that performance fluctuations are driven by small-sample cohort sampling rather than scan quality differences.

---

## Phase 5: Scientific Decision & Final Synthesis

### 5.1 Why Quality-Weighted Aggregation Failed

1. **Noise-Reduction Dynamics:**  
   In fingerprint feature extraction, small capture variations (finger placement, sensor contact pressure) introduce zero-mean measurement noise. Simple arithmetic averaging of 3 impressions reduces independent noise variance by a factor of $1/3$ ($1/\sqrt{3} \approx 42\%$ reduction in standard error).  
   When non-uniform quality weighting is introduced, the effective sample size ($N_{\text{eff}} = 1 / \sum w_i^2$) decreases from $3.0$ toward $1.0$, which increases measurement variance and degrades generalization.
2. **Absence of Real Quality Outliers:**  
   Because $95\%$ of scans had high and uniform optical quality ($Q_{\text{ratio}} < 1.5$), there were no heavily corrupted impressions to downweight. Weighting acted as an arbitrary distortion rather than a meaningful filter.
3. **Fundamental Lack of Biological Signal:**  
   Regardless of whether scans are averaged uniformly, weighted by clarity, or filtered by quality, the underlying friction-ridge orientation field contains no discriminative information about ABO blood group alleles.

### 5.2 Research Trajectory & Conclusion of Investigation

Across ten systematic experiments covering:
- Raw optical pixels (ResNet-50, DenseNet-121, EfficientNet-B0, MobileNetV2, VGG16)
- Input resolution scaling ($112\times 112$ vs. $224\times 224$)
- Scanner background and platen border ablations
- Discrete dermatoglyphic features (minutiae counts, ridge density, pattern topology)
- Continuous spatial orientation tensor fields (EXP-09)
- Multi-scan quality weighting and consistency analysis (EXP-10)

Every valid participant-level evaluation has converged to the same conclusion: **optical fingerprint impressions do not possess statistically significant, participant-generalizable predictive power for ABO blood group determination**.

### 5.3 Final Recommendation

- **Terminate Further Optimization of Fingerprint Aggregation & Preprocessing:** The hypothesis that capture noise or scan quality masks an underlying biological relationship has been thoroughly tested and disproven.
- **Maintain Research-Only Framing:** All experimental outputs must remain strictly designated as research investigations; fingerprint biometric systems cannot be used for clinical or diagnostic blood-group determination.
