# EXP-10: Multi-Scan Quality and Consistency Analysis Report (Corrected)

**Document ID:** `EXP10_MULTI_SCAN_QUALITY_REPORT_CORRECTED.md`  
**Original Reference:** `reports/accuracy_improvement_audit/EXP10_MULTI_SCAN_QUALITY_REPORT.md` (Preserved Unmodified)  
**Correction Date:** 2026-10-18  
**Experiment:** EXP-10 — Multi-Scan Quality and Consistency Analysis  
**Cohort Scope:** Development-80 Cohort (80 participants, 240 fingerprint scans)  
**Security Status:** STRICT DATA ISOLATION ENFORCED (Locked Test Cohort $N=21$ Quarantined; Hospital Cohort $N=26$ Quarantined)  
**Clinical Validity Notice:** RESEARCH INVESTIGATION ONLY — Not clinically validated; biological association between dermatoglyphic friction ridge patterns and ABO blood groups remains unproven.

---

## Document Purpose & Erratum Notice

This document is the **official corrected version** of the EXP-10 evaluation report. It provides an explicit erratum resolving the typographical transcription error in Section 4 of the original draft:
- **Ridge Condition B Confusion Matrix & Recall Correction:** Reconciles the printed confusion matrix and per-class recall for Ridge Condition B with the authoritative machine-readable artifacts ([exp10_participant_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp10_multi_scan_quality/exp10_participant_oof_predictions.csv) and [exp10_evaluation_metrics.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp10_multi_scan_quality/exp10_evaluation_metrics.json)).
- **Clarification of Prediction Flips:** Replaces the inaccurate note that *"only a single O+ prediction flipped"* with the verified fact that **two predictions flipped** (`P0012` A+ to O+, and `P007` B+ to O+), maintaining total accuracy at 29/80 (36.25%) while shifting per-class recall.

The original report [EXP10_MULTI_SCAN_QUALITY_REPORT.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP10_MULTI_SCAN_QUALITY_REPORT.md) remains preserved untouched.

---

## 1. Executive Summary

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
5. **Standardized Scientific Conclusion:**  
   *Under the evaluated datasets, fingerprint representations, and participant-level validation procedures, the experiments did not establish reliable, generalizable blood-group classification performance.*

---

## 2. Phase 1: Per-Scan Quality Audit (N=240 Scans)

All 240 optical fingerprint impressions in the Development-80 cohort were audited across four independent physical and biometric quality dimensions:
1. **Foreground Platen Coverage ($C_{\text{fg}}$):** Mean = $0.8609 \pm 0.1050$ (Min = $0.5844$, Max = $1.0000$).
2. **Ridge Orientation Coherence ($\kappa$):** Mean = $0.6580 \pm 0.1126$ (Min = $0.2664$, Max = $0.8416$).
3. **Ridge Clarity / Local Contrast ($\sigma_{\text{local}}$):** Mean = $59.48 \pm 13.87$ (Min = $15.00$, Max = $78.83$).
4. **Unusable Region Proportion ($P_{\text{unusable}}$):** Mean = $0.1278 \pm 0.1334$ (Min = $0.0161$, Max = $0.6919$).
5. **Composite Technical Quality Score ($Q$):** Mean = $0.4933 \pm 0.1492$ (Min = $0.0339$, Max = $0.7416$).

Serious capture corruption is exceedingly rare: only 3 out of 240 scans ($1.25\%$) had an unusable area $>50\%$. Optical capture quality is uniformly high across the cohort.

---

## 3. Phase 2: Multi-Scan Consistency Analysis (N=80 Participants)

- **Intra-Participant Orientation Collinearity:** Mean pairwise cosine similarity of $[u, v]$ tensor fields = **$0.9355 \pm 0.0987$** (median = $0.9621$).
- **Intra-Participant Quality Symmetry:**
  - Mean quality ratio ($\max Q / \min Q$) = **$1.23 \pm 1.02$**.
  - **$95.0\%$ of participants ($76/80$)** exhibited a quality ratio $< 1.5$.
  - Only **$2$ out of $80$ participants ($2.5\%$)** had substantial quality degradation in one impression ($Q_{\text{ratio}} \ge 2.0$: `P0086` at 2.42, `P0089` at 10.09).
- **Conclusion:** Intra-participant scan quality variation is negligible for $95\%$ of the cohort. The theoretical basis for quality-weighting is unsupported by the sensor data.

---

## 4. Phase 4: Controlled Comparison Results (Reconciled & Corrected)

### 4.1 Controlled Comparison Summary Table

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

### 4.2 Per-Class Recall Breakdown (Corrected)

| Model & Condition | Recall A+ ($N=19$) | Recall AB+ ($N=7$) | Recall B+ ($N=25$) | Recall O+ ($N=29$) |
| :--- | :---: | :---: | :---: | :---: |
| **Majority Baseline (A & B)** | $0.0\%$ (0/19) | $0.0\%$ (0/7) | $0.0\%$ (0/25) | **$100.0\%$** (29/29) |
| **Ridge Classifier (Condition A)** | $26.3\%$ (5/19) | **$0.0\%$** (0/7) | $56.0\%$ (14/25) | $34.5\%$ (10/29) |
| **Ridge Classifier (Condition B - Corrected)** | **$21.1\%$ (4/19)** | **$0.0\%$** (0/7) | $56.0\%$ (14/25) | **$37.9\%$ (11/29)** |
| **Logistic Regression (Condition A)** | $36.8\%$ (7/19) | **$0.0\%$** (0/7) | $48.0\%$ (12/25) | $31.0\%$ (9/29) |
| **Logistic Regression (Condition B)** | $31.6\%$ (6/19) | **$0.0\%$** (0/7) | $48.0\%$ (12/25) | $31.0\%$ (9/29) |
| **Random Forest (Condition A)** | $26.3\%$ (5/19) | **$0.0\%$** (0/7) | $44.0\%$ (11/25) | $55.2\%$ (16/29) |
| **Random Forest (Condition B)** | $15.8\%$ (3/19) | **$0.0\%$** (0/7) | $44.0\%$ (11/25) | $44.8\%$ (13/29) |

### 4.3 Confusion Matrices (Corrected, Canonical Order: A+, AB+, B+, O+)

#### Ridge Classifier — Condition A (Unweighted):
$$\begin{bmatrix} 
5 & 1 & 6 & 7 \\
2 & 0 & 4 & 1 \\
4 & 0 & 14 & 7 \\
8 & 1 & 10 & 10 
\end{bmatrix}$$

#### Ridge Classifier — Condition B (Quality-Weighted, Corrected):
$$\begin{bmatrix} 
4 & 1 & 6 & 8 \\
2 & 0 & 4 & 1 \\
4 & 0 & 14 & 7 \\
8 & 1 & 9 & 11 
\end{bmatrix}$$

*Reconciliation Note on Prediction Flips:*
Between Condition A and Condition B for Ridge Classifier, predictions were identical for 78 out of 80 participants. Exactly **two** participant predictions flipped:
1. Participant `P0012` (True: A+): Condition A predicted A+ (correct); Condition B predicted O+ (incorrect).
2. Participant `P007` (True: O+): Condition A predicted B+ (incorrect); Condition B predicted O+ (correct).  
Because one correct prediction was lost and one was gained, total accuracy remained exactly 29/80 ($36.25\%$), but A+ recall dropped from $26.3\%$ to $21.1\%$ while O+ recall increased from $34.5\%$ to $37.9\%$.

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

---

## 5. Mandatory Research & Clinical Disclaimer

1. **Research Prototype Only:** This system is an investigational computational prototype developed on a small single-center development cohort ($N=80$).
2. **Zero Clinical Validity:** The models are not clinically validated and must never be used for blood group determination, clinical diagnosis, transfusion decisions, or therapeutic interventions.
3. **Serological Verification Notice:** Development cohort labels were self-reported by student participants and have not been validated by clinical hemagglutination records.
4. **Statistical Boundary:** Scan quality variation does not explain classification failure; optical friction ridge impressions do not support predictive blood-group determination.
