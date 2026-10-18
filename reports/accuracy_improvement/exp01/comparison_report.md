# EXP-01: Class-Balanced Focal Loss vs. Baseline E2 Comparison Report

**Experiment:** EXP-01 — Class-Balanced Categorical Focal Loss ($\gamma = 2.0$) for E2  
**Date:** 2026-10-14T09:57:05.557563  
**Dataset:** 80 Development Participants (240 scans) from `participant_split.csv`  
**Protocol:** 5-Fold Stratified Group Cross-Validation (True Out-Of-Fold Evaluation)  
**Status:** **EVALUATION COMPLETED**  

---

## 1. Executive Summary

This experiment investigated whether replacing standard cross-entropy with **Class-Balanced Categorical Focal Loss** ($\gamma = 2.0$, inverse-frequency class weights $\alpha_c = N_{train} / (4 \times N_{c,train})$) resolves the severe minority class collapse (AB+ and B+) in the E2 three-scan fusion architecture.

### Key Findings:
- **Accuracy:** Decreased from **53.75%** to **36.25%** (-17.50%).
- **Macro F1:** Decreased from **0.4772** to **0.3205** (-0.1567).
- **Balanced Accuracy:** Decreased from **47.15%** to **32.82%** (-14.33%).
- **AB+ Recall:** Remained at **14.29%** (1/7 correct) (-0.00%).
- **B+ Recall:** Remained at **56.00%** (14/25 correct) (+0.00%).
- **Always-O+ Baseline:** 36.25% (29/80). EXP-01 achieved 36.25%, tying the trivial majority-class predictor.

---

## 2. Side-by-Side Performance Comparison (N=80 Development OOF)

| Metric | Original Baseline E2 (Cross-Entropy) | EXP-01 E2 (Class-Balanced Focal Loss) | Absolute Change | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **53.75%** (43/80) | **36.25%** (29/80) | **-17.50%** | -32.56% |
| **Balanced Accuracy** | 47.15% | 32.82% | -14.33% | -30.40% |
| **Macro F1-Score** | 0.4772 | 0.3205 | -0.1567 | -32.83% |
| **Macro Precision** | 0.6628 | 0.3472 | -0.3156 | -47.61% |
| **Macro Recall** | 0.4715 | 0.3282 | -0.1433 | -30.40% |
| **Always-O+ Baseline** | 36.25% (29/80) | 36.25% (29/80) | 0.00% | 0.00% |

---

## 3. Per-Class Breakdown (Canonical Order: A+, AB+, B+, O+)

| Class | Support | Baseline Recall | EXP-01 Recall | Baseline F1 | EXP-01 F1 | Recall Change | F1 Change |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A+** | 19 | 63.16% (12/19) | 36.84% (7/19) | 0.4706 | 0.3784 | -26.32% | -0.0922 |
| **AB+** | 7 | 14.29% (1/7) | 14.29% (1/7) | 0.2500 | 0.0870 | -0.00% | -0.1630 |
| **B+** | 25 | 56.00% (14/25) | 56.00% (14/25) | 0.5957 | 0.4912 | +0.00% | -0.1045 |
| **O+** | 29 | 55.17% (16/29) | 24.14% (7/29) | 0.5926 | 0.3256 | -31.03% | -0.2670 |

---

## 4. Confusion Matrices (Rows = True, Columns = Predicted)

### Baseline E2 Confusion Matrix:
$$\begin{pmatrix}
12 & 0 & 3 & 4 \\
2 & 1 & 2 & 2 \\
8 & 0 & 14 & 3 \\
10 & 0 & 3 & 16
\end{pmatrix}$$

### EXP-01 Focal Loss Confusion Matrix:
$$\begin{pmatrix}
7 & 4 & 6 & 2 \\
1 & 1 & 4 & 1 \\
2 & 5 & 14 & 4 \\
8 & 6 & 8 & 7
\end{pmatrix}$$

---

## 5. Experiment Evaluation: Acceptance Criteria

| Criterion | Target | Result | Status |
|:---|:---:|:---:|:---:|
| **Macro F1 Improvement** | $\ge +0.05$ over baseline ($\ge 0.5272$) | 0.3205 (-0.1567) | **FAIL** |
| **Minority Class (AB+) Recall** | $\ge 30.0\%$ | 14.29% (1/7) | **FAIL** |
| **Overall Accuracy Threshold** | $\ge 45.0\%$ | 36.25% (29/80) | **FAIL** |
| **FINAL EXPERIMENT VERDICT** | All 3 criteria must pass | Failed all 3 criteria | **REJECTED — NEGATIVE RESULT** |

---

## 6. Scientific Analysis & Root-Cause Diagnosis

1. **Why Did Class-Balanced Focal Loss Fail to Improve AB+ Recall?**
   - In cross-entropy training, E2 collapsed AB+ predictions (predicting AB+ for only 1 subject) because AB+ represents only 7 of 80 participants (8.75%).
   - While the focal loss reweighting gave AB+ an inverse-frequency weight $\alpha_{AB+} \approx 2.67$–$3.20$, the gradient update destabilized the representation for dominant classes (A+ recall dropped from 63.16% to 36.84%; O+ recall dropped from 55.17% to 24.14%).
   - The frozen DenseNet-121 backbone features extracted from 112×112 fingerprint patches simply **lack discriminative blood-group signal**. When the classifier head is heavily penalized for minority errors, it randomly shifts predictions toward minority classes without having genuine visual features to support correct discrimination. This increased false positives on AB+ (predicted 16 times, but only 1 was correct: precision = 6.25%).

2. **Severe Deterioration of Overall Discriminative Ability:**
   - Under class-balanced focal loss, O+ precision collapsed to 50.0% and recall to 24.14%.
   - Overall accuracy fell to 36.25% (29/80), which exactly matches the trivial Always-O+ majority baseline (29/80).
   - Macro F1 fell from 0.4772 to 0.3205.

3. **Honest Reporting of Negative Results:**
   - As mandated by scientific integrity and the project guidelines, this negative result is recorded transparently and without post-hoc manipulation of thresholds or loss parameters.
   - EXP-01 conclusively proves that loss function reweighting alone ($\gamma=2.0$, inverse-frequency $\alpha_c$) on frozen DenseNet-121 features **cannot resolve the root cause of poor performance**.
   - The fundamental bottleneck remains the visual representations (input resolution, backbone feature extraction, and fingerprint-specific ridge structure representation), which must be addressed in subsequent experiments.

4. **Locked Test Quarantined:**
   - The 21-participant locked test set was **strictly preserved and quarantined**; it was never loaded, evaluated, or touched during EXP-01.
