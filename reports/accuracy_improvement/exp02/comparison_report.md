# EXP-02: Controlled Augmentation vs. Baseline E2 Comparison Report

**Experiment:** EXP-02 — Controlled Data Augmentation for E2  
**Date:** 2026-10-14T12:54:31.253159  
**Dataset:** 80 Development Participants (240 scans) from `participant_split.csv`  
**Protocol:** 5-Fold Stratified Group Cross-Validation (True Out-Of-Fold Evaluation)  
**Status:** **REJECTED**  

---

## 1. Executive Summary

This experiment investigated whether applying **conservative, fingerprint-appropriate data augmentation** during training improves the out-of-fold generalization of the E2 3-scan feature fusion architecture. The augmentation pipeline introduced small platen rotations (+/- 7.2°), subtle translations (+/- 4.5 px), and mild illumination/contrast shifts (+/- 5%) while strictly preserving ridge spatial frequency (zero zoom).

### Key Finding:
- **Accuracy:** Changed from **53.75%** to **47.50%** (-6.25%).
- **Macro F1:** Changed from **0.4772** to **0.4632** (-0.0140).
- **Balanced Accuracy:** Changed from **47.15%** to **43.60%** (-3.55%).
- **AB+ Recall:** Changed from **14.29%** (1/7) to **28.57%** (2/7) (+14.28%).
- **B+ Recall:** Changed from **56.00%** (14/25) to **52.00%** (13/25) (-4.00%).

---

## 2. Side-by-Side Performance Comparison (N=80 Development OOF)

| Metric | Original Baseline E2 (Cross-Entropy) | EXP-02 E2 (Controlled Data Augmentation) | Absolute Change |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **53.75%** (43/80) | **47.50%** (38/80) | **-6.25%** |
| **Balanced Accuracy** | 47.15% | 43.60% | -3.55% |
| **Macro F1-Score** | 0.4772 | 0.4632 | -0.0140 |
| **Macro Precision** | 0.6628 | 0.5906 | -0.0722 |
| **Macro Recall** | 0.4715 | 0.4360 | -0.0355 |
| **Always-O+ Baseline** | 36.25% (29/80) | 36.25% (29/80) | 0.00% |

---

## 3. Per-Class Breakdown (Canonical Order: A+, AB+, B+, O+)

| Class | Support | Baseline Recall | EXP-02 Recall | Baseline F1 | EXP-02 F1 | Recall Change |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A+** | 19 | 63.16% (12/19) | 42.11% | 0.4706 | 0.4000 | -21.05% |
| **AB+** | 7 | 14.29% (1/7) | 28.57% | 0.2500 | 0.4444 | +14.28% |
| **B+** | 25 | 56.00% (14/25) | 52.00% | 0.5957 | 0.5000 | -4.00% |
| **O+** | 29 | 55.17% (16/29) | 51.72% | 0.5926 | 0.5085 | -3.45% |

---

## 4. Confusion Matrices (Rows = True, Columns = Predicted)

### Baseline E2 Confusion Matrix:
$$\begin{pmatrix}
12 & 0 & 3 & 4 \\
2 & 1 & 2 & 2 \\
8 & 0 & 14 & 3 \\
10 & 0 & 3 & 16
\end{pmatrix}$$

### EXP-02 Controlled Augmentation Confusion Matrix:
$$\begin{pmatrix}
8 & 0 & 5 & 6 \\
1 & 2 & 2 & 2 \\
5 & 0 & 13 & 7 \\
7 & 0 & 7 & 15
\end{pmatrix}$$

---

## 5. Experiment Evaluation: Acceptance Criteria

| Criterion | Target | Baseline | EXP-02 Result | Status |
|:---|:---:|:---:|:---:|:---:|
| **Macro F1 Improvement** | $\ge +0.05$ over baseline ($\ge 0.5272$) | 0.4772 | 0.4632 (-0.0140) | **FAIL** |
| **Balanced Accuracy Improvement** | $> 47.15\%$ | 47.15% | 43.60% (-3.55%) | **FAIL** |
| **Minority (AB+) Recall Improvement** | $> 14.29\%$ | 14.29% | 28.57% (+14.28%) | **PASS** |
| **Overall Integrity Audit** | 100% Pass | N/A | All 9 Checks Passed | **PASS** |
| **FINAL EXPERIMENT VERDICT** | All 3 targets met & integrity passed | — | — | **REJECTED** |

---

## 6. Analysis & Scientific Interpretation

1. **Impact on Ridge Representation & Feature Invariance:**
   - Conservative rotation and translation allow the frozen DenseNet-121 backbone embeddings to maintain positional invariance against minor platen alignment variances.
   - Restricting zoom to zero prevented ridge period distortion, ensuring the feature extraction did not introduce synthetic biometric anomalies.
2. **Minority Class Performance (AB+ and B+):**
   - The minority class AB+ has only 7 subjects in the development set. Any movement of a single participant alters recall by 14.29%. The observed performance must be interpreted as a development indicator rather than definitive clinical proof.
3. **Locked Test Status:**
   - In accordance with the project quarantine protocol, the 21-participant locked test set was **NOT evaluated** in this experiment to protect against multiple-testing bias.
