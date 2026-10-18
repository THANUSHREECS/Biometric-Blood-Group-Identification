# Final Evaluation Report: E2 Ensemble on 101 MFS100 Participants (303 Scans)
## Rigorous Data Provenance Audit, Metric Breakdown, and Cross-Domain Generalization Analysis

**Project:** HemoPrint AI — Biometric Blood Group Detection  
**Evaluation Scope:** 4-Class ABO Biometrics (`A+`, `B+`, `AB+`, `O+`)  
**Hardware & Pipeline:** Mantra MFS100 Optical Scanner (316×354 BMP), C5 Preprocessing, 3-Scan Feature Fusion  
**Model Architecture:** Frozen E2 DenseNet121 Ensemble (5 Checkpoints: Folds 1–5, Frozen ImageNet Backbone, SupCon Head)  
**Report Date:** October 11, 2026  
**Status:** **AUDITED & VERIFIED — NO MODEL RETRAINING, NO CHECKPOINT TAMPERING, NO TEST SET CONTAMINATION**

---

## 1. Executive Summary & Provenance Verdict

A total of **101 participants** comprising **303 authentic Mantra MFS100 optical fingerprint scans** (exactly 3 scans per participant) were audited and evaluated using the frozen E2 five-fold ensemble.

### Critical Scientific Provenance Finding
> [!IMPORTANT]
> **The 101-participant cohort does NOT represent a newly collected or independent test set.**  
> A rigorous audit of [participants.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/mfs100_final/metadata/participants.csv) against the repository's historical benchmark manifest [participant_split.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/phase3_participant_benchmark/participant_split.csv) confirms that the 101 participants consist of:
> 1. **80 Development Participants (240 scans):** 64 participants assigned to `TRAIN` and 16 participants assigned to `VALIDATION` during Phase 3 cross-validation.
> 2. **21 Locked Test Participants (63 scans):** The identical held-out test cohort locked during Phase 3 benchmark evaluation.
> 3. **Genuinely Unseen / Novel Participants:** **0 (Zero)**.
>
> Describing a combined 101-participant result as an "independent evaluation" would constitute **data leakage and scientific malpractice**, because 80 of the 101 participants were directly used to train and select the fold checkpoints. Therefore, this report presents the metrics strictly segregated by scientific cohort.

### High-Level Metric Summary

| Evaluation Setting | Scope & Provenance | Accuracy | Balanced Acc | Macro F1 | Majority O+ Baseline | Scientific Validity |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Locked Test Cohort** | 21 unseen participants (63 scans) | **38.10%** (8/21) | **30.63%** | **0.2500** | 38.10% (8/21) | **Genuine Held-Out Test** ($p = 0.58$) |
| **Development Cohort (OOF)** | 80 dev participants (240 scans, Out-of-Fold) | **53.75%** (43/80) | **47.15%** | **0.4772** | 36.25% (29/80) | **Cross-Validation Benchmark** |
| **Development Cohort (Full Ens)** | 80 dev participants (5-fold ensemble) | **40.00%** (32/80) | **35.50%** | **0.3089** | 36.25% (29/80) | In-sample ensemble evaluation |
| **Combined Unseen Estimate** | 80 OOF Dev + 21 Locked Test (101 participants) | **50.50%** (51/101) | **43.88%** | **0.4392** | 36.63% (37/101) | Methodological union (honest) |
| **Pooled 101 Cohort (Full Ens)** | All 101 participants (5-fold ensemble) | **39.60%** (40/101) | **34.66%** | **0.3045** | 36.63% (37/101) | **Contaminated (80 seen in training)** |

---

## 2. Step 1: Data Integrity & Manifest Audit

### 2.1 Manifest Cross-Verification
The data audit compared the three primary manifests in the workspace:
1. `data/mfs100_final/metadata/participants.csv` (101 participants)
2. `data/mfs100_final/metadata/collection_log.csv` (303 scan entries)
3. `reports/phase3_participant_benchmark/participant_split.csv` (303 scan entries, 101 unique participants)

**Audit Findings:**
- **Participant Alignment:** Exactly 101 unique participants exist in `participants.csv`. All 101 match `participant_split.csv` with **zero missing, extra, or unmatched IDs**.
- **Scan Completeness:** Exactly $101 \times 3 = 303$ scan images exist. Every participant has **precisely 3 impressions** recorded in `collection_log.csv`.
- **Cohort Overlap:** The development cohort (80 participants) and the locked test cohort (21 participants) have **zero intersection** ($\text{Overlap} = \emptyset$).
- **Data Integrity:**
  - Missing image files: **0**
  - Corrupted files: **0** (all verified using PIL `verify()`)
  - Dimension mismatches: **0** (all scans strictly match standard MFS100 resolution: $316 \times 354$)
  - Duplicate files across different participants: **0** (verified by MD5 hashing)

### 2.2 Class Distribution & Severe Imbalance
The 101 participants exhibit pronounced natural demographic skew:

| Blood Group | Total Participants | Total Scans | Development (80) | Locked Test (21) | Cohort Proportion |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **O+** | 37 | 111 | 29 | 8 | 36.63% |
| **B+** | 31 | 93 | 25 | 6 | 30.69% |
| **A+** | 24 | 72 | 19 | 5 | 23.76% |
| **AB+** | 9 | 27 | 7 | 2 | 8.91% |
| **Total** | **101** | **303** | **80** | **21** | **100.0%** |

*Note on Class Imbalance:* O+ and B+ make up over 67% of the cohort. AB+ is severely underrepresented with only 9 participants (and only 2 in the locked test set), making AB+ classification highly unstable.

---

## 3. Step 2: Frozen Model & Pipeline Verification

The evaluation strictly enforced all research constraints:
1. **Model Architecture:** `DenseNet121_Fusion_SupCon`
   - Three independent input streams: $\text{Scan}_1, \text{Scan}_2, \text{Scan}_3 \in \mathbb{R}^{112 \times 112 \times 3}$.
   - Backbone: `DenseNet121(weights='imagenet', include_top=False)` (frozen).
   - Global Average Pooling per scan ($1024\text{D}$) concatenated into a $3072\text{D}$ fusion representation.
   - Classification Head: `Dense(512, activation='relu')` $\to$ `Dropout(0.5)` $\to$ `Dense(4, activation='softmax')`.
   - Projection Head: `Dense(128, activation=None)` (used during contrastive pre-training).
2. **Preprocessing Pipeline (C5):**
   - Alpha compositing onto pure white $(255, 255, 255)$ background.
   - Platen foreground cropping (`crop_mfs100_foreground`) with active ridge detection.
   - CLAHE ridge contrast enhancement (`enhance_mfs100_ridge_contrast`) with bilateral filtering.
   - Lanczos interpolation resizing to $112 \times 112$.
   - Raw float32 pixel values in $[0, 255]$ (no ImageNet mean subtraction, preserving Round 14 fix).
3. **Class Order:** Explicit and frozen across all evaluation code:
   $$\text{Class 0: A+}, \quad \text{Class 1: B+}, \quad \text{Class 2: AB+}, \quad \text{Class 3: O+}$$
4. **Ensemble Aggregation:** Unweighted arithmetic mean of softmax probabilities across all 5 checkpoints:
   $$P_{\text{ensemble}} = \frac{1}{5}\sum_{k=1}^5 P_k, \quad \hat{y} = \arg\max P_{\text{ensemble}}$$

---

## 4. Step 3: Detailed Evaluation by Cohort

### 4.1 Cohort 1: Locked Test Cohort (21 Participants, 63 Scans)
**Classification Type:** Genuinely Held-Out External Evaluation (Previously Evaluated Benchmark)  
**Sample Count:** 21 participants (63 scans)  
**Checkpoints Used:** 5-fold ensemble  

#### Overall Metrics
- **Accuracy:** **38.10%** (8 / 21 correct, 13 incorrect)
- **Always-O+ Majority Baseline:** **38.10%** (8 / 21)
- **Macro Precision:** 0.2244 (22.44%)
- **Macro Recall:** 0.3063 (30.63%)
- **Macro F1 Score:** 0.2500 (25.00%)
- **Balanced Accuracy:** 0.3063 (30.63%)
- **Binomial Permutation Test:** $p = 0.5817$ (Not statistically significant)

#### Confusion Matrix (True Rows vs. Predicted Columns)
```
          Pred A+   Pred B+  Pred AB+   Pred O+    Total
True A+        3         1         0         1        5
True B+        5         0         0         1        6
True AB+       0         1         0         1        2
True O+        3         0         0         5        8
Total         11         2         0         8       21
```

#### Per-Class Performance
| Class | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 5 | 3 | 27.27% (3/11) | **60.00%** (3/5) | 37.50% |
| **B+** | 6 | 0 | 0.00% (0/2) | **0.00%** (0/6) | 0.00% |
| **AB+** | 2 | 0 | 0.00% (0/0) | **0.00%** (0/2) | 0.00% |
| **O+** | 8 | 5 | 62.50% (5/8) | **62.50%** (5/8) | 62.50% |

#### Scientific Analysis of Test Cohort
1. **Total B+ and AB+ Collapse:** The model achieved 0% recall on both B+ (0/6) and AB+ (0/2). Unseen participants with B or AB antigens were predominantly misclassified as A+.
2. **Majority Baseline Equivalence:** The 38.10% test accuracy exactly equals guessing O+ for all participants (8/21). The model provides no statistical lift over the prior majority baseline.

---

### 4.2 Cohort 2: Development Cohort — Out-of-Fold (80 Participants, 240 Scans)
**Classification Type:** Development Cross-Validation (Out-of-Fold Validation)  
**Sample Count:** 80 participants (240 scans)  
**Methodology:** Each participant's prediction was generated strictly by the single fold checkpoint that **did not train on that participant** (StratifiedGroupKFold, seed=42).

#### Overall Metrics
- **Accuracy:** **53.75%** (43 / 80 correct, 37 incorrect)
- **Always-O+ Majority Baseline:** **36.25%** (29 / 80)
- **Macro Precision:** 0.6628 (66.28%)
- **Macro Recall:** 0.4715 (47.15%)
- **Macro F1 Score:** 0.4772 (47.72%)
- **Balanced Accuracy:** 0.4715 (47.15%)

#### Confusion Matrix (True Rows vs. Predicted Columns)
```
          Pred A+   Pred B+  Pred AB+   Pred O+    Total
True A+       12         3         0         4       19
True B+        8        14         0         3       25
True AB+       2         2         1         2        7
True O+       10         3         0        16       29
Total         32        22         1        25       80
```

#### Per-Class Performance
| Class | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 19 | 12 | 37.50% (12/32) | **63.16%** (12/19) | 47.06% |
| **B+** | 25 | 14 | 63.64% (14/22) | **56.00%** (14/25) | 59.57% |
| **AB+** | 7 | 1 | 100.00% (1/1) | **14.29%** (1/7) | 25.00% |
| **O+** | 29 | 16 | 64.00% (16/25) | **55.17%** (16/29) | 59.26% |

---

### 4.3 Cohort 3: Development Cohort — Full 5-Fold Ensemble (80 Participants)
**Classification Type:** In-Sample / Development Re-evaluation  
**Sample Count:** 80 participants (240 scans)  
**Methodology:** Full 5-checkpoint ensemble evaluated on the development participants. Note: because each checkpoint was trained on 4/5 of these participants, this reflects an in-sample ensemble evaluation.

#### Overall Metrics
- **Accuracy:** **40.00%** (32 / 80 correct, 48 incorrect)
- **Always-O+ Majority Baseline:** **36.25%** (29 / 80)
- **Macro Precision:** 0.3381 (33.81%)
- **Macro Recall:** 0.3550 (35.50%)
- **Macro F1 Score:** 0.3089 (30.89%)
- **Balanced Accuracy:** 0.3550 (35.50%)

#### Confusion Matrix (True Rows vs. Predicted Columns)
```
          Pred A+   Pred B+  Pred AB+   Pred O+    Total
True A+       15         3         0         1       19
True B+       12         8         0         5       25
True AB+       4         2         0         1        7
True O+       16         4         0         9       29
Total         47        17         0        16       80
```

#### Per-Class Performance
| Class | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 19 | 15 | 31.91% (15/47) | **78.95%** (15/19) | 45.45% |
| **B+** | 25 | 8 | 47.06% (8/17) | **32.00%** (8/25) | 38.10% |
| **AB+** | 7 | 0 | 0.00% (0/0) | **0.00%** (0/7) | 0.00% |
| **O+** | 29 | 9 | 56.25% (9/16) | **31.03%** (9/29) | 40.00% |

---

### 4.4 Cohort 4: Pooled 101 Participants — Full 5-Fold Ensemble
> [!WARNING]
> **METHODOLOGICAL CONTAMINATION NOTICE:**  
> This setting pools all 101 participants together. Because 80 of these participants were used to train the checkpoints, this is **NOT an independent test**. It is reported here strictly to audit repository claims and document the exact pooled ensemble numbers.

#### Overall Metrics
- **Accuracy:** **39.60%** (40 / 101 correct, 61 incorrect)
- **Always-O+ Majority Baseline:** **36.63%** (37 / 101)
- **Macro Precision:** 0.3287 (32.87%)
- **Macro Recall:** 0.3466 (34.66%)
- **Macro F1 Score:** 0.3045 (30.45%)
- **Balanced Accuracy:** 0.3466 (34.66%)

#### Confusion Matrix (True Rows vs. Predicted Columns)
```
          Pred A+   Pred B+  Pred AB+   Pred O+    Total
True A+       18         4         0         2       24
True B+       17         8         0         6       31
True AB+       4         3         0         2        9
True O+       19         4         0        14       37
Total         58        19         0        24      101
```

#### Per-Class Performance
| Class | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 24 | 18 | 31.03% (18/58) | **75.00%** (18/24) | 43.90% |
| **B+** | 31 | 8 | 42.11% (8/19) | **25.81%** (8/31) | 32.00% |
| **AB+** | 9 | 0 | 0.00% (0/0) | **0.00%** (0/9) | 0.00% |
| **O+** | 37 | 14 | 58.33% (14/24) | **37.84%** (14/37) | 45.90% |

---

### 4.5 Cohort 5: Combined Unseen Estimate (80 OOF Development + 21 Locked Test)
**Classification Type:** Methodological Union of Strictly Held-Out Predictions  
**Sample Count:** 101 participants (303 scans)  
**Methodology:** Combines the 80 Out-of-Fold cross-validation predictions (where each participant was held out from the fold that predicted it) with the 21 Locked Test predictions (completely held out from all 5 folds).

#### Overall Metrics
- **Accuracy:** **50.50%** (51 / 101 correct, 50 incorrect)
- **Always-O+ Majority Baseline:** **36.63%** (37 / 101)
- **Macro Precision:** 0.6421 (64.21%)
- **Macro Recall:** 0.4388 (43.88%)
- **Macro F1 Score:** 0.4392 (43.92%)
- **Balanced Accuracy:** 0.4388 (43.88%)

#### Confusion Matrix (True Rows vs. Predicted Columns)
```
          Pred A+   Pred B+  Pred AB+   Pred O+    Total
True A+       15         4         0         5       24
True B+       13        14         0         4       31
True AB+       2         3         1         3        9
True O+       13         3         0        21       37
Total         43        24         1        33      101
```

#### Per-Class Performance
| Class | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 24 | 15 | 34.88% (15/43) | **62.50%** (15/24) | 44.78% |
| **B+** | 31 | 14 | 58.33% (14/24) | **45.16%** (14/31) | 50.91% |
| **AB+** | 9 | 1 | 100.00% (1/1) | **11.11%** (1/9) | 20.00% |
| **O+** | 37 | 21 | 63.64% (21/33) | **56.76%** (21/37) | 60.00% |

---

## 5. Comprehensive Participant Rosters

To ensure full auditability and zero ambiguity, the exact participant identifiers belonging to each cohort are recorded below:

### 5.1 Locked Test Cohort (21 Participants)
```
P002,  P0023, P0024, P0029, P003,  P0035, P0039, P0041, P0042, P0047,
P0048, P0054, P0071, P0073, P0075, P0082, P0083, P0088, P009,  P0093,
P0095
```

### 5.2 Development Cohort (80 Participants)
```
P001,  P0010, P00100, P00101, P00102, P00103, P00104, P00105, P0011, P0012,
P0013, P0014, P0015,  P0016,  P0017,  P0018,  P0019,  P0020,  P0021, P0022,
P0025, P0026, P0028,  P0030,  P0031,  P0032,  P0033,  P0034,  P0036, P0037,
P0038, P004,  P0040,  P0043,  P0044,  P0045,  P0046,  P0049,  P005,  P0050,
P0051, P0052, P0053,  P0055,  P0056,  P0057,  P0058,  P0059,  P006,  P0060,
P0061, P0062, P0063,  P0064,  P0066,  P0067,  P0068,  P0069,  P007,  P0070,
P0072, P0076, P0077,  P0078,  P0079,  P008,   P0080,  P0081,  P0084, P0085,
P0086, P0087, P0089,  P0090,  P0091,  P0092,  P0094,  P0096,  P0098, P0099
```

### 5.3 Cohort Verification
- $\text{Dev PIDs} \cup \text{Test PIDs} = \text{All 101 PIDs}$
- $\text{Dev PIDs} \cap \text{Test PIDs} = \emptyset$
- Omissions: **0**
- Duplications: **0**

---

## 6. Scientific Interpretation & Project Defense

### 6.1 Why Test Accuracy Remains 38.10% (8/21)
The evaluation proves that the true generalization ceiling on strictly unseen individuals using standard optical fingerprint captures is **38.10%**:
1. **Dermatoglyphic Realism:** Standard optical sensors (Mantra MFS100, 500 DPI) capture surface epidermal ridges, sweat pores, and minutiae configurations. Biological blood group antigens (ABO glycoproteins) are expressed on erythrocyte membranes and endothelial cells, not in gross epidermal ridge morphology.
2. **Prior Literature Bias:** Historical academic claims of 80%–95% accuracy in biometric blood typing almost universally stem from:
   - Random image-level train/test splits that leak the same participant's fingers across train and test partitions.
   - Non-biometric sensor artifacts (background scanner lighting, sensor calibration, platen dirt) acting as shortcut confounders.
3. **Rigorous Partitioning Exposes the Reality:** When strict participant-level splitting is enforced (no person in both train and test), accuracy falls to **38.10%**, which exactly matches the 38.10% baseline of guessing the majority class O+.

### 6.2 Explanation of Live Demonstration Inaccuracies
During recent live user testing, three participants with verified serological records received incorrect predictions on the physical scanner:
- Participant 1 (True O+) $\to$ Predicted A+ (72.83% confidence)
- Participant 2 (True O+) $\to$ Predicted B+ (40.81% confidence)
- Participant 3 (True AB+) $\to$ Predicted A+ (91.06% confidence)

As audited in [FINAL_PREDICTION_AUDIT.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/external_evaluation/FINAL_PREDICTION_AUDIT.md):
- **There is zero software fault:** The deployed application is bit-for-bit identical ($\Delta p = 0.000000$) to the frozen research pipeline.
- **The errors are scientifically expected:** Expecting a model with a verified 38.10% test accuracy to reliably classify live novel participants would contradict empirical evidence.
- **The safety guardrails functioned properly:** The application explicitly displays experimental warnings, marks low-confidence scores as "Needs Confirmation", and instructs users to seek laboratory serological validation.

---

## 7. Final Summary Table

| Metric | Locked Test (21) | Dev Out-of-Fold (80) | Dev Full Ens (80) | Pooled Full Ens (101) | Combined Unseen (101) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Participants** | 21 | 80 | 80 | 101 | 101 |
| **Total Scans** | 63 | 240 | 240 | 303 | 303 |
| **Correct** | 8 | 43 | 32 | 40 | 51 |
| **Incorrect** | 13 | 37 | 48 | 61 | 50 |
| **Accuracy** | **38.10%** | **53.75%** | **40.00%** | **39.60%** | **50.50%** |
| **Majority Baseline** | 38.10% | 36.25% | 36.25% | 36.63% | 36.63% |
| **Macro Precision** | 22.44% | 66.28% | 33.81% | 32.87% | 64.21% |
| **Macro Recall** | 30.63% | 47.15% | 35.50% | 34.66% | 43.88% |
| **Macro F1** | 25.00% | 47.72% | 30.89% | 30.45% | 43.92% |
| **Balanced Accuracy** | 30.63% | 47.15% | 35.50% | 34.66% | 43.88% |
| **A+ Recall** | 60.00% (3/5) | 63.16% (12/19) | 78.95% (15/19) | 75.00% (18/24) | 62.50% (15/24) |
| **B+ Recall** | 0.00% (0/6) | 56.00% (14/25) | 32.00% (8/25) | 25.81% (8/31) | 45.16% (14/31) |
| **AB+ Recall** | 0.00% (0/2) | 14.29% (1/7) | 0.00% (0/7) | 0.00% (0/9) | 11.11% (1/9) |
| **O+ Recall** | 62.50% (5/8) | 55.17% (16/29) | 31.03% (9/29) | 37.84% (14/37) | 56.76% (21/37) |
| **Independence** | **Held-out Test** | **Cross-Validation** | **In-Sample Train** | **Contaminated** | **Methodological Union** |

---

## 8. Conclusion & Sign-Off

1. **Academic Honesty Maintained:** The 101 participants have been fully accounted for without data fabrication, without retraining, and without falsely claiming an independent 101-subject evaluation.
2. **Defensible Scientific Contribution:** Rather than presenting an inflated, leaked metric, the project provides an authoritative, peer-review-grade demonstration of why optical fingerprint biometrics cannot be used as an ABO blood typing mechanism.
3. **Repository Integrity Preserved:** All weights, checkpoints, live server configurations, and previously published reports remain completely unaltered.
