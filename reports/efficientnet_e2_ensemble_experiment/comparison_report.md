# Comprehensive Model Comparison Report: EfficientNet-B0 vs. E2 Fusion vs. Equal-Weight Ensemble

**Experiment Name:** Fair Comparison of EfficientNet-B0, E2 Champion Fusion, and Equal-Weight Ensemble  
**Date:** 2026-10-13  
**Target Architecture:** Multi-Cohort Evaluation on Mantra MFS100 Fingerprint Data  
**Framing:** AI-assisted biometric blood group pattern research prototype (strictly non-clinical; not medical validation).  
**Output Directory:** `reports/efficientnet_e2_ensemble_experiment/`

---

## 1. Executive Summary

This study conducted a rigorous, scientifically controlled comparison between:
1. **EfficientNet-B0:** Single-scan pretrained backbone with 3-scan probability averaging.
2. **E2 Champion Fusion:** 3-scan feature fusion network combining 5 cross-validation fold checkpoints.
3. **Equal-Weight Combined Ensemble:** Fixed arithmetic mean ($0.5 \times P_{\text{EffNet}} + 0.5 \times P_{\text{E2}}$) after canonical class-order remapping.

### Primary Experimental Findings:
1. **The Ensemble Does Not Outperform E2 on the Locked Test:**  
   On the quarantined 21-participant locked test set, E2 achieved **38.10%** (8/21), whereas the Equal-Weight Ensemble achieved **33.33%** (7/21). EfficientNet-B0 scored **28.57%** (6/21). Combining the models failed to improve upon E2 and performed below the trivial Always-O+ majority baseline (**38.10%**).
2. **The Ensemble Severely Degrades E2 Development Accuracy:**  
   On the 80 development participants, combining E2 (53.75% out-of-fold accuracy) with EfficientNet-B0 (32.50% zero-shot transfer accuracy) resulted in **38.75%** accuracy—introducing 23 new errors where an individual model was correct but the ensemble failed.
3. **Severe Cross-Domain Degradation on Descriptive MFS100 Cohort:**  
   On the separate 26-participant hospital-confirmed cohort, EfficientNet-B0 achieved **38.46%** (10/26), E2 suffered heavy prediction collapse to A+ achieving **19.23%** (5/26), and the ensemble reached **38.46%** (10/26). Neither model nor ensemble beat the Always-O+ baseline (**42.31%**).
4. **Conclusion:**  
   Combining a sensor-mismatched, unadapted model (EfficientNet-B0) with a specialized feature fusion model (E2) via a fixed equal-weight ensemble provides **no measurable generalization benefit** and consistently damages discrimination.

---

## 2. Historical Benchmarks & Methodological Separation

To preserve historical scientific integrity, previous benchmark figures are reported below alongside their strict evaluation protocols. They must **not** be conflated or presented as equivalent test results:

| Historical Metric | Architecture | Protocol & Dataset | Independent Test Accuracy? |
|:---|:---|:---|:---:|
| **86.02%** | EfficientNet-B0 | Internal image-level test set from the original ~6,000 CMOS optical scanner dataset (`train_abo_positive_4class.py`). | **NO** (Image-level, non-MFS100 domain). |
| **53.75%** | E2 Fusion | Out-of-fold (OOF) cross-validation on 80 MFS100 development participants (5 folds, participant-level). | **NO** (Development OOF estimate). |
| **38.10%** | E2 Fusion | Quarantined locked test evaluation on 21 MFS100 participants (3 scans per subject). | **YES** (Locked test benchmark). |
| **50.50%** | E2 Fusion | Pooled methodological aggregate across 80 development OOF and 21 locked test predictions ($51/101$). | **NO** (Pooled aggregate, not independent test). |

---

## 3. Evaluated Pipelines & Mathematical Formulation

### 3.1 EfficientNet-B0
- **Checkpoint:** `models/abo_positive_4class/best_model.keras` (SHA-256: `7b5ff5d9...`)
- **Input:** Single image resized to $224 \times 224 \times 3$, RGB float32, normalized via `tf.keras.applications.efficientnet.preprocess_input`.
- **Scan Aggregation:** For each participant, all 3 scans are inferred independently; output vectors are averaged:
  $$P_{\text{EffNet}} = \frac{1}{3} \sum_{s=1}^3 P(s)$$
- **Class Order:** Native order is canonical `[0: A+, 1: AB+, 2: B+, 3: O+]`.

### 3.2 E2 Champion Fusion
- **Checkpoints:** 5 fold weights `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5`.
- **Input:** Exactly 3 scans per subject processed through production `transform_c5` (composite on white, crop foreground, CLAHE + bilateral filter, resize to $112\times 112$). Input tensor shape: $(1, 3, 112, 112, 3)$.
- **Ensemble Aggregation:** Average of 5 fold softmax outputs:
  $$P_{\text{E2, native}} = \frac{1}{5} \sum_{k=1}^5 P_k(X)$$
- **Canonical Remapping:** E2 native output order is `[0: A+, 1: B+, 2: AB+, 3: O+]`. Column indices 1 and 2 are explicitly swapped to align with canonical order:
  $$P_{\text{E2, canonical}} = \begin{bmatrix} P_{\text{E2, native}}[0] & (\text{A+}) \\ P_{\text{E2, native}}[2] & (\text{AB+}) \\ P_{\text{E2, native}}[1] & (\text{B+}) \\ P_{\text{E2, native}}[3] & (\text{O+}) \end{bmatrix}$$

### 3.3 Equal-Weight Combined Ensemble
- **Formula:** Arithmetic mean of canonically aligned probability distributions:
  $$P_{\text{combined}} = 0.5 \times P_{\text{EffNet}} + 0.5 \times P_{\text{E2, canonical}}$$
- **Optimization Status:** Fixed 50:50 weighting. No weighting optimization was performed because EfficientNet-B0 lacked valid MFS100 development OOF checkpoints.

---

## 4. Empirical Evaluation Results by Cohort

### 4.1 Cohort 1: Development Cohort (80 Participants, MFS100)
*Status: Descriptive development analysis. E2 is 5-fold OOF; EfficientNet-B0 is zero-shot transfer.*

| Metric | EfficientNet-B0 (Zero-Shot) | E2 Fusion (5-Fold OOF) | Equal-Weight Ensemble | Always-O+ Baseline |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 32.50% (26/80) | **53.75%** (43/80) | 38.75% (31/80) | 36.25% (29/80) |
| **Balanced Accuracy** | 23.91% | **47.15%** | 31.64% | 25.00% |
| **Macro Precision** | 0.1870 | **0.6628** | 0.2995 | 0.0906 |
| **Macro Recall** | 0.2391 | **0.4715** | 0.3164 | 0.2500 |
| **Macro F1** | 0.1885 | **0.4772** | 0.2932 | 0.1330 |

**Per-Class Recall (Development Cohort):**
- **A+ (N=19):** EffNet = 15.79%, E2 = **63.16%**, Ensemble = 47.37%
- **AB+ (N=7):** EffNet = 0.00%, E2 = **14.29%**, Ensemble = 0.00%
- **B+ (N=25):** EffNet = 4.00%, E2 = **56.00%**, Ensemble = 24.00%
- **O+ (N=29):** EffNet = **75.86%**, E2 = 55.17%, Ensemble = 55.17%

---

### 4.2 Cohort 2: Locked Test Set (21 Participants, MFS100)
*Status: Pre-registered, quarantined, un-tuned final evaluation.*

| Metric | EfficientNet-B0 (Zero-Shot) | E2 Fusion (5-Fold Ensemble) | Equal-Weight Ensemble | Always-O+ Baseline |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 28.57% (6/21) | **38.10%** (8/21) | 33.33% (7/21) | **38.10%** (8/21) |
| **Balanced Accuracy** | 20.63% | **30.63%** | 23.75% | 25.00% |
| **Macro Precision** | 0.1333 | **0.2244** | 0.1438 | 0.0952 |
| **Macro Recall** | 0.2063 | **0.3063** | 0.2375 | 0.2500 |
| **Macro F1** | 0.1587 | **0.2500** | 0.1750 | 0.1379 |

**Per-Class Recall (Locked Test Set):**
- **A+ (N=5):** EffNet = 20.00% (1/5), E2 = **60.00%** (3/5), Ensemble = 20.00% (1/5)
- **AB+ (N=2):** EffNet = 0.00% (0/2), E2 = 0.00% (0/2), Ensemble = 0.00% (0/2)
- **B+ (N=6):** EffNet = 0.00% (0/6), E2 = 0.00% (0/6), Ensemble = 0.00% (0/6)
- **O+ (N=8):** EffNet = 62.50% (5/8), E2 = 62.50% (5/8), Ensemble = **75.00%** (6/8)

---

### 4.3 Cohort 3: Separate 26-Participant Cohort (Descriptive Evaluation)
*Status: Hospital serologically confirmed cohort (S001-S026). Descriptive evaluation only; not independent proof of generalizability.*

| Metric | EfficientNet-B0 (Zero-Shot) | E2 Fusion (5-Fold Ensemble) | Equal-Weight Ensemble | Always-O+ Baseline |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | **38.46%** (10/26) | 19.23% (5/26) | **38.46%** (10/26) | **42.31%** (11/26) |
| **Balanced Accuracy** | 29.64% | 20.17% | **31.53%** | 25.00% |
| **Macro Precision** | **0.3136** | 0.2336 | 0.2654 | 0.1058 |
| **Macro Recall** | 0.2964 | 0.2017 | **0.3153** | 0.2500 |
| **Macro F1** | **0.2788** | 0.1548 | 0.2778 | 0.1486 |

---

## 5. Participant Disagreement Analysis

From `reports/efficientnet_e2_ensemble_experiment/participant_disagreement_analysis.csv`:

| Cohort | Total Participants | Models Disagree | Ensemble Corrects EffNet | Ensemble Corrects E2 | Ensemble Introduces New Error |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Development** (80) | 80 | 52 (65.0%) | 13 | 3 | **23** |
| **Locked Test** (21) | 21 | 15 (71.4%) | 1 | 4 | **5** |
| **Separate Cohort** (26) | 26 | 20 (76.9%) | 2 | 6 | **3** |
| **Total Across All Cohorts** | **127** | **87 (68.5%)** | **16** | **13** | **31** |

### Key Takeaway:
In 31 out of 127 cases across all cohorts, at least one of the individual models had the correct prediction, but equal-weight ensembling corrupted the result into an incorrect prediction. The number of errors introduced (31) significantly exceeded the corrections made to either model (16 for EffNet, 13 for E2).

---

## 6. Answers to Core Research Questions

1. **Did the equal-weight ensemble outperform EfficientNet-B0 on the same participants?**  
   - On the Development cohort: Yes (38.75% vs. 32.50%).
   - On the Locked Test cohort: Yes (33.33% vs. 28.57%).
   - On the Separate 26 cohort: Tied (38.46% vs. 38.46%).
2. **Did it outperform E2?**  
   - **NO.** On the Development cohort, it degraded performance by **-15.00%** (38.75% vs. 53.75%).
   - **NO.** On the Locked Test cohort, it degraded performance by **-4.77%** (33.33% vs. 38.10%).
   - Only on the 26 cohort (where E2 collapsed to 19.23%) did the ensemble outperform E2, but only by matching EfficientNet-B0.
3. **Did balanced accuracy or macro F1 improve?**  
   - Relative to E2 on the Locked Test, Balanced Accuracy fell from **30.63% to 23.75%**, and Macro F1 fell from **0.2500 to 0.1750**.
4. **Which blood groups were most frequently misclassified?**  
   - Across all cohorts and models, **AB+** suffered near-total recall collapse (0.0% recall on the locked test for all three pipelines). **B+** was completely missed on the locked test (0.0% recall). Models heavily biased toward **O+** (EffNet) and **A+** (E2).
5. **Was the ensemble comparison conducted without tuning on the locked test?**  
   - **YES.** The 50:50 ensemble weights and canonical remapping were locked prior to evaluation. No grid search, threshold tuning, or weight fitting was performed on the locked test.

---

## 7. Research Integrity Statement

This study constitutes an engineering audit and research benchmark of statistical machine learning algorithms. **Under no circumstances should these results be interpreted as clinical, medical, or diagnostic validation.** Biometric fingerprint images have not demonstrated sufficient biological discrimination to serve as a reliable blood-group testing method.
