# Prediction Error & Failure Mode Analysis

**Investigation:** Accuracy Improvement — Root-Cause Diagnostic  
**Date:** 2026-10-13  
**Status:** COMPLETE & VERIFIED  
**Output Path:** `reports/accuracy_improvement/error_analysis.md`  

---

## 1. Executive Summary

This error analysis examines the predictions from the baseline evaluations across the **Development Cohort (80 participants)**, the **Locked Test Set (21 participants)**, and the **Separate Descriptive Cohort (26 participants)** ($N=127$ total participants).

### Primary Failure Modes Identified:
1. **Severe Class Collapse on AB+ and B+:**  
   Across all models on the locked test set, recall for **AB+** is **0.0%** ($0/2$) and recall for **B+** is **0.0%** ($0/6$). E2 and EfficientNet completely failed to recognize true B+ and AB+ individuals on the locked test.
2. **Extreme Model-Specific Prior Collapses:**  
   - **EfficientNet-B0 collapses to O+:** Predicts O+ for 73.8% of development participants (59/80) and 71.4% of locked test participants (15/21).
   - **E2 collapses to A+ on cross-domain data:** Predicts A+ for 52.4% of locked test participants (11/21) and 61.5% of the separate 26-cohort (16/26).
3. **Scan-to-Scan Variance:**  
   Within-participant scan predictions in EfficientNet-B0 frequently contradict each other (scan 1 predicting A+, scan 2 predicting O+, scan 3 predicting B+), indicating lack of feature stability.

---

## 2. Quantitative Per-Class Error Profile

### 2.1 Locked Test Set ($N=21$) Confusion Summary

| True Class | Support | EfficientNet-B0 Predicted | E2 Fusion Predicted | Equal-Weight Ensemble Predicted |
|:---|:---:|:---:|:---:|:---:|
| **A+** | 5 | 1 A+, 4 O+ (Recall: **20%**) | 3 A+, 1 B+, 1 O+ (Recall: **60%**) | 1 A+, 4 O+ (Recall: **20%**) |
| **AB+** | 2 | 1 A+, 1 O+ (Recall: **0%**) | 1 A+, 1 O+ (Recall: **0%**) | 1 A+, 1 O+ (Recall: **0%**) |
| **B+** | 6 | 1 A+, 5 O+ (Recall: **0%**) | 5 A+, 1 O+ (Recall: **0%**) | 1 A+, 5 O+ (Recall: **0%**) |
| **O+** | 8 | 2 A+, 1 B+, 5 O+ (Recall: **62.5%**) | 3 A+, 5 O+ (Recall: **62.5%**) | 2 A+, 6 O+ (Recall: **75%**) |
| **Total Acc** | **21** | **28.57% (6/21)** | **38.10% (8/21)** | **33.33% (7/21)** |

### 2.2 Predicted Class Distribution vs. True Class Distribution

```
Locked Test Ground Truth:   [A+: 23.8%,  AB+: 9.5%,   B+: 28.6%,  O+: 38.1%]
EfficientNet-B0 Predicted:  [A+: 23.8%,  AB+: 0.0%,   B+: 4.8%,   O+: 71.4%]  <-- O+ Collapse
E2 Fusion Predicted:        [A+: 52.4%,  AB+: 0.0%,   B+: 9.5%,   O+: 38.1%]  <-- A+ Collapse
Equal-Weight Ensemble:      [A+: 23.8%,  AB+: 0.0%,   B+: 0.0%,   O+: 76.2%]  <-- B+ & AB+ Erased
```

---

## 3. Detailed Root-Cause Hypotheses vs. Verified Observations

### 3.1 Observation 1: Total Loss of B+ Sensitivity on the Locked Test
- **Observed Fact:** True B+ subjects represent 28.6% of the test cohort ($6/21$). Every single true B+ subject was misclassified by all three pipelines ($0/6$ correct).
  - EfficientNet predicted O+ for 5 of them and A+ for 1.
  - E2 predicted A+ for 5 of them and O+ for 1.
- **Root-Cause Hypothesis:** In Indian demographic populations, B+ and O+ share high background frequency. Because the training datasets used unweighted or poorly calibrated cross-entropy losses with small sample sizes, the decision boundaries for B+ were eroded by the over-represented O+ prior.

### 3.2 Observation 2: Total Collapse of AB+ Minority Class
- **Observed Fact:** Across the 107 test/evaluation participants (21 locked + 80 dev + 26 cohort), AB+ recall never exceeded 14.3% in any model, and fell to **0.0%** in the locked test.
- **Root-Cause Hypothesis:** Severe sample scarcity. In the 80 development participants, only 7 individuals are AB+. In each 5-fold training split, E2 saw only ~5 AB+ participants ($15$ images) out of 192 training images ($7.8\%$). Deep networks with 512-dimensional dense heads cannot learn a generalized manifold from 5 subjects without heavy class-rebalancing or metric loss weighting.

### 3.3 Observation 3: High-Confidence Failure Modes
- On participant `P002` (True O+): E2 predicted A+ with **66.9% confidence**, and on participant `P0024` (True B+), E2 predicted A+ with **53.3% confidence**.
- The models produce highly confident incorrect predictions because softmax normalizes logits relative to existing class activations; when an out-of-domain impression produces uniformly suppressed features, minor logit differences get exponentiated into artificial high confidence.

---

## 4. Key Takeaways for Model Improvement

1. **Class-Rebalanced Loss Function is Essential:** Training with standard cross-entropy guarantees that minority classes (AB+, B+) are ignored. Focal Loss ($\gamma = 2.0$) or class-weighted cross-entropy ($w_c \propto 1/N_c$) must be tested.
2. **Single-Image EfficientNet Cannot Act as a Direct Classifier:** EfficientNet's severe O+ bias (71%–74% predictions) corrupts the ensemble. It requires explicit fine-tuning on MFS100 development images to reduce prior collapse.
3. **Threshold Calibration / Temperature Scaling:** Softmax probabilities are overconfident. Temperature scaling ($T > 1.0$) on validation logits should be explored during development.
