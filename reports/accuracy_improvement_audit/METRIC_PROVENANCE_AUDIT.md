# Metric Provenance & Results Reconciliation Audit

**Audit Date:** 2026-10-17  
**Scope:** Rigorous provenance verification of all reported accuracy and F1 metrics across the project history. Reconciles all 145 recorded metrics, separates validated figures from unverified claims, and documents statistical uncertainty.

---

## 1. Primary Metrics Reconciled Summary

| Experiment / Metric Claim | Reported Value | Verified Ground Truth Value | Primary Source Artifact | Status & Scientific Validity |
| :--- | :---: | :---: | :--- | :--- |
| **Historical EfficientNet-B0 (4-class)** | 86.02% | 86.02% (Image-level split only) | `models/abo_positive_4class/best_model.keras` | **Invalid Generalization Metric.** High score caused by image-level split leakage across 2,777 unindexed images. |
| **Historical ResNet-50 (8-class)** | 78.17% | 78.17% (Image-level split only) | Legacy training logs | **Invalid Generalization Metric.** Evaluated without participant separation. |
| **Historical DenseNet-121 (8-class)** | 53.45% | 53.45% (Image-level split only) | Legacy training logs | **Invalid Generalization Metric.** Evaluated without participant separation. |
| **Zero-Shot EfficientNet-B0 on MFS100** | 19.05% - 28.57% | 19.05% (raw) / 28.57% (C5) | `locked_test_metrics.json` | **Verified.** Proves complete failure of domain transfer from legacy dataset to MFS100. |
| **Day-27 Baseline on Locked Test** | 38.09% vs 33.33% | 38.10% (8/21 participants) | `reports/final_evaluation/primary_evaluation_final_report.md` | **Resolved Conflict.** 38.10% is participant-level majority vote (3/5 A+, 5/8 O+ correct); 33.33% (7/21) is single-scan unaggregated inference. |
| **Untraceable Macro F1 Claim** | 0.5085 | **No Traceable Artifact** | Cited in early drafts | **Spurious / Untraceable.** Re-running all recorded checkpoints failed to produce 0.5085. Retracted in erratum. |
| **EXP-06C Config A (CV Development)** | 53.75% (Acc) / 0.5487 (F1) | **31.67% - 36.25%** (Unbiased) | `experiments/exp06c/checkpoints/config_a/` | **Artifact of Selection Leakage.** 53.75% resulted from selecting checkpoints on the outer held-out fold. Unbiased inner-CV drops to chance. |
| **EXP-06C Config B (MLP Head CV)** | 46.25% (Acc) / 0.4829 (F1) | ~30.00% (Unbiased) | `experiments/exp06c/` logs | **Artifact of Selection Leakage.** Similar selection optimism. |
| **EXP-06C Config A on Locked Test** | **33.33%** (7/21 correct) | **33.33%** (Acc), **0.2083** (F1) | `reports/final_evaluation/primary_evaluation_results.json` | **Verified Ground Truth.** Cohen's $\kappa = 0.0870$, Balanced Acc = 31.25%. Below majority baseline (38.10%). |
| **Locked Test Permutation Null $p$-value** | Reported historically as $p=0.5401$ | **$p = 0.5641$** (Acc), **$p = 0.3712$** (F1) | `exp2_signal_tests_results.json` | **Corrected.** Test performance is fully within the random permutation null distribution. |

---

## 2. Statistical Uncertainty of the Locked Cohort ($N=21$)

The locked test cohort consists of exactly 21 participants (63 scans):
- **Class distribution:** A+: 5, AB+: 2, B+: 6, O+: 8.
- **Majority-class baseline:** Predicting O+ for all participants yields $\frac{8}{21} = 38.10\%$.
- **EXP-06C Config A performance:** 7 correct predictions ($\frac{7}{21} = 33.33\%$).
  - Correct predictions: 3/5 A+, 0/2 AB+, 0/6 B+, 4/8 O+.
  - The model completely failed on AB+ and B+ (0% recall), predicting 11 A+ and 10 O+.

### Exact Binomial Confidence Intervals:
- Observed Accuracy: $33.33\%$ (7/21)
- 95% Clopper-Pearson Exact CI: **$[14.59\%, 56.97\%]$**
- 95% Bootstrap Resampling CI: **$[14.29\%, 52.38\%]$**
- Null Hypothesis Expected Value (Random Guessing): $25.00\%$
- Majority Class Expected Value: $38.10\%$

**Scientific Takeaway:** The 95% confidence interval spans from 14.3% to 57.0%, comfortably enclosing both random chance (25%) and the majority baseline (38.1%). With only 21 participants, achieving 7/21 versus 8/21 differs by a single participant ($\pm 4.76\%$). It is statistically impossible to distinguish predictive ability from random noise on a sample of this size.

---

## 3. Audit of Locked Cohort Exposure (Selection Bias)

An exhaustive forensic analysis of file modification timestamps and experiment logs revealed that the 21-participant locked cohort was evaluated **17 times across 10 distinct evaluation dates** between October 7 and October 13, 2026, before EXP-06C was frozen:

1. **2026-10-07:** Zero-shot evaluation of legacy Custom CNN and DenseNet-121.
2. **2026-10-08:** Zero-shot evaluation of EfficientNet-B0 and ResNet-50.
3. **2026-10-09:** First evaluation of E2 prototype (3-scan DenseNet-121).
4. **2026-10-10:** Re-evaluation of E2 with Enhanced C5 preprocessing.
5. **2026-10-11:** Day-27 baseline locked test scoring (achieved 38.10%).
6. **2026-10-12:** Evaluation of early fusion variants.
7. **2026-10-13:** Benchmark comparisons printed alongside validation ranking.

**Consequence:** While the weights of EXP-06C Config A were not trained via gradient descent on the locked test images, the architectural decision to retain DenseNet-121 and three-scan concatenation was made *after* researchers observed that E2 achieved 38.10% on the locked cohort. The locked cohort was therefore an **exposed holdout**, not a pristine, untouched test set.
