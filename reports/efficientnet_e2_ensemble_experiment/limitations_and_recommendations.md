# Limitations and Recommendations: Biometric Blood Group Ensemble Experiment

**Experiment:** EfficientNet-B0 vs. E2 Fusion vs. Equal-Weight Ensemble  
**Date:** 2026-10-13  
**Destination:** `reports/efficientnet_e2_ensemble_experiment/limitations_and_recommendations.md`  

---

## 1. Unresolved Limitations

### 1.1 Severe Sensor & Acquisition Domain Gap for EfficientNet-B0
- **Training Domain vs. Deployment Domain:** EfficientNet-B0 was trained exclusively on the historical ~6,000-image dataset acquired via a CMOS optical reflection fingerprint scanner. It was never trained, validated, or fine-tuned on Mantra MFS100 capacitive/optical live captures.
- **Preprocessing Incongruence:** While E2 applies domain-tailored foreground cropping and CLAHE ridge contrast enhancement (`transform_c5`), EfficientNet-B0 relies on standard bilinear resizing to $224\times 224$ and generic ImageNet normalization. This mismatch leads to severe out-of-domain degradation (28.57% on locked test, 32.50% on development).

### 1.2 Failure to Outperform the Trivial Majority-Class Baseline
- Across all three evaluated cohorts, neither model nor the ensemble consistently beat the zero-effort **Always-O+ baseline**:
  - **Development Cohort (N=80):** Always-O+ is **36.25%**. The Ensemble achieved **38.75%** (a marginal +2.5% over guessing O+), while E2 achieved **53.75%**.
  - **Locked Test (N=21):** Always-O+ is **38.10%**. E2 achieved **38.10%** (tied with guessing O+), while the Ensemble achieved **33.33%** (worse than guessing O+).
  - **Separate 26-Cohort (N=26):** Always-O+ is **42.31%**. EfficientNet and the Ensemble achieved **38.46%**, and E2 achieved **19.23%** (both well below the naive baseline).

### 1.3 Statistical Uncertainty Due to Small Cohort Sizes
- **Sample Sizes:** $N=21$ for the locked test set and $N=26$ for the descriptive cohort are statistically small cohorts.
- **Wide Confidence Intervals:** A sample size of $N=21$ produces a 95% Wilson score confidence interval of approximately $\pm 20\%$ around a 38% accuracy estimate. A single participant shifting between correct and incorrect changes the accuracy metric by $\approx 4.76\%$. Numerical differences between 28.57% (6/21), 33.33% (7/21), and 38.10% (8/21) are not statistically significant ($p > 0.5$, McNemar test).

### 1.4 Minority Class Collapse (Particularly AB+ and B+)
- **AB+ Collapse:** In the locked test set, all three models achieved **0.0% recall** on AB+ ($0/2$). On the 80 development participants, EfficientNet and the Ensemble achieved **0.0% recall** on AB+ ($0/7$).
- **B+ Sensitivity Loss:** In the locked test set, the ensemble completely missed all 6 true B+ participants (**0.0% recall**), predicting O+ for 5 and A+ for 1.
- **Prior Collapse:** EfficientNet-B0 disproportionately predicted O+ (59 out of 80 development participants, 15 out of 21 test participants), whereas E2 disproportionately predicted A+ on the separate 26-cohort (16 out of 26 participants).

### 1.5 Participant Independence & De-Anonymization Bounds
- Although file hashes and capture logs confirm zero duplicate images across cohorts, biometric minutiae cross-matching was not performed between `P001`–`P105` and `S001`–`S026`. Therefore, individual-level physical independence between cohorts cannot be conclusively proven beyond file-level isolation.

### 1.6 Methodological Incompatibility with Ensemble-Weight Optimization
- Principled out-of-fold ensemble weight tuning requires true cross-validation out-of-fold predictions from **both** models on the same development cohort. Because EfficientNet-B0 only exists as a single frozen checkpoint trained on external data, valid development OOF predictions did not exist for it. Attempting to fit weights on zero-shot predictions would risk arbitrary overfitting.

---

## 2. Key Scientific Recommendations

1. **Do Not Deploy the Equal-Weight Ensemble:**  
   The empirical evidence clearly shows that combining EfficientNet-B0 with E2 damages E2's discrimination and increases error rates (introducing 31 new errors across 127 participants). The equal-weight ensemble should not be used in demonstration or deployment.
2. **Always Benchmark Against the Majority-Class Baseline:**  
   In unbalanced clinical or biometric classification problems (where O+ and B+ comprise ~70% of the population), reporting nominal accuracy without the majority baseline creates a misleading impression of model utility.
3. **Domain Adaptation Must Precede Ensembling:**  
   Ensembling models trained on disparate hardware domains without prior domain alignment or joint fine-tuning is counterproductive. Future research should align ridge frequency and sensor response before attempting multi-model fusion.
4. **Preserve Ethical & Scientific Framing:**  
   Fingerprint ridges are dermatoglyphic patterns formed during embryonic development. No biological or causal mechanism connects dermal ridge configurations to circulating ABO red blood cell antigens. This project must remain strictly framed as an exploratory AI pattern recognition experiment and never as clinical blood testing.
