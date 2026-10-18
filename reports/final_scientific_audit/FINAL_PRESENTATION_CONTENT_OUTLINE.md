# Final Project Oral Defense: Presentation Content Outline

**Project Title:** Biometric Blood Group Detection Using Optical Fingerprint Images: An Empirical Investigation of Deep Learning, Dermatoglyphic Topology, and Hardware Confounders  
**Document ID:** `FINAL_PRESENTATION_CONTENT_OUTLINE.md`  
**Format:** Slide-by-Slide Defense Deck Outline (18 Slides + Q&A Backup)  
**Target Audience:** Final-Year Project Evaluation Committee, External Examiners, Department Faculty  
**Date:** 2026-10-18  
**Standardized Core Conclusion:**  
> *"Under the evaluated datasets, fingerprint representations, and participant-level validation procedures, the experiments did not establish reliable, generalizable blood-group classification performance."*

---

## Slide 1: Title & Project Overview
- **Visual Header:** Department of Computer Science & Engineering, Final Major Project Defense.
- **Title:** Biometric Blood Group Detection Using Optical Fingerprints: A Methodological and Empirical Investigation.
- **Presenters:** Student Name(s), USN(s).
- **Project Supervisor / Guide:** Faculty Name & Designation.
- **Key Takeaway:** A rigorous empirical evaluation of deep learning, handcrafted dermatoglyphics, and optical sensor physics demonstrating why optical fingerprints cannot reliably classify ABO blood groups.

---

## Slide 2: Critical Research Disclaimer & Clinical Safety
- **Mandatory Statement:** Investigational computational research prototype only.
- **Zero Clinical Diagnostic Validity:** This algorithm must NEVER be used for clinical diagnosis, emergency triage, surgical planning, or transfusion decisions.
- **Clinical Gold Standard:** Certified serological hemagglutination testing remains the indispensable medical standard.
- **Why This Matters:** Highlighting clinical safety upfront establishes ethical maturity and engineering rigor to examiners.

---

## Slide 3: Research Hypothesis & Historical Context
- **The Concept:** Historical anthropometric literature (Cummins & Midlo, 1943) postulated developmental links between embryonic volar pad formation and ABO genotype.
- **The Technical Question:** Can modern commercial optical fingerprint sensors (500 DPI) coupled with advanced computer vision extract generalizable, participant-level blood-group representations?
- **Target Classification:** 4-class ABO Positive (`A+`, `AB+`, `B+`, `O+`).

---

## Slide 4: Dataset Architecture & The Quarantine Strategy
- **Hardware Platform:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI, $316 \times 354$ native resolution).
- **Development-80 Cohort:** Exactly 80 participants, 240 scans (3 sequential impressions per subject).
  - Class Distribution: A+ = 19 ($23.8\%$), AB+ = 7 ($8.8\%$), B+ = 25 ($31.2\%$), O+ = 29 ($36.2\%$).
  - **Zero-Intelligence Majority Baseline (All O+):** **$36.25\%$ accuracy**, **$25.00\%$ balanced accuracy**, **$0.1330$ Macro F1**.
- **Quarantined Cohorts (Zero Access Events):**
  - Locked Test Cohort ($N=21$ participants, 63 scans).
  - External Hospital Cohort ($N=26$ participants, 78 scans).

---

## Slide 5: The "High Accuracy" Trap: Eliminating Data Leakage
- **Historical Claims in Literature:** Many papers report $80\%\text{--}90\%$ accuracy using standard CNNs.
- **The Flaw Exposed:** **Scan-level data leakage**. When 3 scans of the same participant are split randomly across train/test, the CNN memorizes unique finger shape and platen placement.
- **Our Methodological Guard:** Strict 5-Fold `StratifiedGroupKFold`. All 3 scans from any participant reside in exactly one fold.
- **Result of Proper Grouping:** Image-level accuracy immediately collapsed from $85\%$ to near-chance levels.

---

## Slide 6: Phase 1: Deep Convolutional Architectures (EXP-01 – EXP-06D)
- **Architectures Tested:** DenseNet-121, ResNet-50, EfficientNet-B0, MobileNetV2, VGG16.
- **Multi-Scan Concatenation:** 3-scan feature fusion ($3 \times 1024 = 3072$-d embedding).
- **Optimization:** Multi-task Cross-Entropy + Supervised Contrastive Loss ($\tau=0.05, \lambda=0.50$).
- **Apparent Success:** EXP-06C reached $53.75\%$ development accuracy and $0.5487$ Macro F1.
- **The Skeptical Question:** Did the network learn dermatoglyphic biology, or something else?

---

## Slide 7: Forensic Discovery of the Scanner Border Confounder
- **The Experiment:** Decomposing scans into friction-ridge foreground vs. non-ridge platen glass background.
- **The Smoking Gun (One-Way ANOVA across blood groups):**
  - **Border Luminosity:** **$F = 5.996, p = 0.0006$** (Critically Significant).
  - **Glass Background Intensity:** **$F = 3.988, p = 0.0085$** (Statistically Significant).
  - **Foreground Friction Ridges:** **$F = 0.684, p = 0.5641$** (Non-Significant).
- **Explanation:** Temporal collection batches caused ambient lighting and platen wear to correlate with blood groups. CNNs learned background luminous shortcuts!

---

## Slide 8: The Test Cohort Collapse: Confounder Confirmed
- **Validation on Locked Test Cohort:** When evaluated on the 21-participant locked test cohort (where background illumination shifted):
  - Development Accuracy: **$53.75\%$**
  - Locked Test Accuracy: **$33.33\%$** (Below the $36.25\%$ majority baseline).
- **Definitive Conclusion:** CNN performance on raw scans was an artifact of scanner background batch effects, not biological friction ridge signals.

---

## Slide 9: Phase 2: Testing Spatial Resolution Fidelity (EXP-07)
- **Hypothesis:** Did downsampling to $112\times 112$ destroy high-frequency friction ridge detail?
- **Controlled Comparison:** $112\times 112$ ($2.67\text{ px/cycle}$) vs. native $224\times 224$ ($5.33\text{ px/cycle}$) under pre-registered 80-epoch fixed-duration training.
- **Results:**
  - Arm A ($112\times 112$): Balanced Accuracy = **$28.61\%$**, Macro F1 = **$0.2800$**.
  - Arm B ($224\times 224$): Balanced Accuracy = **$26.15\%$**, Macro F1 = **$0.2506$**.
  - Paired Permutation Test: **$p = 0.6225$** (Non-Significant).
- **Verdict:** Resolution was not the bottleneck; higher resolution slightly increased overfitting.

---

## Slide 10: Phase 3: Handcrafted Dermatoglyphic Features (EXP-08)
- **Pipeline:** 24 numeric features (sub-core ridge density, crossing-number minutiae, core-to-delta distance, Poincaré pattern classification).
- **Poincaré Classification Noise:** 74/80 participants ($92.5\%$) classified as WHORL due to optical border noise triggering false-positive singular points.
- **Model Results:**
  - Constant Majority Baseline: **$36.25\%$** (29/80).
  - Random Forest: **$35.00\%$** (28/80, Balanced Acc: $27.06\%$, Macro F1: $0.2603$).
  - Logistic Regression: **$27.50\%$** (22/80, Balanced Acc: $21.15\%$).

---

## Slide 11: Full-Pipeline Permutation Testing (EXP-08)
- **Rigorous Null Hypothesis Test:** 1,000 complete pipeline permutations refitting scaling, imputation, and Random Forest on shuffled labels.
- **Empirical Null Distribution:** Mean Balanced Accuracy = $23.62\%$, Max = $43.20\%$.
- **Exact Finite-Sample $p$-value:** **$p = (253 + 1) / (1001) = \mathbf{0.2537}$**.
- **Takeaway:** Handcrafted biometric features operate fully within the random chance distribution.

---

## Slide 12: Phase 4: Continuous Spatial Orientation Tensor Fields (EXP-09)
- **Novel Representation:** 2D gradient structure tensor decomposed into $[u, v] = [\cos(2\theta)\kappa, \sin(2\theta)\kappa]$ on a $23 \times 20$ grid ($920$ dimensions).
- **Complete Background Elimination:** Binary foreground masking eliminated 100% of platen borders and ambient glass.
- **Results:**
  - Random Forest: Accuracy = **$40.00\%$** (32/80 correct vs. 29/80 baseline), Balanced Accuracy = **$31.37\%$**.
  - Permutation Test: $p(\text{Accuracy} \ge 0.4000) = \mathbf{0.1683}$, $p(\text{BalAcc} \ge 0.3137) = \mathbf{0.1089}$.
  - The gain of 3 participants is statistically non-significant.

---

## Slide 13: Phase 5: Multi-Scan Quality & Consistency (EXP-10)
- **Per-Scan Quality Audit (240 scans):** Mean foreground coverage $86.1\%$, mean coherence $0.658$. Image quality is uniformly high.
- **Intra-Participant Consistency (80 participants):** Mean cosine similarity = **$0.9355$**; $95\%$ of participants have quality ratio $< 1.5$.
- **Controlled Comparison:**
  - Condition A (Unweighted Mean): RF Accuracy = **$40.00\%$**, Balanced Accuracy = **$31.37\%$**.
  - Condition B (Quality-Weighted Mean): RF Accuracy = **$33.75\%$**, Balanced Accuracy = **$26.15\%$**.
- **Why Quality-Weighting Degraded Performance:** Uniform averaging acts as a noise-reduction filter ($1/\sqrt{3} \approx 42\%$ variance reduction); quality-weighting reduces effective sample size.

---

## Slide 14: The Minority Class Problem: Universal AB+ Failure
- **Per-Class Recall Across All Late-Stage Models:**
  - **A+ ($N=19$):** $15\%\text{--}37\%$
  - **B+ ($N=25$):** $32\%\text{--}56\%$
  - **O+ ($N=29$):** $31\%\text{--}55\%$
  - **AB+ ($N=7$):** **$0.0\%$ (0/7 detected across every model)**.
- **Significance:** Despite class-balanced weighting, every model completely fails on the minority class, collapsing onto the two largest classes.

---

## Slide 15: Independent Reproducibility Audit & Reconciliation
- **Verification Rate:** 71 out of 75 metrics ($94.7\%$) reproduced to four decimal places directly from saved prediction CSVs.
- **Transparent Discrepancy Resolution:**
  - EXP-08 fold-local baseline vs global baseline documented.
  - EXP-10 transcription erratum (2 prediction flips, not 1) corrected in dedicated companion report.
- **Data Quarantine Check:** Programmatically confirmed $0$ access events to locked test and hospital cohorts.

---

## Slide 16: What the Evidence Proves vs. What Remains Uncertain
- **What Was Proved:**
  - Optical fingerprints captured at 500 DPI do not provide predictive four-class blood group discrimination.
  - Development CNN accuracy was confounded by scanner border brightness batch effects.
  - The majority baseline ($36.25\%$) remains undefeated.
- **What Remains Uncertain (Epistemic Boundaries):**
  - Labels were self-reported by students (5-10% estimated noise).
  - These results do not disprove potential microscopic, histological, or sweat-pore biochemical markers.

---

## Slide 17: Scientific Rigor: The Value of a Verified Negative Result
- **Engineering Integrity:** In machine learning, diagnosing why a model fails and identifying subtle hardware confounders is a far greater contribution than presenting a brittle, overfitted artifact.
- **Methodological Standards Set:**
  1. Participant-level isolation (zero leakage).
  2. Full-pipeline permutation testing with finite-sample correction.
  3. Pre-registered fixed-duration evaluation.
  4. Complete machine-readable audit trail.

---

## Slide 18: Final Conclusion & Defense Takeaway
- **Standardized Conclusion:**
  > *"Under the evaluated datasets, fingerprint representations, and participant-level validation procedures, the experiments did not establish reliable, generalizable blood-group classification performance."*
- **Final Verdict:** The research program has systematically exhausted visual, topological, resolution, and aggregation hypotheses. The experimental phase is properly and definitively closed.
- **Thank You:** Questions & Discussion.

---

## Backup / Q&A Defense Slides

### Backup Slide A: Why not train another deep model (e.g., Vision Transformer)?
- Without solving ground-truth serological validation and optical border confounders, high-capacity models will merely overfit noise faster.

### Backup Slide B: Exactly how did the optical border confounder work?
- Show ANOVA table: Border intensity $F=5.996, p=0.0006$. Explain chronological batch collection and sensor silicone wear.

### Backup Slide C: McNemar Test Math on EXP-10
- Exact binomtest on discordant pairs: Ridge $b=1, c=1$ ($p=1.0000$); RF $b=7, c=2$ ($p=0.1797$).
