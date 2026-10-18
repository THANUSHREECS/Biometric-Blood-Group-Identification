# Research Paper Preparation Guide
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Target Audience:** Project Teammates, Manuscript Authors, and Research Supervisors  
**Purpose:** Comprehensive guide for authoring a rigorous, scientifically sound, and publication-ready manuscript  
**Integrity Rule:** Publish verified empirical findings and failure-mode analyses; make zero unsupported claims.

---

## 1. Recommended Paper Title Options

Select an objective, scholarly title that accurately reflects an empirical investigation and failure-mode analysis rather than an unverified technological breakthrough:

* **Title Option 1 (Recommended for IEEE/ACM Transactions):**  
  * *"An Empirical Investigation into Dermatoglyphic-Serological Association: Evaluating Deep CNNs and Ridge Orientation Fields for Biometric Blood Group Classification"*
* **Title Option 2 (Recommended for Forensic Science & Biometrics Journals):**  
  * *"Biometric Blood Group Prediction from Optical Fingerprints: A Methodological Audit on Participant-Level Generalization, Confounding Artifacts, and Data Leakage"*
* **Title Option 3 (Recommended for Computer Vision / Medical Informatics):**  
  * *"On the Feasibility of Dermatoglyphic ABO Blood Group Typing: Quantitative Benchmarking of Convolutional Networks, Handcrafted Minutiae, and Optical Confounders"*

---

## 2. Proposed Manuscript Structure & Section-by-Section Evidence Map

### Section I: Introduction
* **Core Narrative:** Introduce the historical dermatoglyphic hypothesis—the conjecture that common embryonic developmental timing (weeks 10–19 of gestation) might induce pleiotropic correlations between friction ridge formations and erythrocyte antigen expression.
* **Problem Statement:** While numerous small-scale studies report extraordinarily high classification accuracies (75%–90%), almost none employ participant-level cross-validation or account for optical scanner batch confounding.
* **Key Contributions to Highlight:**
  1. Systematic multi-architecture evaluation across deep CNNs, contrastive learning, handcrafted minutiae, and continuous orientation tensor fields.
  2. Proof that prior high accuracy claims were artifacts of scan-level data leakage.
  3. Discovery that deep neural networks opportunistically exploit non-ridge optical platen border brightness ($F = 5.996, p = 0.0006$).
  4. Full-pipeline permutation testing establishing that friction ridge patterns do not predict ABO blood groups above chance.

### Section II: Related Work & Prior Literature
* **Topics to Cover:**
  * Dermatoglyphics and medical genetics (historical associations with chromosomal anomalies such as Down syndrome and Turner syndrome).
  * Literature claiming blood group-fingerprint correlations (e.g., ridge density variations, loop/whorl frequency differences).
  * Critiques of methodological rigor in biometric AI: scan-level leakage, small cohorts, and lack of permutation tests.
* **Caveat:** Teammates must perform an independent literature search to find formal journal citations; do not fabricate citations.

### Section III: Dataset Governance & Hardware Acquisition Protocol
* **Hardware:** Mantra MFS100 500 DPI optical scanner (312x512, 8-bit grayscale, 16x24 mm sensor platen).
* **Protocol:** Three-scan left-thumb protocol with mandatory lift-and-reposition between captures.
* **Development Cohort (Dev-80):** 80 student participants, 240 impressions, 4-class distribution: A+ (19), B+ (25), AB+ (7), O+ (29).
* **Mandatory Caveat:** Explicitly acknowledge that blood group labels were self-reported donor survey responses and not validated by laboratory serological testing.

### Section IV: Feature Representations & Machine Learning Methodology
* **Deep Neural Networks:** ResNet-50, DenseNet-121, EfficientNet-B0, VGG16, and ConvNeXt-Tiny under C5 preprocessing.
* **Handcrafted Dermatoglyphic Features (EXP-08):** 24 numeric features (bifurcation counts, ridge ending counts, core/delta topological distances, ridge density).
* **Continuous Spatial Orientation Tensors (EXP-09):** $920$-d vector field $[\cos(2\theta), \sin(2\theta)]$ extracted via Sobel gradients with 100% platen border background masking.
* **Evaluation Protocol:** 5-fold `StratifiedGroupKFold` partitioned strictly by participant ID.
* **Primary Endpoints:** Macro-averaged F1 score and balanced accuracy (to guard against class imbalance).

### Section V: Experimental Results & Statistical Evaluation
* **Quantitative Findings:**
  * EXP-01 to EXP-06D: Apparent CNN development accuracies reached 50%–54% but collapsed on out-of-domain evaluation (33.3%).
  * EXP-07 (Resolution Fidelity): Scaling input from 112x112 to 224x224 did not improve balanced accuracy (28.6% vs 26.1%, $p = 0.6225$).
  * EXP-08 (Minutiae Features): Random Forest achieved 35.0% accuracy vs 36.25% baseline (permutation $p = 0.2537$).
  * EXP-09 (Orientation Tensors): Random Forest achieved 40.0% accuracy vs 36.25% baseline (permutation $p = 0.1089$).
  * EXP-10 (Multi-Scan Quality): Simple arithmetic pooling equaled or outperformed quality-weighted averaging (36.25% vs 33.75%).
* **Minority Class Failure:** 0.0% recall across all models for AB+ (0/7 participants).

### Section VI: Confounding Analysis & Failure Modes
* **Failure Mode 1: Scan-Level Leakage:** Show side-by-side comparison of scan-level random splitting (apparent 85%+) vs participant-level grouping (~36%).
* **Failure Mode 2: Optical Platen Illumination Confounding:** Detail the ANOVA test showing that non-ridge border pixels correlated with blood group labels due to temporal collection clustering.
* **Failure Mode 3: Majority-Class Attraction:** Show confusion matrices demonstrating that models predict B+ and O+ almost exclusively.

### Section VII: Discussion, Epistemic Boundaries & Limitations
* What this study disproves: Standard optical friction-ridge features do not predict ABO blood groups above chance.
* What this study does NOT disprove: Does not rule out sub-micron poroscopic structures, chemical sweat-pore biomarkers, or microscopic epidermal biology.
* Empirical limitations: Self-reported labels, single-institution cohort, sample size of 80 participants.

### Section VIII: Conclusion & Future Research Directions
* Conclude with the standardized scientific statement:  
  *"Under the evaluated datasets, optical fingerprint representations, and participant-level validation procedures, friction ridge patterns did not demonstrate statistically significant or clinically viable blood group prediction."*

---

## 3. Mandatory Figures & Tables for the Paper

### Suggested Table 1: Dataset Cohort Profile
```
Table 1. Experimental Cohort Characteristics and Class Distribution
+---------------+--------------+------------+--------+----------------------------+-----------------------+
| Cohort        | Participants | Scans / P  | Total  | ABO Distribution           | Ground-Truth Source   |
+---------------+--------------+------------+--------+----------------------------+-----------------------+
| Dev-80        | 80           | 3 (Thumb)  | 240    | A:19, B:25, AB:7, O:29     | Self-reported survey  |
| Test-21       | 21           | 3 (Thumb)  | 63     | A:5,  B:7,  AB:2, O:7      | Quarantined survey    |
| Hospital-26   | 26           | 3 (Thumb)  | 78     | A:6,  B:8,  AB:3, O:9      | Hospital context      |
+---------------+--------------+------------+--------+----------------------------+-----------------------+
```

### Suggested Table 2: Benchmark Model Performance on Dev-80 (Consolidated)
```
Table 2. Participant-Level 5-Fold Cross-Validation Performance Across Representations
+------------------------------------+------------+--------------+----------+----------+-------------+
| Representation / Model             | Accuracy   | Balanced Acc | Macro F1 | AB+ Rec. | Perm. p-val |
+------------------------------------+------------+--------------+----------+----------+-------------+
| Majority-Class Zero-Intelligence   | 36.25%     | 25.00%       | 0.1330   | 0.0%     | --          |
| Deep CNN Baseline (EXP-01 Focal)   | 36.25%     | 32.82%       | 0.3205   | 14.3%    | --          |
| ResNet-50 112x112 (EXP-07 Arm A)   | 36.25%     | 28.61%       | 0.2800   | 0.0%     | --          |
| ResNet-50 224x224 (EXP-07 Arm B)   | 33.75%     | 26.15%       | 0.2506   | 0.0%     | p = 0.6225  |
| Handcrafted Minutiae (EXP-08 RF)   | 35.00%     | 27.06%       | 0.2603   | 0.0%     | p = 0.2537  |
| Ridge Orientation Tensor (EXP-09 RF| 40.00%     | 31.37%       | 0.3048   | 0.0%     | p = 0.1089  |
| Quality-Weighted Fusion (EXP-10 RF)| 33.75%     | 26.15%       | 0.2503   | 0.0%     | --          |
+------------------------------------+------------+--------------+----------+----------+-------------+
```

### Suggested Figures
* **Figure 1:** System Overview Diagram (Mantra MFS100 capture $	o$ C5 preprocessing $	o$ feature representation $	o$ consensus inference).
* **Figure 2:** Platen Border Confounder Analysis (Visualizing non-ridge border intensity differences across session batches).
* **Figure 3:** Permutation Distribution Histogram (Plotting the null permutation distribution in EXP-08/EXP-09 with observed classifier performance marked, proving it sits inside the null envelope).
* **Figure 4:** Multi-Scan Quality Consistency (Scatter plot showing intra-participant cosine similarity and uniform quality ratios).

---

## 4. Strictly Forbidden Claims (What Must NEVER Appear in the Paper)

1. **NEVER claim high accuracy as a successful outcome:**  
   * *Forbidden:* "Our deep learning model achieves 88% accuracy in identifying blood groups."  
   * *Required Phrasing:* "While naive scan-level partitioning yields an apparent accuracy of 88%, rigorous participant-level validation reveals that this is an artifact of identity leakage."
2. **NEVER claim clinical or diagnostic utility:**  
   * *Forbidden:* "This technology enables non-invasive blood typing for emergency triage."  
   * *Required Phrasing:* "The system is an investigational research prototype and provides no clinical validity for diagnostic or transfusion use."
3. **NEVER describe the locked test cohort as completely untouched:**  
   * *Forbidden:* "Evaluated on a completely blind, untouched test set."  
   * *Required Phrasing:* "Evaluated on a locked secondary cohort collected during initial platform trials."
4. **NEVER claim serological laboratory ground truth:**  
   * *Forbidden:* "Validated against laboratory-confirmed blood group samples."  
   * *Required Phrasing:* "Labels were obtained through self-reported participant medical history."

---

## 5. Checklist of Information Teammates Must Confirm

Before submitting the paper, the team must confirm:
* [ ] Official Institutional Ethics Clearance / IRB Protocol Number.
* [ ] Formal author list, student registration numbers, and supervisor details.
* [ ] Exact citation for the `dataset_blood_group/` repository.
* [ ] Verified absence of identity overlap between Dev-80 and Test-21.
* [ ] Target journal or conference submission deadline and LaTeX formatting style (e.g., IEEEtran, Springer LNCS).
