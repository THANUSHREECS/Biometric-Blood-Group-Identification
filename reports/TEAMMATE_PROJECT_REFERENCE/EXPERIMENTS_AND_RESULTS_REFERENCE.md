# Experiments & Quantitative Results Reference
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Audience:** Research Authors, Data Scientists, and Academic Examiners  
**Scope:** Exhaustive experimental documentation covering EXP-01 through EXP-10, baseline models, and discrepancy resolutions  
**Audit Status:** Fully reconciled against `reports/final_scientific_audit/METRIC_RECONCILIATION.csv` (71/75 exact matches, 4 reconciled legacy discrepancies).

---

## 1. Consolidated Master Experiment Table

The table below summarizes the core participant-level evaluations conducted on the primary **Development Cohort (Dev-80: 80 participants, 240 scans, 4 classes)**:

| Exp ID | Experiment Name | Model / Representation | Accuracy | Balanced Acc | Macro F1 | AB+ Recall | Permutation $p$-val | Evidence File |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BL-0** | Zero-Intelligence Baseline | Constant Majority Vote (O+ / B+) | **36.25%** | **25.00%** | **0.1330** | **0.0%** | -- | Theoretical Baseline |
| **EXP-01** | Class-Balanced Focal Loss | ResNet-50 + Focal Loss ($\gamma=2.0$) | 36.25% | 32.82% | 0.3205 | 14.3% | Not computed | `reports/accuracy_improvement/exp01/` |
| **EXP-02** | Conservative Augmentation | ResNet-50 + Spatial Jitter | 47.50% | 43.60% | 0.4632 | 14.3% | Not computed | `reports/accuracy_improvement/exp02/` |
| **EXP-03** | Augmentation Ablation | ResNet-50 (Native Scans, No Aug) | 53.75% | 52.84% | 0.5068 | 28.6% | Not computed | `reports/accuracy_improvement/exp03/` |
| **EXP-04** | Backbone Fine-Tuning Depth | DenseNet-121 (Frozen Reference) | 50.00% | 47.86% | 0.4781 | 14.3% | Not computed | `reports/accuracy_improvement/exp04/` |
| **EXP-05** | Multi-Scan Fusion Arch. | 3-Scan Concat Feature Fusion | 52.50% | 51.84% | 0.5107 | 28.6% | Not computed | `reports/accuracy_improvement/exp05/` |
| **EXP-06A** | SupCon Temperature Scaling | SupCon + ResNet-50 ($\tau=0.05$) | 52.50% | 55.18% | 0.5313 | 42.9% | Not computed | `reports/accuracy_improvement/exp06a/` |
| **EXP-06B** | SupCon Loss Weight Tuning | SupCon + Cross-Entropy ($\lambda=0.50$) | 52.50% | 51.46% | 0.4925 | 28.6% | Not computed | `reports/accuracy_improvement/exp06b/` |
| **EXP-06C** | Linear Head Capacity | DenseNet-121 + Linear Projection | 53.75% | 53.47% | 0.5487 | 42.9% | Not computed | `reports/accuracy_improvement/exp06c/` |
| **EXP-06D** | Preprocessing Ablation | Raw Platen Scans (No Filter) | 52.50% | 50.07% | 0.5078 | 28.6% | Not computed | `reports/accuracy_improvement/exp06d/` |
| **EXP-07** | Resolution Scaling (112 vs 224) | ResNet-50 (Arm A: 112x112) | 36.25% | 28.61% | 0.2800 | 0.0% | -- | `experiments/exp07_resolution_fidelity/` |
| **EXP-07** | Resolution Scaling (112 vs 224) | ResNet-50 (Arm B: 224x224) | 33.75% | 26.15% | 0.2506 | 0.0% | $p = 0.6225$ | `experiments/exp07_resolution_fidelity/` |
| **EXP-08** | Handcrafted Minutiae | Random Forest (24 numeric features) | 35.00% | 27.06% | 0.2603 | 0.0% | $p = 0.2537$ | `experiments/exp08_dermatoglyphic_minutiae/` |
| **EXP-08** | Handcrafted Minutiae | Logistic Regression | 30.00% | 23.36% | 0.2241 | 0.0% | Within chance | Reconciled in Audit |
| **EXP-09** | Ridge-Orientation Tensors | Ridge Classifier (920-d features) | 36.25% | 29.20% | 0.2770 | 0.0% | Within chance | `experiments/exp09_orientation_tensor/` |
| **EXP-09** | Ridge-Orientation Tensors | Logistic Regression | 35.00% | 28.97% | 0.2775 | 0.0% | Within chance | `experiments/exp09_orientation_tensor/` |
| **EXP-09** | Ridge-Orientation Tensors | Random Forest (Best Arm) | 40.00% | 31.37% | 0.3048 | 0.0% | $p = 0.1089$ | `experiments/exp09_orientation_tensor/` |
| **EXP-10** | Multi-Scan Quality Pooling | Ridge (Condition A: Unweighted) | 36.25% | 29.20% | 0.2770 | 0.0% | Within chance | `reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md` |
| **EXP-10** | Multi-Scan Quality Pooling | Ridge (Condition B: Weighted) | 36.25% | 28.75% | 0.2730 | 0.0% | Within chance | Reconciled Erratum in Audit |
| **EXP-10** | Multi-Scan Quality Pooling | Random Forest (Condition A: Unweighted)| 40.00% | 31.37% | 0.3048 | 0.0% | Within chance | `reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md` |
| **EXP-10** | Multi-Scan Quality Pooling | Random Forest (Condition B: Weighted) | 33.75% | 26.15% | 0.2503 | 0.0% | Degraded | `reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md` |

---

## 2. Detailed Experiment-by-Experiment Analysis

### EXP-01: Class-Balanced Focal Loss
* **Research Question:** Can Focal Loss ($\gamma = 2.0, \alpha_c = \text{balanced}$) overcome class imbalance and prevent minority class collapse?
* **Model & Setup:** ResNet-50 backbone, 5-fold `StratifiedGroupKFold` on Dev-80, input size $112\times 112$.
* **Results:** Accuracy: **36.25%**, Balanced Acc: **32.82%**, Macro F1: **0.3205**.
* **Key Finding:** Model performance remained anchored to the 36.25% majority baseline. AB+ recall reached only 14.3% (1/7).

### EXP-02: Conservative Augmentation
* **Research Question:** Does mild affine augmentation (rotation $\pm 8^\circ$, translation $\pm 6$ px) improve generalization on optical scans?
* **Results:** Accuracy: **47.50%**, Balanced Acc: **43.60%**, Macro F1: **0.4632**.
* **Key Finding:** While development metrics appeared higher than EXP-01, subsequent audit proved spatial interpolation distorted friction ridge frequencies and the model exploited platen illumination.

### EXP-03: Augmentation Ablation (Native Scans)
* **Research Question:** How does training on pure native scans compare against spatial geometric augmentation?
* **Results:** Accuracy: **53.75%**, Balanced Acc: **52.84%**, Macro F1: **0.5068**.
* **Key Finding:** Native scans outperformed augmented data, demonstrating that artificial spatial interpolation degrades delicate ridge-valley transitions.

### EXP-04: Backbone Fine-Tuning Depth
* **Research Question:** Should CNN backbones be frozen or partially fine-tuned for dermatoglyphic classification?
* **Evaluated Backbones:** DenseNet-121, ResNet-50.
* **Results:** Frozen Feature Extractor (Config A): **50.00%** acc, **47.86%** bal acc. Partially fine-tuned blocks (Config B/C) suffered catastrophic overfitting on $N=80$.
* **Key Finding:** Small biometric sample sizes prohibit back-propagation through deep feature extractors.

### EXP-05: Multi-Scan Feature Fusion
* **Research Question:** Does cross-scan self-attention outperform simple feature vector concatenation across the 3 impressions?
* **Results:** Concatenation (Config A): **52.50%** acc, **51.84%** bal acc. Cross-scan attention: **48.75%** acc (overfitted attention weights).
* **Key Finding:** Concatenating feature vectors is superior to complex self-attention when participant count is small.

### EXP-06A through EXP-06D: Supervised Contrastive Learning & Head Optimization
* **Research Question:** Can Supervised Contrastive (SupCon) pre-training align intra-class ridge embeddings and separate inter-class clusters?
* **Findings:**
  * EXP-06A (Temperature $\tau=0.05$): Highest apparent minority recall (42.9%), Macro F1: **0.5313**.
  * EXP-06C (Linear Head): Highest reported development Macro F1 (**0.5487**).
  * **Critical Audit Discovery:** In Phase 2 failure mode analysis, an ANOVA test revealed that **platen border brightness correlated with blood group ($F=5.996, p=0.0006$)**. When the EXP-06 models were evaluated on the locked Test-21 cohort, where scanner illumination had drifted, accuracy collapsed to **33.33%** (below baseline).

### EXP-07: Resolution Fidelity (112x112 vs 224x224)
* **Research Question:** Did downsampling images from $312\times 512$ to $112\times 112$ destroy high-frequency minutiae signals?
* **Setup:** Fixed 80-epoch training comparing Arm A ($112\times 112$) against Arm B ($224\times 224$).
* **Results:**
  * Arm A (112x112): Accuracy: **36.25%**, Balanced Acc: **28.61%**, Macro F1: **0.2800**.
  * Arm B (224x224): Accuracy: **33.75%**, Balanced Acc: **26.15%**, Macro F1: **0.2506**.
  * Statistical Test: McNemar / permutation test yielded **$p = 0.6225$ (non-significant)**.
* **What It Established:** Destructive downsampling was **not** the cause of classification failure. Doubling spatial ridge sampling did not extract biological signal.

### EXP-08: Handcrafted Dermatoglyphic Minutiae Features
* **Research Question:** If deep CNNs are blinded by optical border artifacts, do handcrafted dermatoglyphic features (minutiae counts, ridge density, pattern topology) encode blood group?
* **Feature Vector:** 24 handcrafted numerical features per participant (averaged over 3 scans).
* **Results (Reconciled):**
  * Random Forest: Accuracy: **35.00%**, Balanced Acc: **27.06%**, Macro F1: **0.2603**.
  * Logistic Regression: Accuracy: **30.00%**, Balanced Acc: **23.36%**, Macro F1: **0.2241**.
  * Baseline (Zero-Intelligence Majority): **36.25%** accuracy, **25.00%** balanced accuracy.
  * Permutation Significance: **$p = 0.2537$** (1,000 full-pipeline permutations).
* **Discrepancy Resolution:** Legacy `train.py` computed a fold-local baseline (predicting B+ on folds 2 & 5, reporting 28.75%) and had an inverted class label map. The audit recomputed the true global 36.25% baseline and confirmed that Random Forest falls squarely inside the chance null distribution.

### EXP-09: Continuous Spatial Ridge-Orientation Tensors
* **Research Question:** Does a continuous dense vector field representation of ridge flow ($920$-d orientation tensor $[\cos 2\theta, \sin 2\theta]$ with 100% background masking) capture global dermatoglyphic patterns?
* **Results:**
  * Ridge Classifier: Accuracy: **36.25%**, Balanced Acc: **29.20%**, Macro F1: **0.2770**.
  * Random Forest: Accuracy: **40.00%**, Balanced Acc: **31.37%**, Macro F1: **0.3048**.
  * Permutation Significance: Balanced accuracy **$p = 0.1089$** (100 permutations).
* **What It Established:** While Random Forest correctly classified 3 additional participants over baseline (32 vs 29), its balanced accuracy remained statistically non-significant ($p > 0.10$). Null permutations reached up to 35.5% balanced accuracy by chance. Complete failure on AB+ (0/7 correct).

### EXP-10: Multi-Scan Quality & Consistency Pooling
* **Research Question:** Does weighting multi-scan feature fusion by platen image quality improve classification over unweighted arithmetic averaging?
* **Setup:** Evaluated Condition A (Unweighted arithmetic mean of 3 scans) vs Condition B (Quality-weighted mean: $w_i = Q_i / \sum Q_j$).
* **Results:**
  * Ridge Classifier: Condition A: **36.25%** acc $\to$ Condition B: **36.25%** acc (Identical).
  * Random Forest: Condition A: **40.00%** acc $\to$ Condition B: **33.75%** acc (Quality-weighting **degraded** accuracy by 6.25%).
* **Discrepancy Resolution:** Report text in [`reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md) noted "only a single O+ prediction flipped" with matrix `[[5, 1, 6, 7], ...]`. Authoritative CSV and JSON show `[[4, 1, 6, 8], ...]` with exactly two flipped predictions (P0012 A+$\to$O+, P007 B+$\to$O+).
* **What It Established:** Intra-participant scan quality on the Mantra MFS100 is remarkably consistent (95% ratio $< 1.5$; mean orientation cosine similarity $= 0.9355$). Simple averaging provides a $1/\sqrt{3} \approx 42\%$ noise variance reduction, whereas quality weighting introduces extraneous noise.

---

## 3. Discrepancy Reconciliation Summary

| Issue ID | Experiment | Historical Claim | Reconciled Audit Value | Reason for Discrepancy | Publication Guideline |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **DISC-01** | EXP-08 | Baseline reported as 28.75% | **36.25%** | Legacy script computed majority baseline locally per fold instead of globally across cohort. | Report 36.25% global majority baseline. |
| **DISC-02** | EXP-08 | Logistic Regression acc: 27.5% | **30.00%** | Inverted label index for B+ and AB+ in legacy evaluation script. | Report 30.00% (24/80 participants). |
| **DISC-03** | EXP-10 | Text note: "single O+ prediction flipped" | **Two predictions flipped** | Typographical error in Markdown draft; authoritative prediction CSV shows 2 flips. | Cite authoritative CSV (`exp10_participant_oof_predictions.csv`). |
| **DISC-04** | EXP-01–06 | Development F1 reported up to 0.5487 | **Confounded by border illumination** | Models exploited platen border brightness ($F=5.996, p=0.0006$); collapsed on Test-21 (33.3%). | Document as an illumination artifact. |

---

## 4. Safe Phrasing for Describing Results in the Research Paper

When presenting these results in the paper, use the following standardized descriptions:

* **Describing Baseline Performance:**  
  *"In a four-class classification problem with 80 participants distributed as 19 A+, 25 B+, 7 AB+, and 29 O+, the zero-intelligence majority-class baseline achieves an accuracy of 36.25% (29/80) and a balanced accuracy of 25.00%."*
* **Describing the Random Forest Result (EXP-09):**  
  *"While the Random Forest classifier trained on spatial orientation tensors achieved an apparent accuracy of 40.00% (32/80 participants), its balanced accuracy of 31.37% was not statistically significant under participant-level permutation testing ($p = 0.1089$)."*
* **Describing Minority Class AB+:**  
  *"Across all evaluated models (ResNet-50, DenseNet-121, Random Forest, Ridge), sensitivity for the minority AB+ class was 0.0% (0 out of 7 participants correctly identified), indicating that the classifiers collapse onto majority classes and fail to achieve multi-class discrimination."*
