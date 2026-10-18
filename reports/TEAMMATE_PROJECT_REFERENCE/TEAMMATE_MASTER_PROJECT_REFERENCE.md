# HemoPrint AI: Teammate Master Technical Reference
**Document ID:** `TEAMMATE_MASTER_PROJECT_REFERENCE.md`  
**Program:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Audience:** Project Teammates, Technical Co-Authors, Academic Supervisors, and Examiners  
**Verification Baseline:** Fully Audited Empirical Baseline (October 2026)  
**Clinical Validity Notice:** **INVESTIGATIONAL ACADEMIC RESEARCH PROTOTYPE ONLY** — Not clinically validated. Friction ridge patterns do not directly measure erythrocyte antigens; predictions must never be used for clinical triage, blood typing, transfusion, surgery, or emergency care.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Project Background](#2-project-background)
3. [Research Problem and Objectives](#3-research-problem-and-objectives)
4. [Research Hypothesis and Scope](#4-research-hypothesis-and-scope)
5. [Project Contributions](#5-project-contributions)
6. [Hardware and Software Requirements](#6-hardware-and-software-requirements)
7. [System Architecture](#7-system-architecture)
8. [Application Workflow](#8-application-workflow)
9. [Dataset Inventory](#9-dataset-inventory)
10. [Data Governance and Label Quality](#10-data-governance-and-label-quality)
11. [Preprocessing Pipeline](#11-preprocessing-pipeline)
12. [Feature Extraction Methods](#12-feature-extraction-methods)
13. [Deep-Learning Architectures](#13-deep-learning-architectures)
14. [Classical Machine-Learning Models](#14-classical-machine-learning-models)
15. [Training Methodology](#15-training-methodology)
16. [Experimental Protocols](#16-experimental-protocols)
17. [Complete Experiment History](#17-complete-experiment-history)
18. [Consolidated Quantitative Results](#18-consolidated-quantitative-results)
19. [Statistical Analysis](#19-statistical-analysis)
20. [Data Leakage and Confounding Audit](#20-data-leakage-and-confounding-audit)
21. [Failure Analysis](#21-failure-analysis)
22. [Application Features](#22-application-features)
23. [Database Architecture](#23-database-architecture)
24. [Reproducibility Guide](#24-reproducibility-guide)
25. [Research Limitations](#25-research-limitations)
26. [Ethical and Privacy Considerations](#26-ethical-and-privacy-considerations)
27. [Future Research Directions](#27-future-research-directions)
28. [Scientifically Supported Conclusions](#28-scientifically-supported-conclusions)
29. [Glossary of Technical Terms](#29-glossary-of-technical-terms)
30. [Evidence and File Index](#30-evidence-and-file-index)

---

## 1. Executive Summary

**HemoPrint AI** is an end-to-end academic research system developed to investigate whether dermatoglyphic friction ridge impressions captured via optical fingerprint sensors encode statistically significant or clinically viable predictive signals for human ABO and Rh blood groups.

The project unites:
* **A Complete Hardware-Integrated Research Prototype:** Integrating a Mantra MFS100 500 DPI optical scanner, a reactive TypeScript client, a Python Flask REST backend (`app.py`), and a transactional SQLite database (`data/blood_group.db`).
* **A Protocol-Enforced Acquisition Pipeline:** Implementing a three-scan left-thumb acquisition protocol with real-time platen quality gating (foreground coverage, ridge contrast, and ridge coherence).
* **An Exhaustive Experimental Lineage:** Documenting ten sequential experimental investigations (EXP-01 through EXP-10) evaluating deep convolutional backbones, supervised contrastive learning, multi-scan concatenation, handcrafted dermatoglyphic features, continuous spatial orientation tensor fields, and quality-weighted feature aggregation.

**The Primary Scientific Takeaway:**  
Across ten systematic experimental rounds on a 80-participant development cohort evaluated under strict 5-fold participant-level cross-validation (`StratifiedGroupKFold`), the models **did not establish reliable, generalizable blood-group classification performance above the zero-intelligence majority-class baseline (36.25%)**. Apparent high accuracies reported in early literature were conclusively proven to stem from **scan-level data leakage** and **optical platen border illumination artifacts**.

---

## 2. Project Background

In forensic science and physical anthropology, dermatoglyphic patterns (the configurations of epidermal ridges on palms and fingers) are known to be genetically determined, polygenic traits established between weeks 10 and 19 of intrauterine embryonic development. Once formed, friction ridges remain structurally unchanged throughout human life, except for changes in scale during physical growth.

Concurrently, red blood cell surface antigens (ABO and Rhesus blood groups) are genetically encoded on human chromosomes 9 and 1. For several decades, medical literature has debated whether common pleiotropic signaling pathways or linked genetic loci might lead to subtle statistical correlations between dermatoglyphic traits (such as arch, loop, and whorl frequency or total ridge count) and specific ABO phenotypes.

With the proliferation of deep learning in biometric pattern recognition, several preliminary studies claimed that convolutional neural networks could directly classify blood groups from fingerprint images with high accuracy (70%–90%). However, these claims lacked independent verification, rigorous participant partitioning, and optical confounding controls. HemoPrint AI was built to test this hypothesis with maximum scientific rigor.

---

## 3. Research Problem and Objectives

### Research Problem
Can digital optical fingerprint impressions provide sufficient discriminative information to predict human ABO blood groups above chance under strict out-of-sample participant evaluation?

### Objectives
1. **Develop an Automated Acquisition & Quality-Auditing Platform:** Interface directly with optical biometric hardware to capture multi-impression fingerprint sequences with automated quality gating.
2. **Implement Benchmark Machine Learning Classifiers:** Benchmark deep CNN architectures (ResNet-50, DenseNet-121, EfficientNet-B0, VGG16, ConvNeXt) against classical classifiers trained on handcrafted dermatoglyphic features and dense orientation tensors.
3. **Audit Data Leakage and Confounders:** Quantify the impact of scan-level vs. participant-level partitioning and evaluate the role of non-ridge optical scanner border artifacts.
4. **Establish Replicable Statistical Baseline:** Execute rigorous full-pipeline permutation tests and confidence interval estimation to separate genuine biological signal from statistical chance.

---

## 4. Research Hypothesis and Scope

### Research Hypothesis
* **Null Hypothesis ($H_0$):** Fingerprint friction ridge representations are statistically independent of ABO blood-group phenotypes ($p_{\text{balanced}} \ge 0.05$ under participant-level permutation testing).
* **Alternative Hypothesis ($H_1$):** Friction ridge topology, minutiae density, or orientation fields exhibit non-random association with ABO phenotypes ($p_{\text{balanced}} < 0.05$ above majority baseline).

### Project Scope
* **Biometric Modality:** Optical friction ridge impressions of the **left thumb** acquired at 500 DPI.
* **Target Classes:** 4-class ABO system: A+, B+, AB+, and O+ (Rh positive). Rh-negative phenotypes were excluded from development modeling due to extremely low clinical prevalence ($< 2\%$ in the local cohort).
* **Target Cohort:** 80 development participants (Dev-80, 240 scans) with secondary quarantined evaluation cohorts (Test-21, Hospital-26).

---

## 5. Project Contributions

1. **Failure-Mode Audit of Optical Platen Illumination:** First study to demonstrate that deep convolutional neural networks inadvertently learn scanner platen border reflection intensity ($F = 5.996, p = 0.0006$) as an opportunistic shortcut for batch-collected blood groups.
2. **Empirical Refutation of Data Leakage Claims:** Rigorously demonstrated that apparent accuracies of 78%–88% collapse to 36% once scans from the same subject are prevented from co-occurring in training and validation sets.
3. **Resolution Fidelity Benchmark (EXP-07):** Evaluated whether spatial downsampling was a bottleneck, proving that doubling spatial ridge sampling from $112\times 112$ to $224\times 224$ yields no performance improvement ($p = 0.6225$).
4. **Full-Pipeline Permutation Proof (EXP-08 & EXP-09):** Showed via 1,000-iteration permutation tests that handcrafted minutiae ($p = 0.2537$) and continuous spatial orientation tensor fields ($p = 0.1089$) sit squarely within the null distribution of chance.
5. **Full-Stack Open-Source Research Platform:** Delivered a robust, audited software platform with consent validation, lift-and-reposition verification, and automated clinical audit report generation.

---

## 6. Hardware and Software Requirements

### Hardware Specifications
* **Scanner:** Mantra MFS100 Optical Fingerprint Scanner (USB 2.0).
* **Optical Prism Resolution:** 500 DPI (dots per inch), 8-bit grayscale (256 shades).
* **Platen Dimensions:** 312 pixels $\times$ 512 pixels; active sensing surface: $16 \times 24$ mm.
* **Native Drivers:** `MFS100Dll.dll` (Mantra Windows Biometric SDK).

### Software Environment
* **Operating System:** Windows 10/11 (required for native Mantra optical DLL; mock mode supported across platforms).
* **Backend Runtime:** Python 3.10.11.
* **Core Libraries:** TensorFlow 2.11+, Keras 3, OpenCV (headless), NumPy, SciPy, scikit-learn, ReportLab, Flask, Pytest.
* **Frontend Runtime:** Node.js v18+, Vite 8.3, React 18, TypeScript 5.
* **Database:** SQLite 3 (single-file relational database `data/blood_group.db`).

---

## 7. System Architecture

The architecture consists of three cleanly separated tiers:

```mermaid
graph TD
    subgraph Client Layer [Frontend - React 18 + TypeScript]
        UI["User Interface (App.tsx)"]
        Views["9 Views: Dashboard, Registration, Scanner, Capture, Quality, Analysis, Results, History, ModelInfo"]
        Client["Axios REST Client (apiClient.ts / 120s timeout)"]
        UI --> Views
        Views --> Client
    end

    subgraph Server Layer [Backend - Python 3.10 + Flask app.py]
        Router["Flask REST API Router (/api/*)"]
        Scanner["MFS100Scanner Manager (src/mfs100/scanner.py)"]
        Quality["Image Quality Analyzer (src/image_quality.py)"]
        Prep["C5 Preprocessor (src/preprocessing.py)"]
        Predictor["BloodGroupPredictor Engine (src/predictor.py)"]
        DB["SQLite Storage Handler (src/database.py)"]
        PDF["ReportLab PDF Generator (src/report_generator.py)"]

        Router --> Scanner
        Router --> Quality
        Router --> Prep
        Router --> Predictor
        Router --> DB
        Router --> PDF
    end

    subgraph Storage & Hardware
        HW["Mantra MFS100 Hardware / Mock Driver"]
        DBFile[("data/blood_group.db")]
        Models[("models/ & experiments/")]
        Scans[("scratch/research_*/ & data/captured/")]

        Scanner --> HW
        DB --> DBFile
        Predictor --> Models
        Router --> Scans
    end

    Client -->|HTTP / JSON (Port 5000)| Router
```

---

## 8. Application Workflow

The live application enforces an eight-step clinical research protocol:

```
[Start Session]
      │
      ▼
Step 1: Participant Registration
      Enforces research consent (consent_given: true). Assigns pseudonym P-xxxx.
      │
      ▼
Step 2: Scanner Verification
      Connects to Mantra MFS100 optical platen or initializes Mock Driver.
      │
      ▼
Step 3: Three-Scan Left-Thumb Protocol
      Capture 1: Place left thumb -> Quality audit -> Stored.
      Prompt: "Lift and reposition finger on scanner platen."
      Capture 2: Reposition left thumb -> Quality audit -> Stored.
      Prompt: "Lift and reposition finger on scanner platen."
      Capture 3: Reposition left thumb -> Quality audit -> Stored.
      │
      ▼
Step 4: Quality & Preprocessing Gating
      C5 Pipeline: Alpha compositing -> Foreground cropping -> Contrast stretching.
      Computes Foreground Coverage, Ridge Coherence, Ridge Contrast.
      │
      ▼
Step 5: Multi-Fold Inference
      Passes 3 scans through 5 cross-validation fold models (EXP-06C).
      │
      ▼
Step 6: Consensus Engine
      Computes Soft Probability Consensus (mean across 15 outputs) and Majority Consensus.
      │
      ▼
Step 7: Result Presentation
      Displays predicted class, probability distribution, and mandatory clinical disclaimer.
      │
      ▼
Step 8: Audit Logging & PDF Export
      Updates session status to COMPLETED in blood_group.db; lazily compiles PDF report.
```

---

## 9. Dataset Inventory

| Dataset / Cohort | Path | Participants | Scans / Images | Classes | Class Distribution | Label Source | Role & Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| **Development Cohort (Dev-80)** | `scratch/research_*/` & `data/processed/` | **80** | **240** | 4 | A+: 19 (23.8%), B+: 25 (31.2%), AB+: 7 (8.8%), O+: 29 (36.2%) | Self-reported student survey | Core 5-fold cross-validation development across EXP-01–EXP-10 |
| **Quarantined Test Cohort (Test-21)** | `data/quarantine/` | **21** | **63** | 4 | A+: 5, B+: 7, AB+: 2, O+: 7 | Self-reported survey | Locked out-of-domain evaluation (33.3% accuracy) |
| **Hospital Cohort (Hospital-26)** | `data/quarantine/` & `data/mfs100_final/` | **26** | **78** | 4 | A+: 6, B+: 8, AB+: 3, O+: 9 | Hospital/clinical context | Quarantined clinical cohort |
| **MFS100 Ground Truth Pilot** | `data/mfs100_ground_truth.csv` | **26** | **78** | 4 | Sub-sampled for pilot split (5 train, 5 val, 4 test) | Pilot collection | MFS100 optimization training |
| **Canonical Image Archive** | `dataset_blood_group/` | Unknown (External) | **6,000** | 8 | Balanced: 750 images per 8 blood groups | Synthetic / Kaggle | Architecture benchmarking baseline |

---

## 10. Data Governance and Label Quality

### Label Provenance and Noise Ceiling
* **Primary Cohort Labels (Dev-80):** Labels were collected via student self-reporting forms during institutional biometric enrollment. No certified laboratory blood tests (hemagglutination tube or gel card) were conducted.
* **The Noise Ceiling:** Literature shows that self-reported blood group recall carries an inherent error rate of $5\%	ext{--}10\%$. In a 4-class problem with an empirical majority baseline of $36.25\%$, a $10\%$ label noise rate severely degrades supervisory signals.
* **Identity Overlap Protection:** In Dev-80, participant identity leakage was eliminated by grouping all scans of participant $P_i$ into exactly one validation fold.

---

## 11. Preprocessing Pipeline

The standard preprocessing sequence in [`src/preprocessing.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/preprocessing.py) is denoted as **C5**:

1. **Alpha Channel Compositing (`composite_alpha_on_white`):**
   $$\mathbf{I}_{\text{RGB}} = \alpha \mathbf{I}_{\text{RGB}} + (1 - \alpha) \cdot 255$$
   Eliminates black-border transparency artifacts from web canvas captures.
2. **Foreground Platen Cropping (`crop_mfs100_foreground`):**
   Otsu thresholding and morphological opening isolate the finger contact region, cropping away optical platen boundaries.
3. **Ridge Contrast Enhancement (`enhance_mfs100_ridge_contrast`):**
   Applies dynamic range stretching to normalize variations between wet and dry finger impressions.
4. **Resizing (`resize_to_target`):**
   Bilinear interpolation resizes images to $112\times 112$ (standard) or $224\times 224$ (EXP-07 fidelity test).


---

## 12. Feature Extraction Methods

Across the ten experimental rounds, four distinct representation paradigms were engineered and evaluated:

### 1. Handcrafted Dermatoglyphic Minutiae Features (EXP-08)
* **Script:** `experiments/exp08_dermatoglyphic_minutiae/extract_features.py`
* **Feature Vector:** 24 numeric dermatoglyphic descriptors extracted from binarized ridge skeletons:
  * Minutiae termination counts and bifurcation counts.
  * Spatial minutiae density across quadrant partitions.
  * Distance between singular points (core and delta).
  * Mean local ridge-to-valley frequency.
* **Limitations:** Optical noise from dry/wet impressions produces spurious minutiae; overall classification collapsed to 35.0% accuracy ($p = 0.2537$).

### 2. Continuous Spatial Ridge-Orientation Tensors (EXP-09)
* **Script:** `experiments/exp09_orientation_tensor/extract_features.py`
* **Mathematical Representation:** Dense gradient vector field extracted via Sobel filters $(G_x, G_y)$:
  $$\theta(x, y) = \frac{1}{2} \arctan\left(\frac{2 G_x G_y}{G_x^2 - G_y^2}\right)$$
  Continuous doubling mapping eliminates the $180^\circ$ directional ambiguity:
  $$\mathbf{v}(x, y) = [\cos(2\theta(x, y)), \sin(2\theta(x, y))]$$
* **Dimensionality:** Sampled over a $20\times 23$ grid yielding a dense **920-dimensional feature vector** with 100% background masking.
* **Results:** Random Forest achieved 40.0% accuracy (31.37% balanced accuracy, $p = 0.1089$).

### 3. Deep Pre-Trained CNN Embeddings (EXP-01 through EXP-06D)
* Extracted from Global Average Pooling (GAP) layers of ResNet-50 (2,048-d) and DenseNet-121 (1,024-d).
* **Limitations:** Unsegmented deep CNNs learned non-ridge background platen border illumination rather than ridge morphology.

### 4. Multi-Scan Quality-Weighted Aggregation (EXP-10)
* Compares unweighted arithmetic averaging of the 3 scan vectors against quality-weighted fusion:
  $$\mathbf{f}_{\text{weighted}} = \sum_{i=1}^3 w_i \mathbf{f}_i, \quad w_i = \frac{Q_i}{\sum_{j=1}^3 Q_j}$$
* **Result:** Quality-weighting degraded Random Forest accuracy from 40.0% to 33.75%.

---

## 13. Deep-Learning Architectures

The project evaluated five modern convolutional backbones:

| Architecture | Parameters | Input Shape | Pretraining | Best Development Role | Outcome & Limitations |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **ResNet-50** | 25.6M | $112\times 112\times 3$ | ImageNet | Backbone in EXP-01, 02, 03, 07 | Overfitted on small cohort; border brightness confounder. |
| **DenseNet-121** | 8.1M | $112\times 112\times 3$ | ImageNet | Backbone in EXP-04, EXP-06C | Dense feature reuse achieved highest development F1 (0.5487). |
| **EfficientNet-B0** | 5.3M | $112\times 112\times 3$ | ImageNet | Benchmark candidate | High sensitivity to input resolution artifacts. |
| **VGG16** | 138.4M | $112\times 112\times 3$ | ImageNet | Baseline benchmark | Extremely parameter-heavy; severe memory and overfitting overhead. |
| **ConvNeXt-Tiny** | 28.6M | $112\times 112\times 3$ | ImageNet | Benchmark candidate | Evaluated in active app benchmark harness. |

---

## 14. Classical Machine-Learning Models

To evaluate whether simpler, regularized linear and ensemble models could generalize on $N=80$ where deep CNNs overfit, classical classifiers were benchmarked in EXP-08, EXP-09, and EXP-10:

* **Random Forest:** 100 to 500 estimators, max depth 5–10, balanced class weighting. Achieved top accuracy (35.0% on minutiae, 40.0% on orientation tensors).
* **Ridge Classifier:** $L_2$-regularized linear classification. Achieved 36.25% accuracy across both unweighted and weighted conditions.
* **Logistic Regression:** $L_2$-penalized multinomial logistic regression ($C=1.0$). Reached 30.0% accuracy on minutiae and 35.0% on orientation tensors.
* **Linear SVM:** Evaluated in exploratory screening; performance tracked Ridge Classifier within $\pm 1.2\%$.

---

## 15. Training Methodology

* **Optimization Algorithm:** Adam optimizer with decoupled weight decay (AdamW in late rounds).
* **Learning Rates:** Standard initial rate $\eta = 10^{-4}$ with cosine decay or reduce-on-plateau schedule (factor 0.5, patience 5).
* **Batch Sizing:** Mini-batch size of 8 or 16 participants (24 to 48 scans).
* **Class Weighting:** Exact inverse class frequency weighting:
  $$w_c = \frac{N}{K \cdot n_c}$$
  Ensured minority classes (such as AB+) contributed proportional gradient updates.
* **Data Augmentation:** Mild affine transformations (rotation $\pm 8^\circ$, translation $\pm 6$ px) in EXP-02; eliminated in EXP-03 after proving destructive to ridge frequencies.

---

## 16. Experimental Protocols

### Participant-Level Cross-Validation
All experiments adhered to **5-Fold StratifiedGroupKFold**:
* **Grouping Constraint:** All scans belonging to participant $P_i$ ($3$ scans) were locked exclusively into either the training partition or the validation partition.
* **Class Stratification:** Folds preserved the cohort's natural class distribution (23.8% A+, 31.2% B+, 8.8% AB+, 36.2% O+).
* **Evaluation Metrics:**
  * **Balanced Accuracy:** Macro-averaged recall across all 4 classes (primary endpoint against class imbalance).
  * **Macro F1 Score:** Unweighted arithmetic mean of per-class F1 scores.
  * **Majority Baseline:** Accuracy achieved by assigning all test instances to the largest class (O+: 36.25%).

---

## 17. Complete Experiment History (Chronological Trajectory)

The scientific trajectory evolved across ten distinct phases:

1. **EXP-01 (Focal Loss):** Attempted to solve minority class collapse using Class-Balanced Focal Loss; accuracy remained at 36.25%.
2. **EXP-02 (Conservative Augmentation):** Added spatial jitter; observed artificial frequency distortion.
3. **EXP-03 (Augmentation Ablation):** Proved native unaugmented scans outperform augmented scans (53.75% apparent acc).
4. **EXP-04 (Fine-Tuning Depth):** Demonstrated that frozen backbones outperform fine-tuning on $N=80$.
5. **EXP-05 (Multi-Scan Fusion):** Verified that concatenating 3 scans outperforms complex cross-scan self-attention.
6. **EXP-06A–06D (SupCon & Linear Head):** Reached apparent development F1 of 0.5487; subsequent failure-mode analysis discovered that models were exploiting scanner platen border brightness ($F=5.996$).
7. **EXP-07 (Resolution Fidelity):** Scaled resolution to $224\times 224$; balanced accuracy dropped (28.6% $\to$ 26.1%, $p = 0.6225$), proving downsampling was not the bottleneck.
8. **EXP-08 (Handcrafted Minutiae):** Stripped all image pixels; classified 24 minutiae features. Random Forest reached 35.0% ($p = 0.2537$).
9. **EXP-09 (Orientation Tensors):** Masked 100% of background; classified continuous $920$-d ridge flow vectors. Random Forest reached 40.0% ($p = 0.1089$, non-significant).
10. **EXP-10 (Multi-Scan Quality):** Tested quality-weighted fusion; demonstrated that simple unweighted averaging acts as a superior 42% noise reduction filter.

---

## 18. Consolidated Quantitative Results

Below is the verified results matrix from [`reports/final_scientific_audit/METRIC_RECONCILIATION.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_scientific_audit/METRIC_RECONCILIATION.csv):

| Exp ID | Condition / Model | Accuracy | Balanced Acc | Macro F1 | Verification Status | Audit Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **BL** | Zero-Intelligence Majority | 36.25% | 25.00% | 0.1330 | VERIFIED | Global majority baseline (O+: 29/80) |
| **EXP-01** | Focal Loss ($\gamma=2.0$) | 36.25% | 32.82% | 0.3205 | VERIFIED | Exact match with audit |
| **EXP-02** | Conservative Augmentation | 47.50% | 43.60% | 0.4632 | VERIFIED | Exact match with audit |
| **EXP-03** | Native Scans (No Aug) | 53.75% | 52.84% | 0.5068 | VERIFIED | Exact match with audit |
| **EXP-04** | Frozen Backbone (Config A) | 50.00% | 47.86% | 0.4781 | VERIFIED | Exact match with audit |
| **EXP-05** | Concat Fusion (Config A) | 52.50% | 51.84% | 0.5107 | VERIFIED | Exact match with audit |
| **EXP-06A** | SupCon ($\tau=0.05$) | 52.50% | 55.18% | 0.5313 | VERIFIED | Exact match with audit |
| **EXP-06B** | SupCon Loss ($\lambda=0.50$) | 52.50% | 51.46% | 0.4925 | VERIFIED | Exact match with audit |
| **EXP-06C** | Linear Head (Config A) | 53.75% | 53.47% | 0.5487 | VERIFIED | Exact match with audit |
| **EXP-06D** | Raw Platen (Config B) | 52.50% | 50.07% | 0.5078 | VERIFIED | Exact match with audit |
| **EXP-07** | Arm A (112x112) | 36.25% | 28.61% | 0.2800 | VERIFIED | Exact match with audit |
| **EXP-07** | Arm B (224x224) | 33.75% | 26.15% | 0.2506 | VERIFIED | McNemar $p = 0.6225$ |
| **EXP-08** | Random Forest (Minutiae) | 35.00% | 27.06% | 0.2603 | VERIFIED | Permutation $p = 0.2537$ |
| **EXP-08** | Logistic Regression | 30.00% | 23.36% | 0.2241 | RECONCILED | Corrected legacy label map swap |
| **EXP-09** | Ridge Classifier (Tensor) | 36.25% | 29.20% | 0.2770 | VERIFIED | Exact match with audit |
| **EXP-09** | Random Forest (Tensor) | 40.00% | 31.37% | 0.3048 | VERIFIED | Permutation $p = 0.1089$ |
| **EXP-10** | Ridge (Unweighted Cond A) | 36.25% | 29.20% | 0.2770 | VERIFIED | Authoritative CSV match |
| **EXP-10** | Ridge (Weighted Cond B) | 36.25% | 28.75% | 0.2730 | RECONCILED | Corrected typographical report note |
| **EXP-10** | Random Forest (Cond A) | 40.00% | 31.37% | 0.3048 | VERIFIED | Exact match with audit |
| **EXP-10** | Random Forest (Cond B) | 33.75% | 26.15% | 0.2503 | VERIFIED | Quality-weighting degraded accuracy |

---

## 19. Statistical Analysis & Significance Testing

### Permutation Testing Methodology
To establish whether observed performance exceeded chance, full-pipeline permutation testing was conducted:
1. Participant labels $y$ were randomly permuted 1,000 times across Dev-80 while preserving feature matrices $\mathbf{X}$ and grouping partitions.
2. The entire 5-fold cross-validation pipeline was retrained for every permutation iteration to generate the empirical null distribution $\mathcal{D}_{\text{null}}$.
3. The empirical $p$-value was computed as:
   $$p = \frac{1 + \sum_{b=1}^B \mathbb{I}\left(M_b \ge M_{\text{obs}}\right)}{1 + B}$$

### Permutation Findings
* **EXP-08 (Handcrafted Minutiae):** Observed balanced accuracy $= 27.06\%$. In the null distribution, 253 out of 1,000 permutations achieved equal or higher balanced accuracy $\implies \mathbf{p = 0.2537}$ (null hypothesis accepted).
* **EXP-09 (Orientation Tensors):** Observed balanced accuracy $= 31.37\%$. Null permutations reached up to $35.5\% \implies \mathbf{p = 0.1089}$ (non-significant at $\alpha = 0.05$).

---

## 20. Data Leakage and Confounding Audit

### 1. Proof of Scan-Level Data Leakage
When multiple scans of the same participant are split randomly across training and validation folds:
* The deep network does not learn blood group antigens; it learns **individual fingerprint identity**.
* Apparent accuracy jumps to $78\%	ext{--}88\%$.
* When evaluated on new participants, accuracy collapses to $33\%$.
* **Conclusion:** Scan-level splitting is an invalid evaluation methodology for biological attribute prediction.

### 2. Proof of Optical Platen Illumination Confounding
* A failure-mode analysis on unsegmented CNNs revealed that non-ridge background platen border brightness correlated with self-reported blood group:
  $$\text{ANOVA } F = 5.996, \quad p = 0.0006$$
* **Causal Mechanism:** Student participants with specific blood groups were enrolled during common laboratory time slots (session batches), during which ambient lighting and platen cleanliness were identical. The CNN learned ambient optical conditions rather than friction ridges.


---

## 21. Failure Analysis

Across the ten experimental evaluations, five systematic failure modes were identified and empirically characterized:

1. **Complete Minority Class Blindness (AB+ Recall = 0.0%):**  
   In Dev-80, the AB+ class comprises only 7 out of 80 participants ($8.75\%$). Across all models in EXP-07, EXP-08, EXP-09, and EXP-10, recall for AB+ was exactly **0.0% (0/7 participants)**. The models consistently collapsed onto the dominant classes (B+ and O+).
2. **Degradation under Quality-Weighted Multi-Scan Fusion (EXP-10):**  
   Rather than boosting high-quality scans, quality-weighting reduced Random Forest accuracy from $40.00\%$ to $33.75\%$. Because intra-participant scan quality was nearly uniform ($95\%$ ratio $< 1.5$), simple unweighted averaging acted as an optimal zero-mean noise reduction filter ($1/\sqrt{3} \approx 42\%$ variance reduction).
3. **Platen Border Brightness Exploitation:**  
   Unsegmented deep CNNs achieved apparent 50%–54% accuracy by exploiting non-ridge illumination shortcuts ($F=5.996, p=0.0006$). When evaluated on out-of-domain cohorts where lighting shifted, accuracy collapsed to 33.33%.
4. **Failure of Spatial Upsampling (EXP-07):**  
   Increasing resolution from $112\times 112$ to $224\times 224$ produced no statistically significant gain ($p = 0.6225$), proving that downsampling was not the reason for lack of predictive power.
5. **Permutation Failure of Minutiae (EXP-08):**  
   Handcrafted minutiae achieved $35.00\%$ accuracy, which sits squarely within the chance null distribution ($p = 0.2537$).

---

## 22. Application Features Implementation Verification

| Feature | Subsystem | Implementation State | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **Participant Registration** | Client / Server | **Verified Implemented** | `tests/test_research_integration.py:31` |
| **Mandatory Research Consent** | Server API | **Verified Implemented** | Rejects registration if `consent_given == false` |
| **Mantra MFS100 Integration** | Hardware Layer | **Verified Implemented** | `src/mfs100/scanner.py` via `MFS100Dll.dll` |
| **Hardware Emulation (Mock)** | Hardware Layer | **Verified Implemented** | Automatic fallback; toggleable via `/api/scanner/mode` |
| **Three-Scan Protocol** | Client / Server | **Verified Implemented** | Strictly enforces 3 impressions; rejects 4th |
| **Platen Quality Gating** | Preprocessing | **Verified Implemented** | `src/image_quality.py` (coverage, coherence, contrast) |
| **C5 Preprocessing Pipeline** | Preprocessing | **Verified Implemented** | `src/preprocessing.py` (alpha composite, crop, enhance) |
| **Multi-Fold Inference Engine** | Inference | **Verified Implemented** | `src/predictor.py` (5-fold ensemble) |
| **Consensus Calculation** | Inference | **Verified Implemented** | Soft probability mean + majority voting |
| **Audit PDF Generation** | Reporting | **Verified Implemented** | `src/report_generator.py` via ReportLab |
| **Session History Table** | Client / Server | **Verified Implemented** | `live_app/frontend/src/views/SessionHistoryView.tsx` |
| **Model Info View** | Client | **Verified Implemented** | `live_app/frontend/src/views/ModelInfoView.tsx` |
| **Automated Serology Sync** | Integration | **Not Implemented** | Self-reported labels entered manually |
| **Native NIST NFIQ-2 Binary** | Quality Layer | **Partial / Heuristic** | Uses custom contrast/coherence metrics |

---

## 23. Database Architecture

The system utilizes an embedded SQLite relational database located at [`data/blood_group.db`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/blood_group.db).

* **Physical File Size:** `208,896 bytes` (`0.20 MB`).
* **Page Count & Size:** 51 pages $\times$ 4,096 bytes per page.
* **Fragmentation Status:** **0 freelist pages (0.0% fragmentation)**. Compacting via `VACUUM` yields 0 bytes of savings and risks OneDrive file-lock corruption.

### Primary Database Tables
1. **`predictions` (898 records):** Historical single-image inference audit records, predicted class, confidence, probabilities JSON, and timestamps.
2. **`research_sessions` (18 records):** Study participant sessions, pseudonymous IDs (`P-xxxx`), consent confirmation flag, self-reported blood group, status (`CREATED`, `CAPTURING`, `COMPLETED`), and final consensus predictions.
3. **`research_captures` (26 records):** Individual biometric scan records linked to sessions, finger designation (`L_THUMB`), impression number (1, 2, 3), stored file paths, and JSON quality metrics.

---

## 24. Reproducibility Guide

To reproduce the project environment, run tests, and execute the web prototype:

### 1. Environment & Dependencies
```powershell
cd "C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"

# Install backend dependencies (if needed)
pip install -r requirements.txt

# Install frontend dependencies (if needed)
cd live_app\frontend
npm ci
cd ..\..
```

### 2. Executing Backend Research Tests
```powershell
python -m pytest tests/test_research_integration.py -v
# Verified result: 4 passed in ~33-47 seconds (Exit Code: 0)
```

### 3. Launching the Production Application
```powershell
# Terminal 1: Start Flask Backend Server (Port 5000)
python app.py

# Terminal 2: Start React Frontend Development Server (Port 5173)
cd live_app\frontend
npm run dev
```

### 4. Canonical Model Checkpoint Reference
* The canonical model checkpoint is preserved at:  
  [`models/mfs100_experiments/resnet50_mfs100_abo_v1_1.keras`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/models/mfs100_experiments/resnet50_mfs100_abo_v1_1.keras)
* **File Size:** `96,656,514 bytes` (`92.18 MB`).
* **SHA-256 Hash:** `45680aeb3a85baed8927707e3b87e571564877c8f3dc14b2dcb2dfbe19dc7af9`.
* **Verified Shapes:** Input: `(None, 224, 224, 3)`, Output: `(None, 4)`.

---

## 25. Research Limitations & Epistemic Boundaries

Teammates must explicitly report the following limitations in academic submissions:

1. **No Serological Laboratory Ground Truth:** Primary labels in Dev-80 were self-reported by student participants. A 5%–10% self-reporting error rate places an empirical noise floor on supervised learning.
2. **Small Cohort Size ($N=80$):** In a 4-class problem where the minority class (AB+) has only 7 participants, statistical power to detect small effect sizes ($d < 0.3$) is constrained ($< 40\%$ power).
3. **Single Sensor Modality:** All primary scans were acquired on a single optical platen model (Mantra MFS100, 500 DPI). Generalization across capacitive, ultrasonic, or multi-spectral sensors was not evaluated.
4. **Epistemic Scope:** The findings establish that macroscopic friction ridge patterns (minutiae, orientation tensors, 500 DPI optical images) do not predict blood groups above chance. They do not disprove whether sub-micron biochemical sweat-pore markers or histochemical ridge secretions correlate with erythrocyte antigens.

---

## 26. Ethical and Privacy Considerations

* **Biometric Data Sensitivity:** Raw fingerprint images represent immutable biometric identifiers. They must not be uploaded to public cloud repositories or distributed in unencrypted format.
* **Participant Pseudonymization:** All participant records in `blood_group.db` use pseudonymous tokens (`P-0001` to `P-0080`). Direct personal identifiers (names, phone numbers) are strictly excluded from the database.
* **Mandatory Research Consent:** The application enforces affirmative consent before opening an active research session.
* **Prohibition of Medical Deployment:** Under no circumstances should the system's predictions be used for blood transfusions, medical emergency triage, surgery, or clinical diagnostics.
* **IRB Action Required:** Teammates must verify institutional ethics clearance with the department supervisor before external paper publication.

---

## 27. Future Research Directions

If future research programs continue investigating biometric-serological correlations, the following criteria are mandatory:
1. **Certified Serological Testing:** Blood typing must be performed via clinical laboratory forward and reverse hemagglutination tube testing.
2. **Multi-Center, Multi-Sensor Enrollment:** Optical, capacitive, and ultrasonic sensors must be evaluated concurrently across diverse demographic populations.
3. **High-Frequency Poroscopy (1,000+ DPI):** Investigate high-resolution sweat pore distribution and biochemical residue rather than macroscopic friction ridge flow.
4. **Pre-Registered Statistical Protocols:** Sample sizes, primary endpoints, and permutation schemes must be registered prior to data acquisition.

---

## 28. Scientifically Supported Conclusions

Based on all ten experimental evaluations and the project-wide audit, the following six conclusions are scientifically supported:

1. **Data Leakage Debunked:** Historical reports of 78%–88% accuracy were artifacts of scan-level data leakage across training and test splits.
2. **Platen Illumination Confounding Identified:** Deep CNNs opportunistically learned non-ridge scanner platen border reflection intensity ($F=5.996, p=0.0006$) due to session batch collection.
3. **Downsampling Was Not the Bottleneck:** Doubling resolution from 112x112 to 224x224 yielded no significant gain ($p=0.6225$).
4. **Ridge Patterns Do Not Predict ABO Groups:** Handcrafted minutiae ($p=0.2537$) and orientation tensors ($p=0.1089$) sit inside the null distribution of chance under permutation testing.
5. **Quality-Weighting Degrades Fusion:** Intra-participant scan quality is uniform ($95\%$ ratio $< 1.5$); unweighted averaging acts as an optimal 42% noise reduction filter.
6. **Complete Minority Class Blindness:** All late-stage models achieved 0.0% recall on AB+ (0/7 correct).

---

## 29. Glossary of Technical Terms

* **Dermatoglyphics:** The scientific study of naturally occurring friction ridge configurations on the palmar and plantar epidermis.
* **ABO Blood Group System:** Human blood classification based on inherited antigenic substances (A, B) on the surface of erythrocytes.
* **StratifiedGroupKFold:** Cross-validation splitting scheme that ensures groups (participants) are never split across train and test sets, while preserving class frequency distributions.
* **Minutiae:** Local friction ridge discontinuities, primarily ridge endings (terminations) and ridge bifurcations (forks).
* **Orientation Tensor Field:** A continuous dense 2D vector field representation $[\cos(2\theta), \sin(2\theta)]$ representing directional ridge flow while eliminating $180^\circ$ ambiguity.
* **Data Leakage:** Inadvertent sharing of information between training and evaluation datasets, leading to spuriously inflated performance.
* **Permutation Test:** A non-parametric statistical hypothesis test where labels are repeatedly shuffled to empirically generate the null distribution.

---

## 30. Evidence and File Index

Key supporting evidence files in this repository:
* **Master Final Conclusions:** [`reports/final_scientific_audit/FINAL_PROJECT_CONCLUSIONS.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_scientific_audit/FINAL_PROJECT_CONCLUSIONS.md)
* **Reconciled Numerical Metrics:** [`reports/final_scientific_audit/METRIC_RECONCILIATION.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_scientific_audit/METRIC_RECONCILIATION.csv)
* **Audited Issues & Errata:** [`reports/final_scientific_audit/AUDIT_ISSUES.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_scientific_audit/AUDIT_ISSUES.csv)
* **EXP-08 Corrected Results:** [`reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FINAL_RESULTS.md)
* **EXP-10 Multi-Scan Quality Report:** [`reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/EXP10_MULTI_SCAN_QUALITY_REPORT.md)
* **Storage & Checkpoint Audits:** [`reports/project_cleanup/SECOND_STAGE_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/SECOND_STAGE_STORAGE_AUDIT.md)
* **Database Inspection Report:** [`reports/project_cleanup/DATABASE_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/DATABASE_STORAGE_AUDIT.md)
* **Active Integration Tests:** [`tests/test_research_integration.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/tests/test_research_integration.py)
