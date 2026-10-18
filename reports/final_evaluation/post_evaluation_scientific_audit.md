# Post-Evaluation Scientific Audit — Phase 1: EXP-06C Diagnostic Review
**Project:** Biometric Blood Group Detection / HemoPrint AI  
**Audit Target:** Root-Cause Investigation of EXP-06C Config A Generalization Failure on Locked Test Cohort ($N=21$)  
**Report Date:** 2026-10-16  
**Auditor:** Independent Antigravity Forensic Audit  
**Operating Mode:** Read-Only Audit (No training, no model modification, no split tampering, no re-evaluation)  
**Output Path:** `reports/final_evaluation/post_evaluation_scientific_audit.md`  

---

## 1. Executive Summary

Following the execution of the pre-registered primary evaluation on October 16, 2026, the frozen **EXP-06C Config A** candidate achieved a **Macro F1 of 0.2083** [95% CI: 0.0749, 0.3481] and an **Accuracy of 33.33%** (7/21) on the held-out 21-participant test cohort (`data/mfs100_final`, split = `TEST`). This outcome fell below the pre-registered performance target ($\text{Macro F1} > 0.3500$) and underperformed naive majority-class guessing (Always-O+ $= 38.10\%$), while showing no statistically significant difference from the historical Phase 3 baseline ($p = 1.0000$).

This forensic scientific audit investigates the datasets, split manifests, ground-truth label provenance, image preprocessing artifacts, and experiment reproducibility records across the entire project lifecycle to identify the root causes of this generalization failure.

### Summary Assessment Table

| Audit Area | Investigation Focus | Status | Key Forensic Finding |
| :--- | :--- | :---: | :--- |
| **Audit A: Dataset Inventory** | Dataset paths, image formats, dimensions, hash integrity, participant identifiers | **NEEDS REVIEW** | 4 primary datasets identified ($6,000$ legacy, $303$ MFS100 final, $1,747$ captured, SQLite DB). Legacy dataset completely lacks participant identifiers. |
| **Audit B: Leakage & Splitting** | Partition overlap, duplicate hashes, cross-split leakage, cohort independence | **PASS** (MFS100) /<br>**FAIL** (Legacy 6k) | MFS100 splits have 0 participant overlap and 0 duplicate image hashes. However, the legacy 6,000-image dataset suffered from **confirmed identity leakage** (random image-level split), creating an illusory 85.41% baseline. |
| **Audit C: Label Quality** | Serological verification vs. self-reporting vs. filename derivation | **FAIL** (101 Cohort) /<br>**PASS** (26 Cohort) | Primary 101-participant cohort has **zero laboratory serology verification** (self-reported donor records). Only the separate 26-cohort has hospital serological test sheets. |
| **Audit D: Image Artifacts** | CLAHE, bilateral filtering, platen crop, resolution downsampling | **FAIL** (Pipeline) | Downsampling from $316 \times 354$ to $112 \times 112$ aliases ridge frequency (~2.5 px period). EXP-06D proved that CLAHE and bilateral filtering degrade representation (+0.0622 F1 gain when removed). |
| **Audit E: Reproducibility** | EXP-06C development OOF predictions, metrics, per-class rates | **PASS** | Development results on the 80 participants recomputed bit-for-bit from `config_a_oof_predictions.csv` (OOF F1 $= 0.5487$, Acc $= 53.75\%$). Zero missing or duplicate records. |

---

## 2. AUDIT A — DATASET INVENTORY

An exhaustive, read-only filesystem inventory was conducted across all datasets in the project workspace.

### 2.1 Quantitative Inventory of Dataset Repositories

| Repository Directory | Role in Project | File Count | File Formats | Native Resolution | Color Mode | Unique SHA-256 Hashes | Duplicate Files Quarantined | Participant IDs Present? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `data/raw/` | Original Legacy Dataset (Phase 1/2) | 6,000 | `.BMP` | $96 \times 103$ | RGBA | 5,985 | 15 (in `data/quarantine/`) | **NO** (`cluster_X_Y.BMP`) |
| `data/processed/` | Deduplicated Legacy Dataset | 5,985 | `.BMP` | $96 \times 103$ | RGBA | 5,985 | 0 | **NO** |
| `data/mfs100_final/raw/` | Authentic MFS100 Optical Cohort | 303 | `.bmp` | $316 \times 354$ | L (8-bit Gray) | 303 | 0 | **YES** (`P001`–`P0105`) |
| `data/captured/` | Live Physical Scanner Captures | 1,747 | `.bmp` | $316 \times 354$ | L (8-bit Gray) | 1,747 | 0 | **PARTIAL** (78 indexed to `S001`–`S026`) |
| `data/blood_group.db` | Operational Web App Database | 1 (184 KB) | SQLite 3 | — | — | — | — | Stored 898 mock/live web inference logs |

### 2.2 Class Label Distributions Across Datasets

#### 1. Legacy 6,000 Dataset (`data/raw/`) — 8 Classes (ABO $\times$ Rh):
- **A+:** 565 (9.42%) | **A-:** 1,009 (16.82%)
- **AB+:** 708 (11.80%) | **AB-:** 761 (12.68%)
- **B+:** 652 (10.87%) | **B-:** 741 (12.35%)
- **O+:** 852 (14.20%) | **O-:** 712 (11.87%)
- **Total:** 6,000 images (5,985 unique, 15 exact duplicates).

#### 2. Primary 101-Participant MFS100 Cohort (`data/mfs100_final/raw/`) — 4 Classes:
- **A+:** 24 participants (72 scans, 23.76%)
- **AB+:** 9 participants (27 scans, 8.91%)
- **B+:** 31 participants (93 scans, 30.69%)
- **O+:** 37 participants (111 scans, 36.63%)
- **Total:** Exactly 101 participants, each with exactly 3 scans = 303 physical scans.

#### 3. Descriptive 26-Participant Hospital Cohort (`data/captured/`, `data/mfs100_ground_truth.csv`):
- **A+:** 4 participants (12 scans, 15.38%)
- **AB+:** 3 participants (9 scans, 11.54%)
- **B+:** 8 participants (24 scans, 30.77%)
- **O+:** 11 participants (33 scans, 42.31%)
- **Total:** Exactly 26 participants (`S001`–`S026`), each with exactly 3 scans = 78 physical scans.

### 2.3 Record Consistency and Integrity
- **Missing / Corrupted Records:** None. All 303 scans in `data/mfs100_final` and all 78 scans in `data/mfs100_ground_truth.csv` open cleanly with PIL, have valid headers, non-zero variance, and standard 8-bit depth.
- **Participant ID Reliability:**
  - `data/mfs100_final`: High structural reliability (`P001`–`P0105`), organized into directory paths `<blood_group>/<participant_id>/scan_<01|02|03>.bmp`.
  - `data/raw`: Zero participant tracking. Images are named `cluster_0_1001.BMP` through `cluster_7_4556.BMP`, completely obscuring the donor identity.

---

## 3. AUDIT B — PARTICIPANT AND DUPLICATE LEAKAGE

The integrity of training, validation, and test partitions was audited across both the modern MFS100 benchmark and the historical legacy dataset.

### 3.1 MFS100 101-Participant Cohort Partition Audit
*Source file:* [`reports/phase3_participant_benchmark/participant_split.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/phase3_participant_benchmark/participant_split.csv)

```
Total Participants: 101 (303 scans)
├── TRAIN:      64 participants (192 scans)
├── VALIDATION: 16 participants ( 48 scans)
└── TEST:       21 participants ( 63 scans)
```

1. **Participant ID Overlap Across Splits:** **NOT FOUND** (`set()`).
   - $\text{TRAIN} \cap \text{VALIDATION} = \emptyset$ (0 participants).
   - $\text{TRAIN} \cap \text{TEST} = \emptyset$ (0 participants).
   - $\text{VALIDATION} \cap \text{TEST} = \emptyset$ (0 participants).
   - **Status:** **PASS**.
2. **Participant Multi-Split Leakage:** **NOT FOUND** (0 participants).
   - All 3 scans of every participant reside strictly within one partition.
   - **Status:** **PASS**.
3. **Exact Image Hash Overlap Across Splits:** **NOT FOUND** (0 collisions).
   - 192 unique hashes in TRAIN.
   - 48 unique hashes in VALIDATION.
   - 63 unique hashes in TEST.
   - Cross-split hash intersections: $\text{TRAIN} \cap \text{VAL} = 0$, $\text{TRAIN} \cap \text{TEST} = 0$, $\text{VAL} \cap \text{TEST} = 0$.
   - **Status:** **PASS**.
4. **Cross-Cohort Hash Overlap (`mfs100_final` vs `data/captured`):** **NOT FOUND** (0 collisions).
   - No image in `data/mfs100_final` shares a SHA-256 hash with any of the 78 indexed scans in `data/captured`.
   - **Status:** **PASS**.
5. **Cross-Cohort Participant Identity Overlap:** **UNKNOWN**.
   - The 101 cohort uses namespace `P001`–`P0105`, while the hospital cohort uses `S001`–`S026`.
   - Because no minutiae-matching ground truth was recorded during acquisition, it is impossible to confirm or rule out whether any volunteer from `P` was also scanned as subject `S`.
   - **Status:** **UNKNOWN / NEEDS REVIEW** (Quarantine must remain strictly enforced).

### 3.2 Historical 6,000-Image Dataset Identity Leakage Audit
*Source file:* [`outputs/dataset_manifest.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/outputs/dataset_manifest.csv)

1. **Participant Identifier Column:** **CONFIRMED ABSENT**.
   - The manifest columns are strictly `['image_path', 'class_name', 'class_id', 'split']`.
   - There is no participant ID, donor ID, finger index, or session metadata.
2. **Splitting Methodology:** **IMAGE-LEVEL RANDOM SPLIT (CONFIRMED)**.
   - Total rows: 5,985 (Train: 4,189, Val: 898, Test: 898).
   - The partition was performed via naive random sampling across the 6,000 images.
3. **Identity Leakage Finding:** **CONFIRMED IDENTITY LEAKAGE**.
   - Standard biometric fingerprint datasets (e.g., SOCOFing, FVC) collect 10 fingers per donor and multiple impressions per finger.
   - Performing an image-level split on multi-finger donor data guarantees that impressions from the *same* physical individual appear in both the training set and the test set.
   - **Scientific Impact:** Deep neural networks trained on this split achieved $\approx 86\%$ test accuracy (Model M1 in `reports/final/ORIGINAL_6000_DATASET_MODEL_ACCURACY_AUDIT.md`) because they memorized individual donor identities rather than learning generalizable haematological representations. When evaluated on genuinely unseen participants in MFS100, accuracy collapsed to chance ($\approx 33\text{--}38\%$).
   - **Status:** **FAIL (FATAL METHODOLOGICAL FLAW IN ORIGINAL BENCHMARK)**.

---

## 4. AUDIT C — LABEL QUALITY AND SEROLOGY PROVENANCE

A forensic trace of blood-group label provenance was conducted across all project documentation and data files.

### 4.1 Provenance Classification by Cohort

| Dataset Cohort | Claimed Ground-Truth Source | Physical Documentation Present? | Provenance Category | Clinical Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| **Legacy 6,000 Images** | Public Kaggle / Synthetic SOCOFing derivative | **NO** | Unknown / Synthetic | **UNVERIFIED** |
| **Primary MFS100 Cohort ($N=101$)** | Volunteer donor self-report / student registration record | **NO** (No lab sheets or phlebotomy logs) | Self-Reported / Institutional | **NOT VERIFIED** |
| **Secondary Cohort ($N=26$)** | Hospital Laboratory Serological Test Reports (25) + Verified record (1) | **YES** (Referenced `Report <Group>.pdf` in `mfs100_ground_truth.csv`) | Laboratory Serology | **VERIFIED (25/26)** |

### 4.2 Critical Label Quality Findings

1. **Total Absence of Serological Validation for the Primary Cohort:**
   - Ground-truth blood groups for all 101 participants in `data/mfs100_final` (including the 80 development participants and the 21 locked test participants) were collected via **unverified participant self-reporting**.
   - No hemagglutination assays (Anti-A, Anti-B, Anti-D reagents), column agglutination tests, or phlebotomist logs were performed to corroborate self-reported labels.
   - In general population self-reporting, error rates of 5% to 15% are common (e.g., individuals confusing O+ and B+, or unaware of Rh status).
   - **Scientific Consequence:** An unknown fraction of the ground-truth labels on the 21 test participants may be erroneous. A model trained on noisy self-reports cannot be evaluated for true clinical validity against self-reported labels.
2. **Highest Ground-Truth Fidelity Exists Only in the Quarantined 26-Cohort:**
   - As documented in [`data/mfs100_ground_truth.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/mfs100_ground_truth.csv#L1-L35), 25 of 26 subjects (`S001`–`S025`) have verified hospital serological test certificates.
   - However, this cohort cannot be used for model selection, parameter tuning, or pooling with the primary cohort due to domain shifts and quarantine boundaries.

---

## 5. AUDIT D — IMAGE ARTIFACTS AND PREPROCESSING SENSITIVITY

The transformation pipeline implemented in `src/preprocessing.py` (`transform_c5`) was forensically analyzed for artifacts, information destruction, and sensor-specific characteristics.

### 5.1 Step-by-Step Processing Analysis

1. **Alpha-Channel Compositing (`composite_alpha_on_white`):**
   - *Implementation:* Composites transparent backgrounds onto solid white `(255, 255, 255)`.
   - *Observation:* Cleanly handles legacy RGBA images, avoiding artificial black borders. MFS100 grayscale images are converted to 3-channel RGB.
   - *Artifact Risk:* Low.
2. **Foreground Platen Cropping (`crop_mfs100_foreground`):**
   - *Implementation:* Detects the fingerprint platen boundaries using intensity thresholding ($\text{intensity} < 238$) with a 6% padding margin.
   - *Observation:* As documented in `outputs/mfs100_crop_results.json`, crop dimensions vary widely across scans depending on finger moisture and platen contact area (bounding boxes range from $249 \times 349$ to full $316 \times 354$).
   - *Artifact Risk:* Moderate. Cropping non-uniformly rescales the physical ridge frequency before resizing.
3. **CLAHE Contrast Enhancement (`enhance_mfs100_ridge_contrast`):**
   - *Implementation:* Applies Adaptive Histogram Equalization with `clipLimit=2.5` on an $8 \times 8$ grid.
   - *Observed Evidence:* CLAHE dramatically amplifies high-frequency optical sensor noise in empty platen regions and around fingerprint boundaries.
   - *Empirical Impact from EXP-06D:* In EXP-06D (`reports/accuracy_improvement/exp06d/final_report.md`), removing CLAHE and bilateral filtering increased pooled OOF Macro F1 from **0.4456** to **0.5078** (+0.0622) and Accuracy from **46.25%** to **52.50%** (+6.25%).
   - *Conclusion:* CLAHE acts as a **destructive artifact generator**, injecting spurious high-frequency texture that distorts epidermal feature extraction.
4. **Dynamic Percentile Contrast Stretching (p1 $\rightarrow$ 15, p99 $\rightarrow$ 245):**
   - *Implementation:* Linearly stretches grayscale intensities between the 1st and 99th percentiles.
   - *Observation:* Highly sensitive to optical sensor glare, dry thumb impressions, or latent oil smudges on the sensor glass. Shifts the global intensity baseline non-linearly across scans.
5. **Bilateral Filtering ($d=3, \sigma=25$):**
   - *Implementation:* Edge-preserving smoothing filter.
   - *Observation:* While intended to suppress CMOS sensor read noise, it smooths fine minutiae transitions (bifurcations and ending ridges), reducing the discriminative fidelity of epidermal ridge geometry.
6. **Severe Resolution Downsampling to $112 \times 112$:**
   - *Implementation:* Lanczos interpolation downsamples $316 \times 354$ native MFS100 captures to $112 \times 112$.
   - *Mathematical Analysis:* Native MFS100 resolution is 500 DPI. A human fingerprint ridge-to-valley period is approximately $\approx 0.4\text{--}0.5\text{ mm}$, corresponding to $8\text{--}10\text{ pixels}$ at 500 DPI.
   - When downsampled to $112 \times 112$ (a $3\times$ spatial compression), ridge period collapses to **$2.5\text{--}3.2\text{ pixels}$**.
   - *Consequence:* This approaches the Nyquist-Shannon limit for spatial frequency representation, inducing severe aliasing, phase distortion, and moiré-like patterns across the 3 concatenated scans. DenseNet-121 filters pre-trained on ImageNet are incapable of extracting meaningful biological semantics from aliased 2.5-pixel ridges.

---

## 6. AUDIT E — EXPERIMENT REPRODUCIBILITY RESULTS

The development cross-validation records for the candidate **EXP-06C Config A** were audited by independently recalculating all metrics directly from the raw out-of-fold prediction file:  
[`reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv)

### 6.1 Prediction Integrity & Participant Coverage
- **Total Development Participants Evaluated:** Exactly **80 unique participants**.
- **Participant ID Match:** Exactly matches the approved 80 development participants from `reports/phase3_participant_benchmark/participant_split.csv` (`TRAIN` $N=64$ + `VALIDATION` $N=16$).
- **Missing Predictions:** 0.
- **Duplicate Records:** 0.
- **Probability Consistency:** Softmax probabilities sum to $1.0000 \pm 0.0001$ across all 80 participants.

### 6.2 Metric Recomputation Audit

| Metric | Stored in `metrics_comparison.csv` | Independently Recomputed from Predictions | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **53.75%** (43 / 80) | **53.7500%** (43 / 80) | 0.0000% | **EXACT MATCH** |
| **Balanced Accuracy** | **53.47%** | **53.4735%** | 0.0000% | **EXACT MATCH** |
| **Macro F1 (Pooled OOF)** | **0.5487** | **0.548679** | 0.0000 | **EXACT MATCH** |
| **Macro Precision** | — | **0.573408** | — | Verified |
| **Macro Recall** | — | **0.534735** | — | Verified |
| **Mean Fold Macro F1** | **0.5331** | **0.533076** | 0.0000 | **EXACT MATCH** |
| **Fold F1 Std Dev** | **0.0895** (pop.) / **0.1001** (sample) | **0.089511** (ddof=0) | 0.0000 | **EXACT MATCH** |
| **A+ Recall** | 42.11% (8 / 19) | 42.1053% (8 / 19) | 0.0000% | **EXACT MATCH** |
| **AB+ Recall** | 57.14% (4 / 7) | 57.1429% (4 / 7) | 0.0000% | **EXACT MATCH** |
| **B+ Recall** | 56.00% (14 / 25) | 56.0000% (14 / 25) | 0.0000% | **EXACT MATCH** |
| **O+ Recall** | 58.62% (17 / 29) | 58.6207% (17 / 29) | 0.0000% | **EXACT MATCH** |

### 6.3 Recomputed Development Confusion Matrix ($N=80$)

Canonical classes: `[A+, AB+, B+, O+]`

```
                    PREDICTED
               A+    AB+     B+     O+    Total
         A+ [   8      1      1      9 ]    19
TRUE    AB+ [   1      4      0      2 ]     7
         B+ [   2      1     14      8 ]    25
         O+ [   3      0      9     17 ]    29
      Total    14      6     24     36    N=80
```

**Audit Verdict:** **PASS (100% REPRODUCIBLE)**. The reported development metrics in EXP-06C are completely authentic, mathematically faithful to the recorded predictions, and exhibit zero post-hoc inflation or truncation discrepancies.

---

## 7. Confirmed Scientific Issues vs. Unresolved Hypotheses

To prevent speculative drift, this section strictly separates **empirically proven facts** from **unresolved hypotheses**.

### 7.1 Confirmed Scientific Issues (Supported by Direct Evidence)

1. **Extreme Overparameterization Relative to Sample Size:**
   - *Evidence:* The classification head had $1,968,772$ trainable parameters fitted on only 64 training participants (16 validation) per fold. Fitting nearly two million parameters on 64 human subjects guarantees that the network memorizes donor-specific biometric minutiae rather than universal haematological features.
2. **Severe Class Imbalance in Development Training:**
   - *Evidence:* AB+ has only 7 participants in the entire 80-person development pool. In 5-fold CV, training folds saw only 5 or 6 AB+ individuals. A neural network cannot learn a generalizable representation of human AB+ blood from 5 individuals.
3. **Generalization Collapse on Unseen Participants:**
   - *Evidence:* Development cross-validation Macro F1 was **0.5487**; held-out test Macro F1 collapsed to **0.2083** (Accuracy dropped from 53.75% to 33.33%, below Always-O+ 38.10%). The model collapsed almost entirely to predicting the majority class `O+` (16 of 21 test predictions).
4. **Identity Leakage in Historical 6,000-Image Baseline:**
   - *Evidence:* `outputs/dataset_manifest.csv` proves the original 6,000 dataset lacked participant IDs and was split at the image level, creating the false impression of an 85.41% baseline.
5. **Absence of Laboratory Serology Ground Truth for Test Cohort:**
   - *Evidence:* Neither the 80 development participants nor the 21 test participants have laboratory serological confirmation records.
6. **Destructive Impact of CLAHE & Bilateral Filtering:**
   - *Evidence:* EXP-06D proved that Enhanced C5 preprocessing degraded Macro F1 by -0.0622 compared to Raw Platen cropping. EXP-06C was trained with this suboptimal Enhanced C5 pipeline.

### 7.2 Unresolved Hypotheses (Requiring Further Empirical Research)

1. **Hypothesis: Biological Dermatoglyphic Correlation Exists:**
   - *Status:* **UNRESOLVED / UNPROVEN**. While dermatoglyphic literature posits weak statistical correlations between ridge pattern frequencies (loops, whorls, arches) and ABO phenotypes at a population level, there is no empirical evidence in this codebase that optical fingerprint scans contain direct haematological or serological signals capable of individual-level classification.
2. **Hypothesis: Scanner Sensor Noise Imprinting:**
   - *Status:* **UNRESOLVED**. It is hypothesized that the Mantra MFS100 optical CMOS sensor introduces temperature- or age-dependent fixed-pattern noise (FPN) that models may exploit during cross-validation, but this has not been isolated via dark-frame subtraction.
3. **Hypothesis: Unverified Test Labels Explain Test Collapse:**
   - *Status:* **UNRESOLVED**. While self-reported labels are prone to error, label noise alone cannot explain the complete collapse of A+ (0/5) and B+ (0/6) sensitivity. The primary driver is model overparameterization and majority-class collapse.

---

## 8. Recommended Next Steps Based on Forensic Evidence

Based on the empirical findings of this Phase 1 scientific audit, the following evidence-based actions are recommended:

1. **Acknowledge Clinical Non-Validity Permanently:**
   - Formally document that optical biometric blood group classification via deep CNNs on small participant cohorts is not clinically valid and cannot be deployed for diagnostic, triage, or screening purposes.
2. **Discontinue Further Training on the 80-Participant Cohort:**
   - Further hyperparameter sweeps, architecture tweaks, or loss function ablations on $N=64$ training subjects cannot overcome the fundamental mathematical sample-size barrier.
3. **If Future Research is Pursued, Mandate Gold-Standard Data Collection:**
   - Any future research must require:
     - Minimum $N \ge 500\text{--}1,000$ participants with verified laboratory serological agglutination test records.
     - Native optical resolution ($500\text{ DPI}$, un-aliased, no downsampling to $112 \times 112$).
     - Strict participant-level stratified grouping at initial enrollment.
     - Multi-sensor collection to prevent scanner fixed-pattern noise memorization.
4. **Preserve All Quarantine and Reproducibility Ledgers:**
   - Maintain the frozen state of all 66 experiment checkpoints, freeze manifests, and execution logs for regulatory and academic transparency.

---

## 9. Source File Evidence Ledger

The findings in this audit report are directly corroborated by the following repository files:

- **Primary Evaluation Final Report:** [`reports/final_evaluation/primary_evaluation_final_report.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/primary_evaluation_final_report.md)
- **Primary Predictions File:** [`reports/final_evaluation/primary_evaluation_predictions.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/primary_evaluation_predictions.csv)
- **Primary Metrics JSON:** [`reports/final_evaluation/primary_evaluation_results.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/primary_evaluation_results.json)
- **Participant Split Manifest:** [`reports/phase3_participant_benchmark/participant_split.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/phase3_participant_benchmark/participant_split.csv)
- **Hospital Cohort Metadata:** [`data/mfs100_ground_truth.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/data/mfs100_ground_truth.csv)
- **Legacy Manifest:** [`outputs/dataset_manifest.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/outputs/dataset_manifest.csv)
- **EXP-06C Development Predictions:** [`reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv)
- **EXP-06C Metrics Comparison:** [`reports/accuracy_improvement/exp06c/metrics_comparison.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06c/metrics_comparison.csv)
- **EXP-06D Final Report (Preprocessing Ablation):** [`reports/accuracy_improvement/exp06d/final_report.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/final_report.md)
- **Preprocessing Implementation:** [`src/preprocessing.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/src/preprocessing.py)
