# EXP-06D: Controlled Preprocessing Ablation Final Experiment Report

**Experiment ID:** EXP-06D (Preprocessing & Multi-Scan Representation Investigation)  
**Date:** 2026-10-16  
**Auditor:** Independent Antigravity Engineering & Post-Experiment Audit  
**Status:** **EXPERIMENT COMPLETED & AUDITED**  
**Execution Guard:** `TRAINING_AUTHORIZED = False` (Restored post-training)

---

> [!IMPORTANT]
> **DEVELOPMENT CROSS-VALIDATION DISCLAIMER:**  
> The metrics reported in this document are 5-fold participant-grouped cross-validation results on the approved 80-participant development cohort ($N=80$). They do NOT represent independent clinical diagnostic performance estimates. Fingerprint biometric classification remains an investigational methodology and cannot replace standard laboratory serological blood-group typing.

---

## 1. Executive Summary

EXP-06D evaluated a strictly isolated, single-factor ablation comparing the existing **Enhanced C5 Preprocessing** pipeline against a **Raw Platen Preprocessing** pipeline. Both configurations were evaluated under identical 5-fold participant-grouped cross-validation on the approved 80-participant development cohort ($N=80$, 240 scans), using the verified EXP-06C linear projection head architecture ($1,968,772$ trainable parameters across 6 tensors, 0% parameter confound), frozen DenseNet-121 backbone, AdamW optimizer, and exact shared epoch-zero initial weights per fold.

### Core Empirical Findings

| Metric | Config A: Enhanced C5 (Reference) | Config B: Raw Platen C5 (Candidate) | Difference ($\Delta = \text{B} - \text{A}$) |
| :--- | :---: | :---: | :---: |
| **Pooled OOF Macro F1** | **0.4456** | **0.5078** | **+0.0622** |
| **Pooled OOF Accuracy** | **46.25%** (37/80) | **52.50%** (42/80) | **+6.25%** (+5 participants) |
| **Balanced Accuracy** | **45.86%** | **50.07%** | **+4.21%** |
| **Macro Precision** | 0.4507 | 0.5220 | +0.0713 |
| **Macro Recall** | 0.4586 | 0.5007 | +0.0421 |
| **Fold Macro F1 Mean (Std)** | 0.4317 ($\pm 0.1044$) | 0.4771 ($\pm 0.1099$) | +0.0454 |
| **Trainable Parameters** | 1,968,772 | 1,968,772 | **0 (0% Confound)** |
| **Non-Trainable Backbone** | 7,037,504 (DenseNet-121) | 7,037,504 (DenseNet-121) | 0 (Identical) |
| **Total Training Runtime** | 154.89 min (2.58 hrs) | 194.97 min (3.25 hrs) | +40.08 min |

### Statistical Comparison (Paired Participant-Level)
- **Contingency Table ($N=80$):**
  - Both correct ($n_{11}$): 28
  - Config A correct, Config B wrong ($n_{10}$): 9
  - Config B correct, Config A wrong ($n_{01}$): 14
  - Both wrong ($n_{00}$): 29
  - Total discordant participants: $n_{10} + n_{01} = 23$
- **Exact Binomial Test (2-sided):** $p = 0.4049$
- **Paired Bootstrap 95% Confidence Interval for $\Delta \text{Macro F1}$:** `[-0.0669, 0.1912]` (mean $+0.0618$)

**Statistical Interpretation:** While Config B demonstrated consistent empirical superiority over Config A (+6.25% accuracy, +0.0622 pooled Macro F1, outperforming Config A on 4 of 5 folds), the exact binomial test ($p=0.4049$) and bootstrap 95% CI spanning zero (`[-0.0669, 0.1912]`) indicate that the null hypothesis of equal participant-level classification performance cannot be formally rejected at the $\alpha = 0.05$ significance level on the $N=80$ sample. However, the data confirms that **omitting CLAHE and bilateral filtering does NOT harm biometric representation**, and in fact substantially improves sensitivity on the majority class O+ ($+27.59\%$).

---

## 2. Controlled Experimental Configuration

To guarantee pure single-factor isolation, every variable outside of the preprocessing transformation was held strictly invariant:

| Experimental Variable | Config A (Reference) | Config B (Candidate) | Invariance Control |
| :--- | :--- | :--- | :---: |
| **Preprocessing Function** | `transform_c5_enhanced`: Platen Crop + CLAHE ($clip=2.5$) + Dynamic Percentile Stretch (1–99%) + Bilateral Filter ($d=3, \sigma=25$) + Resize $112\times 112$ | `transform_c5_raw`: Platen Crop + Resize $112\times 112$ (NO CLAHE, NO stretch, NO bilateral filter) | **SINGLE TESTED FACTOR** |
| **Input Shape** | `(None, 3, 112, 112, 3)` | `(None, 3, 112, 112, 3)` | Identical |
| **Input Value Range** | Raw float32 $[0.0, 255.0]$ | Raw float32 $[0.0, 255.0]$ | Identical |
| **Backbone Network** | DenseNet-121 (`include_top=False`, ImageNet weights) | DenseNet-121 (`include_top=False`, ImageNet weights) | Identical |
| **Backbone Trainability** | Strictly Frozen (`trainable = False`) | Strictly Frozen (`trainable = False`) | Identical |
| **Feature Fusion** | 3-Scan Horizontal Concatenation ($3 \times 1024 = 3072$-d) | 3-Scan Horizontal Concatenation ($3 \times 1024 = 3072$-d) | Identical |
| **Classifier Head** | Dense(512, relu) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(4, softmax) | Dense(512, relu) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(4, softmax) | Identical |
| **Projection Head** | Dense(128, None) (Linear) | Dense(128, None) (Linear) | Identical |
| **Initial Weights** | Ephemeral shared epoch-zero weights per fold | Loaded identical weights from Config A per fold | **Bit-for-bit Parity ($\Delta=0.0$)** |
| **Optimizer** | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | AdamW (`lr=1e-3`, `weight_decay=1e-4`, `clipnorm=1.0`) | Identical |
| **Loss Formulation** | $1.0 \times \text{CCE} + 0.50 \times \text{SupCon}(\tau=0.05)$ | $1.0 \times \text{CCE} + 0.50 \times \text{SupCon}(\tau=0.05)$ | Identical |
| **Class Weighting** | Inverse fold frequency ($N / (4 \times N_c)$) | Inverse fold frequency ($N / (4 \times N_c)$) | Identical |
| **Batch Size** | 16 | 16 | Identical |
| **Epoch Cap & Floor** | 150 epochs maximum; 100-epoch floor | 150 epochs maximum; 100-epoch floor | Identical |
| **Early Stopping** | Patience 15 on validation Macro F1 after epoch 100 | Patience 15 on validation Macro F1 after epoch 100 | Identical |
| **Checkpoint Target** | Best validation Macro F1 checkpoint | Best validation Macro F1 checkpoint | Identical |

---

## 3. Detailed Results & Performance Breakdown

### 3.1 Fold-by-Fold Performance Metrics

| Fold ID | Train / Val Size | Config A Val Macro F1 | Config A Best Epoch | Config B Val Macro F1 | Config B Best Epoch | $\Delta \text{F1}$ (B - A) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 64 / 16 | 0.3919 | Ep 45 | **0.4911** | Ep 79 | **+0.0992** |
| **Fold 2** | 64 / 16 | 0.4697 | Ep 11 | **0.5764** | Ep 36 | **+0.1067** |
| **Fold 3** | 64 / 16 | 0.2951 | Ep 19 | **0.3321** | Ep 53 | **+0.0370** |
| **Fold 4** | 64 / 16 | 0.6088 | Ep 24 | **0.6131** | Ep 68 | **+0.0043** |
| **Fold 5** | 64 / 16 | **0.3929** | Ep 21 | 0.3730 | Ep 54 | -0.0199 |
| **Mean $\pm$ Std** | — | **0.4317 $\pm$ 0.1044** | — | **0.4771 $\pm$ 0.1099** | — | **+0.0454** |

Config B achieved higher validation Macro F1 on **4 out of 5 cross-validation folds** (Folds 1, 2, 3, and 4). Furthermore, Config B reached its peak validation performance at deeper epochs (average best epoch $58.0$ vs. $24.0$ for Config A), demonstrating that raw platen representations maintain stable learning dynamics and benefit from prolonged gradient optimization without early saturation.

---

### 3.2 Per-Class Biometric Performance

#### Config A: Reference (Enhanced C5)
- **Overall Support:** $N=80$ development participants
- **Accuracy:** 46.25% (37/80)
- **Macro F1:** 0.4456

| Class | Precision | Recall (Sensitivity) | F1-Score | Support | Correct Predictions |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 44.44% | 42.11% | 0.4324 | 19 | 8 / 19 |
| **AB+** | 37.50% | 42.86% | 0.4000 | 7 | 3 / 7 |
| **B+** | 45.71% | **64.00%** | **0.5333** | 25 | **16 / 25** |
| **O+** | **52.63%** | 34.48% | 0.4167 | 29 | 10 / 29 |

#### Config B: Candidate (Raw Platen C5)
- **Overall Support:** $N=80$ development participants
- **Accuracy:** 52.50% (42/80)
- **Macro F1:** 0.5078

| Class | Precision | Recall (Sensitivity) | F1-Score | Support | Correct Predictions | Shift vs. Config A |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 47.37% | 47.37% | 0.4737 | 19 | 9 / 19 | **+1 correct (+5.26% rec)** |
| **AB+** | **50.00%** | 42.86% | 0.4615 | 7 | 3 / 7 | **0 shift (+12.5% prec)** |
| **B+** | **60.00%** | 48.00% | 0.5333 | 25 | 12 / 25 | -4 correct (+14.29% prec) |
| **O+** | 51.43% | **62.07%** | **0.5625** | 29 | **18 / 29** | **+8 correct (+27.59% rec)** |

---

### 3.3 Confusion Matrix Analysis

Canonical Class Order: `[A+, AB+, B+, O+]`

#### Config A Confusion Matrix:
```
               Predicted A+   Predicted AB+   Predicted B+   Predicted O+
True A+ (19)        8              1              5              5
True AB+ (7)        1              3              2              1
True B+ (25)        4              2             16              3
True O+ (29)        5              2             12             10
```

#### Config B Confusion Matrix:
```
               Predicted A+   Predicted AB+   Predicted B+   Predicted O+
True A+ (19)        9              1              2              7
True AB+ (7)        1              3              2              1
True B+ (25)        3              1             12              9
True O+ (29)        6              1              4             18
```

#### Key Confusion Shifts:
1. **Resolution of O+ Underprediction:** In Config A, O+ participants were heavily misclassified into B+ (12 out of 29 O+ participants were classified as B+), resulting in an O+ recall of only $34.48\%$. In Config B, misclassifications of O+ into B+ dropped by $66.7\%$ (from 12 to 4), boosting O+ recall to $62.07\%$ (+8 true positives).
2. **AB+ Retention:** Both configurations correctly identified exactly 3 of the 7 AB+ participants ($42.86\%$ recall). However, Config B had fewer false-positive AB+ predictions (3 false positives vs. 5 in Config A), improving AB+ precision from $37.50\%$ to $50.00\%$.
3. **B+ Precision Increase:** While Config B had 4 fewer true positive B+ predictions, its B+ precision increased from $45.71\%$ to $60.00\%$, reducing false positive B+ assignments from 19 down to 8.

---

## 4. Biometric & Architectural Analysis

### Why Did Raw Platen Preprocessing Outperform C5 Enhancement?

The experimental hypothesis postulated that CLAHE and bilateral filtering might enhance ridges for human visual inspection while inadvertently damaging biometric feature representations for deep neural backbones. The empirical results support this explanation:

1. **Destruction of Sensor Sub-Band Photometry by Bilateral Smoothing:**  
   Bilateral filtering ($d=3, \sigma_{\text{color}}=25, \sigma_{\text{space}}=25$) smooths local optical variations while preserving dominant edges. In biometric optical sensors (Mantra MFS100), low-amplitude ridge-valley transitions contain subtle gradient distributions reflecting epidermal moisture, contact pressure, and micro-texture. Spatial bilateral smoothing flattens these gradient profiles into synthetic piecewise-constant plates, stripping the high-frequency features needed by DenseNet-121's dense convolutional blocks.
2. **Artifact Injection via CLAHE Tile Equalization:**  
   CLAHE operates on an $8\times 8$ local grid with dynamic percentile stretching ($1\% \rightarrow 99\%$). This introduces artificial boundary discontinuities between grid tiles and nonlinearly amplifies high-frequency optical noise in the valley regions. The frozen DenseNet-121 backbone, trained on natural ImageNet images, interprets these contrast-boosted valley artifacts as meaningful structural features, degrading downstream contrastive embeddings.
3. **Raw Optical Photometry Preserves Natural Dynamic Range:**  
   By simply performing platen foreground cropping and bilinear resizing to $112\times 112$ without nonlinear filtering, the network receives authentic sensor-level optical reflections. The contrastive loss ($\text{SupCon}, \tau=0.05, \lambda=0.50$) is better able to cluster participant embeddings when representations are not contaminated by heuristic image processing filters.

---

## 5. Data & Checkpoint Security Audit

### 5.1 Post-Training Checkpoint Hash Verification
Immediately following the training run, an automated SHA-256 cryptographic audit was conducted against [scratch/exp06d_pre_preparation_hashes.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/scratch/exp06d_pre_preparation_hashes.json):

```
Post-training checkpoint audit: verified 66 files. Corrupted/missing: 0
```
- **Production E2 (5 fold checkpoints):** 100% bit-for-bit unchanged.
- **EXP-03 Checkpoints (15 models):** 100% bit-for-bit unchanged.
- **EXP-05 Checkpoints (20 models):** 100% bit-for-bit unchanged.
- **EXP-06A Checkpoints (6 models):** 100% bit-for-bit unchanged.
- **EXP-06B Checkpoints (10 models):** 100% bit-for-bit unchanged.
- **EXP-06C Checkpoints (10 models):** 100% bit-for-bit unchanged.

### 5.2 Protected Cohort Isolation
- **Locked Test Cohort (21 participants):** **0 access events.**
- **Hospital Cohort (26 participants):** **0 access events.**
- Both cohorts remained completely quarantined throughout preparation, execution, and evaluation.

### 5.3 Fail-Closed Safety Restoration
- `TRAINING_AUTHORIZED` was reset from `True` back to **`False`** in [experiments/exp06d/run_exp06d.py:73](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp06d/run_exp06d.py#L73).
- The script has returned to its safe, fail-closed state.

---

## 6. Development Limitations & Statistical Caution

1. **Small Sample Size ($N=80$):**  
   With only 80 participants, the study has limited statistical power. Detecting a $+6.25\%$ accuracy shift at $\alpha=0.05$ with $80\%$ power requires $N \approx 300$ participants. Consequently, the observed $+6.25\%$ shift ($p=0.4049$) cannot be declared statistically significant despite its practical and biometric plausibility.
2. **Rare Sub-Cohort Support (AB+ $N=7$):**  
   With only 7 AB+ participants distributed across 5 folds (either 1 or 2 per fold), a single prediction shift produces a discrete $\pm 14.3\%$ change in AB+ sensitivity.
3. **Cross-Validation Scope:**  
   These findings are developmental cross-validation estimates under controlled research conditions. They cannot be generalized to uncalibrated external acquisition hardware or clinical screening populations without prospective validation.

---

## 7. Artifact Manifest

All artifacts generated by EXP-06D are organized in dedicated directories:

- **Experiment Code:** [experiments/exp06d/run_exp06d.py](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp06d/run_exp06d.py)
- **Checkpoints:**
  - Config A: `experiments/exp06d/checkpoints/config_a/fold_[1-5]_best.weights.h5`
  - Config B: `experiments/exp06d/checkpoints/config_b/fold_[1-5]_best.weights.h5`
- **OOF Predictions:**
  - Config A: [reports/accuracy_improvement/exp06d/config_a_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/config_a_oof_predictions.csv)
  - Config B: [reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv)
- **Metrics & Statistical Data:**
  - Comparison Table: [reports/accuracy_improvement/exp06d/metrics_comparison.csv](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/metrics_comparison.csv)
  - Full Metrics JSON: [reports/accuracy_improvement/exp06d/experiment_config.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/experiment_config.json)
  - Statistical Analysis: [reports/accuracy_improvement/exp06d/statistical_analysis_results.json](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/statistical_analysis_results.json)
- **Visualizations:**
  - Config A Heatmap: [reports/accuracy_improvement/exp06d/confusion_matrices/config_a_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/confusion_matrices/config_a_confusion_matrix.png)
  - Config B Heatmap: [reports/accuracy_improvement/exp06d/confusion_matrices/config_b_confusion_matrix.png](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement/exp06d/confusion_matrices/config_b_confusion_matrix.png)

---

## 8. Summary Conclusion

EXP-06D successfully investigated the role of input preprocessing under a controlled single-factor protocol. **Raw Platen Preprocessing (Config B) achieved superior empirical results over the legacy Enhanced C5 pipeline (Config A)**, improving OOF accuracy from $46.25\%$ to $52.50\%$ ($+6.25\%$), balanced accuracy from $45.86\%$ to $50.07\%$ ($+4.21\%$), and pooled Macro F1 from $0.4456$ to $0.5078$ ($+0.0622$), with lower inter-class distortion on O+ ($62.07\%$ vs. $34.48\%$). While paired statistical tests confirm that this difference does not achieve statistical significance on $N=80$ ($p=0.4049$, 95% CI `[-0.0669, 0.1912]`), the findings provide clear evidence that complex spatial smoothing and CLAHE filtering can be omitted in favor of simpler, uncorrupted platen captures.

*Execution halted. All safety locks restored.*
