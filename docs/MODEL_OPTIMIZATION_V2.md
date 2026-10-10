# Model Optimization Strategy V2

**Generated**: 2026-09-04  
**Scope**: Advanced Optimization on Train (4189) & Validation (898) splits ONLY.  
**Test Set Status**: Completely LOCKED and untouched (Preserved Baseline: 85.41% test acc, 0.8591 macro F1).

---

## 1. Executive Summary & Optimization Trajectory

Through systematic diagnostic analysis and controlled validation-only experimentation, validation accuracy has progressed through four distinct tiers:

1. **Initial ResNet50 Baseline**: **60.02%** Val Accuracy (0.5691 Macro F1)
2. **Phase 3 ResNet50 Improved (Single Model)**: **79.73%** Val Accuracy (0.7993 Macro F1)
3. **Phase F Initial Ensemble (3-Way Soft Voting)**: **85.19%** Val Accuracy (0.8547 Macro F1)
4. **Phase F/G Champion Ensemble V2**: **86.41%** Val Accuracy (**0.8663** Macro F1)

---

## 2. Controlled Experimental Phases & Findings

### Phase A: Ensemble Diagnostic Insights
- Detailed findings recorded in `docs/ENSEMBLE_DIAGNOSTIC.md`.
- **Error Correlation**: Extremely low ($r = 0.210$ between ResNet50 and VGG16), confirming strong architectural complementarity.
- **Class-Specific Dominance**: VGG16 dominates on `A-` (88.1% recall) and `B+` (84.5%), while ResNet50 dominates on `A+` (89.4%), `B-` (96.4%), and `O-` (92.5%).

---

### Phase C: Higher Input Resolution Benchmark

| Model | 224Ã—224 Val Accuracy | 256Ã—256 Val Accuracy | Net Impact | Decision |
|---|:---:|:---:|:---:|:---:|
| **ResNet50** | **79.73%** | 76.39% | -3.34% | **Retain 224Ã—224** |
| **VGG16** | **78.62%** | 76.73% | -1.89% | **Retain 224Ã—224** |

**Rationale**: Raw fingerprint bitmaps in this dataset are natively $\sim 96 \times 103$ pixels. Upscaling beyond $224 \times 224$ (e.g. to $256 \times 256$) produces noticeable interpolation blur and artificial gradients that degrade ridge minutiae clarity. Standard $224 \times 224$ matched to the backbone receptive field is optimal.

---

### Phase D: Fingerprint-Specific Preprocessing Experiments

Evaluated on the exact same 898 validation images:

| Preprocessing Pipeline | ResNet50 Val Acc | ResNet50 Macro F1 | VGG16 Val Acc | Finding |
|---|:---:|:---:|:---:|---|
| **Exp 1: Baseline** (Alpha on white + Resize) | 79.73% | 0.7993 | **78.62%** | Reference standard |
| **Exp 2: Mild Contrast (1.2Ã—)** | **80.96%** | **0.8115** | 77.95% | **+1.23% gain on ResNet50** |
| **Exp 3: Tuned CLAHE** (clip=2.0, grid=8Ã—8) | 80.40% | 0.8060 | 78.40% | +0.67% gain on ResNet50 |
| **Exp 4: Mild Sharpening** (UnsharpMask) | 78.73% | 0.7908 | 77.06% | Artificial edge noise hurts both |

**Decision**: Integrate Mild Contrast Enhancement ($1.2\times$) specifically into the ResNet50 inference branch, lifting single-model performance above $80.9\%$.

---

### Phase E: Class Imbalance Optimization
- Mild class balancing ($w_{smooth} = 0.6 \cdot (w_{balanced} - 1) + 1.0$) struck the optimal tradeoff between protecting minority classes (`A-`, `O-`) without causing precision collapse on common classes.
- Zero class weighting causes severe majority-class bias.
- Extreme inverse-frequency weighting creates false positives on rare classes.

---

### Phase G: 4-Way Ensemble Expansion Test (DenseNet121 Inclusion)

| Configuration | Val Accuracy | Val Macro F1 | Status |
|---|:---:|:---:|:---:|
| **3-Way Uniform** (ResNet + VGG + EffNet) | **84.52%** | **0.8474** | Robust |
| **4-Way Uniform** (Adding DenseNet121) | 83.96% | 0.8415 | **-0.56% dilution** |

**Decision**: Exclude DenseNet121. DenseNet121 has a $42.1\%$ single-model error rate; injecting its probability distributions degrades the high-precision consensus of the top 3 models.

---

### Phase F: Champion Ensemble V2 Configuration

Using a fine-grained simplex search (step = $0.02$) with contrast-enhanced ResNet50 feature extraction:

- **Weights**:
  - **ResNet50 (1.2Ã— Contrast)**: **0.58**
  - **VGG16**: **0.38**
  - **EfficientNetB0**: **0.04**
- **Optimal Temperature**: $T = 1.00$
- **Validation Accuracy**: **86.41%** (gain of **+26.39%** over baseline 60.02%, **+1.22%** over Ensemble V1)
- **Validation Macro F1**: **0.8663** (gain of **+29.72%** over baseline 0.5691)

---

## 3. Per-Class Validation Breakdown (Champion Ensemble V2)

| Blood Group | Validation Support | Precision | Recall | F1-Score |
|:---:|:---:|:---:|:---:|:---:|
| **A+** | 85 | 0.8750 | 0.8235 | **0.8485** |
| **A-** | 151 | 0.9421 | 0.7550 | **0.8382** |
| **AB+** | 106 | 0.8762 | 0.8679 | **0.8720** |
| **AB-** | 114 | 0.8125 | 0.9123 | **0.8595** |
| **B+** | 97 | 0.9451 | 0.8866 | **0.9149** |
| **B-** | 111 | 0.9153 | 0.9730 | **0.9432** |
| **O+** | 128 | 0.7899 | 0.8516 | **0.8195** |
| **O-** | 106 | 0.7949 | 0.8774 | **0.8341** |
| **Macro Average** | **898** | **0.8689** | **0.8684** | **0.8663** |

---

## 4. Key Artifacts
- Optimization Summary JSON: `outputs/ensemble_optimization_v2_results.json`
- Diagnostic Report: `docs/ENSEMBLE_DIAGNOSTIC.md`
- Preprocessing Benchmarks: `outputs/preprocessing_experiments_results.json`
- Resolution Benchmarks: `outputs/resolution_experiments_results.json`
- Experiment Log: `outputs/model_experiments.csv`
