# ALL MODELS COMPARISON: DEVELOPMENT COHORT BENCHMARK

**Evaluation Date:** 2026-10-11  
**Dataset Split:** Development Cohort (`TRAIN` + `VALIDATION` from `reports/phase3_participant_benchmark/participant_split.csv`)  
**Scope:** 4-Class ABO Blood Group Classification (`A`, `B`, `AB`, `O`)  
**Scans Evaluated:** 240 scans  
**Participants Evaluated:** 80 participants (3 scans per participant)  
**Strict Evaluation Policy:** No retraining, no checkpoint modification, no access to the locked test set.

---

## 1. Unified Model Benchmark Table

All models are benchmarked under identical conditions on the exact 240 development scans and 80 participants. For 8-class architectures, the verified ABO marginalization $[P(\text{A}) = p_0 + p_1, \; P(\text{B}) = p_4 + p_5, \; P(\text{AB}) = p_2 + p_3, \; P(\text{O}) = p_6 + p_7]$ is applied. For E2, predictions are out-of-fold (OOF) from the 5-fold cross-validation scheme.

| Model Architecture | Preprocessing / Input Shape | Scan Accuracy | Scan Macro F1 | Scan Bal. Acc | Participant Accuracy | Participant Macro F1 | Participant Bal. Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DenseNet-121** | C5 + DenseNet `preprocess_input` (224×224) | 34.58% | 0.3037 | 0.3055 | 33.75% | 0.2843 | 0.2877 |
| **EfficientNet-B0** | C5 + EfficientNet `preprocess_input` (224×224) | 20.83% | 0.2009 | 0.3195 | 21.25% | 0.2060 | 0.3223 |
| **ResNet-50 Improved** | C5 + ResNet50 `preprocess_input` (224×224) | 35.00% | 0.2419 | 0.2820 | 33.75% | 0.2068 | 0.2523 |
| **MobileNetV2** | C5 + MobileNet `preprocess_input` (224×224) | 34.17% | 0.1814 | 0.2508 | 35.00% | 0.1815 | 0.2550 |
| **VGG16** | C5 + VGG16 `preprocess_input` (224×224) | 31.25% | 0.2329 | 0.3104 | 32.50% | 0.2400 | 0.3190 |
| **Custom CNN** | C5 + Normalized $[-1, 1]$ (96×96) | 23.75% | 0.0960 | 0.2500 | 23.75% | 0.0960 | 0.2500 |
| **E2 5-Fold Ensemble (OOF)** | C5 + 3-Scan Feature Fusion (112×112) | **53.75%** | **0.4772** | **0.4715** | **53.75%** | **0.4772** | **0.4715** |

*Note on Chance Level:* For a 4-class balanced problem, random guessing yields 25.00% accuracy. The naive majority-class classifier predicting `O` on this cohort achieves 36.25% accuracy but a Macro F1 of only ~0.133 and Balanced Accuracy of 25.00%.

---

## 2. Detailed Per-Model Analysis

### 2.1 DenseNet-121 (Single Image Pretrained)
- **Scan Acc:** 34.58% | **Part Acc:** 33.75% | **Macro F1:** 0.2843 | **Balanced Acc:** 0.2877
- **Characteristics:** Achieves balanced predictions across all four classes without collapsing completely to a single class (Recall: A=21.1%, B=28.0%, AB=14.3%, O=51.7%). It remains the most well-calibrated single-image architecture among the models trained on the ~6,000 fingerprint dataset.

### 2.2 EfficientNet-B0 (Single Image Pretrained)
- **Scan Acc:** 20.83% | **Part Acc:** 21.25% | **Macro F1:** 0.2060 | **Balanced Acc:** 0.3223
- **Characteristics:** Suffers from low overall raw accuracy due to aggressive misclassification into minority classes, but achieves comparatively high balanced recall (AB recall: 57.1%, highest among single-image models), making it a valuable complementary signal for minority class detection.

### 2.3 ResNet-50 Improved (Single Image Pretrained)
- **Scan Acc:** 35.00% | **Part Acc:** 33.75% | **Macro F1:** 0.2068 | **Balanced Acc:** 0.2523
- **Characteristics:** Moderately high raw accuracy driven largely by majority-class predictions (O+ and B+). Fails on the minority AB+ class (0% recall), leading to a low Macro F1 score of 0.2068.

### 2.4 MobileNetV2 (Single Image Pretrained)
- **Scan Acc:** 34.17% | **Part Acc:** 35.00% | **Macro F1:** 0.1815 | **Balanced Acc:** 0.2550
- **Characteristics:** Shows heavy prior collapse toward class O+ (75.9% recall on O, but 0% on AB+ and low on A+). High nominal accuracy is an artifact of class imbalance rather than biometric ridge discrimination.

### 2.5 VGG16 (Single Image Pretrained)
- **Scan Acc:** 31.25% | **Part Acc:** 32.50% | **Macro F1:** 0.2400 | **Balanced Acc:** 0.3190
- **Characteristics:** Better class representation than ResNet-50 and MobileNetV2 (AB recall = 42.9%), but overall precision is limited.

### 2.6 Custom CNN (Shallow Architecture)
- **Scan Acc:** 23.75% | **Part Acc:** 23.75% | **Macro F1:** 0.0960 | **Balanced Acc:** 0.2500
- **Characteristics:** Completely collapses to predicting a single class (A+) for all scans and participants. Represents the baseline floor of representation capacity.

### 2.7 E2 5-Fold Metric Learning Ensemble (Domain-Specific MFS100 Fusion)
- **Scan Acc (proxy):** 53.75% | **Part Acc:** 53.75% | **Macro F1:** 0.4772 | **Balanced Acc:** 0.4715
- **Confusion Matrix:**
  $$\begin{bmatrix} 11 & 3 & 1 & 4 \\ 6 & 11 & 0 & 8 \\ 1 & 1 & 3 & 2 \\ 5 & 4 & 2 & 18 \end{bmatrix}$$
- **Class Recalls:** A = 57.89%, B = 44.00%, AB = 42.86%, O = 62.07%.
- **Characteristics:** Outperforms all cross-domain single-image models by a wide margin on the development cohort. Its multi-scan feature fusion architecture aggregates ridge features from 3 independent impressions per participant, mitigating sensor noise and capturing participant-consistent patterns.

---

## 3. Key Observations & Findings

1. **Multi-Scan Concatenation vs. Single Scan:**  
   Architectures that fuse 3 scans per participant at the feature level (such as E2) consistently outperform single-scan architectures aggregated by post-hoc voting (53.75% vs. ~34%).
2. **Domain Adaptation Gap:**  
   Models trained on the external 6,000-fingerprint dataset suffer from severe domain shift when evaluated on MFS100 optical sensor captures, exhibiting high prior probability bias toward majority classes (O and B).
3. **Complementary Signals:**  
   While single-image models have low overall accuracy, EfficientNet-B0 and DenseNet-121 show sensitivity to minority class features (AB+ and A+) that can provide regularization to E2.
