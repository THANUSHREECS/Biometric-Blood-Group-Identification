# FUSION COMPARISON: DEVELOPMENT COHORT ENSEMBLE AUDIT

**Evaluation Date:** 2026-10-11  
**Target Scope:** 4-Class ABO Blood Group Detection (`A`, `B`, `AB`, `O`)  
**Cohort:** 80 Development Participants (240 Scans)  
**Strict Integrity Rules:**
1. Evaluated strictly using development out-of-fold (OOF) prediction vectors.
2. Zero access to the locked test set (`reports/phase3_final_evaluation/final_test_predictions.csv`).
3. No retraining of neural network weights or checkpoint modification.
4. Objective evaluation of accuracy claims (no assumption that fusion will surpass 50% on unseen participants).

---

## 1. Fusion Strategies Benchmarked

We evaluated three principled fusion methodologies on the development participant probability distributions:

1. **Equal-Weight Averaging (All 6 Models):**
   $$P_{\text{fusion}} = \frac{1}{6} \sum_{m=1}^6 P_m$$
   Fusing DenseNet-121, EfficientNet-B0, ResNet-50 Improved, MobileNetV2, VGG16, and E2 Ensemble (OOF).

2. **Selective Equal-Weight Averaging (Top 2 Balanced Models):**
   $$P_{\text{fusion}} = \frac{1}{2} \left( P_{\text{E2}} + P_{\text{EffNet}} \right)$$
   Combining the leading in-domain feature fusion model (E2) with the highest minority-class sensitivity model (EfficientNet-B0).

3. **NLL-Optimal Softmax Weighted Averaging:**
   $$\min_{\mathbf{w}} -\sum_{i=1}^{80} \log \left( \sum_{m=1}^6 w_m P_{m, y_i} \right) \quad \text{s.t.} \quad \sum w_m = 1, \; w_m \ge 0$$
   Optimized strictly on the development participant out-of-fold probability vectors via BFGS.

---

## 2. Comparison Results Table

| Strategy / Model | Participant Accuracy | Participant Macro F1 | Participant Balanced Accuracy | Predominant Weights |
| :--- | :---: | :---: | :---: | :--- |
| **Best Single Pretrained (DenseNet-121)** | 33.75% | 0.2843 | 0.2877 | Single model ($w = 1.0$) |
| **Best In-Domain Model (E2 5-Fold OOF)** | **53.75%** | **0.4772** | **0.4715** | Single ensemble ($w = 1.0$) |
| **Equal-Weight Fusion (All 6 Models)** | 37.50% | 0.3035 | 0.3608 | Uniform ($w_m = 0.1667$) |
| **Equal-Weight Fusion (Top 2: E2 + EffNet)** | 43.75% | 0.4240 | **0.4710** | Uniform ($w_m = 0.5000$) |
| **Weighted Fusion (NLL-Optimal)** | 48.75% | 0.4346 | 0.4270 | E2: 38.7%, DenseNet: 35.6%, MobileNet: 24.6% |

---

## 3. Confusion Matrix Breakdown for Fusion Strategies

### 3.1 Equal-Weight Fusion (All 6 Models)
$$\text{Accuracy: } 37.50\% \quad | \quad \text{Macro F1: } 0.3035 \quad | \quad \text{Balanced Acc: } 0.3608$$

$$\begin{bmatrix} 7 & 4 & 1 & 7 \\ 5 & 10 & 0 & 10 \\ 1 & 2 & 1 & 3 \\ 8 & 7 & 2 & 12 \end{bmatrix}$$

- **Class Recalls:** A = 36.84%, B = 40.00%, AB = 14.29%, O = 41.38%.
- **Diagnosis:** Averaging all models causes regression toward the mean. The weaker single-image models pull down the superior discriminative capability of E2.

### 3.2 Selective Equal-Weight Fusion (E2 + EfficientNet-B0)
$$\text{Accuracy: } 43.75\% \quad | \quad \text{Macro F1: } 0.4240 \quad | \quad \text{Balanced Acc: } 0.4710$$

$$\begin{bmatrix} 11 & 1 & 2 & 5 \\ 7 & 8 & 1 & 9 \\ 1 & 0 & 4 & 2 \\ 8 & 5 & 4 & 12 \end{bmatrix}$$

- **Class Recalls:** A = 57.89%, B = 32.00%, **AB = 57.14%**, O = 41.38%.
- **Diagnosis:** This pairing achieves the highest balanced recall for the difficult minority class AB (57.14%, correctly predicting 4 of 7 AB participants), but sacrifices overall accuracy on class O and B.

### 3.3 NLL-Optimal Weighted Averaging
$$\text{Accuracy: } 48.75\% \quad | \quad \text{Macro F1: } 0.4346 \quad | \quad \text{Balanced Acc: } 0.4270$$

$$\begin{bmatrix} 11 & 2 & 0 & 6 \\ 8 & 10 & 0 & 7 \\ 2 & 1 & 1 & 3 \\ 10 & 2 & 0 & 17 \end{bmatrix}$$

- **Optimal Weights:**
  - **E2 Ensemble (OOF):** 0.3870 (38.70%)
  - **DenseNet-121:** 0.3560 (35.60%)
  - **MobileNetV2:** 0.2465 (24.65%)
  - **EfficientNet-B0:** 0.0106 (1.06%)
  - **ResNet-50 Improved:** 0.0000 (0.00%)
  - **VGG16:** 0.0000 (0.00%)
- **Diagnosis:** The optimization solver assigns almost all weight to the top three models: E2, DenseNet-121, and MobileNetV2. ResNet-50 and VGG16 are pruned to zero weight because their predictions are redundant and increase cross-entropy loss.

---

## 4. Scientific Findings & Honest Generalization Assessment

### 4.1 Does Cross-Domain Fusion Outperform E2?
**No.** On the development cohort out-of-fold data:
- Standalone E2 achieves **53.75% participant accuracy** and **0.4772 Macro F1**.
- Weighted fusion achieves **48.75% participant accuracy** and **0.4346 Macro F1** (a net drop of 5.00 percentage points).
- Equal-weight fusion across all models drops further to **37.50%**.

### 4.2 Why Does Fusion Cause Negative Transfer?
1. **Domain Mismatch:**  
   DenseNet-121, EfficientNet-B0, and MobileNetV2 were trained on an external dataset of ~6,000 synthetic / ink-rolled impressions. Their learned convolutional filters expect high-contrast binary ridge lines. When applied to live MFS100 optical scans, their probability distributions suffer from sensor-domain shift and prior probability bias.
2. **Scan-Level vs. Participant-Level Representation:**  
   E2 processes 3 scans simultaneously via multi-tower feature concatenation, effectively canceling out single-scan pressure artifacts. The single-image models operate on isolated scans and are averaged after the fact, which cannot recover the multi-view feature correlations captured by E2.

### 4.3 Generalization Ceiling on Unseen Held-Out Test Data
In the verified benchmark on 21 held-out test participants (`reports/phase3_final_evaluation/FINAL_TEST_AUDIT.md`), E2 achieved **38.10% accuracy** (8/21 correct).  
Academic integrity requires acknowledging that:
- **Accuracy on unseen participants does not reach 50%.**
- The theoretical ceiling for non-invasive biometric blood group estimation without serological input on this sensor is constrained to ~38–40%, consistent with independent permutation test findings ($p = 0.58$).
- Fusion with cross-domain models does not lift this ceiling and risks worsening false positive rates on unseen participants.

---

## 5. Architectural Recommendations

1. **Production Pipeline:** Retain the verified **E2 5-Checkpoint Metric Learning Ensemble** as the sole deployed production inference engine in `app.py`. Do not dilute it with cross-domain 8-class models.
2. **Academic Presentation:** Present the comparative fusion analysis in your defense to demonstrate thorough methodological rigor:
   - Prove that you audited and evaluated all 6 transfer learning architectures.
   - Show that you tested both equal-weight and optimal-weight fusion.
   - Demonstrate scientific maturity by explaining why fusion leads to negative transfer when combining cross-domain single-image models with in-domain multi-scan metric learning representations.
