# FINAL MODEL CARD: E2 THREE-SCAN FEATURE FUSION ENSEMBLE

**Model Identifier:** HemoPrint AI Champion Pipeline (E2)  
**Version:** 1.0.0-frozen  
**Release Date:** 2026-10-11  
**Intended Use:** Academic Exploratory Research Prototype for Non-Invasive Biometric Blood Group Analysis  
**Clinical Status:** **STRICTLY NON-DIAGNOSTIC / NON-CLINICAL**. Blood group determination must always be confirmed using standard clinical serological laboratory testing.

---

## 1. Model Architecture & Pipeline Specification

The E2 champion model is a multi-view deep neural network designed to capture multi-impression fingerprint ridge minutiae from three consecutive optical captures per participant.

```
Input: [Scan 1, Scan 2, Scan 3] (Shape: 3 x 112 x 112 x 3, float32 [0, 255])
                    │
   ┌────────────────┼────────────────┐
   ▼                ▼                ▼
Scan 1           Scan 2           Scan 3
   │                │                │
   ▼                ▼                ▼
Frozen           Frozen           Frozen
DenseNet-121     DenseNet-121     DenseNet-121
Backbone         Backbone         Backbone
   │                │                │
   ▼                ▼                ▼
GlobalAvgPool2D  GlobalAvgPool2D  GlobalAvgPool2D
(1024-d)         (1024-d)         (1024-d)
   │                │                │
   └────────────────┬────────────────┘
                    ▼
          Concatenate Layer (3072-d)
                    │
                    ▼
          Dense (512 units, ReLU)
                    │
                    ▼
          Dropout (rate = 0.5)
                    │
                    ▼
          Dense (4 units, Softmax) ──> [P(A+), P(B+), P(AB+), P(O+)]
```

### 1.1 Input Tensor & Preprocessing (C5 Pipeline)
- **Input Tensor Shape:** `(1, 3, 112, 112, 3)` representing three distinct finger impressions.
- **Preprocessing (C5 Specification):**
  1. Alpha compositing onto pure white background (`composite_alpha_on_white`).
  2. Otsu-threshold foreground bounding box localization and cropping (`crop_mfs100_foreground`).
  3. CLAHE adaptive histogram equalization and bilateral ridge filtering (`enhance_mfs100_ridge_contrast`).
  4. Aspect-ratio preserving target resizing to $112 \times 112$ pixels (`resize_to_target`).
  5. Conversion to `np.float32` array with pixel intensity values in $[0.0, 255.0]$.

### 1.2 Target Classes & Explicit Ordering
- **Class 0:** `A+`
- **Class 1:** `B+`
- **Class 2:** `AB+`
- **Class 3:** `O+`

---

## 2. Checkpoint Inventory & Cryptographic SHA-256 Locks

The production ensemble consists of five cross-validation fold checkpoints. All checkpoint weights are cryptographically locked:

| Fold Index | File Path | File Size | SHA-256 Checksum |
| :---: | :--- | :---: | :--- |
| **Fold 1** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | 52,910,112 bytes | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` |
| **Fold 2** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | 52,910,112 bytes | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` |
| **Fold 3** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | 52,910,112 bytes | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` |
| **Fold 4** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | 52,910,112 bytes | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` |
| **Fold 5** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | 52,910,112 bytes | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` |

- **Inference Aggregation Rule:** Flat unweighted mean probability across the 5 checkpoints:
  $$P_{\text{ensemble}} = \frac{1}{5} \sum_{k=1}^5 P_k$$
  $$\hat{y} = \arg\max(P_{\text{ensemble}})$$

---

## 3. Quantitative Performance

### 3.1 Development Cohort (Out-Of-Fold Evaluation, 80 Participants)
*Evaluated on 80 participants (240 scans) using StratifiedGroupKFold cross-validation with zero participant leakage across folds.*

- **Participant Accuracy:** $\mathbf{53.75\%}$ (43/80 correct)
- **Macro F1 Score:** $\mathbf{0.4772}$
- **Balanced Accuracy:** $\mathbf{0.4715}$

#### Development Confusion Matrix:
$$\begin{bmatrix} 12 & 3 & 0 & 4 \\ 8 & 14 & 0 & 3 \\ 2 & 2 & 1 & 2 \\ 10 & 3 & 0 & 16 \end{bmatrix}$$

| Blood Group | Support | Correct | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A+** | 19 | 12 | 0.3750 | 0.6316 | 0.4706 |
| **B+** | 25 | 14 | 0.6364 | 0.5600 | 0.5957 |
| **AB+** | 7 | 1 | 1.0000 | **0.1429** | 0.2500 |
| **O+** | 29 | 16 | 0.6400 | 0.5517 | 0.5926 |
| **Macro Average** | **80** | **43** | **0.6629** | **0.4715** | **0.4772** |

### 3.2 Locked Test Set Evaluation (21 Completely Held-Out Participants)
*Evaluated on 21 participants (63 scans) strictly held out from training, hyperparameter tuning, and early stopping.*

- **Participant Accuracy:** $\mathbf{38.10\%}$ (8/21 correct)
- **Macro F1 Score:** $\mathbf{0.2442}$
- **Balanced Accuracy:** $\mathbf{0.3241}$
- **Comparison to Majority-Class Baseline:** Predicting class `O+` unconditionally on this test distribution achieves $38.10\%$ ($8/21$).
- **Permutation Significance Test:** $p = 0.58$ ($1,000$ Monte Carlo label permutations). The observed test performance is statistically indistinguishable from chance/majority-class heuristics.

---

## 4. Known Technical & Biological Limitations

1. **Generalization Ceiling (38.10%):** While the model reaches 53.75% during cross-validated development optimization, true out-of-distribution performance on unseen individuals drops to 38.10%.
2. **Deficient Minority-Class Recall:** The model exhibits severe sensitivity deficits on rare blood groups, correctly identifying only 1 out of 7 AB+ participants (14.29% recall).
3. **No Definitive Dermatoglyphic-Antigen Biological Link:** Scientific permutation tests ($p = 0.58$) demonstrate that epidermal minutiae do not possess a causal biological correspondence with ABO antigens. Predictions on unseen individuals are susceptible to background demographic priors.
4. **Sensor Surface Sensitivity:** Finger skin moisture, ambient optical reflection, and contact pressure alter ridge valley contrast and can perturb the 3-scan feature representation.

---

## 5. Mandatory Medical & Diagnostic Disclaimer

> [!CAUTION]
> **NOT A CLINICAL DIAGNOSTIC DEVICE**  
> This system is an academic research demonstration prototype designed to explore multi-view metric learning on optical biometric sensors. It has **NOT** been evaluated, validated, or approved by any medical regulatory authority (such as CDSCO, FDA, or CE).  
> **Under no circumstances should any prediction generated by this software be used for clinical decision-making, emergency blood transfusion matching, or patient diagnosis. Blood group must always be determined using standard clinical laboratory serology tests.**

---

## 6. Reproduction Instructions

To verify the frozen model checkpoints and reproduce the evaluation metrics:
```bash
# 1. Verify checkpoint SHA-256 integrity
python -c "
import hashlib, os, json
with open('reports/phase3_final_evaluation/frozen_model_config.json') as f:
    cfg = json.load(f)
for cp in cfg['checkpoint_paths']:
    with open(cp, 'rb') as fp:
        print(cp, hashlib.sha256(fp.read()).hexdigest())
"

# 2. Run the verified development out-of-fold benchmark
python scratch/audit_e2_oof_deep.py

# 3. Verify parity with locked test benchmark
python scripts/provenance/final_evaluation_e2.py
```
