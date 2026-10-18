# Prioritized Improvement Plan: Controlled Experiments

**Investigation:** Accuracy Improvement — Root-Cause Diagnostic  
**Date:** 2026-10-13  
**Status:** COMPLETE & VERIFIED  
**Output Path:** `reports/accuracy_improvement/experiment_plan.md`  

---

## 1. Executive Summary

Based on the diagnostic findings across datasets, domain gaps, preprocessing, error distributions, and data boundaries, this document defines **five ranked, controlled experiments** aimed at improving biometric classification accuracy on MFS100 fingerprint data.

### Ground Rules Observed:
- No existing model checkpoint is modified or overwritten.
- All experiments will be implemented in isolated experimental scripts (`experiments/` or `scratch/`).
- The 21-participant locked test set remains completely quarantined during model development.
- The 80 development participants represent the sole training and validation pool.
- Model selection decisions will be guided by **Balanced Accuracy** and **Macro F1**, not raw accuracy.

---

## 2. Ranked Experiment Matrix

| Rank | Experiment Identifier | Target Problem | Core Technique | Primary Metric Targeted | Expected Feasibility |
|:---:|:---|:---|:---|:---:|:---:|
| **1** | **EXP-01: Class-Balanced Loss for E2** | Total recall collapse on AB+ and B+ | Focal Loss ($\gamma=2.0$) or Inverse-Frequency Class Weighting | Macro F1, AB+ / B+ Recall | High (loss function modification only) |
| **2** | **EXP-02: Preprocessing Domain Alignment for EffNet** | EfficientNet ingests raw uncropped frames with dark borders | Apply E2 foreground cropping (`transform_c5`) at $224 \times 224$ | Dev Balanced Accuracy | High (preprocessing pipeline change) |
| **3** | **EXP-03: Low-Learning-Rate Fine-Tuning of EffNet on MFS100** | Sensor domain shift (optical vs. MFS100) | Two-stage fine-tuning (frozen backbone $\rightarrow$ top blocks) on 64 dev participants | OOF Dev Accuracy, Prior Rebalancing | Medium (training required on dev pool) |
| **4** | **EXP-04: Realistic Fingerprint Augmentation** | Augmented data lacks real physical sensor contact dynamics | Add elastic deformation, Gaussian blur, and morphological pressure simulation | Invariance to pressure/moisture | High (augmentation layer in pipeline) |
| **5** | **EXP-05: High-Resolution E2 Backbone ($224 \times 224$)** | Extreme downsampling ($112 \times 112$) destroys ridge minutiae | Scale E2 fusion architecture to ingest $224 \times 224 \times 3$ per scan | Spatial ridge frequency retention | Medium (requires retraining E2 on dev pool) |

---

## 3. Detailed Experiment Specifications

### Experiment 1: Class-Balanced / Focal Loss for E2 (Rank 1)
- **1. Target Problem:** E2 achieves 0.0% recall on AB+ and B+ on the locked test, collapsing predictions heavily into A+.
- **2. Supporting Evidence:** The 80 development participants contain only 7 AB+ subjects ($8.8\%$). In standard cross-entropy, the gradients from O+ (29 subjects) and B+ (25 subjects) overwhelm the minority class representation.
- **3. Files & Functions Involved:**
  - Script: `scripts/experiment6_fusion_supcon.py`
  - Function: Loss computation (lines 140–165).
- **4. Required Changes:**
  - Create experimental script: `experiments/exp01_class_balanced_e2.py`.
  - Replace categorical cross-entropy with Focal Loss:
    $$L_{\text{focal}} = - \alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0$$
    with inverse class frequency weighting $\alpha_t = \frac{N}{4 \times N_c}$.
- **5. Permitted Data:** 80 development participants evaluated via 5-fold cross-validation (`StratifiedGroupKFold`). Locked test strictly quarantined.
- **6. Baseline:** Baseline E2 OOF performance (Macro F1: 0.4772, Balanced Acc: 47.15%, AB+ Recall: 14.29%).
- **7. Metrics to Report:** OOF Macro F1, Balanced Accuracy, Per-class recall (especially AB+ and B+), Confusion Matrix.
- **8. Risks:** Potential over-correction where the model predicts AB+ excessively, dropping overall accuracy.
- **9. Acceptance Criteria:** Statistically significant increase in validation Macro F1 ($\ge +0.05$) and AB+ recall ($\ge 30\%$) without dropping overall accuracy below 45%.

---

### Experiment 2: Preprocessing Domain Alignment for EfficientNet-B0 (Rank 2)
- **1. Target Problem:** EfficientNet-B0 was evaluated on full uncropped MFS100 frames containing heavy black platen borders and ambient glass reflection, inducing 74% O+ prior collapse.
- **2. Supporting Evidence:** `domain_shift_analysis.md` revealed that MFS100 raw images have 55% foreground coverage and $11\times$ lower Laplacian sharpness than the original dataset.
- **3. Files & Functions Involved:**
  - Module: `src/preprocessing.py`
  - Harness: `scripts/evaluate_models.py` $\rightarrow$ `load_and_preprocess_effnet()`.
- **4. Required Changes:**
  - Create experimental harness: `experiments/exp02_effnet_preprocessing_alignment.py`.
  - Modify preprocessing to apply `crop_mfs100_foreground()` and `enhance_mfs100_ridge_contrast()` before resizing with Lanczos to $224 \times 224$.
- **5. Permitted Data:** 80 development participants. Frozen checkpoint `models/abo_positive_4class/best_model.keras` (no retraining).
- **6. Baseline:** Current EfficientNet-B0 development performance (Accuracy: 32.50%, Balanced Acc: 23.91%, Macro F1: 0.1885).
- **7. Metrics to Report:** Balanced Accuracy, Macro Precision, Macro F1, Predicted Class Distribution.
- **8. Risks:** CLAHE contrast enhancement might amplify noise artifacts that the ImageNet backbone misinterprets.
- **9. Acceptance Criteria:** Development Balanced Accuracy increases above 30.0% and O+ prediction fraction drops below 55%.

---

### Experiment 3: Two-Stage Fine-Tuning of EfficientNet-B0 on MFS100 Development Pool (Rank 3)
- **1. Target Problem:** EfficientNet-B0 was trained entirely on a different optical hardware domain and has never adapted to Mantra MFS100 sensor physics.
- **2. Supporting Evidence:** Cross-domain transfer accuracy is only 28.57% (worse than the 38.10% Always-O+ baseline).
- **3. Files & Functions Involved:**
  - Script: `scripts/train_abo_positive_4class.py`.
- **4. Required Changes:**
  - Create experimental script: `experiments/exp03_effnet_mfs100_finetune.py`.
  - Load existing checkpoint `models/abo_positive_4class/best_model.keras`.
  - Stage 1: Freeze backbone, train classification head with Adam (learning rate $= 10^{-4}$) for 10 epochs on 64 development participants.
  - Stage 2: Unfreeze top 20 layers of EfficientNet-B0 with very low learning rate ($10^{-5}$) with Cosine Decay and Early Stopping on the 16 validation participants.
- **5. Permitted Data:** 64 development participants for training, 16 development participants for validation. Locked test strictly quarantined.
- **6. Baseline:** Frozen EfficientNet-B0 development accuracy (32.50%) and validation accuracy.
- **7. Metrics to Report:** Validation Accuracy, Macro F1, Balanced Accuracy, Per-class recall.
- **8. Risks:** Overfitting to the 64 training participants ($N=192$ images).
- **9. Acceptance Criteria:** Validation Macro F1 $\ge 0.40$ on the held-out 16 validation participants without identity memorization.

---

### Experiment 4: Realistic Fingerprint Augmentation Layer (Rank 4)
- **1. Target Problem:** Current training data augmentation ($\pm 2.7\%$ rotation/zoom) is too rigid and fails to reflect live-scan pressure variations, skin dryness, and sensor noise.
- **2. Supporting Evidence:** Within-participant scans exhibit substantial intensity and ridge thickness variance ($158 \pm 86$).
- **3. Files & Functions Involved:**
  - `src/preprocessing.py` and augmentation blocks in model builders.
- **4. Required Changes:**
  - Implement a domain-specific augmentation pipeline:
    - Random morphological dilation/erosion (kernel $2\times 2$, probability $0.3$) to simulate finger contact pressure.
    - Random Gaussian blur ($\sigma \in [0.5, 1.5]$, probability $0.3$) to simulate skin moisture.
    - Random Gaussian sensor noise ($\mu=0, \sigma=10$, probability $0.2$).
- **5. Permitted Data:** 80 development participants in 5-fold cross-validation.
- **6. Baseline:** Baseline E2 5-fold OOF performance.
- **7. Metrics to Report:** 5-Fold OOF Accuracy, Macro F1, Balanced Accuracy.
- **8. Risks:** Excessive distortion could destroy authentic ridge valley continuity.
- **9. Acceptance Criteria:** Measurable reduction in within-participant prediction variance and OOF Macro F1 $\ge 0.50$.

---

### Experiment 5: High-Resolution E2 Architecture ($224 \times 224$) (Rank 5)
- **1. Target Problem:** Compressing $316 \times 354$ scans into $112 \times 112$ reduces ridge spatial wavelength below Nyquist limits, eliminating minutiae.
- **2. Supporting Evidence:** 500 DPI fingerprint standards require $\ge 8$ pixels per ridge period; at $112\times 112$, the period collapses to $<3$ pixels.
- **3. Files & Functions Involved:**
  - Model builder in `scripts/experiment6_fusion_supcon.py` $\rightarrow$ `build_E2()`.
- **4. Required Changes:**
  - Update input tensor to $(3, 224, 224, 3)$.
  - Adjust feature concatenation (DenseNet-121 outputs $7\times 7 \times 1024$ GAP features = 3072 features, same head dimension).
  - Retrain on 80 development participants via 5-fold cross-validation.
- **5. Permitted Data:** 80 development participants only.
- **6. Baseline:** $112\times 112$ E2 baseline (53.75% OOF accuracy).
- **7. Metrics to Report:** OOF Accuracy, Macro F1, Balanced Accuracy, training memory footprint.
- **8. Risks:** Increased GPU memory footprint and potential overfitting given the larger receptive field.
- **9. Acceptance Criteria:** OOF Macro F1 $\ge 0.52$ and improved per-class balance.

---

## 4. Experiment Execution Roadmap

```
Step 1: EXP-01 (Class-Balanced Loss for E2) 
        ├── Verify OOF Macro F1 on 80 Dev Pool
        └── If successful, freeze as candidate E2-v2

Step 2: EXP-02 (Preprocessing Domain Alignment for EffNet)
        ├── Test on 80 Dev Pool without retraining
        └── Measure reduction in O+ prior bias

Step 3: EXP-03 (Domain Fine-Tuning of EffNet on Dev Pool)
        ├── Train head + top layers on 64 Dev Train, evaluate on 16 Dev Val
        └── Measure genuine cross-domain transfer

Step 4: EXP-04 & EXP-05 (Augmentation & High-Resolution Scaling)
        └── Implement if Steps 1-3 validate improved ridge feature learning

Step 5: FINAL LOCKED TEST BENCHMARK
        └── ONLY AFTER candidate pipeline is completely frozen on development data.
```
