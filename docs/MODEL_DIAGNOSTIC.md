# Baseline Model Diagnostic Report
**Project:** Biometric Blood Group Identification from Fingerprint Images  
**Baseline Model:** ResNet50 Transfer Learning  
**Diagnostic Date:** 2026-09-04  

---

## 1. Baseline Metrics Summary

| Metric | Value |
|---|---|
| Test Accuracy | **57.57%** (517 / 898 correct) |
| Test Loss (SCCE) | 1.1395 |
| Macro Precision | 0.6299 |
| Macro Recall | 0.6116 |
| Macro F1-Score | **0.5691** |
| Weighted F1-Score | 0.5473 |

### Per-Class Performance
| Class | Support | Correct | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A+  | 85  | 80  | 94.1% | 0.597 | 0.941 | 0.731 |
| **A-**  | **152** | **23**  | **15.1%** | **0.742** | **0.151** | **0.251** |
| AB+ | 106 | 93  | 87.7% | 0.484 | 0.877 | 0.624 |
| AB- | 114 | 60  | 52.6% | 0.690 | 0.526 | 0.597 |
| B+  | 97  | 66  | 68.0% | 0.600 | 0.680 | 0.638 |
| B-  | 111 | 49  | 44.1% | 0.925 | 0.441 | 0.598 |
| O+  | 127 | 66  | 51.9% | 0.489 | 0.520 | 0.504 |
| O-  | 106 | 80  | 75.5% | 0.513 | 0.755 | 0.611 |

---

## 2. Confusion Matrix Analysis

### Key Confusion Patterns (test set)

| True → Predicted | Count | Problem |
|---|---|---|
| A- → O+ | **60** | ⚠️ Severe: 39.5% of A- images predicted as O+ |
| A- → AB+ | **32** | ⚠️ Major: 21.1% of A- images predicted as AB+ |
| A- → O-  | **18** | Significant confusion |
| B- → A+  | **16** | B- misidentified as A+ |
| O+ → A+  | **25** | O+ misidentified as A+ |
| AB- → O- | **38** | AB- heavily confused with O- |
| O+ → AB+ | **23** | O+ confused with AB+ |

### Root Causes of A- Catastrophic Failure
1. **A-** has the largest class count in training (705 samples) but worst accuracy — this rules out insufficient data
2. The A- confusion with O+ (60 misclassifications) suggests the model learned **low-level visual features** that overlap between A- and O+ fingerprint images
3. This indicates the classification signal in fingerprint images is genuinely weak/ambiguous for certain blood types
4. The baseline model **barely converged** — with only 8 epochs, the head never properly learned A- discrimination

---

## 3. Training Configuration Analysis

### What the Baseline Did
| Parameter | Value | Issue |
|---|---|---|
| Stage 1 epochs | 5 | Too few — head not converged |
| Stage 2 epochs | 3 | Far too few for meaningful fine-tuning |
| Total epochs | **8** | Critically undertrained |
| Stage 1 LR | 1e-3 | Appropriate |
| Stage 2 LR | 1e-5 | Appropriate but too short |
| Fine-tuned layers | conv5 only | Insufficient — only top residual block |
| Best val accuracy | **60.02%** | Model stopped too early |
| Early stopping | Not used in Stage 1 | Checkpoint saved suboptimally |

### Key Diagnosis
The baseline is **primarily underfitting** from premature termination, not true overfitting.
The 79% training vs 60% validation gap is:
- **Partly** genuine overfitting to training augmentation patterns
- **Mostly** the model still improving when training was cut short at epoch 8

---

## 4. Dataset Analysis

### Class Distribution (Training Set)
| Class | Train Count | Percentage | Imbalance Ratio |
|---|---|---|---|
| A-  | 705 | 16.8% | 1.78× (most common) |
| O+  | 596 | 14.2% | 1.51× |
| AB- | 532 | 12.7% | 1.35× |
| B-  | 519 | 12.4% | 1.31× |
| O-  | 495 | 11.8% | 1.25× |
| AB+ | 493 | 11.8% | 1.25× |
| B+  | 454 | 10.8% | 1.15× |
| A+  | 395 |  9.4% | 1.00× (least common) |

**Imbalance is mild (1.78× max ratio)**. The class imbalance is NOT the primary problem — A- has the MOST training samples but worst accuracy. This confirms the issue is training duration, not data scarcity.

### Image Resolution Concern
| Property | Value |
|---|---|
| Source image size | **96×103 pixels** (majority) |
| Model input size | 224×224 pixels |
| Upsampling factor | ~2.3× |

**Critical observation**: The images are being upsampled from ~96×103 to 224×224. This is a significant stretch and may introduce artifacts. Consider testing models with native 96×96 input (Custom CNN) to avoid upsampling distortion.

---

## 5. Identified Problems and Recommended Solutions

### Problem 1: Severely Undertrained Baseline
**Evidence**: 8 total epochs; validation loss still declining at epoch 8  
**Solution**: Extend training to 20-30 epochs with EarlyStopping(patience=7)

### Problem 2: Insufficient Fine-Tuning Depth  
**Evidence**: Only conv5 blocks unfrozen (ResNet50 has 5 conv stages)  
**Solution**: Unfreeze conv4 + conv5 for ResNet50; for other architectures, unfreeze a larger upper portion

### Problem 3: Poor A- Discrimination
**Evidence**: 15% recall despite having 705 training samples  
**Solution**: 
- More epochs to allow convergence
- Monitor per-class F1 during training (not just accuracy)
- Try mild class weighting adjustments

### Problem 4: Potential Upsampling Artifacts
**Evidence**: Source images are 96×103, upsampled 2.3× to 224×224  
**Solution**: Test smaller input resolution for Custom CNN (96×96 or 128×128)

### Problem 5: No Learning Rate Warmup
**Evidence**: Abrupt LR changes between stages  
**Solution**: Use ReduceLROnPlateau during both stages; consider cosine decay

---

## 6. Scientific Context

This project investigates whether fingerprint ridge patterns encode blood group information. The scientific basis is **not established** and remains research-level. Expected outcomes:
- A dataset with genuine signal may achieve 75-90% accuracy with proper training
- If the visual patterns are weakly correlated with blood group, ceiling may be 65-75%
- The 15% A- recall suggests some classes may genuinely overlap in fingerprint appearance

> **Note**: All predictions must be accompanied by: *"AI-based preliminary estimation only. This result must be confirmed using a standard laboratory blood typing test before medical use."*

---

## 7. Recommended Improvement Strategy

1. **Primary fix**: Train longer (20-30 epochs per stage) with proper EarlyStopping
2. **Architecture diversity**: Test EfficientNetB0 and MobileNetV2 which are better at small/simple datasets
3. **Better fine-tuning**: Unfreeze more backbone layers with appropriate LR
4. **Monitor per-class performance**: Use validation macro-F1 as primary metric (not just accuracy)
5. **Ensemble**: Combine top 2-3 models if individual performance plateaus

**Expected realistic improvement**: From 57.57% → 70-80% with proper training.  
**Aspirational target**: 90%+ requires the dataset to contain strong fingerprint-blood group signal.
