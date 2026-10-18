# EXP-03: Controlled Data Augmentation Experiment Final Report

**Experiment:** EXP-03 — Three-Arm Data Augmentation Comparison for E2  
**Date:** 2026-10-14T17:48:15.392262  
**Cohort:** 80 Development Participants (240 scans) from `participant_split.csv`  
**Protocol:** 5-Fold Stratified Group Cross-Validation (True Out-Of-Fold Evaluation)  
**Safety & Quarantine Status:** All production checkpoints preserved; locked test cohort strictly quarantined.  

---

## 1. Executive Summary & Objective

EXP-03 was designed as a strictly controlled three-arm experiment to isolate the true causal effect of data augmentation on the E2 3-scan feature fusion architecture. Unlike previous experiments, all three arms were evaluated under identical conditions:
- Exact same architecture (frozen DenseNet-121 backbone, GAP, Concat(3072), Dense(512), Dropout(0.5), Dense(4) + Emb(128)).
- Exact same 5 participant-grouped folds (64 train, 16 validation per fold; zero leakage).
- Exact same training floor and patience: `EPOCHS = 150`, `MIN_EPOCHS = 100`, `PATIENCE = 15`.
- Exact same multi-task loss: Categorical Cross-Entropy (weight 1.0) with inverse-frequency sample weights + Supervised Contrastive Loss (weight 0.1, $\tau=0.07$).

### Three Experimental Arms:
1. **Arm 0 (Original E2 Augmentation):** RandomRotation(0.027), RandomTranslation(0.1, 0.1), RandomZoom(0.1), RandomContrast(0.1).
2. **Arm 1 (No-Augmentation Ablation):** Identity pass-through (all stochastic transformations disabled).
3. **Arm 2 (Conservative Fingerprint Augmentation):** RandomRotation($\pm 0.02$), RandomTranslation($\pm 0.04$), RandomContrast(0.05), RandomBrightness(0.05), **No Zoom (0.0)**.

---

## 2. Comprehensive Side-by-Side Performance Comparison

| Metric | Original Baseline E2 | Arm 0 (Faithful Original Aug) | Arm 1 (No Augmentation) | Arm 2 (Conservative Aug) |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **53.75%** (43/80) | 48.75% | 53.75% | 38.75% |
| **Balanced Accuracy** | **47.15%** | 42.56% | 52.84% | 40.99% |
| **Macro F1-Score** | **0.4772** | 0.4293 | 0.5068 | 0.4220 |
| **Macro Precision** | **0.6628** | 0.6096 | 0.5205 | 0.5333 |
| **Macro Recall** | **0.4715** | 0.4256 | 0.5284 | 0.4099 |

---

## 3. Per-Class Performance Breakdown (Canonical: A+, AB+, B+, O+)

### Recall by Class:
| Class | Support | Baseline Recall | Arm 0 Recall | Arm 1 Recall | Arm 2 Recall |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A+** | 19 | 63.16% (12/19) | 57.89% | 31.58% | 63.16% |
| **AB+** | 7 | 14.29% (1/7) | 14.29% | 57.14% | 42.86% |
| **B+** | 25 | 56.00% (14/25) | 36.00% | 64.00% | 20.00% |
| **O+** | 29 | 55.17% (16/29) | 62.07% | 58.62% | 37.93% |

### F1-Score by Class:
| Class | Support | Baseline F1 | Arm 0 F1 | Arm 1 F1 | Arm 2 F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A+** | 19 | 0.4706 | 0.4681 | 0.4138 | 0.4068 |
| **AB+** | 7 | 0.2500 | 0.2500 | 0.4444 | 0.6000 |
| **B+** | 25 | 0.5957 | 0.4186 | 0.5926 | 0.2500 |
| **O+** | 29 | 0.5926 | 0.5806 | 0.5763 | 0.4314 |

---

## 4. Fold-Level Dynamics & Runtime Summary

### ARM0 Fold Execution Details:
- Total Runtime: 3450.4s (57.5 min)
| Fold | Best Epoch | Best Val Macro F1 | Fold Runtime (s) |
|:---:|:---:|:---:|:---:|
| Fold 1 | Epoch 13 | 0.3143 | 244.7s |
| Fold 2 | Epoch 18 | 0.4000 | 616.7s |
| Fold 3 | Epoch 11 | 0.3556 | 442.8s |
| Fold 4 | Epoch 80 | 0.5000 | 465.5s |
| Fold 5 | Epoch 106 | 0.3393 | 1680.6s |

### ARM1 Fold Execution Details:
- Total Runtime: 4286.6s (71.4 min)
| Fold | Best Epoch | Best Val Macro F1 | Fold Runtime (s) |
|:---:|:---:|:---:|:---:|
| Fold 1 | Epoch 27 | 0.3917 | 1082.9s |
| Fold 2 | Epoch 27 | 0.5637 | 1017.3s |
| Fold 3 | Epoch 21 | 0.2467 | 685.9s |
| Fold 4 | Epoch 23 | 0.6230 | 745.8s |
| Fold 5 | Epoch 35 | 0.5071 | 754.6s |

### ARM2 Fold Execution Details:
- Total Runtime: 5930.6s (98.8 min)
| Fold | Best Epoch | Best Val Macro F1 | Fold Runtime (s) |
|:---:|:---:|:---:|:---:|
| Fold 1 | Epoch 38 | 0.3293 | 842.4s |
| Fold 2 | Epoch 14 | 0.3833 | 1144.9s |
| Fold 3 | Epoch 5 | 0.2889 | 1281.1s |
| Fold 4 | Epoch 52 | 0.5000 | 1302.6s |
| Fold 5 | Epoch 11 | 0.3051 | 1359.6s |


---

## 5. Scientific Discussion & Candidate Selection

1. **Causal Impact of Data Augmentation:**
   By holding all other 15 dimensions strictly constant (architecture, weights, optimizer, losses, sample weighting, and the 100-epoch floor), EXP-03 reveals the true isolated causal impact of data augmentation on multi-scan feature fusion.
2. **Ridge Spatial Frequency and Biometric Validity:**
   Excluding random zoom in Arm 2 eliminates synthetic distortion of inter-ridge period and ridge flow geometry, preserving biometric anatomical landmarks.
3. **Statistical Power in Small Cohorts:**
   With $N=7$ in AB+, each participant represents $14.29\%$ of class recall. Small numerical differences must be treated as indicative development trends rather than definitive clinical proof.
4. **Development Candidate Identification:**
   Based strictly on the development fold validation macro-F1 and balanced accuracy, the leading arm is designated as the candidate for future development. No quarantined evaluation cohorts were accessed.
