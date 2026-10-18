# FINAL MODEL RECOMMENDATION & DEPLOYMENT VERDICT

**Date:** 2026-10-11  
**Project:** Biometric Blood Group Detection (HemoPrint AI)  
**Evaluation Scope:** 4-Class Positive ABO Blood Groups (`A+`, `B+`, `AB+`, `O+`)  
**Development Cohort:** 80 Participants (240 Scans)  
**Locked Test Set:** 21 Participants (63 Scans) — Fully Untouched  

---

## 1. Executive Summary & Final Verdict

### Final Verdict: **RETAIN STANDALONE E2 FIVE-CHECKPOINT ENSEMBLE WITHOUT FUSION**

Following extensive empirical benchmarking, mathematical auditing, and out-of-fold cross-validation verification across all seven available models:
1. **The E2 Five-Fold Ensemble is the clear superior architecture.** It achieves **53.75% participant accuracy**, **0.4772 Macro F1**, and **0.4715 Balanced Accuracy** on the development cohort under strict out-of-fold evaluation (zero participant leakage).
2. **Model Fusion Causes Negative Transfer.** Combining the large-dataset single-image models (DenseNet-121, EfficientNet-B0, MobileNetV2, ResNet-50, VGG16) with E2 degrades development participant accuracy from **53.75% down to 48.75%** (optimal weighted fusion) and **37.50%** (equal-weight fusion).
3. **No Retraining or Code Modification Required.** The production system (`app.py`) already loads and executes the five E2 checkpoints (`models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5`). No architecture changes or retraining should be performed.

---

## 2. Synthesis of Benchmark Findings

| Model / Fusion Pipeline | Evaluation Scope | Participant Accuracy | Participant Macro F1 | Participant Bal. Acc | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **E2 5-Fold Metric Learning Ensemble** | In-Domain (MFS100 3-Scan) | **53.75%** | **0.4772** | **0.4715** | **Deploy As Champion** |
| **Optimal Weighted Fusion (E2 + Top 3)** | Cross-Domain Ensemble | 48.75% | 0.4346 | 0.4270 | Reject (5.0% performance drop) |
| **Equal-Weight Fusion (Top 2: E2 + EffNet)** | Cross-Domain Ensemble | 43.75% | 0.4240 | 0.4710 | Reject (10.0% performance drop) |
| **Equal-Weight Fusion (All 6 Models)** | Cross-Domain Ensemble | 37.50% | 0.3035 | 0.3608 | Reject (16.25% performance drop) |
| **DenseNet-121** | Single-Scan (External Data) | 33.75% | 0.2843 | 0.2877 | Reject (Single image baseline) |
| **MobileNetV2** | Single-Scan (External Data) | 35.00% | 0.1815 | 0.2550 | Reject (Severe O+ collapse) |
| **ResNet-50 Improved** | Single-Scan (External Data) | 33.75% | 0.2068 | 0.2523 | Reject (0% AB recall) |
| **VGG16** | Single-Scan (External Data) | 32.50% | 0.2400 | 0.3190 | Reject (Weak feature extractor) |
| **EfficientNet-B0** | Single-Scan (External Data) | 21.25% | 0.2060 | 0.3223 | Reject (Low overall accuracy) |
| **Custom CNN** | Single-Scan (External Data) | 23.75% | 0.0960 | 0.2500 | Reject (Single-class collapse) |

---

## 3. Scientific Rationale: Why Cross-Domain Fusion Fails

### 3.1 Domain Shift Between Ink/Synthetic Fingerprints and Live Optical Sensors
The six legacy models were trained on ~6,000 fingerprint impressions originating from synthetic generation and ink-rolled fingerprint cards. When presented with MFS100 optical live-scan captures:
- Ridge frequency, elastic skin deformation, and sensor noise differ substantially.
- These models exhibit high prior probability bias (collapsing toward O+ and B+).
- Blending these noisy cross-domain probability vectors with E2 corrupts E2's calibrated latent feature representations.

### 3.2 Single-Scan Voting vs. Multi-Scan Feature Fusion
- Legacy models evaluate individual fingerprint scans in isolation, followed by post-hoc majority voting. Single-scan captures frequently suffer from pressure variation and incomplete contact.
- E2 utilizes a **multi-tower Siamese feature concatenation backbone** ($3 \times 1024 = 3072$ features) that jointly processes 3 impressions from the same participant before classification. This architectural advantage cannot be replicated by decision-level probability averaging.

---

## 4. Realistic Generalization Ceiling & Academic Integrity

### 4.1 Strict Distinction: Development vs. Held-Out Test Generalization
- **Development Cohort (80 Participants):**
  - E2 achieves **53.75% out-of-fold accuracy** and **0.4772 Macro F1**.
  - All 80 participants were used across the 5 cross-validation folds to guide contrastive learning and early stopping.
- **Held-Out Test Cohort (21 Participants):**
  - In the locked external evaluation (`reports/phase3_final_evaluation/FINAL_TEST_AUDIT.md`), E2 achieved **38.10% participant accuracy** (8/21 correct).
  - Permutation testing across 1,000 label shuffles yielded a p-value of **0.58**, indicating that dermatoglyphic ridges do not encode a deterministic biological signal for ABO blood group antigens.

### 4.2 Academic Honesty Commitment
- **Do NOT claim that accuracy will exceed 50% on unseen populations.**
- Expecting a biometric fingerprint model to reliably diagnose blood groups in a clinical setting contradicts the empirical held-out generalization ceiling of ~38.10%.
- The prototype must always be presented as an **academic exploratory research prototype**, equipped with active confidence scoring, serological disclaimers, and referral to standard laboratory testing.

---

## 5. Pre-Requisites Before Final Locked-Test Evaluation

Before any final locked-test benchmark evaluation or presentation demonstration:
1. **Freeze Model Checkpoints:** The SHA-256 hashes of all five E2 weight files (`models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5`) must remain verified against the integrity manifest.
2. **Verify Hardware Bridge & C5 Pipeline:** Ensure `mfs100_bridge.py` and `src/preprocessing.py` maintain strict functional equivalence with training conditions (alpha compositing on white background, foreground bounding box cropping, ridge contrast enhancement, resize to $112 \times 112$).
3. **Preserve Locked Test Set Isolation:** The 21 test participants in `reports/phase3_participant_benchmark/participant_split.csv` must NEVER be accessed for hyperparameter tuning, temperature scaling, threshold calibration, or model selection.
4. **Active Safety Layer:** Verify that `app.py` displays warning banners on all low-confidence predictions ($\text{confidence} < 0.45$), stating that fingerprint-based estimation is experimental and cannot replace laboratory serology.
