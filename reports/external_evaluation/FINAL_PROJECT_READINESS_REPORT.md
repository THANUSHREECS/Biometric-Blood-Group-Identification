# Final Project Readiness Report: Executive Summary

## Overall Project Status
The Biometric Blood Group Detection project (4-class: A+, B+, AB+, O+) using the Mantra MFS100 fingerprint scanner is prepared for final evaluation. The application has been fully audited to ensure no untested claims are presented to users.

## Current Evidence (Verified Metrics)

| Evaluation Setting | Model / Architecture | Samples / Scope | Metric / Accuracy | Baseline & Chance | Status & Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Dataset)** | **Model A: EfficientNetB0** (not deployed) | 415 test images (random image split) | **86.02%** (95% CI: [82.65%, 88.92%]) | Majority baseline: 30.6% | Standalone single-image model; not deployed in production pipeline |
| **Dataset Benchmark** | **Model B: E2 DenseNet121 Ensemble** (deployed) | Dataset test set | **Not reliably measured** | Chance: ~25.0% | Multi-scan 5-fold ensemble; training script recovered at `scripts/experiment6_fusion_supcon.py` (9 Oct 2026) |
| **Locked External Test** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 21 Locked Test Participants (63 scans, 3 per person) | **38.10%** (8/21, Wilson CI: [20.75%, 59.12%]) | Always-O+ baseline: 38.10% (8/21, exact binomial p=0.5817) | True locked test set; exact reproduction of `FINAL_TEST_REPORT.md` (8/21 correct) |
| **Development Cohort (OOF)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (240 scans, Out-of-Fold) | **53.75%** (43/80, Wilson CI: [42.90%, 64.25%]) | Always-O+ baseline: 36.25% (29/80) | StratifiedGroupKFold cross-validation; checkpoint selection makes this optimistic |
| **Live Pooled (Contaminated) — INVALID** | **Model B: E2 DenseNet121 Ensemble** (deployed, **old mismatched pipeline**) | 101 Participants (pooled: 80 dev + 21 locked test) | **27.72% — INVALID** (obtained with `densenet.preprocess_input` mismatching training; 80 of 101 were development participants) | Always-O+ baseline: 36.63% (37/101) | Fixed pipeline (Round 14): 40/101 = 39.60%; see AUDIT_SUMMARY.md |
| **Live Exploratory** | **Model A: EfficientNetB0** | 101 Participants (avg of 3 scans) | **30.69%** (Wilson 95% CI: [22.54%, 40.26%]) | Always-O+ baseline: 36.63%; Prior-matched chance: 33.54% | Exploratory single-image model averaged over 3 scans |
| **Pseudo-Group Split (ORB)** | Logistic Regression / DenseNet | 898 samples | **Invalid (72.53%)** | Baseline: ~25% | Grouping collapsed with 87.7% false-link rate |

*Note: The proxy metadata artifact classifier achieved 44.58% on a random split, and pseudo-subject grouping reduced the proxy pixel classifier accuracy from 85% to 58.46%, indicating significant dataset leakage in random splits.*

## Remaining Limitations & Audit Notes
1. **Model Distinction:** Model A (EfficientNetB0, 224x224, 86.02%) is not deployed. Model B (E2 DenseNet121 5-fold ensemble, 112x112) is deployed.
2. **Participant Breakdown & Script Provenance:** Of 101 participants, 80 were used in E2 development and 21 formed the locked test cohort (8/21 = 38.10%, equal to always-O+). E2 training script recovered at `scripts/experiment6_fusion_supcon.py` (9 Oct 2026).
3. **Class Order Verification:** E2 deployed class order is `['A+', 'B+', 'AB+', 'O+']` (`src/config.py` line 7, `src/predictor.py` line 40), exactly matching the training script (`scripts/experiment6_fusion_supcon.py` line 35). Model A used `['A+', 'AB+', 'B+', 'O+']` (`scripts/evaluate_abo_positive_4class.py` line 56).
4. **Cross-Domain Generalization:** On the 21 locked test participants, accuracy is 38.10% (matching the 38.10% O+ baseline). The 27.72% figure (Round 12, 101 pooled) was obtained with **mismatched normalisation** (`densenet.preprocess_input` applied to a model trained without it) and is **invalid**. Fixed pipeline (Round 14): 101 pooled = 39.60%, 21 test = 38.10%.
5. **Clinical Validity:** The system is purely a research prototype and cannot be used for clinical diagnosis. 

## Tests Executed and Results
1. **Model Loading & Config:** `test_integration.py` confirmed that the E2 DenseNet121_Fusion_SupCon model initializes successfully.
2. **3-Scan Consensus Inference:** Validated that passing 3 scans properly computes 4-class probabilities summing to 1.
3. **Test Suite Recovery:** 32 failing tests were audited. Outdated API usages were fixed.

## Exact Commands to Launch the Project
1. Open a terminal in the project root directory.
2. Ensure the virtual environment is activated.
3. Start the application: `python app.py`

## Conclusion
The live system does not demonstrate reliable cross-domain four-class blood group classification. Clinical validity is not claimed.
