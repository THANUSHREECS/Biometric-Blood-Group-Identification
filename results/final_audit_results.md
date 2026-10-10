# HemoPrint AI: Final Audit Results Table

| Evaluation Setting | Model / Architecture | Samples / Scope | Metric / Accuracy | Baseline & Chance | Status & Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Dataset)** | **Model A: EfficientNetB0** (not deployed) | 415 test images (random image split) | **86.02%** (95% CI: [82.65%, 88.92%]) | Majority baseline: 30.6% | Standalone single-image model; not deployed in production pipeline |
| **Dataset Benchmark** | **Model B: E2 DenseNet121 Ensemble** (deployed) | Dataset test set | **Not reliably measured** | Chance: ~25.0% | Multi-scan 5-fold ensemble; training script recovered at `scripts/experiment6_fusion_supcon.py` (9 Oct 2026) |
| **Locked External Test** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 21 Locked Test Participants (63 scans, 3 per person) | **38.10%** (8/21, Wilson CI: [20.75%, 59.12%]) | Always-O+ baseline: 38.10% (8/21, exact binomial p=0.5817) | True locked test set; exact reproduction of `FINAL_TEST_REPORT.md` (8/21 correct) |
| **Development Cohort (OOF)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (240 scans, Out-of-Fold) | **53.75%** (43/80, Wilson CI: [42.90%, 64.25%]) | Always-O+ baseline: 36.25% (29/80) | StratifiedGroupKFold cross-validation; checkpoint selection makes this optimistic |
| **Live Pooled (Contaminated)** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 101 Participants (pooled: 80 dev + 21 locked test) | **27.72%** (Wilson 95% CI: [19.93%, 37.15%]) | Always-O+ baseline: 36.63% (37/101); Prior-matched chance: 24.79% | Contaminated evaluation: includes 80 participants used in E2 training/selection |
| **Live Exploratory** | **Model A: EfficientNetB0** | 101 Participants (avg of 3 scans) | **30.69%** (Wilson 95% CI: [22.54%, 40.26%]) | Always-O+ baseline: 36.63%; Prior-matched chance: 33.54% | Exploratory single-image model averaged over 3 scans |
| **Pseudo-Group Split (ORB)** | Logistic Regression / DenseNet | 898 samples | **Invalid (72.53%)** | Baseline: ~25% | Pseudo-grouping collapsed with 87.7% false-link rate |

### Additional Artifact Tests (Random Split, 4-Class)
*   **(a) Border/corner-pixel stats only**: 20.82% (95% CI: [18.15%, 23.50%])
*   **(b) BMP header + file size + timestamp only**: 16.93% (95% CI: [14.36%, 19.38%])
*   **(c) Ridge-area stats only**: 19.04% (95% CI: [16.59%, 21.60%])

### Audit Findings & Model Provenance
1.  **Model Separation:** Model A (`models/abo_positive_4class/best_model.keras`, EfficientNetB0, 224x224) achieved 86.02% on the 415-image random split, but is not deployed. Model B (`src/predictor.py`, 5-fold DenseNet121 Siamese ensemble, 112x112) is the deployed runtime system.
2.  **Participant Breakdown & Training Provenance:** Of the 101 participants in `data/mfs100_final`, 80 were used in E2 development (training and fold selection), and 21 formed the locked test cohort. The E2 training script has been recovered at `scripts/experiment6_fusion_supcon.py` (originally `scratch/experiment6_fusion_supcon.py`, 9 Oct 2026).
3.  **Origin of 38.10%:** The 38.10% accuracy metric previously cited is exactly 8/21 correct on the 21 locked-test participants (`reports/phase3_final_evaluation/FINAL_TEST_REPORT.md`), equal to the always-O+ majority baseline (8/21 = 38.10%).
4.  **Class Order Verification:** E2 deployed class order is `['A+', 'B+', 'AB+', 'O+']` (`src/config.py` line 7, `src/predictor.py` line 40), exactly matching the training script (`scripts/experiment6_fusion_supcon.py` line 35). Model A used `['A+', 'AB+', 'B+', 'O+']` (`scripts/evaluate_abo_positive_4class.py` line 56).
5.  **Biological Plausibility:** There is no established physiological mechanism linking epidermal ridge macroscopic patterns to ABO blood groups. Our empirical data on unseen participants show no generalizable signal beyond class baselines.
