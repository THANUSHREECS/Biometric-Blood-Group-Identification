# HemoPrint AI: Final Audit Results Table
*(Updated Round 15 — branch fix-preprocessing-mismatch)*

## Summary Table

| Evaluation Setting | Model / Architecture | Samples / Scope | Metric / Accuracy | Baseline & Chance | Status & Audit Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Split (Dataset)** | **Model A: EfficientNetB0** (not deployed) | 415 test images (random image split, public fingerprint dataset) | **86.02%** (95% CI: [82.65%, 88.92%]) | Majority baseline: 30.6% | Standalone single-image model; **not deployed**. Subject independence not verifiable (no subject IDs). Metadata artifact classifiers achieve 44.58% (random split), indicating dataset leakage. |
| **Dataset Benchmark** | **Model B: E2 DenseNet121 Ensemble** (deployed) | Dataset test set (public fingerprint dataset) | **Not reliably measured** | Chance: ~25.0% | Multi-scan 5-fold ensemble; training script at scratch/experiment6_fusion_supcon.py. Not evaluated on this dataset in valid participant-split manner. |
| **21 Locked Test Participants** | **Model B: E2 DenseNet121 Ensemble** (deployed) | 21 Locked Test Participants (63 scans, 3 per person) | **38.10%** (8/21; Wilson 95% CI: [20.75%, 59.12%]) | Always-O+ baseline: 38.10% (8/21); binomial p = 0.5817 | **Valid**: True locked test, exact reproduction of FINAL_TEST_REPORT.md. Equal to always-O+ baseline. Pred dist: A+=4, B+=1, AB+=0, O+=16. |
| **21 Locked Test — post-fix deployed path** | **Model B: E2 DenseNet121 Ensemble** (deployed, fixed pipeline) | 21 Locked Test Participants | **38.10%** (8/21) | Always-O+ baseline: 8/21 = 38.10%; binomial p = 1.0000 | **Valid**: Fixed pipeline (no preprocess_input). Pred dist: A+=11, B+=2, AB+=0, O+=8. Prob diff vs experiment pipeline: **0.0000000000** (float32). |
| **80 Development Cohort (OOF)** | **Model B: E2 DenseNet121 Ensemble** | 80 Development Participants (240 scans, Out-of-Fold via StratifiedGroupKFold) | **53.75%** (43/80; Wilson CI: [42.90%, 64.25%]) | Always-O+ baseline: 36.25% (29/80) | **Optimistic** (OOF from training script only). Not an external test. |
| **101 Pooled — INVALID** | **Model B: E2 DenseNet121 Ensemble** (old mismatched pipeline) | 101 Participants (pooled: 80 dev + 21 locked test) | **27.72% — INVALID** | Always-O+ baseline: 36.63% (37/101) | **Invalid**: obtained with densenet.preprocess_input applied to a model trained without it (normalisation mismatch). Also contaminated: 80 of 101 were development participants. See AUDIT_SUMMARY.md Round 14. |
| **101 Pooled — post-fix deployed path** | **Model B: E2 DenseNet121 Ensemble** (fixed pipeline) | 101 Participants (80 dev + 21 locked test, contaminated) | **39.60%** (40/101) | Always-O+ baseline: 36.63% (37/101); binomial p = 0.5369 | **Contaminated** (80 dev participants not independent). Fixed pipeline; pixel-identical to training pipeline. |
| **Live Exploratory** | **Model A: EfficientNetB0** | 101 Participants (avg of 3 scans) | **30.69%** (Wilson 95% CI: [22.54%, 40.26%]) | Always-O+ baseline: 36.63%; Prior-matched chance: 33.54% | Exploratory; Model A not deployed. |
| **Pseudo-Group Split (ORB)** | Logistic Regression / DenseNet | 898 samples | **Invalid (72.53%)** | Baseline: ~25% | Pseudo-grouping collapsed with 87.7% false-link rate. |

---

### Additional Artifact Tests (Random Split, 4-Class, Model A public dataset)

- **(a) Border/corner-pixel stats only**: 20.82% (95% CI: [18.15%, 23.50%])
- **(b) BMP header + file size + timestamp only**: 16.93% (95% CI: [14.36%, 19.38%])
- **(c) Ridge-area stats only**: 19.04% (95% CI: [16.59%, 21.60%])

---

### Audit Findings & Model Provenance

1. **Model Separation:** Model A (EfficientNetB0, 224x224) achieved 86.02% on the 415-image random split, but is **not deployed**. Model B (5-fold DenseNet121 Siamese ensemble, 112x112) is the deployed runtime system.
2. **Participant Breakdown & Training Provenance:** Of the 101 participants, 80 were used in E2 development and 21 formed the locked test cohort. The E2 training script is at scratch/experiment6_fusion_supcon.py.
3. **Origin of 38.10%:** 8/21 correct on 21 locked-test participants (FINAL_TEST_REPORT.md), equal to the always-O+ majority baseline.
4. **27.72% is invalid:** Obtained in Round 12 using densenet.preprocess_input on a model trained without it. Fixed pipeline gives 39.60% for the same 101 participants (contaminated).
5. **Post-fix pipeline equality (Round 15):** scripts/round15_step1_pipeline_equality.py confirms 0.0000000000 max abs probability diff and 101/101 (100%) top-1 match between experiment pipeline and deployed pipeline, across all 303 scans (float32, no rounding).
6. **Class Order Verification:** E2 deployed class order is [A+, B+, AB+, O+] (src/config.py:7, src/predictor.py:40), exactly matching the training script (scratch/experiment6_fusion_supcon.py:35). Model A used [A+, AB+, B+, O+].
7. **Biological Plausibility:** There is no established physiological mechanism linking epidermal ridge macroscopic patterns to ABO blood groups. Empirical data on 21 unseen participants show no signal above the majority-class baseline.
