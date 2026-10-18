# Executive Audit Summary & Research Roadmap

**Audit Date:** 2026-10-17  
**Project:** Biometric Bloodgroup Detection Using Fingerprint Images  
**Hardware:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI)  
**Target Classes:** A+, AB+, B+, O+

---

## 1. What We Know with Scientific Certainty

1. **The Historical 86% Accuracy Was An Artifact of Split Leakage and Shortcuts:**
   - In the legacy 6,000-image dataset, impressions from the same physical fingers were distributed across both train and test splits due to lack of participant identifiers. The models learned individual finger identity, not blood groups.
   - Furthermore, 42 trivial non-ridge global image statistics (background brightness, alpha transparency channels, border padding) could predict bloodgroup labels with 51% balanced accuracy without inspecting fingerprint ridges.
2. **The 53.75% Development Score Was an Illusion of Checkpoint Selection Leakage:**
   - During EXP-03 through EXP-06C, the best model checkpoints were selected by maximizing Macro F1 directly on the outer 16-participant held-out validation folds across 100 epochs.
   - Permuting labels to pure random noise produced a mean validation accuracy of 46.77% simply through callback selection on stochastic noise.
   - When honest, leakage-free evaluation protocols are applied (nested inner validation or fixed training durations), the true development accuracy of EXP-06C drops to 31.67%–36.25%.
3. **The Final Candidate Failed on the Locked Cohort:**
   - On the 21-participant locked test cohort, EXP-06C Config A achieved **33.33% accuracy (7/21 correct)** and a **Macro F1 of 0.2083**.
   - Predicting the majority class (O+) for all participants yields **38.10% (8/21)**. The model performed below zero-intelligence majority guessing.
   - A 1,000-run permutation test yielded $p = 0.5641$ (accuracy) and $p = 0.3712$ (Macro F1), establishing that test performance is indistinguishable from random chance.
4. **The Locked Cohort Is An Exposed Holdout:**
   - The 21-participant test cohort was evaluated 17 times across 10 distinct evaluation dates between October 7 and October 13, 2026. The architectural choice of DenseNet-121 and three-scan concatenation was influenced by seeing early test scores. It is not an untouched, pristine test cohort.
5. **Preprocessing Was Severely Destructive:**
   - Downsampling 500 DPI scans to $112 \times 112$ pixels compressed the human friction ridge period down to $\approx 2.67\text{ pixels}$, causing spatial aliasing.
   - EXP-06D proved that removing CLAHE and bilateral filtering improved out-of-fold Macro F1 by $+0.0622$, confirming that aggressive image enhancements destroyed minute biometric features.

---

## 2. What Remains Uncertain (Open Hypotheses)

1. **Does A Non-Trivial Dermatoglyphic Bloodgroup Phenotype Actually Exist in Optical Scans?**
   - While medical literature documents minor statistical differences in population-level ridge densities and pattern distributions across ABO groups, it has never been established that an optical fingerprint scan contains sufficient mutual information to determine an individual's blood type.
2. **Impact of Self-Reported Label Noise:**
   - All 101 participants in the primary MFS100 cohort have self-reported labels. Because self-reported blood typing carries an estimated 10%–25% clinical error rate, we cannot ascertain how much of the model's failure stems from noisy labels versus an absence of biological signal.
3. **Impact of Native Resolution:**
   - Whether preserving native $224 \times 224$ or $316 \times 354$ resolution under a strict, leakage-free validation protocol can extract genuine signal above majority chance remains an untested hypothesis.

---

## 3. What Should Happen Next

1. **Do NOT Immediately Train or Sweeps Models:**
   - Do not run random architectures or deep neural network sweeps without resolving data fidelity and evaluation protocols.
2. **Execute EXP-07 as a Controlled Benchmark:**
   - Run the preregistered EXP-07 experiment on the Dev-80 cohort comparing $112 \times 112$ against $224 \times 224$ under strict nested inner-validation (or fixed 80 epochs).
   - If EXP-07 fails to exceed majority chance ($p > 0.05$), it will provide conclusive empirical evidence that deep learning on optical scans cannot predict ABO blood groups with the current dataset.
3. **Audit the Quarantined Hospital Cohort:**
   - Conduct an administrative and consent review to determine whether cross-cohort participant identity overlap can be ruled out between the 26 hospital participants and the MFS100 cohort.
   - If confirmed clean, this cohort should serve as a true, untouched, serologically verified blind test set for any future candidate.
4. **Establish Honest Academic Framing for the Final Report:**
   - Frame the engineering project as a rigorous scientific investigation into biometric feasibility and dataset shortcuts.
   - Highlight the discovery and elimination of checkpoint-selection leakage, image downsampling aliasing, and split contamination as major, commendable engineering contributions.
