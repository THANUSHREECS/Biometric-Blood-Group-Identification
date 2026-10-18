# High-Priority Next Experiment Proposal (EXP-07)

**Title:** Resolution Scaling and Preprocessing Fidelity Benchmark under Leakage-Free Nested Cross-Validation  
**Target Identifier:** EXP-07-RES-FIDELITY  
**Audit Date:** 2026-10-17  
**Status:** Preregistered Proposal (Strictly no execution permitted during audit phase)

---

## 1. Research Question
Does preserving optical sensor spatial fidelity (scaling input resolution from $112 \times 112$ to native/near-native $224 \times 224$ and eliminating destructive CLAHE/bilateral filtering) yield a statistically significant improvement in out-of-fold ABO bloodgroup classification accuracy when evaluated under a strictly leakage-free validation protocol?

---

## 2. Hypothesis
- **Null Hypothesis ($H_0$):** Friction ridge features at $224 \times 224$ resolution with raw intensity normalization do not improve out-of-fold classification performance over random permutation chance ($p \ge 0.05$ under a 1,000-run permutation test).
- **Alternative Hypothesis ($H_1$):** Because human fingerprint ridge period at 500 DPI requires $>5\text{ pixels}$ to avoid Nyquist aliasing, preserving ridge-valley geometry at $224 \times 224$ without smoothing will produce a non-trivial classification signal exceeding the majority-class baseline and the empirical noise ceiling.

---

## 3. Dataset Required
- **Cohort:** Dev-80 cohort of the Primary MFS100 dataset (80 participants, 240 scans).
- **Absolute Exclusion:** The 21-participant locked test cohort is **strictly quarantined and excluded**. Under no circumstances will the 21 locked participants be accessed, loaded, or scored during this experiment.
- **Quarantine Notice:** The 26-participant hospital cohort remains quarantined pending identity audit and will not be accessed.

---

## 4. Eligibility and Exclusion Criteria
- **Inclusion Criteria:** All 80 participants in the established Dev-80 split with exactly three readable right-thumb scans from the Mantra MFS100 scanner.
- **Exclusion Criteria:** Any scan with severe saturation (>95% pure black or white pixels) or corrupted image headers (already validated clean).

---

## 5. Participant-Level Split Strategy
- **Outer Split:** 5-fold Stratified Group K-Fold grouped strictly by `participant_id`.
  - In each outer fold: 64 participants for training/inner-validation, 16 participants strictly reserved for out-of-fold scoring.
  - Class stratification maintained across all folds: A+, B+, AB+, O+.
- **Zero Identity Overlap:** Scans from the same participant are guaranteed to never appear in both training and outer validation.

---

## 6. Training & Validation Protocol
- **Architecture:** Identical to EXP-06C Config A baseline to isolate resolution and preprocessing as the sole independent variables:
  - Input: $(1, 3, 224, 224, 3)$ vs baseline $(1, 3, 112, 112, 3)$.
  - Backbone: DenseNet-121 (pretrained ImageNet, frozen weights).
  - Three-scan feature extraction followed by concatenation ($3 \times 1024 = 3072$-d embedding).
  - Classification Head: Linear projection ($3072 \to 4$ logits) with $L_2$ weight decay ($\lambda = 10^{-4}$).
  - Optimizer: AdamW, initial learning rate $10^{-3}$, cosine annealing to $10^{-6}$.

---

## 7. Checkpoint-Selection Method (Leakage-Free)
To permanently eliminate the selection leakage identified in the audit, two independent evaluation regimes will be run:
1. **Regime A (Fixed Duration, No Checkpointing):**
   - Each fold trains for exactly **80 epochs**.
   - No callback tracks outer-validation metrics.
   - The final model at Epoch 80 is evaluated exactly once on the held-out 16 participants.
2. **Regime B (Nested Inner-Validation):**
   - The 64 training participants in each fold are subdivided into an inner-train set (48 participants) and an inner-validation set (16 participants).
   - Checkpoints are saved based on maximum Macro F1 on the **inner-validation set only**.
   - The selected checkpoint is evaluated exactly once on the outer 16 participants.

---

## 8. Baselines for Comparison
1. **Historical EXP-06C Config A Baseline ($112 \times 112$, C5 Preprocessing):** Unbiased fixed-duration score is 36.25% (OOF Macro F1: 0.3278).
2. **Majority-Class Baseline:** Predicting the most frequent class (O+) across Dev-80 yields $33.75\%$ (27/80).
3. **Random Guessing Baseline:** $25.00\%$.
4. **Empirical Permutation Noise Baseline:** Mean performance obtained by permuting labels across 100 runs under the identical protocol.

---

## 9. Metrics
- Primary Metric: Out-of-Fold (OOF) Balanced Accuracy (to account for class imbalance).
- Secondary Metrics: OOF Macro F1-score, Multi-class Cohen's Kappa ($\kappa$), and Expected Calibration Error (ECE).
- Per-class precision, recall, and confusion matrix.

---

## 10. Statistical Uncertainty & Power Analysis
- Total OOF sample size: $N = 80$ participants.
- A score of 45.0% (36/80 correct) on $N=80$ yields a 95% Wilson score confidence interval of $[34.5\%, 56.0\%]$.
- To achieve statistical significance against the majority baseline (33.75%) at $\alpha = 0.05$ with $N=80$, the model must achieve at least **$46.25\%$ accuracy (37/80 correct)**.
- Permutation test: 1,000 label permutations will be computed to derive an exact, non-parametric $p$-value for both Balanced Accuracy and Macro F1.

---

## 11. Leakage Safeguards
1. **Automated Assertion Guards:** The script will programmatically assert that zero test participant IDs (from the 21-person locked cohort) appear in any training, validation, or inference dataset.
2. **Zero Outer-Metric Visibility:** The training loop will be completely forbidden from passing `val_dataset` to the Keras `.fit()` method during Regime A, preventing any gradient or callback exposure.
3. **Deterministic Seed:** All fold splits and data shuffling will use fixed random seed `20261017`.

---

## 12. Stopping Conditions
- Fixed training duration of 80 epochs per fold (Regime A) or 100 inner-epochs (Regime B).
- Total training will terminate automatically upon completion of the 5 outer folds.

---

## 13. Expected Runtime & Resource Requirements
- **Hardware:** Single NVIDIA RTX GPU (or Apple Silicon / Intel Core i7 with CUDA/Metal support).
- **Memory Footprint:** Peak VRAM $\approx 3.2\text{ GB}$ at batch size 8 for $(3, 224, 224, 3)$ inputs.
- **Wall-Clock Runtime:** Approximately $18\text{ to }25\text{ minutes}$ for complete 5-fold cross-validation.

---

## 14. Outcomes & Decision Criteria
- **Outcome A (Hypothesis Supported):** OOF Balanced Accuracy $> 46.25\%$ and permutation test $p < 0.05$.
  - *Interpretation:* Preserving friction ridge spatial resolution provides actionable signal. Resolution scaling is validated as a genuine performance driver.
- **Outcome B (Hypothesis Rejected):** OOF Balanced Accuracy $\le 36.25\%$ and permutation test $p > 0.05$.
  - *Interpretation:* Increasing optical resolution and removing filtering does not improve bloodgroup prediction. Strongly indicates that DenseNet-121 feature space on optical scans does not contain measurable ABO phenotypes.

---

## 15. Absolute Scientific Limitations
Regardless of outcome:
1. This experiment **cannot** prove clinical efficacy or medical safety.
2. This experiment **cannot** rule out self-reported label noise in the Dev-80 cohort.
3. This experiment **cannot** be declared an unbiased real-world test, because an independent, unexposed external test cohort does not currently exist.
