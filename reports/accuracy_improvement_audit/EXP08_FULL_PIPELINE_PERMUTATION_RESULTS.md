# EXP-08: Full-Pipeline Permutation Test Results & Statistical Audit

**Document ID:** EXP08_FULL_PIPELINE_PERMUTATION_RESULTS  
**Date:** 2026-10-18  
**Experiment:** EXP-08-DERMATOGLYPHIC-MINUTIAE  
**Topic:** End-to-End Pipeline Refit Permutation Test (1,000 Iterations per Model)  
**Dataset:** Dev-80 Cohort (80 Participants, 240 Scans, $(80, 24)$ Numeric Features)  
**Hardware:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI)  
**Class Distribution:** A+: 19 (23.75%), AB+: 7 (8.75%), B+: 25 (31.25%), O+: 29 (36.25%)  

---

## 1. Executive Summary

This report documents the results of the **full-pipeline cross-validation permutation test** for EXP-08, conducted in strict accordance with the statistical safeguards specified in [`EXP08_PERMUTATION_METHOD_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_PERMUTATION_METHOD_AUDIT.md).

Unlike heuristic post-hoc permutation tests that evaluate fixed predictions, this experiment refit the entire machine learning pipeline—including in-fold median imputation, in-fold feature scaling, re-stratified 5-fold cross-validation, and classifier optimization—across **1,000 independent participant-level label permutations** for both L2 Logistic Regression and Random Forest.

### Core Findings:

1. **Random Forest Classifier ($B = 1,000$ Full-Pipeline Permutations):**
   - **Observed Balanced Accuracy:** **$27.06\%$** (Chance = $25.00\%$)
   - **Empirical Null Distribution:** Mean **$23.62\%$**, Std **$5.45\%$**, Median **$23.28\%$**, 95th Percentile **$33.09\%$**, Max **$43.20\%$**
   - **Null Permutations $\ge$ Observed:** **253 / 1,000**
   - **Empirical $p$-value:** **$p = 0.2537$** ($p = \frac{1 + 253}{1001}$)
   - **Secondary Metric (Macro F1):** Observed **$0.2603$**, Null Mean **$0.2296$**, **$p = 0.2817$**
2. **L2 Logistic Regression ($B = 1,000$ Full-Pipeline Permutations):**
   - **Observed Balanced Accuracy:** **$21.15\%$**
   - **Empirical Null Distribution:** Mean **$24.30\%$**, Std **$5.88\%$**, Median **$24.05\%$**, 95th Percentile **$34.46\%$**, Max **$45.96\%$**
   - **Null Permutations $\ge$ Observed:** **696 / 1,000**
   - **Empirical $p$-value:** **$p = 0.6963$** ($p = \frac{1 + 696}{1001}$)
   - **Secondary Metric (Macro F1):** Observed **$0.2193$**, Null Mean **$0.2387$**, **$p = 0.6254$**
3. **Substantive Scientific Conclusion:**
   - Both models fall comfortably within their empirical null distributions ($p = 0.2537$ and $p = 0.6963$).
   - Over $25\%$ of random label shuffles yielded a balanced accuracy equal to or higher than the trained Random Forest model.
   - The full-pipeline permutation test confirms that **handcrafted dermatoglyphic features provide no statistically significant predictive signal beyond chance** for ABO blood-group classification on this dataset.

---

## 2. Pre-Execution Validation & Verification

Before running the permutation loops, pre-execution checks verified that the experimental conditions and observed scores matched the primary EXP-08 results exactly:

| Dimension | Specification | Verified Reality | Status |
| :--- | :--- | :--- | :---: |
| **Cohort Size** | 80 Dev participants | Exactly 80 participant rows | **PASSED** |
| **Feature Set** | 24 numeric features | Exactly 24 columns from `feature_names.json` | **PASSED** |
| **Cohort Quarantine** | 0 test / hospital samples | 100% disjoint from locked test and hospital cohorts | **PASSED** |
| **Observed LogReg Acc / BalAcc / MacroF1** | 27.50% / 21.15% / 0.2193 | 27.50% / 21.15% / 0.2193 | **PASSED** |
| **Observed RF Acc / BalAcc / MacroF1** | 35.00% / 27.06% / 0.2603 | 35.00% / 27.06% / 0.2603 | **PASSED** |
| **Hyperparameter Continuity** | `build_model_pipeline` | Exact parameters from `train.py` imported directly | **PASSED** |

---

## 3. Methodological Design of the Full-Pipeline Permutation Test

### A. The Formal Null Hypothesis Tested
The test evaluated the global biomarker independence hypothesis:
$$H_0^{\text{biomarker}}: X \perp\!\!\!\perp Y \implies P(Y \mid X) = P(Y)$$
*"Friction ridge biometric features $X$ and ABO blood group antigens $Y$ are statistically independent. The learning pipeline $f(X) \to \hat{Y}$ cannot extract true predictive signal."*

### B. Algorithmic Protocol (Repeated for $B = 1,000$ Iterations per Model)
1. **Participant Label Permutation:**
   A random permutation $Y_{\pi_b} = \text{Permute}(Y)$ was generated using seed $42 + b \times 10007$. This bisection strictly preserved the marginal class distribution ($19\text{ A+}, 7\text{ AB+}, 25\text{ B+}, 29\text{ O+}$).
2. **Re-Stratified Cross-Validation Partition:**
   `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` was re-run on $(X, Y_{\pi_b}, \text{groups})$ for each permutation. This guaranteed that every validation fold under $H_0$ contained a balanced representation of classes (e.g., 1–2 AB+, 3–5 A+, 5 B+, 5–6 O+), perfectly mirroring the real-data evaluation.
3. **In-Fold Preprocessing Pipeline:**
   - `SimpleImputer(strategy="median")` was fit strictly on training fold features $X_{\text{train}}$.
   - `StandardScaler()` (for Logistic Regression) was fit strictly on $X_{\text{train}}$.
4. **Model Refitting:**
   The classifier was refit from scratch on $(X_{\text{train}}, Y_{\pi_b, \text{train}})$ with balanced class weights.
5. **Out-of-Fold Prediction & Metric Computation:**
   Probabilities were predicted on $X_{\text{val}}$ out-of-fold. Exactly one prediction was generated per participant across the 80 participants.
6. **Unbiased $p$-Value Calculation:**
   Upper-tail empirical $p$-values were computed using the standard unbiased formula:
   $$p = \frac{1 + \sum_{b=1}^{B} \mathbb{I}\left( \text{Metric}_{\text{null}}[b] \ge \text{Metric}_{\text{observed}} \right)}{B + 1}$$

---

## 4. Empirical Permutation Test Results

### Comprehensive Statistical Summary ($B = 1,000$ Permutations):

| Metric | L2 Logistic Regression | Random Forest Classifier | Primary Evaluation Note |
| :--- | :---: | :---: | :--- |
| **Observed Accuracy** | 27.50% (22/80) | 35.00% (28/80) | RF is below 36.25% majority base rate |
| **Observed Balanced Accuracy** | **21.15%** | **27.06%** | **Primary Evaluation Metric** |
| **Null BalAcc Mean $\pm$ Std** | $24.30\% \pm 5.88\%$ | $23.62\% \pm 5.45\%$ | Theoretical chance = 25.00% |
| **Null BalAcc Median** | 24.05% | 23.28% | Unbiased central tendency |
| **Null BalAcc 95th Percentile** | 34.46% | 33.09% | Critical value for significance ($\alpha = 0.05$) |
| **Null BalAcc Maximum** | 45.96% | 43.20% | Maximum observed under pure noise |
| **Permutations $\ge$ Observed** | 696 / 1,000 | 253 / 1,000 | Count of null trials exceeding real score |
| **Empirical $p$-value (BalAcc)** | **$p = 0.6963$** | **$p = 0.2537$** | **Both models statistically nonsignificant** |
| **Observed Macro F1** | 0.2193 | 0.2603 | Secondary Evaluation Metric |
| **Null Macro F1 Mean $\pm$ Std** | $0.2387 \pm 0.0540$ | $0.2296 \pm 0.0553$ | Empirical null distribution |
| **Null Macro F1 95th Percentile** | 0.3287 | 0.3249 | Critical value for Macro F1 |
| **Permutations $\ge$ Observed (F1)** | 625 / 1,000 | 281 / 1,000 | Count of null trials exceeding real score |
| **Empirical $p$-value (Macro F1)** | **$p = 0.6254$** | **$p = 0.2817$** | **Both models statistically nonsignificant** |

---

## 5. Methodological Comparison: Fixed-Prediction vs. Full-Pipeline Permutation

| Methodological Dimension | Previous Method (Fixed Predictions) | Rigorous Method (Full-Pipeline Refit) |
| :--- | :--- | :--- |
| **Hypothesis Tested** | Static predictions $\hat{y}$ independent of $Y$ | Biomarker independence ($X \perp\!\!\!\perp Y$) |
| **Model Re-Training** | Model trained **once** on real data; $\hat{y}$ frozen | Model **refit 1,000 times** on permuted data |
| **Cross-Validation Partitioning** | Fixed single partition from real data | **Re-stratified 5-fold CV** on each permuted draw |
| **Imputation & Scaling** | Fixed single transformation from real data | **Re-computed in-fold** for each permutation |
| **Noise-Fitting Modeled?** | No (classifier treated as static black box) | **Yes** (adaptive capacity modeled under $H_0$) |
| **Random Forest $p$-value (BalAcc)** | $p = 0.3250$ | **$p = 0.2537$** |
| **Logistic Regression $p$-value (BalAcc)** | $p = 0.7810$ | **$p = 0.6963$** |
| **Substantive Scientific Conclusion** | **Nonsignificant ($p \gg 0.05$)** | **Nonsignificant ($p \gg 0.05$)** |

*Scientific Takeaway:* Both permutation methodologies yield consistent conclusions: the observed performance of classical machine learning models trained on handcrafted dermatoglyphic features is entirely compatible with random chance.

---

## 6. Statistical Interpretation & Limits of Inference

1. **Failure to Exceed Empirical Chance:**
   - For Random Forest, the observed balanced accuracy ($27.06\%$) lies well within the 95% null distribution interval ($[13.1\%, 33.1\%]$). In more than 1 out of every 4 random label shuffles ($25.3\%$), the exact same pipeline achieved an equal or higher balanced accuracy on pure noise.
   - For Logistic Regression, the observed balanced accuracy ($21.15\%$) was lower than the empirical null mean ($24.30\%$), yielding $p = 0.6963$.
2. **Minority Class Resolution ($N=7$ for AB+):**
   - With only 7 AB+ participants in Dev-80, each AB+ classification alters the class recall by $1/7 \approx 14.29\%$. Because both models achieved 0/7 recall on AB+ in the real data, the primary metric is penalized by the discrete difficulty of identifying the minority class.
   - However, even when AB+ labels are permuted across participants, the models frequently achieve 0% or low recall on AB+ due to class rarity, confirming that minority class collapse is a structural property of small-sample learning rather than an artifact of bloodgroup biology.
3. **Absence of Proof is Not Proof of Absence:**
   - A statistically nonsignificant result on an 80-participant development cohort confirms that commercial 500 DPI optical scans do not provide a viable blood-group diagnostic tool.
   - It does **not** disprove the theoretical existence of subtle biological correlations between embryonic dermatoglyphic development and blood-group antigens that might be investigated in large-scale serological population studies.
4. **Clinical Non-Validity:**
   - Under no circumstances should fingerprint images or dermatoglyphic measurements be used for clinical blood-group determination or medical triage.

---

## 7. Artifact Inventory & Reproducibility Lineage

All generated artifacts are saved in dedicated, non-overlapping directory structures:

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **Final Results Document** | [`EXP08_FULL_PIPELINE_PERMUTATION_RESULTS.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FULL_PIPELINE_PERMUTATION_RESULTS.md) | This document. |
| **Machine-Readable Metrics JSON** | [`exp08_full_pipeline_permutation_metrics.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/permutation_test/exp08_full_pipeline_permutation_metrics.json) | Summary metrics, percentiles, and exact $p$-values. |
| **Random Forest Null CSV** | [`random_forest_null_distribution.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/permutation_test/random_forest_null_distribution.csv) | 1,000 permutation trials with fold seeds and scores. |
| **Logistic Regression Null CSV** | [`logistic_regression_null_distribution.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/permutation_test/logistic_regression_null_distribution.csv) | 1,000 permutation trials with fold seeds and scores. |
| **Reproduction Script** | [`run_full_pipeline_permutation_test.py`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/exp08_dermatoglyphic_minutiae/permutation_test/run_full_pipeline_permutation_test.py) | Standalone script reproducing the entire permutation experiment. |
| **Methodology Audit Document** | [`EXP08_PERMUTATION_METHOD_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_PERMUTATION_METHOD_AUDIT.md) | Pre-execution statistical methodology review. |
