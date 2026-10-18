# Pre-Registered Independent Evaluation Protocol
## Biometric Blood Group Detection / HemoPrint AI

**Document Type:** Formal Pre-Registered Experimental Protocol  
**Candidate Model Under Test:** `EXP-06C Config A` (5-Fold Ensemble)  
**Protocol Version:** 1.0.0 (Pre-Registered Prior to Protected Data Access)  
**Date of Pre-Registration:** 2026-10-16  
**Auditor & Protocol Author:** Independent Technical & Scientific Auditor  
**Pre-Registration Status:** **LOCKED & PRE-REGISTERED — ZERO REAL TEST ACCESS PERMITTED**  

---

> [!IMPORTANT]
> ### PRE-REGISTRATION COMMITMENT & SCIENTIFIC INTEGRITY
> To eliminate publication bias, $p$-hacking, post-hoc threshold adjustment, and selective metric reporting, this protocol formally locks all evaluation parameters, primary endpoints, statistical tests, decision rules, and rejection criteria **PRIOR** to loading any protected evaluation data.
> 
> Any deviation from the specifications established in this document invalidates the evaluation.

---

## 1. Evaluation Cohorts & Independence Boundaries

Evaluation will be executed sequentially on two distinct, strictly segregated cohorts. **Automatic pooling of the two cohorts is strictly prohibited.**

### 1.1 Primary Evaluation Cohort: Locked 21-Participant Test Set
- **Role:** Primary Benchmark for Internal Held-Out Subject Generalization under identical acquisition conditions.
- **Population:** Exactly 21 human participants (63 scans, Left Thumb protocol).
- **Source Manifest:** `reports/phase3_participant_benchmark/participant_split.csv` (`split == 'TEST'`).
- **Participant IDs:** `P002`, `P0023`, `P0024`, `P0029`, `P003`, `P0035`, `P0039`, `P0041`, `P0042`, `P0047`, `P0048`, `P0054`, `P0071`, `P0073`, `P0075`, `P0082`, `P0083`, `P0088`, `P009`, `P0093`, `P0095`.
- **Class Distribution:** `A+`: 5 (23.8%), `AB+`: 2 (9.5%), `B+`: 6 (28.6%), `O+`: 8 (38.1%).
- **Majority Class (Always-O+) Benchmark:** **38.10%** (8 / 21).
- **Historical Baseline Benchmark:** Phase 3 Model B scored **38.10% Accuracy** (8/21), **0.2500 Macro F1**, **30.62% Balanced Accuracy**.
*[Evidence: Verified from source — `reports/external_evaluation/FOUR_CLASS_TEST_RESULTS.md`]*

### 1.2 Secondary Evaluation Cohort: Separate 26-Participant Hospital Cohort
- **Role:** Secondary Exploratory Benchmark for Domain-Shifted Generalization with Serological Confirmation.
- **Population:** Exactly 26 human participants (78 scans, Left Thumb protocol).
- **Source Manifest:** `data/mfs100_ground_truth.csv`.
- **Participant IDs:** `S001` through `S026`.
- **Class Distribution:** `A+`: 4 (15.4%), `AB+`: 3 (11.5%), `B+`: 8 (30.8%), `O+`: 11 (42.3%).
- **Majority Class (Always-O+) Benchmark:** **42.31%** (11 / 26).
- **Historical Baseline Benchmark:** Baseline E2 scored **19.23% Accuracy** (5/26), **0.1548 Macro F1**, **20.17% Balanced Accuracy**.
*[Evidence: Verified from source — `reports/efficientnet_e2_ensemble_experiment/separate_26_cohort_metrics.json`]*

---

## 2. Frozen Candidate & Inference Mechanism

### 2.1 Model Checkpoint Ensemble
The evaluation will utilize strictly the 5 frozen fold checkpoints of **EXP-06C Config A** (`DenseNet121_Fusion_LinearHead`):
- Fold 1: `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc`
- Fold 2: `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c`
- Fold 3: `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1`
- Fold 4: `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3`
- Fold 5: `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6`
*[Evidence: Verified from source — computed and confirmed in candidate freeze manifest]*

### 2.2 Canonical Class Indexing
```python
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
INDEX_MAPPING = {0: "A+", 1: "AB+", 2: "B+", 3: "O+"}
```

### 2.3 Prediction Aggregation Algorithm
1. For each participant, exactly 3 raw scans are preprocessed via `src.preprocessing.transform_c5` (alpha composite $\rightarrow$ platen crop $\rightarrow$ CLAHE $\rightarrow$ percentile stretch $\rightarrow$ bilateral filter $\rightarrow$ resize $112\times 112$, raw float32 $[0, 255]$).
2. The 3 scans are assembled into a single tensor $\mathbf{X}_i \in \mathbb{R}^{1 \times 3 \times 112 \times 112 \times 3}$.
3. Forward inference is executed on all 5 checkpoints:
   $$\mathbf{P}_{\text{ensemble}}(y=c \mid \mathbf{X}_i) = \frac{1}{5} \sum_{k=1}^5 \operatorname{Softmax}_c(\text{Model}_k(\mathbf{X}_i))$$
4. The predicted label is assigned via strict argmax:
   $$\hat{y}_i = \operatorname{CANONICAL\_CLASSES}[\operatorname{argmax}_c \mathbf{P}_{\text{ensemble}}(y=c \mid \mathbf{X}_i)]$$
*[Evidence: Verified by executed synthetic test — `experiments/final_evaluation/evaluate_frozen_candidate.py`]*

---

## 3. Pre-Registered Endpoints & Metrics

### 3.1 Primary Endpoint: Macro-Averaged F1-Score
- **Formula:**
  $$\text{Macro F1} = \frac{1}{4} \sum_{c \in \{\text{A+}, \text{AB+}, \text{B+}, \text{O+}\}} \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
  *(With zero-division defined as 0.0 when precision + recall = 0).*
- **Primary Threshold:** To claim that the candidate demonstrates non-trivial biometric discriminability on the primary cohort ($N=21$), the model must achieve:
  $$\text{Macro F1}_{\text{Test}} > 0.3500 \quad \text{AND} \quad \text{Macro F1}_{\text{Test}} > \text{Macro F1}_{\text{Phase 3 Baseline}} (0.2500)$$

### 3.2 Secondary Endpoints
1. **Overall Accuracy:** $\frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)}{N}$.
2. **Balanced Accuracy (Macro Recall):** $\frac{1}{4} \sum_{c} \text{Recall}_c$.
3. **Per-Class Metrics:** Support ($N_c$), Precision, Recall, and F1 for each of the 4 canonical classes.
4. **Confusion Matrix:** Formatted strictly with canonical rows (True) and columns (Predicted): `[A+, AB+, B+, O+]`.

---

## 4. Statistical Methods & Uncertainty Estimation

### 4.1 Confidence Interval Estimation (Non-Parametric Bootstrap)
- **Method:** $B = 2,000$ paired participant-level resamples with replacement.
- **Random Seed:** Hardcoded `seed = 42`.
- **Interval Bounds:** 2.5th percentile (lower bound) and 97.5th percentile (upper bound) to establish empirical 95% bootstrap confidence intervals for Macro F1, Balanced Accuracy, and Overall Accuracy.

### 4.2 Hypothesis Testing vs. Historical Baseline
- **Method:** Paired two-sided exact binomial test on discordant participant pairs between EXP-06C Config A and the historical Phase 3 baseline predictions on the 21 test participants.
- **Null Hypothesis ($H_0$):** Both models have equal probability of being correct on discordant pairs ($\pi_{\text{discordant}} = 0.5$).
- **Alternative Hypothesis ($H_1$):** Model correctness differs ($\pi_{\text{discordant}} \neq 0.5$).
- **Significance Level:** $\alpha = 0.05$.

### 4.3 Statistical Power & Sample Size Disclosure

> [!WARNING]
> ### LOW STATISTICAL POWER ON SMALL SAMPLE SIZES
> - On the primary test cohort ($N=21$), each participant represents $1/21 \approx 4.76\%$ of overall accuracy.
> - On the secondary cohort ($N=26$), each participant represents $1/26 \approx 3.85\%$.
> - For rare classes (AB+ $N=2$ in Cohort 1; $N=3$ in Cohort 2), a single prediction shift produces a $\pm 50.0\%$ or $\pm 33.3\%$ shift in recall.
> - A post-hoc power calculation demonstrates that detecting an accuracy shift of $+10.0\%$ (e.g. from $38.1\%$ to $48.1\%$) at $\alpha = 0.05$ with $80\%$ power requires $N \approx 150$ participants.
> - **Scientific Interpretation Standard:** Due to limited power, confidence intervals will be wide. A non-significant $p$-value ($p > 0.05$) indicates **insufficient statistical power to resolve the difference**, NOT proof of equivalence.

---

## 5. Data Handling, Integrity & Invalidation Rules

### 5.1 Input Validation Rules
1. **Three-Scan Completeness:** Exactly 3 scan files must be located for each participant. If any participant has $< 3$ scans, the evaluation halts immediately.
2. **Dimension & Channel Invariance:** All images must load as 8-bit grayscale or RGB with minimum dimensions $112 \times 112$.
3. **No Imputation / No Data Replacement:** Synthetic interpolation, data augmentation, or scan substitution is strictly prohibited.

### 5.2 Criteria for Evaluation Invalidation
The evaluation will be declared **NULL AND VOID** if any of the following occur:
1. Any candidate checkpoint SHA-256 hash fails to match the freeze manifest.
2. The primary and secondary cohort data are merged or pooled.
3. Any hyperparameter, temperature, loss weight, or decision threshold is adjusted after observing test outputs.
4. Any participant from the development cohort ($N=80$) is included in the evaluation sample.
5. Less than 100% of cohort participants are evaluated.

### 5.3 Prohibition on Post-Test Model Selection or Re-Tuning
- **Single-Pass Evaluation:** The evaluation script must run from start to finish without pauses, interactive prompts, or conditional re-runs.
- **No Threshold Calibration:** Softmax probabilities are mapped to classes strictly via $\operatorname{argmax}$. No probability thresholds, temperature scaling, or prior class adjustments may be fitted on test outputs.
- **Finality:** The results generated on the first execution are final and unalterable.

---

## 6. Pre-Registration Certification

- **Candidate Frozen:** `EXP-06C Config A` (5 checkpoints).
- **Primary Cohort Pre-Registered:** 21 locked development test participants.
- **Secondary Cohort Pre-Registered:** 26 hospital participants (unpooled).
- **Execution Script Prepared:** `experiments/final_evaluation/evaluate_frozen_candidate.py`.
- **Status:** **PRE-REGISTRATION COMPLETE — AWAITING EXPLICIT USER AUTHORIZATION TO EXECUTE.**
