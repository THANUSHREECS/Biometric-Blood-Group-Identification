# Final Pre-Evaluation Protocol Audit
## Biometric Blood Group Detection / HemoPrint AI

**Document Type:** Formal Read-Only Pre-Evaluation Protocol Audit  
**Candidate Model Under Audit:** `EXP-06C Config A` (DenseNet-121 3-Scan Concat + Linear Projection Head)  
**Target Evaluation Cohort:** Primary Locked 21-Participant Test Cohort ($N=21$, 63 scans)  
**Audit Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Audit Type:** Strict Read-Only Audit — **Zero Real-Cohort Evaluation Authorized or Executed**  
**Execution Guards:** `TRAINING_AUTHORIZED = False`, `EVALUATION_AUTHORIZED = False`  

---

> [!IMPORTANT]
> ### READ-ONLY AUDIT DECLARATION
> This audit was conducted exclusively through static code analysis, cryptographic hashing, and documentation verification.
> 
> - **Zero protected fingerprint images were opened or read.**
> - **Zero protected ground-truth labels were loaded into memory.**
> - **Zero real-cohort inference or predictions were executed.**
> - **Both `TRAINING_AUTHORIZED` and `EVALUATION_AUTHORIZED` remain strictly False.**

---

## 1. Audit Classification Matrix

Every requirement from the audit charter was evaluated and classified as **PASS**, **FAIL**, or **NEEDS REVIEW**, backed by physical file paths and source evidence:

| Requirement / Item | Audit Criteria | Audit Classification | Evidence Summary & Source Paths |
| :--- | :--- | :---: | :--- |
| **1. Source & Protocol Inspection** | Evaluator code and written protocol inspected | **PASS** | Read-only inspection of [evaluate_frozen_candidate.py](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/experiments/final_evaluation/evaluate_frozen_candidate.py) and [preregistered_evaluation_protocol.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/preregistered_evaluation_protocol.md). |
| **2. Candidate & Checkpoint Parity** | Candidate is EXP-06C Config A; 5 checkpoints match freeze manifest | **PASS** | Confirmed: All 5 checkpoint paths and byte sizes in `FROZEN_CHECKPOINT_LEDGER` match [candidate_freeze_manifest.md](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/final_evaluation/candidate_freeze_manifest.md). |
| **3. Canonical Class Order** | Fixed canonical order `[A+, AB+, B+, O+]` enforced | **PASS** | Hardcoded in `evaluate_frozen_candidate.py:44` (`CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]`) and protocol Section 2.2. |
| **4. Architectural & Aggregation Parity** | 3-scan input, `transform_c5` preprocessing, frozen backbone, 5-fold soft voting match experiment design | **PASS** | Input shape `(3, 112, 112, 3)`, raw float32 $[0, 255]$ via `src.preprocessing.transform_c5`, 3072-d concat, and soft-voting probability averaging verified in `evaluate_frozen_candidate.py:112-192`. |
| **5. Primary & Secondary Metrics** | Primary metric is Macro F1; secondary metrics include Acc, Balanced Acc, per-class P/R/F1, CM, bootstrap CI | **PASS** | Verified in `evaluate_frozen_candidate.py:202-230, 290-316` and protocol Section 3. |
| **6. Prohibition on Post-Test Tuning** | Single-pass evaluation only; zero threshold calibration, zero model selection, zero retraining | **PASS** | Pre-registered in protocol Section 5.3; evaluator contains zero threshold adjustment parameters or retuning loops. |
| **7a. Primary Cohort Size** | Accurately documents $N=21$ participants (63 scans) | **PASS** | Documented in `cohort_provenance_review.md:46-55` and `preregistered_evaluation_protocol.md:32-41`. |
| **7b. Serology Verification Limitation** | Discloses that primary cohort blood groups lack physical laboratory serology sheets | **PASS** | Documented as **NOT VERIFIED** for clinical laboratory serology in `cohort_provenance_review.md:58-65` and protocol Section 1.1. |
| **7c. Historical Model Exposure** | Documents that Phase 3 Model B was previously evaluated on this cohort | **PASS** | Documented in `cohort_provenance_review.md:73-82`: Model B scored $38.10\%$ accuracy, $0.2500$ Macro F1. |
| **7d. Candidate Prior Exposure** | EXP-06C Config A exposure documented based on logs, not assumptions | **PASS** | Task logs (`task-1475`, `task-1482`) and `evaluation_execution_history_audit.md` prove candidate has **ZERO** prior evaluation exposure. |
| **7e. Prohibition on Cohort Pooling** | Separate 26-person hospital cohort must NOT be pooled with primary cohort | **PASS** | Explicitly pre-registered as separate evaluation in protocol Section 1.2 and `cohort_provenance_review.md:120-132`. |
| **7f. Cross-Cohort Overlap Unverified** | Cross-cohort subject identity overlap documented as unverified | **PASS** | Documented as **NOT VERIFIED** in `cohort_provenance_review.md:67-71, 107-111` (biometric minutiae de-duplication across splits unperformed). |
| **7g. Clinical Validity Disclaimer** | Explicitly documents that research is not clinically validated | **PASS** | Prominent non-clinical disclaimer placed in header of all 5 evaluation reports and protocol Section 1. |
| **8. Checkpoint SHA-256 Parity** | Checkpoint hashes recomputed on disk match freeze manifest | **PASS** | All 5 candidate checkpoints re-hashed on disk; 100% bit-for-bit parity confirmed against manifest. |
| **9. Authorization Guards Enforced** | `TRAINING_AUTHORIZED = False`, `EVALUATION_AUTHORIZED = False` | **PASS** | Confirmed hardcoded at lines 41 and 42 of `experiments/final_evaluation/evaluate_frozen_candidate.py`. |
| **10. Zero Real-Data Access / Predictions** | Zero images loaded, zero labels read, zero predictions generated | **PASS** | Confirmed: No real prediction files exist in `reports/final_evaluation/` or repository. |

---

## 2. In-Depth Requirement Verification & Evidence

### Requirement 1 & 2: Candidate Specification and Checkpoint Parity
- **Candidate:** `EXP-06C Config A` (frozen DenseNet-121 backbone, 3-scan horizontal concatenation, linear projection head).
- **Physical Checkpoint Hash Parity:**
  The SHA-256 hashes of all five fold checkpoints were recomputed directly on disk on 2026-10-16T22:01:48:
  - **Fold 1:** `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5`  
    *Disk Hash:* `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc`  
    *Manifest Hash:* `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc`  
    *Status:* **PASS (100% Match, 52,899,216 bytes)**
  - **Fold 2:** `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5`  
    *Disk Hash:* `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c`  
    *Manifest Hash:* `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c`  
    *Status:* **PASS (100% Match, 52,899,472 bytes)**
  - **Fold 3:** `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5`  
    *Disk Hash:* `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1`  
    *Manifest Hash:* `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1`  
    *Status:* **PASS (100% Match, 52,899,472 bytes)**
  - **Fold 4:** `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5`  
    *Disk Hash:* `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3`  
    *Manifest Hash:* `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3`  
    *Status:* **PASS (100% Match, 52,899,472 bytes)**
  - **Fold 5:** `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5`  
    *Disk Hash:* `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6`  
    *Manifest Hash:* `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6`  
    *Status:* **PASS (100% Match, 52,899,472 bytes)**

### Requirement 3: Canonical Class Ordering
- In `experiments/final_evaluation/evaluate_frozen_candidate.py:44`:
  ```python
  CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
  CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
  ```
- In `reports/final_evaluation/preregistered_evaluation_protocol.md:58-62`:
  ```python
  CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
  INDEX_MAPPING = {0: "A+", 1: "AB+", 2: "B+", 3: "O+"}
  ```
- *Status:* **PASS.** Canonical mapping is 100% consistent across code, protocol, and confusion matrix indexing.

### Requirement 4: Input Structure, Preprocessing, and Ensemble Parity
- **Input Tensor:** Three scans per participant stacked to `(1, 3, 112, 112, 3)`.
- **Preprocessing:** `src.preprocessing.transform_c5` executes alpha composite $\rightarrow$ MFS100 platen crop $\rightarrow$ CLAHE ($clip=2.5$) $\rightarrow$ percentile stretch ($1-99\%$) $\rightarrow$ bilateral filter ($d=3, \sigma=25$) $\rightarrow$ resize $112\times 112$. Returns raw `float32` array in $[0.0, 255.0]$ without `preprocess_input`.
- **Ensemble Aggregation:** Unweighted arithmetic mean across the 5 fold models:
  $$\mathbf{P}_{\text{ensemble}}(y=c \mid \mathbf{X}) = \frac{1}{5} \sum_{k=1}^5 \mathbf{p}_k(c)$$
  Probability vector normalized to strictly sum to $1.000000 \pm 10^{-6}$.
- *Status:* **PASS.**

### Requirement 5: Metrics & Statistical Protocol Parity
- **Primary Endpoint:** Macro F1 (macro-averaged across all 4 classes). Pre-registered threshold: $\text{Macro F1} > 0.3500$ and exceeding Phase 3 baseline ($0.2500$).
- **Secondary Endpoints:** Overall Accuracy, Balanced Accuracy, per-class Precision, Recall, F1-Score, Support, and Confusion Matrix `[A+, AB+, B+, O+]`.
- **Uncertainty Quantification:** Non-parametric bootstrap ($B=2,000$ resamples with replacement, `seed=42`) generating 95% percentile confidence intervals for Macro F1, Accuracy, and Balanced Accuracy.
- **Statistical Comparison:** Exact two-sided binomial test on discordant pairs against the historical Phase 3 baseline.
- *Status:* **PASS.**

### Requirement 6: Prohibition on Post-Test Model Selection or Re-Tuning
- Protocol Section 5.3 strictly prohibits post-test threshold calibration, class prior re-weighting, temperature scaling, or re-training.
- Evaluator contains zero conditional loops for model selection and writes outputs on the first pass.
- *Status:* **PASS.**

### Requirement 7: Accuracy of Provenance & Context Disclosures
1. **Primary Cohort Size:** Exactly 21 human participants, 63 scans. (*PASS*)
2. **Serology Status:** Lack of physical serology lab sheets explicitly disclosed; marked **NOT VERIFIED** for clinical serology. (*PASS*)
3. **Prior Model Exposure:** Documented that Phase 3 Model B previously scored $38.10\%$ accuracy (8/21) and $0.2500$ Macro F1 on this cohort. (*PASS*)
4. **Candidate Exposure Based on Logs:** Forensic task logs (`task-1475`, `task-1482`) and `evaluation_execution_history_audit.md` prove that EXP-06C Config A has had **ZERO prior evaluation exposure**. Both previous invocations aborted before data access. (*PASS*)
5. **Prohibition on Pooling:** Explicit alert in `cohort_provenance_review.md` and `preregistered_evaluation_protocol.md` mandates that the 26-participant hospital cohort must remain separate and never pooled with the 21-participant cohort. (*PASS*)
6. **Cross-Cohort Overlap:** Stated as **NOT VERIFIED** because biometric minutiae matching across the `P...` and `S...` namespaces was not performed. (*PASS*)
7. **Clinical Disclaimer:** Prominently stated in all reports that fingerprint biometric classification is an investigational hypothesis and has no clinical diagnostic validity. (*PASS*)

### Requirement 8, 9 & 10: Checkpoints, Guards, and Zero Real Data Access
- **Checkpoint Hashes:** Recomputed on disk and verified bit-for-bit unchanged. (*PASS*)
- **Guards:** In `experiments/final_evaluation/evaluate_frozen_candidate.py`:
  - Line 41: `TRAINING_AUTHORIZED = False` (*PASS*)
  - Line 42: `EVALUATION_AUTHORIZED = False` (*PASS*)
- **Zero Real Data Access:** Verified via directory scan:
  - `reports/final_evaluation/primary_evaluation_predictions.csv`: **DOES NOT EXIST** (`exists = False`)
  - `reports/final_evaluation/primary_evaluation_results.json`: **DOES NOT EXIST** (`exists = False`)
  - *Status:* **PASS.**

---

## 3. Residual Risks & Audit Verdict

| Risk Area | Risk Assessment | Audit Verdict |
| :--- | :--- | :---: |
| **Statistical Power on $N=21$** | Each participant represents $4.76\%$ of accuracy. AB+ ($N=2$) is subject to $\pm 50\%$ recall swings. | **DISCLOSED & PRE-REGISTERED (PASS)** |
| **Code Execution Safety** | Evaluator cannot run without both variable edit (`EVALUATION_AUTHORIZED = True`) and CLI flag. | **VERIFIED (PASS)** |
| **Data Quarantine** | Neither cohort has been accessed; checkpoints are bit-for-bit locked. | **VERIFIED (PASS)** |

---

## 4. Final Audit Certification

All requirements from the charter have been classified as **PASS**. There are **zero FAIL or NEEDS REVIEW** items.

The evaluation protocol, candidate checkpoints, and evaluator implementation are certified fully compliant and ready for execution upon explicit authorization.

*Audit complete. All systems safely halted.*
