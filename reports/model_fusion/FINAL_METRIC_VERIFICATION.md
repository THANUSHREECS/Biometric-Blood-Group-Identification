# FINAL METRIC VERIFICATION: DENSENET-121 DEVELOPMENT EVALUATION

**Evaluation Timestamp:** 2026-10-11  
**Dataset Split:** Development Set (`TRAIN` + `VALIDATION` from `reports/phase3_participant_benchmark/participant_split.csv`)  
**Target Scope:** 4 ABO Blood Groups (`A`, `B`, `AB`, `O`)  
**Explicit Class Ordering:** `[A, B, AB, O]`  
**Total Scans Evaluated:** 240  
**Total Participants Evaluated:** 80 (3 scans per participant)  

---

## 1. Ground Truth Cohort Distribution

The development cohort consists of 80 participants and 240 scans. All participants in this cohort are Rh-positive, mapped into the four ABO antigen categories:

| Blood Group | Class Index | Scan Count | Scan Prevalence | Participant Count | Participant Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A** (A+) | 0 | 57 | 23.75% | 19 | 23.75% |
| **B** (B+) | 1 | 75 | 31.25% | 25 | 31.25% |
| **AB** (AB+) | 2 | 21 | 8.75% | 7 | 8.75% |
| **O** (O+) | 3 | 87 | 36.25% | 29 | 36.25% |
| **Total** | — | **240** | **100.00%** | **80** | **100.00%** |

---

## 2. Verified 8-Class to 4-Class ABO Marginalization

The checkpoint `models/densenet121_blood_group.keras` was trained on the 8-class system defined in `models/class_names.json`:
- $0 = \text{A+}$
- $1 = \text{A-}$
- $2 = \text{AB+}$
- $3 = \text{AB-}$
- $4 = \text{B+}$
- $5 = \text{B-}$
- $6 = \text{O+}$
- $7 = \text{O-}$

The final layer outputs an 8-dimensional softmax probability vector $P = [p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7]$.  
The 4-class ABO marginal probabilities are calculated by summing the positive and negative Rh components for each antigen group:

$$P(\text{A}) = p_0 + p_1$$
$$P(\text{B}) = p_4 + p_5$$
$$P(\text{AB}) = p_2 + p_3$$
$$P(\text{O}) = p_6 + p_7$$

The resulting vector $[P(\text{A}), P(\text{B}), P(\text{AB}), P(\text{O})]$ satisfies $\sum P = 1.0$, and the scan prediction is $\hat{y} = \arg\max P$.

---

## 3. Scan-Level Verification (240 Scans)

### 3.1 Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

$$\text{Class Order: } [0: \text{A}, \; 1: \text{B}, \; 2: \text{AB}, \; 3: \text{O}]$$

| True \ Pred | Pred A (0) | Pred B (1) | Pred AB (2) | Pred O (3) | Row Total (True) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True A (0)** | **14** | 18 | 7 | 18 | **57** |
| **True B (1)** | 11 | **21** | 3 | 40 | **75** |
| **True AB (2)** | 3 | 6 | **4** | 8 | **21** |
| **True O (3)** | 21 | 16 | 6 | **44** | **87** |
| **Col Total (Pred)** | **49** | **61** | **20** | **110** | **240** |

### 3.2 Performance Metrics

- **Overall Accuracy:** $\frac{14 + 21 + 4 + 44}{240} = \frac{83}{240} = \mathbf{0.3458}$ (**34.58%**)
- **Macro F1 Score:** $\mathbf{0.3037}$
- **Balanced Accuracy:** $\mathbf{0.3055}$

| Class | True Support | Correct (Diag) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | 57 | 14 | 0.2857 (14/49) | 0.2456 (14/57) | 0.2642 |
| **B** | 75 | 21 | 0.3443 (21/61) | 0.2800 (21/75) | 0.3088 |
| **AB** | 21 | 4 | 0.2000 (4/20) | 0.1905 (4/21) | 0.1951 |
| **O** | 87 | 44 | 0.4000 (44/110) | 0.5057 (44/87) | 0.4467 |
| **Macro Avg** | **240** | **83** | **0.3075** | **0.3055** | **0.3037** |

---

## 4. Five Consistency Checks (Verification Audit)

Every required consistency check was verified mathematically from the exact saved prediction arrays:

1. **Confusion Matrix Sum vs. Total Scans:**
   $$\sum_{i,j} C_{i,j} = 14+18+7+18+11+21+3+40+3+6+4+8+21+16+6+44 = 240$$
   *Status:* **PASSED** (240 == 240).

2. **Diagonal Sum Divided by Total vs. Accuracy:**
   $$\frac{\text{Trace}(C)}{N} = \frac{14 + 21 + 4 + 44}{240} = \frac{83}{240} = 0.345833\dots \equiv \text{Accuracy}$$
   *Status:* **PASSED**.

3. **Per-Class Recall vs. Diagonal / Row Sum:**
   - Class A: $\frac{14}{57} = 0.245614 \equiv \text{Recall}_A$
   - Class B: $\frac{21}{75} = 0.280000 \equiv \text{Recall}_B$
   - Class AB: $\frac{4}{21} = 0.190476 \equiv \text{Recall}_{AB}$
   - Class O: $\frac{44}{87} = 0.505747 \equiv \text{Recall}_O$
   *Status:* **PASSED** (Exact equality for all 4 classes).

4. **Macro F1 vs. Mean of Per-Class F1 Scores:**
   $$\frac{0.264151 + 0.308824 + 0.195122 + 0.446701}{4} = 0.303699 \equiv \text{Macro F1}$$
   *Status:* **PASSED**.

5. **Balanced Accuracy vs. Mean Per-Class Recall:**
   $$\frac{0.245614 + 0.280000 + 0.190476 + 0.505747}{4} = 0.305459 \equiv \text{Balanced Accuracy}$$
   *Status:* **PASSED**.

---

## 5. Participant-Level Verification (80 Participants)

### 5.1 Aggregation Method
- Each participant has exactly 3 scans recorded in `participant_split.csv`.
- **Voting Mechanism:** Majority vote across the 3 scan predictions.
- **Tie-Breaking Rule:** In the event that all 3 scans predict different classes (a 1-1-1 tie), the tie is broken deterministically by selecting the lowest class index ($\arg\max$ over sorted unique prediction labels).
- **Missing Predictions:** None. All 80 participants have 3 valid scans ($N = 80$).

### 5.2 Participant Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

| True \ Pred | Pred A (0) | Pred B (1) | Pred AB (2) | Pred O (3) | Row Total (True) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True A (0)** | **4** | 6 | 3 | 6 | **19** |
| **True B (1)** | 4 | **7** | 1 | 13 | **25** |
| **True AB (2)** | 1 | 2 | **1** | 3 | **7** |
| **True O (3)** | 7 | 5 | 2 | **15** | **29** |
| **Col Total (Pred)** | **16** | **20** | **7** | **37** | **80** |

### 5.3 Participant-Level Metrics

- **Overall Participant Accuracy:** $\frac{4 + 7 + 1 + 15}{80} = \frac{27}{80} = \mathbf{0.3375}$ (**33.75%**)
- **Participant Macro F1:** $\mathbf{0.2843}$
- **Participant Balanced Accuracy:** $\mathbf{0.2877}$

| Class | True Support | Correct (Diag) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | 19 | 4 | 0.2500 (4/16) | 0.2105 (4/19) | 0.2286 |
| **B** | 25 | 7 | 0.3500 (7/20) | 0.2800 (7/25) | 0.3111 |
| **AB** | 7 | 1 | 0.1429 (1/7) | 0.1429 (1/7) | 0.1429 |
| **O** | 29 | 15 | 0.4054 (15/37) | 0.5172 (15/29) | 0.4545 |
| **Macro Avg** | **80** | **27** | **0.2871** | **0.2877** | **0.2843** |

---

## 6. Audit Conclusion

The earlier reported discrepancies (where a confusion matrix summed to 191 entries or where DenseNet-121 collapsed to predicting only AB+) were traced to:
1. Subsetting errors that excluded negative-Rh or unstratified samples.
2. Directly argmaxing 8-class outputs without the required Rh-positive/Rh-negative ABO marginalization.

When the mathematically rigorous ABO marginalization is applied to the exact 240 development scans, **DenseNet-121 achieves 34.58% scan-level accuracy (Macro F1 = 0.3037) and 33.75% participant-level accuracy (Macro F1 = 0.2843)**. All five consistency checks pass without deviation.
