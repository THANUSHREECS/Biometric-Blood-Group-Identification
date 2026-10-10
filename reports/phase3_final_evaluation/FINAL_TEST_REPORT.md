# FINAL INDEPENDENT EVALUATION: E2 CHAMPION ENSEMBLE

## 1. DATASET PROPERTIES
- **Total Participants:** 21 (Strictly locked test participants)
- **Total Scans:** 63 (3 per participant)
- **Ground Truth Distribution:**
  - A+: 5 participants
  - B+: 6 participants
  - AB+: 2 participants
  - O+: 8 participants

## 2. PRIMARY METRICS
- **Accuracy:** 38.10%
- **Macro F1:** 0.2500
- **Weighted F1:** 0.3274
- **Balanced Accuracy:** 0.3062

## 3. UNCERTAINTY (10,000 Bootstraps)
*Calculated at the participant level.*
- **Accuracy 95% CI:** [19.05%, 57.14%]
- **Macro F1 95% CI:** [0.1304, 0.4057]

*Note on limitations: Due to the exceptionally small test set (21 participants), the 95% confidence intervals are massive. A shift in just 2 participants drastically swings the performance. Furthermore, AB+ had only 2 representatives, meaning many bootstrap samples contained zero AB+ cases, making Macro F1 highly unstable and statistically precarious.*

## 4. PER-CLASS PERFORMANCE
| Class | Precision | Recall | F1-Score | True Count | Predicted Count |
|-------|-----------|--------|----------|------------|-----------------|
| **A+**  | 27.27% | 60.00% | 37.50% | 5 | 11 |
| **B+**  | 0.00% | 0.00% | 0.00% | 6 | 2 |
| **AB+** | 0.00% | 0.00% | 0.00% | 2 | 0 |
| **O+**  | 62.50% | 62.50% | 62.50% | 8 | 8 |

## 5. BASELINE COMPARISON
We are comparing the final multi-modal DenseNet121 Contrastive Metric Learning Ensemble against the Day 27 Baseline (Standard CNN). Both models were evaluated on the **exact same 21 unseen participants**.

- **Day 27 Baseline Accuracy:** 38.09%
- **Final E2 Ensemble Accuracy:** 38.10% (Identical)

- **Day 27 Baseline Macro F1:** 0.3125
- **Final E2 Ensemble Macro F1:** 0.2500 (Worse)

## 6. SCIENTIFIC INTERPRETATION

**1. Generalization Failure:** 
Despite sophisticated data-firewalls, contrastive metric learning, participant-aware negative sampling, and three-scan feature fusion, the final model failed to generalize to strictly unseen participants. The extensive development improvements (reaching 53% accuracy and 0.42 Macro F1 during validation) did not translate to the locked test set. 

**2. Metric Collapse:** 
The model suffered a complete recall collapse on the B+ and AB+ classes (0% recall for both). The contrastive projection head completely failed to map the subtle, purported blood-group textures of the unseen B/AB individuals into the learned clusters. Instead, the model massively overpredicted A+ (predicting 11 participants as A+ when only 5 existed). 

**3. Biological Plausibility:** 
This experiment provides extremely strong empirical evidence against the hypothesis that fingerprint ridge patterns contain a biologically deterministic, generalizable macroscopic signature for the ABO blood group that can be resolved by a standard MFS100 scanner. Any signal detected during development was likely a mathematical artifact of the specific cross-validation splits, rather than a universal biological biomarker. A successful computational evaluation does not establish a biological mechanism, and in this case, the computational evaluation itself entirely rejected the hypothesis on the independent test set. 

**4. Conclusion:**
There is no evidence that the model is clinically valid, nor that fingerprint images can reliably detect blood groups. The 38% accuracy is essentially indistinguishable from a naive random guess distributed around the class priors.
