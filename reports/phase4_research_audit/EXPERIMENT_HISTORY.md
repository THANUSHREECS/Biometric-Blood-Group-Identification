# EXPERIMENT HISTORY SUMMARY

## 1. PRE-MFS100 RESEARCH (Canonical Dataset)
- **Original Claim:** 85.41% Accuracy, 0.8590 Macro F1.
- **Provenance Analysis:** Searching the project history reveals this was achieved on an 898-image test set from a "6,000-image dataset (5,985 cleaned samples)". The dataset was synthetic/scanned canonical images ($96 \times 103$), *not* real optical captures. The evaluation split was likely at the image-level (which, in a 10-finger-per-person dataset, causes massive participant leakage).

## 2. PHASE 3 MFS100 DEVELOPMENT (101 Participants)
*Note: All development used the 80 Dev participants (64 Train / 16 Val).*

**Experiment 0: Day 27 Baseline**
- **Hypothesis:** Does the 85.41% model generalize to MFS100 real captures?
- **Result:** Locked Test Accuracy: 38.09%, Macro F1: 0.3125.
- **Verdict:** Massive domain gap and likely failure to generalize biologically.

**Experiment 1: Architecture Benchmark**
- **Hypothesis:** Can a different CNN backbone improve MFS100 feature extraction?
- **Result:** DenseNet121 and EfficientNet emerged as the strongest candidates, but testing indicated massive overfitting. 
- **Verdict:** DenseNet121 chosen as base.

**Experiment 2: Transfer Learning Strategy**
- **Hypothesis:** Given only 64 training participants, should the backbone be frozen?
- **Result:** T1 (Frozen ImageNet Backbone) outperformed fine-tuning (T2/T3). 
- **Verdict:** Freezing is necessary to prevent catastrophic overfitting on 64 participants.

**Experiment 3: Preprocessing Benchmark**
- **Hypothesis:** Does ridge-contrast enhancement help?
- **Result:** C5 (Composite + Crop + Contrast Enhancement + Resize) achieved ~50% Validation Accuracy.
- **Verdict:** C5 adopted.

**Experiment 4: Three-Scan Feature Fusion**
- **Hypothesis:** Does aggregating 3 left-thumb scans provide more signal?
- **Result:** Feature Concatenation (F1) outperformed Probability Averaging (F0). Val Macro F1: 0.3770.

**Experiment 5: Supervised Contrastive Metric Learning**
- **Hypothesis:** Can contrastive loss cluster the minority classes (AB+, B+) better?
- **Result:** Joint loss (Classification + SupCon) improved minority recall during cross-validation.

**Experiment 6: Final Champion E2 Development**
- **Hypothesis:** Combine Fusion (Exp 4) and Metric Learning (Exp 5).
- **Result (5-Fold CV):** Mean Val Accuracy: 53.75%, Mean Val Macro F1: 0.4240. 

**Experiment 7: Independent Test Evaluation (Final)**
- **Hypothesis:** Evaluate frozen E2 ensemble on 21 locked participants.
- **Result:** Accuracy: 38.10%, Macro F1: 0.2500. 
- **Verdict:** Generalization failed completely. The model collapsed on B+ and AB+. Performance matched random/baseline exactly.
