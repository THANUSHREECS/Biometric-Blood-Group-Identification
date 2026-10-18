# Data Split & Training Boundary Audit

**Investigation:** Accuracy Improvement — Root-Cause Diagnostic  
**Date:** 2026-10-13  
**Status:** COMPLETE & VERIFIED  
**Output Path:** `reports/accuracy_improvement/data_split_audit.md`  

---

## 1. Executive Summary

A critical prerequisite before designing any model improvement experiment is to establish unambiguous boundaries on which data can legitimately be used for training, hyperparameter tuning, model selection, and testing.

### Core Protocol Constraints:
1. **The 21-Participant Locked Test Set is Strictly Quarantined:**  
   It must NEVER be used for training, hyperparameter search, feature selection, loss function tuning, or early stopping. It is reserved exclusively for final evaluation after an experimental pipeline is completely frozen.
2. **The 26-Participant Cohort is Restricted to Descriptive Verification:**  
   Because physical-individual independence from the 101-participant cohort cannot be conclusively proven via minutiae matching, it must NOT be used for model selection or tuning.
3. **The 80 Development Participants Form the Sole Legitimate Training Pool:**  
   All new model training, domain adaptation, feature extraction, and architectural modifications must be conducted exclusively on the 80 development participants from `reports/phase3_participant_benchmark/participant_split.csv`.

---

## 2. Quantitative Partition Registry

```
Total Available MFS100 Participants (N = 101 from participant_split.csv)
├── Development Pool (N = 80, 240 Scans) [PERMITTED FOR MODEL EXPERIMENTS]
│   ├── Canonical TRAIN Split (N = 64, 192 Scans)
│   └── Canonical VALIDATION Split (N = 16, 48 Scans)
│   └── Permitted Protocol: 5-Fold Stratified Group Cross-Validation
└── Locked Test Set (N = 21, 63 Scans) [STRICTLY QUARANTINED / FINAL TEST ONLY]

External Descriptive Cohort (N = 26, 78 Scans from mfs100_ground_truth.csv)
└── Role: Secondary Descriptive Verification of Frozen Models [NO TUNING]
```

### Participant Counts by Blood Group

| Split | A+ | AB+ | B+ | O+ | Total Participants | Total Scans |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dev: TRAIN** | 15 | 5 | 20 | 24 | **64** | 192 |
| **Dev: VALIDATION** | 4 | 2 | 5 | 5 | **16** | 48 |
| **Total Dev Pool** | **19** | **7** | **25** | **29** | **80** | **240** |
| **Locked Test (Quarantined)** | 5 | 2 | 6 | 8 | **21** | **63** |
| **Descriptive Cohort** | 4 | 3 | 8 | 11 | **26** | **78** |

---

## 3. Training & Validation Rules for Proposed Experiments

1. **Participant Grouping Rule:**  
   All 3 scans of any participant must remain in the exact same fold. No participant's scans may be split across training and validation partitions.
2. **Evaluation Metrics During Development:**  
   All experimental selections must be driven by **Macro F1** and **Balanced Accuracy** across the 80 development participants (via 5-fold cross-validation or the canonical 16-participant validation set), **never** by raw accuracy alone.
3. **Requirement for Additional Data:**  
   $N=64$ training participants is statistically small for training deep neural networks from scratch. Therefore:
   - Full end-to-end retraining from random initialization is mathematically doomed to overfit.
   - Experiments must focus on **transfer learning, frozen/partially-unfrozen backbones, LoRA (Low-Rank Adaptation), or metric-loss heads**.
   - For long-term production improvements, collecting at least 200–500 additional independently verified MFS100 participants is strongly recommended.
