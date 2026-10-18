# EXP-06C: Expected Artifacts & Audit Registry

**Document Version:** 1.0.0  
**Preparation Date:** 2026-10-16  
**Experiment Identifier:** `EXP-06C` (Projection Head Architecture Ablation)  
**Lead Auditor:** Senior ML Research Scientist & Verification Specialist  

---

> [!CAUTION]
> ### RESEARCH USE ONLY — STRICT NON-CLINICAL DISCLAIMER
> All models, configurations, and artifact paths documented herein are academic research prototypes designed to study machine learning representations on optical biometric features. They **must not** be used for clinical blood grouping, medical diagnostics, patient transfusion matching, or triage.

---

## 1. Directory Structure & Storage Isolation

All newly generated experimental artifacts for EXP-06C will be isolated into dedicated directories to prevent any collision with or overwriting of existing checkpoints.

```
major project/
├── experiments/
│   └── exp06c/
│       ├── run_exp06c.py                     # Audited execution script
│       └── checkpoints/                      # Dedicated checkpoint directory
│           ├── config_a/                     # Config A (Linear Reference)
│           │   ├── fold_1_best.weights.h5
│           │   ├── fold_2_best.weights.h5
│           │   ├── fold_3_best.weights.h5
│           │   ├── fold_4_best.weights.h5
│           │   └── fold_5_best.weights.h5
│           └── config_b/                     # Config B (2-Layer Nonlinear MLP)
│               ├── fold_1_best.weights.h5
│               ├── fold_2_best.weights.h5
│               ├── fold_3_best.weights.h5
│               ├── fold_4_best.weights.h5
│               └── fold_5_best.weights.h5
│
└── reports/
    └── accuracy_improvement/
        ├── exp06c_preparation/              # Pre-training audit & safety package
        │   ├── experiment_plan.md
        │   ├── configuration_comparison.md
        │   ├── safety_audit.md
        │   ├── synthetic_test_report.md
        │   └── expected_artifacts.md
        │
        └── exp06c/                          # Post-training reports & predictions
            ├── metrics_comparison.csv       # Summary metrics comparison table
            ├── experiment_config.json       # Complete epoch-by-epoch training logs
            ├── statistical_analysis_results.json # Full paired statistical tests & hashes
            ├── final_report.md              # Comprehensive final experimental report
            ├── config_a_oof_predictions.csv # 80 participant OOF predictions (Config A)
            ├── config_b_oof_predictions.csv # 80 participant OOF predictions (Config B)
            └── confusion_matrices/          # Visual heatmaps
                ├── config_a_confusion_matrix.png
                └── config_b_confusion_matrix.png
```

---

## 2. Checkpoint Specifications & Expected Registry

A total of **10 new model weight checkpoints** (2 configurations $\times$ 5 folds) will be produced upon execution:

| Configuration | Fold | Checkpoint Path | Architecture | Target Metric Monitored |
|:---|:---:|:---|:---|:---:|
| **Config A** (Linear) | Fold 1 | `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5` | Linear Projection (`3072x128`) | Held-out Val Macro F1 |
| | Fold 2 | `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5` | Linear Projection (`3072x128`) | Held-out Val Macro F1 |
| | Fold 3 | `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5` | Linear Projection (`3072x128`) | Held-out Val Macro F1 |
| | Fold 4 | `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5` | Linear Projection (`3072x128`) | Held-out Val Macro F1 |
| | Fold 5 | `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5` | Linear Projection (`3072x128`) | Held-out Val Macro F1 |
| **Config B** (MLP) | Fold 1 | `experiments/exp06c/checkpoints/config_b/fold_1_best.weights.h5` | 2-Layer MLP (`3072x512` -> `512x128`) | Held-out Val Macro F1 |
| | Fold 2 | `experiments/exp06c/checkpoints/config_b/fold_2_best.weights.h5` | 2-Layer MLP (`3072x512` -> `512x128`) | Held-out Val Macro F1 |
| | Fold 3 | `experiments/exp06c/checkpoints/config_b/fold_3_best.weights.h5` | 2-Layer MLP (`3072x512` -> `512x128`) | Held-out Val Macro F1 |
| | Fold 4 | `experiments/exp06c/checkpoints/config_b/fold_4_best.weights.h5` | 2-Layer MLP (`3072x512` -> `512x128`) | Held-out Val Macro F1 |
| | Fold 5 | `experiments/exp06c/checkpoints/config_b/fold_5_best.weights.h5` | 2-Layer MLP (`3072x512` -> `512x128`) | Held-out Val Macro F1 |

---

## 3. Prediction & Evaluation Files Schema

### 3.1 Out-of-Fold Prediction CSV Schema
Each configuration will output an OOF prediction file (`{config}_oof_predictions.csv`) with exactly **80 rows** corresponding to the 80 development participants:

| Column Name | Type | Description | Verification Rule |
|:---|:---:|:---|:---|
| `participant_id` | String | Unique participant identifier | Exactly 80 unique IDs; matches dev set; 0 test IDs |
| `fold_id` | Integer | Cross-validation fold index | Integer $\in \{1, 2, 3, 4, 5\}$ |
| `true_label` | String | Ground-truth blood group | String $\in \{\text{'A+'}, \text{'AB+'}, \text{'B+'}, \text{'O+'}\}$ |
| `predicted_label` | String | Model predicted blood group | String $\in \{\text{'A+'}, \text{'AB+'}, \text{'B+'}, \text{'O+'}\}$ |
| `canonical_probabilities`| JSON String | Probabilities in canonical order | List of 4 floats: `[P(A+), P(AB+), P(B+), P(O+)]`, sum to $1.0$ |
| `native_probabilities` | JSON String | Probabilities in native model order | List of 4 floats: `[P(A+), P(B+), P(AB+), P(O+)]`, sum to $1.0$ |

---

## 4. Post-Training Audit Checklist

Following training execution, the following verification gates must be audited before declaring experiment completion:

- [ ] **Cohort Quarantine:** Exactly 80 participant predictions per config. Zero overlap with 21 locked test or 26 hospital participants.
- [ ] **Prediction Completeness:** Every participant has exactly 1 OOF prediction per configuration.
- [ ] **Protected Checkpoint Immutability:** SHA-256 hashes of all 56 baseline checkpoints verified bit-for-bit identical before and after training.
- [ ] **New Checkpoint Existence:** Exactly 10 new `.weights.h5` files exist, are readable, and have valid non-zero file sizes.
- [ ] **Metric Recomputation:** Primary pooled OOF Macro F1 and secondary metrics recomputed directly from the prediction CSVs.
- [ ] **Statistical Hypothesis Testing:** Paired Exact Binomial / continuity-corrected McNemar tests executed with exact $p$-values reported.
- [ ] **Historical Reference Comparison:** Config A compared with EXP-06B Arm D historical performance ($0.4925$ Macro F1, $52.50\%$ accuracy).
- [ ] **Scratch Cleanup:** All ephemeral template files (`scratch/exp06c_init_shared_clf_fold_*.json`) cleaned up upon completion.
