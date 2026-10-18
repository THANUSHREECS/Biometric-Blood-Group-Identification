# Audit Erratum: Prior Evaluations of the Locked MFS100 Test Cohort

**Target Document:** `reports/leakage_audit/leakage_audit_report.md` (Frozen Audit Report, Section 1.1 / Section 2)  
**Date of Erratum:** October 17, 2026  
**Document Status:** Formal Erratum (New file in `reports/project_report/`; original frozen audit files remain unmodified)  
**Reference Artifact:** `reports/project_report/locked_cohort_usage.csv`  
**Mapping Data File:** `reports/project_report/erratum_artifact_map.json` (Manually curated artifact and checkpoint mapping)  
**Statistical Source:** `experiments/leakage_audit/exp2_signal_tests_results.json`

---

## 1. Context and Problem Statement

In `reports/leakage_audit/leakage_audit_report.md` (Section 1.1, line 19), the report states:
> *"No new inference or tuning was conducted on the locked MFS100 test cohort (21 participants) in this audit; it was evaluated once previously and its saved predictions were read."*

A comprehensive audit of repository evaluation artifacts (`reports/project_report/locked_cohort_usage.csv`) demonstrates that the statement **"it was evaluated once previously" is factually unsupported by the repository evidence**.

While the first half of the statement remains true—no new inference or hyperparameter tuning on the locked test cohort was executed during the leakage audit itself—the locked test cohort was **not** evaluated only once prior to the audit. In reality, as derived from `locked_cohort_usage.csv`, the 21 locked test participants were evaluated across **17 prediction artifacts (15 occurring before EXP-03 began, and 2 after EXP-03, one of which is a read-only permutation re-analysis of saved predictions)** from which **10 distinct evaluation runs** can be established involving **10 distinct verifiable model checkpoint paths before EXP-03 (plus 1 unspecified narrative baseline)** across **6 distinct project phases** prior to and including the EXP-06C Config A locked evaluation.

Critically, **the E2 champion model was scored on the locked cohort across three separate evaluation events between October 10 and October 13, 2026. Because EXP-03 through EXP-06D (which developed E2 variants, culminating in EXP-06C Config A) began on October 14, 2026, Config A's underlying model family and feature representations had already been scored in three recorded evaluations on the locked cohort before the accuracy improvement program began.**

---

## 2. Itemized Prediction Artifacts and Distinct Evaluation Runs on the Locked Test Cohort

### 2.1 Distinct Evaluation Runs Established from Repository Artifacts

A rigorous inspection of scripts, prediction CSVs, and metrics JSON files reveals that the 17 prediction artifacts correspond to **10 distinct evaluation runs**:

1. **Day 27 Live Zero-Shot Run (1 run):**  
   The participant-level file (`reports/final_mfs100/final_zero_shot_participant_predictions.csv`) and scan-level file (`reports/final_mfs100/final_zero_shot_scan_predictions.csv`) were generated together by a **single evaluation run** executed by `scratch/final_zero_shot_eval.py` on 2026-10-07.
2. **Phase 3 Architecture Benchmark Runs (7 runs):**  
   7 architectures (ConvNeXt-Tiny, Custom CNN, DenseNet-121, EfficientNet-B0, EfficientNetV2-B0, ResNet-50, VGG16) fine-tuned on Train 64 were evaluated independently on the 21 locked participants on 2026-10-07 by `scratch/experiment1_architecture.py`.
3. **E2 Champion Ensemble Run (1 run, 3 recorded results):**  
   E2 was evaluated to produce `reports/phase3_final_evaluation/final_test_predictions.csv` (committed in `c1aac08`). The subsequent evaluation records in `reports/final/eval_101_raw_results.json` and `reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv` contain identical participant predictions and metrics (38.10% accuracy, 0.2500 macro F1). Therefore, they are **repeated looks at the same predictions from a single evaluation run**, rather than new runs.
4. **Day 27 Narrative Baseline in `FINAL_TEST_REPORT.md` (0 backing runs):**  
   The Day 27 baseline figures cited in `FINAL_TEST_REPORT.md` (38.09% accuracy, 0.3125 macro F1) have **no backing run** or prediction CSV in the repository; they exist solely as text in markdown.
5. **EfficientNet-B0 Single-Scan Evaluation Run (1 run):**  
   `reports/efficientnet_e2_ensemble_experiment/efficientnet_predictions.csv` represents a distinct evaluation run evaluating the original synthetic checkpoint `models/abo_positive_4class/best_model.keras` on 2026-10-13.
6. **Equal-Weight Ensemble (0 new inference runs):**  
   `equal_weight_ensemble_predictions.csv` was mathematically derived as an arithmetic probability average of the existing E2 and EfficientNet predictions.
7. **EXP-06C Config A Primary Evaluation Run (1 run):**  
   Candidate evaluation run executed on 2026-10-16 (`reports/final_evaluation/primary_evaluation_predictions.csv`).
8. **Leakage Audit Permutation Signal Test (0 new inference runs):**  
   Read-only permutation test re-analyzing saved predictions from `primary_evaluation_predictions.csv`.

**Total Distinct Evaluation Runs:** Exactly **10 distinct evaluation runs** involving active model inference on the locked test cohort are established by repository evidence (9 pre-EXP-03, 1 candidate evaluation).

---

### 2.2 Derivation Code and Output from `locked_cohort_usage.csv`

The counts are derived programmatically using the manually curated mapping defined in `reports/project_report/erratum_artifact_map.json`:

```python
import json
import pandas as pd

with open("reports/project_report/erratum_artifact_map.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

summary_docs = set(meta["summary_docs"])
checkpoint_map = meta["checkpoint_map"]

df = pd.read_csv("reports/project_report/locked_cohort_usage.csv")

scoring_rows = df[~df["artifact_path"].isin(summary_docs)].copy()
scoring_rows["checkpoint"] = scoring_rows["artifact_path"].map(checkpoint_map)

before_exp03 = scoring_rows[scoring_rows["available_before_exp03_06d"] == "yes"]
after_exp03 = scoring_rows[scoring_rows["available_before_exp03_06d"] == "no"]

atomic_checkpoints_before = set()
for cp in before_exp03["checkpoint"].dropna():
    for part in cp.split(" + "):
        atomic_checkpoints_before.add(part.strip())

print(f"Total prediction artifacts: {len(scoring_rows)}")
print(f"Prediction artifacts before EXP-03: {len(before_exp03)}")
print(f"Prediction artifacts after EXP-03: {len(after_exp03)} (one primary evaluation, one read-only re-analysis)")
print(f"Distinct checkpoint paths before EXP-03 ({len(atomic_checkpoints_before)}):")
for i, p in enumerate(sorted(atomic_checkpoints_before), 1):
    print(f"  {i:2d}. {p}")
```

**Execution Output:**
```text
Total prediction artifacts: 17
Prediction artifacts before EXP-03: 15
Prediction artifacts after EXP-03: 2 (one primary evaluation, one read-only re-analysis)
Distinct checkpoint paths before EXP-03 (10):
   1. models/abo_positive_4class/best_model.keras
   2. models/day27_full_dataset/four_class/best_model.keras
   3. models/phase3_architecture_benchmark/convnext_tiny/best_model.keras
   4. models/phase3_architecture_benchmark/custom_cnn/best_model.keras
   5. models/phase3_architecture_benchmark/densenet121/best_model.keras
   6. models/phase3_architecture_benchmark/efficientnet_b0/best_model.keras
   7. models/phase3_architecture_benchmark/efficientnetv2_b0/best_model.keras
   8. models/phase3_architecture_benchmark/resnet50/best_model.keras
   9. models/phase3_architecture_benchmark/vgg16/best_model.keras
  10. models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5
(Plus 1 unspecified narrative baseline in FINAL_TEST_REPORT.md; Config A's checkpoint was created on Oct 16)
```

*(Note: The artifact classification into summary vs. prediction artifacts and checkpoint associations are manually curated in `reports/project_report/erratum_artifact_map.json` based on repository inspection).*

---

### 2.3 Verification of Checkpoints and SHA-256 Hashes

Every checkpoint path listed in `checkpoint_map` was verified to exist on disk. SHA-256 hashes of the core model weights were computed as follows:

| Model / Checkpoint File | File Path | Exists | SHA-256 Hash |
|---|---|---|---|
| **E2 Fold 1 Weights** | `models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5` | Yes | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` |
| **E2 Fold 2 Weights** | `models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5` | Yes | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` |
| **E2 Fold 3 Weights** | `models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5` | Yes | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` |
| **E2 Fold 4 Weights** | `models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5` | Yes | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` |
| **E2 Fold 5 Weights** | `models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5` | Yes | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` |
| **Config A Fold 1 Weights** | `experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5` | Yes | `ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc` |
| **Config A Fold 2 Weights** | `experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5` | Yes | `615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c` |
| **Config A Fold 3 Weights** | `experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5` | Yes | `e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1` |
| **Config A Fold 4 Weights** | `experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5` | Yes | `58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3` |
| **Config A Fold 5 Weights** | `experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5` | Yes | `6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6` |
| **EffNet Pretrained Model** | `models/abo_positive_4class/best_model.keras` | Yes | `7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9` |

**Script Lines Loading E2 Weights:**
- `scratch/final_evaluation_e2.py:80-82`:
  ```python
  for cp in checkpoint_paths:
      cp_abs = os.path.join(project_root, cp.replace("/", os.sep))
      model.load_weights(cp_abs)
  ```
- `scratch/run_full_experiment.py:131-133`:
  ```python
  for fold in range(1, 6):
      cp = E2_CKPT_DIR / f"fold_{fold}_best.weights.h5"
      e2_model.load_weights(str(cp))
  ```

---

### 2.4 Table of Prediction Artifacts on the Locked Test Cohort

Derived from `reports/project_report/locked_cohort_usage.csv` and `erratum_artifact_map.json`:

| Model Evaluated | Prediction Artifact Path | File Mtime | Git Tracked Date | Model Checkpoint Path (Where Known) | Model Identity Across Files |
|---|---|---|---|---|---|
| **Day 27 Model A (participant)** | `reports/final_mfs100/final_zero_shot_participant_predictions.csv` | 2026-10-07 20:05:25 | `c1aac08` (2026-10-10) | `models/day27_full_dataset/four_class/best_model.keras` (`scratch/final_zero_shot_eval.py:19-24`) | Purports to be same baseline model as Phase 3 report, but metrics conflict (33.33% / 0.1875 here vs 38.09% / 0.3125 in Phase 3 report). |
| **Day 27 Model A (scan-level)** | `reports/final_mfs100/final_zero_shot_scan_predictions.csv` | 2026-10-07 20:05:25 | `c1aac08` (2026-10-10) | `models/day27_full_dataset/four_class/best_model.keras` (`scratch/final_zero_shot_eval.py:19-24`) | Single-scan zero-shot predictions (63 scans) of Day 27 Model A (Acc: 0.3492, Macro F1: 0.1792). |
| **ConvNeXt-Tiny (Pretrained)** | `reports/phase3_architecture_benchmark/convnext_tiny_test_preds.csv` | 2026-10-07 22:09:15 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/convnext_tiny/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Unique checkpoint fine-tuned on 64 Train participants. |
| **Custom CNN (Phase 3 Benchmark)** | `reports/phase3_architecture_benchmark/custom_cnn_test_preds.csv` | 2026-10-07 21:00:22 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/custom_cnn/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Distinct from Day 27 Model A; trained from scratch on 64 MFS100 Train participants (Acc: 0.2857, Macro F1: 0.1200). |
| **DenseNet-121 (Pretrained)** | `reports/phase3_architecture_benchmark/densenet121_test_preds.csv` | 2026-10-07 21:38:17 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/densenet121/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Single-scan DenseNet-121 fine-tuned on 64 Train participants; distinct from E2 3-scan fusion. |
| **EfficientNet-B0 (Benchmark)** | `reports/phase3_architecture_benchmark/efficientnet_b0_test_preds.csv` | 2026-10-07 21:43:17 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/efficientnet_b0/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Fine-tuned on 64 Train participants (Acc: 0.1905, Macro F1: 0.1250); distinct from original synthetic checkpoint. |
| **EfficientNetV2-B0 (Pretrained)** | `reports/phase3_architecture_benchmark/efficientnetv2_b0_test_preds.csv` | 2026-10-07 21:48:06 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/efficientnetv2_b0/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Unique checkpoint fine-tuned on 64 Train participants. |
| **ResNet-50 (Pretrained)** | `reports/phase3_architecture_benchmark/resnet50_test_preds.csv` | 2026-10-07 21:28:25 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/resnet50/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Unique checkpoint fine-tuned on 64 Train participants. |
| **VGG16 (Pretrained)** | `reports/phase3_architecture_benchmark/vgg16_test_preds.csv` | 2026-10-07 21:17:11 | `c1aac08` (2026-10-10) | `models/phase3_architecture_benchmark/vgg16/best_model.keras` (`scratch/experiment1_architecture.py:215, 232`) | Unique checkpoint fine-tuned on 64 Train participants. |
| **E2 5-Fold Ensemble (`DenseNet121_Fusion_SupCon`)** | `reports/phase3_final_evaluation/final_test_predictions.csv` | 2026-10-11 13:00:45 | `c1aac08` (2026-10-10) | `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5` | Original predictions of E2 on locked cohort. |
| **Day 27 Baseline (Standard CNN)** | `reports/phase3_final_evaluation/FINAL_TEST_REPORT.md` | 2026-10-10 15:00:47 | `c1aac08` (2026-10-10) | Unspecified in file (narrative only) | Stated in markdown narrative only; no backing prediction CSV exists. |
| **E2 5-Fold Ensemble (`DenseNet121_Fusion_SupCon`)** | `reports/final/eval_101_raw_results.json` | 2026-10-11 19:08:12 | Untracked | `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5` | Identical participant predictions as Phase 3 Final Evaluation (repeated look). |
| **EfficientNet-B0 (Single-scan)** | `reports/efficientnet_e2_ensemble_experiment/efficientnet_predictions.csv` | 2026-10-13 23:03:26 | Untracked | `models/abo_positive_4class/best_model.keras` (`scratch/run_full_experiment.py:38, 87`) | Loaded original-dataset synthetic checkpoint; distinct from architecture benchmark model. |
| **E2 5-Fold Ensemble (`DenseNet121_Fusion_SupCon`)** | `reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv` | 2026-10-13 23:03:26 | Untracked | `models/phase3_fusion_metric_learning/E2/fold_[1-5]_best.weights.h5` | Identical participant predictions as Phase 3 Final Evaluation (repeated look). |
| **Equal-Weight Ensemble (EffNet-B0 + E2)** | `reports/efficientnet_e2_ensemble_experiment/equal_weight_ensemble_predictions.csv` | 2026-10-13 23:03:26 | Untracked | Combination of `models/abo_positive_4class/best_model.keras` and E2 weights | 50/50 probability average of the two preceding prediction arrays. |
| **EXP-06C Config A (`DenseNet121_Fusion_LinearHead`)** | `reports/final_evaluation/primary_evaluation_predictions.csv` | 2026-10-16 22:08:38 | Untracked | `experiments/exp06c/checkpoints/config_a/fold_[1-5]_best.weights.h5` | Pre-registered candidate; exact same predictions re-analyzed in leakage audit. |
| **EXP-06C Config A (Saved Predictions)** | `experiments/leakage_audit/exp2_signal_tests_results.json` | 2026-10-16 22:54:21 | Untracked | Read from `primary_evaluation_predictions.csv` | Exact same predictions as primary candidate evaluation (read-only permutation test, no new inference). |

---

## 3. Detailed Dating Audit: Git First-Added Commit vs. Filesystem mtime

For every locked-cohort artifact in `locked_cohort_usage.csv`, the filesystem modification time (`mtime`) is compared directly against the first Git commit that added the file (`git log --diff-filter=A --format="%h %ci" -- <file>`):

| Artifact Path | Filesystem `mtime` | First Added Git Commit (`git log --diff-filter=A`) | Git Status | Primary Evidence Basis |
|---|---|---|---|---|
| `reports/final_mfs100/final_zero_shot_participant_predictions.csv` | 2026-10-07 20:05:25 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/final_mfs100/final_zero_shot_scan_predictions.csv` | 2026-10-07 20:05:25 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/final_mfs100/FINAL_ZERO_SHOT_MFS100_REPORT.md` | 2026-10-10 15:00:47 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/convnext_tiny_test_preds.csv` | 2026-10-07 22:09:15 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/custom_cnn_test_preds.csv` | 2026-10-07 21:00:22 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/densenet121_test_preds.csv` | 2026-10-07 21:38:17 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/efficientnet_b0_test_preds.csv` | 2026-10-07 21:43:17 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/efficientnetv2_b0_test_preds.csv` | 2026-10-07 21:48:06 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/resnet50_test_preds.csv` | 2026-10-07 21:28:25 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_architecture_benchmark/vgg16_test_preds.csv` | 2026-10-07 21:17:11 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_final_evaluation/final_test_predictions.csv` | 2026-10-11 13:00:45 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_final_evaluation/final_test_classification_report.csv` | 2026-10-11 13:00:45 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/phase3_final_evaluation/FINAL_TEST_REPORT.md` | 2026-10-10 15:00:47 | `c1aac08 2026-10-10 12:50:50 +0530` | Clean/Committed | **Git Commit Date (`c1aac08`)** |
| `reports/final/eval_101_raw_results.json` | 2026-10-11 19:08:12 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/final/E2_101_PARTICIPANT_EVALUATION.md` | 2026-10-11 19:09:32 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime + In-file text (`October 11, 2026`) |
| `reports/efficientnet_e2_ensemble_experiment/locked_test_metrics.json` | 2026-10-13 22:58:45 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv` | 2026-10-13 23:03:26 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/efficientnet_e2_ensemble_experiment/efficientnet_predictions.csv` | 2026-10-13 23:03:26 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/efficientnet_e2_ensemble_experiment/equal_weight_ensemble_predictions.csv` | 2026-10-13 23:03:26 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/efficientnet_e2_ensemble_experiment/participant_disagreement_analysis.csv` | 2026-10-13 23:03:26 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime only |
| `reports/final_evaluation/primary_evaluation_predictions.csv` | 2026-10-16 22:08:38 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime (Post-EXP-03) |
| `reports/final_evaluation/primary_evaluation_results.json` | 2026-10-16 22:08:44 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime + In-file date (`2026-10-16`) |
| `reports/final_evaluation/primary_evaluation_final_report.md` | 2026-10-16 22:12:17 | Untracked / Not in Git history | Untracked (`??`) | Filesystem mtime + In-file date (`2026-10-16`) |
| `experiments/leakage_audit/exp2_signal_tests_results.json` | 2026-10-16 22:54:21 | Untracked / Not in Git history | Staged (`A `) | Post-EXP-03 (Audit execution date) |

### 3.1 Unreliability of Filesystem mtimes and Primacy of Git Commit History

The comparison above demonstrates conclusively that **filesystem modification times (`mtime`) are unreliable as ground-truth chronology**:
- In `reports/phase3_final_evaluation/FINAL_TEST_REPORT.md`, the file has a filesystem mtime of `2026-10-10 15:00:47 +0530`, yet its first-added Git commit (`c1aac08`) is timestamped **`2026-10-10 12:50:50 +0530`**—more than two hours *earlier*.
- In `reports/phase3_final_evaluation/final_test_predictions.csv`, the disk mtime is `2026-10-11 13:00:45 +0530`, yet running `git diff c1aac08 -- reports/phase3_final_evaluation/final_test_predictions.csv` produces 0 diff lines (the committed file from October 10 is byte-for-byte identical to the file on disk).
- Any speculative explanations for these mtime discrepancies (such as local execution of evaluation scripts or Python `.pyc` bytecode creation timestamps) are **strictly unverified**. 
- Therefore, the **first-added Git commit date (`c1aac08` at `2026-10-10 12:50:50 +0530`) constitutes the primary, authoritative evidence of historical provenance** for all tracked benchmark and evaluation files.

### 3.2 Sourcing of the EXP-03 Start Time
The EXP-03 start timestamp (**`2026-10-14 13:08:28`**) is sourced directly from the filesystem modification time of the earliest preparation file in EXP-03: `reports/accuracy_improvement/exp03_preparation/configuration_diff.json`. The primary experiment execution script `experiments/exp03/run_exp03.py` was created at `13:59:53`, and the first fold checkpoint (`fold_1_best.weights.h5`) was written at `14:01:49`.

---

## 4. Scientific Impact and Clarification

1. **Repeated Cohort Exposure:**  
   The phrase "evaluated once previously" created the inaccurate impression that the locked test cohort was an unprobed, virgin test set that was accessed exactly once across the research history. In reality, multiple preceding model architectures and ensembles were benchmarked on these exact 21 participants across 10 distinct evaluation runs between October 7 and October 13, 2026.

2. **Integrity of Candidate Selection & Backbone Lineage:**  
   The candidate selection for EXP-06C Config A (finalized on October 16, 2026) was performed using 5-fold cross-validation on the 80 development participants (`Dev 80`), without direct parameter fitting on the 21 locked test participants. However, researchers and developers had prior visibility into the performance of earlier architectures on the locked test set.

   **Audit of TEST Split Usage Across Development Experiments:**
   - **Experiments 2 through 6 load the TEST split strictly for firewall assertions, with no inference or evaluation on TEST:**
     - `scratch/experiment2_transfer_learning.py:47, 53-54`:
       ```python
       test_df = split_df[split_df["split"] == "TEST"].copy()
       assert len(test_pids) == 21, f"Expected 21 test participants, found {len(test_pids)}"
       assert len(set(dev_pids).intersection(set(test_pids))) == 0, "Leakage between dev and test!"
       ```
     - `scratch/experiment3_preprocessing.py`: Does not reference the TEST split at all (lines 46–48 filter exclusively for `TRAIN` and `VALIDATION`).
     - `scratch/experiment4_fusion.py:48, 54-55`:
       ```python
       test_df = split_df[split_df["split"] == "TEST"].copy()
       assert len(test_pids) == 21, f"Expected 21 test participants, found {len(test_pids)}"
       assert len(set(dev_pids).intersection(set(test_pids))) == 0, "Leakage between dev and test!"
       ```
     - `scratch/experiment5_supcon.py` and `scratch/experiment6_fusion_supcon.py`: Contain no references to the TEST split.
   
   - **Experiment 1 Architecture Benchmark Printed Locked-Test Metrics Beside Validation Rankings:**
     In `reports/phase3_architecture_benchmark/EXPERIMENT1_REPORT.md`:
     Lines 3–7 (Validation Rankings):
     ```markdown
     ## Validation Rankings
     | Architecture | Val Macro F1 | Val Accuracy | Val O+ Share |
     | EfficientNet-B0 | 0.5667 | 0.7500 | 0.5625 |
     | DenseNet121 | 0.3173 | 0.5000 | 0.4375 |
     ```
     Lines 15–19 (Final Locked Test Performance printed directly below):
     ```markdown
     ## Final Locked Test Performance
     | Architecture | Test Macro F1 | Test Accuracy | Test O+ Share |
     | EfficientNet-B0 | 0.1250 | 0.1905 | 0.3810 |
     | DenseNet121 | 0.3750 | 0.5238 | 0.5714 |
     ```
     Lines 27–29:
     ```markdown
     ## Best Architecture: EfficientNet-B0
     - Test Accuracy: 0.1905
     - Test Macro F1: 0.1250
     ```
   
   - **Backbone Selection Context:**
     Although EfficientNet-B0 won the validation benchmark (Val F1 0.5667 vs 0.3173), DenseNet-121 was selected as the base backbone for all subsequent experiments (Experiments 3, 4, 6, and E2). In `reports/phase4_research_audit/EXPERIMENT_HISTORY.md:17-18`, the text states:
     > *"DenseNet121 and EfficientNet emerged as the strongest candidates, but testing indicated massive overfitting. Verdict: DenseNet121 chosen as base."*
     
     The observation that EfficientNet-B0 collapsed on the locked test set (0.1250 test F1) while DenseNet-121 achieved 0.3750 test F1 is **consistent with locked-test results influencing the backbone choice; not established by any file.**

3. **Leakage Audit Invariance & Statistical Results:**  
   The core finding of the leakage audit remains completely sound: the audit itself did **not** perform any new model training, parameter fitting, or inference runs on the locked test cohort. The audit strictly loaded the saved prediction artifact `reports/final_evaluation/primary_evaluation_predictions.csv` to run permutation tests.
   
   As recorded in `experiments/leakage_audit/exp2_signal_tests_results.json`:
   - Observed accuracy: `0.3333` (7/21 correct)
   - Observed macro F1: `0.2083`
   - Permutation null p-value for accuracy: **`0.5641`** (`primary_test_21_participants.permutation_test.empirical_p_accuracy`)
   - Permutation null p-value for macro F1: **`0.3712`** (`primary_test_21_participants.permutation_test.empirical_p_macro_f1`)
   - Accuracy 95% bootstrap CI: **`[14.3%, 52.4%]`** (`primary_test_21_participants.bootstrap_ci_95.accuracy.ci_95`: `[0.1429, 0.5238]`)
   
   Therefore, candidate performance on the locked test set is **within the permutation null; 21 participants; CI [14.3%, 52.4%]**.

---

## 5. Corrected Formulation

The sentence in `reports/leakage_audit/leakage_audit_report.md` Section 1.1 should be understood as:

> *"No new inference or tuning was conducted on the locked MFS100 test cohort (21 participants) in this audit; the audit solely re-analyzed saved predictions from the October 16 candidate evaluation. As documented in `reports/project_report/locked_cohort_usage.csv`, the locked test cohort was represented across 17 prediction artifacts (corresponding to 10 distinct evaluation runs, 9 of which occurred between October 7 and October 13 across 10 distinct checkpoint paths before EXP-03, including Day 27, baseline architectures, E2, and the EfficientNet+E2 ensemble) prior to the selection and final evaluation of EXP-06C Config A."*
