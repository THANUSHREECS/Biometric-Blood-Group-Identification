# HemoPrint AI — Comprehensive Project Storage Audit

**Audit Date:** October 18, 2026  
**Repository:** HemoPrint AI (`major project`)  
**Scope:** Complete recursive filesystem audit across all project folders, datasets, models, scripts, dependencies, and reports.

---

## 1. Executive Summary & Storage Totals

| Metric | Measured Value | Human-Readable |
| :--- | :--- | :--- |
| **Total Project Size** | `21,450,975,674 bytes` | **20.46 GB** (19.98 GiB) |
| **Total File Count** | `28,906 files` | 28,906 |
| **Total Directory Count**| `971 directories` | 971 |
| **Model Weights & Checkpoints** | `17,761,905,042 bytes` | **16.94 GB** (82.8% of total storage) |
| **Biometric Datasets & Scans** | `1,129,750,407 bytes` | **1.05 GB** (5.3% of total storage) |
| **Git Version Control History** | `1,581,490,311 bytes` | **1.47 GB** (7.4% of total storage) |
| **Scratch & Session Data** | `512,104,773 bytes` | **488.38 MB** (2.4% of total storage) |
| **Frontend Dependencies (`node_modules`)** | `133,660,672 bytes` | **127.47 MB** (0.6% of total storage) |
| **Evaluation Reports & Audits** | `60,992,510 bytes` | **58.17 MB** (0.3% of total storage) |
| **Code, Templates & Configurations** | `4,071,959 bytes` | **3.88 MB** (<0.1% of total storage) |

---

## 2. Directory Breakdown (Descending by Storage Usage)

| Directory Path | File Count | Size (MB) | Size (Bytes) | % of Project | Primary Contents |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`models/`** | 108 | 11,086.31 MB | 11,624,843,380 | 54.19% | Pretrained CNNs (ResNet-50, VGG-16, DenseNet-121, ConvNeXt, EfficientNet-B0), 5-fold cross-validation checkpoints |
| **`experiments/`** | 185 | 6,356.45 MB | 6,665,218,662 | 31.07% | EXP-01 through EXP-10 code, logs, and fold checkpoints (`exp03`, `exp04`, `exp05`, `exp06a-d`) |
| **`.git/`** | 99 | 1,508.23 MB | 1,581,490,311 | 7.37% | Git commit history, tree objects, packfiles |
| **`data/`** | 14,085 | 564.51 MB | 591,928,807 | 2.76% | `captured/` (118.7 MB), `mfs100_final/` (32.7 MB), `processed/` (176.9 MB), `raw/` (234.5 MB), `blood_group.db` |
| **`scratch/`** | 553 | 488.38 MB | 512,104,773 | 2.39% | Active session captures (`research_*`), EXP-06d shared initial weights, EXP-05 synthetic test suites, scratch utilities |
| **`dataset_blood_group/`** | 6,000 | 234.50 MB | 245,892,800 | 1.15% | Original canonical benchmark dataset (6,000 8-bit bitmap fingerprint images across 8 blood groups) |
| **`live_app/`** | 6,661 | 128.11 MB | 134,335,695 | 0.63% | React 19 + TypeScript frontend application, Vite tooling, `node_modules` (127.47 MB), `dist` build (0.38 MB) |
| **`reports/`** | 988 | 58.17 MB | 60,992,510 | 0.28% | Scientific audits, EXP-01–EXP-10 reports, 871 participant PDF examination records, `final_scientific_audit/` |
| **`outputs/`** | 205 | 28.33 MB | 29,709,122 | 0.14% | Domain analysis, confusion matrices, ROC curves, calibration outputs, model performance metrics |
| **`scripts/`** | 100 | 1.49 MB | 1,561,833 | 0.01% | Reproducibility scripts, data pipelines, model evaluation, leakage audit verification |
| **`tests/`** | 51 | 0.29 MB | 300,113 | <0.01% | Integration tests, unit tests, protocol verification tests |
| **`src/`** | 23 | 0.15 MB | 157,400 | <0.01% | Flask backend core, feature extraction, predictor adapter, database models, MFS100 bridge integration |
| **`docs/`** | 10 | 0.35 MB | 370,613 | <0.01% | Project architecture documentation and specifications |
| **`templates/` & `static/`**| 13 | 0.10 MB | 100,064 | <0.01% | Legacy HTML/JS interfaces |
| **Project Root (`.`)** | 25 | 1.28 MB | 1,342,617 | <0.01% | `app.py`, `README.md`, `requirements.txt`, root environment files |

---

## 3. Largest Individual Files (>50 MB)

| File Path | Size (MB) | Purpose & Preservation Status |
| :--- | :---: | :--- |
| `.git/objects/pack/pack-eca7cb6...pack` | 1,294.84 MB | Git repository packfile (**Protected** — VCS integrity) |
| `models/experiments/combined_ensemble_best.keras` | 421.63 MB | Ensemble baseline checkpoint (**Protected** — Benchmark baseline) |
| `models/resnet50_improved_blood_group.keras` | 264.84 MB | Pretrained ResNet-50 improved model (**Protected** — Model artifact) |
| `models/experiments/resnet50_long_epoch_best.keras` | 264.84 MB | Long-epoch ResNet-50 benchmark (**Protected** — Reproducibility) |
| `models/blood_group_resnet50.keras` | 210.74 MB | Baseline ResNet-50 model (**Protected** — Canonical baseline) |
| `models/phase3_transfer_learning/VGG16/T4/fold_1_best.keras` | 168.53 MB | VGG-16 T4 Fold 1 checkpoint (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T4/fold_2_best.keras` | 168.53 MB | VGG-16 T4 Fold 2 checkpoint (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T4/fold_3_best.keras` | 168.53 MB | VGG-16 T4 Fold 3 checkpoint (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T4/fold_4_best.keras` | 168.53 MB | VGG-16 T4 Fold 4 checkpoint (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T4/fold_5_best.keras` | 168.53 MB | VGG-16 T4 Fold 5 checkpoint (**Protected** — Phase 3 experimental evidence) |
| `models/vgg16_blood_group.keras` | 156.78 MB | Baseline VGG-16 model (**Protected** — Canonical baseline) |
| `models/experiments/vgg16_long_epoch_best.keras` | 156.78 MB | VGG-16 long epoch model (**Protected** — Experimental baseline) |
| `experiments/exp04/checkpoints/config_c/fold_1_best.weights.h5` | 120.49 MB | EXP-04 Config C Fold 1 weights (**Protected** — Multi-scan experimental evidence) |
| `experiments/exp04/checkpoints/config_c/fold_2_best.weights.h5` | 120.49 MB | EXP-04 Config C Fold 2 weights (**Protected** — Multi-scan experimental evidence) |
| `experiments/exp04/checkpoints/config_c/fold_3_best.weights.h5` | 120.49 MB | EXP-04 Config C Fold 3 weights (**Protected** — Multi-scan experimental evidence) |
| `experiments/exp04/checkpoints/config_c/fold_4_best.weights.h5` | 120.49 MB | EXP-04 Config C Fold 4 weights (**Protected** — Multi-scan experimental evidence) |
| `experiments/exp04/checkpoints/config_c/fold_5_best.weights.h5` | 120.49 MB | EXP-04 Config C Fold 5 weights (**Protected** — Multi-scan experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T2/fold_1..5_best.keras` | 110.27 MB ea. | VGG-16 T2 5-fold models (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_transfer_learning/VGG16/T3/fold_1..5_best.keras` | 110.27 MB ea. | VGG-16 T3 5-fold models (**Protected** — Phase 3 experimental evidence) |
| `models/phase3_architecture_benchmark/convnext_tiny/best_model.keras` | 106.67 MB | ConvNeXt-Tiny benchmark (**Protected** — Architecture audit) |
| `.git/objects/pack/pack-c70c361...pack` | 105.66 MB | Git repository packfile (**Protected** — VCS integrity) |
| `models/abo_rh_experiments/two_stage_abo_rh_best_val.keras` | 97.20 MB | Two-stage ABO/Rh model (**Protected** — Experimental evidence) |

---

## 4. Potentially Removable vs. Preserved Storage Analysis

The vast majority (99.99%) of project storage consists of irreplaceable scientific artifacts:
1. **Model Weights & Experimental Checkpoints (16.94 GB):** Under Phase 2 Rule 4, all weights and models must be preserved bit-for-bit to guarantee reproducibility of all academic claims in the research publication.
2. **Datasets & Biometric Captures (1.05 GB):** Under Phase 2 Rules 1 & 2, raw and processed biometric files must never be removed or modified.
3. **Git History (1.47 GB):** Version control history establishes tamper-proof provenance across all project phases.
4. **Research Reports (58.17 MB):** Preserved per Rules 6 & 8.
5. **Disposable Cache & Temporary Files (911.81 KB):** Python bytecode (`.pyc`), Pytest runtime caches, and transient CLI text tests are completely safe to remove.
