# HemoPrint AI — Preserved Research Assets Register

**Audit Date:** October 18, 2026  
**Status:** Frozen & Protected  
**Mandate:** Strict adherence to Phase 2 Preservation Rules to ensure complete academic reproducibility and publication integrity.

---

## 1. Biometric Datasets & Fingerprint Captures (Rules 1 & 2)

| Asset Group | Directory / File Path | Item Count | Storage Size | Preservation Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Canonical Benchmark Dataset** | `dataset_blood_group/` | 6,000 files | 234.50 MB | The original 8-class optical fingerprint dataset (A+, A-, B+, B-, AB+, AB-, O+, O-). Kept bit-for-bit unchanged. |
| **Raw Repository Mirror** | `data/raw/` | 6,000 files | 234.50 MB | Working mirror of canonical BMP files mapped to metadata. |
| **Processed Biometric Cache** | `data/processed/` | 5,985 files | 176.90 MB | Preprocessed 112×112 CLAHE/bilateral-filtered impressions used across experiments. |
| **MFS100 Captured Impressions**| `data/captured/` | 1,773 files | 118.70 MB | Live and benchmark 500 DPI optical captures from physical Mantra MFS100 scanner sessions. |
| **MFS100 Final Cohort** | `data/mfs100_final/` | 306 files | 32.72 MB | Standardized 3-impression Left Thumb research cohort (N=102 subjects). |
| **Quarantined Test Cohorts** | `data/quarantine/` | 16 files | 1.08 MB | Strictly locked external evaluation splits (N=21 held-out test cohort; N=26 hospital cohort). Untouched ($0$ access). |
| **Reference Sample Fingerprints**| `data/sample/` | 8 files | 0.40 MB | Verified canonical sample impressions (A+, B+, AB+, O+) used for scanner demo and initialization. |
| **Live Session Captures** | `scratch/research_*/` | 116 folders | 35.58 MB | Real-time session capture directories containing scan BMPs and prediction metadata. |

---

## 2. Model Checkpoints & Neural Network Weights (Rule 4)

| Model Family / Experiment | Location | Checkpoint Format | Size | Preservation Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **EXP-06C Active Model** | `experiments/exp06c/checkpoints/` | `.weights.h5` | 575.85 MB | Primary 5-fold consensus model for live application inference (Config A frozen DenseNet-121 + linear head). |
| **EXP-06D Head Initialization** | `scratch/exp06d_init_shared_weights_*.json` | `.json` (tensors) | 208.45 MB | Verified shared initial weights required by `experiments/exp06d/run_exp06d.py:654`. |
| **EXP-06A & EXP-06B Ensembles** | `experiments/exp06a/`, `exp06b/` | `.weights.h5` | 2,018.03 MB | Baseline and augmentation ablation models establishing negative empirical results. |
| **EXP-04 & EXP-05 Multi-Scan Models** | `experiments/exp04/`, `exp05/` | `.weights.h5` | 1,954.55 MB | Multi-scan concatenation and supervised contrastive pretraining architectures. |
| **EXP-03 Pretrained Checkpoints**| `experiments/exp03/` | `.weights.h5` | 756.90 MB | Supervised contrastive feature extractor checkpoints. |
| **Phase 3 Transfer Learning** | `models/phase3_transfer_learning/` | `.keras` | ~4,200 MB | VGG-16, ResNet-50, and DenseNet-121 fold models establishing cross-fold stability. |
| **Standalone Benchmark Baselines**| `models/*.keras` | `.keras` | ~6,800 MB | ResNet-50, VGG-16, EfficientNet-B0, and ConvNeXt reference baselines. |
| **EXP-05 Synthetic Validation** | `scratch/exp05_synthetic/` | `.weights.h5` | 198.38 MB | Documented parameter count and contrastive loss safety test suites. |
| **Audit Checkpoints** | `scratch/test_init_fold_1.weights.h5` | `.weights.h5` | 42.93 MB | Preserved under Rule 4 strict non-deletion mandate. |

---

## 3. Databases & Research Session Records (Rule 9)

| Database Asset | File Path | Records Contained | Preservation Rationale |
| :--- | :--- | :---: | :--- |
| **HemoPrint SQLite Database** | `data/blood_group.db` | Participants, Sessions, Impressions, Inference Logs | Primary persistent datastore recording pseudonymous participant registrations, informed consents, MFS100 quality scores, and predictions. |

---

## 4. Evaluation Outputs, Provenance Records & Metrics (Rule 5)

| Category | Locations | Key Artifacts |
| :--- | :--- | :--- |
| **Experimental Metrics** | `outputs/`, `results/` | `evaluation_results.json`, `dataset_manifest.csv`, `dataset_report.json`, confusion matrices, ROC plots |
| **Leakage & Domain Audits** | `experiments/leakage_audit/` | Feature caches (`dev_densenet121_features.npz`), perceptual hashes, permutation test distributions |
| **Participant PDF Reports** | `reports/blood_group_report_*.pdf` | 871 individual participant audit PDFs generated during research sessions (47.70 MB) |
| **EXP-07–EXP-10 Evidence** | `experiments/exp07..10/`, `reports/` | Resolution fidelity, minutiae ridge-density, orientation tensors, and multi-scan consistency reports |

---

## 5. Audit Artifacts & Research Reports (Rules 6 & 8)

| Directory | Core Purpose & Contents |
| :--- | :--- |
| **`reports/final_scientific_audit/`** | Strictly preserved final academic evidence audit, issue registers, metric reconciliations, and execution logs. |
| **`reports/final_evaluation/`** | Primary evaluation report and readiness audit. |
| **`reports/accuracy_improvement/`** | Detailed preparation, execution, and statistical validation reports for Experiments 01 through 06d. |
| **`reports/leakage_audit/`** | Forensic evidence documenting scanner platen brightness artifacts and scan leakage. |

---

## 6. Reproduction Scripts & Notebooks (Rule 7)

| Script Directory | Purpose |
| :--- | :--- |
| **`scripts/`** (100 scripts) | End-to-end evaluation, data preparation, calibration experiments, leakage audits, and cross-validation runners. |
| **`experiments/`** (185 scripts/configs)| Experiment runners for EXP-01 through EXP-10. |
| **`scratch/*.py`** (92 scripts) | Documented verification scripts referenced in reports (`clean_estimate`, `pairwise_comparison`, etc.). |

---

## 7. Application Code & Dependency Lockfiles (Rules 10 & 11)

| Asset | Location | Purpose |
| :--- | :--- | :--- |
| **Flask Backend Application** | `app.py`, `src/` | REST API, predictor service, MFS100 scanner integration, image quality evaluation. |
| **React + TypeScript Frontend** | `live_app/frontend/src/` | 9 full-screen responsive views, navigation, API client, design system. |
| **MFS100 32-bit Bridge** | `src/mfs100_bridge/bridge.py` | Out-of-process 32-bit driver translation service. |
| **Dependency Declarations** | `requirements.txt`, `package.json`, `package-lock.json` | Exact pinned environment definitions for reproducible builds. |
| **Version Control Metadata** | `.git/` | Full repository commit history and provenance tree. |
