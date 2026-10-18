# Second-Stage Storage Optimization Audit
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Audit Mode:** Strict Non-Destructive Audit (0 files modified, deleted, or relocated)  
**Total Current Repository Footprint:** **`21,450,063,233 bytes`** (**`19.98 GiB`**) across **`28,798 files`**

---

## Executive Summary

Following the initial phase cleanup which removed disposable bytecode caches and zero-byte crash folders (saving ~891 KB), a deep multi-phase physical audit was conducted across all 28,798 files in the repository to identify the root causes of the 19.98 GiB footprint.

### Key Discoveries
1. **The Primary Storage Driver is Deep Learning Weights (84.3%):**
   * **292 model checkpoint files** (`.keras`, `.weights.h5`, `.h5`) account for **`17.23 GiB`** of disk space.
   * **97 of the top 100 largest files in the repository are model checkpoints.**
2. **Verified Exact Duplication:**
   * A full cryptographic SHA-256 hash across all 292 checkpoints revealed **one exact duplicate pair**:
     * `models/mfs100_experiments/v1_1_checkpoints/epoch_22.keras` (92.18 MB) is byte-for-byte identical to `models/mfs100_experiments/resnet50_mfs100_abo_v1_1.keras` (`SHA-256: 45680aeb3a85baed8927707e3b87e571564877c8f3dc14b2dcb2dfbe19dc7af9`).
3. **Massive Intermediate Training Epoch Storage (2.70 GiB):**
   * The directory `models/mfs100_experiments/v1_1_checkpoints/` stores all 30 consecutive training epochs (`epoch_01.keras` through `epoch_30.keras`), totaling **`2,765.40 MB`** (`2.70 GiB`).
   * Epochs 01–21 and 23–30 have **0 references** in any application code, research script, or published audit report. Epoch 22 is already preserved under the canonical name `resnet50_mfs100_abo_v1_1.keras`.
4. **Git Object Packfiles (7.4%):**
   * The `.git/objects/pack/` directory stores **`1.51 GiB`** (principally `pack-eca7cb...pack` at 1.29 GB), representing historical Git commits that tracked binary models in earlier repository revisions.
5. **Database is Already 100% Compact (0.00% Waste):**
   * `data/blood_group.db` is only **`0.20 MB`** (208,896 bytes) containing 898 predictions, 26 captures, and 18 sessions.
   * `PRAGMA freelist_count` is **0** (0 free pages). Compacting or running `VACUUM` would save **0 bytes** and introduces severe OneDrive file-lock and sync corruption risks.
6. **Frontend Dependency Burden:**
   * `live_app/frontend/node_modules/` occupies **`127.47 MB`** (5,401 files). Rebuilding is 100% deterministic via the verified `package-lock.json`.

---

## 1. Storage Distribution Profile

```
+-----------------------------------------------------------------------------------+
| COMPONENT                                 SIZE          % OF REPO   CATEGORY      |
+-----------------------------------------------------------------------------------+
| Model Checkpoints (292 files)             17.23 GiB     84.3%       Research/ML   |
| Git Object Database (.git/pack/)           1.51 GiB      7.4%       VCS Metadata  |
| Datasets & Sensor Captures (data/ + raw)  565.00 MB      2.8%       Ground Truth  |
| Scratch Biometric Sessions & Vectors      488.38 MB      2.4%       Active Data   |
| Frontend Dependencies (node_modules)      127.47 MB      0.6%       Build Tooling |
| Research Reports & Audit Trails            58.17 MB      0.3%       Evidence/Docs |
| Output Tables, Plots & CSVs                28.33 MB      0.1%       Metrics/Plots |
| Operational Database (blood_group.db)       0.20 MB     <0.01%      Relational DB |
+-----------------------------------------------------------------------------------+
| TOTAL REPOSITORY FOOTPRINT                19.98 GiB    100.0%                     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Model Checkpoint Audit (models/ & experiments/)

All 292 model checkpoints were cataloged, categorized by experimental role, and cross-referenced against all source code, reproduction scripts, and published reports.

### Checkpoint Role Taxonomy

| Role | Count | Total Size | Status & Reproducibility Impact |
| :--- | :---: | :---: | :--- |
| **Active Inference Models** | **16** | **`542.42 MB`** | **CRITICAL.** Directly imported and loaded by `live_app/backend/main.py` for live inference, architecture benchmarking, and preprocessing ablation. **Must never be touched.** |
| **Best-Performing & Fold Checkpoints** | **229** | **`12.77 GiB`** | **HIGH.** K-fold cross validation weights (Fold 1–5 for VGG16, ResNet50, ConvNeXt, DenseNet, EfficientNet across transfer learning tiers T1–T4 and experiments Exp01–Exp10). Essential for recreating publication tables. |
| **Intermediate Epoch Checkpoints** | **30** | **`2.70 GiB`** | **LOW.** Per-epoch dumps (`epoch_01.keras` to `epoch_30.keras` in `mfs100_experiments`). Only epoch 22 was selected. 29 epochs are completely unreferenced. |
| **Historical Baselines & Ablations** | **16** | **`1.19 GiB`** | **HIGH.** Multi-task ABO/Rh models, two-stage classifiers, and domain adaptation baselines cited in final audit reports. |
| **CV Fold Weight Vectors** | **1** | **`0.04 GiB`** | **MODERATE.** Shared weight vectors for fold initialization reproducibility. |
| **Total** | **292** | **`17.23 GiB`** | Full inventory detailed in [`MODEL_CHECKPOINT_AUDIT.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/MODEL_CHECKPOINT_AUDIT.csv). |

### Active Inference Models in `live_app/backend/main.py`
The following 16 models are strictly required to operate the running web application:
1. `models/abo_positive_4class/best_model.keras` (31.13 MB)
2. `models/day27_full_dataset/eight_class/best_model.keras` (5.32 MB)
3. `models/day27_full_dataset/four_class/best_model.keras` (5.30 MB)
4. `models/phase3_architecture_benchmark/convnext_tiny/best_model.keras` (106.67 MB)
5. `models/phase3_architecture_benchmark/custom_cnn/best_model.keras` (4.56 MB)
6. `models/phase3_architecture_benchmark/densenet121/best_model.keras` (28.35 MB)
7. `models/phase3_architecture_benchmark/efficientnetv2_b0/best_model.keras` (23.56 MB)
8. `models/phase3_architecture_benchmark/efficientnet_b0/best_model.keras` (16.32 MB)
9. `models/phase3_architecture_benchmark/resnet50/best_model.keras` (90.71 MB)
10. `models/phase3_architecture_benchmark/vgg16/best_model.keras` (56.25 MB)
11. `models/phase3_preprocessing/P0_C5/best_model.keras` (28.35 MB)
12. `models/phase3_preprocessing/P1_GRAYSCALE/best_model.keras` (28.35 MB)
13. `models/phase3_preprocessing/P2_CLAHE/best_model.keras` (28.35 MB)
14. `models/phase3_preprocessing/P3_GABOR/best_model.keras` (28.35 MB)
15. `models/phase3_preprocessing/P4_DENOISE_CONTRAST/best_model.keras` (28.35 MB)
16. `models/phase3_preprocessing/P5_GRAYSCALE_CLAHE/best_model.keras` (28.35 MB)

---

## 3. Experiment Output Audit (experiments/ and outputs/)

Inspection of `experiments/` and `outputs/` identified **412 files** totaling **`6.23 GiB`**:
* **Checkpoints inside Experiments:** 115 files (**`6,314.12 MB`** / `6.17 GiB`), including fold checkpoints for Exp01, Exp02, Exp04, Exp06, and Exp07.
* **Evaluation Metrics & Predictions:** 44 files (**`1.93 MB`**), including `.npy` arrays, `.csv` prediction matrices, and `.json` test evaluations. All are directly cited in the final scientific audit and manuscript results.
* **Plots & Confusion Matrices:** 91 files (**`22.75 MB`**), including ROC curves, PR curves, and confusion matrix visualizations.
* **Training Logs & Artifacts:** 162 files (**`45.72 MB`**), including training histories, loss curves, and fold split definitions.

*Conclusion:* Experiment outputs outside model checkpoints take only **`70.4 MB`**. Deleting plots or prediction files provides almost zero storage benefit while destroying the empirical evidence needed to substantiate the research paper. **All experiment plots and metric tables must be preserved.**

---

## 4. Database Storage & Compaction Audit

An exhaustive physical inspection of `data/blood_group.db` showed:
* File size: `208,896 bytes` (`0.20 MB`).
* Total pages: 51 pages (4,096 bytes per page).
* Freelist pages: **0 pages** (0 bytes).
* **Compaction Verdict:** There is zero unused or free space inside the database. Compaction (`VACUUM`) yields **0 bytes of space savings**, but introduces severe operational hazards:
  * OneDrive lock conflicts (`[WinError 5] Access is denied`).
  * Process contention if the backend daemon is querying the DB.
  * Risk of participant record corruption during write replacement.
* Full schema and verification logs are recorded in [`DATABASE_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/DATABASE_STORAGE_AUDIT.md).

---

## 5. Frontend Storage Audit

An inspection of `live_app/frontend/` showed:
* `node_modules/`: **`127.47 MB`** across 5,401 files.
* Production bundle (`dist/`): **`0.38 MB`** across 5 files.
* Vite cache (`.vite`): **`7.06 MB`**.
* Source files (`src/`): **`0.17 MB`** across 20 files.
* Lockfile integrity: [`package-lock.json`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/live_app/frontend/package-lock.json) is intact.
* Rebuild verification: The production bundle compiles cleanly from source in **`1.52 seconds`**.
* Detailed analysis is documented in [`FRONTEND_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/FRONTEND_STORAGE_AUDIT.md).

---

## 6. Publication Package Strategy (Separating Monorepo from Release)

The existing repository operates as an **all-inclusive scientific monorepo** containing training lineage, exploratory experiments, Git history, local UI dependencies, and raw datasets (~20 GiB).

To prepare for research paper submission (e.g., IEEE / Springer / Nature Scientific Data / Zenodo), the project should NOT attempt to crush all historical evidence into a degraded monorepo. Instead, a clean **Research Publication Package** architecture should be established:

```
+-----------------------------------------------------------------------------------------+
| RESEARCH MONOREPO (Internal Archive)             PUBLIC REPLICATION BUNDLE (Release)    |
| Size: ~20 GiB (Preserved on Private Drive)       Target Size: ~350 MB to 1.1 GB         |
+-----------------------------------------------------------------------------------------+
| [x] Full Git history (.git/ 1.51 GB)             [ ] Excluded (.git pruned)             |
| [x] All 292 checkpoints (17.23 GB)              [ ] Only Best Models & Baselines (~542MB)|
| [x] 30 intermediate epochs (2.70 GB)             [ ] Excluded (Redundant epochs)        |
| [x] Raw biometric captures (scratch/ 488 MB)     [ ] Excluded (Privacy & IRB compliance)|
| [x] Quarantined cohorts (data/quarantine)        [ ] Excluded (Protected test sets)     |
| [x] Node modules (127 MB)                        [ ] Excluded (Use package-lock.json)   |
| [x] All reproduction scripts & notebooks         [x] Included (Complete pipeline)       |
| [x] Canonical dataset (234 MB)                   [x] Included or Zenodo DOI link        |
| [x] Final scientific audit reports               [x] Included (reports/final_audit/)    |
| [x] Evaluation metrics, plots & confusion matrices[x] Included (outputs/ & experiments/)  |
+-----------------------------------------------------------------------------------------+
```

### Ethical & Privacy Constraints
* **Participant Fingerprint Images:** Biometric samples in `scratch/research_*/` and `data/captured/` must **not** be included in an unrestricted public zip without documented participant informed consent, Institutional Review Board (IRB) approval, and de-identification.
* **Quarantined Test Cohorts:** Test cohorts ($N=21$ and hospital $N=26$) must remain isolated to prevent future data snooping or post-hoc over-fitting.

---

## 7. Safest Potential Storage Savings (Ranked by Risk)

If you decide in a subsequent phase to reclaim storage on your local disk, the following table presents an objectively ranked list based on risk, reproducibility, and application impact:

| Rank | Target Candidate Path / Directory | Reclaimable Size | Risk Level | Evidence & Justification | Impact on Live App | Impact on Paper Reproducibility |
| :---: | :--- | :---: | :---: | :--- | :--- | :--- |
| **1** | `models/mfs100_experiments/v1_1_checkpoints/epoch_22.keras` | **`92.18 MB`** | **Zero Risk** | **Verified 100% Duplicate.** SHA-256 hash `45680aeb3a85...` matches `models/mfs100_experiments/resnet50_mfs100_abo_v1_1.keras` bit-for-bit. | None. Unreferenced in app. | None. Canonical model already preserved under target name. |
| **2** | `models/mfs100_experiments/v1_1_checkpoints/` (remaining 29 unreferenced epochs: `epoch_01` to `epoch_21`, `epoch_23` to `epoch_30`) | **`2,673.22 MB`** (`~2.61 GiB`) | **Very Low** | Intermediate training iterations. 0 references across all scripts, reports, and code. Final weights and metrics are logged. | None. Not used by backend or frontend. | Low. Epoch curves are recorded in training history JSON. Any individual epoch can be re-trained from script. |
| **3** | `live_app/frontend/dist/` | **`0.38 MB`** | **Very Low** | Compiled frontend assets. Can be rebuilt from source via `npm run build` in 1.52 seconds. | None for dev server (`npm run dev`). | None. Frontend UI is operational tooling, not experimental evidence. |
| **4** | `live_app/frontend/node_modules/` | **`127.47 MB`** | **Low (Dev Impact)** | 5,401 third-party Node packages. Fully locked by `package-lock.json`. Reinstallable via `npm ci`. | Dev server will not run until `npm ci` is executed. | None. Standard software dependency management practice. |
| **5** | Historical Git Packfile Optimization (`git gc --aggressive --prune=now`) | **`~400–800 MB`** (Estimated) | **Moderate** | Compacts loose git objects and packs. Modifies `.git/` internal pack structure without changing commit history. | None. | None. Git commit hashes remain identical. |
| **6** | Fold Checkpoints of Superseded Benchmark Architectures in `models/phase3_transfer_learning/` | **`~3.5–5.0 GiB`** | **High** | Multi-gigabyte VGG16/ResNet intermediate fold checkpoints. If retained on an external archive, local working tree is slimmed down. | None if active models are retained. | High if external archive is lost. Required for full K-fold table recalculation. |

---

## 8. Master Inventory Reports Created

All structured data generated during this audit has been formatted and committed to [`reports/project_cleanup/`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup):
1. [`SECOND_STAGE_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/SECOND_STAGE_STORAGE_AUDIT.md) — This master optimization synthesis and architectural roadmap.
2. [`LARGE_FILES_INVENTORY.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/LARGE_FILES_INVENTORY.csv) — Complete sorted audit of the top 100 largest files with references, categories, and purposes.
3. [`MODEL_CHECKPOINT_AUDIT.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/MODEL_CHECKPOINT_AUDIT.csv) — Audit of all 292 model checkpoints, SHA-256 hashes, duplicate flags, roles, and retention recommendations.
4. [`EXPERIMENT_OUTPUT_AUDIT.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/EXPERIMENT_OUTPUT_AUDIT.csv) — Audit of all 412 experiment files across `experiments/` and `outputs/`.
5. [`DATABASE_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/DATABASE_STORAGE_AUDIT.md) — Deep SQLite physical geometry, freelist evaluation, and compaction risk assessment.
6. [`FRONTEND_STORAGE_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/project_cleanup/FRONTEND_STORAGE_AUDIT.md) — Node modules, build bundles, Vite cache, and packaging advice.
