# HemoPrint AI — Project Cleanup Execution Log

**Execution Timestamp:** 2026-10-18 14:46:11  
**Method:** Strict explicit allowlist execution (no broad recursive deletion).  
**Scope:** Removal of disposable Python bytecode caches (`__pycache__`), test execution runtime caches (`.pytest_cache`), transient CLI scratch files, test diagnostics dumps, and empty abandoned session folders.

---

## 1. Summary of Actions & Storage Impact

| Metric | Before Cleanup | After Cleanup | Net Difference (Freed) |
| :--- | :--- | :--- | :--- |
| **Total Project Size** | `21,450,975,674 bytes` (20,457.24 MB) | `21,450,063,233 bytes` (20,456.37 MB) | **`912,441 bytes` (891.06 KB)** |
| **Total Files** | `28,906 files` | `28,798 files` | **`108 files deleted`** |
| **Total Directories** | `971 directories` | `941 directories` | **`30 directories deleted`** |
| **Execution Status** | &mdash; | &mdash; | **100% SUCCESS (0 errors)** |

---

## 2. Itemized Deletion Register

All removals were performed against a verified explicit allowlist. No broad wildcard or unrestrained recursive commands were used.

### A. Specific Disposable Test & Scratch Files (5 files)
| Relative Path | Size | Reason for Removal | Status |
| :--- | :---: | :--- | :---: |
| `tests/dump_lines.txt` | 80,983 bytes | Ephemeral line-dump diagnostic file generated during manual test recovery in `scripts/recover_tests.py`. Not used by test runner or application. | **DELETED** |
| `scratch/b64_test.txt` | 11 bytes | Transient one-line CLI test file ("Hello World"). | **DELETED** |
| `scratch/test_b64.txt` | 11 bytes | Transient one-line CLI test file. | **DELETED** |
| `scratch/stress_test_inputs.txt` | 25 bytes | Transient one-line CLI stress test parameter string. | **DELETED** |
| `scratch/test_herestring.py` | 73 bytes | Transient two-line script testing quote preservation in PowerShell. | **DELETED** |

### B. Python Bytecode Cache Directories (22 `__pycache__` directories, 97 files)
| Directory Path | File Count | Size Freed | Status |
| :--- | :---: | :---: | :---: |
| `scripts/__pycache__` | 21 files | 208,439 bytes | **DELETED** |
| `tests/__pycache__` | 30 files | 131,063 bytes | **DELETED** |
| `scratch/__pycache__` | 10 files | 51,926 bytes | **DELETED** |
| `src/__pycache__` | 8 files | 48,152 bytes | **DELETED** |
| `scripts/calibration_experiments/__pycache__` | 1 files | 39,852 bytes | **DELETED** |
| `experiments/exp06d/__pycache__` | 1 files | 35,008 bytes | **DELETED** |
| `experiments/exp07_resolution_fidelity/__pycache__` | 4 files | 34,203 bytes | **DELETED** |
| `__pycache__` (root) | 4 files | 32,377 bytes | **DELETED** |
| `experiments/exp08_dermatoglyphic_minutiae/__pycache__` | 5 files | 31,509 bytes | **DELETED** |
| `experiments/__pycache__` | 1 files | 30,818 bytes | **DELETED** |
| `experiments/exp03/__pycache__` | 1 files | 27,072 bytes | **DELETED** |
| `experiments/exp06c/__pycache__` | 1 files | 26,400 bytes | **DELETED** |
| `experiments/exp06b/__pycache__` | 1 files | 26,057 bytes | **DELETED** |
| `experiments/exp04/__pycache__` | 1 files | 25,791 bytes | **DELETED** |
| `experiments/exp08_dermatoglyphic_minutiae/permutation_test/__pycache__` | 2 files | 17,141 bytes | **DELETED** |
| `scripts/abo_rh_experiments/__pycache__` | 1 files | 17,082 bytes | **DELETED** |
| `experiments/final_evaluation/__pycache__` | 1 files | 15,560 bytes | **DELETED** |
| `reports/project_report/__pycache__` | 1 files | 14,081 bytes | **DELETED** |
| `src/mfs100/__pycache__` | 3 files | 11,579 bytes | **DELETED** |
| `live_app/backend/__pycache__` | 1 files | 10,173 bytes | **DELETED** |
| `experiments/mfs100_domain_adaptation/__pycache__` | 2 files | 9,430 bytes | **DELETED** |
| `src/mfs100_bridge/__pycache__` | 3 files | 8,347 bytes | **DELETED** |

### C. Test Execution Cache Directory (1 directory, 6 files)
| Directory Path | File Count | Size Freed | Status |
| :--- | :---: | :---: | :---: |
| `.pytest_cache/` | 6 files | 11,282 bytes | **DELETED** |

### D. Empty Abandoned Session Directories (6 directories, 0 files)
| Directory Path | File Count | Size Freed | Status |
| :--- | :---: | :---: | :---: |
| `scratch/research_1b858ea0-2113-4ce1-8fe9-d3812cefae03` | 0 files | 0 bytes | **DELETED** |
| `scratch/research_492c16db-d318-40af-9e93-1365de14106d` | 0 files | 0 bytes | **DELETED** |
| `scratch/research_75da9ebf-1d4a-4aea-a123-21c5e3c9cb5e` | 0 files | 0 bytes | **DELETED** |
| `scratch/research_9267a0e0-651f-4c6e-86ce-b675d3565653` | 0 files | 0 bytes | **DELETED** |
| `scratch/research_bece8657-1131-4251-aca0-995b0a544a3e` | 0 files | 0 bytes | **DELETED** |
| `scratch/research_f33c6966-783b-4c15-8b1a-d30cc8f7cff7` | 0 files | 0 bytes | **DELETED** |

---

## 3. Preserved Entities Verification

Every mandatory preservation item under Phase 2 was verified intact post-cleanup:
1. `dataset_blood_group/`: 100% intact (6,000 canonical images preserved bit-for-bit).
2. `data/captured/` (1,773 files) and `data/mfs100_final/` (306 files): 100% intact.
3. `data/quarantine/` (N=21 and N=26 locked cohorts): Untouched ($0$ access).
4. `models/`: All `.keras` baseline and transfer learning checkpoints preserved.
5. `experiments/`: All checkpoints for EXP-03, EXP-04, EXP-05, and EXP-06a-d preserved.
6. `scratch/exp06d_init_shared_weights_*.json`: Preserved for EXP-06d reproducibility.
7. `scratch/exp05_synthetic/`: Preserved for parameter verification and dry-run testing.
8. `scratch/clean_estimate_cache/`: Preserved for clean permutation tests.
9. `scratch/research_*/` (116 active sessions): Preserved for session inspection.
10. `data/blood_group.db`: Preserved intact.
11. `reports/final_scientific_audit/`: 100% untouched.
12. Application source code (`app.py`, `src/`, `live_app/`): Preserved intact.
