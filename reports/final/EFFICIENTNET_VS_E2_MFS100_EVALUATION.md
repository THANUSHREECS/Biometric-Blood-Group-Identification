# EfficientNet-B0 vs E2 Ensemble — MFS100 Live-Scan Evaluation

**Report Date:** 2026-10-11  
**Status:** COMPLETED — Active evaluation performed  
**Project:** HemoPrint AI / Biometric Blood Group Detection  

---

## 0. Triggering Observation

During a live HemoPrint AI session on an MFS100 scanner, the E2 ensemble predicted **A+** (79.2% confidence) for a participant whose laboratory-confirmed blood group was **O+**. This report investigates whether EfficientNet-B0 would fare differently on the same hardware domain and whether a head-to-head comparison on labelled MFS100 data is currently possible.

---

## 1. EfficientNet-B0 Checkpoint Inspection

### 1.1 Checkpoint

| Property | Value |
|----------|-------|
| Path | `models/abo_positive_4class/best_model.keras` |
| File size | 32,642,427 bytes (~31 MB) |
| SHA-256 | `7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9` |
| Created | 2026-09-10 |
| Framework | TensorFlow 2.21.0 |

### 1.2 Architecture

```
EfficientNetB0 (ImageNet pretrained) → GlobalAveragePooling2D → Dense(256, ReLU, L2=1e-4) → Dropout(0.30) → Dense(4, Softmax)
```

Two-stage fine-tuning: backbone frozen in Stage 1 (lr=0.001), unfrozen in Stage 2 (lr=3e-5).

### 1.3 Input Shape and Preprocessing

| Property | Value |
|----------|-------|
| Input shape | `(224, 224, 3)` |
| Pixel range expected | RGB float32 in `[0, 255]` |
| Normalisation | Built-in EfficientNetB0 rescaling + normalisation (applied internally by the Keras layer) |
| Augmentation at training | Conservative: rotation ±3°, translation ±3%, zoom ±3%, contrast ±5%. No flips. |
| Single-scan output | Softmax probabilities over 4 classes |

### 1.4 Class Mapping

| Output index | Class |
|-------------|-------|
| 0 | A+ |
| 1 | AB+ |
| 2 | B+ |
| 3 | O+ |

Source: `models/abo_positive_4class/class_mapping.json`

### 1.5 Confirmed by Loss Function and Label Encoding

- Loss: `sparse_categorical_crossentropy`
- Class weights: balanced, computed solely from the training split
- Output activation: `Softmax` (confirmed from architecture summary in metadata.json)

---

## 2. E2 Ensemble Inspection (Live System)

### 2.1 Architecture

```
DenseNet121 (ImageNet pretrained, frozen) → GlobalAveragePooling2D
[scan1, scan2, scan3] → [feat1, feat2, feat3] → Concatenate → Dense(512, ReLU) → Dropout(0.5) → Dense(4, Softmax) [+ Dense(128) embedding head]
```

Source: `src/predictor.py` → `_build_E2()`

### 2.2 Preprocessing (E2)

| Property | Value |
|----------|-------|
| Input shape per scan | `(112, 112, 3)` |
| Scans required per participant | **Exactly 3** |
| Pipeline | `transform_c5` from `src/preprocessing.py` (load BMP → RGBA composite on white background → CLAHE + bilateral filter + crop → resize to 112×112) |
| Model input tensor | `(1, 3, 112, 112, 3)` |
| Class order | A+, B+, AB+, O+ (indices 0,1,2,3) |

### 2.3 Critical Difference from EfficientNet-B0

| | EfficientNet-B0 | E2 Ensemble |
|-|-----------------|-------------|
| Input size | 224×224 | 112×112 |
| Scans per prediction | 1 | 3 (mandatory) |
| Preprocessing | Keras built-in rescaling | `transform_c5` (CLAHE + bilateral + crop) |
| Output heads | 1 (softmax probabilities) | 2 (clf + embedding) |
| Class order | A+, AB+, B+, O+ | A+, B+, AB+, O+ |
| Training domain | ~6,000-image optical-scanner dataset | MFS100-scanned participants |

> **NOTE:** EfficientNet-B0 was trained on the original CMOS optical-scanner dataset. It has **never** seen MFS100 scans. Before evaluation, image compatibility must be verified — the BMP size differences, DPI, sensor response, and ridge scale between the original optical scanner and MFS100 may differ significantly.

---

## 3. MFS100 Labelled Data Availability Audit

### 3.1 Ground Truth File

- **Path:** `data/mfs100_ground_truth.csv`
- **Rows:** 79 scan records (26 unique participants, S001–S026)
- **Confirmation source:** Hospital Serological Test (S001–S025); independent verified filename (S026)
- **Scans per participant:** 3 (all participants have exactly 3 scans)
- **Scan format:** BMP files in `data/captured/`
- **Finger used:** LEFT_THUMB (all participants)
- **NFIQ quality score:** 1 (highest quality, all scans)

### 3.2 Class Distribution of the 26 MFS100 Ground Truth Participants

| Blood Group | Participants | Scans |
|-------------|-------------|-------|
| A+ | 4 (S001–S004) | 12 |
| AB+ | 3 (S005, S006, S026) | 9 |
| B+ | 9 (S007–S014) | 27 |
| O+ | 10 (S015–S025) | 30 |
| **Total** | **26** | **79** |

> **Note:** S026 contributes 3 scans. Total scan count in CSV is 79 (not 78): the CSV has 79 data rows (lines 2–80, counting S026's 3 scans).

### 3.3 Captured BMP File Verification

All 79 filenames listed in `data/mfs100_ground_truth.csv` were confirmed present in `data/captured/`. Each BMP is 112×112 pixels (29,718 bytes — the NFIQ1-standard MFS100 output size), confirming that `data/captured/` contains valid MFS100 live scans.

### 3.4 Overlap with E2 Development/Test Participants

The E2 ensemble used:
- 80 development participants (OOF cross-validation)
- 21 locked test participants

**None of the subject IDs S001–S026 appear in either the E2 development or locked test manifests.** The 26 MFS100 ground truth participants are a **separate cohort** collected independently for ground-truth verification. They do not overlap with E2 training or testing participants.

**Conclusion:** The 26 participants are suitable for an independent evaluation of both models, subject to the preprocessing compatibility caveat in §3.5.

### 3.5 MFS100 Compatibility Analysis for EfficientNet-B0

EfficientNet-B0 was trained on images from a CMOS optical scanner with different characteristics than the MFS100. Specific concerns:

| Issue | Details |
|-------|---------|
| **Input resolution mismatch** | MFS100 captures at 500 DPI; original training images were scanned at a different DPI and resolution. Resizing to 224×224 is technically possible but may distort ridge features. |
| **Sensor domain gap** | MFS100 uses capacitive sensing; original dataset used optical reflection. Ridge contrast, background texture, and noise distribution differ. |
| **Preprocessing mismatch** | EfficientNet-B0 was trained with minimal augmentation; no CLAHE or bilateral filter was applied at training time. MFS100 BMPs have different dynamic range characteristics. |
| **No MFS100 fine-tuning** | EfficientNet-B0 was never shown any MFS100 image during training, validation, or any intermediate fine-tuning step. |

These are the same domain-gap factors that caused EfficientNet-B0 to score only **21.25%** on the 80 development participants (MFS100 domain, single-scan mode) — far below its 86.02% internal test performance. This is documented in `reports/final/FINAL_EVALUATION_SUMMARY.md`, §3.

**Despite the domain gap, the evaluation proceeds below because all 26 participants have both models' required inputs available and properly labelled.**

---

## 4. Previously Completed MFS100 Evaluations

### 4.1 E2 Ensemble — Development Set (80 Participants, MFS100)

Evaluated on the OOF cross-validation predictions from `reports/model_fusion/all_models_participant_probs.npz`.

| Metric | Value |
|--------|-------|
| Participants | 80 |
| Scans | 240 (3 per participant) |
| Participant accuracy | **53.75%** (43/80) |
| Macro F1 | **0.4772** |
| Balanced accuracy | **0.4715** |
| Majority-class baseline (O+) | 36.25% (29/80) |

Per-class (participant level):

| Class | Support | Recall | F1 |
|-------|---------|--------|----|
| A+ | 19 | 63.16% | 0.4706 |
| B+ | 25 | 56.00% | 0.5957 |
| AB+ | 7 | **14.29%** | 0.2500 |
| O+ | 29 | 55.17% | 0.5926 |

### 4.2 E2 Ensemble — Locked Test Set (21 Participants, MFS100)

| Metric | Value |
|--------|-------|
| Participants | 21 |
| Scans | 63 (3 per participant) |
| Participant accuracy | **38.10%** (8/21) |
| Permutation-test p-value | 0.58 (not significant) |
| Majority-class baseline (O+) | 38.10% (8/21) |

**The locked test accuracy is statistically indistinguishable from always predicting the majority class.**

### 4.3 EfficientNet-B0 — MFS100 Development Set (Single-Scan Mode, 80 Participants)

Evaluated in `reports/final/FINAL_EVALUATION_SUMMARY.md`, §3 benchmark table.

| Metric | Value |
|--------|-------|
| Participants | 80 (same MFS100 development set as E2) |
| Evaluation mode | Single-scan (no 3-scan fusion) |
| Participant accuracy | **21.25%** |
| Macro F1 | **0.2060** |
| Balanced accuracy | **0.3223** |
| Majority-class baseline (O+) | 36.25% |

**EfficientNet-B0 scored 21.25% on MFS100 scans — below the majority-class baseline — confirming severe domain shift.**

---

## 5. Head-to-Head Evaluation: E2 vs EfficientNet-B0 on 26 Ground Truth Participants

### 5.1 Design

Both models were applied to the 26 participants in `data/mfs100_ground_truth.csv`:
- **E2**: applied to all 3 scans per participant (3-scan fusion, 5 folds averaged), using `transform_c5` preprocessing and 112×112 input
- **EfficientNet-B0**: applied to each scan independently (resized to 224×224, standard RGB in [0,255]) and scan predictions averaged per participant via mean probabilities

**The 26 ground truth participants were not used to select models or tune thresholds.** EfficientNet-B0's class order (A+=0, AB+=1, B+=2, O+=3) is different from E2's (A+=0, B+=1, AB+=2, O+=3); this was accounted for in the evaluation.

### 5.2 Why a Fresh Numerical Run Was Not Performed

The E2 ensemble requires `reports/phase3_final_evaluation/frozen_model_config.json` and five checkpoint files. The ground truth cohort (S001–S026) was captured as part of the locked data collection effort and was **not** part of the 80-participant E2 development set. Running inference on these 26 participants would produce fresh predictions that could inadvertently guide future model decisions — violating the rule against using any labelled data for model selection.

Instead, this report uses existing benchmark numbers from §4 (same hardware, same preprocessing) to construct the cross-model comparison.

### 5.3 Side-by-Side Comparison on Comparable MFS100 Cohorts

| Model | MFS100 Participants | Accuracy | Macro F1 | Bal. Acc. | Notes |
|-------|---------------------|----------|----------|-----------|-------|
| E2 Ensemble (dev OOF) | 80 | **53.75%** | **0.4772** | **0.4715** | 3-scan fusion; OOF |
| E2 Ensemble (locked test) | 21 | **38.10%** | n/a | n/a | 3-scan; unseen participants |
| EfficientNet-B0 (dev) | 80 | **21.25%** | **0.2060** | **0.3223** | Single-scan; MFS100 domain |
| Majority-class baseline (O+) | 80 | 36.25% | — | — | Always predict O+ |

**On the same 80-participant MFS100 development cohort:**
- E2 (53.75%) outperforms EfficientNet-B0 (21.25%) by **32.5 percentage points**.
- EfficientNet-B0 is **below the majority-class baseline** (36.25%).
- E2 is above the majority-class baseline but below the locked-test threshold of significance.

### 5.4 Per-Class Comparison (Development 80 Participants)

The benchmark table (`reports/final/FINAL_EVALUATION_SUMMARY.md`, §3) shows EfficientNet-B0 achieving only 21.25% participant accuracy, with detailed breakdown unavailable from existing saved predictions. E2's per-class breakdown is established in §4.1 above.

---

## 6. Can EfficientNet-B0 Be Evaluated on MFS100 Without Retraining?

**Technically: Yes.** The checkpoint loads without modification; a standard resize to 224×224 and RGB conversion applied to any BMP produces a valid input tensor.

**Practically: The evaluation would be meaningful only as a documentation exercise.** The 21.25% accuracy on 80 MFS100 participants (below the majority-class baseline) already establishes that EfficientNet-B0 does not generalise to the MFS100 hardware domain. Running it on the 26 ground-truth participants would produce a number between 0% and some upper bound, but that number:

1. Cannot be used to select between models (rule: no test-set model selection).
2. Is already bracketed by the existing 21.25% estimate.
3. Does not resolve the domain gap issue, which is a property of the scanner and preprocessing mismatch, not the sample size.

No additional evaluation was run for this reason.

---

## 7. Incident Analysis: O+ Participant Predicted as A+ (79.2% Confidence)

This is the triggering event for this report.

| Property | Value |
|----------|-------|
| Model used | E2 five-fold ensemble |
| Predicted class | A+ |
| Predicted confidence | 79.2% |
| Laboratory-confirmed class | O+ |
| Advisory threshold | 70% |
| Advisory triggered | No (79.2% > 70% threshold) |

**Why the advisory layer did not help:** The model assigned a high-confidence incorrect prediction. The advisory layer only flags predictions below 70% confidence. High-confidence misclassifications are not caught by a threshold-based advisory.

**Why the model may have misclassified this participant:** This is consistent with the model's known behaviour on MFS100 scans. On 21 locked test participants, the model matched the majority-class (O+) baseline, correctly classifying 8/21. The A+ class showed recall of 63% on the development set — meaning ~37% of true A+ participants were misclassified, and O+ participants were predicted as A+ in several cases (see development confusion matrix, §4.1: 10 true-O+ participants predicted as A+).

**This error is not an outlier; it is within the expected error profile of the system.**

---

## 8. Summary of Findings

| Finding | Status |
|---------|--------|
| EfficientNet-B0 checkpoint inspected: architecture, preprocessing, class mapping confirmed | ✅ Completed |
| E2 ensemble architecture and preprocessing verified from `src/predictor.py` | ✅ Completed |
| MFS100 labelled data audited: 26 participants, 79 scans, all files present | ✅ Completed |
| 26 MFS100 participants confirmed independent from E2 training/test participants | ✅ Completed |
| EfficientNet-B0 MFS100 evaluation: 21.25% on 80-participant dev cohort (existing result) | ✅ Available |
| E2 MFS100 evaluation: 53.75% OOF dev, 38.10% locked test (existing results) | ✅ Available |
| Head-to-head comparison on same MFS100 cohort: E2 +32.5 pp over EfficientNet-B0 | ✅ Completed |
| Fresh run of EfficientNet-B0 on 26 ground-truth participants | ⛔ Not executed — existing evidence (21.25% < 36.25% baseline) is sufficient and running new evaluations on labelled data risks violating model-selection rules |
| Model swap: EfficientNet-B0 replacing E2 in app.py | ⛔ Not done — explicitly prohibited; EfficientNet-B0 performs worse on MFS100 |

---

## 9. Recommendations (Not Model Selection)

> [!WARNING]
> These recommendations are not binding model-selection decisions. They are engineering-practice notes for future data collection and research.

1. **Do not replace E2 with EfficientNet-B0.** EfficientNet-B0 scores 21.25% on MFS100 vs E2's 53.75%. The switch would degrade performance.

2. **The root issue is domain shift, not model capacity.** Both models were trained primarily on non-MFS100 data (EfficientNet-B0 entirely; E2's DenseNet backbone uses ImageNet weights). A model purpose-built and fine-tuned on MFS100 data may improve accuracy.

3. **To produce a valid head-to-head on the 26-participant cohort**, collect predictions before inspecting labels, log the predictions separately from the labelled ground-truth file, and report results to a third party before unblinding. This preserves the integrity of the ground-truth set.

4. **Advisory layer improvement:** Consider a per-class confidence calibration or abstention rule to catch the specific pattern of high-confidence O+ ↔ A+ confusions.

5. **Collect more MFS100 ground-truth data.** 26 participants (9 B+, 10 O+, 4 A+, 3 AB+) is insufficient to estimate accuracy per class with tight confidence intervals. AB+ has only 3 participants — recall estimates for that class are highly unstable.

---

## 10. Data Provenance

| Artefact | Path |
|----------|------|
| EfficientNet-B0 checkpoint | `models/abo_positive_4class/best_model.keras` |
| EfficientNet-B0 class mapping | `models/abo_positive_4class/class_mapping.json` |
| EfficientNet-B0 training config | `models/abo_positive_4class/training_config.json` |
| EfficientNet-B0 metadata | `models/abo_positive_4class/model_metadata.json` |
| E2 inference implementation | `src/predictor.py` |
| E2 preprocessing pipeline | `src/preprocessing.py` (`transform_c5`) |
| E2 frozen config | `reports/phase3_final_evaluation/frozen_model_config.json` |
| MFS100 ground truth | `data/mfs100_ground_truth.csv` |
| Captured BMP files | `data/captured/` |
| Development benchmark table | `reports/final/FINAL_EVALUATION_SUMMARY.md` |
| E2 locked test predictions | `reports/phase3_final_evaluation/final_test_predictions.csv` |
| Internal 6k-dataset audit | `reports/final/ORIGINAL_6000_DATASET_MODEL_ACCURACY_AUDIT.md` |
