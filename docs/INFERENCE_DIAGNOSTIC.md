# Inference Diagnostic, Weight Reconciliation & Test Integrity Report

**Project**: Biometric Blood Group Identification System  
**Date**: 2026-09-04  
**Status**: Completed Diagnostic Investigation & Integrity Reconciliation  
**Test Set Policy**: **STRICTLY LOCKED** — No further test set evaluations permitted.

---

## 1. Reconciliation of Ensemble Weights & Historical Evaluations

A thorough audit of all experiment scripts, validation logs, and output artifacts was conducted to resolve the history between the **50 / 35 / 15** and **58 / 38 / 4** weight configurations.

### Summary Comparison Table

| Property | Historical Test Evaluation (Evaluation #1) | Diagnostic Test Evaluation (Evaluation #2) |
|---|---|---|
| **Date & Timestamp** | `2026-09-04T14:46:16.504421` | `2026-09-04T19:23:52.730272` |
| **Script Executed** | `scripts/evaluate_final_model.py` | `scripts/evaluate_champion_v2.py` |
| **Result File** | `outputs/final_evaluation_results.json` | `outputs/FINAL_MODEL_EVALUATION.json` |
| **ResNet50 Weight** | **0.50** (50%) | **0.58** (58%) |
| **VGG16 Weight** | **0.35** (35%) | **0.38** (38%) |
| **EfficientNetB0 Weight** | **0.15** (15%) | **0.04** (4%) |
| **ResNet50 Preprocessing** | Standard `resnet50.preprocess_input` (No contrast adjustment) | `1.2× Contrast` + `resnet50.preprocess_input` |
| **VGG16 Preprocessing** | Standard `vgg16.preprocess_input` | Standard `vgg16.preprocess_input` |
| **EffNetB0 Preprocessing**| Standard `efficientnet.preprocess_input` | Standard `efficientnet.preprocess_input` |
| **Validation Metric** | Val Acc: **85.19%**, Val Macro F1: **0.8547** | Val Acc: **86.41%**, Val Macro F1: **0.8663** |
| **Validation Origin** | `scripts/evaluate_ensemble.py` | `scripts/optimize_ensemble_v2.py` |
| **Test Accuracy (898)** | **85.41%** (767 / 898 correct) | **85.41%** (767 / 898 correct) |
| **Test Loss (SCCE)** | **0.4527** | **0.4274** |
| **Test Macro F1** | **0.8591** | **0.8590** |
| **Correct / Total** | 767 / 898 | 767 / 898 |

---

### Questions Directly Answered

#### Q1: Which weights produced the previously reported 85.41% result?
- **Configuration**: **50% ResNet50 / 35% VGG16 / 15% EfficientNetB0**.
- **Preprocessing**: Standard ImageNet Caffe/Torch preprocessing for all three backbones (no contrast enhancement).
- **Execution**: Evaluated on the 898 test samples on `2026-09-04T14:46:16` via `scripts/evaluate_final_model.py`.
- **Artifact**: Preserved in `outputs/final_evaluation_results.json`.

#### Q2: Which weights produced the latest 85.41% result?
- **Configuration**: **58% ResNet50 (with 1.2× contrast) / 38% VGG16 / 4% EfficientNetB0**.
- **Execution**: Evaluated on the 898 test samples on `2026-09-04T19:23:52` via `scripts/evaluate_champion_v2.py`.
- **Artifact**: Preserved in `outputs/FINAL_MODEL_EVALUATION.json`.

#### Q3: When and how were the 58 / 38 / 4 weights selected?
- In `scripts/optimize_ensemble_v2.py` (documented in `docs/MODEL_OPTIMIZATION_V2.md`), Phase D benchmarked four preprocessing filters on single models, finding that $1.2\times$ contrast lifted ResNet50 validation accuracy from $79.73\%$ to $80.96\%$.
- Phase F performed a fine-grained simplex grid search (step = 0.02) over weights of ResNet50 (with contrast), VGG16, and EfficientNetB0.
- The highest validation score was achieved at $w_{res} = 0.58$, $w_{vgg} = 0.38$, $w_{eff} = 0.04$, yielding $86.41\%$ validation accuracy and $0.8663$ validation Macro F1.

#### Q4: Were the 58 / 38 / 4 weights selected using validation data only?
- **YES. Strictly validation data only.**
- In `scripts/optimize_ensemble_v2.py` lines 37–40:
  ```python
  def load_val_data():
      df = pd.read_csv(DATASET_MANIFEST_PATH)
      val_df = df[df["split"] == "val"].copy()
      return val_df["image_path"].tolist(), np.array(val_df["class_id"].astype(int).tolist())
  ```
- The test set was never loaded or referenced during weight optimization. Output saved to `outputs/ensemble_optimization_v2_results.json`.

---

## 2. Correction of the Test-Set History

### Formal Acknowledgment of Multiple Test Evaluations
The previous project documentation claimed: *"Single Final Test Run — The locked test split was evaluated exactly once"*.

**This statement is now explicitly corrected:**
- **Test Evaluation #1** (`2026-09-04T14:46:16`): Conducted via `scripts/evaluate_final_model.py` with weights `(0.50, 0.35, 0.15)` without contrast enhancement.
- **Test Evaluation #2** (`2026-09-04T19:23:52`): Conducted via `scripts/evaluate_champion_v2.py` during diagnostic investigation with weights `(0.58, 0.38, 0.04)` and $1.2\times$ contrast on ResNet50.

Both evaluations evaluated the locked 898 test set samples and both achieved **85.41% accuracy (767/898 correct)**. Both result files are permanently preserved as historical records:
- Evaluation #1: `outputs/final_evaluation_results.json`
- Evaluation #2: `outputs/FINAL_MODEL_EVALUATION.json`

---

## 3. Strict Lock on Future Test Evaluations

**Policy Effective Immediately:**
1. The 898-image test set is **100% frozen and locked**.
2. **No further executions** of `evaluate_final_model.py`, `evaluate_champion_v2.py`, or any other test-set evaluation scripts will be performed.
3. All future debugging, preprocessing refinement, threshold tuning, ensemble modifications, or calibrations must be conducted **strictly on the 898-sample validation split (`split == "val"`) or unit test fixtures**.

---

## 4. Verification of the Preprocessing Pipeline

### Concrete Code Alignment Evidence

#### Training / Evaluation Function (`scripts/evaluate_champion_v2.py` lines 83–91)
```python
for p in b_paths:
    with Image.open(p) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, target_size=(224, 224))
        if enhance_contrast:
            resized = ImageEnhance.Contrast(resized).enhance(1.2)
    arr = np.array(resized, dtype=np.float32)
    b_imgs.append(prep_fn(arr))
```

#### Production Inference Pipeline (`src/preprocessing.py` lines 147–167)
```python
raw_pil = load_image_to_pil(image_input)
rgb_pil = composite_alpha_on_white(raw_pil)
resized_pil = resize_to_target(rgb_pil, target_size=IMAGE_SIZE)

# ResNet50 path: 1.2x contrast enhancement (matches evaluate_champion_v2.py)
resnet_pil = ImageEnhance.Contrast(resized_pil).enhance(1.2)
resnet_array = np.expand_dims(np.array(resnet_pil, dtype=np.float32), axis=0)
resnet_tensor = tf.keras.applications.resnet50.preprocess_input(resnet_array)

# VGG16 and EfficientNetB0: no contrast enhancement
base_array = np.expand_dims(np.array(resized_pil, dtype=np.float32), axis=0)
vgg_tensor = tf.keras.applications.vgg16.preprocess_input(base_array.copy())
effnet_tensor = tf.keras.applications.efficientnet.preprocess_input(base_array.copy())
```

### Preprocessing Checks
1. **ResNet50**: Receives `ImageEnhance.Contrast(resized_pil).enhance(1.2)` followed by `resnet50.preprocess_input` (BGR conversion and ImageNet zero-centering).
2. **VGG16**: Receives standard `vgg16.preprocess_input` directly from `resized_pil` (no contrast alteration).
3. **EfficientNetB0**: Receives standard `efficientnet.preprocess_input` directly from `resized_pil` (no contrast alteration).
4. **No Double Preprocessing**: `resized_pil` is created once from `rgb_pil`. `ImageEnhance.Contrast` is applied to a separate variable `resnet_pil`. The base array used for VGG16 and EfficientNetB0 is constructed independently from `resized_pil`.
5. **Class Ordering Invariance**: Confirmed identical across `models/class_names.json`, `src/config.py`, `scripts/evaluate_final_model.py`, `scripts/evaluate_champion_v2.py`, and `src/predictor.py`:
   - `0: A+`, `1: A-`, `2: AB+`, `3: AB-`, `4: B+`, `5: B-`, `6: O+`, `7: O-`

---

## 5. Mantra MFS100 Hardware vs Training Data Verification

### Confirmation of Genuine Hardware Capture (Not Mock Data)
We audited the capture files in `data/captured/` and compared their signatures:

| Dimension / Attribute | Mock Emulation Capture | Physical MFS100 Scanner Capture |
|---|---|---|
| **Source** | Read from `data/sample/<class>.bmp` | Streamed from native SDK `AutoCapture()` |
| **Width × Height** | `96 × 103` px (or `241 × 298` for O-) | **`316 × 354` px** |
| **Color Mode** | 3-channel RGB (`Mode: RGB`) | **1-channel 8-bit Grayscale (`Mode: L`)** |
| **Byte Size on Disk** | 29,718 bytes (or 215,806 bytes) | **112,942 bytes** ($316 \times 354 + 1078$ header) |
| **Hardware Serial** | N/A (Emulated) | **`11215389`** (MANTRA MFS100) |
| **Quality & NFIQ** | Fixed mock constants | Dynamic sensor feedback (Quality: 75–85, NFIQ: 1) |

**Conclusion**: Physical captures (e.g. `data/captured/capture_1788528658550.bmp`) are confirmed **genuine hardware captures from the physical Mantra MFS100 device**.

---

### Side-by-Side Comparison: Physical MFS100 Capture vs Dataset Training Images

Measurements taken from `data/captured/capture_1788528658550.bmp` (Physical MFS100) vs dataset training samples in `data/processed/`:

| Feature | Physical MFS100 Capture | Dataset Training Images | Discrepancy & Inference Impact |
|---|---|---|---|
| **Resolution** | `316 × 354` px | `96 × 103` px | Physical sensor has $\sim 3.3\times$ higher native linear resolution. |
| **Aspect Ratio (W / H)**| `0.8927` | `0.9320` | Closely aligned ($\Delta < 4\%$). When resized to $224 \times 224$, distortion is minimal. |
| **Color Channels** | 1-channel Grayscale (`Mode: L`, 8-bit) | 3-channel RGB (`Mode: RGB`, 24-bit) | Dataset stores grayscale ridges inside 3 identical RGB channels. Our pipeline converts `Mode: L` $\to$ `RGB` before inference. |
| **Pixel Range** | $[0, 250]$ | $[0, 255]$ | Both span the full dynamic range. |
| **Mean Pixel Intensity**| $173.4$ | $151.5$ to $189.2$ (mean $\sim 160.7$) | Mean intensity is within the exact empirical range of training classes. |
| **Pixel Std Deviation** | $91.56$ | $89.1$ to $109.2$ (mean $\sim 101.4$) | Ridge-to-valley contrast magnitude is closely comparable. |
| **Background** | White / Bright sensor platen ($239.2 \pm 21.0$) | White / Bright background ($209.4 \pm 47.5$) | Both datasets use light backgrounds with dark ridge lines. |
| **Crop / Framing** | Full sensor platen; print surrounded by empty platen border if finger is small | Tight crop around fingerprint core / pattern area | **Key Difference**: Training images are closely cropped around the fingerprint. Hardware captures contain varying amounts of white border depending on finger placement. |
| **Orientation** | Upright distal phalanx | Upright distal phalanx | Orientation is identical. |

*Note: Per instructions, no model modifications or retraining have been made based on this comparison.*

---

## 6. Verification Status Summary

1. **Determinism**: 10 repeated inferences on the same input image yielded max probability variance strictly equal to $0.0$ (bit-identical).
2. **Byte / Path Invariance**: Loading an image from raw BMP bytes vs. a file path produces identical probability distributions.
3. **Automated Unit Tests**: All 38 tests in the regression suite pass (`Ran 38 tests in 28.549s — OK`).
4. **Test Set Lock**: The test set is completely frozen.
