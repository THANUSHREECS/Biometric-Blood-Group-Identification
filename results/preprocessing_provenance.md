# Preprocessing Provenance
*(Created Round 16, branch `fix-preprocessing-mismatch`)*

## Per-Script Preprocessing Step Inventory

### 1. `scratch/experiment6_fusion_supcon.py` (E2 Training Script, timestamp 2026-10-09 16:11:16)

| Step | File:Line | Code | Description |
|---|---|---|---|
| Import helpers | `scratch/experiment6_fusion_supcon.py:23` | `from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target` | Imports standard preprocessing subroutines |
| Function definition | `scratch/experiment6_fusion_supcon.py:55` | `def transform_c5(img_path):` | C5 transformation pipeline entry point |
| Load image | `scratch/experiment6_fusion_supcon.py:56` | `pil_img = load_image_to_pil(img_path)` | Opens BMP image using PIL |
| Alpha composite | `scratch/experiment6_fusion_supcon.py:57` | `rgb_img = composite_alpha_on_white(pil_img)` | Composites alpha channel onto solid white background |
| Foreground crop | `scratch/experiment6_fusion_supcon.py:58` | `cropped, _ = crop_mfs100_foreground(rgb_img)` | Bounding-box margin crop of fingerprint ridge foreground |
| Ridge contrast (CLAHE) | `scratch/experiment6_fusion_supcon.py:59` | `enhanced = enhance_mfs100_ridge_contrast(cropped)` | CLAHE ridge enhancement |
| Resize | `scratch/experiment6_fusion_supcon.py:60` | `resized = resize_to_target(enhanced, target_size=IMAGE_SIZE)` | LANCZOS interpolation to `(112, 112)` |
| Normalisation / Return | `scratch/experiment6_fusion_supcon.py:61` | `return np.array(resized, dtype=np.float32)` | Raw float32 values in `[0, 255]`; **no `preprocess_input`** |

---

### 2. `scratch/final_evaluation_e2.py` (Evaluation Script for Saved CSV, timestamp 2026-10-09)

| Step | File:Line | Code | Description |
|---|---|---|---|
| Import helpers | `scratch/final_evaluation_e2.py:12` | `from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target` | Imports preprocessing subroutines |
| Function definition | `scratch/final_evaluation_e2.py:30` | `def transform_c5(img_path):` | C5 evaluation loader |
| Load image | `scratch/final_evaluation_e2.py:31` | `pil_img = load_image_to_pil(img_path)` | Opens BMP image |
| Alpha composite | `scratch/final_evaluation_e2.py:32` | `rgb_img = composite_alpha_on_white(pil_img)` | Alpha compositing on white |
| Foreground crop | `scratch/final_evaluation_e2.py:33` | `cropped, _ = crop_mfs100_foreground(rgb_img)` | Bounding-box margin crop |
| Ridge contrast (CLAHE) | `scratch/final_evaluation_e2.py:34` | `enhanced = enhance_mfs100_ridge_contrast(cropped)` | CLAHE ridge enhancement |
| Resize | `scratch/final_evaluation_e2.py:35` | `resized = resize_to_target(enhanced, target_size=(112, 112))` | LANCZOS interpolation to `(112, 112)` |
| Normalisation / Return | `scratch/final_evaluation_e2.py:36` | `return np.array(resized, dtype=np.float32)` | Raw float32 values in `[0, 255]`; **no `preprocess_input`** |

---

### 3. `scratch/audit_e2_ensemble.py` (Audit Evaluation Script)

| Step | File:Line | Code | Description |
|---|---|---|---|
| Import helpers | `scratch/audit_e2_ensemble.py:11` | `from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target` | Imports preprocessing subroutines |
| Function definition | `scratch/audit_e2_ensemble.py:22` | `def transform_c5(img_path):` | C5 evaluation loader |
| Load image | `scratch/audit_e2_ensemble.py:23` | `pil_img = load_image_to_pil(img_path)` | Opens BMP image |
| Alpha composite | `scratch/audit_e2_ensemble.py:24` | `rgb_img = composite_alpha_on_white(pil_img)` | Alpha compositing on white |
| Foreground crop | `scratch/audit_e2_ensemble.py:25` | `cropped, _ = crop_mfs100_foreground(rgb_img)` | Bounding-box margin crop |
| Ridge contrast (CLAHE) | `scratch/audit_e2_ensemble.py:26` | `enhanced = enhance_mfs100_ridge_contrast(cropped)` | CLAHE ridge enhancement |
| Resize | `scratch/audit_e2_ensemble.py:27` | `resized = resize_to_target(enhanced, target_size=IMAGE_SIZE)` | LANCZOS interpolation to `(112, 112)` |
| Normalisation / Return | `scratch/audit_e2_ensemble.py:28` | `return np.array(resized, dtype=np.float32)` | Raw float32 values in `[0, 255]`; **no `preprocess_input`** |

---

### 4. `src/preprocessing.py::transform_c5` (Deployed Implementation, Post-Fix Round 14)

| Step | File:Line | Code | Description |
|---|---|---|---|
| Function definition | `src/preprocessing.py:247` | `def transform_c5(image_input):` | Production pipeline entry point |
| Load image | `src/preprocessing.py:254` | `raw_pil = load_image_to_pil(image_input)` | Loads PIL image from path, bytes, or array |
| Alpha composite | `src/preprocessing.py:255` | `rgb_pil = composite_alpha_on_white(raw_pil)` | Alpha compositing on white |
| Foreground crop | `src/preprocessing.py:256` | `cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)` | Bounding-box margin crop with metadata tracking |
| Ridge contrast (CLAHE) | `src/preprocessing.py:257` | `enhanced_pil = enhance_mfs100_ridge_contrast(cropped_pil)` | CLAHE ridge enhancement |
| Resize | `src/preprocessing.py:258` | `resized_pil = resize_to_target(enhanced_pil, target_size=(112, 112))` | LANCZOS interpolation to `(112, 112)` |
| Normalisation | `src/preprocessing.py:259` | `img_array = np.array(resized_pil, dtype=np.float32)` | Raw float32 values in `[0, 255]`; **no `preprocess_input`** |
| Return | `src/preprocessing.py:260` | `return img_array, resized_pil, crop_meta` | Returns tuple of array, display PIL image, and metadata |

---

## Provenance Analysis & Config Verification

1. **Exact Pipeline Concordance**:
   All four scripts execute the exact identical sequence of subroutines: `load_image_to_pil` $\rightarrow$ `composite_alpha_on_white` $\rightarrow$ `crop_mfs100_foreground` $\rightarrow$ `enhance_mfs100_ridge_contrast` $\rightarrow$ `resize_to_target(..., (112, 112))` using LANCZOS $\rightarrow$ `np.array(..., dtype=np.float32)`.
   Across all 303 scans in the dataset, the pixel difference between the experiment loader and the deployed loader is identically `0.00000000`.

2. **Root Cause of the Normalisation Bug (Pre-Round 14)**:
   In commit `c1aac08`, the author added `tf.keras.applications.densenet.preprocess_input` at line 276 of `src/preprocessing.py`. This zero-centered and scaled inputs to approximately `[-2.1, 2.6]`, whereas the model checkpoints were trained directly on raw `[0, 255]` float32 values. Removing `preprocess_input` in Round 14 restored exact numerical parity with the training and evaluation scripts.

3. **Config String vs. Code**:
   `models/phase3_fusion_metric_learning/E2/frozen_model_config.json` stated:
   `"composite_alpha_on_white -> crop_mfs100_foreground -> enhance_mfs100_ridge_contrast -> resize (112,112) Lanczos"`.
   The config description accurately reflected the training script (`scratch/experiment6_fusion_supcon.py`). The bug was solely that the initial `src/preprocessing.py` implementation appended an unlisted `densenet.preprocess_input` call. With that call removed, the deployed code, the frozen config description, and the training script are in full alignment.
