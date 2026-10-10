# Biometric Blood Group Identification System Using Fingerprint Sensor
## Project & Dataset Audit Report (Step 1)

**Audit Date**: September 3, 2026  
**Auditor**: Lead Developer (Deep Learning & Systems Engineer)  
**Workspace**: `c:\Users\thanu\OneDrive\Desktop\major project`  
**Dataset Directory**: `dataset_blood_group`

---

## 1. Executive Summary

A comprehensive, non-destructive audit of the workspace, fresh fingerprint dataset, system environment, and hardware peripherals was conducted. 

### Key Findings at a Glance:
- **Total Images**: Exactly 6,000 fingerprint images found across 8 blood group classes.
- **Classes**: Exactly 8 target classes detected (`A+`, `A-`, `AB+`, `AB-`, `B+`, `B-`, `O+`, `O-`).
- **File Format**: 100% Windows Bitmap (`.BMP`).
- **Color Mode**: 5,956 RGBA images (99.27%) and 44 RGB images (0.73%).
- **Dimensions**: 5,956 images are `96 Ãƒâ€” 103` pixels; 44 images are `241 Ãƒâ€” 298` pixels.
- **Image Corruption**: 0 corrupted or unreadable images. All 6,000 files passed PIL integrity verification.
- **Duplicates**: 11 exact duplicate clusters containing 15 redundant duplicate files. **0 cross-class duplicates**.
- **Class Imbalance**: Significant imbalance detected. `A-` contains 1,009 samples whereas `A+` contains only 565 samples (~1.79Ãƒâ€” ratio).
- **Physical Hardware**: **Mantra MFS100 USB fingerprint scanner is physically plugged in and recognized** (`USB\VID_2C0F&PID_1005\89D28C5E19D4`, Status: OK).
- **Driver / SDK**: Driver package found at `C:\Program Files\Mantra\MFS100\Driver`. Official `MFS100Dll.dll` identified with image capture exports.
- **Python / Framework**: Python 3.10.11 (64-bit), TensorFlow 2.21.0 (oneDNN CPU), Keras 3.12.4, Flask 3.1.3, scikit-learn 1.7.2, Pillow 12.3.0. Additional packages needed: `opencv-python`, `seaborn`, `reportlab`.

---

## 2. Dataset Structure and Inventory

The raw dataset directory `dataset_blood_group` contains 8 subdirectories:

| Directory | Target Blood Group | Class ID | File Count | Percentage of Dataset |
|:---|:---:|:---:|:---:|:---:|
| `A+` | A+ | 0 | 565 | 9.42% |
| `A-` | A- | 1 | 1,009 | 16.82% |
| `AB+` | AB+ | 2 | 708 | 11.80% |
| `AB-` | AB- | 3 | 761 | 12.68% |
| `B+` | B+ | 4 | 652 | 10.87% |
| `B-` | B- | 5 | 741 | 12.35% |
| `O+` | O+ | 6 | 852 | 14.20% |
| `O-` | O- | 7 | 712 | 11.87% |
| **Total** | **8 Classes** | Ã¢â‚¬â€ | **6,000** | **100.00%** |

### Class Imbalance Analysis:
- The largest class is **A-** (1,009 images), while the smallest class is **A+** (565 images).
- Stratified sampling is essential for train/validation/test splits (70% / 15% / 15%).
- Mild class weighting (`compute_class_weight`) should be evaluated during training to prevent bias towards majority classes without causing instability.

---

## 3. Filename Conventions and Sample Identification

- Filename pattern across all classes: `cluster_{cluster_id}_{image_id}.BMP`
  - `cluster_0_*.BMP` -> Class `A+`
  - `cluster_1_*.BMP` -> Class `A-`
  - `cluster_2_*.BMP` -> Class `B+`
  - `cluster_3_*.BMP` -> Class `B-`
  - `cluster_4_*.BMP` -> Class `AB+`
  - `cluster_5_*.BMP` -> Class `AB-`
  - `cluster_6_*.BMP` -> Class `O+`
  - `cluster_7_*.BMP` -> Class `O-`
- Every image in the dataset has a unique numerical suffix across the entire dataset (0 to 5,999).
- No metadata indicating subject/donor identity exists beyond the cluster and image ID. Therefore, duplicate detection is the primary safeguard against data leakage between splits.

---

## 4. Image Properties and Preprocessing Considerations

### 4.1 Resolution and Dimensions
- **(96, 103)**: 5,956 images (99.27%)
- **(241, 298)**: 44 images (0.73%)
- **Model Target**: ResNet50 requires `(224, 224, 3)` input.
- **Resizing**: Resizing from `96 Ãƒâ€” 103` and `241 Ãƒâ€” 298` to `224 Ãƒâ€” 224` must preserve fingerprint ridge clarity using high-quality interpolation (e.g., Lanczos or Bilinear).

### 4.2 Color Channels & Alpha Transparency (CRITICAL FINDING)
- **Mode Distribution**:
  - `RGBA` (4-channel): 5,956 images
  - `RGB` (3-channel): 44 images
- **Grayscale Content**: In both RGBA and RGB files, $R = G = B$ across all non-transparent pixels. They are grayscale fingerprint impressions stored in color container formats.
- **Alpha Channel Analysis**:
  - In the 5,956 RGBA images, approximately 7.89% of pixels have `alpha == 0` (located at margins and borders).
  - The RGB values where `alpha == 0` are `(0, 0, 0)` (black).
  - **Severe Pitfall Avoided**: Standard `Image.convert("RGB")` or OpenCV `cv2.imread` blindly strips the alpha channel, leaving those transparent border pixels solid black, creating high-contrast artificial rectangular edges that distort CNN feature maps.
  - **Solution**: The central preprocessing pipeline will composite RGBA images onto a clean white `(255, 255, 255)` background using the alpha mask before resizing, matching physical fingerprint paper and optical scanner platens.

---

## 5. Duplicate and Leakage Audit

A cryptographic SHA-256 hash was computed for all 6,000 files.

### Exact Duplicate Findings:
- Total duplicate clusters: **11**
- Total redundant images: **15**
- Unique distinct images: **5,985**
- **Cross-Class Duplicates**: **0** (No image belongs to more than one blood group).

#### Intra-Class Duplicate Breakdown:
1. `A-`: 1 pair (`cluster_1_120.BMP` == `cluster_1_1694.BMP`)
2. `AB+`: 3 pairs (`cluster_4_3003.BMP` == `cluster_4_325.BMP`, `cluster_4_193.BMP` == `cluster_4_397.BMP`, `cluster_4_4296.BMP` == `cluster_4_5675.BMP`)
3. `AB-`: 1 pair (`cluster_5_2347.BMP` == `cluster_5_4907.BMP`)
4. `B+`: 1 cluster of 5 identical files (`cluster_2_2511.BMP`, `cluster_2_4268.BMP`, `cluster_2_5333.BMP`, `cluster_2_5356.BMP`, `cluster_2_829.BMP`)
5. `O+`: 1 pair (`cluster_6_3415.BMP` == `cluster_6_720.BMP`)
6. `O-`: 4 pairs/clusters (`cluster_7_1468.BMP` == `cluster_7_1819.BMP` == `cluster_7_5380.BMP`, `cluster_7_3871.BMP` == `cluster_7_4235.BMP`, `cluster_7_2270.BMP` == `cluster_7_5326.BMP`, `cluster_7_1040.BMP` == `cluster_7_5447.BMP`)

### Perceptual Duplicate Check:
- Difference hashing (`dHash`) evaluated across classes showed 0 cross-class near duplicates.
- Redundant duplicates will be tracked in `outputs/dataset_manifest.csv` and deduplicated so that no duplicate image leaks across training, validation, or test sets.

---

## 6. Hardware & Mantra MFS100 Scanner Audit

### Physical Device Status:
- Device: **Mantra MFS100 Optical Fingerprint Scanner**
- Connection: USB 2.0
- Plug and Play Info: `InstanceId: USB\VID_2C0F&PID_1005\89D28C5E19D4`, Status: `OK`.
- The physical scanner is actively connected to the host machine.

### Driver & DLL Inventory:
- Installed Driver Directory: `C:\Program Files\Mantra\MFS100\Driver`
- Test Utility & DLLs: `C:\Program Files\Mantra\MFS100\Driver\MFS100Test\`
  - `MFS100Dll.dll` (Official C/C++ API DLL)
  - `MANTRA.MFS100.dll` (.NET assembly)
- Key Exported Functions identified in `MFS100Dll.dll`:
  - `MFS100Init`, `MFS100Uninit`
  - `MFS100DeviceConnected`
  - `MFS100AutoCapture`, `MFS100GetFrame`
  - `MFS100ConvertRawToBmp`
  - `MFS100GetWidth`, `MFS100GetHeight`, `MFS100GetQuality`

### Architecture & Compatibility Finding:
- The installed `MFS100Dll.dll` in `MFS100Test` is **32-bit (x86)**.
- The Python runtime is **64-bit (x64)** (`Python 3.10.11`).
- In Windows, a 64-bit process cannot directly load a 32-bit DLL in-process via `ctypes` (returns `[WinError 193] %1 is not a valid Win32 application`).
- **Planned Adapter Design**:
  1. Primary: Configurable `MFS100Scanner` abstraction in `src/mfs100/scanner.py`.
  2. Mock Mode (`MFS100_MOCK=true`): Fully functional mock capture reading genuine sample fingerprint images from `data/sample/`, passing through the exact same preprocessing and prediction pipeline.
  3. Live Mode (`MFS100_MOCK=false`): Communicates with physical device via 64-bit DLL path / out-of-process 32-bit bridge / local RD service.

---

## 7. Compute & Hardware Acceleration Environment

- **OS**: Windows 11 (64-bit)
- **CPU**: Multi-core x86_64 with AVX2 and oneDNN acceleration enabled.
- **TensorFlow**: Version 2.21.0
- **GPU**: TensorFlow >= 2.11 discontinued native Windows GPU support (`GPUs Available: []`). Training will execute on CPU using oneDNN optimizations.
- **Training Strategy**:
  - ResNet50 backbone feature extraction will be leveraged.
  - Efficient batch processing (`tf.data` pipeline with prefetching).
  - Two-stage transfer learning: Stage 1 feature extraction (frozen backbone), Stage 2 fine-tuning top residual blocks with low learning rate ($10^{-5}$) and early stopping.

---

## 8. Existing Code and Workspace State

- No previous source code existed in `c:\Users\thanu\OneDrive\Desktop\major project` prior to this audit.
- No files were deleted, moved, or altered in `dataset_blood_group`.
- Scratch verification scripts were written exclusively to the isolated IDE brain directory.

---

## 9. Recommended Step-by-Step Implementation Roadmap

1. **Step 2 Ã¢â‚¬â€ Dataset Cleaning & Validation (`scripts/prepare_dataset.py`)**:
   - Preserve `dataset_blood_group` into `data/raw/`.
   - Process images into `data/processed/`: handle alpha transparency compositing on white background, normalize formats.
   - Quarantine or drop the 15 redundant duplicate files.
   - Generate `outputs/dataset_report.json`.

2. **Step 3 & 4 Ã¢â‚¬â€ Deduplication & Stratified Split**:
   - Create 70% train (~4,190), 15% val (~897), 15% test (~898) stratified split across all 8 classes.
   - Generate `outputs/dataset_manifest.csv` with columns: `image_path,class_name,class_id,split`.
   - Save class mapping in `models/class_names.json`.

3. **Step 5 Ã¢â‚¬â€ Central Preprocessing (`src/preprocessing.py`)**:
   - Implement single unified pipeline: load -> alpha composite on white -> resize to 224x224 -> RGB conversion -> ResNet50 `preprocess_input`.

4. **Step 6 Ã¢â‚¬â€ ResNet50 Transfer Learning (`scripts/train_model.py`)**:
   - ResNet50 ImageNet backbone -> GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.4) -> Dense(8, softmax).
   - SparseCategoricalCrossentropy loss.
   - Stage 1: Freeze backbone, train classification head.
   - Stage 2: Unfreeze top layers, fine-tune with small learning rate.
   - Save best model checkpoint: `models/blood_group_resnet50.keras`.
   - Save training curves: `outputs/training_accuracy.png`, `outputs/training_loss.png`.

5. **Step 7 Ã¢â‚¬â€ Real Model Evaluation (`scripts/evaluate_model.py`)**:
   - Evaluate exclusively on the untouched 15% test split.
   - Compute real accuracy, loss, precision, recall, F1-score, confusion matrix (`outputs/confusion_matrix.png`).
   - Save genuine results to `outputs/evaluation_results.json` and `outputs/training_results.json`.

6. **Step 8 Ã¢â‚¬â€ Prediction Engine (`src/predictor.py`)**:
   - `BloodGroupPredictor` class: loads model & class mapping once, returns predicted blood group, confidence score, and complete 8-class probability distribution.
   - Configurable confidence threshold (0.60) with medical advisory flag.

7. **Step 9 Ã¢â‚¬â€ Flask Backend (`app.py`) & SQLite History (`src/database.py`)**:
   - RESTful endpoints: `/api/status`, `/api/upload`, `/api/predict`, `/api/capture`, `/api/history`, `/api/report/<id>`.
   - SQLite DB (`data/blood_group.db`) recording prediction audit trail without sensitive PII.

8. **Step 10 Ã¢â‚¬â€ Healthcare Web Interface (`templates/`, `static/`)**:
   - Modern, responsive dashboard with MFS100 scanner status, capture button, upload zone, prediction cards, probability bar charts, and history table.
   - Prominent medical disclaimer banner on every view.

9. **Step 11 Ã¢â‚¬â€ PDF Clinical Report Generation (`src/report_generator.py`)**:
   - ReportLab PDF generator containing timestamp, input mode, fingerprint preview, predicted blood group, confidence, 8-class breakdown, and mandatory disclaimer.

10. **Step 12-14 Ã¢â‚¬â€ MFS100 Adapter & Mock Mode (`src/mfs100/`)**:
    - Unified scanner interface (`connect`, `disconnect`, `is_connected`, `capture`, `get_device_info`).
    - Configurable `MFS100_MOCK=true/false` support.
    - Sample images in `data/sample/` for reproducible testing.

11. **Step 15-17 Ã¢â‚¬â€ End-to-End Automated Testing (`tests/`)**:
    - `test_preprocessing.py`, `test_predictor.py`, `test_api.py`.
    - Verification of upload flow, mock capture flow, DB logging, and PDF output.

12. **Step 18 Ã¢â‚¬â€ Final Documentation & README**:
    - Comprehensive B.E. major project README with architecture diagrams, dataset audit citations, genuine performance metrics, and operational manuals.
