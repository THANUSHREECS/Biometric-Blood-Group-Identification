# REPRODUCIBILITY AUDIT

## 1. DATASET MANIFESTS
- **Directory:** `data/mfs100_final/`
- **Total Images:** 303 physical image files.
- **Participant Split Manifest:** `reports/phase3_participant_benchmark/participant_split.csv`
- **Split Configuration:** 64 Train / 16 Validation / 21 Locked Test. (3 scans per participant).

## 2. PARTICIPANT-LEVEL SPLITS
All participant-level cross-validation throughout the diagnostic experiments (Exp 8, 9, 10) was rigorously controlled using:
- **Library:** `scikit-learn`
- **Class:** `StratifiedGroupKFold`
- **Arguments:** `n_splits=5, shuffle=True, random_state=42`
- **Grouping Array:** `participant_id`
This guarantees mathematically identical fold assignments across all comparative scripts.

## 3. MODEL CHECKPOINT LOCATIONS
- **Experiment 6 (Champion E2) Code Archive:** `scratch/experiment6_fusion_supcon.py`
- **Frozen Test Evaluation Pipeline:** `scratch/final_evaluation_e2.py`
- **Pre-Final Audit Configuration:** `reports/phase3_final_evaluation/frozen_model_config.json`

## 4. PREPROCESSING IMPLEMENTATION
The C5 preprocessing pipeline is preserved in `src/preprocessing.py`:
- `load_image_to_pil()`
- `composite_alpha_on_white()`
- `crop_mfs100_foreground()`
- `enhance_mfs100_ridge_contrast()` (CLAHE)
- `resize_to_target()` ($112 \times 112$)

## 5. EXPERIMENT SCRIPTS
Diagnostic scripts are fully preserved:
- `scratch/experiment8/run_experiment8.py` (Linear Classifiers)
- `scratch/experiment9/run_experiment9.py` (Handcrafted Features)
- `scratch/experiment10/run_experiment10.py` (Augmentation and Regularization)

## 6. RANDOM SEEDS & SOFTWARE
- **Random Seeds:** `random_state=42` was hardcoded in all fold generators.
- **Deep Learning:** TensorFlow/Keras backend.
- **Machine Learning:** Scikit-Learn 1.x.
- **Image Processing:** OpenCV (`cv2`), `skimage`.

## 7. METRIC CALCULATION METHODS
- Metrics computed globally across folded predictions using `sklearn.metrics`.
- **Primary Metric:** `f1_score(average='macro', zero_division=0)`
- **Aggregation:** In single-scan evaluation, model prediction probabilities were mathematically averaged (`np.mean(axis=0)`) per participant before computing the final `argmax` class prediction.

## 8. KNOWN DISCREPANCIES & ISSUES
- The original "85.41%" benchmark referenced in the earliest phases of the repository pertained to a completely different 6,000-image canonical dataset (likely SOCOFing), split at the image-level (which inherently leaks identity). This score is historically accurate to the previous repository state but scientifically invalid for the real-world MFS100 optical problem space.
