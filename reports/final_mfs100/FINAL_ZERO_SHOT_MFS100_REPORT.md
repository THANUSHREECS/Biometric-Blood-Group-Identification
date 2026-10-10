# FINAL ZERO-SHOT MFS100 REPORT

## 1. Objective
Evaluate how well the previously frozen blood-group model generalizes to completely new participants captured using the real Mantra MFS100 scanner, treated as an independent target-domain evaluation dataset.

## 2. Frozen Model
Day 27 Model A (`models/day27_full_dataset/four_class/best_model.keras`).

## 3. Frozen Preprocessing
C5 preprocessing pipeline (`src/preprocessing.py`): `composite_alpha_on_white` -> `crop_mfs100_foreground` -> `enhance_mfs100_ridge_contrast` -> `resize_to_target` (112x112).

## 4. Dataset
The frozen final MFS100 dataset comprising 101 participants and 303 images. 

## 5. Participant-Level Methodology
For every participant, each of their 3 scans was preprocessed and predicted separately. The final participant-level prediction was calculated by averaging the probability vectors across the 3 scans and selecting the `argmax`.

## 6. Three-Scan Aggregation
Averaging probabilities allows the model to produce a more robust single prediction for the participant across multiple captures, instead of relying on a single highest-confidence scan or majority voting.

## 7. Scan-Level Exploratory Results
Exploratory results on the 303 individual scans are available in `final_zero_shot_scan_predictions.csv`. These do not represent 303 independent samples.

## 8. Participant-Level Primary Results
- Accuracy: 30.69%
- Macro F1: 0.1503
- O+ prediction share: 86.14%

## 9. Confusion Matrix
Please refer to `final_zero_shot_confusion_matrix.png` and `final_zero_shot_confusion_matrix_normalized.png`.

## 10. Per-Class Metrics
- **A+ Recall:** 8.33%
- **B+ Recall:** 0.00%
- **AB+ Recall:** 0.00%
- **O+ Recall:** 78.38%

## 11. Confidence Analysis
Participant-level confidence data is stored in `final_zero_shot_participant_predictions.csv`. Note: Model confidence is not a probability of medical correctness.

## 12. Three-Scan Consistency
- 3/3 prediction agreement: 94 participants
- 2/3 agreement: 7 participants
- 1/3 agreement: 0 participants

## 13. Bootstrap Uncertainty (Participant Level)
- 95% CI for Accuracy: [21.78%, 40.59%]
- 95% CI for Macro F1: [0.1017, 0.2041]

## 14. Leakage Audit
Status: **PASS**. (See `FINAL_ZERO_SHOT_LEAKAGE_AUDIT.md`)

## 15. Limitations
The model heavily over-predicts the O+ class (86.14% prediction share) on the new target domain, showing significant domain shift degradation. The historical exploratory benchmark of 51.28% on 78 burst captures is not comparable, as the current evaluation enforces participant-level independence. 

## 16. Interpretation
The zero-shot generalization of the Day 27 model to completely new participants on the MFS100 scanner is extremely poor (30.69% accuracy). The model's predictions have collapsed largely into a single majority class (O+). The high prediction consistency (94 participants with 3/3 agreement) highlights that the model's failure is systematic and robust across repeated captures of the same participant, not a result of noisy captures. Target-domain learning/adaptation is strictly necessary to recover performance.
