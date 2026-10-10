# Wrong Prediction Investigation Report

## 1. Incident Overview
- **Observed Prediction:** A+ (69.5%), B+ (16.1%), AB+ (5.6%), O+ (8.8%)
- **Ground Truth Supplied:** O+
- **Result:** The system incorrectly predicted A+.

## 2. Pipeline Discrepancies Discovered
During the investigation of the research vs. deployed inference pipelines, two critical software discrepancies were found that completely invalidated the hardware test:

### Discrepancy A: Missing ImageNet Preprocessing
The E2 model (DenseNet121) was originally trained utilizing Keras's built-in ImageNet preprocessing layer `tf.keras.applications.densenet.preprocess_input`. This function normalizes raw pixel arrays into a specific scale and mean-variance distribution expected by the DenseNet backbone.
However, the deployed `transform_c5` pipeline in `src/preprocessing.py` was returning raw `0-255` float values and feeding them directly into the model. By bypassing the required normalization, the model's first convolutional layers experienced massive, out-of-distribution activation spikes, resulting in arbitrary "garbage" predictions (e.g., confidently predicting A+ indiscriminately).

### Discrepancy B: Class Index Mapping Mismatch
The original training configuration (`train_abo_positive_4class.py`) mapped the four classes alphabetically as:
- 0: A+
- 1: AB+
- 2: B+
- 3: O+

However, the deployed `src/config.py` incorrectly hardcoded the mapping as:
- 0: A+
- 1: B+
- 2: AB+
- 3: O+

This mismatch would cause any B+ prediction to be labeled AB+, and vice-versa. (Note: Since O+ is mapped to index 3 in both configurations, this specific bug did not cause the A+ misprediction; that was solely due to the preprocessing failure).

## 3. Investigation Tasks Executed
- **Audited the Scans:** I verified the bridge logs and predictor to confirm the three scans were genuine, distinct hardware captures processed in sequence. No reuse or overwriting occurred.
- **Checked Checkpoints:** The 5 E2 checkpoints were correctly loaded.
- **Checked Dashboard Metrics:** The dashboard is correctly parsing the backend probability array. It displays raw Softmax probability, not calibrated confidence or dataset accuracy.

## 4. Fixes Applied
1. **Preprocessing Fix:** Added `tf.keras.applications.densenet.preprocess_input` to the end of `transform_c5` in `src/preprocessing.py`.
2. **Mapping Fix:** Corrected `CLASS_NAMES` in `src/config.py` and `test_integration.py` to match the exact `[A+, AB+, B+, O+]` order established during training.

## 5. Conclusion & Recommendations
- **Cause of Wrong Prediction:** A genuine software bug. The missing pixel normalization completely scrambled the model's inference mathematics.
- **Is the model still unreliable?** Yes. Even with the pipeline bugs fixed, the E2 frozen model remains a research prototype with a measured empirical accuracy of only 38.10% on unseen participants.
- **Next Steps for the Demonstration:** The software bugs have been patched. You must restart the Flask server (`python app.py`) and perform the hardware test again. If the model still predicts A+ (or anything other than O+), it must be honestly reported as a model generalization failure, in line with the established 38.10% accuracy metrics.
