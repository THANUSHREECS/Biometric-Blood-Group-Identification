# RECOMMENDED ACTION PLAN: REMAINING WORK

Based on the Project Readiness Audit, the following tasks are ranked by their necessity for a successful external evaluation. 

*Note: No new models are required to be trained. The focus is entirely on application alignment.*

## 1. ESSENTIAL FIXES (High Priority, Low Implementation Risk)
These must be resolved for the application to function correctly during the live demonstration.

**Task 1.1: Align Application Configuration to MFS100 Target**
- **Action:** Update `src/config.py` to target 4 classes (`A+`, `B+`, `AB+`, `O+`) instead of the legacy 8 classes.
- **Action:** Update `IMAGE_SIZE` and `INPUT_SHAPE` in `config.py` from $224 \times 224$ to $112 \times 112$.
- **Scientific Value:** Critical. The application will crash if the input shapes and class arrays do not match the final E2 model structure.

**Task 1.2: Point Predictor to the E2 Champion Model**
- **Action:** Modify `src/predictor.py` or the environment variables to load the frozen `Experiment 6 E2` checkpoint rather than the legacy `Champion Ensemble V2`.
- **Scientific Value:** Critical. Demonstrating the legacy 85% model on synthetic data misrepresents the actual 7th-semester MFS100 research effort.

**Task 1.3: Update Application Preprocessing**
- **Action:** Ensure that `/api/capture` and `/api/upload` route inputs through the documented C5 preprocessing pipeline (`transform_c5`).
- **Scientific Value:** Critical. The E2 model expects cropped, CLAHE-enhanced inputs.

## 2. REQUIRED SCIENTIFIC DISCLAIMERS (Medium Priority, Zero Risk)
**Task 2.1: Update UI Text**
- **Action:** Modify `templates/index.html` to clearly state that the model operates at ~38% accuracy in the real world and that the hypothesis of fingerprint-to-blood-group mapping was scientifically rejected for this hardware.
- **Scientific Value:** High. External evaluators will deduct marks if a student presents a 38% accurate system as a "successful medical tool." Honesty about the scientific findings will yield higher marks.

## 3. OPTIONAL IMPROVEMENTS (Low Priority, Skip if Time is Limited)
- **Task 3.1: Clean up unused legacy models.** Delete or archive the VGG16 and EfficientNet weights from the deployment folder to save disk space and improve startup time.
- **Task 3.2: Database Purge.** Clear the SQLite database of any legacy 8-class predictions to ensure the history page only shows clean 4-class results during the demo.

**Final Recommendation:** Execute the Essential Fixes (Task 1.1, 1.2, 1.3) by simply updating the python variables in `config.py` and `predictor.py`. Do not retrain any models. Then proceed to the Viva evaluation.
