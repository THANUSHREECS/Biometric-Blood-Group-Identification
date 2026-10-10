# DEMONSTRATION PLAN FOR EXTERNAL EVALUATION

## 1. SELECTED DEMONSTRATION MODEL
The model selected for the final-year external evaluation demonstration is the **Experiment 6 E2 Champion** (Three-Scan Feature Concatenation + Supervised Contrastive Learning + Classification).
- **Architecture:** Frozen DenseNet121 Backbone
- **Input:** $112 \times 112$ C5-preprocessed optical scan
- **Output:** 4 Classes (A+, B+, AB+, O+)

*Note: This model was selected based entirely on rigorous, locked cross-validation decisions made during Phase 3, avoiding any hyperparameter tuning on the locked test set.*

## 2. NARRATIVE AND DISCLAIMER STRATEGY
During the external evaluation demonstration, the project must be explicitly framed as an **investigative scientific prototype**, not a commercially viable or medically valid product.

**Key Talking Points:**
1. **The Hypothesis:** The project set out to test a historical biometric hypothesisÃ¢â‚¬â€that macroscopic fingerprint ridge patterns encode genetic ABO blood group traits.
2. **The Illusion of Success:** Initial testing on canonical/synthetic datasets yielded 85% accuracy. However, rigorous participant-level splitting revealed this was a machine-learning artifact (data leakage).
3. **The Reality of Biology:** When rigorously tested on strictly independent individuals using optical hardware (MFS100), the model yielded 38.1% accuracy.
4. **The Conclusion:** We successfully built the hardware pipeline, deep learning architecture, and evaluation framework. The framework successfully *disproved* the hypothesis for 500 DPI optical captures. The model's failure to generalize is a scientific finding, not an engineering defect.

## 3. LIVE DEMONSTRATION SCRIPT
1. **Hardware Integration:** Connect the MFS100 scanner and demonstrate live physical capture using the Left Thumb.
2. **Consensus Mode:** Demonstrate the `/api/capture/consensus` feature to show how the application merges multiple scans to handle sensor noise.
3. **Inference & Visualization:** Show the PDF report generation and SQLite history.
4. **Transparent Evaluation:** Point to the Confidence Advisory Layer in the UI and explicitly state that the model operates at ~38% accuracy in the real world, heavily relying on the fallback O+ and A+ majority classes.
