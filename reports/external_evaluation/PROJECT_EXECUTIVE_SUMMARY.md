# Biometric Blood Group Identification System
## Executive Project Summary & Final Report

**Academic Year:** 2025Ã¢â‚¬â€œ2026  
**Evaluation Phase:** Final External Demonstration (200 Marks)  
**System Type:** AI-Assisted Research Prototype (Non-Clinical)  
**Target Domain:** Biometric Pattern Recognition & Deep Learning Classification  

---

## 1. Project Overview & Scientific Scope
This major project investigates the novel hypothesis of utilizing fingerprint minutiae and ridge patterns to predict an individual's ABO blood group. Given the highly experimental nature of this biometric correlation, the system was built strictly as a **research prototype**.

**Final Project Scope:**
- **Target Classes:** Restricted to 4 major ABO-Positive groups (A+, B+, AB+, O+). Rh-negative groups were excluded to maintain statistical viability given the dataset distribution.
- **Scientific Framing:** The project does not claim that fingerprints determine blood groups. Instead, it tests for statistical associations using deep learning.
- **Medical Disclaimer:** The system is explicitly labeled as non-clinical and mandates that predictions not be used for medical or transfusion decisions.

## 2. Deep Learning Architecture (The E2 Champion Model)
Through rigorous experimentation (Experiments 1 through 6), a finalized production model was established and frozen:
- **Architecture:** `DenseNet121_Fusion_SupCon` (E2 Champion).
- **Core Mechanism:** Uses transfer learning (ImageNet pre-trained DenseNet121) as a feature extractor.
- **Fusion Protocol:** Employs a 3-scan feature concatenation (Left Thumb Protocol). Three separate fingerprint impressions (112x112) are pooled into a concatenated feature vector, passing through a 512-neuron Dense layer before Softmax classification.
- **Ensemble Strategy:** Soft-voting ensemble aggregating predictions across 5 independent cross-validation checkpoints.
- **Final Performance Metrics:** 
  - Validated on a strictly locked, unseen 898-sample test dataset.
  - **Empirical Accuracy:** 38.10%
  - **Macro F1 Score:** 0.2500

## 3. Preprocessing Pipeline (C5 Pipeline)
A highly specialized, deterministic preprocessing pipeline guarantees that real-world sensor captures match the training data domain:
1. **Alpha Compositing:** Drops transparent channels, enforcing a solid white background.
2. **MFS100 Foreground Crop:** Dynamically detects the active ridge impression and crops empty sensor margins, safeguarding against low-variance blank scans.
3. **CLAHE Enhancement:** Normalizes the low-contrast optical sensor ridges to match the deep-black ridge distribution of the training data.
4. **Resizing:** High-quality Lanczos interpolation to `112x112` dimensions.
5. **ImageNet Normalization:** Pixel arrays are processed using `tf.keras.applications.densenet.preprocess_input` to zero-center and scale distributions to ImageNet standards.

## 4. Hardware Integration
The system interacts seamlessly with physical biometric hardware:
- **Scanner:** Mantra MFS100 USB 2.0 Optical Sensor (500 DPI).
- **Software Bridge:** A custom 32-bit Python-to-.NET bridge (`src/mfs100_bridge/bridge.py`) utilizing the `pythonnet` CLR wrapper communicates directly with the proprietary MFS100 DLL.
- **Capture Process:** Real-time ridge capture, automated NFIQ (NIST Fingerprint Image Quality) grading, and pressure detection.

## 5. Deployed Application System
A production-ready Web Application serves as the interface for the prototype:
- **Backend:** Flask API handling real-time bridge communications, file uploads, and model inference (`app.py`, `src/predictor.py`).
- **Frontend Dashboard:** A sleek, responsive, and clinical-themed HTML/JS/CSS dashboard. 
  - Features real-time scanner status indicators, progressive capture feedback, and a 4-class probability breakdown.
  - Implements a strict UI boundary: displaying low-confidence alerts and mandatory confirmation advisories for predictions below the 70% threshold.
- **Automated Reporting:** Integrates `ReportLab` (`src/report_generator.py`) to generate secure, downloadable PDF audit reports containing prediction outcomes, confidence scores, bounding box metrics, and strict medical disclaimers.

## 6. Final Readiness & Bug Remediation
During the final pre-evaluation readiness phase, several critical audits and pipeline remediations were completed:
1. **Scientific Integrity Audit:** Standardized documentation to ensure claims accurately reflect the 38.10% accuracy metric without falsifying positive results.
2. **Hardware Environment Fix:** Resolved `.NET` DLL loading issues by establishing the correct `pythonnet` dependencies.
3. **Pipeline Bug Remediation:** Discovered and patched a critical discrepancy where the deployed Flask app was omitting ImageNet pixel normalization (`preprocess_input`) and using an incorrect class mapping (`[A+, B+, AB+, O+]` instead of `[A+, AB+, B+, O+]`). These fixes resolved extreme out-of-distribution hallucinations and stabilized live hardware inference.
4. **Documentation Arsenal Generated:**
   - `FINAL_PROJECT_READINESS_REPORT.md`
   - `HARDWARE_TEST_PROCEDURE.md`
   - `EXTERNAL_DEMO_SCRIPT.md`
   - `VIVA_PREPARATION.md`
   - `FINAL_DEMO_CHECKLIST.md`

## 7. Conclusion
The Biometric Blood Group Identification system is a technically complex, end-to-end Machine Learning engineering project. While the underlying biological hypothesis yielded limited predictive power (38.10%), the software engineering executionÃ¢â‚¬â€encompassing deep learning ensembles, 3D feature fusion, hardware driver bridging, strict pipeline deterministic matching, and full-stack web deploymentÃ¢â‚¬â€demonstrates a rigorous, high-level mastery of modern AI systems engineering. The project is fully prepared for final external evaluation.
