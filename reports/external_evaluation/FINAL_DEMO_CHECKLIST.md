# Final Demo Checklist

Use this checklist to ensure a flawless demonstration during the external evaluation.

## 1. Environment Setup
- [ ] Laptop is plugged into a power source.
- [ ] Python virtual environment is activated (`venv\Scripts\activate`).
- [ ] All required dependencies are installed (`pip install -r requirements.txt`).
- [ ] No other applications are using port 5000.

## 2. Hardware Preparation (If Applicable)
- [ ] Mantra MFS100 scanner is plugged into the USB port.
- [ ] The MFS100 driver is installed and the device is recognized by Windows.
- [ ] The scanner lens is clean (use a microfiber cloth).
- [ ] The 32-bit hardware bridge is running: `python -m src.mfs100_bridge.bridge`.

## 3. Application Launch
- [ ] Start the Flask server: `python app.py`.
- [ ] Open the dashboard in a modern browser: `http://127.0.0.1:5000`.
- [ ] Verify the UI loads correctly and the medical disclaimer is visible.
- [ ] Verify the model tag reads "E2 Research Ensemble" with the research prototype advisory badge.

## 4. Software & Mock Testing Verification
- [ ] Set scanner mode to "Mock Mode".
- [ ] Test a single file upload via the File Upload tab.
- [ ] Verify prediction completes and UI updates.
- [ ] Click "Download Clinical PDF Report".
- [ ] Open the PDF and verify formatting, data, and the 38.10% accuracy metric disclaimer.

## 5. Live Scanner Verification (If Hardware Connected)
- [ ] Switch scanner mode to "Physical MFS100".
- [ ] Select "3-Scan Consensus Mode" (Left Thumb Protocol).
- [ ] Capture 3 impressions successfully.
- [ ] Verify probabilities sum to ~100% and a prediction is rendered.
- [ ] Generate the PDF report for the live scan.

## 6. Code & Data Integrity
- [ ] Confirm no locked test dataset files are open or being accessed.
- [ ] Confirm `app.js` and `app.py` correctly point to the E2 model.
- [ ] Ensure `test_integration.py` runs without errors before the examiners arrive.

## 7. Presentation Readiness
- [ ] Review `EXTERNAL_DEMO_SCRIPT.md`.
- [ ] Review `VIVA_PREPARATION.md`.
- [ ] Have the Project Report Document physically printed or open in a PDF viewer.
- [ ] Be prepared to honestly discuss the 38.10% accuracy as a valid scientific finding.

**CRITICAL RULE:** Do NOT attempt to alter code, retrain models, or change confidence thresholds during the demo to artificially inflate the accuracy. Scientific honesty is paramount.
