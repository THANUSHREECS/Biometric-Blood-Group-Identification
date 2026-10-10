# Hardware Test Procedure

This document provides step-by-step instructions for testing the physical Mantra MFS100 scanner and the complete three-scan workflow. Since the scanner is not always available during software development, these manual tests must be performed to guarantee full end-to-end integration readiness.

## Test Procedure

1. **Connect the MFS100 scanner**
   - Plug the Mantra MFS100 scanner into a USB port on the host machine.
   
2. **Start the required bridge or service**
   - Open a terminal and run `python -m src.mfs100_bridge.bridge` to launch the 32-bit driver bridge.
   - Run `python app.py` in a separate terminal to start the Flask application.
   
3. **Verify device detection**
   - Open the web dashboard at `http://127.0.0.1:5000`.
   - Ensure the UI indicates "Scanner: Physical MFS100 (Online)".
   - Click "Device Diagnostic" to confirm serial number and resolution details.

4. **Capture the first fingerprint**
   - Select the "3-Scan Consensus Mode".
   - Place your LEFT THUMB centrally on the scanner.
   - Click "Capture Fingerprint".
   - Verify the system captures the first impression successfully.

5. **Capture the second fingerprint**
   - Keep your thumb steady on the scanner. 
   - Verify the system captures the second distinct impression automatically after the short delay.

6. **Capture the third fingerprint**
   - Keep your thumb steady on the scanner.
   - Verify the system captures the final third impression.

7. **Verify that all three captures are genuine and distinct**
   - In the logs or consensus breakdown card, ensure 3 separate impressions were processed with unique quality metrics.

8. **Verify that all three images pass through C5 preprocessing**
   - Check the console logs or "Crop Metadata" section in the UI to ensure MFS100 foreground crop, alpha white compositing, CLAHE ridge enhancement, and resize to 112x112 happened correctly.

9. **Verify that the five-checkpoint E2 ensemble produces four probabilities**
   - On the web dashboard, verify that probabilities are generated for A+, B+, AB+, and O+. 
   - Confirm the sum of probabilities is ~100%.

10. **Verify the prediction is displayed and the PDF is generated**
    - Ensure the final predicted blood group is prominently displayed on the interface.
    - Click "Download Clinical PDF Report" and confirm the PDF is generated with the accurate data, images, and medical disclaimer.

**Note:** Mark any test step that has not actually been performed as PENDING during the actual demonstration.
