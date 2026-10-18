# HARDWARE VERIFICATION CHECKLIST: MANTRA MFS100 & DUAL-PROCESS BRIDGE

**Date:** 2026-10-11  
**Target Hardware:** Mantra MFS100 Optical Fingerprint Scanner (500 DPI, USB 2.0)  
**System Architecture:** Dual-Process Architecture (32-Bit C# Bridge + 64-Bit Flask AI Engine)  

---

## 1. Dual-Process Architecture Specification

The Mantra MFS100 hardware SDK is distributed exclusively as 32-bit native Windows libraries (`MANTRA.MFS100.dll`). Because modern deep learning frameworks (TensorFlow 2.11+) require a 64-bit Python runtime, HemoPrint AI separates hardware acquisition and neural network inference into two communicating processes:

```
┌────────────────────────────────────────────────────────┐
│                   Mantra MFS100 USB                    │
│             Optical Fingerprint Sensor                 │
└───────────────────────────┬────────────────────────────┘
                            │ USB 2.0 Raw Capture
                            ▼
┌────────────────────────────────────────────────────────┐
│        PROCESS 1: 32-Bit Hardware Bridge               │
│   Executable: C:\Python310-32\python.exe               │
│   Script: python -m src.mfs100_bridge.bridge           │
│   Port: http://127.0.0.1:5100                          │
│   Role: Loads 32-bit MANTRA.MFS100.dll via pythonnet   │
│         Exposes REST API: /status, /capture, /info     │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP JSON / Base64 BMP
                            ▼
┌────────────────────────────────────────────────────────┐
│        PROCESS 2: 64-Bit Production AI Server          │
│   Executable: 64-Bit Virtualenv Python                 │
│   Script: python app.py                                │
│   Port: http://127.0.0.1:5000                          │
│   Role: Flask Dashboard, C5 Preprocessor,              │
│         E2 5-Checkpoint Metric Learning Ensemble       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Verification Status

> [!IMPORTANT]
> **VERIFICATION DISCLOSURE**  
> - **Dual-Process HTTP Bridge:** **VERIFIED ONLINE & OPERATIONAL**. Process communication over `127.0.0.1:5100` and `127.0.0.1:5000` is functioning seamlessly.
> - **Simulated / Mock Acquisition Pipeline:** **VERIFIED PASSING (100% Tests Passed)**. Automated integration tests successfully acquire synthetic optical scans, execute C5 preprocessing, and run 5-checkpoint E2 inference.
> - **Physical Optical USB Scanner Verification:** **CONDITIONAL**. Physical USB device capture is contingent upon the physical Mantra MFS100 unit being connected to a live USB 2.0/3.0 port and driver initialization by the host OS. A physical hardware test must be completed on-site with the live scanner plugged in before demonstration.

---

## 3. Pre-Demonstration Hardware Verification Checklist

Use this checklist during morning setup prior to any viva or live presentation:

### Phase A: Physical Hardware & Windows Drivers
- [ ] **A.1:** Plug the Mantra MFS100 USB cable into a direct motherboard USB port (avoid unpowered USB hubs).
- [ ] **A.2:** Open Windows Device Manager (`devmgmt.msc`) and confirm `Mantra MFS100 Optical Fingerprint Scanner` is listed under Universal Serial Bus devices with no warning triangle.
- [ ] **A.3:** Confirm Mantra MFS100 Driver v9.2.0.0+ is installed in `C:\Program Files\Mantra\MFS100\Driver`.

### Phase B: 32-Bit Bridge Daemon Verification
- [ ] **B.1:** Confirm Process 1 is running:
  ```powershell
  python -m src.mfs100_bridge.bridge
  ```
- [ ] **B.2:** Test the bridge health endpoint via browser or PowerShell:
  ```powershell
  curl http://127.0.0.1:5100/status
  ```
  Expected output: `{"sdk_loaded": true, "status": "online", "device_connected": true}`.

### Phase C: 64-Bit Flask Production Server Verification
- [ ] **C.1:** Confirm Process 2 is running:
  ```powershell
  python app.py
  ```
- [ ] **C.2:** Navigate to `http://localhost:5000` in Google Chrome or Microsoft Edge.
- [ ] **C.3:** Verify the scanner status widget on the clinical dashboard displays a green indicator: `Mantra MFS100: CONNECTED`.

### Phase D: Live 3-Scan Acquisition & Preprocessing Test
- [ ] **D.1:** Prompt the participant to clean their index finger with a dry tissue (ensuring no excess oil or moisture).
- [ ] **D.2:** Place finger on the optical prism when the green sensor LED illuminates.
- [ ] **D.3:** Capture Impression 1 (Scanner buzzer will beep upon successful acquisition).
- [ ] **D.4:** Lift finger and reposition for Impression 2.
- [ ] **D.5:** Lift finger and reposition for Impression 3.
- [ ] **D.6:** Verify that the UI displays the 3 cropped and CLAHE-contrast-enhanced ridge previews.

### Phase E: Ensemble Inference & Safety Layer
- [ ] **E.1:** Verify that inference completes in $< 2.0$ seconds.
- [ ] **E.2:** Verify that the predicted blood group (`A+`, `B+`, `AB+`, or `O+`) is displayed alongside the 4-class probability distribution bar chart.
- [ ] **E.3:** If confidence is $< 0.70$, verify that the amber advisory banner appears: `Confirmation Recommended: Low Confidence`.
- [ ] **E.4:** Verify that the mandatory laboratory serology confirmation disclaimer is visible at the bottom of the screen and on downloaded PDF reports.

---

## 4. Fault Handling & Fallback Protocols

| Symptom | Root Cause | Immediate Remediation |
| :--- | :--- | :--- |
| **"Scanner Disconnected" in UI** | USB disconnected or bridge process stopped | 1. Re-seat USB cable.<br>2. Check `python -m src.mfs100_bridge.bridge` terminal for disconnect logs.<br>3. Restart bridge. |
| **Bridge reports `load_error`** | Missing 32-bit Python environment | Ensure command uses `C:\Python310-32\python.exe`. 64-bit Python cannot load the 32-bit Mantra DLL. |
| **"Low Quality Image" error** | Finger placed too lightly or dry skin | Ask participant to apply moderate, even pressure. If finger is very dry, rub palms together to restore natural moisture. |
| **Physical scanner unavailable** | Hardware fault during live demo | In `src/config.py`, set `MFS100_MOCK = True`. Restart `app.py`. The application will seamlessly serve synthetic optical impressions for demonstration purposes. |

---

## 5. Mandatory Medical Disclaimer

> [!CAUTION]
> **RESEARCH PROTOTYPE DISCLAIMER**  
> The Mantra MFS100 optical scanner is an optical identity biometric device, not an in-vitro diagnostic instrument. Fingerprint acquisition cannot detect circulating red blood cell agglutinins or ABO surface antigens. Predictions must always be verified by clinical laboratory testing.
