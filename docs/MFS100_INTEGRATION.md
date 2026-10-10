# Mantra MFS100 Hardware Integration Guide

## 1. Architectural Overview

The biometric blood group identification system uses an **out-of-process 32-bit HTTP bridge architecture** to integrate the physical Mantra MFS100 optical fingerprint scanner with the 64-bit Python machine learning environment.

```
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š               Physical Mantra MFS100 Scanner           Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                           Ã¢â€â€š USB Cable / Driver
                           Ã¢â€“Â¼
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š         32-bit Bridge Service (127.0.0.1:5100)         Ã¢â€â€š
Ã¢â€â€š  - Runtime: C:\Python310-32\python.exe (32-bit x86)    Ã¢â€â€š
Ã¢â€â€š  - Managed Wrapper: MANTRA.MFS100.dll (v9.0.2.5)       Ã¢â€â€š
Ã¢â€â€š  - Native Driver: MFS100Dll.dll (32-bit C)             Ã¢â€â€š
Ã¢â€â€š  - Framework: Flask + pythonnet (clr)                  Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Â¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
                           Ã¢â€â€š Localhost HTTP REST API
                           Ã¢â€â€š (Base64 BMP payload)
                           Ã¢â€“Â¼
Ã¢â€Å’Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€
Ã¢â€â€š          Main Clinical Flask Web Application           Ã¢â€â€š
Ã¢â€â€š  - Runtime: 64-bit Python (Python 3.10)                Ã¢â€â€š
Ã¢â€â€š  - Port: 127.0.0.1:5000                                Ã¢â€â€š
Ã¢â€â€š  - Adapter: src/mfs100/scanner.py                      Ã¢â€â€š
Ã¢â€â€š  - Central Preprocessing: PIL image standardization    Ã¢â€â€š
Ã¢â€â€š  - AI Model: ResNet50 (224Ãƒâ€”224 RGB, Softmax 8-class)   Ã¢â€â€š
Ã¢â€â€š  - Storage: SQLite audit database (predictions.db)     Ã¢â€â€š
Ã¢â€â€š  - Clinical Documentation: ReportLab PDF Generator     Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€Ëœ
```

---

## 2. Why the 32-bit Bridge Architecture Was Required

The Mantra MFS100 SDK distribution (`C:\Program Files\Mantra\MFS100\Driver\MFS100Test\`) provides:
1. `MANTRA.MFS100.dll` Ã¢â‚¬â€ 32-bit .NET managed wrapper
2. `MFS100Dll.dll` Ã¢â‚¬â€ 32-bit Win32 native C library

Loading a 32-bit native or managed DLL directly into a 64-bit Python process produces:
`[WinError 193] %1 is not a valid Win32 application`

Because no 64-bit native `MFS100Dll.dll` is provided by the vendor, the 32-bit HTTP bridge isolates the 32-bit SDK in its native architecture and serves standard JSON/HTTP over `127.0.0.1:5100`. The 64-bit Flask web server consumes the bridge endpoints without any architectural conflicts.

---

## 3. How to Run the System

### Step 1: Start the 32-bit MFS100 Bridge
In a terminal, run using the 32-bit Python executable:
```powershell
C:\Python310-32\python.exe -m src.mfs100_bridge.bridge
```
Output:
```text
2026-09-04 18:19:32 [INFO] [MFS100-Bridge] MANTRA.MFS100 .NET assembly loaded successfully.
2026-09-04 18:19:32 [INFO] [MFS100-Bridge] Starting MFS100 32-bit Bridge on http://127.0.0.1:5100
```

### Step 2: Start the Main Flask Application
In a second terminal, run using your normal 64-bit Python environment:
```powershell
python app.py
```
Visit the clinical dashboard in your browser:
```
http://127.0.0.1:5000/
```

---

## 4. Bridge REST API Reference (Port 5100)

| Method | Route | Description |
|---|---|---|
| `GET` | `/status` | Bridge health, Python architecture, SDK load state, and connection status |
| `POST` | `/connect` | Initializes MFS100 scanner (`MFS100.Init()`) |
| `POST` | `/disconnect` | Releases scanner handle (`MFS100.Uninit()`) |
| `GET` | `/device-info` | Hardware specs (Serial, Make, Model, Width, Height) |
| `POST` | `/capture` | Triggers biometric auto-capture with quality threshold and timeout |

### Example Capture Response
```json
{
  "success": true,
  "image_b64": "Qk1+GAAAAAAA...",
  "quality": 82,
  "nfiq": 1,
  "width": 316,
  "height": 354,
  "format": "BMP"
}
```

---

## 5. Mock Mode vs. Physical Hardware Support

The application supports both modes seamlessly:
- **Mock Mode (`MFS100_MOCK=true`)**:
  - Automatically enabled if the bridge is offline or when selected in the UI.
  - Pulls authentic fingerprint samples from `data/sample/`.
  - Supports simulating specific blood group classes for automated testing and demonstrations.
- **Physical Hardware Mode (`MFS100_MOCK=false`)**:
  - Automatically queries the bridge at `http://127.0.0.1:5100`.
  - Triggers physical scanner optical illumination and fingerprint auto-capture.
  - Verifies image integrity (BMP magic bytes, dimensions, minimum file size).

You can switch between Mock Mode and Physical Hardware at any time directly from the clinical dashboard using the **"Switch to Physical Scanner"** / **"Switch to Mock Mode"** button.

---

## 6. Biometric AI Pipeline Verification

Captured images from the physical MFS100 scanner follow the exact same verified clinical pipeline as uploaded images:
1. **Raw Acquisition**: 316Ãƒâ€”354 8-bit grayscale BMP from physical scanner.
2. **Standardization**: Converted to 224Ãƒâ€”224 RGB standardized tensor via `src/preprocessing.py`.
3. **Inference**: Evaluated by the trained deep neural network model (`models/blood_group_resnet50.keras`).
4. **Reliability Check**: Confidence score evaluated against clinical threshold (60.0%). Low-confidence predictions flag an alert.
5. **Audit Trail**: Every transaction recorded in SQLite database (`predictions.db`).
6. **Clinical Report**: Automated PDF report generated via ReportLab (`reports/`).

---

## 7. Verification Results

All 38 test cases across unit, bridge, scanner adapter, and end-to-end integration pass with 100% success:
```text
Ran 38 tests in 18.016s
OK
```
Bridge unit tests in 32-bit Python:
```text
Ran 5 tests in 2.884s
OK
```
Hardware communication confirmed:
- Physical device detected: **Mantra MFS100**
- Serial Number: **11215389**
- Dimensions: **316 Ãƒâ€” 354**
