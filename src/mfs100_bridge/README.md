# Mantra MFS100 32-bit HTTP Bridge

## Overview
This bridge is an isolated 32-bit Python service that loads the 32-bit `MANTRA.MFS100.dll` via `pythonnet` and exposes a REST HTTP API on `http://127.0.0.1:5100`.

This architecture enables any 64-bit Python application (such as the main Blood Group Prediction Flask server) to communicate seamlessly with the physical MFS100 scanner without running into 32-bit/64-bit DLL architecture conflicts (`[WinError 193]`).

## Environment
- **Python Executable:** `C:\Python310-32\python.exe` (32-bit x86)
- **Dependencies:** `flask`, `pythonnet`, `pillow`
- **Host / Port:** `127.0.0.1:5100`

## How to Start the Bridge
Run from project root using the 32-bit Python executable:
```powershell
C:\Python310-32\python.exe -m src.mfs100_bridge.bridge
```

## API Endpoints

### 1. `GET /status`
Returns bridge health, Python architecture, SDK load state, and device status.

### 2. `POST /connect`
Initializes connection to physical scanner via `MFS100.Init()`. Returns device serial, model, make, width, and height.

### 3. `POST /disconnect`
Releases scanner handle via `MFS100.Uninit()`.

### 4. `GET /device-info`
Returns connected device specs.

### 5. `POST /capture`
Triggers optical scan via `MFS100.AutoCapture()`. Returns base64 encoded BMP image bytes, quality (0-100), and NFIQ score (1-5).
