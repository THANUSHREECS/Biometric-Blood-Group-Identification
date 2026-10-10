# MFS100 SDK Audit Report

**Project:** Blood Group Prediction — MFS100 Integration
**Audit Date:** 2026-09-04
**Auditor:** Automated SDK Inspection
**Status:** Complete

---

## 1. SDK Location

```
C:\Program Files\Mantra\MFS100\
Driver\
    Driver\
        Win-10-X64\MFS100CI.dll   (64-bit USB class driver)
        Win-10-X86\MFS100CI.dll   (32-bit USB class driver)
    MFS100Test\                   SDK DLLs and test application
        MFS100Dll.dll              (32-bit native C) core SDK
        MANTRA.MFS100.dll          (32-bit .NET C#) managed wrapper
        iengine_ansi_iso.dll       (32-bit native C) ANSI/ISO template engine
        MANTRA.MFS100.Test.exe     (32-bit .NET) official test app
        FingerData\
            FingerImage.bmp        112,942 bytes - SAMPLE CAPTURE IMAGE
            RawData.raw            111,864 bytes
            ISOImage.iso           111,910 bytes
            WSQImage.wsq           10,434 bytes
            AnsiTemplate.ansi      368 bytes
            ISOTemplate.iso        366 bytes
    Redist\
        vcredist_x86.exe           Visual C++ 2013 x86 runtime (required)
        vcredist_x64.exe           Visual C++ 2013 x64 runtime

C:\Program Files\Mantra\RDService\MFS100\
    ConfigMantraRDService.exe      UIDAI/STQC Registered Device config tool
    BouncyCastle.Crypto.dll        Crypto for Aadhaar signing
    Newtonsoft.Json.dll            JSON for RD service protocol
    mantra.loger.dll
```

## 2. SDK Version

| Component | Version | Architecture | Type |
|---|---|---|---|
| MANTRA.MFS100.dll | 9.0.2.5 | x86 (32-bit) | .NET Managed Assembly (C#) |
| MFS100Dll.dll | 9.0.2.5 | x86 (32-bit) | Native Win32 C DLL |
| iengine_ansi_iso.dll | - | x86 (32-bit) | Native Win32 C DLL |
| MFS100CI.dll (Win-10-X64) | - | x86-64 (64-bit) | Kernel USB CI driver |
| MFS100CI.dll (Win-10-X86) | - | x86 (32-bit) | Kernel USB CI driver |
| MANTRA.MFS100.Test.exe | - | x86 (32-bit) | .NET Test Application |
| Company | Mantra Softech India Pvt. Ltd. | | Copyright 2016 |

## 3. DLL Architecture - The Critical Issue

Your main Python environment:
  python.exe -> 64-bit (x86-64)
  Path: C:\Users\thanu\AppData\Local\Programs\Python\Python310\python.exe

MFS100Dll.dll -> 32-bit (x86)

This is why [WinError 193] occurs:
A 64-bit process cannot load a 32-bit DLL via ctypes.CDLL.

No 64-bit MFS100Dll.dll exists in the installation.

MANTRA.MFS100.dll contains both MANTRA.Dll (wraps 32-bit) and MANTRA.Dll64
(wraps 64-bit) classes. However MANTRA.Dll64 requires a 64-bit MFS100Dll.dll
which does NOT ship with this SDK. Only 32-bit exists.

No 32-bit Python installed on this machine. Only Python 3.10 64-bit.
A 32-bit Python must be installed separately for the bridge process.

## 4. Complete API Surface

### 4.1 MANTRA.MFS100 class (recommended application interface)

Lifecycle:
  Init()                    Initialize with auto-detected device
  Init(serialNo)            Initialize with specific serial number
  Init(serialNo, key)       Initialize with license key
  Uninit()                  Release device
  IsConnected()             bool - check connection status
  Dispose()                 IDisposable cleanup

Device Information:
  GetDeviceInfo()           Returns DeviceInfo {SerialNo, Model, Make, Width, Height}
  GetSDKVersion()           Returns SDK version string

Capture:
  StartCapture(quality, timeout)   Start live preview, fire OnCaptureCompleted
  StopCapture()                    Stop ongoing capture
  AutoCapture(quality, timeout)    Blocking auto-capture with quality threshold

Events:
  OnPreview           += handler(FingerData)         Live preview frames
  OnCaptureCompleted  += handler(FingerData, int)    int = error code
  OnMFS100Attached    += handler()                   USB plug event
  OnMFS100Detached    += handler()                   USB unplug event

Utility:
  MatchISO(probe, gallery)
  MatchANSI(probe, gallery)
  RotateImage(data, width, height, angle)
  BitmapToBytes(bitmap)
  RawToBitmapBytes(raw, w, h)      Convert raw -> BMP bytes
  BytesToBitmap(data)
  GetErrorMsg(errorCode)
  GetCertification()

### 4.2 FingerData - Capture Result Object

  FingerImage     byte[]    BMP image bytes -- THIS IS WHAT WE NEED FOR THE CNN
  RawData         byte[]    Raw grayscale pixel data
  ISOTemplate     byte[]    ISO 19794-2 minutiae template
  ISOImage        byte[]    ISO 19794-4 fingerprint image
  ANSITemplate    byte[]    ANSI 378 minutiae template
  WSQImage        byte[]    WSQ compressed image
  Quality         int       Quality score 0-100
  Nfiq            int       NFIQ quality score 1-5 (lower is better)
  InWidth         int       Image width pixels
  InHeight        int       Image height pixels
  InArea          int       Finger area pixels
  Resolution      int       DPI resolution
  GrayScale       int
  Bpp             int       Bits per pixel
  WSQCompressRatio double

### 4.3 DeviceInfo Object

  SerialNo   string    Hardware serial number
  Model      string    Device model e.g. "MFS100"
  Make       string    Manufacturer e.g. "MANTRA"
  Width      int       Sensor width pixels
  Height     int       Sensor height pixels

### 4.4 Error Codes (MANTRA.MFS100Exception)

  MFS100_E_SUCCESS                          =  0
  MFS100_E_NO_DEVICE                        = -1   Scanner not connected
  MFS100_E_INVALIDPARAM                     = -2
  MFS100_E_NULLPARAM                        = -3
  MFS100_E_MEMORY                           = -4
  MFS100_E_CORRUPT_SERIAL                   = -5
  MFS100_E_UNKNOWN_SENSOR                   = -6
  MFS100_E_LOAD_FIRMWARE_FAILED             = -7
  MFS100_E_NOT_INITIALIZED                  = -8
  MFS100_E_SYNC_PROBLEM                     = -9
  MFS100_E_OTHER                            = -10
  MFS100_E_TIMEOUT                          = -11  Capture timeout
  MFS100_E_NOT_GOOD_IMAGE                   = -12  Image quality too low
  MFS100_E_BLANKIMAGE                       = -13  No finger / blank image
  MFS100_E_EXTRACTOR_INIT_FAILED            = -14
  MFS100_E_FILE_IO                          = -15
  MFS100_E_BAD_LICENSE                      = -16
  MFS100_E_BAD_FORMAT                       = -17
  MFS100_E_BAD_VALUE                        = -18
  MFS100_E_BAD_TEMPLATE                     = -19
  MFS100_E_READ_ONLY                        = -20
  MFS100_E_NOT_DEFINED                      = -21
  MFS100_E_NULL_TEMPLATE                    = -22
  MFS100_E_NOT_IMPLEMENTED                  = -23
  MFS100_E_BAD_QUALITY                      = -24
  MFS100_E_CAPTURING_STOPPED               = -25
  MFS100_E_LATENT_FINGER                    = -26  Latent print detected
  MFS100_E_ALREADY_START_STOP              = -27
  MFS100_E_INVALID_KEY                      = -28
  MFS100_E_IMPROPER_OR_NO_FINGER_PLACEMENT = -29
  MFS100_E_UNHANDLEDEXCEPTION              = -30
  MFS100_E_INVALID_REQUEST                  = -31

## 5. Sample Fingerprint Image Format (Confirmed)

File: FingerImage.bmp (actual scanner capture)

| Property | Value |
|---|---|
| Format | Windows BMP |
| Width | 316 pixels |
| Height | 354 pixels |
| Bit depth | 8 bits per pixel |
| Color space | Grayscale (8-bit palette) |
| Size | 112,942 bytes |

CNN pipeline conversion required:
  8-bit grayscale BMP (316x354)
  -> PIL Image mode L
  -> convert('RGB')
  -> resize to 224x224
  -> per-model normalization (ResNet50/VGG16/EfficientNetB0)

This is fully supportable by the existing preprocessing.py pipeline.

## 6. Required Dependencies for 32-bit Bridge

| Dependency | Status | Action |
|---|---|---|
| 32-bit Python 3.8-3.10 | NOT INSTALLED | Must install from python.org |
| pythonnet (clr) | NOT INSTALLED | pip install pythonnet (in 32-bit Python) |
| flask | NOT INSTALLED in 32-bit env | pip install flask (in 32-bit Python) |
| pillow | NOT INSTALLED in 32-bit env | pip install pillow (in 32-bit Python) |
| MANTRA.MFS100.dll | PRESENT | C:\Program Files\Mantra\MFS100\Driver\MFS100Test\ |
| MFS100Dll.dll | PRESENT | Same directory |
| iengine_ansi_iso.dll | PRESENT | Same directory |
| Visual C++ 2013 x86 Runtime | LIKELY INSTALLED | Redist\vcredist_x86.exe if missing |
| MFS100 USB driver | INSTALLED | MFS100CI.dll loaded via kernel |

## 7. Recommended Architecture: pythonnet + MANTRA.MFS100.dll

32-bit Python
    -> import clr (pythonnet)
    -> MANTRA.MFS100.dll  (.NET managed wrapper)
    -> P/Invoke internally to MFS100Dll.dll (native 32-bit C)
    -> USB
    -> Physical Scanner

Reason: Uses the official high-level C# API. AutoCapture() handles quality
checking and timeout internally. FingerData.FingerImage returns ready-to-use
BMP byte array. Best-documented code path matching SDK examples.

## 8. Bridge API Design

Bound to 127.0.0.1 only. Configurable port (default 5100).

  GET  /status       {"running": true, "version": "..."}
  POST /connect      {"success": true, "device": {...}} or {"error": "..."}
  POST /disconnect   {"success": true}
  GET  /device-info  {"serial": "...", "model": "...", "make": "...", "width": N, "height": N}
  POST /capture      {"success": true, "image_b64": "...", "quality": N, "nfiq": N}
                  or {"error": "...", "error_code": N}

Image returned as base64-encoded BMP bytes (no temp files needed).

## 9. Action Items Before Implementation

1. Install 32-bit Python 3.10 from python.org (REQUIRED)
2. Install pythonnet in 32-bit Python (REQUIRED)
3. Install flask and pillow in 32-bit Python (REQUIRED)
4. Verify Visual C++ 2013 x86 runtime (VERIFY)
5. Confirm MFS100 USB driver loaded in Device Manager (VERIFY)
6. Create src/mfs100_bridge/ directory structure (READY)
7. Write bridge.py (READY after deps installed)
8. Refactor src/mfs100/scanner.py as real/mock adapter (READY)

## 10. Files That Must NOT Be Modified

  models/blood_group_resnet50.keras
  models/vgg16_blood_group.keras
  models/efficientnetb0_blood_group.keras
  outputs/final_evaluation_results.json   (85.41% official result)

## 11. Mock Mode

When MFS100_MOCK=true:
  - Bridge is NOT started
  - Scanner adapter returns FingerData\FingerImage.bmp as sample
  - UI clearly shows MOCK MODE warning
  - input_type in database = "MFS100_MOCK"
