Ã¯Â»Â¿# FINAL DEMO GUIDE
## Biometric Blood Group Identification System Ã¢â‚¬â€ Step-by-Step Demonstration Procedure

**Document Type:** Demonstration Procedure Guide
**Project:** B.E. Final Year Major Project Ã¢â‚¬â€ Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Audience:** Evaluators, Viva Panel, Project Demonstration

---

## QUICK-START CHECKLIST (Before Beginning)

- [ ] Mantra MFS100 USB scanner connected to PC
- [ ] MFS100 driver installed (`C:\Program Files\Mantra\MFS100\Driver`)
- [ ] 32-bit Python at `C:\Python310-32\python.exe`
- [ ] Project folder: `C:\Users\thanu\OneDrive\Desktop\major project`
- [ ] Left thumb clean and dry

---

## STEP A Ã¢â‚¬â€ Start the MFS100 32-bit Bridge

Open **Terminal 1** (Command Prompt or PowerShell):

```
cd "C:\Users\thanu\OneDrive\Desktop\major project"
C:\Python310-32\python.exe -m src.mfs100_bridge.bridge
```

**Expected output:**
```
[*] MFS100 Bridge starting on port 5100...
[+] MFS100 Bridge running. Waiting for requests.
```

> Keep this terminal open. Do NOT close it during the demo.

---

## STEP B Ã¢â‚¬â€ Start the Flask Backend

Open **Terminal 2** (separate window):

```
cd "C:\Users\thanu\OneDrive\Desktop\major project"
python app.py
```

**Expected output:**
```
[+] Loaded Champion Ensemble weights: {'resnet50': 0.58, 'vgg16': 0.38, 'efficientnetb0': 0.04}
[*] Initializing Champion Ensemble V2 (ResNet50 + VGG16 + EfficientNetB0)...
[+] Champion Ensemble V2 loaded successfully! Locked Accuracy: 85.41%
 * Running on http://127.0.0.1:5000
```

> Model loading takes 30Ã¢â‚¬â€œ60 seconds. Wait for the "Running on" line.

---

## STEP C Ã¢â‚¬â€ Open the Clinical Dashboard

Open a browser and navigate to:
```
http://127.0.0.1:5000
```

The clinical dashboard will load with:
- System title: **Biometric Blood Group Identification System**
- Model status panel (Champion Ensemble V2, 85.41%)
- Scanner status panel
- Medical disclaimer banner

---

## STEP D Ã¢â‚¬â€ Verify Scanner Status

On the dashboard, check the **System Status** panel:
- **Model:** Champion Ensemble V2 | 85.41% | Active
- **Scanner:** Connected (Hardware Mode) OR Mock Mode

If scanner is not connected:
1. Click the **Connect Scanner** button
2. Status should update to show MFS100 hardware connected

If running without physical hardware:
1. Toggle to **Mock Mode** using the mode switch
2. The dashboard will label all captures as MOCK

---

## STEP E Ã¢â‚¬â€ Select Real MFS100 Mode

Ensure the scanner toggle is set to **Physical MFS100** (not Mock Mode).

The dashboard will display the **LEFT THUMB protocol instruction:**
> "Standard Acquisition Finger: LEFT THUMB Ã¢â‚¬â€ Place your LEFT THUMB centrally on the MFS100 sensor platen."

---

## STEP F Ã¢â‚¬â€ Follow LEFT THUMB Protocol

Before capturing, verify:
1. **Left thumb** is selected (not other fingers)
2. Left thumb is **clean and dry**
3. Finger is placed **centrally** on the oval MFS100 sensor platen
4. Press with **moderate, flat pressure** (no rocking or rolling)
5. Keep the thumb **still** until capture completes

---

## STEP G Ã¢â‚¬â€ Capture Fingerprint (Single Scan)

Click the **Capture Fingerprint** button on the dashboard.

**What happens:**
1. Flask sends capture request to 32-bit bridge
2. Bridge invokes MFS100 SDK
3. Scanner captures fingerprint impression
4. Image saved to `data/captured/`
5. Preprocessing: foreground crop Ã¢â€ â€™ CLAHE ridge enhancement Ã¢â€ â€™ 224Ãƒâ€”224 resize
6. Champion Ensemble V2 inference (ResNet50 58% + VGG16 38% + ENB0 4%)
7. Result returned to UI in ~2Ã¢â‚¬â€œ4 seconds

**If "Finger not placed properly" error appears:**
- Remove and re-place the thumb centrally
- Ensure firm, flat contact
- Click Capture again

---

## STEP H Ã¢â‚¬â€ Show Quality / NFIQ Information

After capture, the dashboard shows quality metadata:
- **NFIQ Score:** 1 (Excellent) to 5 (Poor) Ã¢â‚¬â€ lower is better
- **Quality Score:** 0Ã¢â‚¬â€œ100 Ã¢â‚¬â€ higher is better
- **Pressure Status:** OPTIMAL, TOO_LIGHT, or TOO_HEAVY

**For demonstrators:** Point out that NFIQ 1 and quality > 80 indicate an excellent scan, while NFIQ 3+ or quality < 70 would trigger the consensus recommendation banner.

---

## STEP I Ã¢â‚¬â€ Show the Prediction Result

The prediction result card displays:
- **Blood Group** (large, prominent): e.g., O+
- **Confidence:** e.g., 94.59%
- **Status:** NORMAL or LOW_CONFIDENCE

If **NORMAL:** Green status Ã¢â‚¬â€ prediction is above the 60% confidence threshold.

If **LOW_CONFIDENCE:** Red/amber warning Ã¢â‚¬â€ confidence is below 60%.
> "Low-confidence prediction (41.6% < 60%). Please recapture or confirm with standard laboratory blood typing."

**Key talking point:** The model never hides uncertainty. Low-confidence results are always flagged.

---

## STEP J Ã¢â‚¬â€ Show ABO/Rh Analysis

Below the main prediction, the analytical decomposition shows:
- **Derived ABO Group:** e.g., O (66.2% combined probability)
- **Rh Factor:** + or Ã¢Ë†â€™
- **ABO Probabilities:** e.g., A: 12.3%, B: 5.1%, AB: 3.2%, O: 79.4%

If the Rh Uncertainty Advisory is triggered:
> Amber banner: "Rh factor (Ã‚Â±) determination is uncertain. Standard serological laboratory blood typing is required to confirm Rh factor."

**Key talking point:** ABO/Rh decomposition is an analytical layer only. It does NOT change the model's primary 8-class prediction.

---

## STEP K Ã¢â‚¬â€ Show Confidence Warning (If Applicable)

For low-confidence scans, demonstrate:
1. The red LOW_CONFIDENCE badge
2. The advisory message recommending recapture
3. The Smart NFIQ Consensus Tip (appears when quality < 80 or NFIQ Ã¢â€°Â¥ 2):
   > "Consider enabling 3-Scan Consensus for improved accuracy"

---

## STEP L Ã¢â‚¬â€ Run 3-Scan Consensus

Check the **Enable 3-Scan LEFT THUMB Consensus** checkbox.

Click **Start 3-Scan Consensus**.

**What happens:**
1. Scanner captures **3 separate LEFT THUMB impressions** in sequence (with 0.4 s gap between each)
2. Each impression is independently processed and predicted
3. The 3 probability distributions are averaged
4. Consensus prediction = argmax of averaged distribution

Result shows:
- Consensus blood group and confidence
- **Agreement rate:** e.g., "3/3 impressions agreed" or "2/3 impressions agreed"
- Average quality across all 3 scans
- DB label: `MFS100 Consensus - Left Thumb (3 Impressions)`

**Key talking point:** Consensus reduces the effect of single-touch angle variance and pressure variation.

---

## STEP M Ã¢â‚¬â€ Generate and View PDF Report

Click the **Download Report** button (or navigate to the report URL).

The clinical PDF contains:
- Report ID (BG-XXXXXX format)
- Timestamp
- Input source (with LEFT THUMB note for hardware)
- Model name: Champion Ensemble V2
- Confidence percentage
- Predicted blood group (32pt display)
- Derived ABO group and Rh factor
- Rh Uncertainty Advisory (amber box, if applicable)
- Full 8-class probability table
- Mandatory medical disclaimer (red box at top)
- No claim of clinical diagnosis anywhere in the document

**Key talking point for viva:** Show the medical disclaimer. Emphasize that the system explicitly states it is "AI-based preliminary estimation only" and mandates laboratory confirmation.

---

## STEP N Ã¢â‚¬â€ Open Audit History

Click **History** in the navigation or navigate to:
```
http://127.0.0.1:5000/history
```

The audit page shows:
- All 200+ prediction records in reverse chronological order
- Columns: ID, Timestamp, Input Type, Blood Group, Confidence, Status, Finger Used
- `finger_used = LEFT_THUMB` for hardware captures
- `finger_used = NULL` for uploads and mock captures
- Full audit trail from all demo sessions

---

## STEP O Ã¢â‚¬â€ Demonstrate Upload Mode

1. Click the **Upload Image** tab on the dashboard
2. Select a fingerprint image file (.bmp / .png / .jpg)
3. The system:
   - Runs standard preprocessing (no hardware crop)
   - Runs Champion Ensemble V2 inference
   - Returns prediction with ABO/Rh analysis
   - `finger_used` is NOT set (remains NULL in database)
   - No LEFT THUMB metadata in upload PDF

**Key talking point:** Upload mode and hardware mode share the same model and preprocessing foundation. The only differences are: no foreground crop, no ridge enhancement, no finger_used metadata.

---

## STEP P Ã¢â‚¬â€ Demonstrate Mock Mode

1. Click the **Toggle Mock Mode** button on the dashboard
2. Status updates to: **MOCK MODE Ã¢â‚¬â€ Physical Scanner Not Active**
3. Click **Capture (Mock)**

The system:
- Uses sample fingerprint images from `data/sample/` (one per blood group class)
- Runs identical preprocessing and inference pipeline
- Labels result and DB record clearly as **MFS100 (Mock)**
- `finger_used` is NOT set

**Key talking point:** Mock mode is identical to real hardware mode except for the image source. It is used for demos when the MFS100 is unavailable and for all automated testing.

---

## COMMON DEMO TALKING POINTS

| Topic | Key Statement |
|---|---|
| Why fingerprints? | Dermatoglyphic ridge patterns show statistical correlation with blood group phenotype in published literature. |
| Why not 100% accurate? | Biological variation and limited training data mean probabilistic estimation, not deterministic classification. |
| What is 85.41%? | Official one-time evaluation on 767/898 locked test images Ã¢â‚¬â€ honest, not fabricated. |
| Why Champion Ensemble? | Ensemble soft-voting outperforms any single model by 5Ã¢â‚¬â€œ28 percentage points. |
| What is LEFT THUMB? | Acquisition protocol for demo reproducibility only Ã¢â‚¬â€ the model never receives finger identity. |
| What about AB+? | AB+ has a 6.60% O- confusion rate on validation data Ã¢â‚¬â€ openly documented and mitigated by LOW_CONFIDENCE flagging. |
| Is this a medical device? | NO. Academic research prototype only. All results require mandatory laboratory confirmation. |

---

## TROUBLESHOOTING

| Problem | Solution |
|---|---|
| Bridge not starting | Check C:\Python310-32 exists; check MFS100 driver installed |
| "Finger not placed properly" | Re-place LEFT THUMB with firm, flat, central contact |
| Model loading slow | Normal Ã¢â‚¬â€ 3 large models (482 MB total) loading from disk |
| LOW_CONFIDENCE result | Expected behavior Ã¢â‚¬â€ recapture or use 3-scan consensus |
| Scanner shows "disconnected" | Click Connect Scanner button; check USB cable |
| Mock mode always on | Set environment variable MFS100_MOCK=false and restart |

---

*This document is part of the B.E. Final Year Major Project Ã¢â‚¬â€ Biometric Blood Group Identification System.*
*For evaluator, viva panel, and demonstration use. September 2026.*
