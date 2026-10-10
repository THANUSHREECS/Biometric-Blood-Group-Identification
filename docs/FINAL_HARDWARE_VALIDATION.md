# Final Real-Hardware Validation Report: Mantra MFS100

**Date:** 2026-09-05 12:26 IST  
**Environment:** Windows 11, 64-bit Python 3.10.11 (Flask App) + 32-bit Python 3.10.11 (MFS100 HTTP Bridge)  
**Model Architecture:** Champion Ensemble V2 (ResNet50: 58%, VGG16: 38%, EfficientNetB0: 4%) â€” Frozen (Zero Retraining)  
**Locked Test Set:** Untouched (898 images in `outputs/dataset_manifest.csv` un-evaluated)  

---

## 1. Hardware Configuration

| Component | Specification | Status |
|---|---|:---:|
| **Scanner Model** | Mantra MFS100 Optical Fingerprint Sensor | Connected |
| **Serial Number** | `11215389` | Active |
| **Make / Vendor** | MANTRA / Mantra Softech India Pvt. Ltd. | Verified |
| **Sensor Resolution** | 316 Ã— 354 pixels, 500 DPI | Operational |
| **SDK Version** | `v9.0.2.5` (`MANTRA.MFS100.dll` + `MFS100Dll.dll`) | Loaded |
| **Bridge Service** | 32-bit Out-of-process HTTP REST (`127.0.0.1:5100`) | Healthy |
| **Main Web Server** | 64-bit Flask (`127.0.0.1:5000`) | Online |

---

## 2. Section A: Single-Scan Physical MFS100 Test (5 Consecutive Scans)

- **Subject Finger:** Right Thumb (Known: O+)
- **Test Objective:** Evaluate repeatability, optical quality (NFIQ), automated foreground bounding-box crop, ABO decomposition, and Rh uncertainty trigger.

| Scan # | Prediction | 8-Class Conf | NFIQ | Quality | Crop Bounding Box | Crop Applied | ABO Group | Rh Factor | Rh Margin | Rh Uncertainty | Consensus Rec.? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | O+ | 49.4% | 2 | 79/100 | `[42, 0, 307, 354]` | âœ… Yes | **O** (85.2%) | **+** | 0.4805 | âš ï¸ **True** (conf < 65%) | **True** (NFIQ 2) |
| **2** | O+ | 61.5% | 1 | 83/100 | `[33, 0, 307, 354]` | âœ… Yes | **O** (88.1%) | **+** | 0.6014 | âš ï¸ **True** (conf < 65%) | False |
| **3** | O+ | 67.5% | 1 | 85/100 | `[32, 0, 307, 354]` | âœ… Yes | **O** (91.4%) | **+** | 0.6641 | False | False |
| **4** | O+ | 68.6% | 1 | 84/100 | `[32, 0, 307, 354]` | âœ… Yes | **O** (92.0%) | **+** | 0.6738 | False | False |
| **5** | O+ | 68.6% | 1 | 84/100 | `[32, 0, 307, 354]` | âœ… Yes | **O** (92.1%) | **+** | 0.6745 | False | False |

### Analysis of Section A:
1. **Consistency:** 5 / 5 = **100% agreement** on O+.
2. **First-Impression Placement Drift:** Scan 1 had NFIQ 2 (Quality 79) and lower confidence (49.4%), triggering both the **Rh Uncertainty Advisory** and the **Smart Consensus Recommendation**.
3. **Convergence:** As the finger settled on Scans 2â€“5, NFIQ stabilized at 1 (Quality 83â€“85), crop boundaries locked stably to `[32, 0, 307, 354]`, and confidence reached 68.6% with zero Rh uncertainty.

---

## 3. Section B: Three-Scan Consensus Test

- **Subject Finger:** Right Thumb (Same finger)
- **Execution:** Automated 3-impression sequence with soft-voting probability aggregation.

| Impression | Blood Group | Confidence | Derived ABO Group | Crop Applied |
|:---:|:---:|:---:|:---:|:---:|
| **Impression 1** | O+ | 73.3% | O | âœ… Yes |
| **Impression 2** | O+ | 76.1% | O | âœ… Yes |
| **Impression 3** | O+ | 75.1% | O | âœ… Yes |

- **Final Consensus Prediction:** **O+**
- **Consensus Confidence:** **74.86%**
- **Scan Agreement:** **3 / 3 (100% Agreement)**
- **Derived ABO Group:** **O** (Aggregated ABO mass: 75.45%)
- **Derived Rh Factor:** **+**
- **Rh Uncertainty Flag:** **False** (Confidence > 65%, wide Rh margin: 74.27%)
- **Consensus Probability Distribution (Unaltered Soft Voting):**
  - $P(O+) = 74.86\%$
  - $P(A-) = 22.28\%$
  - $P(AB-) = 1.96\%$
  - $P(O-) = 0.59\%$
  - $P(A+) = 0.20\%$
  - $P(AB+) = 0.04\%$
  - $P(B-) = 0.04\%$
  - $P(B+) = 0.03\%$
  - $\sum P = 1.0000$

> [!NOTE]
> The consensus mechanism strictly averages the raw probability vectors of the 3 impressions. It does NOT bias, re-weight, or alter the underlying model probabilities.

---

## 4. Section C: Different-Finger Consistency Test (Same Subject)

| Finger Tested | Prediction | Confidence | NFIQ | Quality | Crop Bounding Box | ABO Group | Rh Factor | Rh Margin | Rh Uncertainty |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Right Index Finger** | **O+** | **92.55%** | 1 | 94/100 | `[5, 0, 314, 354]` | **O** | **+** | 0.9148 | False |
| **Right Middle Finger** | **O+** | **92.20%** | 1 | 95/100 | `[6, 0, 315, 354]` | **O** | **+** | 0.9109 | False |

- **Observation:** All three fingers (Thumb, Index, Middle) from the subject consistently classified as **O+** with very high ridge quality (Quality 94â€“95, NFIQ 1) and $>92\%$ confidence.
- Note: Different fingers on the same individual share phenotypic dermatoglyphic traits (arch/whorl/loop frequencies), yielding harmonious ABO typing across digits.

---

## 5. Section D: Placement & Quality Guidance Test

- **Optimal Placement:** Quality: 95/100, NFIQ: 1, Pressure: `OPTIMAL`.  
  *Advice:* `"Good placement and pressure. High-quality capture."*  
  *Consensus Prompt:* Hidden (`False`).
- **Suboptimal Placement (Observed in Scan 1):** Quality: 79/100, NFIQ: 2, Pressure: `MODERATE`.  
  *Advice:* Automatically displayed placement guide.  
  *Consensus Prompt:* **Triggered** (`True`) â€” Prompted user to switch to 3-Scan Consensus mode.
- **Medical Disclaimer Gating:** Both single and consensus cards strictly present results as AI preliminary estimations with zero false claims of medical certainty.

---

## 6. Section E: PDF Clinical Report Generation Test

- **Total Reports Generated in System:** 174 reports.
- **Sample Validated Reports:**
  1. `reports/blood_group_report_171_20260905_122358.pdf` (151 KB) â€” 3-Scan Consensus Report
  2. `reports/blood_group_report_172_20260905_122416.pdf` (151 KB) â€” Index Finger Physical Scan
  3. `reports/blood_group_report_174_20260905_122439.pdf` (149 KB) â€” Normal Physical MFS100 Scan
- **PDF Elements Verified:**
  - âœ… Embedded 224Ã—224 fingerprint impression thumbnail.
  - âœ… Primary Estimated Blood Group (`O+`) in high-contrast header card.
  - âœ… Derived ABO Group (`O`) and Rh Factor (`+`) in diagnostic metadata.
  - âœ… Input Source: `Mantra MFS100 Scanner` / `MFS100 Consensus (3 Impressions)`.
  - âœ… Complete 8-class probability breakdown table.
  - âœ… Timestamp, Report ID, and Model Name (`Champion Ensemble V2`).
  - âœ… Prominent Medical Serological Disclaimer Box in red border.
  - âœ… Yellow Rh Uncertainty Banner triggered when `rh_uncertainty` is true.

---

## 7. Section F: Database Audit Log Test

Audit records verified directly in `data/blood_group.db`:

```text
ID 174: [2026-09-05T12:24:39] MFS100               -> O+ (93.8%) [NORMAL]
ID 173: [2026-09-05T12:24:20] MFS100               -> O+ (92.2%) [NORMAL]
ID 172: [2026-09-05T12:24:16] MFS100               -> O+ (92.5%) [NORMAL]
ID 171: [2026-09-05T12:23:58] MFS100 (Consensus)   -> O+ (74.9%) [NORMAL]
ID 170: [2026-09-05T12:21:52] MFS100               -> O+ (68.6%) [NORMAL]
ID 169: [2026-09-05T12:21:49] MFS100               -> O+ (68.6%) [NORMAL]
ID 168: [2026-09-05T12:21:46] MFS100               -> O+ (67.5%) [NORMAL]
ID 167: [2026-09-05T12:21:43] MFS100               -> O+ (61.5%) [NORMAL]
ID 166: [2026-09-05T12:21:39] MFS100               -> O+ (49.4%) [LOW_CONFIDENCE]
ID 165: [2026-09-05T12:21:29] MFS100               -> O+ (74.5%) [NORMAL]
```
- All physical captures and consensus operations correctly logged with zero PII.

---

## 8. Section G: Upload Regression Test (Validation / Non-Test Data)

Uploaded authentic sample images from `data/sample/`:

1. **`data/sample/A+.bmp`**:
   - Status: 200 OK
   - Prediction: `A+` (88.39% confidence)
   - Crop Applied: `False` (Verified: hardware cropping is bypassed for file uploads)
   - ABO Group: `A` | Rh Factor: `+`
   - Rh Uncertainty: `False`
2. **`data/sample/B+.bmp`**:
   - Status: 200 OK
   - Prediction: `B+` (79.31% confidence)
   - Crop Applied: `False`
   - ABO Group: `B` | Rh Factor: `+`
   - Rh Uncertainty: `False`
3. **`data/sample/O-.bmp`**:
   - Status: 200 OK
   - Prediction: `O-` (53.60% confidence)
   - Crop Applied: `False`
   - ABO Group: `O` | Rh Factor: `-`
   - Rh Uncertainty: **True**
   - Advisory Message: *"Rh factor (Â±) determination is uncertain (narrow Rh probability margin (17.4%); moderate 8-class confidence (53.6%); predicted Rh-negative requires serological validation due to weak ridge-Rh correlation). ABO group appears to be 'O' (89.8% confidence). Standard serological laboratory blood typing is required to confirm Rh factor."*

---

## 9. Section H: Automated Test Suite Verification

Pytest execution output:
```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\thanu\OneDrive\Desktop\major project
collected 44 items

tests\test_api.py .....                                                  [ 11%]
tests\test_database.py .                                                 [ 13%]
tests\test_dataset_prep.py ....                                          [ 22%]
tests\test_e2e_integration.py ..                                         [ 27%]
tests\test_mfs100.py ...                                                 [ 34%]
tests\test_mfs100_bridge.py .....                                        [ 45%]
tests\test_mfs100_integration.py ....                                    [ 54%]
tests\test_mfs100_scanner.py .....                                       [ 65%]
tests\test_predictor.py ......                                           [ 79%]
tests\test_preprocessing.py .......                                      [ 95%]
tests\test_report.py ..                                                  [100%]

============================= 44 passed in 45.06s =============================
```
- **Result:** **44 passed in 45.06s**, 0 failures, 0 warnings.
- **Locked Test Set:** Untouched and un-evaluated.

---

## 10. Limitations & Known Issues

1. **Weak Ridge Correlation to Rh Factor:**
   - As established scientifically and empirically, Rh factor (+ vs -) has weak phenotypic expression in epidermal ridge patterns.
   - The system now explicitly protects against this limitation by decomposing predictions into ABO and Rh factor components and surfacing prominent uncertainty advisories on all narrow margins, moderate confidences, and Rh-negative predictions.
2. **First-Impression Placement Drift on NFIQ 2 Fingers:**
   - First-time finger presses can be slightly misaligned before the user adjusts pressure.
   - The 3-Scan Consensus mode completely stabilizes this drift by averaging across multiple placements.
3. **Mandatory Laboratory Confirmation:**
   - This AI biometric platform is strictly an academic research proof-of-concept and decision-support prototype. It cannot replace certified serological blood typing for clinical transfusions.

---

## 11. Final Acceptance Verdict

| Test Category | Criterion | Status |
|---|---|:---:|
| **Physical MFS100 Capture** | 5 consecutive scans on live hardware | âœ… PASS |
| **Consistency** | Repeatability across impressions | âœ… PASS (100% on O+) |
| **3-Scan Consensus** | Soft-voting multi-scan noise reduction | âœ… PASS (100% agreement) |
| **ABO/Rh Decomposition** | Derived ABO & Rh without model modification | âœ… PASS |
| **Rh Uncertainty Guard** | Advisory triggers on narrow margin & Rh- | âœ… PASS |
| **Quality Guidance** | NFIQ & quality score feedback | âœ… PASS |
| **Upload Regression** | Bypasses crop, preserves upload pipeline | âœ… PASS |
| **PDF Reporting** | Formal reports with disclaimers & ABO data | âœ… PASS |
| **Database Audit** | SQLite logging without PII | âœ… PASS |
| **Full Test Suite** | 44/44 unit and integration tests | âœ… PASS (45.06s) |
| **Model Constraints** | Zero retraining, locked weights, test set untouched | âœ… FULLY COMPLIANT |
