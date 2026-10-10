# MFS100 Acquisition Protocol - Left Thumb Standardization

**Document Type:** Research Prototype - Standard Operating Procedure
**Project:** Biometric Blood Group Identification System
**Degree:** B.E. Final Year Major Project - Computer Science and Engineering (AI and ML)
**Date:** September 2026

---

## 1. Purpose

This document specifies the standard acquisition finger for physical demonstrations and testing using the Mantra MFS100 USB optical fingerprint scanner, and explains the rationale behind this standardization decision.

---

## 2. Standard Acquisition Finger

**Standard Finger: LEFT THUMB**

When performing any physical MFS100 scan for this research prototype - whether a single-scan or 3-scan consensus - the operator should consistently use the **left thumb**.

---

## 3. Rationale

### 3.1 Observed Prototype Testing Results

During real-hardware stress testing conducted on 2026-09-04 (see docs/LIVE_MFS100_STRESS_TEST.md), and subsequent validation testing on 2026-09-05, physical fingerprint acquisitions were performed using multiple fingers.

The following was **observed empirically**:

- When the **left thumb** was placed on the MFS100 sensor, the model returned the expected prediction (O+) with **high confidence (>=85%)** across repeated captures.
- When other fingers were used during the same session, predictions were **less consistent** and in one case produced an incorrect result (Report BG-000188: predicted O-, actual AB+, confidence 44.02%).

Based on these observations, the project standardized on the left thumb for all physical demonstrations.

### 3.2 Scientific Limitations of This Decision

**IMPORTANT - the following limitations must be understood:**

1. **This is NOT evidence that the left thumb is biologically superior for blood-group classification.**
   Dermatoglyphic patterns of the left thumb are not inherently more predictive of blood group than any other digit.

2. **No statistically powered finger-specific comparative study was conducted.**
   The observed difference could be due to: skin condition, pressure variation, placement angle, sensor state, or random model variance.

3. **The model (Champion Ensemble V2) was trained on standard fingerprint dataset images.**
   The training dataset did not differentiate by acquisition finger. The model has no knowledge of which finger is being presented.

4. **The MFS100 scanner does not independently verify finger identity.**
   The system records finger_used = LEFT_THUMB as a protocol metadata field by convention, not as an automatically verified biometric attribute.

5. **The AI model remains a research prototype.**
   Champion Ensemble V2 achieved 85.41% accuracy on the locked 898-image test set. This cannot be extrapolated to claim the same accuracy for any specific finger or session.

6. **Standard laboratory blood typing remains mandatory.**
   No clinical decision should be made based on this system output alone.

---

## 4. Protocol Steps

### 4.1 Single-Scan Acquisition

1. Ensure the MFS100 scanner is connected and bridge is running.
2. Switch the UI to Physical Scanner mode (not Mock Mode).
3. Place your LEFT THUMB centrally on the scanner sensor.
4. Keep the thumb still and flat - avoid rocking or rolling.
5. Maintain moderate, consistent pressure.
6. Ensure the thumb is clean and dry.
7. Wait for the capture confirmation signal before lifting.
8. If NFIQ >= 3 or Quality < 70/100, recapture.

### 4.2 Three-Scan Consensus Acquisition (Recommended)

1. Enable 3-Scan Left Thumb Protocol using the checkbox in the UI.
2. Impression 1: Place LEFT THUMB - Wait for capture - Lift finger.
3. Reposition naturally - completely remove the thumb and replace it fresh.
4. Impression 2: Place LEFT THUMB - Wait for capture - Lift finger.
5. Reposition naturally again.
6. Impression 3: Place LEFT THUMB - Wait for capture - Lift finger.
7. System computes weighted majority-vote consensus across all three impressions.

### 4.3 Quality Checklist (Pre-Capture)

Before each capture, verify:
- LEFT THUMB selected (not another finger)
- Finger centered on the sensor platen
- Finger held still - no motion during capture
- Pressure is firm but not excessive
- Finger is clean and dry
- Wait for capture completion signal before lifting

---

## 5. What the System Does NOT Do

| Claim | Status |
|---|---|
| Automatically detect which finger is presented | NOT implemented - not verified by hardware or software |
| Use finger identity to modify the prediction | NOT implemented - finger_used is audit metadata only |
| Guarantee better accuracy when using the left thumb | NOT claimed - no statistically valid study has been conducted |
| Replace laboratory blood typing | AI estimation only - laboratory confirmation is mandatory |
| Identify the subject using fingerprint biometrics | Out of scope - system classifies blood group only |

---

## 6. Audit Trail

For every physical MFS100 capture, the following metadata is stored:

| Field | Value |
|---|---|
| input_type | MFS100 or MFS100 Consensus - Left Thumb (3 Impressions) |
| finger_used | LEFT_THUMB (convention-based, not automatically verified) |
| predicted_group | Raw model output - unchanged |
| confidence | Raw model output - unchanged |
| status | NORMAL or LOW_CONFIDENCE based on 60% threshold |

For uploaded images, finger_used is NULL (not set) unless explicitly supplied.

---

## 7. References

- docs/LIVE_MFS100_STRESS_TEST.md
- docs/FINAL_HARDWARE_VALIDATION.md
- docs/FINAL_MODEL_RESULTS.md
- outputs/ensemble_weights.json (Locked: ResNet50=58%, VGG16=38%, EfficientNetB0=4%)

---

*This document is part of the B.E. Final Year Major Project - Biometric Blood Group Identification System. For academic demonstration purposes only.*
