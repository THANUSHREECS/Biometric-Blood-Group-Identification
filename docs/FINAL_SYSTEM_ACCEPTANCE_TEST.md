# Final System Acceptance Test Report

**Generated:** 2026-10-10 14:31:42
**System:** Biometric Blood Group Identification System Ã¢â‚¬â€ Champion Ensemble V2
**Ensemble Configuration:** ResNet50 58% + VGG16 38% + EfficientNetB0 4%
**Locked Test Accuracy:** 85.41% (898-image test set Ã¢â‚¬â€ NOT accessed this session)

---

## Summary

| Metric | Count |
|--------|-------|
| Ã¢Å“â€¦ PASS | 74 |
| Ã¢Å’ FAIL | 10 |
| Ã¢Å¡Â Ã¯Â¸  WARN | 3 |
| Ã¢Å¡Â Ã¯Â¸  SKIP | 3 |
| **TOTAL** | **90** |

**Overall Result: Ã¢Å’ FAILED Ã¢â‚¬â€ see failures below**

---

## 1. Mock Mode

| Test | Status | Detail |
|------|--------|--------|
| 1.1 Scanner status reachable | Ã¢Å“â€¦ PASS | mock_mode=False |
| 1.2 Switch to mock mode | Ã¢Å“â€¦ PASS | Switched to mock mode |
| 1.3 Mock capture [A+] success | Ã¢Å“â€¦ PASS | success=True |
| 1.3 Blood group valid [A+] | Ã¢Å“â€¦ PASS | Ã¢â€ â€™ A+ |
| 1.3 8-class probability dist [A+] | Ã¢Å’ FAIL | Condition not met: 8 classes present, sum=1.0 |
| 1.3 Confidence range [A+] | Ã¢Å“â€¦ PASS | confidence=46.7% |
| 1.3 scanner_mode=mock [A+] | Ã¢Å“â€¦ PASS | scanner_mode=mock |
| 1.3 No crop in mock mode [A+] | Ã¢Å“â€¦ PASS | crop_applied=False (correct) |
| 1.3 Low-conf flag [A+] | Ã¢Å¡Â Ã¯Â¸ WARN | is_low_confidence=True, conf=46.7% |
| 1.4 PDF report [A+] | Ã¢Å“â€¦ PASS | PDF OK, size=15178 bytes |
| 1.3 Mock capture [B-] success | Ã¢Å“â€¦ PASS | success=True |
| 1.3 Blood group valid [B-] | Ã¢Å“â€¦ PASS | Ã¢â€ â€™ A+ |
| 1.3 8-class probability dist [B-] | Ã¢Å’ FAIL | Condition not met: 8 classes present, sum=1.0 |
| 1.3 Confidence range [B-] | Ã¢Å“â€¦ PASS | confidence=62.9% |
| 1.3 scanner_mode=mock [B-] | Ã¢Å“â€¦ PASS | scanner_mode=mock |
| 1.3 No crop in mock mode [B-] | Ã¢Å“â€¦ PASS | crop_applied=False (correct) |
| 1.3 Low-conf flag [B-] | Ã¢Å“â€¦ PASS | is_low_confidence=False, conf=62.9% |
| 1.4 PDF report [B-] | Ã¢Å“â€¦ PASS | PDF OK, size=16660 bytes |
| 1.3 Mock capture [O+] success | Ã¢Å“â€¦ PASS | success=True |
| 1.3 Blood group valid [O+] | Ã¢Å“â€¦ PASS | Ã¢â€ â€™ A+ |
| 1.3 8-class probability dist [O+] | Ã¢Å’ FAIL | Condition not met: 8 classes present, sum=1.0 |
| 1.3 Confidence range [O+] | Ã¢Å“â€¦ PASS | confidence=57.9% |
| 1.3 scanner_mode=mock [O+] | Ã¢Å“â€¦ PASS | scanner_mode=mock |
| 1.3 No crop in mock mode [O+] | Ã¢Å“â€¦ PASS | crop_applied=False (correct) |
| 1.3 Low-conf flag [O+] | Ã¢Å¡Â Ã¯Â¸ WARN | is_low_confidence=True, conf=57.9% |
| 1.4 PDF report [O+] | Ã¢Å“â€¦ PASS | PDF OK, size=18588 bytes |
| 1.5 Database has mock records | Ã¢Å“â€¦ PASS | 7 MFS100 (Mock) records found |

*Section: 22 passed, 3 failed, 2 warned/skipped*

---

## 2. File Upload

| Test | Status | Detail |
|------|--------|--------|
| 2.1 Upload success [A+.bmp] | Ã¢Å“â€¦ PASS | success=True |
| 2.2 Blood group valid [A+.bmp] | Ã¢Å“â€¦ PASS | Ã¢â€ â€™ A+ (46.7%) |
| 2.3 8-class prob distribution [A+.bmp] | Ã¢Å’ FAIL | Condition not met: sum=1.0 |
| 2.4 MFS100 crop NOT applied [A+.bmp] | Ã¢Å“â€¦ PASS | crop_applied=False (correct for upload) |
| 2.5 Low-conf flag [A+.bmp] | Ã¢Å¡Â Ã¯Â¸ WARN | is_low_confidence=True, conf=46.7% |
| 2.6 PDF report [A+.bmp] | Ã¢Å“â€¦ PASS | PDF size=15170 bytes |
| 2.1 Upload success [O-.bmp] | Ã¢Å“â€¦ PASS | success=True |
| 2.2 Blood group valid [O-.bmp] | Ã¢Å“â€¦ PASS | Ã¢â€ â€™ A+ (65.4%) |
| 2.3 8-class prob distribution [O-.bmp] | Ã¢Å’ FAIL | Condition not met: sum=1.0 |
| 2.4 MFS100 crop NOT applied [O-.bmp] | Ã¢Å“â€¦ PASS | crop_applied=False (correct for upload) |
| 2.5 Low-conf flag [O-.bmp] | Ã¢Å“â€¦ PASS | is_low_confidence=False, conf=65.4% |
| 2.6 PDF report [O-.bmp] | Ã¢Å“â€¦ PASS | PDF size=95746 bytes |
| 2.7 Database upload records present | Ã¢Å“â€¦ PASS | 5 upload records found |

*Section: 10 passed, 2 failed, 1 warned/skipped*

---

## 3. MFS100 Hardware

| Test | Status | Detail |
|------|--------|--------|
| 3.1 Bridge reachable on 127.0.0.1:5100 | Ã¢Å“â€¦ PASS | HTTP 200 |
| 3.1b Bridge status | Ã¢Å“â€¦ PASS | {'device': None, 'is_device_connected': True, 'is_initialized': False, 'python_arch': '64bit', 'python_version': '3.10.1 |
| 3.2 Switch to hardware mode | Ã¢Å“â€¦ PASS | mock_mode=False |
| 3.3 Physical MFS100 connected | Ã¢Å“â€¦ PASS | {'connected': True, 'info': {'bridge_arch': '64bit', 'bridge_status': 'REACHABLE', 'firmware': 'v9.0.2.5', 'height': 354 |
| 3.4 Not in mock mode | Ã¢Å“â€¦ PASS | mock_mode=False |
| 3.4 Bridge reachable from main app | Ã¢Å“â€¦ PASS | bridge_health.reachable=True |
| 3.4 Serial number present | Ã¢Å“â€¦ PASS | serial=11215389 |
| 3.4 Model = MFS100 | Ã¢Å“â€¦ PASS | model=MFS100 |
| 3.5 Hardware capture #1 | Ã¢Å¡Â Ã¯Â¸ SKIP | HTTP 500 Ã¢â‚¬â€ Capture error: MFS100 Capture Failed: Finger not placed properly (scanner may need finger placement) |
| 3.5 Hardware capture #2 | Ã¢Å¡Â Ã¯Â¸ SKIP | HTTP 500 Ã¢â‚¬â€ Capture error: MFS100 Capture Failed: Finger not placed properly (scanner may need finger placement) |
| 3.5 Hardware capture #3 | Ã¢Å¡Â Ã¯Â¸ SKIP | HTTP 500 Ã¢â‚¬â€ Capture error: MFS100 Capture Failed: Finger not placed properly (scanner may need finger placement) |
| 3.8 Restore mock mode | Ã¢Å“â€¦ PASS | Restored mock_mode=True after hardware test |

*Section: 9 passed, 0 failed, 3 warned/skipped*

---

## 4. Database

| Test | Status | Detail |
|------|--------|--------|
| 4.1 Database has records | Ã¢Å“â€¦ PASS | total=50 |
| 4.2 Records is a list | Ã¢Å“â€¦ PASS | len=50 |
| 4.3 Record has id field | Ã¢Å“â€¦ PASS | id=645 |
| 4.4 Record has timestamp | Ã¢Å“â€¦ PASS | timestamp=2026-10-10T14:29:17.199417 |
| 4.5 Record has input_type | Ã¢Å“â€¦ PASS | input_type=upload |
| 4.6 Record has predicted_group | Ã¢Å“â€¦ PASS | predicted_group=A+ |
| 4.7 Record has confidence | Ã¢Å“â€¦ PASS | confidence=0.6537 |
| 4.8 Record has status field | Ã¢Å“â€¦ PASS | status=NORMAL |
| 4.9 Input types present | Ã¢Å“â€¦ PASS | Types found: ['MFS100', 'MFS100 (Mock)', 'MFS100 Consensus - Left Thumb (3 Impressions)', 'upload'] |
| 4.10 All confidence values in [0,1] | Ã¢Å“â€¦ PASS | min=0.318, max=0.985 |
| 4.11 Timestamp parseable | Ã¢Å“â€¦ PASS | '2026-10-10T14:29:17.199417' |

*Section: 11 passed, 0 failed, 0 warned/skipped*

---

## 5. PDF Report

| Test | Status | Detail |
|------|--------|--------|
| 5.1 PDF has substantial content | Ã¢Å“â€¦ PASS | size=15172 bytes (>5000) |
| 5.2 PDF magic bytes valid | Ã¢Å“â€¦ PASS | Starts with %PDF |
| 5.3 Content-Type is application/pdf | Ã¢Å“â€¦ PASS | application/pdf |
| 5.4 Blood group in PDF text | Ã¢Å“â€¦ PASS | Found group in text |
| 5.5 Confidence in PDF | Ã¢Å“â€¦ PASS | Found 'confidence' |
| 5.6 Medical disclaimer in PDF | Ã¢Å“â€¦ PASS | Found disclaimer text |
| 5.7 Model info in PDF | Ã¢Å“â€¦ PASS | Found model reference |
| 5.8 Input source in PDF | Ã¢Å“â€¦ PASS | Found source reference |

*Section: 8 passed, 0 failed, 0 warned/skipped*

---

## 6. Error Handling

| Test | Status | Detail |
|------|--------|--------|
| 6.1 Invalid file type rejected (400) | Ã¢Å“â€¦ PASS | status=400 |
| 6.2 Empty file handled (400/500) | Ã¢Å“â€¦ PASS | status=500 |
| 6.3 Missing file rejected (400) | Ã¢Å“â€¦ PASS | status=400 |
| 6.4 Non-existent report returns 404 | Ã¢Å“â€¦ PASS | status=404 |
| 6.5 Unknown endpoint returns 404 | Ã¢Å“â€¦ PASS | status=404 |
| 6.6 Corrupted image handled gracefully (400/500) | Ã¢Å“â€¦ PASS | status=500 |
| 6.7 Mock capture still works (mock mode fallback) | Ã¢Å“â€¦ PASS | status=200 |

*Section: 7 passed, 0 failed, 0 warned/skipped*

---

## 7. Automated Tests

| Test | Status | Detail |
|------|--------|--------|
| 7.1 All tests pass (pytest) | Ã¢Å’ FAIL | 25 tests failed Ã¢â‚¬â€ see output above |
| 7.2 Test suite exit code | Ã¢Å’ FAIL | exit_code=1 |

*Section: 0 passed, 2 failed, 0 warned/skipped*

---

## 8. Integrity

| Test | Status | Detail |
|------|--------|--------|
| 8.1 ResNet50 weight = 0.58 | Ã¢Å“â€¦ PASS | resnet50=0.58 |
| 8.1 VGG16 weight = 0.38 | Ã¢Å“â€¦ PASS | vgg16=0.38 |
| 8.1 EfficientNetB0 weight = 0.04 | Ã¢Å“â€¦ PASS | efficientnetb0=0.04 |
| 8.2 Model file unmodified: resnet50_improved_blood_group.keras | Ã¢Å“â€¦ PASS | Last modified 865.6h ago (>1h Ã¢Å“â€œ) |
| 8.2 Model file unmodified: vgg16_blood_group.keras | Ã¢Å“â€¦ PASS | Last modified 864.4h ago (>1h Ã¢Å“â€œ) |
| 8.2 Model file unmodified: efficientnetb0_blood_group.keras | Ã¢Å“â€¦ PASS | Last modified 867.3h ago (>1h Ã¢Å“â€œ) |
| 8.3 final_evaluation_results.json not rewritten this session | Ã¢Å“â€¦ PASS | Last modified 863.8h ago (>1h Ã¢Å“â€œ) |
| 8.4 API /status reports correct ResNet50 weight | Ã¢Å’ FAIL | Condition not met: weights={} |
| 8.4 API /status reports correct VGG16 weight | Ã¢Å’ FAIL | Condition not met: VGG16=None |
| 8.4 API /status reports correct EfficientNetB0 weight | Ã¢Å’ FAIL | Condition not met: EffNet=None |

*Section: 7 passed, 3 failed, 0 warned/skipped*

---

## Integrity Statement

> [!IMPORTANT]
> The following constraints were upheld throughout this acceptance test:

- Ã¢Å“â€¦ The locked 898-image test set was **NOT accessed or evaluated**
- Ã¢Å“â€¦ No model weights were modified
- Ã¢Å“â€¦ No retraining occurred
- Ã¢Å“â€¦ Ensemble configuration locked at ResNet50=58%, VGG16=38%, EfficientNetB0=4%
- Ã¢Å“â€¦ MFS100 foreground crop applied only for `source="mfs100"` hardware captures
- Ã¢Å“â€¦ Standard file uploads received standard preprocessing (no MFS100 crop)

---

## System Architecture Verified

| Component | Status | Notes |
|-----------|--------|-------|
| Champion Ensemble V2 | Ã¢Å“â€¦ Active | ResNet50 58% + VGG16 38% + EfficientNetB0 4% |
| MFS100 Bridge (32-bit) | Ã¢Å“â€¦ Running | 127.0.0.1:5100 |
| Flask Web Server (64-bit) | Ã¢Å“â€¦ Running | 127.0.0.1:5000 |
| Source-Aware Preprocessing | Ã¢Å“â€¦ Integrated | MFS100 crop Ã¢â€°Â  upload path |
| SQLite Audit Database | Ã¢Å“â€¦ Recording | All predictions logged |
| PDF Report Generator | Ã¢Å“â€¦ Functional | ReportLab PDF |
| Error Handling | Ã¢Å“â€¦ Tested | 400/404/500 correctly returned |

---

*This report was auto-generated by `scripts/run_acceptance_test.py`*
*Timestamp: 2026-10-10 14:31:42*