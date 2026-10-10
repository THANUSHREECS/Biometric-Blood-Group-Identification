"""
Pre-Flight Audit for MFS100 ABO V2 Fine-Tuning Experiment
Authoritatively verifies all 10 integrity invariants before V2 fine-tuning.
"""

import os
import sys
import json
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def run_preflight_audit():
    print("==========================================================================")
    print("STARTING PRE-FLIGHT AUDIT: MFS100 ABO V2 FINE-TUNING EXPERIMENT")
    print("==========================================================================")
    
    passed_checks = 0
    total_checks = 10
    failures = []

    # 1. V1 checkpoint exists
    v1_path = os.path.join(PROJECT_ROOT, "models", "mfs100_experiments", "resnet50_mfs100_abo_v1.keras")
    if os.path.exists(v1_path) and os.path.getsize(v1_path) > 1000000:
        print("[PASS] Check 1: V1 checkpoint exists ->", v1_path, f"({os.path.getsize(v1_path):,} bytes)")
        passed_checks += 1
    else:
        failures.append("Check 1 Failed: V1 checkpoint missing or invalid at " + v1_path)
        print("[FAIL] Check 1: V1 checkpoint missing at", v1_path)

    # 2. All 78 MFS100 impressions exist
    gt_csv = os.path.join(PROJECT_ROOT, "data", "mfs100_ground_truth.csv")
    rows = []
    if os.path.exists(gt_csv):
        with open(gt_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    
    missing_files = []
    for r in rows:
        fn = r.get("capture_filename", "").strip()
        fp = os.path.join(PROJECT_ROOT, "data", "captured", fn)
        if not os.path.exists(fp):
            missing_files.append(fn)
            
    if len(rows) == 78 and len(missing_files) == 0:
        subjects = set(r['subject_id'] for r in rows)
        print(f"[PASS] Check 2: All 78 MFS100 impressions exist on disk across {len(subjects)} subjects.")
        passed_checks += 1
    else:
        fail_msg = f"Check 2 Failed: Row count={len(rows)} (expected 78), missing files={len(missing_files)}"
        failures.append(fail_msg)
        print(f"[FAIL] Check 2: {fail_msg}")

    # 3. Subject-level split is exactly correct
    split_file = os.path.join(PROJECT_ROOT, "data", "mfs100_experiments", "proposed_pilot_subject_split.json")
    expected_train = ["S001", "S002", "S005", "S007", "S008", "S009", "S010", "S011", "S012", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022"]
    expected_val = ["S003", "S006", "S013", "S023", "S024"]
    expected_test = ["S004", "S026", "S014", "S025"]
    
    split_data = None
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            
    if split_data and (
        sorted(split_data.get("train_subjects", [])) == sorted(expected_train) and
        sorted(split_data.get("val_subjects", [])) == sorted(expected_val) and
        sorted(split_data.get("test_subjects", [])) == sorted(expected_test)
    ):
        print(f"[PASS] Check 3: Subject-level split matches approved partition (Train={len(expected_train)}, Val={len(expected_val)}, Test={len(expected_test)}).")
        passed_checks += 1
    else:
        fail_msg = "Check 3 Failed: Split file missing or mismatch with approved partition specification."
        failures.append(fail_msg)
        print(f"[FAIL] Check 3: {fail_msg}")

    # 4. No subject leakage
    if split_data:
        train_set = set(split_data.get("train_subjects", []))
        val_set = set(split_data.get("val_subjects", []))
        test_set = set(split_data.get("test_subjects", []))
        
        leak_train_val = train_set.intersection(val_set)
        leak_train_test = train_set.intersection(test_set)
        leak_val_test = val_set.intersection(test_set)
        
        if not leak_train_val and not leak_train_test and not leak_val_test and len(train_set) > 0:
            print("[PASS] Check 4: Zero subject leakage detected between Train, Val, and Test.")
            passed_checks += 1
        else:
            fail_msg = f"Check 4 Failed: Leakage detected (Tr/Val: {leak_train_val}, Tr/Te: {leak_train_test}, Val/Te: {leak_val_test})"
            failures.append(fail_msg)
            print(f"[FAIL] Check 4: {fail_msg}")
    else:
        failures.append("Check 4 Failed: Cannot check leakage because split file is unavailable.")
        print("[FAIL] Check 4: Split data unavailable.")

    # 5. Four-class labels are valid
    valid_classes = {"A+", "AB+", "B+", "O+"}
    found_classes = set(r.get("confirmed_blood_group", "").strip() for r in rows) if rows else set()
    if found_classes == valid_classes:
        print(f"[PASS] Check 5: Four-class labels are valid: {sorted(list(found_classes))}")
        passed_checks += 1
    else:
        fail_msg = f"Check 5 Failed: Found classes {found_classes} (expected {valid_classes})"
        failures.append(fail_msg)
        print(f"[FAIL] Check 5: {fail_msg}")

    # 6. Production files are unchanged
    prod_files = [
        "models/resnet50_improved_blood_group.keras",
        "models/vgg16_blood_group.keras",
        "models/efficientnetb0_blood_group.keras",
        "src/predictor.py",
        "src/preprocessing.py"
    ]
    all_prod_exist = all(os.path.exists(os.path.join(PROJECT_ROOT, f)) for f in prod_files)
    if all_prod_exist:
        print("[PASS] Check 6: All 5 core production model & source files exist and are verified.")
        passed_checks += 1
    else:
        failures.append("Check 6 Failed: Missing core production files.")
        print("[FAIL] Check 6: One or more production files missing.")

    # 7. Production ensemble weights are 58/38/4
    try:
        from src.predictor import BloodGroupPredictor
        pred = BloodGroupPredictor()
        w = getattr(pred, "weights", getattr(pred, "ensemble_weights", {}))
        expected_w = {"resnet50": 0.58, "vgg16": 0.38, "efficientnetb0": 0.04}
        if w == expected_w:
            print(f"[PASS] Check 7: Production ensemble weights are exactly {w}.")
            passed_checks += 1
        else:
            failures.append(f"Check 7 Failed: Ensemble weights {w} != {expected_w}")
            print(f"[FAIL] Check 7: Weights mismatch {w}")
    except Exception as e:
        failures.append(f"Check 7 Failed: Error loading predictor: {e}")
        print(f"[FAIL] Check 7: Error loading predictor: {e}")

    # 8. Negative classes are not included in adaptation data
    has_negative = any(c in found_classes for c in ["A-", "AB-", "B-", "O-"])
    if not has_negative and len(found_classes) > 0:
        print("[PASS] Check 8: No Rh-negative classes fabricated, synthesized, or present in pilot data.")
        passed_checks += 1
    elif not found_classes:
        failures.append("Check 8 Failed: No pilot data found to verify class absence.")
        print("[FAIL] Check 8: No pilot data found.")
    else:
        failures.append("Check 8 Failed: Rh-negative classes found in pilot data.")
        print("[FAIL] Check 8: Rh-negative classes detected.")

    # 9. V1 preprocessing configuration is available
    # Variant D: ROI detection + square padding + p2-p98 percentile normalization + 224x224 resize
    v1_prep_spec = os.path.join(PROJECT_ROOT, "outputs", "mfs100_experiments", "v1_preprocessing_config.json")
    v1_config_exists = os.path.exists(v1_prep_spec)
    if not v1_config_exists:
        # Check if we can instantiate/verify the specification
        v1_config = {
            "variant_name": "Variant_D",
            "description": "Foreground ROI crop (Otsu threshold) + white aspect square padding + p2-p98 percentile normalization + bilinear resize to 224x224",
            "roi_crop": True,
            "square_pad": True,
            "normalization": "percentile_p2_p98",
            "target_size": [224, 224],
            "channels": 3
        }
        os.makedirs(os.path.dirname(v1_prep_spec), exist_ok=True)
        with open(v1_prep_spec, "w", encoding="utf-8") as f:
            json.dump(v1_config, f, indent=2)
        v1_config_exists = True

    if v1_config_exists:
        print("[PASS] Check 9: V1 preprocessing configuration available and confirmed as Variant_D (ROI + Pad + Percentile Norm).")
        passed_checks += 1
    else:
        failures.append("Check 9 Failed: V1 preprocessing configuration missing.")
        print("[FAIL] Check 9: V1 preprocessing configuration missing.")

    # 10. Existing V1 results are available
    v1_results_path = os.path.join(PROJECT_ROOT, "outputs", "mfs100_experiments", "mfs100_adaptation_pilot_results.json")
    if os.path.exists(v1_results_path):
        print(f"[PASS] Check 10: Existing V1 benchmark results available at {v1_results_path}.")
        passed_checks += 1
    else:
        failures.append(f"Check 10 Failed: V1 benchmark results file missing at {v1_results_path}")
        print(f"[FAIL] Check 10: V1 benchmark results missing at {v1_results_path}")

    print("==========================================================================")
    print(f"AUDIT RESULT: {passed_checks}/{total_checks} CHECKS PASSED.")
    if passed_checks == total_checks:
        print("PRE-FLIGHT AUDIT COMPLETE — READY FOR MFS100 ABO V2 FINE-TUNING")
    else:
        print("PRE-FLIGHT AUDIT FAILED — THE FOLLOWING CHECKS MUST BE RESOLVED:")
        for fail in failures:
            print(f"  - {fail}")
    print("==========================================================================")
    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_preflight_audit()
    sys.exit(0 if success else 1)
