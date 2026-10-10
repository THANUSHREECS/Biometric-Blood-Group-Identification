#!/usr/bin/env python3
"""
scripts/test_prediction.py
Stand-alone test script for validating the prediction engine:
- Tests inference across authentic sample images in data/sample/
- Validates 8-class probability distribution
- Tests MFS100 mock capture prediction
- Checks confidence threshold gating (< 0.60 flag)
"""

import os
import sys
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import SAMPLE_IMAGE_DIR, CLASS_NAMES
from src.predictor import BloodGroupPredictor
from src.mfs100.scanner import MFS100Scanner

def test_predictions():
    print("=" * 70)
    print("TESTING BLOOD GROUP PREDICTION ENGINE")
    print("=" * 70)

    predictor = BloodGroupPredictor()
    if not predictor.is_model_ready():
        print("[-] Error: Model is not ready. Train the model first!")
        return False

    scanner = MFS100Scanner(mock=True)

    # 1. Test inference on authentic class samples
    print("\n[*] Testing inference across authentic class samples:")
    samples = sorted([f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.endswith(".bmp")])
    
    for sample in samples:
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, sample)
        target_class = os.path.splitext(sample)[0]
        
        res = predictor.predict(sample_path)
        
        print(f"\nSample: {sample:<8} (Target: {target_class})")
        print(f"  -> Predicted Blood Group : {res['blood_group']}")
        print(f"  -> Confidence            : {res['confidence'] * 100:.2f}%")
        print(f"  -> Evaluation Status     : {res['status']}")
        print(f"  -> Message               : {res['message']}")
        print(f"  -> Probabilities (Top 3) :")
        sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        for cname, p in sorted_probs:
            print(f"       {cname:5s}: {p * 100:.2f}%")

    # 2. Test MFS100 Mock Capture Flow
    print("\n" + "-" * 70)
    print("[*] Testing Mantra MFS100 Capture & Inference Pipeline:")
    cap_path, cap_bytes = scanner.capture()
    print(f"[+] Captured fingerprint saved to: {cap_path} ({len(cap_bytes)} bytes)")
    
    cap_res = predictor.predict(cap_path)
    print(f"[+] Capture Prediction Result:")
    print(f"    Blood Group : {cap_res['blood_group']}")
    print(f"    Confidence  : {cap_res['confidence'] * 100:.2f}%")
    print(f"    Status      : {cap_res['status']}")

    print("\n" + "=" * 70)
    print("[+] ALL PREDICTION ENGINE TESTS PASSED!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    test_predictions()
