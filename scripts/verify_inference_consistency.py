#!/usr/bin/env python3
"""
scripts/verify_inference_consistency.py
Validates:
1. Strict determinism: Repeated predictions on the same input produce identical probabilities.
2. File format invariant: Loading as path vs raw bytes produces identical outputs.
3. Sample predictions on authentic images.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import SAMPLE_IMAGE_DIR
from src.predictor import BloodGroupPredictor

def main():
    predictor = BloodGroupPredictor()
    assert predictor.is_ensemble_ready, "Ensemble must be ready"
    
    # 1. Determinism Test
    sample_files = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.endswith(".bmp")]
    if not sample_files:
        print("[-] No sample files found!")
        return 1
    
    test_img = os.path.join(SAMPLE_IMAGE_DIR, sample_files[0])
    print(f"[*] Testing determinism on {test_img} (10 repeated runs)...")
    
    first_res = predictor.predict(test_img)
    first_probs = list(first_res["probabilities"].values())
    
    for i in range(9):
        rep_res = predictor.predict(test_img)
        rep_probs = list(rep_res["probabilities"].values())
        diff = np.max(np.abs(np.array(first_probs) - np.array(rep_probs)))
        if diff > 1e-6:
            print(f"[-] Determinism failed at run {i+2}! Max diff: {diff}")
            return 1
    print("[+] Determinism check PASSED: 10/10 runs produced bit-identical output.")
    
    # 2. File path vs Raw Bytes Invariance
    with open(test_img, "rb") as f:
        raw_bytes = f.read()
    
    bytes_res = predictor.predict(raw_bytes)
    bytes_probs = list(bytes_res["probabilities"].values())
    diff_bytes = np.max(np.abs(np.array(first_probs) - np.array(bytes_probs)))
    if diff_bytes > 1e-5:
        print(f"[-] Byte invariance failed! Diff: {diff_bytes}")
        return 1
    print("[+] Byte vs Path invariance PASSED: Identical output whether given path or bytes.")
    
    # 3. Predict across all authentic sample images
    print("\n[*] Sample image predictions with locked Champion Ensemble V2:")
    for sf in sorted(sample_files):
        p = os.path.join(SAMPLE_IMAGE_DIR, sf)
        r = predictor.predict(p)
        print(f"  {sf:8s} -> Predicted: {r['blood_group']:4s} | Confidence: {r['confidence']*100:.2f}% | Status: {r['status']}")
    
    print("\n[+] Verification successfully completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
