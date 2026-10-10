import os
import sys
import numpy as np

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor import BloodGroupPredictor
import src.config as config

def run_tests():
    print("--- STARTING INTEGRATION TESTS ---")
    
    p = BloodGroupPredictor()
    
    # Check 1: Model Loading & Config
    print("\n[Test 1] Model Loading & Config")
    assert p.is_model_ready(), "Model failed to load"
    assert config.CLASS_NAMES == ["A+", "AB+", "B+", "O+"], f"Invalid classes: {config.CLASS_NAMES}"
    assert len(p.checkpoint_paths) == 5, f"Expected 5 checkpoints, got {len(p.checkpoint_paths)}"
    print("PASS")

    # Check 2: 3-Scan Consensus Inference
    print("\n[Test 2] 3-Scan Consensus Inference")
    dummy_img1 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    dummy_img2 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    dummy_img3 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    
    res = p.predict_consensus([dummy_img1, dummy_img2, dummy_img3])
    probs = res["probabilities"]
    print(f"Probabilities: {probs}")
    
    sum_probs = sum(probs.values())
    print(f"Sum of probabilities: {sum_probs}")
    assert abs(sum_probs - 1.0) < 1e-3, f"Probabilities do not sum to 1: {sum_probs}"
    assert len(probs) == 4, "Expected 4 class probabilities"
    print("PASS")
    
    # Check 3: Single-Scan Mode Validation
    print("\n[Test 3] Single-Scan Experimental Labeling")
    res_single = p.predict(dummy_img1)
    print(f"Message: {res_single['message']}")
    assert res_single.get("is_replicated_scan") is True, "is_replicated_scan flag missing"
    assert "EXPERIMENTAL" in res_single["message"], "Message does not state experimental status"
    print("PASS")
    
    print("\n--- ALL TESTS COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    run_tests()
