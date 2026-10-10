import os
import sys
import time
import base64
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.mfs100.scanner import MFS100Scanner
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

def transform_c5(image_input, target_size=(112, 112)):
    pil_img = load_image_to_pil(image_input)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=target_size)
    return np.array(resized, dtype=np.float32)

def main():
    print("Testing Real Capture via MFS100 Bridge...")
    
    scanner = MFS100Scanner(mock=False, bridge_url="http://127.0.0.1:5100")
    
    health = scanner.check_bridge_health()
    if not health.get("reachable"):
        print("FAIL: Bridge unreachable.")
        sys.exit(1)
        
    print("Bridge is Reachable.")
    
    if not scanner.is_connected():
        if not scanner.connect():
            print("FAIL: Could not connect to scanner through bridge.")
            sys.exit(1)
            
    print("Scanner Connected.")
    
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "live_mfs100_tests")
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = os.path.join(reports_dir, f"test_capture_{timestamp}.bmp")
    
    print("Please place your LEFT THUMB on the scanner...")
    
    try:
        # capture hardware image directly
        saved_path, img_bytes = scanner._capture_hardware(save_path)
        print(f"Capture successful. Saved to {saved_path}")
    except Exception as e:
        print(f"FAIL: Capture failed: {e}")
        sys.exit(1)
        
    print("Loading Day 27 Model...")
    model_path = os.path.join(PROJECT_ROOT, "models", "day27_full_dataset", "four_class", "best_model.keras")
    model = tf.keras.models.load_model(model_path)
    
    print("Running C5 Preprocessing...")
    input_array = transform_c5(saved_path, target_size=(112, 112))
    input_tensor = np.expand_dims(input_array, axis=0)
    
    print("Running Inference...")
    preds = model.predict(input_tensor, verbose=0)[0]
    
    FOUR_CLASSES = ["A+", "AB+", "B+", "O+"]
    CLASS_MAPPING = {i: c for i, c in enumerate(FOUR_CLASSES)}
    
    pred_idx = np.argmax(preds)
    pred_class = CLASS_MAPPING[pred_idx]
    confidence = float(preds[pred_idx])
    
    print(f"Prediction: {pred_class} (Confidence: {confidence:.2f})")
    print("PASS: Real capture pipeline works.")

if __name__ == "__main__":
    main()
