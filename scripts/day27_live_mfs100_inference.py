import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target
from src.mfs100.scanner import MFS100Scanner

def transform_c5(image_input, target_size=(112, 112)):
    # C5 Preprocessing from day27_experiments.py
    pil_img = load_image_to_pil(image_input)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=target_size)
    return np.array(resized, dtype=np.float32)

FOUR_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_MAPPING = {i: c for i, c in enumerate(FOUR_CLASSES)}
MODEL_PATH = "models/day27_full_dataset/four_class/best_model.keras"
OUTPUT_DIR = "reports/live_mfs100_tests"

def display_prediction(preds, img_path, preprocessed_shape):
    pred_idx = np.argmax(preds)
    pred_class = CLASS_MAPPING[pred_idx]
    confidence = preds[pred_idx] * 100
    
    print("\n========================================")
    print("MFS100 LIVE BLOOD GROUP PREDICTION")
    print("========================================")
    print(f"\nPrediction: {pred_class}")
    print(f"\nConfidence: {confidence:.2f}%")
    print("\nProbabilities:\n")
    
    # Sort for display matching prompt: A+, B+, AB+, O+
    display_order = ["A+", "B+", "AB+", "O+"]
    probs_dict = {CLASS_MAPPING[i]: preds[i]*100 for i in range(len(FOUR_CLASSES))}
    
    for c in display_order:
        print(f"{c:4s} : {probs_dict[c]:.2f}%")
        
    print("\n========================================")
    print(f"\nCaptured image: {img_path}")
    print(f"Preprocessed image: {preprocessed_shape}")
    print("Inference completed successfully.")
    print("\nResearch prototype — fingerprint-based prediction is experimental and")
    print("not medically validated. Do not use this result for medical decisions.")
    
    return pred_class, confidence, probs_dict

def save_result(timestamp, capture_number, image_path, pred_class, confidence, probs_dict, actual_label):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "live_predictions.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    row = {
        "timestamp": timestamp,
        "capture_number": capture_number,
        "image_path": image_path,
        "predicted_class": pred_class,
        "confidence": confidence,
        "prob_A_plus": probs_dict["A+"],
        "prob_B_plus": probs_dict["B+"],
        "prob_AB_plus": probs_dict["AB+"],
        "prob_O_plus": probs_dict["O+"],
        "actual_label": actual_label
    }
    
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def main():
    parser = argparse.ArgumentParser(description="Day 27 Live MFS100 Inference")
    parser.add_argument("--image", type=str, help="Path to a previously captured image for testing")
    args = parser.parse_args()

    print("DAY 27 LIVE MFS100 TEST")
    print("-----------------------")
    print("Model:")
    print(MODEL_PATH)
    print("\nClasses:")
    for i in range(len(FOUR_CLASSES)):
        print(f"{i} -> {FOUR_CLASSES[i]}")
    print("\nPreprocessing:")
    print("C5")
    print("Input:")
    print("112x112x3")
    print("\nScanner:")
    print("Mantra MFS100")
    print("\nStatus:")
    print("READY\n")

    print("[1] Loading Day 27 model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[2] Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    out_shape = model.output_shape
    if out_shape[-1] != 4:
        print(f"Error: Model output shape is {out_shape}, expected 4 classes.")
        return
        
    print("[3] Class mapping verified.")
    print("[4] Loading C5 preprocessing...")

    if args.image:
        print(f"\n[TEST IMAGE MODE] Processing {args.image}")
        if not os.path.exists(args.image):
            print(f"Error: Image {args.image} not found.")
            return
            
        pil_img = load_image_to_pil(args.image)
        captured_shape = f"{pil_img.width}x{pil_img.height}"
        print(f"Captured image shape: {captured_shape}")
        
        input_array = transform_c5(args.image, target_size=(112, 112))
        print(f"Preprocessed image shape: {input_array.shape}")
        
        input_tensor = np.expand_dims(input_array, axis=0)
        print(f"Model input shape: {input_tensor.shape}")
        
        preds = model.predict(input_tensor, verbose=0)[0]
        
        display_prediction(preds, args.image, input_array.shape)
        return

    print("[5] MFS100 SDK initialization...")
    try:
        scanner = MFS100Scanner(mock=False)
        if not scanner.is_connected():
            connected = scanner.connect()
            if not connected:
                print("Scanner initialization failed.")
                print(scanner.get_device_info().get('status'))
                return
        print("[6] Scanner detected.")
    except Exception as e:
        print(f"Scanner initialization failed: {e}")
        return

    print("[7] Ready for fingerprint capture.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    capture_count = 1
    
    while True:
        input("\nPress Enter to capture a fingerprint (or Ctrl+C to exit)...")
        
        print("\nWaiting for finger...")
        try:
            # save in a separate live-test directory
            temp_path = os.path.join(OUTPUT_DIR, f"live_capture_{int(time.time())}.bmp")
            saved_path, img_bytes = scanner._capture_hardware(temp_path)
        except Exception as e:
            print(f"Capture failed: {e}")
            continue
            
        pil_img = load_image_to_pil(saved_path)
        captured_shape = f"{pil_img.width}x{pil_img.height}"
        print(f"Captured image shape: {captured_shape}")
        
        input_array = transform_c5(saved_path, target_size=(112, 112))
        print(f"Preprocessed image shape: {input_array.shape}")
        
        input_tensor = np.expand_dims(input_array, axis=0)
        print(f"Model input shape: {input_tensor.shape}")
        
        preds = model.predict(input_tensor, verbose=0)[0]
        
        pred_class, confidence, probs_dict = display_prediction(preds, saved_path, input_array.shape)
        
        actual_label = "UNKNOWN"
        knows_gt = input("\nDo you know the laboratory-confirmed blood group for this sample? (Y/N): ").strip().upper()
        if knows_gt == 'Y':
            gt = input("Enter actual blood group (A+, B+, AB+, O+): ").strip().upper()
            if gt in FOUR_CLASSES:
                actual_label = gt
            else:
                print("Invalid input. Storing as UNKNOWN.")
                
        timestamp_str = datetime.now().isoformat()
        save_result(timestamp_str, capture_count, saved_path, pred_class, confidence, probs_dict, actual_label)
        capture_count += 1
        
        again = input("\nScan another fingerprint? (Y/N): ").strip().upper()
        if again != 'Y':
            break

if __name__ == "__main__":
    main()
