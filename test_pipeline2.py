
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import live_app.backend.main as main
import numpy as np

print("Loading resources...")
main.load_resources()

images = [
    "reports/live_mfs100_tests/live_capture_1790484664.bmp",
    "reports/live_mfs100_tests/live_capture_1790485052.bmp"
]

for img in images:
    input_array = main.transform_c5(img, target_size=(112, 112))
    input_tensor = np.expand_dims(input_array, axis=0)
    preds = main.model.predict(input_tensor, verbose=0)[0]
    
    pred_idx = np.argmax(preds)
    pred_class = main.CLASS_MAPPING[pred_idx]
    confidence = float(preds[pred_idx])
    
    print(f"Image: {img}")
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence * 100:.6f}%")
    print("-" * 20)

print("Scanner Status:", main.get_scanner_status())
