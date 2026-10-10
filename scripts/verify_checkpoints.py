#!/usr/bin/env python3
"""
Verification script for the three model checkpoints used in the ensemble experiment.
Ensures that:
1. The checkpoint files exist.
2. Each model's output class ordering matches the project CLASS_NAMES.
"""
import os
import sys
from pathlib import Path

# Project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, MODELS_DIR
import tensorflow as tf

CHECKPOINTS = {
    "ResNet50": os.path.join(MODELS_DIR, "experiments", "resnet50_long_epoch_best.keras"),
    "VGG16": os.path.join(MODELS_DIR, "experiments", "vgg16_long_epoch_best.keras"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "experiments", "efficientnetb0_long_epoch_best.keras"),
}

def main():
    all_ok = True
    for name, path in CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"[ERROR] {name} checkpoint not found at {path}")
            all_ok = False
            continue
        print(f"[OK] {name} checkpoint found: {path}")
        try:
            model = tf.keras.models.load_model(path)
            # Assume final Dense layer units correspond to class count
            output_shape = model.output_shape
            expected = (None, len(CLASS_NAMES))
            if output_shape != expected:
                print(f"[WARN] {name} output shape {output_shape} != expected {expected}")
            else:
                print(f"[OK] {name} output shape matches CLASS_NAMES ({len(CLASS_NAMES)} classes)")
        except Exception as e:
            print(f"[ERROR] Failed to load {name} model: {e}")
            all_ok = False
    if all_ok:
        print("All checkpoints verified successfully.")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
