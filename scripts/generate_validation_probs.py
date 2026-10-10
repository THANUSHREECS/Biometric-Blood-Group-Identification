#!/usr/bin/env python3
"""
Generate validation probability predictions for the three models used in the ensemble experiment.
Outputs NumPy arrays saved under `outputs/`:
- val_probs_resnet50.npy
- val_probs_vgg16.npy
- val_probs_efficientnetb0.npy
Also saves the true validation labels as `val_labels.npy` for later use.
"""
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Set project root (scripts directory is two levels below project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, DATASET_MANIFEST_PATH, OUTPUTS_DIR
# import removed: make_preprocess_fn not needed
from scripts.train_models import (
    load_split,
    create_tf_dataset,
    INPUT_SIZES,
    MODEL_BUILDERS,
)

# Model specifications
MODELS = {
    "resnet50": {
        "arch": "resnet50",
        "checkpoint": os.path.join(MODELS_DIR, "experiments", "resnet50_long_epoch_best.keras"),
        "output": os.path.join(OUTPUTS_DIR, "val_probs_resnet50.npy"),
    },
    "vgg16": {
        "arch": "vgg16",
        "checkpoint": os.path.join(MODELS_DIR, "experiments", "vgg16_long_epoch_best.keras"),
        "output": os.path.join(OUTPUTS_DIR, "val_probs_vgg16.npy"),
    },
    "efficientnetb0": {
        "arch": "efficientnetb0",
        "checkpoint": os.path.join(MODELS_DIR, "experiments", "efficientnetb0_long_epoch_best.keras"),
        "output": os.path.join(OUTPUTS_DIR, "val_probs_efficientnetb0.npy"),
    },
}

def load_validation_data(arch_name, target_size, batch_size=32):
    # Load validation split paths and labels
    val_paths, val_labels = load_split(DATASET_MANIFEST_PATH, "val")
    # No augmentation for validation
    val_ds = create_tf_dataset(
        val_paths,
        val_labels,
        arch_name,
        target_size,
        batch_size,
        is_training=False,
    )
    return val_ds, np.array(val_labels)

def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Save true labels once (they are the same for all models)
    labels_saved = False
    for name, spec in MODELS.items():
        arch = spec["arch"]
        checkpoint = spec["checkpoint"]
        output_path = spec["output"]
        if not os.path.isfile(checkpoint):
            print(f"[ERROR] Checkpoint for {name} not found at {checkpoint}")
            sys.exit(1)
        # Determine input size for the architecture
        target_size = INPUT_SIZES[arch]
        # Load validation dataset
        val_ds, val_labels = load_validation_data(arch, target_size)
        if not labels_saved:
            label_path = os.path.join(OUTPUTS_DIR, "val_labels.npy")
            np.save(label_path, val_labels)
            labels_saved = True
        # Load model
        model = tf.keras.models.load_model(checkpoint)
        # Predict probabilities
        probs = model.predict(val_ds, verbose=1)
        # Ensure shape is (N, num_classes)
        np.save(output_path, probs)
        print(f"[DONE] Saved {name} validation probabilities to {output_path}")

if __name__ == "__main__":
    main()
