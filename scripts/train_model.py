#!/usr/bin/env python3
"""
scripts/train_model.py
Production training script for Biometric Blood Group Identification using ResNet50:
- Loads train and validation splits from outputs/dataset_manifest.csv
- Uses centralized preprocessing (src/preprocessing.py)
- Mild realistic data augmentation applied only during training
- Two-stage Transfer Learning:
    Stage 1: Frozen ResNet50 backbone, train classification head
    Stage 2: Unfreeze upper residual blocks (conv5), fine-tune with low lr
- Computes mild class weights to mitigate class imbalance
- ModelCheckpoint (saving best val_accuracy), EarlyStopping, ReduceLROnPlateau
- Saves models/blood_group_resnet50.keras
- Saves outputs/training_accuracy.png, outputs/training_loss.png, outputs/training_results.json
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
    DATASET_MANIFEST_PATH,
    MODEL_PATH,
    OUTPUTS_DIR,
    TRAINING_RESULTS_PATH
)
from src.preprocessing import load_and_preprocess_for_training

def build_data_augmentation():
    """
    Mild, realistic data augmentation that preserves ridge flow and orientation.
    No aggressive warping or distortion.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.03, fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(0.05)
    ], name="data_augmentation")

def load_data_split(manifest_path, split_name):
    df = pd.read_csv(manifest_path)
    split_df = df[df["split"] == split_name].copy()
    if len(split_df) == 0:
        raise ValueError(f"No records found for split: {split_name}")
    paths = split_df["image_path"].tolist()
    labels = split_df["class_id"].astype(int).tolist()
    return paths, labels

def create_tf_dataset(paths, labels, batch_size, is_training=False, augment_layer=None):
    """
    Builds a high-performance tf.data pipeline using our centralized image preprocessor.
    """
    def _py_load_image(path_tensor):
        p = path_tensor.numpy().decode("utf-8")
        img_arr = load_and_preprocess_for_training(p, target_size=IMAGE_SIZE)
        return img_arr

    def _tf_map(path, label):
        img = tf.py_function(_py_load_image, inp=[path], Tout=tf.float32)
        img.set_shape(INPUT_SHAPE)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2048), seed=42)
    
    dataset = dataset.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and augment_layer is not None:
        dataset = dataset.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_resnet50_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE):
    """
    ResNet50 Architecture:
    Input -> ResNet50 (include_top=False, weights='imagenet') -> GlobalAveragePooling2D
          -> Dense(256, relu) -> Dropout(0.4) -> Dense(8, softmax)
    """
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    # Freeze backbone for Stage 1
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_ResNet50")
    return model, base_model

def plot_and_save_history(history_dict, output_dir=OUTPUTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy plot
    plt.figure(figsize=(9, 5))
    plt.plot(history_dict["accuracy"], label="Training Accuracy", marker="o", color="#2563eb")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy", marker="s", color="#10b981")
    plt.title("Biometric Blood Group Identification - Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "training_accuracy.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"[+] Saved accuracy plot to: {acc_path}")

    # 2. Loss plot
    plt.figure(figsize=(9, 5))
    plt.plot(history_dict["loss"], label="Training Loss", marker="o", color="#dc2626")
    plt.plot(history_dict["val_loss"], label="Validation Loss", marker="s", color="#f59e0b")
    plt.title("Biometric Blood Group Identification - Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse Categorical Crossentropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss plot to: {loss_path}")

def train(args):
    print("=" * 70)
    print("STEP 6: RESNET50 TRANSFER LEARNING MODEL TRAINING")
    print("=" * 70)
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Stage 1 Epochs: {args.stage1_epochs}, Stage 2 Epochs: {args.stage2_epochs}")

    # 1. Load splits from manifest
    if not os.path.exists(DATASET_MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found at {DATASET_MANIFEST_PATH}. Run scripts/prepare_dataset.py first!")

    train_paths, train_labels = load_data_split(DATASET_MANIFEST_PATH, "train")
    val_paths, val_labels = load_data_split(DATASET_MANIFEST_PATH, "val")
    print(f"[+] Loaded {len(train_paths)} training samples and {len(val_paths)} validation samples.")

    # 2. Compute mild class weights
    classes_arr = np.array(train_labels)
    unique_classes = np.unique(classes_arr)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes_arr
    )
    # Mild dampening to prevent gradient spikes
    # weight_mild = 0.6 * (raw_weight - 1.0) + 1.0
    mild_class_weights = {
        int(c): float(np.round(0.6 * (w - 1.0) + 1.0, 4))
        for c, w in zip(unique_classes, raw_weights)
    }
    print("\nMild Class Weights for Training:")
    for cid, w in mild_class_weights.items():
        print(f"  Class {CLASS_NAMES[cid]:5s} (ID {cid}): {w:.4f}")

    # 3. Create datasets
    augment_layer = build_data_augmentation()
    train_ds = create_tf_dataset(train_paths, train_labels, args.batch_size, is_training=True, augment_layer=augment_layer)
    val_ds = create_tf_dataset(val_paths, val_labels, args.batch_size, is_training=False)

    # 4. Build Model
    model, base_model = build_resnet50_model(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    model.summary()

    # 5. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    history_combined = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": []
    }

    # =========================================================================
    # STAGE 1: Train Classification Head (Backbone Frozen)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Training Classification Head (Backbone Frozen)")
    print("=" * 60)
    base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.stage1_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=mild_class_weights,
        callbacks=[checkpoint_cb, reduce_lr_cb],
        verbose=1
    )

    for k in history_combined:
        history_combined[k].extend(hist1.history[k])

    best_val_acc_s1 = max(hist1.history["val_accuracy"])
    print(f"\n[+] Stage 1 Best Val Accuracy: {best_val_acc_s1:.4f}")

    # =========================================================================
    # STAGE 2: Fine-Tuning Top Residual Blocks
    # =========================================================================
    if args.stage2_epochs > 0:
        print("\n" + "=" * 60)
        print("STAGE 2: Fine-Tuning Top Layers of ResNet50 (conv5_block*)")
        print("=" * 60)

        # Unfreeze base model
        base_model.trainable = True

        # Keep BatchNormalization layers frozen for stable fine-tuning
        # Unfreeze only conv5 blocks (upper layers)
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            elif "conv5" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        trainable_count = sum(len(layer.weights) for layer in model.layers if layer.trainable)
        print(f"[+] Fine-tuning configured with top residual blocks unfreezed.")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.stage2_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.stage2_epochs,
            class_weight=mild_class_weights,
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
            verbose=1
        )

        for k in history_combined:
            history_combined[k].extend(hist2.history[k])

    # 6. Load Best Model Checkpoint
    if os.path.exists(MODEL_PATH):
        print(f"\n[*] Reloading best checkpoint from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)

    # 7. Final Validation Evaluation
    final_val_loss, final_val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n[+] Final Best Model Validation Accuracy: {final_val_acc:.4f} (Loss: {final_val_loss:.4f})")

    # 8. Save Training History Curves
    plot_and_save_history(history_combined, output_dir=OUTPUTS_DIR)

    # 9. Save Training Results Metadata
    training_summary = {
        "model_name": "ResNet50_TransferLearning",
        "timestamp": datetime.datetime.now().isoformat(),
        "image_size": list(IMAGE_SIZE),
        "input_shape": list(INPUT_SHAPE),
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "dataset_size": len(train_paths) + len(val_paths),
        "train_count": len(train_paths),
        "validation_count": len(val_paths),
        "stage1_epochs": len(hist1.history["accuracy"]),
        "stage2_epochs": len(hist2.history["accuracy"]) if args.stage2_epochs > 0 else 0,
        "total_epochs": len(history_combined["accuracy"]),
        "best_val_accuracy": float(final_val_acc),
        "best_val_loss": float(final_val_loss),
        "learning_rates": {
            "stage1": args.stage1_lr,
            "stage2": args.stage2_lr
        },
        "augmentation": {
            "rotation_factor": 0.03,
            "translation_factor": 0.03,
            "zoom_factor": 0.03,
            "brightness_factor": 0.05
        },
        "preprocessing": "RGBA alpha composite on white background, 224x224 RGB, resnet50.preprocess_input",
        "model_path": MODEL_PATH
    }

    with open(TRAINING_RESULTS_PATH, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"[+] Saved training results to: {TRAINING_RESULTS_PATH}")
    print("\n[+] TRAINING PIPELINE COMPLETE!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 for Blood Group Identification")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--stage1_epochs", type=int, default=8, help="Epochs for Stage 1 head training")
    parser.add_argument("--stage2_epochs", type=int, default=6, help="Epochs for Stage 2 fine-tuning")
    parser.add_argument("--stage1_lr", type=float, default=1e-3, help="Learning rate for Stage 1")
    parser.add_argument("--stage2_lr", type=float, default=1e-5, help="Learning rate for Stage 2")

    args = parser.parse_args()
    train(args)
