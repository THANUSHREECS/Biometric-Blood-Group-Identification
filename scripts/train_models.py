#!/usr/bin/env python3
"""
scripts/train_models.py
Multi-Architecture Model Comparison Training Framework for Blood Group Classification.

Supports 7 architectures:
  1. custom_cnn       - Custom CNN baseline (no pretrained weights)
  2. mobilenetv2      - MobileNetV2 transfer learning
  3. efficientnetb0   - EfficientNetB0 transfer learning
  4. densenet121      - DenseNet121 transfer learning
  5. resnet50         - ResNet50 improved (more epochs + deeper fine-tuning than baseline)
  6. vgg16            - VGG16 transfer learning
  7. vgg19            - VGG19 transfer learning

Training Protocol:
  - Stage 1: Freeze backbone, train classification head
  - Stage 2: Unfreeze upper backbone layers, fine-tune with low LR
  - EarlyStopping, ModelCheckpoint, ReduceLROnPlateau on val_accuracy
  - Conservative fingerprint-safe augmentation (rotation ±3°, translate 3%, zoom 3%)
  - Mild class weights to handle imbalance
  - Each model gets its own preprocess_input function
  - Results logged to outputs/model_experiments.csv

RULES:
  - Test set is NEVER touched during training
  - Validation set is used for model selection ONLY
  - Baseline model (blood_group_resnet50.keras) is NEVER overwritten

Usage:
  # Pilot mode (quick 3+3 epoch rank test):
  python scripts/train_models.py --pilot --arch all

  # Pilot for specific architectures:
  python scripts/train_models.py --pilot --arch mobilenetv2 efficientnetb0

  # Full training for specific architectures:
  python scripts/train_models.py --arch mobilenetv2 efficientnetb0 densenet121

  # Full training, single arch:
  python scripts/train_models.py --arch resnet50
"""

import os
import sys
import csv
import json
import time
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
)
from src.preprocessing import composite_alpha_on_white, resize_to_target

# ============================================================================
# CONSTANTS
# ============================================================================

ALL_ARCHS = [
    "custom_cnn",
    "mobilenetv2",
    "efficientnetb0",
    "densenet121",
    "resnet50",
    "vgg16",
    "vgg19",
]

# Model-specific preprocessing functions
PREPROCESS_FUNCS = {
    "custom_cnn":     lambda x: (x / 127.5) - 1.0,   # [-1, 1] normalized
    "mobilenetv2":    tf.keras.applications.mobilenet_v2.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    "densenet121":    tf.keras.applications.densenet.preprocess_input,
    "resnet50":       tf.keras.applications.resnet50.preprocess_input,
    "vgg16":          tf.keras.applications.vgg16.preprocess_input,
    "vgg19":          tf.keras.applications.vgg19.preprocess_input,
}

# Architecture-specific input sizes
INPUT_SIZES = {
    "custom_cnn":     (96, 96),    # Native resolution to avoid upsampling
    "mobilenetv2":    (224, 224),
    "efficientnetb0": (224, 224),
    "densenet121":    (224, 224),
    "resnet50":       (224, 224),
    "vgg16":          (224, 224),
    "vgg19":          (224, 224),
}

EXPERIMENT_CSV = os.path.join(OUTPUTS_DIR, "model_experiments.csv")
CSV_HEADER = [
    "experiment_id", "model", "preprocessing", "augmentation",
    "class_weight_strategy", "learning_rate", "fine_tuning_strategy",
    "epochs", "best_val_accuracy", "best_val_loss", "val_macro_f1",
    "training_time_min", "parameters", "notes"
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_split(manifest_path, split_name):
    """Load image paths and labels for a given split from the manifest CSV."""
    df = pd.read_csv(manifest_path)
    split_df = df[df["split"] == split_name].copy()
    if len(split_df) == 0:
        raise ValueError(f"No records found for split: {split_name}")
    paths = split_df["image_path"].tolist()
    labels = split_df["class_id"].astype(int).tolist()
    return paths, labels


# ============================================================================
# PREPROCESSING & DATASET PIPELINE
# ============================================================================

def make_preprocess_fn(arch_name, target_size):
    """
    Returns a Python function that loads an image file, composites alpha,
    resizes, and applies the architecture-specific preprocess_input.
    """
    preprocess_fn = PREPROCESS_FUNCS[arch_name]

    def _load_and_preprocess(path_str):
        from PIL import Image
        with Image.open(path_str) as img:
            rgb_img = composite_alpha_on_white(img)
            resized_img = resize_to_target(rgb_img, target_size=target_size)
        arr = np.array(resized_img, dtype=np.float32)
        arr = preprocess_fn(arr)
        return arr

    return _load_and_preprocess


def build_augmentation_layer(arch_name):
    """
    Conservative augmentation safe for fingerprint ridge structure.
    No horizontal/vertical flipping (scientifically unjustified for fingerprints).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, fill_mode="nearest"),       # ±~10.8°
        tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.03, fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(0.05),
        tf.keras.layers.RandomContrast(0.05),
    ], name="fingerprint_augmentation")


def create_tf_dataset(paths, labels, arch_name, target_size, batch_size,
                      is_training=False, augment_layer=None):
    """
    Builds a tf.data pipeline with model-specific preprocessing.
    Augmentation is only applied during training.
    Validation/test data is NEVER augmented.
    """
    load_fn = make_preprocess_fn(arch_name, target_size)

    def _py_load(path_tensor):
        path_str = path_tensor.numpy().decode("utf-8")
        return load_fn(path_str)

    def _tf_map(path, label):
        img = tf.py_function(_py_load, inp=[path], Tout=tf.float32)
        img.set_shape((*target_size, 3))
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 2048), seed=42)

    dataset = dataset.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and augment_layer is not None:
        dataset = dataset.map(
            lambda x, y: (augment_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ============================================================================
# CLASS WEIGHT COMPUTATION
# ============================================================================

def compute_mild_class_weights(train_labels):
    """
    Compute mildly dampened class weights to handle imbalance without
    creating extreme gradient spikes.
    Formula: weight = 0.6 * (balanced_weight - 1.0) + 1.0
    """
    classes_arr = np.array(train_labels)
    unique_classes = np.unique(classes_arr)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes_arr
    )
    mild_weights = {
        int(c): float(np.round(0.6 * (w - 1.0) + 1.0, 4))
        for c, w in zip(unique_classes, raw_weights)
    }
    print("\n  Mild Class Weights:")
    for cid, w in mild_weights.items():
        print(f"    {CLASS_NAMES[cid]:5s} (ID {cid}): {w:.4f}")
    return mild_weights


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_custom_cnn(input_shape=(96, 96, 3), num_classes=NUM_CLASSES):
    """
    Custom CNN with 4 Conv blocks + GlobalAveragePooling head.
    Trained from scratch (no pretrained weights).
    Uses 96x96 native resolution to preserve spatial information without
    aggressive upsampling.
    """
    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool2")(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool3")(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Activation("relu", name="relu4")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool4")(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    return model, None  # No base model to unfreeze


def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_MobileNetV2")
    return model, base


def build_efficientnetb0(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_EfficientNetB0")
    return model, base


def build_densenet121(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_DenseNet121")
    return model, base


def build_resnet50_improved(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Improved ResNet50: Different save path from baseline to preserve baseline model.
    Unfreezes conv4+conv5 during fine-tuning (vs. baseline which only used conv5).
    """
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_ResNet50_Improved")
    return model, base


def build_vgg16(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_VGG16")
    return model, base


def build_vgg19(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="fingerprint_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="blood_group_output")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="BloodGroup_VGG19")
    return model, base


MODEL_BUILDERS = {
    "custom_cnn":     build_custom_cnn,
    "mobilenetv2":    build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
    "densenet121":    build_densenet121,
    "resnet50":       build_resnet50_improved,
    "vgg16":          build_vgg16,
    "vgg19":          build_vgg19,
}

# Model save paths (baseline resnet50 path is PRESERVED; improved version gets new name)
MODEL_SAVE_PATHS = {
    "custom_cnn":     os.path.join(MODELS_DIR, "custom_cnn_blood_group.keras"),
    "mobilenetv2":    os.path.join(MODELS_DIR, "mobilenetv2_blood_group.keras"),
    "efficientnetb0": os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
    "densenet121":    os.path.join(MODELS_DIR, "densenet121_blood_group.keras"),
    "resnet50":       os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
    "vgg16":          os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
    "vgg19":          os.path.join(MODELS_DIR, "vgg19_blood_group.keras"),
}


# ============================================================================
# FINE-TUNING STRATEGIES
# ============================================================================

def apply_finetune_strategy(arch_name, base_model):
    """
    Unfreeze an appropriate upper portion of each backbone for Stage 2.
    BatchNormalization layers are kept frozen for training stability.
    Returns a description of what was unfrozen.
    """
    if base_model is None:
        return "N/A (custom_cnn trained from scratch)"

    # First unfreeze all
    base_model.trainable = True

    if arch_name == "mobilenetv2":
        # Unfreeze last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        description = "last 30 layers (BatchNorm frozen)"

    elif arch_name == "efficientnetb0":
        # Unfreeze last 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        description = "last 50 layers (BatchNorm frozen)"

    elif arch_name == "densenet121":
        # Unfreeze last dense block (approximately last 50 layers)
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        description = "last dense block (~50 layers, BatchNorm frozen)"

    elif arch_name == "resnet50":
        # Unfreeze conv4 and conv5 blocks (improved over baseline which only used conv5)
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            elif "conv4" in layer.name or "conv5" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        description = "conv4 + conv5 blocks (BatchNorm frozen)"

    elif arch_name in ("vgg16", "vgg19"):
        # Unfreeze last 4 convolutional blocks (block4 and block5)
        for layer in base_model.layers:
            if "block5" in layer.name or "block4" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        description = "block4 + block5 conv layers"

    else:
        # Default: unfreeze last 30 layers, freeze BN
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        description = "last 30 layers (default, BatchNorm frozen)"

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  [+] Fine-tune: {description} ({trainable_count} trainable layers in backbone)")
    return description


# ============================================================================
# PLOTTING
# ============================================================================

def save_training_curves(history_dict, arch_name, output_dir=OUTPUTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(history_dict["accuracy"], label="Training Accuracy", marker="o", color="#2563eb")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy", marker="s", color="#10b981")
    plt.title(f"{arch_name.upper()} — Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(output_dir, f"{arch_name}_training_accuracy.png")
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # Loss
    plt.figure(figsize=(9, 5))
    plt.plot(history_dict["loss"], label="Training Loss", marker="o", color="#dc2626")
    plt.plot(history_dict["val_loss"], label="Validation Loss", marker="s", color="#f59e0b")
    plt.title(f"{arch_name.upper()} — Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(output_dir, f"{arch_name}_training_loss.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()

    print(f"  [+] Saved training curves: {acc_path}, {loss_path}")


def save_confusion_matrix(y_true, y_pred, arch_name, split_name="val", output_dir=OUTPUTS_DIR):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, linewidths=0.5
    )
    plt.title(f"{arch_name.upper()} — Confusion Matrix ({split_name} set)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Blood Group", fontsize=11, fontweight="bold")
    plt.ylabel("True Blood Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{arch_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"  [+] Saved confusion matrix: {cm_path}")
    return cm


# ============================================================================
# EXPERIMENT LOGGING
# ============================================================================

def log_experiment(experiment_id, arch_name, best_val_acc, best_val_loss,
                   val_macro_f1, epochs_trained, training_time_min,
                   param_count, is_pilot, finetune_desc, stage1_lr, stage2_lr,
                   notes=""):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Ensure header exists
    if not os.path.exists(EXPERIMENT_CSV):
        with open(EXPERIMENT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    mode = "pilot" if is_pilot else "full"
    preprocess_name = f"{arch_name}.preprocess_input"
    augmentation = "rotation±3%,translate3%,zoom3%,brightness±5%,contrast±5%"
    cw_strategy = "mild_balanced (0.6*(w-1)+1)"
    lr_str = f"s1={stage1_lr}, s2={stage2_lr}"

    row = [
        experiment_id, arch_name, preprocess_name, augmentation,
        cw_strategy, lr_str, finetune_desc, epochs_trained,
        round(float(best_val_acc), 4), round(float(best_val_loss), 4),
        round(float(val_macro_f1), 4), round(float(training_time_min), 2),
        int(param_count), f"[{mode}] {notes}"
    ]

    with open(EXPERIMENT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"  [+] Experiment logged: {experiment_id} | val_acc={best_val_acc:.4f} | macro_f1={val_macro_f1:.4f}")


# ============================================================================
# VALIDATION METRICS COMPUTATION
# ============================================================================

def compute_val_metrics(model, val_ds, val_labels):
    """Compute validation predictions and metrics without touching the test set."""
    all_probs = []
    for batch_imgs, _ in val_ds:
        probs = model.predict(batch_imgs, verbose=0)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    pred_labels = np.argmax(all_probs, axis=1)
    true_labels = np.array(val_labels)

    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    val_acc = accuracy_score(true_labels, pred_labels)

    print(f"  Validation accuracy (post-train check): {val_acc:.4f}")
    print(f"  Validation Macro F1:                   {f1_macro:.4f}")

    clf_report = classification_report(
        true_labels, pred_labels,
        target_names=CLASS_NAMES, zero_division=0
    )
    print(f"\n  Classification Report:\n{clf_report}")

    return pred_labels, true_labels, float(f1_macro), float(val_acc)


# ============================================================================
# SINGLE ARCHITECTURE TRAINING
# ============================================================================

def train_architecture(arch_name, args, is_pilot=False, experiment_counter=None):
    """
    Full two-stage training for a single architecture.
    Returns a results dict with val accuracy and macro F1.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING: {arch_name.upper()} | mode={'PILOT' if is_pilot else 'FULL'}")
    print("=" * 70)

    target_size = INPUT_SIZES[arch_name]
    input_shape = (*target_size, 3)
    save_path = MODEL_SAVE_PATHS[arch_name]

    # Pilot vs full epoch counts
    if is_pilot:
        s1_epochs = 3
        s2_epochs = 3
    else:
        if arch_name == "custom_cnn":
            s1_epochs = args.s1_epochs
            s2_epochs = 0   # No backbone to fine-tune
        elif arch_name in ("vgg16", "vgg19"):
            s1_epochs = args.s1_epochs
            s2_epochs = args.s2_epochs
        else:
            s1_epochs = args.s1_epochs
            s2_epochs = args.s2_epochs

    stage1_lr = args.stage1_lr
    stage2_lr = args.stage2_lr

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_paths, train_labels = load_split(DATASET_MANIFEST_PATH, "train")
    val_paths, val_labels = load_split(DATASET_MANIFEST_PATH, "val")
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")

    mild_class_weights = compute_mild_class_weights(train_labels)
    augment_layer = build_augmentation_layer(arch_name)

    train_ds = create_tf_dataset(train_paths, train_labels, arch_name, target_size,
                                  args.batch_size, is_training=True, augment_layer=augment_layer)
    val_ds = create_tf_dataset(val_paths, val_labels, arch_name, target_size,
                                args.batch_size, is_training=False)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    build_fn = MODEL_BUILDERS[arch_name]
    model, base_model = build_fn(input_shape=input_shape, num_classes=NUM_CLASSES)
    param_count = model.count_params()
    print(f"  Total parameters: {param_count:,}")

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=7 if not is_pilot else 3,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3 if not is_pilot else 2,
        min_lr=1e-7,
        verbose=1
    )

    history_combined = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    start_time = time.time()

    # -----------------------------------------------------------------------
    # STAGE 1: Train Classification Head (Backbone Frozen)
    # -----------------------------------------------------------------------
    print(f"\n  --- STAGE 1: Head training ({s1_epochs} max epochs) ---")
    if base_model is not None:
        base_model.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage1_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    s1_callbacks = [checkpoint_cb, reduce_lr_cb]
    if arch_name == "custom_cnn":
        s1_callbacks.append(early_stop_cb)  # Custom CNN uses early stop from start

    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=s1_epochs,
        class_weight=mild_class_weights,
        callbacks=s1_callbacks,
        verbose=1
    )

    for k in history_combined:
        history_combined[k].extend(hist1.history.get(k, []))

    best_s1_val = max(hist1.history["val_accuracy"]) if hist1.history["val_accuracy"] else 0.0
    print(f"\n  Stage 1 best val accuracy: {best_s1_val:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 2: Fine-Tuning (for pretrained models only)
    # -----------------------------------------------------------------------
    finetune_desc = "N/A (custom_cnn)"
    if base_model is not None and s2_epochs > 0:
        print(f"\n  --- STAGE 2: Fine-tuning ({s2_epochs} max epochs) ---")
        finetune_desc = apply_finetune_strategy(arch_name, base_model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=stage2_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=s2_epochs,
            class_weight=mild_class_weights,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
            verbose=1
        )

        for k in history_combined:
            history_combined[k].extend(hist2.history.get(k, []))

        best_s2_val = max(hist2.history["val_accuracy"]) if hist2.history["val_accuracy"] else 0.0
        print(f"\n  Stage 2 best val accuracy: {best_s2_val:.4f}")

    elapsed_min = (time.time() - start_time) / 60.0
    total_epochs = len(history_combined["accuracy"])

    # -----------------------------------------------------------------------
    # Load best checkpoint and evaluate on validation
    # -----------------------------------------------------------------------
    if os.path.exists(save_path):
        print(f"\n  [*] Loading best checkpoint from {save_path}...")
        model = tf.keras.models.load_model(save_path)

    best_val_acc = max(history_combined["val_accuracy"]) if history_combined["val_accuracy"] else 0.0
    best_val_loss = min(history_combined["val_loss"]) if history_combined["val_loss"] else 9.99

    # Compute macro F1 on validation set
    print(f"\n  Computing validation metrics (macro F1)...")
    val_pred_labels, val_true_labels, val_macro_f1, _ = compute_val_metrics(
        model, val_ds, val_labels
    )

    # -----------------------------------------------------------------------
    # Save confusion matrix and training curves
    # -----------------------------------------------------------------------
    save_confusion_matrix(val_true_labels, val_pred_labels, arch_name, split_name="val")
    save_training_curves(history_combined, arch_name)

    # -----------------------------------------------------------------------
    # Log experiment
    # -----------------------------------------------------------------------
    exp_id = experiment_counter if experiment_counter else f"{arch_name}_001"
    notes = f"best_epoch={history_combined['val_accuracy'].index(max(history_combined['val_accuracy']))+1}"
    log_experiment(
        experiment_id=exp_id,
        arch_name=arch_name,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        val_macro_f1=val_macro_f1,
        epochs_trained=total_epochs,
        training_time_min=elapsed_min,
        param_count=param_count,
        is_pilot=is_pilot,
        finetune_desc=finetune_desc,
        stage1_lr=stage1_lr,
        stage2_lr=stage2_lr,
        notes=notes
    )

    result = {
        "arch": arch_name,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "val_macro_f1": val_macro_f1,
        "epochs": total_epochs,
        "training_time_min": elapsed_min,
        "parameters": param_count,
        "model_path": save_path,
        "is_pilot": is_pilot
    }

    print(f"\n  {'='*50}")
    print(f"  DONE: {arch_name.upper()}")
    print(f"  Val Accuracy: {best_val_acc:.4f} | Macro F1: {val_macro_f1:.4f} | Time: {elapsed_min:.1f} min")
    print(f"  {'='*50}")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Architecture Blood Group Model Training Framework"
    )
    parser.add_argument(
        "--arch", nargs="+", default=["mobilenetv2"],
        choices=ALL_ARCHS + ["all"],
        help="Architectures to train. Use 'all' for all architectures."
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: 3+3 epochs per model to quickly rank architectures")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--s1_epochs", type=int, default=20,
                        help="Stage 1 (head training) max epochs")
    parser.add_argument("--s2_epochs", type=int, default=20,
                        help="Stage 2 (fine-tuning) max epochs")
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=2e-5)

    args = parser.parse_args()

    # Resolve 'all'
    archs_to_train = ALL_ARCHS if "all" in args.arch else args.arch

    print("=" * 70)
    print("BLOOD GROUP MODEL COMPARISON TRAINING FRAMEWORK")
    print("=" * 70)
    print(f"Mode: {'PILOT' if args.pilot else 'FULL TRAINING'}")
    print(f"Architectures: {archs_to_train}")
    print(f"Batch size: {args.batch_size}")
    if not args.pilot:
        print(f"Stage 1 epochs (max): {args.s1_epochs}")
        print(f"Stage 2 epochs (max): {args.s2_epochs}")
    print(f"Stage 1 LR: {args.stage1_lr} | Stage 2 LR: {args.stage2_lr}")
    print(f"Test set: LOCKED - not used during training")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    all_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    for i, arch_name in enumerate(archs_to_train):
        exp_id = f"{timestamp}_{arch_name}_{'pilot' if args.pilot else 'full'}"
        try:
            result = train_architecture(
                arch_name=arch_name,
                args=args,
                is_pilot=args.pilot,
                experiment_counter=exp_id
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n  [!] ERROR training {arch_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "arch": arch_name,
                "best_val_accuracy": 0.0,
                "best_val_loss": 99.0,
                "val_macro_f1": 0.0,
                "error": str(e)
            })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE — SUMMARY (VALIDATION PERFORMANCE ONLY)")
    print("=" * 70)
    print(f"{'Model':<16} | {'Val Acc':>9} | {'Macro F1':>9} | {'Epochs':>7} | {'Time(min)':>10} | {'Parameters':>12}")
    print("-" * 75)

    results_sorted = sorted(all_results, key=lambda r: r.get("best_val_accuracy", 0), reverse=True)
    for r in results_sorted:
        if "error" in r:
            print(f"{r['arch']:<16} | {'ERROR':>9} | {'':>9} | {'':>7} | {'':>10} | {'':>12}")
        else:
            print(
                f"{r['arch']:<16} | {r['best_val_accuracy']:>9.4f} | "
                f"{r['val_macro_f1']:>9.4f} | {r['epochs']:>7} | "
                f"{r['training_time_min']:>10.1f} | {r['parameters']:>12,}"
            )

    print("-" * 75)
    if results_sorted and "error" not in results_sorted[0]:
        best = results_sorted[0]
        print(f"\nBest architecture so far: {best['arch'].upper()} | val_acc={best['best_val_accuracy']:.4f} | macro_f1={best['val_macro_f1']:.4f}")

    # Save summary JSON
    summary_path = os.path.join(OUTPUTS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "pilot" if args.pilot else "full",
            "architectures_trained": archs_to_train,
            "results": all_results
        }, f, indent=2)
    print(f"\n[+] Summary saved to: {summary_path}")
    print("\nNOTE: Test set was NOT evaluated. Model selection is based on validation performance.")
    print("      Run compare_models.py to generate the comparison report.")


if __name__ == "__main__":
    main()
