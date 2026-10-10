#!/usr/bin/env python3
"""
scripts/train_abo_positive_4class.py
====================================
4-Class ABO-Positive Fingerprint Classification Prototype Training Script.

Scientific Framing:
AI-assisted research prototype investigating statistical associations
between fingerprint images and ABO blood-group labels.
This is NOT a clinically validated blood-group test.
Fingerprint images do NOT biologically determine blood group.

Protected Classes (ABO-Positive):
  A+, AB+, B+, O+

Excluded Classes (Rh-Negative):
  A-, AB-, B-, O-

Canonical Class Mapping:
  {"A+": 0, "AB+": 1, "B+": 2, "O+": 3}

Strict Protections:
  - Additive experiment only. Production models are untouched.
  - Preprocessing verified for EfficientNetB0: [0, 255] RGB float32.
  - Class weights computed ONLY from TRAIN split.
  - Conservative augmentation (TRAIN only, no flips).
  - Two-stage training with checkpointing on validation metrics.
  - Baselines computed from actual TRAIN distribution.
  - TEST split is strictly excluded during training and model selection.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

import tensorflow as tf

# Resolve project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import composite_alpha_on_white, resize_to_target

# Paths
DATASET_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "outputs", "dataset_manifest.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "abo_positive_4class")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Constants & Class Mapping
FOUR_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_ID = {
    "A+": 0,
    "AB+": 1,
    "B+": 2,
    "O+": 3
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)

EXPECTED_COUNTS = {
    "train": {"A+": 395, "AB+": 493, "B+": 454, "O+": 596, "total": 1938},
    "val":   {"A+": 85,  "AB+": 106, "B+": 97,  "O+": 128, "total": 416},
    "test":  {"A+": 85,  "AB+": 106, "B+": 97,  "O+": 127, "total": 415},
}


def compute_file_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_and_verify_dataset():
    """
    Loads dataset manifest and filters strictly the 4 ABO-positive classes.
    Verifies that split counts match canonical partitions and no overlap exists.
    """
    print("[*] Loading dataset manifest from:", DATASET_MANIFEST_PATH)
    if not os.path.exists(DATASET_MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {DATASET_MANIFEST_PATH}")

    df = pd.read_csv(DATASET_MANIFEST_PATH)
    filtered_df = df[df["class_name"].isin(FOUR_CLASSES)].copy().reset_index(drop=True)

    # Normalize image paths
    resolved_paths = []
    for p in filtered_df["image_path"]:
        if os.path.exists(p):
            resolved_paths.append(p)
        else:
            # Fallback relative resolution
            base_rel = os.path.relpath(p, "C:/Users/thanu/OneDrive/Desktop/major project") if "OneDrive/Desktop" in p else p
            local_p = os.path.join(PROJECT_ROOT, base_rel)
            if os.path.exists(local_p):
                resolved_paths.append(local_p)
            else:
                raise FileNotFoundError(f"Could not locate image file: {p}")
    filtered_df["image_path"] = resolved_paths
    filtered_df["class_id_4class"] = filtered_df["class_name"].map(CLASS_TO_ID)

    # Check partition counts
    for split_name in ["train", "val", "test"]:
        sub = filtered_df[filtered_df["split"] == split_name]
        counts = sub["class_name"].value_counts().to_dict()
        total = len(sub)
        print(f"[*] Split '{split_name}': total={total}, counts={counts}")
        exp = EXPECTED_COUNTS[split_name]
        if total != exp["total"]:
            print(f"[!] Warning: {split_name} count {total} differs from expected {exp['total']}")
        for cname in FOUR_CLASSES:
            if counts.get(cname, 0) != exp[cname]:
                print(f"[!] Warning: {split_name} class {cname} count {counts.get(cname, 0)} != {exp[cname]}")

    # Verify zero partition leakage
    train_paths = set(filtered_df[filtered_df["split"] == "train"]["image_path"])
    val_paths = set(filtered_df[filtered_df["split"] == "val"]["image_path"])
    test_paths = set(filtered_df[filtered_df["split"] == "test"]["image_path"])

    assert len(train_paths.intersection(val_paths)) == 0, "Leakage between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Leakage between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Leakage between val and test!"
    print("[+] Verified strict partition isolation. Zero image overlap detected.")

    return filtered_df


def load_split_arrays(split_df):
    """
    Loads images and labels into numpy arrays.
    Preprocessing convention:
      - RGB conversion
      - Alpha compositing onto clean white background
      - Lanczos interpolation resize to (224, 224)
      - Float32 array in [0, 255] range (internal EfficientNetB0 rescaling/normalization).
    """
    paths = split_df["image_path"].tolist()
    labels = split_df["class_id_4class"].to_numpy(dtype=np.int32)
    cnames = split_df["class_name"].tolist()

    imgs = []
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            rgb = composite_alpha_on_white(img)
            resized = resize_to_target(rgb, (224, 224))
            arr = np.array(resized, dtype=np.float32)
            imgs.append(arr)

    return np.array(imgs, dtype=np.float32), labels, paths, cnames


def compute_and_save_baselines(y_train, y_val, y_test):
    """
    Calculates Majority-Class and Random Baselines from ACTUAL TRAIN DISTRIBUTION.
    """
    print("\n" + "=" * 70)
    print("CALCULATING 4-CLASS BASELINES (FROM ACTUAL TRAIN DISTRIBUTION)")
    print("=" * 70)

    train_counts = {int(k): int(v) for k, v in Counter(y_train).items()}
    n_train = int(len(y_train))
    majority_class_id = int(max(train_counts, key=train_counts.get))
    majority_class_name = ID_TO_CLASS[majority_class_id]
    majority_train_pct = float(train_counts[majority_class_id]) / float(n_train)

    print(f"[*] Train class counts: {train_counts}")
    print(f"[*] Majority class in train: {majority_class_name} (ID {majority_class_id}) with {train_counts[majority_class_id]}/{n_train} ({majority_train_pct:.2%})")

    # Majority Baseline on Val
    y_pred_maj_val = np.full_like(y_val, majority_class_id)
    acc_maj_val = accuracy_score(y_val, y_pred_maj_val)
    p_maj_val, r_maj_val, f1_maj_val, _ = precision_recall_fscore_support(
        y_val, y_pred_maj_val, average="macro", zero_division=0
    )

    # Majority Baseline on Test
    y_pred_maj_test = np.full_like(y_test, majority_class_id)
    acc_maj_test = accuracy_score(y_test, y_pred_maj_test)
    p_maj_test, r_maj_test, f1_maj_test, _ = precision_recall_fscore_support(
        y_test, y_pred_maj_test, average="macro", zero_division=0
    )

    # Uniform Random Baseline (theoretical)
    acc_uniform_random = 0.25
    macro_f1_uniform_random = 0.25

    # Stratified Random Baseline (train proportions)
    train_probs = [float(train_counts[i]) / float(n_train) for i in range(NUM_CLASSES)]
    val_counts = {int(k): int(v) for k, v in Counter(y_val).items()}
    test_counts = {int(k): int(v) for k, v in Counter(y_test).items()}
    acc_strat_val = sum(train_probs[i] * (val_counts[i] / float(len(y_val))) for i in range(NUM_CLASSES))
    acc_strat_test = sum(train_probs[i] * (test_counts[i] / float(len(y_test))) for i in range(NUM_CLASSES))

    baseline_metrics = {
        "scientific_context": {
            "description": "Baseline benchmarks for 4-class ABO-positive fingerprint classification prototype.",
            "purpose": "Provides lower-bound performance context before model evaluation. Not used in model tuning.",
            "classes": FOUR_CLASSES,
            "train_distribution": {ID_TO_CLASS[i]: int(train_counts[i]) for i in range(NUM_CLASSES)},
            "majority_class": majority_class_name,
            "majority_class_id": int(majority_class_id)
        },
        "majority_class_baseline": {
            "strategy": f"Always predict majority class from train ({majority_class_name})",
            "validation": {
                "accuracy": round(float(acc_maj_val), 4),
                "macro_precision": round(float(p_maj_val), 4),
                "macro_recall": round(float(r_maj_val), 4),
                "macro_f1": round(float(f1_maj_val), 4)
            },
            "test": {
                "accuracy": round(float(acc_maj_test), 4),
                "macro_precision": round(float(p_maj_test), 4),
                "macro_recall": round(float(r_maj_test), 4),
                "macro_f1": round(float(f1_maj_test), 4)
            }
        },
        "random_baseline": {
            "uniform_random": {
                "strategy": "Equal probability prediction across 4 classes (p = 0.25)",
                "expected_accuracy": float(acc_uniform_random),
                "expected_macro_f1": float(macro_f1_uniform_random)
            },
            "stratified_random": {
                "strategy": "Prediction sampled from empirical train class prior distribution",
                "train_priors": {ID_TO_CLASS[i]: round(float(train_probs[i]), 4) for i in range(NUM_CLASSES)},
                "validation_expected_accuracy": round(float(acc_strat_val), 4),
                "test_expected_accuracy": round(float(acc_strat_test), 4)
            }
        }
    }

    baseline_path = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"[+] Baseline metrics saved to: {baseline_path}")
    print(f"    - Majority Baseline Val Acc:  {acc_maj_val:.2%} (Macro F1: {f1_maj_val:.4f})")
    print(f"    - Uniform Random Baseline Acc: {acc_uniform_random:.2%}")
    print(f"    - Stratified Random Val Acc:  {acc_strat_val:.2%}")

    return baseline_metrics


def build_augmentation_layer():
    """
    Conservative augmentation safe for fingerprint ridge structure.
    NO horizontal or vertical flips.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.03, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.03, 0.03, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.03, fill_mode="nearest"),
        tf.keras.layers.RandomContrast(0.05),
    ], name="conservative_augmentation")


def build_efficientnetb0_4class_model():
    """
    Builds the 4-class ABO-positive EfficientNetB0 classification model.
    Architecture:
      Input (224, 224, 3) [0, 255 RGB]
          ↓
      EfficientNetB0 ImageNet backbone (internal rescaling & normalization)
          ↓
      Global Average Pooling 2D
          ↓
      Dense(256, activation="relu", kernel_regularizer=l2(1e-4))
          ↓
      Dropout(0.30)
          ↓
      Dense(4, activation="softmax", name="abo_positive_4class_output")
    """
    print("[*] Building EfficientNetB0 backbone (weights='imagenet')...")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="fingerprint_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap_pool")(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_256"
    )(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_0.30")(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="abo_positive_4class_output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ABO_Positive_4Class_EfficientNetB0")
    return model, base_model


def train_model(X_train, y_train, X_val, y_val):
    """
    Executes Two-Stage Training:
      Stage 1: Frozen backbone, train classification head (Adam lr=1e-3).
      Stage 2: Unfreeze top ~25 layers of backbone, fine-tune with Adam lr=3e-5.
    Uses balanced class weights computed ONLY from TRAIN split.
    Uses conservative augmentation on TRAIN only.
    """
    print("\n" + "=" * 70)
    print("STARTING TWO-STAGE TRAINING PIPELINE")
    print("=" * 70)

    # Compute balanced class weights strictly on TRAIN
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3]),
        y=y_train
    )
    class_weights = {i: float(w) for i, w in enumerate(raw_weights)}
    print(f"[*] Balanced class weights (from TRAIN only):")
    for i, w in class_weights.items():
        print(f"    {ID_TO_CLASS[i]}: {w:.4f}")

    # Build model
    model, base_model = build_efficientnetb0_4class_model()

    augment_layer = build_augmentation_layer()
    batch_size = 32

    # Build tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=42)
    train_ds = train_ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # ----------------------------------------------------
    # STAGE 1: Freeze backbone, train classification head
    # ----------------------------------------------------
    print("\n[*] --- STAGE 1: Head Training (Backbone Frozen) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    stage1_epochs = 8
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        verbose=1
    )

    # ----------------------------------------------------
    # STAGE 2: Unfreeze top ~25 layers, fine-tune with low LR
    # ----------------------------------------------------
    print("\n[*] --- STAGE 2: Fine-Tuning (Unfreeze Top ~25 Layers) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False
    for layer in base_model.layers[-25:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    best_checkpoint_path = os.path.join(MODELS_DIR, "best_model.keras")

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    stage2_epochs = 12
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    # Merge history
    combined_history = {
        "stage1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in history2.history.items()}
    }

    if not os.path.exists(best_checkpoint_path):
        model.save(best_checkpoint_path)

    plot_training_history(combined_history)

    history_path = os.path.join(OUTPUTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"[+] Training history saved to: {history_path}")

    return best_checkpoint_path, combined_history


def plot_training_history(history_data):
    """Generates and saves training and validation curves."""
    s1_acc = history_data["stage1"]["accuracy"]
    s1_val_acc = history_data["stage1"]["val_accuracy"]
    s1_loss = history_data["stage1"]["loss"]
    s1_val_loss = history_data["stage1"]["val_loss"]

    s2_acc = history_data["stage2"]["accuracy"]
    s2_val_acc = history_data["stage2"]["val_accuracy"]
    s2_loss = history_data["stage2"]["loss"]
    s2_val_loss = history_data["stage2"]["val_loss"]

    all_acc = s1_acc + s2_acc
    all_val_acc = s1_val_acc + s2_val_acc
    all_loss = s1_loss + s2_loss
    all_val_loss = s1_val_loss + s2_val_loss

    epochs = range(1, len(all_acc) + 1)
    s1_len = len(s1_acc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, all_acc, "b-o", label="Train Accuracy", linewidth=2)
    axes[0].plot(epochs, all_val_acc, "r--s", label="Val Accuracy", linewidth=2)
    axes[0].axvline(x=s1_len + 0.5, color="gray", linestyle=":", label="Stage 2 Transition")
    axes[0].set_title("4-Class ABO-Positive: Accuracy Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Loss
    axes[1].plot(epochs, all_loss, "b-o", label="Train Loss", linewidth=2)
    axes[1].plot(epochs, all_val_loss, "r--s", label="Val Loss", linewidth=2)
    axes[1].axvline(x=s1_len + 0.5, color="gray", linestyle=":", label="Stage 2 Transition")
    axes[1].set_title("4-Class ABO-Positive: Loss Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Training history curve saved to: {plot_path}")


def evaluate_split(model, X, y, paths, cnames, split_name="validation"):
    """
    Evaluates model on given split and generates comprehensive metrics and predictions.
    """
    print(f"\n[*] Evaluating on {split_name} ({len(y)} samples)...")
    probs = model.predict(X, batch_size=32, verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    acc = accuracy_score(y, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y, preds, average="weighted", zero_division=0)

    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(y, preds, average=None, zero_division=0)
    cm = confusion_matrix(y, preds, labels=list(range(NUM_CLASSES)))

    per_class_dict = {}
    for i in range(NUM_CLASSES):
        cls_name = ID_TO_CLASS[i]
        mask = (y == i)
        cls_confs = confs[mask]
        cls_corr_confs = confs[mask & (y == preds)]
        per_class_dict[cls_name] = {
            "class_id": i,
            "class_name": cls_name,
            "support": int(s_cls[i]),
            "correct": int(cm[i, i]),
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "mean_confidence": round(float(np.mean(cls_confs)), 4) if len(cls_confs) > 0 else 0.0,
            "mean_confidence_correct": round(float(np.mean(cls_corr_confs)), 4) if len(cls_corr_confs) > 0 else 0.0
        }

    hardest_class = min(per_class_dict.keys(), key=lambda c: per_class_dict[c]["f1"])

    correct_mask = (y == preds)
    conf_analysis = {
        "overall_mean_confidence": round(float(np.mean(confs)), 4),
        "overall_median_confidence": round(float(np.median(confs)), 4),
        "overall_min_confidence": round(float(np.min(confs)), 4),
        "overall_max_confidence": round(float(np.max(confs)), 4),
        "mean_confidence_correct": round(float(np.mean(confs[correct_mask])), 4) if np.sum(correct_mask) > 0 else 0.0,
        "mean_confidence_incorrect": round(float(np.mean(confs[~correct_mask])), 4) if np.sum(~correct_mask) > 0 else 0.0,
        "high_confidence_errors_ge_80": int(np.sum((~correct_mask) & (confs >= 0.80))),
        "high_confidence_errors_ge_90": int(np.sum((~correct_mask) & (confs >= 0.90)))
    }

    metrics = {
        "split": split_name,
        "sample_count": len(y),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "hardest_class": hardest_class,
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "confidence_analysis": conf_analysis
    }

    pred_records = []
    for i in range(len(y)):
        pred_records.append({
            "image_path": paths[i],
            "filename": os.path.basename(paths[i]),
            "ground_truth_id": int(y[i]),
            "ground_truth_class": cnames[i],
            "predicted_id": int(preds[i]),
            "predicted_class": ID_TO_CLASS[preds[i]],
            "confidence": round(float(confs[i]), 4),
            "is_correct": bool(y[i] == preds[i]),
            "prob_A+": round(float(probs[i, 0]), 4),
            "prob_AB+": round(float(probs[i, 1]), 4),
            "prob_B+": round(float(probs[i, 2]), 4),
            "prob_O+": round(float(probs[i, 3]), 4),
        })
    pred_df = pd.DataFrame(pred_records)

    plot_confusion_matrix(cm, split_name)

    return metrics, pred_df, cm


def plot_confusion_matrix(cm, split_name):
    """Plots and saves confusion matrix."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=FOUR_CLASSES, yticklabels=FOUR_CLASSES,
        cbar=False, annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title(f"4-Class ABO-Positive Confusion Matrix ({split_name.capitalize()})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()

    out_file = "confusion_matrix_val.png" if "val" in split_name else "confusion_matrix_test.png"
    cm_path = os.path.join(OUTPUTS_DIR, out_file)
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")


def save_metadata_and_configs(best_model_path):
    """Saves class mapping, training config, and model metadata."""
    model_sha256 = compute_file_sha256(best_model_path)

    class_mapping_data = {
        "class_to_id": CLASS_TO_ID,
        "id_to_class": {str(k): v for k, v in ID_TO_CLASS.items()},
        "classes": FOUR_CLASSES,
        "num_classes": NUM_CLASSES,
        "scientific_framing": "4-class ABO-positive fingerprint classification prototype",
        "disclaimer": "AI-assisted research prototype investigating statistical associations between fingerprint images and ABO blood-group labels. Not a clinically validated blood-group test."
    }
    with open(os.path.join(MODELS_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_mapping_data, f, indent=2)

    training_config = {
        "backbone": "EfficientNetB0 (ImageNet pretrained)",
        "input_shape": list(INPUT_SHAPE),
        "preprocessing_convention": "Standard RGB float32 in [0, 255]; built-in EfficientNetB0 rescaling and normalization",
        "num_classes": NUM_CLASSES,
        "classes": FOUR_CLASSES,
        "optimizer": "Adam",
        "stage1_learning_rate": 1e-3,
        "stage2_learning_rate": 3e-5,
        "batch_size": 32,
        "loss": "sparse_categorical_crossentropy",
        "class_weights": "balanced computed solely from train split",
        "augmentation": "Conservative rotation (0.03), translation (0.03), zoom (0.03), contrast (0.05). No flips.",
        "early_stopping_monitor": "val_loss",
        "two_stage_training": True
    }
    with open(os.path.join(MODELS_DIR, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    model_metadata = {
        "model_name": "ABO_Positive_4Class_EfficientNetB0",
        "model_filename": "best_model.keras",
        "model_sha256": model_sha256,
        "created_at": datetime.now().isoformat(),
        "framework": f"TensorFlow {tf.__version__}",
        "architecture_summary": "EfficientNetB0 -> GAP -> Dense(256, ReLU, L2 1e-4) -> Dropout(0.30) -> Dense(4, Softmax)",
        "scientific_framing": "AI-assisted research prototype investigating statistical associations between fingerprint images and ABO blood-group labels.",
        "clinical_disclaimer": "This is NOT a clinically validated blood-group test. Fingerprints do not biologically determine blood group."
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(model_metadata, f, indent=2)

    print(f"[+] Saved model metadata, class mapping, and training config to {MODELS_DIR}")
    return model_sha256


def main():
    print("=" * 80)
    print("4-CLASS ABO-POSITIVE FINGERPRINT CLASSIFICATION EXPERIMENT")
    print("AI-Assisted Research Prototype (Non-Clinical)")
    print("=" * 80)

    # 1. Load and verify dataset
    df = load_and_verify_dataset()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[*] Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 2. Extract split arrays
    print("[*] Loading Train data into memory...")
    X_train, y_train, paths_train, cnames_train = load_split_arrays(train_df)
    print("[*] Loading Validation data into memory...")
    X_val, y_val, paths_val, cnames_val = load_split_arrays(val_df)
    print("[*] Loading Test data into memory (for baseline and locked test evaluation)...")
    X_test, y_test, paths_test, cnames_test = load_split_arrays(test_df)

    # 3. Calculate and save baselines
    baseline_metrics = compute_and_save_baselines(y_train, y_val, y_test)

    # 4. Train model (Two-stage EfficientNetB0)
    best_model_path, history = train_model(X_train, y_train, X_val, y_val)

    # 5. Save model metadata
    model_sha256 = save_metadata_and_configs(best_model_path)

    # 6. Evaluate on Validation (Primary Model Selection)
    print("\n[*] Loading best checkpoint for validation evaluation:", best_model_path)
    best_model = tf.keras.models.load_model(best_model_path, compile=False)

    val_metrics, val_preds_df, val_cm = evaluate_split(
        best_model, X_val, y_val, paths_val, cnames_val, split_name="validation"
    )

    # Save validation metrics
    val_metrics_path = os.path.join(OUTPUTS_DIR, "val_metrics.json")
    with open(val_metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    val_preds_path = os.path.join(OUTPUTS_DIR, "val_predictions.csv")
    val_preds_df.to_csv(val_preds_path, index=False)

    val_report_str = classification_report(
        y_val, val_preds_df["predicted_id"].to_numpy(),
        target_names=FOUR_CLASSES, zero_division=0
    )
    val_report_path = os.path.join(OUTPUTS_DIR, "classification_report_val.txt")
    with open(val_report_path, "w") as f:
        f.write("4-Class ABO-Positive Fingerprint Classification Prototype\n")
        f.write("Validation Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(val_report_str)

    print(f"[+] Validation metrics saved to: {val_metrics_path}")
    print(f"    - Validation Accuracy:        {val_metrics['accuracy']:.2%}")
    print(f"    - Validation Macro Precision: {val_metrics['macro_precision']:.4f}")
    print(f"    - Validation Macro Recall:    {val_metrics['macro_recall']:.4f}")
    print(f"    - Validation Macro F1:        {val_metrics['macro_f1']:.4f}")
    print(f"    - Hardest Class:              {val_metrics['hardest_class']}")

    print("\n[+] Training phase complete. Proceeding to evaluate_abo_positive_4class.py for locked test evaluation.")


if __name__ == "__main__":
    main()
