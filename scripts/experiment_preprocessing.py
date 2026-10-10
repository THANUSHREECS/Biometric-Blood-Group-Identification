#!/usr/bin/env python3
"""
scripts/experiment_preprocessing.py
Phase D: Controlled Fingerprint Preprocessing Experiments (Validation Split Only)

Experiments:
  Exp 1: Baseline Preprocessing (Composite on white -> Resize -> Model Preprocess)
  Exp 2: Mild Contrast Enhancement (PIL ImageEnhance.Contrast factor=1.2)
  Exp 3: Carefully Tuned CLAHE (OpenCV CLAHE clipLimit=2.0, tileGridSize=(8,8) on L-channel)
  Exp 4: Mild Sharpening (UnsharpMask radius=1.0, percent=120, threshold=3)

Evaluates on the exact same validation split (898 images).
Does NOT touch the test set.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import CLASS_NAMES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target


def preprocess_exp1_baseline(pil_img, target_size=(224, 224)):
    """Exp 1: Baseline composite and resize."""
    rgb = composite_alpha_on_white(pil_img)
    return resize_to_target(rgb, target_size=target_size)


def preprocess_exp2_contrast(pil_img, target_size=(224, 224), factor=1.2):
    """Exp 2: Mild contrast enhancement."""
    rgb = composite_alpha_on_white(pil_img)
    resized = resize_to_target(rgb, target_size=target_size)
    enhancer = ImageEnhance.Contrast(resized)
    return enhancer.enhance(factor)


def preprocess_exp3_clahe(pil_img, target_size=(224, 224), clip_limit=2.0, grid_size=(8, 8)):
    """Exp 3: CLAHE on L-channel in LAB space to preserve color/channel balance."""
    rgb = composite_alpha_on_white(pil_img)
    resized = resize_to_target(rgb, target_size=target_size)
    arr = np.array(resized)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def preprocess_exp4_sharpening(pil_img, target_size=(224, 224)):
    """Exp 4: Mild unsharp masking."""
    rgb = composite_alpha_on_white(pil_img)
    resized = resize_to_target(rgb, target_size=target_size)
    return resized.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))


PREPROCESS_EXPERIMENTS = {
    "Exp 1: Baseline": preprocess_exp1_baseline,
    "Exp 2: Mild Contrast (1.2x)": preprocess_exp2_contrast,
    "Exp 3: Tuned CLAHE (clip=2.0)": preprocess_exp3_clahe,
    "Exp 4: Mild Sharpening (Unsharp)": preprocess_exp4_sharpening,
}


def load_val_data():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df[df["split"] == "val"].copy()
    return val_df["image_path"].tolist(), np.array(val_df["class_id"].astype(int).tolist())


def evaluate_pipeline_on_model(model_name, model_path, val_paths, val_labels, batch_size=32):
    print(f"\n=======================================================")
    print(f"Evaluating Preprocessing Variations for {model_name.upper()}")
    print(f"=======================================================")

    model = tf.keras.models.load_model(model_path)
    if "resnet" in model_name:
        native_prep = tf.keras.applications.resnet50.preprocess_input
    elif "vgg" in model_name:
        native_prep = tf.keras.applications.vgg16.preprocess_input
    elif "eff" in model_name:
        native_prep = tf.keras.applications.efficientnet.preprocess_input
    else:
        native_prep = lambda x: (x / 127.5) - 1.0

    exp_results = {}

    for exp_name, prep_fn in PREPROCESS_EXPERIMENTS.items():
        all_probs = []
        n = len(val_paths)
        for i in range(0, n, batch_size):
            b_paths = val_paths[i:i+batch_size]
            b_imgs = []
            for p in b_paths:
                with Image.open(p) as img:
                    proc_pil = prep_fn(img, target_size=(224, 224))
                arr = np.array(proc_pil, dtype=np.float32)
                b_imgs.append(native_prep(arr))
            probs = model.predict(np.stack(b_imgs, axis=0), verbose=0)
            all_probs.append(probs)

        probs = np.vstack(all_probs)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(val_labels, preds)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            val_labels, preds, average="macro", zero_division=0
        )

        exp_results[exp_name] = {
            "accuracy": float(acc),
            "macro_f1": float(f1_m),
            "macro_precision": float(p_m),
            "macro_recall": float(r_m),
            "probs": probs
        }
        print(f"  {exp_name:<32}: Val Acc = {acc*100:.2f}% | Macro F1 = {f1_m:.4f}")

    return exp_results


def main():
    print("=" * 70)
    print("PHASE D: FINGERPRINT PREPROCESSING EXPERIMENTS (VALIDATION ONLY)")
    print("=" * 70)

    val_paths, val_labels = load_val_data()
    print(f"[+] Loaded {len(val_paths)} validation samples\n")

    resnet_path = os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras")
    vgg_path = os.path.join(MODELS_DIR, "vgg16_blood_group.keras")

    res_results = evaluate_pipeline_on_model("resnet50", resnet_path, val_paths, val_labels)
    vgg_results = evaluate_pipeline_on_model("vgg16", vgg_path, val_paths, val_labels)

    # Summary table
    print("\n" + "=" * 70)
    print("PREPROCESSING BENCHMARK SUMMARY (VALIDATION SPLIT)")
    print("=" * 70)
    print(f"{'Preprocessing Experiment':<32} | {'ResNet50 Val Acc':>16} | {'VGG16 Val Acc':>14}")
    print("-" * 70)
    for exp_name in PREPROCESS_EXPERIMENTS.keys():
        r_acc = res_results[exp_name]["accuracy"] * 100
        v_acc = vgg_results[exp_name]["accuracy"] * 100
        print(f"{exp_name:<32} | {r_acc:>15.2f}% | {v_acc:>13.2f}%")

    out_json = os.path.join(OUTPUTS_DIR, "preprocessing_experiments_results.json")
    save_data = {
        "resnet50": {k: {"acc": v["accuracy"], "f1": v["macro_f1"]} for k, v in res_results.items()},
        "vgg16": {k: {"acc": v["accuracy"], "f1": v["macro_f1"]} for k, v in vgg_results.items()},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[+] Results saved to: {out_json}")


if __name__ == "__main__":
    main()
