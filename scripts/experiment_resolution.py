#!/usr/bin/env python3
"""
scripts/experiment_resolution.py
Phase C: Input Resolution Benchmark (Validation Split Only)

Compares:
  - 224x224 (Baseline)
  - 256x256 (Higher detail)
on the exact same validation split.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import CLASS_NAMES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target

RESOLUTIONS = [(224, 224), (256, 256)]


def load_val_data():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df[df["split"] == "val"].copy()
    return val_df["image_path"].tolist(), np.array(val_df["class_id"].astype(int).tolist())


def evaluate_resolution(model_name, model_path, native_prep, val_paths, val_labels, batch_size=32):
    print(f"\n=======================================================")
    print(f"Evaluating Resolutions for {model_name.upper()}")
    print(f"=======================================================")

    model = tf.keras.models.load_model(model_path)
    res_metrics = {}

    for res in RESOLUTIONS:
        all_probs = []
        n = len(val_paths)
        for i in range(0, n, batch_size):
            b_paths = val_paths[i:i+batch_size]
            b_imgs = []
            for p in b_paths:
                with Image.open(p) as img:
                    rgb = composite_alpha_on_white(img)
                    resized = resize_to_target(rgb, target_size=res)
                arr = np.array(resized, dtype=np.float32)
                b_imgs.append(native_prep(arr))
            probs = model.predict(np.stack(b_imgs, axis=0), verbose=0)
            all_probs.append(probs)

        probs = np.vstack(all_probs)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(val_labels, preds)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            val_labels, preds, average="macro", zero_division=0
        )
        res_str = f"{res[0]}x{res[1]}"
        res_metrics[res_str] = {
            "accuracy": float(acc),
            "macro_f1": float(f1_m),
            "probs": probs
        }
        print(f"  Resolution {res_str:<10}: Val Acc = {acc*100:.2f}% | Macro F1 = {f1_m:.4f}")

    return res_metrics


def main():
    print("=" * 70)
    print("PHASE C: RESOLUTION EXPERIMENTS (VALIDATION SPLIT ONLY)")
    print("=" * 70)

    val_paths, val_labels = load_val_data()
    resnet_path = os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras")
    vgg_path = os.path.join(MODELS_DIR, "vgg16_blood_group.keras")

    r_results = evaluate_resolution(
        "resnet50", resnet_path,
        tf.keras.applications.resnet50.preprocess_input,
        val_paths, val_labels
    )
    v_results = evaluate_resolution(
        "vgg16", vgg_path,
        tf.keras.applications.vgg16.preprocess_input,
        val_paths, val_labels
    )

    print("\n" + "=" * 70)
    print("RESOLUTION BENCHMARK SUMMARY (VALIDATION SPLIT)")
    print("=" * 70)
    print(f"{'Resolution':<12} | {'ResNet50 Val Acc':>16} | {'VGG16 Val Acc':>14}")
    print("-" * 70)
    for res in RESOLUTIONS:
        res_str = f"{res[0]}x{res[1]}"
        r_acc = r_results[res_str]["accuracy"] * 100
        v_acc = v_results[res_str]["accuracy"] * 100
        print(f"{res_str:<12} | {r_acc:>15.2f}% | {v_acc:>13.2f}%")

    out_json = os.path.join(OUTPUTS_DIR, "resolution_experiments_results.json")
    save_data = {
        "resnet50": {k: {"acc": v["accuracy"], "f1": v["macro_f1"]} for k, v in r_results.items()},
        "vgg16": {k: {"acc": v["accuracy"], "f1": v["macro_f1"]} for k, v in v_results.items()},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[+] Results saved to: {out_json}")


if __name__ == "__main__":
    main()
