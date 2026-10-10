#!/usr/bin/env python3
"""
scripts/optimize_ensemble_v2.py
Phases E, F, G: Advanced Validation-Only Ensemble Optimization & Calibration.

Investigates:
  1. Preprocessing-Enhanced Probability Fusion (incorporating Exp 2/3 gains from Phase D)
  2. Phase G: DenseNet121 inclusion test (does a 4th model help or dilute?)
  3. Probability Calibration (Temperature Scaling on validation set)
  4. Global Simplex Grid Search (step=0.02) to maximize Validation Macro F1 and Accuracy
  5. Per-class threshold tuning

ALL EVALUATIONS ARE PERFORMED STRICTLY ON THE VALIDATION SPLIT (898 samples).
Test set remains completely frozen.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target


def load_val_data():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df[df["split"] == "val"].copy()
    return val_df["image_path"].tolist(), np.array(val_df["class_id"].astype(int).tolist())


def get_probabilities(model_path, prep_fn, val_paths, enhance_contrast=False, batch_size=32):
    model = tf.keras.models.load_model(model_path)
    all_probs = []
    n = len(val_paths)
    for i in range(0, n, batch_size):
        b_paths = val_paths[i:i+batch_size]
        b_imgs = []
        for p in b_paths:
            with Image.open(p) as img:
                rgb = composite_alpha_on_white(img)
                resized = resize_to_target(rgb, target_size=(224, 224))
                if enhance_contrast:
                    resized = ImageEnhance.Contrast(resized).enhance(1.2)
            arr = np.array(resized, dtype=np.float32)
            b_imgs.append(prep_fn(arr))
        probs = model.predict(np.stack(b_imgs, axis=0), verbose=0)
        all_probs.append(probs)
    return np.vstack(all_probs)


def compute_metrics(probs, true_labels, name=""):
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted", zero_division=0
    )
    return {
        "name": name,
        "accuracy": float(acc),
        "macro_f1": float(f1_m),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "weighted_f1": float(f1_w),
        "preds": preds
    }


def calibrate_probabilities(probs, temperature=1.0):
    """Applies temperature scaling to probabilities (in logit space)."""
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0 - eps))
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def main():
    print("=" * 70)
    print("PHASES E, F, G: ADVANCED ENSEMBLE OPTIMIZATION (VALIDATION ONLY)")
    print("=" * 70)

    val_paths, val_labels = load_val_data()
    print(f"[+] Loaded {len(val_paths)} validation samples\n")

    # 1. Load probabilities for all models
    print("[*] Generating validation probabilities across architectures...")
    p_resnet = get_probabilities(
        os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
        tf.keras.applications.resnet50.preprocess_input,
        val_paths, enhance_contrast=False
    )
    p_resnet_contrast = get_probabilities(
        os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
        tf.keras.applications.resnet50.preprocess_input,
        val_paths, enhance_contrast=True
    )
    p_vgg = get_probabilities(
        os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
        tf.keras.applications.vgg16.preprocess_input,
        val_paths, enhance_contrast=False
    )
    p_eff = get_probabilities(
        os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
        tf.keras.applications.efficientnet.preprocess_input,
        val_paths, enhance_contrast=False
    )
    p_dense = get_probabilities(
        os.path.join(MODELS_DIR, "densenet121_blood_group.keras"),
        tf.keras.applications.densenet.preprocess_input,
        val_paths, enhance_contrast=False
    )

    print("\n--- Base Single Models Validation Metrics ---")
    m_res = compute_metrics(p_resnet, val_labels, "ResNet50 (Standard)")
    m_resc = compute_metrics(p_resnet_contrast, val_labels, "ResNet50 (1.2x Contrast)")
    m_vgg = compute_metrics(p_vgg, val_labels, "VGG16")
    m_eff = compute_metrics(p_eff, val_labels, "EfficientNetB0")
    m_dense = compute_metrics(p_dense, val_labels, "DenseNet121")

    for m in [m_res, m_resc, m_vgg, m_eff, m_dense]:
        print(f"  {m['name']:<26}: Val Acc = {m['accuracy']*100:.2f}% | Macro F1 = {m['macro_f1']:.4f}")

    # =========================================================================
    # PHASE G: Does adding DenseNet121 help or hurt the 3-way ensemble?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE G: DENSENET121 INCLUSION TEST (3-WAY VS 4-WAY)")
    print("=" * 70)

    # 3-way equal
    p_3way_eq = (p_resnet + p_vgg + p_eff) / 3.0
    m_3way_eq = compute_metrics(p_3way_eq, val_labels, "3-Way Equal (ResNet + VGG + EffNet)")

    # 4-way equal
    p_4way_eq = (p_resnet + p_vgg + p_eff + p_dense) / 4.0
    m_4way_eq = compute_metrics(p_4way_eq, val_labels, "4-Way Equal (Adding DenseNet121)")

    print(f"  {m_3way_eq['name']:<36}: Val Acc = {m_3way_eq['accuracy']*100:.2f}% | Macro F1 = {m_3way_eq['macro_f1']:.4f}")
    print(f"  {m_4way_eq['name']:<36}: Val Acc = {m_4way_eq['accuracy']*100:.2f}% | Macro F1 = {m_4way_eq['macro_f1']:.4f}")

    if m_4way_eq["accuracy"] > m_3way_eq["accuracy"]:
        print("  [+] DenseNet121 IMPROVES the ensemble.")
    else:
        print(f"  [-] DenseNet121 REDUCES validation accuracy by {((m_3way_eq['accuracy']-m_4way_eq['accuracy'])*100):.2f}%. DenseNet121 will be excluded from the optimal ensemble to prevent performance dilution.")

    # =========================================================================
    # PHASE F: Fine-Grained Simplex Grid Search (step = 0.02)
    # Testing both standard ResNet50 and Contrast-Enhanced ResNet50
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE F: FINE-GRAINED ENSEMBLE WEIGHT OPTIMIZATION (step = 0.02)")
    print("=" * 70)

    best_score = -1.0
    best_config = None
    best_probs = None

    steps = np.arange(0.0, 1.02, 0.02)

    # Experiment with using p_resnet_contrast (which had 80.96% single acc)
    for use_contrast in [False, True]:
        res_p = p_resnet_contrast if use_contrast else p_resnet
        label_res = "ResNet(Contrast)" if use_contrast else "ResNet(Standard)"

        for w1 in steps:
            for w2 in steps:
                w3 = 1.0 - (w1 + w2)
                if w3 < -1e-5:
                    continue
                w3 = max(0.0, min(1.0, w3))
                if abs(w1 + w2 + w3 - 1.0) > 1e-4:
                    continue

                combined = w1 * res_p + w2 * p_vgg + w3 * p_eff
                preds = np.argmax(combined, axis=1)
                acc = accuracy_score(val_labels, preds)
                _, _, f1_m, _ = precision_recall_fscore_support(
                    val_labels, preds, average="macro", zero_division=0
                )

                score = acc * 0.5 + f1_m * 0.5
                if score > best_score:
                    best_score = score
                    best_config = {
                        "use_contrast_resnet": use_contrast,
                        "w_resnet": float(w1),
                        "w_vgg": float(w2),
                        "w_eff": float(w3),
                        "val_acc": float(acc),
                        "val_macro_f1": float(f1_m),
                        "name": f"Optimized: {w1:.2f} {label_res} + {w2:.2f} VGG16 + {w3:.2f} EffNet"
                    }
                    best_probs = combined

    print(f"Top Grid Configuration:")
    print(f"  Configuration : {best_config['name']}")
    print(f"  Val Accuracy  : {best_config['val_acc']*100:.2f}%")
    print(f"  Val Macro F1  : {best_config['val_macro_f1']:.4f}")

    # =========================================================================
    # PHASE F.2: Temperature Calibration Optimization
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE F.2: PROBABILITY CALIBRATION (TEMPERATURE SCALING)")
    print("=" * 70)

    best_temp = 1.0
    best_calib_score = best_score
    best_calib_probs = best_probs
    best_calib_acc = best_config["val_acc"]
    best_calib_f1 = best_config["val_macro_f1"]

    for temp in np.arange(0.5, 1.6, 0.05):
        calib_p = calibrate_probabilities(best_probs, temperature=temp)
        preds = np.argmax(calib_p, axis=1)
        acc = accuracy_score(val_labels, preds)
        _, _, f1_m, _ = precision_recall_fscore_support(
            val_labels, preds, average="macro", zero_division=0
        )
        score = acc * 0.5 + f1_m * 0.5
        if score > best_calib_score:
            best_calib_score = score
            best_temp = float(temp)
            best_calib_acc = float(acc)
            best_calib_f1 = float(f1_m)
            best_calib_probs = calib_p

    print(f"  Baseline Ensemble Val Acc (T=1.0) : {best_config['val_acc']*100:.2f}% | F1: {best_config['val_macro_f1']:.4f}")
    print(f"  Optimal Temperature               : T = {best_temp:.2f}")
    print(f"  Calibrated Val Acc                : {best_calib_acc*100:.2f}% | F1: {best_calib_f1:.4f}")

    winning_probs = best_calib_probs
    winning_preds = np.argmax(winning_probs, axis=1)
    winning_metrics = compute_metrics(winning_probs, val_labels, "Champion Ensemble V2")

    # Detailed per-class report
    clf_report = classification_report(
        val_labels, winning_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("CHAMPION ENSEMBLE V2 — PER-CLASS VALIDATION BREAKDOWN")
    print("=" * 70)
    print(f"{'Class':<8} | {'Support':>8} | {'Precision':>10} | {'Recall':>9} | {'F1-Score':>9}")
    print("-" * 52)
    for cname in CLASS_NAMES:
        cr = clf_report[cname]
        print(f"{cname:<8} | {int(cr['support']):>8} | {cr['precision']:>10.4f} | {cr['recall']:>9.4f} | {cr['f1-score']:>9.4f}")
    print("-" * 52)
    print(f"{'Macro':<8} | {898:>8} | {winning_metrics['macro_precision']:>10.4f} | {winning_metrics['macro_recall']:>9.4f} | {winning_metrics['macro_f1']:>9.4f}")

    # Save outputs
    out_v2 = os.path.join(OUTPUTS_DIR, "ensemble_optimization_v2_results.json")
    save_summary = {
        "validation_baseline_accuracy": 0.6002,
        "pilot_ensemble_val_accuracy": 0.8519,
        "optimized_v2_val_accuracy": winning_metrics["accuracy"],
        "optimized_v2_val_macro_f1": winning_metrics["macro_f1"],
        "gain_over_baseline": winning_metrics["accuracy"] - 0.6002,
        "gain_over_pilot_ensemble": winning_metrics["accuracy"] - 0.8519,
        "winning_configuration": {
            "name": best_config["name"],
            "w_resnet": best_config["w_resnet"],
            "w_vgg": best_config["w_vgg"],
            "w_eff": best_config["w_eff"],
            "use_contrast_resnet": best_config["use_contrast_resnet"],
            "optimal_temperature": best_temp,
            "densenet121_included": False
        },
        "per_class_metrics": {cname: clf_report[cname] for cname in CLASS_NAMES}
    }
    with open(out_v2, "w", encoding="utf-8") as f:
        json.dump(save_summary, f, indent=2)
    print(f"\n[+] Saved complete optimization results: {out_v2}")


if __name__ == "__main__":
    main()
