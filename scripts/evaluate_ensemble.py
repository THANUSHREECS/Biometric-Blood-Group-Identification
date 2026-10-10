#!/usr/bin/env python3
"""
scripts/evaluate_ensemble.py
Validation-Only Multi-Model Soft-Voting Ensemble Optimization.

Evaluates combinations of top trained models on the VALIDATION split:
  1. ResNet50 (Improved)
  2. VGG16
  3. EfficientNetB0
  (and optionally DenseNet121)

Tests:
  - Equal-weighted voting
  - Ranked pairs (ResNet50 + VGG16)
  - Grid-searched optimal weights (maximizing Val Macro F1 / Val Acc)

NOTE:
  - Test set is NEVER touched here. Validation set only.
  - Generates outputs/ensemble_val_results.json and outputs/ensemble_weights.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target

PREPROCESS_FUNCS = {
    "resnet50":       tf.keras.applications.resnet50.preprocess_input,
    "vgg16":          tf.keras.applications.vgg16.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    "densenet121":    tf.keras.applications.densenet.preprocess_input,
}

INPUT_SIZES = {
    "resnet50":       (224, 224),
    "vgg16":          (224, 224),
    "efficientnetb0": (224, 224),
    "densenet121":    (224, 224),
}

MODEL_PATHS = {
    "resnet50":       os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
    "vgg16":          os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
    "efficientnetb0": os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
    "densenet121":    os.path.join(MODELS_DIR, "densenet121_blood_group.keras"),
}


def load_val_split():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df[df["split"] == "val"].copy()
    if len(val_df) == 0:
        raise ValueError("No val records found in manifest!")
    paths = val_df["image_path"].tolist()
    labels = val_df["class_id"].astype(int).tolist()
    print(f"[+] Loaded {len(paths)} validation samples")
    return paths, labels


def get_model_probabilities(arch_name, model_path, image_paths, batch_size=32):
    """
    Runs inference in batches and returns probability matrix of shape (N, NUM_CLASSES).
    """
    print(f"[*] Loading {arch_name} from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    target_size = INPUT_SIZES[arch_name]
    prep_fn = PREPROCESS_FUNCS[arch_name]

    n = len(image_paths)
    all_probs = []

    print(f"[*] Computing probabilities for {arch_name} ({n} images)...")
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_paths = image_paths[start_idx:end_idx]
        batch_arr = []
        for p in batch_paths:
            with Image.open(p) as img:
                rgb = composite_alpha_on_white(img)
                resized = resize_to_target(rgb, target_size=target_size)
            arr = np.array(resized, dtype=np.float32)
            batch_arr.append(prep_fn(arr))
        batch_tensor = np.stack(batch_arr, axis=0)
        probs = model.predict(batch_tensor, verbose=0)
        all_probs.append(probs)

    return np.vstack(all_probs)


def evaluate_probs(probs, true_labels, name=""):
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(true_labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted", zero_division=0
    )
    return {
        "name": name,
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "weighted_f1": float(f1_wt),
        "preds": preds
    }


def main():
    print("=" * 70)
    print("ENSEMBLE OPTIMIZATION & EVALUATION (VALIDATION SPLIT ONLY)")
    print("=" * 70)

    val_paths, val_labels = load_val_split()

    models_to_evaluate = ["resnet50", "vgg16", "efficientnetb0"]
    probs_dict = {}

    for arch in models_to_evaluate:
        mpath = MODEL_PATHS[arch]
        if not os.path.exists(mpath):
            print(f"[!] Warning: model file missing: {mpath}")
            continue
        probs = get_model_probabilities(arch, mpath, val_paths)
        probs_dict[arch] = probs

    # Evaluate individual models on val set
    print("\n--- Individual Model Validation Metrics ---")
    single_results = {}
    for arch, p in probs_dict.items():
        res = evaluate_probs(p, val_labels, name=f"Single: {arch}")
        single_results[arch] = res
        print(f"  {arch:<16}: Val Acc = {res['accuracy']:.4f} | Macro F1 = {res['macro_f1']:.4f}")

    p_res = probs_dict["resnet50"]
    p_vgg = probs_dict["vgg16"]
    p_eff = probs_dict["efficientnetb0"]

    ensemble_configs = []

    # 1. Equal-weighted 2-model: ResNet50 + VGG16
    p_res_vgg = 0.5 * p_res + 0.5 * p_vgg
    res_2m = evaluate_probs(p_res_vgg, val_labels, name="Ensemble: ResNet50 (0.5) + VGG16 (0.5)")
    ensemble_configs.append((res_2m, {"resnet50": 0.5, "vgg16": 0.5, "efficientnetb0": 0.0}))

    # 2. Equal-weighted 3-model: ResNet50 + VGG16 + EfficientNetB0
    p_3m_eq = (p_res + p_vgg + p_eff) / 3.0
    res_3m_eq = evaluate_probs(p_3m_eq, val_labels, name="Ensemble: Equal 3-way (0.33 / 0.33 / 0.33)")
    ensemble_configs.append((res_3m_eq, {"resnet50": 1/3, "vgg16": 1/3, "efficientnetb0": 1/3}))

    # 3. Accuracy-weighted: weights proportional to (val_acc - 0.5)^2
    # resnet ~ 0.30^2 = 0.09, vgg ~ 0.286^2 = 0.082, eff ~ 0.206^2 = 0.042
    w_res = (single_results["resnet50"]["accuracy"] - 0.5) ** 2
    w_vgg = (single_results["vgg16"]["accuracy"] - 0.5) ** 2
    w_eff = (single_results["efficientnetb0"]["accuracy"] - 0.5) ** 2
    w_sum = w_res + w_vgg + w_eff
    w_res, w_vgg, w_eff = w_res / w_sum, w_vgg / w_sum, w_eff / w_sum

    p_acc_wt = w_res * p_res + w_vgg * p_vgg + w_eff * p_eff
    res_acc_wt = evaluate_probs(
        p_acc_wt, val_labels,
        name=f"Ensemble: Acc-Weighted ({w_res:.2f} / {w_vgg:.2f} / {w_eff:.2f})"
    )
    ensemble_configs.append((res_acc_wt, {"resnet50": float(w_res), "vgg16": float(w_vgg), "efficientnetb0": float(w_eff)}))

    # 4. Grid search over 3 weights with step 0.05
    print("\n[*] Running grid search over weight simplex (step = 0.05)...")
    best_grid_acc = 0.0
    best_grid_f1 = 0.0
    best_grid_weights = None
    best_grid_res = None
    best_grid_probs = None

    steps = np.arange(0.0, 1.05, 0.05)
    grid_records = []

    for w1 in steps:
        for w2 in steps:
            w3 = 1.0 - (w1 + w2)
            if w3 < -1e-6:
                continue
            w3 = max(0.0, min(1.0, w3))
            if abs(w1 + w2 + w3 - 1.0) > 1e-4:
                continue

            combined = w1 * p_res + w2 * p_vgg + w3 * p_eff
            preds = np.argmax(combined, axis=1)
            acc = accuracy_score(val_labels, preds)
            _, _, f1_macro, _ = precision_recall_fscore_support(
                val_labels, preds, average="macro", zero_division=0
            )

            grid_records.append((w1, w2, w3, acc, f1_macro))

            # Prioritize macro F1, tie-break with accuracy
            score = f1_macro * 0.5 + acc * 0.5
            current_best_score = best_grid_f1 * 0.5 + best_grid_acc * 0.5
            if score > current_best_score:
                best_grid_acc = acc
                best_grid_f1 = f1_macro
                best_grid_weights = (float(w1), float(w2), float(w3))
                best_grid_probs = combined

    w1_opt, w2_opt, w3_opt = best_grid_weights
    res_grid = evaluate_probs(
        best_grid_probs, val_labels,
        name=f"Ensemble: Grid-Optimal ({w1_opt:.2f} ResNet / {w2_opt:.2f} VGG16 / {w3_opt:.2f} EffNet)"
    )
    ensemble_configs.append((res_grid, {
        "resnet50": w1_opt,
        "vgg16": w2_opt,
        "efficientnetb0": w3_opt
    }))

    print("\n" + "=" * 70)
    print("ENSEMBLE COMPARISON RESULTS (VALIDATION SET)")
    print("=" * 70)
    print(f"{'Configuration':<46} | {'Val Acc':>9} | {'Macro F1':>9}")
    print("-" * 70)

    # Print baseline
    print(f"{'Baseline ResNet50 (8 epochs, locked)':<46} | {'60.02%':>9} | {'0.5691':>9}")
    print(f"{'ResNet50 (Improved, Single)':<46} | {single_results['resnet50']['accuracy']*100:>8.2f}% | {single_results['resnet50']['macro_f1']:>9.4f}")
    print(f"{'VGG16 (Single)':<46} | {single_results['vgg16']['accuracy']*100:>8.2f}% | {single_results['vgg16']['macro_f1']:>9.4f}")
    print(f"{'EfficientNetB0 (Single)':<46} | {single_results['efficientnetb0']['accuracy']*100:>8.2f}% | {single_results['efficientnetb0']['macro_f1']:>9.4f}")
    print("-" * 70)

    best_ensemble_item = None
    best_ensemble_score = -1.0

    for item, w in ensemble_configs:
        print(f"{item['name']:<46} | {item['accuracy']*100:>8.2f}% | {item['macro_f1']:>9.4f}")
        score = item['accuracy'] + item['macro_f1']
        if score > best_ensemble_score:
            best_ensemble_score = score
            best_ensemble_item = (item, w)

    best_res, best_w = best_ensemble_item

    print("=" * 70)
    print(f"WINNING CONFIGURATION: {best_res['name']}")
    print(f"  Val Accuracy: {best_res['accuracy']*100:.2f}% (gain of +{best_res['accuracy']*100 - 60.02:.2f}% over baseline)")
    print(f"  Val Macro F1: {best_res['macro_f1']:.4f}")
    print("=" * 70)

    # Detailed per-class report for winning ensemble
    clf_report = classification_report(
        val_labels, best_res["preds"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\nWinning Ensemble Per-Class Metrics (Validation Set):")
    for cls in CLASS_NAMES:
        m = clf_report[cls]
        print(f"  {cls:5s}: Precision={m['precision']:.3f} | Recall={m['recall']:.3f} | F1={m['f1-score']:.3f} (n={int(m['support'])})")

    # Confusion matrix plot
    cm = confusion_matrix(val_labels, best_res["preds"])
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title(f"Ensemble Confusion Matrix (Validation Set)\nAccuracy: {best_res['accuracy']*100:.2f}% | Macro F1: {best_res['macro_f1']:.4f}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n[+] Saved confusion matrix to: {cm_path}")

    # Save weights & metadata
    ensemble_weights_path = os.path.join(OUTPUTS_DIR, "ensemble_weights.json")
    with open(ensemble_weights_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "soft_voting_ensemble",
            "val_accuracy": best_res["accuracy"],
            "val_macro_f1": best_res["macro_f1"],
            "weights": best_w,
            "models": {
                "resnet50": MODEL_PATHS["resnet50"],
                "vgg16": MODEL_PATHS["vgg16"],
                "efficientnetb0": MODEL_PATHS["efficientnetb0"]
            },
            "per_class_report": {cls: clf_report[cls] for cls in CLASS_NAMES}
        }, f, indent=2)
    print(f"[+] Saved ensemble weights configuration to: {ensemble_weights_path}")

    ensemble_results_path = os.path.join(OUTPUTS_DIR, "ensemble_val_results.json")
    with open(ensemble_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "single_models": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in single_results.items()},
            "winning_ensemble": {
                "name": best_res["name"],
                "weights": best_w,
                "accuracy": best_res["accuracy"],
                "macro_f1": best_res["macro_f1"]
            },
            "per_class_metrics": {cls: clf_report[cls] for cls in CLASS_NAMES}
        }, f, indent=2)
    print(f"[+] Saved ensemble validation results to: {ensemble_results_path}")


if __name__ == "__main__":
    main()
