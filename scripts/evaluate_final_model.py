#!/usr/bin/env python3
"""
scripts/evaluate_final_model.py
FINAL TEST EVALUATION — runs ONLY ONCE on the locked test set after user approval.

This script must only be run AFTER:
  1. All model training is complete
  2. The final model has been selected based on VALIDATION performance
  3. The user has explicitly approved the final model selection

The test set is the 898 untouched images in the 'test' split of the manifest CSV.
It is never used for model selection, hyperparameter tuning, or architecture comparison.

Usage:
  python scripts/evaluate_final_model.py --model_path models/mobilenetv2_blood_group.keras --arch mobilenetv2
  python scripts/evaluate_final_model.py --model_path models/efficientnetb0_blood_group.keras --arch efficientnetb0

Outputs:
  outputs/final_evaluation_results.json
  outputs/final_confusion_matrix.png
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.config import (
    CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
)
from src.preprocessing import composite_alpha_on_white, resize_to_target

INPUT_SIZES = {
    "custom_cnn":     (96, 96),
    "mobilenetv2":    (224, 224),
    "efficientnetb0": (224, 224),
    "densenet121":    (224, 224),
    "resnet50":       (224, 224),
    "vgg16":          (224, 224),
    "vgg19":          (224, 224),
}

PREPROCESS_FUNCS = {
    "custom_cnn":     lambda x: (x / 127.5) - 1.0,
    "mobilenetv2":    tf.keras.applications.mobilenet_v2.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    "densenet121":    tf.keras.applications.densenet.preprocess_input,
    "resnet50":       tf.keras.applications.resnet50.preprocess_input,
    "vgg16":          tf.keras.applications.vgg16.preprocess_input,
    "vgg19":          tf.keras.applications.vgg19.preprocess_input,
}

# Baseline results for comparison
BASELINE = {
    "model": "ResNet50 (baseline, 8 epochs)",
    "test_accuracy": 0.5757,
    "test_loss": 1.1395,
    "macro_precision": 0.6299,
    "macro_recall": 0.6116,
    "macro_f1": 0.5691,
    "weighted_f1": 0.5473,
    "correct": 517,
    "incorrect": 381,
    "total": 898,
}


def load_test_split():
    """Load ONLY the test split from the manifest. NEVER loads train or val."""
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test records found in manifest!")
    print(f"[+] Loaded {len(test_df)} test samples (locked, untouched split)")
    return test_df["image_path"].tolist(), test_df["class_id"].astype(int).tolist()


def preprocess_image(path_str, arch_name):
    from PIL import Image
    target_size = INPUT_SIZES[arch_name]
    preprocess_fn = PREPROCESS_FUNCS[arch_name]
    with Image.open(path_str) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, target_size=target_size)
    arr = np.array(resized, dtype=np.float32)
    return preprocess_fn(arr)


def plot_confusion_matrix(cm, output_path, model_name):
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, linewidths=0.5
    )
    plt.title(f"FINAL MODEL: {model_name}\nConfusion Matrix — TEST SET (898 untouched samples)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Blood Group", fontsize=11, fontweight="bold")
    plt.ylabel("True Blood Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix: {output_path}")


def evaluate_final_model(model_path, arch_name, batch_size=32):
    print("=" * 70)
    print("FINAL TEST EVALUATION — LOCKED TEST SET (898 samples)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Architecture: {arch_name}")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print()
    print("IMPORTANT: This evaluation uses the untouched test split.")
    print("The test set was NEVER used for training, validation, or")
    print("model selection. This is a genuine, unbiased evaluation.")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    print(f"\n[*] Loading model...")
    model = tf.keras.models.load_model(model_path)
    print(f"[+] Model loaded: {model.count_params():,} parameters")

    # Load ONLY test split
    test_paths, test_labels = load_test_split()
    total_test = len(test_paths)

    # Run predictions
    print(f"\n[*] Running predictions on {total_test} test images...")
    all_probs = []
    for i in range(0, total_test, batch_size):
        batch_paths = test_paths[i:i+batch_size]
        batch_imgs = [preprocess_image(p, arch_name) for p in batch_paths]
        batch_tensor = np.stack(batch_imgs, axis=0)
        probs = model.predict(batch_tensor, verbose=0)
        all_probs.append(probs)
        if (i // batch_size) % 5 == 0:
            print(f"  Processed {min(i+batch_size, total_test)}/{total_test}...")

    all_probs = np.concatenate(all_probs, axis=0)
    pred_labels = np.argmax(all_probs, axis=1)
    true_labels = np.array(test_labels)

    # Compute test loss
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(scce(true_labels, all_probs).numpy())

    # Compute metrics
    test_acc = accuracy_score(true_labels, pred_labels)
    correct = int(np.sum(pred_labels == true_labels))
    incorrect = total_test - correct

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    clf_report = classification_report(
        true_labels, pred_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Per-class metrics
    per_class = {}
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES[cid]
        c_total = int(np.sum(true_labels == cid))
        c_correct = int(cm[cid, cid])
        per_class[cname] = {
            "class_id": cid,
            "total_samples": c_total,
            "correct_predictions": c_correct,
            "accuracy": round(c_correct / c_total, 4) if c_total > 0 else 0.0
        }

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Samples  : {total_test}")
    print(f"Correct Predictions : {correct} ({test_acc*100:.2f}%)")
    print(f"Incorrect           : {incorrect} ({(1-test_acc)*100:.2f}%)")
    print(f"Test Loss (SCCE)    : {test_loss:.4f}")
    print(f"Test Accuracy       : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Macro Precision     : {prec_macro:.4f}")
    print(f"Macro Recall        : {rec_macro:.4f}")
    print(f"Macro F1-Score      : {f1_macro:.4f}")
    print(f"Weighted F1-Score   : {f1_weighted:.4f}")

    print("\nPer-Class Breakdown:")
    print(f"{'Class':<8} | {'Support':>8} | {'Correct':>8} | {'Accuracy':>9} | "
          f"{'Precision':>10} | {'Recall':>9} | {'F1':>9}")
    print("-" * 78)
    for cname in CLASS_NAMES:
        m = per_class[cname]
        cr = clf_report[cname]
        print(
            f"{cname:<8} | {m['total_samples']:>8} | {m['correct_predictions']:>8} | "
            f"{m['accuracy']:>9.4f} | {cr['precision']:>10.4f} | "
            f"{cr['recall']:>9.4f} | {cr['f1-score']:>9.4f}"
        )

    # -----------------------------------------------------------------------
    # Comparison against baseline
    # -----------------------------------------------------------------------
    improvement = test_acc - BASELINE["test_accuracy"]
    print("\n" + "=" * 70)
    print("COMPARISON AGAINST BASELINE")
    print("=" * 70)
    print(f"{'Metric':<22} | {'Baseline ResNet50':>18} | {'Final Model':>12} | {'Improvement':>12}")
    print("-" * 72)
    print(f"{'Test Accuracy':<22} | {BASELINE['test_accuracy']*100:>17.2f}% | {test_acc*100:>11.2f}% | {improvement*100:>+11.2f}%")
    print(f"{'Macro F1':<22} | {BASELINE['macro_f1']:>18.4f} | {f1_macro:>12.4f} | {f1_macro-BASELINE['macro_f1']:>+12.4f}")
    print(f"{'Macro Precision':<22} | {BASELINE['macro_precision']:>18.4f} | {prec_macro:>12.4f} | {prec_macro-BASELINE['macro_precision']:>+12.4f}")
    print(f"{'Macro Recall':<22} | {BASELINE['macro_recall']:>18.4f} | {rec_macro:>12.4f} | {rec_macro-BASELINE['macro_recall']:>+12.4f}")
    print(f"{'Correct Predictions':<22} | {BASELINE['correct']:>18} | {correct:>12} | {correct-BASELINE['correct']:>+12}")

    if improvement > 0:
        print(f"\n[+] Improvement over baseline: +{improvement*100:.2f}% test accuracy")
    else:
        print(f"\n[!] No improvement over baseline: {improvement*100:.2f}% test accuracy")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    cm_path = os.path.join(OUTPUTS_DIR, "final_confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path, arch_name.upper())

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_model": {
            "arch": arch_name,
            "model_path": model_path,
            "parameters": model.count_params()
        },
        "test_evaluation": {
            "test_total": total_test,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(prec_weighted), 4),
            "weighted_recall": round(float(rec_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class_metrics": per_class,
            "classification_report": clf_report,
            "confusion_matrix": cm.tolist()
        },
        "baseline_comparison": {
            "baseline_model": BASELINE["model"],
            "baseline_test_accuracy": BASELINE["test_accuracy"],
            "baseline_macro_f1": BASELINE["macro_f1"],
            "improvement_accuracy": round(float(improvement), 4),
            "improvement_f1": round(float(f1_macro - BASELINE["macro_f1"]), 4)
        }
    }

    results_path = os.path.join(OUTPUTS_DIR, "final_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved final evaluation results: {results_path}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print(f"  Test Accuracy: {test_acc*100:.2f}% (vs baseline 57.57%)")
    print(f"  Macro F1:      {f1_macro*100:.2f}% (vs baseline 56.91%)")
    print("=" * 70)

    return results


def evaluate_final_ensemble(weights_json_path, batch_size=32):
    print("=" * 70)
    print("FINAL TEST EVALUATION — ENSEMBLE MODEL (898 untouched samples)")
    print("=" * 70)
    print(f"Weights Config: {weights_json_path}")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print()
    print("IMPORTANT: This evaluation uses the untouched test split.")
    print("The test set was NEVER used for training, validation, or")
    print("model selection. This is a genuine, unbiased evaluation.")
    print("=" * 70)

    if not os.path.exists(weights_json_path):
        raise FileNotFoundError(f"Ensemble weights file not found: {weights_json_path}")

    with open(weights_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    weights_dict = config["weights"]
    models_dict = config["models"]

    test_paths, test_labels = load_test_split()
    total_test = len(test_paths)
    true_labels = np.array(test_labels)

    weighted_probs = np.zeros((total_test, NUM_CLASSES), dtype=np.float32)
    active_models_info = {}

    for arch, weight in weights_dict.items():
        if weight <= 0.0:
            continue
        mpath = models_dict[arch]
        print(f"\n[*] Loading {arch} (weight={weight:.3f}) from {mpath}...")
        model = tf.keras.models.load_model(mpath)
        active_models_info[arch] = {
            "model_path": mpath,
            "weight": float(weight),
            "parameters": int(model.count_params())
        }

        arch_probs = []
        for i in range(0, total_test, batch_size):
            batch_paths = test_paths[i:i+batch_size]
            batch_imgs = [preprocess_image(p, arch) for p in batch_paths]
            batch_tensor = np.stack(batch_imgs, axis=0)
            probs = model.predict(batch_tensor, verbose=0)
            arch_probs.append(probs)

        arch_probs = np.concatenate(arch_probs, axis=0)
        weighted_probs += weight * arch_probs
        print(f"    Computed test probabilities for {arch}")

    pred_labels = np.argmax(weighted_probs, axis=1)

    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(scce(true_labels, weighted_probs).numpy())

    test_acc = accuracy_score(true_labels, pred_labels)
    correct = int(np.sum(pred_labels == true_labels))
    incorrect = total_test - correct

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    clf_report = classification_report(
        true_labels, pred_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class = {}
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES[cid]
        c_total = int(np.sum(true_labels == cid))
        c_correct = int(cm[cid, cid])
        per_class[cname] = {
            "class_id": cid,
            "total_samples": c_total,
            "correct_predictions": c_correct,
            "accuracy": round(c_correct / c_total, 4) if c_total > 0 else 0.0
        }

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS — ENSEMBLE")
    print("=" * 70)
    print(f"Total Test Samples  : {total_test}")
    print(f"Correct Predictions : {correct} ({test_acc*100:.2f}%)")
    print(f"Incorrect           : {incorrect} ({(1-test_acc)*100:.2f}%)")
    print(f"Test Loss (SCCE)    : {test_loss:.4f}")
    print(f"Test Accuracy       : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"Macro Precision     : {prec_macro:.4f}")
    print(f"Macro Recall        : {rec_macro:.4f}")
    print(f"Macro F1-Score      : {f1_macro:.4f}")
    print(f"Weighted F1-Score   : {f1_weighted:.4f}")

    print("\nPer-Class Breakdown:")
    print(f"{'Class':<8} | {'Support':>8} | {'Correct':>8} | {'Accuracy':>9} | "
          f"{'Precision':>10} | {'Recall':>9} | {'F1':>9}")
    print("-" * 78)
    for cname in CLASS_NAMES:
        m = per_class[cname]
        cr = clf_report[cname]
        print(
            f"{cname:<8} | {m['total_samples']:>8} | {m['correct_predictions']:>8} | "
            f"{m['accuracy']:>9.4f} | {cr['precision']:>10.4f} | "
            f"{cr['recall']:>9.4f} | {cr['f1-score']:>9.4f}"
        )

    improvement = test_acc - BASELINE["test_accuracy"]
    print("\n" + "=" * 70)
    print("COMPARISON AGAINST BASELINE")
    print("=" * 70)
    print(f"{'Metric':<22} | {'Baseline ResNet50':>18} | {'Final Ensemble':>14} | {'Improvement':>12}")
    print("-" * 74)
    print(f"{'Test Accuracy':<22} | {BASELINE['test_accuracy']*100:>17.2f}% | {test_acc*100:>13.2f}% | {improvement*100:>+11.2f}%")
    print(f"{'Macro F1':<22} | {BASELINE['macro_f1']:>18.4f} | {f1_macro:>14.4f} | {f1_macro-BASELINE['macro_f1']:>+12.4f}")
    print(f"{'Macro Precision':<22} | {BASELINE['macro_precision']:>18.4f} | {prec_macro:>14.4f} | {prec_macro-BASELINE['macro_precision']:>+12.4f}")
    print(f"{'Macro Recall':<22} | {BASELINE['macro_recall']:>18.4f} | {rec_macro:>14.4f} | {rec_macro-BASELINE['macro_recall']:>+12.4f}")
    print(f"{'Correct Predictions':<22} | {BASELINE['correct']:>18} | {correct:>14} | {correct-BASELINE['correct']:>+12}")

    if improvement > 0:
        print(f"\n[+] Improvement over baseline: +{improvement*100:.2f}% test accuracy")
    else:
        print(f"\n[!] No improvement over baseline: {improvement*100:.2f}% test accuracy")

    cm_path = os.path.join(OUTPUTS_DIR, "final_confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path, "Soft-Voting Ensemble (ResNet50 + VGG16 + EfficientNetB0)")

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_model": {
            "model_type": "soft_voting_ensemble",
            "weights": weights_dict,
            "active_models": active_models_info
        },
        "test_evaluation": {
            "test_total": total_test,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(prec_weighted), 4),
            "weighted_recall": round(float(rec_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class_metrics": per_class,
            "classification_report": clf_report,
            "confusion_matrix": cm.tolist()
        },
        "baseline_comparison": {
            "baseline_model": BASELINE["model"],
            "baseline_test_accuracy": BASELINE["test_accuracy"],
            "baseline_macro_f1": BASELINE["macro_f1"],
            "improvement_accuracy": round(float(improvement), 4),
            "improvement_f1": round(float(f1_macro - BASELINE["macro_f1"]), 4)
        }
    }

    results_path = os.path.join(OUTPUTS_DIR, "final_evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved final evaluation results: {results_path}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print(f"  Test Accuracy: {test_acc*100:.2f}% (vs baseline 57.57%)")
    print(f"  Macro F1:      {f1_macro*100:.2f}% (vs baseline 56.91%)")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Final test evaluation for the selected blood group model or ensemble"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to single .keras model file"
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=list(INPUT_SIZES.keys()),
        help="Architecture name for single model"
    )
    parser.add_argument(
        "--ensemble", type=str, default=None,
        help="Path to ensemble_weights.json configuration file"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.ensemble:
        evaluate_final_ensemble(args.ensemble, args.batch_size)
    elif args.model_path and args.arch:
        evaluate_final_model(args.model_path, args.arch, args.batch_size)
    else:
        parser.error("Must provide either --ensemble <path> OR both --model_path and --arch")


if __name__ == "__main__":
    main()

