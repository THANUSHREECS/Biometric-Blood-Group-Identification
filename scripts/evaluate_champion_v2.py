#!/usr/bin/env python3
"""
scripts/evaluate_champion_v2.py
Official Final Locked Test Evaluation: Champion Ensemble V2.

Evaluates the frozen Champion Ensemble V2 on the untouched 898-image test set:
  - ResNet50 (Improved, 1.2x Contrast): 0.58
  - VGG16 (Native Preprocessing):       0.38
  - EfficientNetB0 (Native Preprocessing): 0.04

Compares against:
  - Baseline ResNet50: 57.57% (517 / 898)
  - Previous Official Ensemble V1: 85.41% (767 / 898)

Outputs:
  - outputs/FINAL_MODEL_EVALUATION.json
  - outputs/FINAL_CONFUSION_MATRIX.png
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

# Official Baselines
BASELINE_ORIGINAL = {
    "name": "Original ResNet50 (8 epochs)",
    "accuracy": 0.5757,
    "correct": 517,
    "macro_f1": 0.5691,
    "macro_precision": 0.6299,
    "macro_recall": 0.6116,
    "test_loss": 1.1395
}

BASELINE_ENSEMBLE_V1 = {
    "name": "Previous Official Ensemble V1 (0.50 ResNet / 0.35 VGG / 0.15 EffNet)",
    "accuracy": 0.8541,
    "correct": 767,
    "macro_f1": 0.8591,
    "macro_precision": 0.8648,
    "macro_recall": 0.8602,
    "test_loss": 0.4527
}


def load_test_split():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test samples found in manifest!")
    print(f"[+] Loaded {len(test_df)} test samples (untouched locked test split)")
    return test_df["image_path"].tolist(), np.array(test_df["class_id"].astype(int).tolist())


def get_model_test_probs(model_path, prep_fn, test_paths, enhance_contrast=False, batch_size=32):
    print(f"[*] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    all_probs = []
    n = len(test_paths)
    for i in range(0, n, batch_size):
        b_paths = test_paths[i:i+batch_size]
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
    return np.vstack(all_probs), int(model.count_params())


def main():
    print("=" * 75)
    print("OFFICIAL LOCKED TEST EVALUATION: CHAMPION ENSEMBLE V2")
    print("=" * 75)
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("Strict Protocol: Single execution on locked 898 test images.")
    print("=" * 75)

    test_paths, test_labels = load_test_split()
    total_test = len(test_paths)

    # 1. Compute probabilities for each active model
    print("\n[*] Running test predictions for ResNet50 (Improved, 1.2x Contrast)...")
    p_resnet, params_res = get_model_test_probs(
        os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
        tf.keras.applications.resnet50.preprocess_input,
        test_paths, enhance_contrast=True
    )

    print("[*] Running test predictions for VGG16 (Native Preprocessing)...")
    p_vgg, params_vgg = get_model_test_probs(
        os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
        tf.keras.applications.vgg16.preprocess_input,
        test_paths, enhance_contrast=False
    )

    print("[*] Running test predictions for EfficientNetB0 (Native Preprocessing)...")
    p_eff, params_eff = get_model_test_probs(
        os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
        tf.keras.applications.efficientnet.preprocess_input,
        test_paths, enhance_contrast=False
    )

    # 2. Frozen Champion Ensemble V2 Weights
    w_res = 0.58
    w_vgg = 0.38
    w_eff = 0.04

    weighted_probs = w_res * p_resnet + w_vgg * p_vgg + w_eff * p_eff
    pred_labels = np.argmax(weighted_probs, axis=1)

    # 3. Compute Metrics
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(scce(test_labels, weighted_probs).numpy())

    test_acc = accuracy_score(test_labels, pred_labels)
    correct = int(np.sum(pred_labels == test_labels))
    incorrect = total_test - correct

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, pred_labels, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        test_labels, pred_labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(test_labels, pred_labels, labels=list(range(NUM_CLASSES)))
    clf_report = classification_report(
        test_labels, pred_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class = {}
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES[cid]
        c_total = int(np.sum(test_labels == cid))
        c_corr = int(cm[cid, cid])
        cr = clf_report[cname]
        per_class[cname] = {
            "class_id": cid,
            "support": c_total,
            "correct": c_corr,
            "accuracy": round(c_corr / c_total, 4) if c_total > 0 else 0.0,
            "precision": round(float(cr["precision"]), 4),
            "recall": round(float(cr["recall"]), 4),
            "f1_score": round(float(cr["f1-score"]), 4)
        }

    # 4. Print Results
    print("\n" + "=" * 75)
    print("CHAMPION ENSEMBLE V2 — FINAL TEST RESULTS")
    print("=" * 75)
    print(f"Total Test Samples     : {total_test}")
    print(f"Correct Predictions    : {correct} ({test_acc*100:.2f}%)")
    print(f"Incorrect Predictions  : {incorrect} ({(1-test_acc)*100:.2f}%)")
    print(f"Test Loss (SCCE)       : {test_loss:.4f}")
    print(f"Test Accuracy          : {test_acc*100:.2f}%")
    print(f"Macro Precision        : {prec_macro:.4f}")
    print(f"Macro Recall           : {rec_macro:.4f}")
    print(f"Macro F1-Score         : {f1_macro:.4f}")
    print(f"Weighted F1-Score      : {f1_wt:.4f}")

    print("\nPer-Class Performance on Locked Test Split:")
    print(f"{'Class':<8} | {'Support':>8} | {'Correct':>8} | {'Accuracy':>9} | {'Precision':>10} | {'Recall':>9} | {'F1':>9}")
    print("-" * 75)
    for cname in CLASS_NAMES:
        m = per_class[cname]
        print(
            f"{cname:<8} | {m['support']:>8} | {m['correct']:>8} | "
            f"{m['accuracy']*100:>8.2f}% | {m['precision']:>10.4f} | "
            f"{m['recall']:>9.4f} | {m['f1_score']:>9.4f}"
        )

    # 5. Three-Way Comparison Table
    print("\n" + "=" * 75)
    print("THREE-WAY COMPARISON (ORIGINAL BASELINE vs OFFICIAL V1 vs CHAMPION V2)")
    print("=" * 75)
    print(f"{'Metric':<22} | {'Original Baseline':>18} | {'Official V1':>13} | {'Champion V2':>13}")
    print("-" * 75)
    print(f"{'Test Accuracy':<22} | {BASELINE_ORIGINAL['accuracy']*100:>17.2f}% | {BASELINE_ENSEMBLE_V1['accuracy']*100:>12.2f}% | {test_acc*100:>12.2f}%")
    print(f"{'Correct / 898':<22} | {BASELINE_ORIGINAL['correct']:>18} | {BASELINE_ENSEMBLE_V1['correct']:>13} | {correct:>13}")
    print(f"{'Macro F1':<22} | {BASELINE_ORIGINAL['macro_f1']:>18.4f} | {BASELINE_ENSEMBLE_V1['macro_f1']:>13.4f} | {f1_macro:>13.4f}")
    print(f"{'Macro Recall':<22} | {BASELINE_ORIGINAL['macro_recall']:>18.4f} | {BASELINE_ENSEMBLE_V1['macro_recall']:>13.4f} | {rec_macro:>13.4f}")
    print(f"{'Macro Precision':<22} | {BASELINE_ORIGINAL['macro_precision']:>18.4f} | {BASELINE_ENSEMBLE_V1['macro_precision']:>13.4f} | {prec_macro:>13.4f}")
    print(f"{'Test Loss':<22} | {BASELINE_ORIGINAL['test_loss']:>18.4f} | {BASELINE_ENSEMBLE_V1['test_loss']:>13.4f} | {test_loss:>13.4f}")

    diff_v1 = test_acc - BASELINE_ENSEMBLE_V1["accuracy"]
    diff_orig = test_acc - BASELINE_ORIGINAL["accuracy"]

    print("\n" + "-" * 75)
    if diff_v1 > 0:
        print(f"[+] Champion V2 IMPROVED over Official V1 by +{diff_v1*100:.2f}% test accuracy (+{correct - BASELINE_ENSEMBLE_V1['correct']} samples)!")
    elif diff_v1 == 0:
        print(f"[*] Champion V2 MATCHED Official V1 exactly at {test_acc*100:.2f}% test accuracy.")
    else:
        print(f"[-] Champion V2 test accuracy is {diff_v1*100:.2f}% compared to Official V1 (85.41%).")
    print(f"[+] Total gain over Original Baseline: +{diff_orig*100:.2f}% test accuracy (+{correct - BASELINE_ORIGINAL['correct']} samples).")
    print("-" * 75)

    # 6. Save Confusion Matrix
    cm_path = os.path.join(OUTPUTS_DIR, "FINAL_CONFUSION_MATRIX.png")
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, linewidths=0.5
    )
    plt.title(f"CHAMPION ENSEMBLE V2 — FINAL TEST CONFUSION MATRIX\nTest Accuracy: {test_acc*100:.2f}% | Macro F1: {f1_macro:.4f} (898 samples)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Blood Group", fontsize=11, fontweight="bold")
    plt.ylabel("True Blood Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved final confusion matrix: {cm_path}")

    # 7. Save Final JSON
    results_json_path = os.path.join(OUTPUTS_DIR, "FINAL_MODEL_EVALUATION.json")
    results_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_identity": {
            "name": "Champion Ensemble V2",
            "type": "soft_voting_ensemble",
            "weights": {
                "resnet50_contrast_enhanced": w_res,
                "vgg16_native": w_vgg,
                "efficientnetb0_native": w_eff
            },
            "models": {
                "resnet50": {
                    "path": os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
                    "preprocessing": "alpha_white + resize_224 + 1.2x_contrast + resnet50.preprocess_input",
                    "weight": w_res,
                    "parameters": params_res
                },
                "vgg16": {
                    "path": os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
                    "preprocessing": "alpha_white + resize_224 + vgg16.preprocess_input",
                    "weight": w_vgg,
                    "parameters": params_vgg
                },
                "efficientnetb0": {
                    "path": os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
                    "preprocessing": "alpha_white + resize_224 + efficientnet.preprocess_input",
                    "weight": w_eff,
                    "parameters": params_eff
                }
            }
        },
        "validation_benchmark": {
            "validation_accuracy": 0.8641,
            "validation_macro_f1": 0.8663
        },
        "test_evaluation": {
            "total_samples": total_test,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "test_accuracy": round(float(test_acc), 4),
            "test_loss": round(float(test_loss), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(prec_wt), 4),
            "weighted_recall": round(float(rec_wt), 4),
            "weighted_f1": round(float(f1_wt), 4),
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist()
        },
        "comparisons": {
            "original_resnet50_baseline": {
                "accuracy": BASELINE_ORIGINAL["accuracy"],
                "correct": BASELINE_ORIGINAL["correct"],
                "gain_accuracy": round(float(diff_orig), 4),
                "gain_correct": correct - BASELINE_ORIGINAL["correct"]
            },
            "previous_official_ensemble_v1": {
                "accuracy": BASELINE_ENSEMBLE_V1["accuracy"],
                "correct": BASELINE_ENSEMBLE_V1["correct"],
                "gain_accuracy": round(float(diff_v1), 4),
                "gain_correct": correct - BASELINE_ENSEMBLE_V1["correct"]
            }
        }
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"[+] Saved final evaluation results to: {results_json_path}")


if __name__ == "__main__":
    main()
