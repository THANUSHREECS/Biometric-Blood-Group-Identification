#!/usr/bin/env python3
"""
scripts/evaluate_model.py
Production evaluation script for Biometric Blood Group Identification:
- Evaluates exclusively on the untouched 15% test split (898 samples)
- Uses the identical preprocessing pipeline from src/preprocessing.py
- Computes genuine, non-fabricated metrics:
    * Test Loss
    * Test Accuracy
    * Precision, Recall, F1-Score (Macro, Weighted, Per-Class)
    * Confusion Matrix & Per-Class Accuracy
    * Total Correct vs Incorrect Counts
- Generates outputs/confusion_matrix.png
- Generates outputs/evaluation_results.json
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    IMAGE_SIZE,
    INPUT_SHAPE,
    NUM_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    MODEL_PATH,
    DATASET_MANIFEST_PATH,
    OUTPUTS_DIR,
    EVALUATION_RESULTS_PATH
)
from src.preprocessing import load_and_preprocess_for_training

def load_test_split(manifest_path=DATASET_MANIFEST_PATH):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No records found with split == 'test'!")

    paths = test_df["image_path"].tolist()
    labels = test_df["class_id"].astype(int).tolist()
    class_names = test_df["class_name"].tolist()
    return paths, labels, class_names

def plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5
    )
    plt.title("Biometric Blood Group Identification - Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Blood Group", fontsize=11, fontweight="bold")
    plt.ylabel("True Blood Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to: {output_path}")

def evaluate_model(model_path=MODEL_PATH, manifest_path=DATASET_MANIFEST_PATH, batch_size=32):
    print("=" * 70)
    print("STEP 7: COMPREHENSIVE MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first!")

    print(f"[*] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[*] Loading test set from: {manifest_path}")
    test_paths, test_labels, test_class_names = load_test_split(manifest_path)
    total_test = len(test_paths)
    print(f"[+] Found {total_test} test images across {NUM_CLASSES} classes.")

    # Preprocess and evaluate batch by batch
    print("[*] Generating predictions on test set...")
    all_preds_probs = []
    
    # Process in batches
    for i in range(0, total_test, batch_size):
        batch_paths = test_paths[i:i+batch_size]
        batch_imgs = [load_and_preprocess_for_training(p, target_size=IMAGE_SIZE) for p in batch_paths]
        batch_tensor = np.stack(batch_imgs, axis=0)
        probs = model.predict(batch_tensor, verbose=0)
        all_preds_probs.append(probs)

    all_preds_probs = np.concatenate(all_preds_probs, axis=0)
    pred_labels = np.argmax(all_preds_probs, axis=1)
    true_labels = np.array(test_labels)

    # 1. Compute Test Loss
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = float(scce(true_labels, all_preds_probs).numpy())

    # 2. Compute Accuracy
    test_accuracy = float(accuracy_score(true_labels, pred_labels))
    correct_count = int(np.sum(pred_labels == true_labels))
    incorrect_count = int(total_test - correct_count)

    # 3. Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )

    # 4. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES)))

    # 5. Per-class accuracy
    per_class_metrics = {}
    for cid in range(NUM_CLASSES):
        cname = CLASS_NAMES[cid]
        c_total = int(np.sum(true_labels == cid))
        c_correct = int(cm[cid, cid])
        c_acc = float(c_correct / c_total) if c_total > 0 else 0.0
        
        per_class_metrics[cname] = {
            "class_id": cid,
            "total_samples": c_total,
            "correct_predictions": c_correct,
            "accuracy": round(c_acc, 4)
        }

    # 6. Classification Report
    clf_report = classification_report(
        true_labels,
        pred_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Print Summary Table
    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS (MEASURED & GENUINE)")
    print("=" * 70)
    print(f"Total Test Samples    : {total_test}")
    print(f"Correct Predictions   : {correct_count} ({test_accuracy * 100:.2f}%)")
    print(f"Incorrect Predictions : {incorrect_count} ({(1 - test_accuracy) * 100:.2f}%)")
    print(f"Test Loss (SCCE)      : {test_loss:.4f}")
    print(f"Test Accuracy         : {test_accuracy:.4f}")
    print(f"Macro Precision       : {precision_macro:.4f}")
    print(f"Macro Recall          : {recall_macro:.4f}")
    print(f"Macro F1-Score        : {f1_macro:.4f}")
    print(f"Weighted F1-Score     : {f1_weighted:.4f}")

    print("\nPer-Class Breakdown:")
    print(f"{'Class':<8} | {'Support':<9} | {'Correct':<9} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    for cname in CLASS_NAMES:
        m = per_class_metrics[cname]
        cr = clf_report[cname]
        print(f"{cname:<8} | {m['total_samples']:<9} | {m['correct_predictions']:<9} | {m['accuracy']:<10.4f} | {cr['precision']:<10.4f} | {cr['recall']:<10.4f} | {cr['f1-score']:<10.4f}")

    print("=" * 75)

    # 7. Save Confusion Matrix Plot
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_path=cm_plot_path)

    # 8. Save Evaluation Results JSON
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": model_path,
        "test_total": total_test,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "test_accuracy": round(test_accuracy, 4),
        "test_loss": round(test_loss, 4),
        "metrics_macro": {
            "precision": round(float(precision_macro), 4),
            "recall": round(float(recall_macro), 4),
            "f1_score": round(float(f1_macro), 4)
        },
        "metrics_weighted": {
            "precision": round(float(precision_weighted), 4),
            "recall": round(float(recall_weighted), 4),
            "f1_score": round(float(f1_weighted), 4)
        },
        "per_class_metrics": per_class_metrics,
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist()
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved complete evaluation metrics to: {EVALUATION_RESULTS_PATH}")

    return results

if __name__ == "__main__":
    evaluate_model()
