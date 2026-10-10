#!/usr/bin/env python3
"""
scripts/compare_models.py
Reads outputs/model_experiments.csv and generates:
  - docs/MODEL_COMPARISON.md    : Comparison table (val only, no test leakage)
  - outputs/model_comparison.json : Structured comparison data
  - outputs/model_comparison.png  : Bar chart of val accuracy and macro F1

Also generates a detailed confusion matrix analysis for the best model on the
validation set (using the saved .keras model).

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --best_arch mobilenetv2
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
)
from src.preprocessing import composite_alpha_on_white, resize_to_target

EXPERIMENT_CSV = os.path.join(OUTPUTS_DIR, "model_experiments.csv")
COMPARISON_JSON = os.path.join(OUTPUTS_DIR, "model_comparison.json")
COMPARISON_PNG = os.path.join(OUTPUTS_DIR, "model_comparison.png")
MODEL_COMPARISON_DOC = os.path.join(
    os.path.join(os.path.dirname(__file__), "..", "docs"), "MODEL_COMPARISON.md"
)

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

MODEL_SAVE_PATHS = {
    "custom_cnn":     os.path.join(MODELS_DIR, "custom_cnn_blood_group.keras"),
    "mobilenetv2":    os.path.join(MODELS_DIR, "mobilenetv2_blood_group.keras"),
    "efficientnetb0": os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
    "densenet121":    os.path.join(MODELS_DIR, "densenet121_blood_group.keras"),
    "resnet50":       os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
    "vgg16":          os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
    "vgg19":          os.path.join(MODELS_DIR, "vgg19_blood_group.keras"),
}

# Baseline (original, never retrained)
BASELINE_INFO = {
    "model": "resnet50_baseline",
    "best_val_accuracy": 0.6002,
    "val_macro_f1": 0.5691,
    "training_time_min": None,
    "parameters": None,
    "notes": "[BASELINE] Original 8-epoch training (57.57% test acc)"
}


def load_val_split():
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    val_df = df[df["split"] == "val"]
    return val_df["image_path"].tolist(), val_df["class_id"].astype(int).tolist()


def preprocess_image(path_str, arch_name):
    from PIL import Image
    target_size = INPUT_SIZES[arch_name]
    preprocess_fn = PREPROCESS_FUNCS[arch_name]
    with Image.open(path_str) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, target_size=target_size)
    arr = np.array(resized, dtype=np.float32)
    return preprocess_fn(arr)


def evaluate_model_on_val(arch_name, model, val_paths, val_labels, batch_size=32):
    """Evaluate a model on the validation set and return pred_labels, metrics."""
    all_probs = []
    for i in range(0, len(val_paths), batch_size):
        batch_paths = val_paths[i:i+batch_size]
        batch_imgs = [preprocess_image(p, arch_name) for p in batch_paths]
        batch_tensor = np.stack(batch_imgs, axis=0)
        probs = model.predict(batch_tensor, verbose=0)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    pred_labels = np.argmax(all_probs, axis=1)
    true_labels = np.array(val_labels)

    _, _, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="macro", zero_division=0
    )
    val_acc = accuracy_score(true_labels, pred_labels)
    return pred_labels, true_labels, float(f1_macro), float(val_acc)


def plot_comparison_chart(df_exp, output_path=COMPARISON_PNG):
    """Bar chart comparing val accuracy and macro F1 across models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Deduplicate: keep best run per model
    best_runs = df_exp.sort_values("best_val_accuracy", ascending=False).drop_duplicates("model")
    best_runs = best_runs.sort_values("best_val_accuracy", ascending=True)

    models = best_runs["model"].tolist()
    val_accs = best_runs["best_val_accuracy"].tolist()
    f1s = best_runs["val_macro_f1"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.barh(x - width/2, val_accs, width, label="Val Accuracy", color="#2563eb", alpha=0.85)
    bars2 = ax.barh(x + width/2, f1s, width, label="Macro F1", color="#10b981", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Blood Group Model Comparison — Validation Performance\n(Test set NOT included — locked until final evaluation)",
                 fontsize=12, fontweight="bold")
    ax.axvline(x=0.5757, color="#dc2626", linestyle="--", alpha=0.7, label="Baseline ResNet50 test acc (57.57%)")
    ax.legend(fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Annotate bars
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=8, color="#1e3a5f")
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=8, color="#065f46")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart: {output_path}")


def plot_best_model_confusion_matrix(arch_name, y_true, y_pred, output_dir=OUTPUTS_DIR):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(10, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar=True, linewidths=0.5
    )

    # Highlight diagonal (correct predictions)
    plt.title(f"BEST MODEL: {arch_name.upper()} — Confusion Matrix (Validation Set)\n"
              f"(Test set remains locked)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Blood Group", fontsize=11, fontweight="bold")
    plt.ylabel("True Blood Group", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"best_model_{arch_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[+] Saved best model confusion matrix: {cm_path}")
    return cm


def generate_comparison_markdown(df_exp, best_arch, comparison_data, output_path):
    """Generate docs/MODEL_COMPARISON.md with the full comparison table."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# Model Comparison Report")
    lines.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n> ⚠️ **Test set is LOCKED** — no test accuracy appears in this table.")
    lines.append("> Model selection is based exclusively on validation performance.")
    lines.append("> Only the final selected model will be evaluated on the test set.\n")

    lines.append("## Comparison Table\n")
    lines.append("| Model | Best Val Accuracy | Macro F1 | Training Time | Parameters | Notes |")
    lines.append("|---|---|---|---|---|---|")

    # Baseline row
    lines.append(
        f"| **resnet50_baseline** (locked) | 60.02% | 56.91% | — | — | "
        f"Original 8-epoch baseline (test acc: 57.57%) |"
    )

    # Experiment rows
    best_per_model = df_exp.sort_values("best_val_accuracy", ascending=False).drop_duplicates("model")
    best_per_model = best_per_model.sort_values("best_val_accuracy", ascending=False)

    for _, row in best_per_model.iterrows():
        marker = " 🏆" if row["model"] == best_arch else ""
        params_str = f"{int(row['parameters']):,}" if pd.notna(row.get("parameters")) else "—"
        time_str = f"{row['training_time_min']:.1f} min" if pd.notna(row.get("training_time_min")) else "—"
        notes_clean = str(row.get("notes", "")).replace("|", "/")
        lines.append(
            f"| **{row['model']}{marker}** | {row['best_val_accuracy']*100:.2f}% | "
            f"{row['val_macro_f1']*100:.2f}% | {time_str} | {params_str} | {notes_clean} |"
        )

    lines.append(f"\n🏆 **Recommended Model**: `{best_arch}`")
    lines.append(f"  - Best Val Accuracy: {comparison_data.get('best_val_accuracy', 0)*100:.2f}%")
    lines.append(f"  - Val Macro F1: {comparison_data.get('val_macro_f1', 0)*100:.2f}%\n")

    lines.append("## Confusion Analysis (Validation Set)\n")
    lines.append("See generated confusion matrix images in `outputs/`:\n")
    for _, row in best_per_model.iterrows():
        lines.append(f"- `outputs/{row['model']}_confusion_matrix.png`")

    lines.append("\n## Per-Class Analysis\n")
    if "per_class_metrics" in comparison_data:
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for cls, m in comparison_data["per_class_metrics"].items():
            lines.append(
                f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1-score']:.3f} | {int(m['support'])} |"
            )

    lines.append("\n## Scientific Note\n")
    lines.append(
        "> This project is a research/prototype system investigating whether fingerprint images "
        "can be used to classify blood group labels. Results are not medically validated.\n"
        "> **All predictions must be confirmed using standard laboratory blood typing before medical use.**"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Saved MODEL_COMPARISON.md: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained blood group models")
    parser.add_argument("--best_arch", type=str, default=None,
                        help="Manually specify the best architecture to analyze in detail")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("=" * 70)
    print("MODEL COMPARISON REPORT GENERATOR")
    print("=" * 70)
    print("NOTE: Test set is NEVER evaluated here. Validation set only.")

    # Load experiment CSV
    if not os.path.exists(EXPERIMENT_CSV):
        print(f"[!] No experiment CSV found at {EXPERIMENT_CSV}")
        print("    Run scripts/train_models.py first.")
        return

    df_exp = pd.read_csv(EXPERIMENT_CSV, encoding="utf-8", encoding_errors="replace")
    if len(df_exp) == 0:
        print("[!] Experiment CSV is empty. Run train_models.py first.")
        return

    print(f"\n[+] Loaded {len(df_exp)} experiment records from {EXPERIMENT_CSV}")

    # Determine best architecture by validation accuracy
    best_row = df_exp.sort_values("best_val_accuracy", ascending=False).iloc[0]
    best_arch = args.best_arch if args.best_arch else best_row["model"]
    print(f"[+] Best architecture: {best_arch} (val_acc={best_row['best_val_accuracy']:.4f})")

    # Load val split
    val_paths, val_labels = load_val_split()

    # Evaluate best model on val set for detailed per-class metrics
    best_model_path = MODEL_SAVE_PATHS.get(best_arch)
    comparison_data = {}

    if best_model_path and os.path.exists(best_model_path):
        print(f"\n[*] Loading best model: {best_model_path}")
        best_model = tf.keras.models.load_model(best_model_path)

        print("[*] Evaluating best model on validation set...")
        pred_labels, true_labels, f1_macro, val_acc = evaluate_model_on_val(
            best_arch, best_model, val_paths, val_labels, args.batch_size
        )

        clf_report = classification_report(
            true_labels, pred_labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        print(f"\nBest Model ({best_arch}) Validation Metrics:")
        print(f"  Accuracy:  {val_acc:.4f}")
        print(f"  Macro F1:  {f1_macro:.4f}")
        print(f"\n  Per-class report:")
        for cls in CLASS_NAMES:
            m = clf_report[cls]
            print(f"    {cls:5s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} (n={int(m['support'])})")

        # Detailed confusion matrix for best model
        cm = plot_best_model_confusion_matrix(best_arch, true_labels, pred_labels)

        comparison_data = {
            "best_arch": best_arch,
            "best_val_accuracy": float(val_acc),
            "val_macro_f1": float(f1_macro),
            "per_class_metrics": {
                cls: clf_report[cls] for cls in CLASS_NAMES
            },
            "confusion_matrix": cm.tolist()
        }
    else:
        print(f"[!] Best model file not found: {best_model_path}")
        comparison_data = {
            "best_arch": best_arch,
            "best_val_accuracy": float(best_row["best_val_accuracy"]),
            "val_macro_f1": float(best_row["val_macro_f1"]),
        }

    # Generate comparison chart
    plot_comparison_chart(df_exp, COMPARISON_PNG)

    # Generate comparison JSON
    all_results = []
    best_per_model = df_exp.sort_values("best_val_accuracy", ascending=False).drop_duplicates("model")
    for _, row in best_per_model.iterrows():
        all_results.append({
            "model": row["model"],
            "best_val_accuracy": float(row["best_val_accuracy"]),
            "val_macro_f1": float(row["val_macro_f1"]),
            "epochs": int(row["epochs"]) if pd.notna(row.get("epochs")) else None,
            "training_time_min": float(row["training_time_min"]) if pd.notna(row.get("training_time_min")) else None,
            "parameters": int(row["parameters"]) if pd.notna(row.get("parameters")) else None,
            "notes": str(row.get("notes", ""))
        })

    comparison_json = {
        "generated": pd.Timestamp.now().isoformat(),
        "best_arch": best_arch,
        "baseline": BASELINE_INFO,
        "models": all_results,
        "best_model_detail": comparison_data
    }

    with open(COMPARISON_JSON, "w") as f:
        json.dump(comparison_json, f, indent=2)
    print(f"[+] Saved comparison JSON: {COMPARISON_JSON}")

    # Generate markdown report
    generate_comparison_markdown(df_exp, best_arch, comparison_data, MODEL_COMPARISON_DOC)

    # Final summary
    print("\n" + "=" * 70)
    print("COMPARISON REPORT COMPLETE")
    print("=" * 70)
    print(f"Best model (validation): {best_arch.upper()}")
    print(f"Val accuracy:            {comparison_data.get('best_val_accuracy', 0)*100:.2f}%")
    print(f"Val macro F1:            {comparison_data.get('val_macro_f1', 0)*100:.2f}%")
    print(f"\nBaseline ResNet50 test acc: 57.57%")
    print(f"\nFiles generated:")
    print(f"  - {MODEL_COMPARISON_DOC}")
    print(f"  - {COMPARISON_JSON}")
    print(f"  - {COMPARISON_PNG}")
    print("\n" + "=" * 70)
    print("NEXT STEP: Present comparison to user and wait for approval")
    print("before running evaluate_final_model.py on the locked test set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
