#!/usr/bin/env python3
"""
Evaluation script for the Biometric Blood Group Detection project.
Compares EfficientNet-B0 (single-scan avg) and the E2 three-scan fusion ensemble
on the MFS100 26-participant cohort, adhering to strict test-set protection.
All preprocessing, class mapping, and participant-split handling follow the
original training pipelines.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ----- Project layout -----
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
MODELS_ROOT = PROJECT_ROOT / "models"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5

# Paths to required files
GT_CSV = DATA_ROOT / "mfs100_ground_truth.csv"
SPLIT_CSV = REPORTS_ROOT / "phase3_participant_benchmark" / "participant_split.csv"
EFFNET_CKPT = MODELS_ROOT / "abo_positive_4class" / "best_model.keras"
E2_CKPT_DIR = MODELS_ROOT / "phase3_fusion_metric_learning" / "E2"
E2_CHECKPOINTS = [E2_CKPT_DIR / f"fold_{i}_best.weights.h5" for i in range(1, 6)]

# ----- Helper utilities -----

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Class mappings
EFFNET_CLASS_TO_ID = {"A+": 0, "AB+": 1, "B+": 2, "O+": 3}
E2_CLASS_TO_ID = {"A+": 0, "B+": 1, "AB+": 2, "O+": 3}
ID_TO_CLASS = {v: k for k, v in EFFNET_CLASS_TO_ID.items()}

# ----- Preprocessing & Model Definitions -----
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def load_and_preprocess_effnet(img_path: Path) -> np.ndarray:
    """Load image and preprocess for EfficientNet-B0 (224x224 RGB)."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return efficientnet_preprocess(arr)

def P2(img_path):
    """Current deployed transform_c5 (112x112)."""
    arr, _, _ = transform_c5(img_path)
    return arr

def preprocess_three_scans(scan_paths):
    """Apply C5 preprocessing (P2) to each scan and stack. Shape: (3, 112, 112, 3)."""
    processed = [P2(str(p)) for p in scan_paths]
    return np.stack(processed, axis=0)

def build_E2():
    """Build E2 three-scan fusion model architecture matching checkpoints."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
        tf.keras.layers.RandomContrast(0.1)
    ])
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1 = data_augmentation(s1, training=False)
    a2 = data_augmentation(s2, training=False)
    a3 = data_augmentation(s3, training=False)
    base = tf.keras.applications.DenseNet121(include_top=False, weights=None)
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    f1, f2, f3 = pool(base(a1)), pool(base(a2)), pool(base(a3))
    concat = tf.keras.layers.Concatenate()([f1, f2, f3])
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

# ----- Data loading -----
if not GT_CSV.is_file():
    raise FileNotFoundError(f"Ground-truth CSV not found: {GT_CSV}")

participants = {}
with GT_CSV.open(newline="") as f:
    for row in pd.read_csv(f).itertuples(index=False):
        pid = row.subject_id
        label = row.confirmed_blood_group
        img_path = Path(row.capture_filename)
        if not img_path.is_absolute():
            img_path = DATA_ROOT / "captured" / img_path
        participants.setdefault(pid, {"scans": [], "label": label})["scans"].append(img_path)

# Validate that all image files exist
missing_images = []
total_images = 0
for pid, info in participants.items():
    for p in info["scans"]:
        total_images += 1
        if not p.is_file():
            missing_images.append(p)

if missing_images:
    print(f"[!] Missing {len(missing_images)} out of {total_images} images.")
    for m in missing_images:
        print(f"    Missing: {m}")
    raise FileNotFoundError(f"{len(missing_images)} image files not found.")
else:
    print(f"[+] All {total_images} image files found for {len(participants)} participants.")

for pid, info in participants.items():
    if len(info["scans"]) != 3:
        raise ValueError(f"Participant {pid} has {len(info['scans'])} scans; expected 3.")

# Load participant split
if not SPLIT_CSV.is_file():
    raise FileNotFoundError(f"Participant split file missing: {SPLIT_CSV}.")

split_df = pd.read_csv(SPLIT_CSV)
train_ids = set(split_df[split_df["split"].str.lower() == "train"]["participant_id"])
test_ids = set(split_df[split_df["split"].str.lower() == "test"]["participant_id"])

if not test_ids.intersection(participants.keys()):
    print("[!] No matching IDs in participant_split.csv; using all ground-truth participants as test set.")
    test_ids = set(participants.keys())
else:
    if not test_ids.issubset(participants.keys()):
        missing = test_ids - participants.keys()
        raise ValueError(f"Test split contains unknown participants: {missing}")
    if train_ids & test_ids:
        raise ValueError("Train and test splits overlap - this violates the protocol.")

# ----- Model loading -----
if not EFFNET_CKPT.is_file():
    raise FileNotFoundError(f"EfficientNet checkpoint not found: {EFFNET_CKPT}")
print("Loading EfficientNet-B0 model ...")
EffNet = tf.keras.models.load_model(str(EFFNET_CKPT))

print("Building E2 fusion model ...")
E2_model = build_E2()
for cp in E2_CHECKPOINTS:
    if not cp.is_file():
        raise FileNotFoundError(f"E2 checkpoint missing: {cp}")

# ----- Evaluation -----
results_dir = REPORTS_ROOT / "model_comparison_results"
results_dir.mkdir(parents=True, exist_ok=True)

effnet_preds = {}
e2_preds = {}
combined_preds = {}

print(f"Evaluating {len(test_ids)} participants...")
for pid in sorted(test_ids):
    info = participants[pid]
    scans = info["scans"]
    true_label_str = info["label"]
    true_id = EFFNET_CLASS_TO_ID[true_label_str]

    # EfficientNet: average probabilities across the 3 scans
    probs_list = []
    for sp in scans:
        arr = load_and_preprocess_effnet(sp)
        prob = EffNet.predict(arr[np.newaxis, ...], verbose=0)[0]
        probs_list.append(prob)
    effnet_prob = np.mean(probs_list, axis=0)

    # E2: preprocess each scan, stack to (1, 3, 112, 112, 3), ensemble over 5 folds
    stacked = preprocess_three_scans(scans)
    X = stacked[np.newaxis, ...]
    e2_all = []
    for cp in E2_CHECKPOINTS:
        E2_model.load_weights(str(cp))
        clf, _ = E2_model.predict(X, verbose=0)
        e2_all.append(clf[0])
    e2_prob_raw = np.mean(e2_all, axis=0)

    # Reorder E2 probs (A+, B+, AB+, O+) to match EfficientNet order (A+, AB+, B+, O+)
    reorder = [
        e2_prob_raw[E2_CLASS_TO_ID["A+"]],
        e2_prob_raw[E2_CLASS_TO_ID["AB+"]],
        e2_prob_raw[E2_CLASS_TO_ID["B+"]],
        e2_prob_raw[E2_CLASS_TO_ID["O+"]],
    ]
    e2_prob = np.array(reorder, dtype=np.float32)

    # Combined ensemble (simple average)
    combined_prob = (effnet_prob + e2_prob) / 2.0

    effnet_preds[pid] = {
        "true_id": true_id,
        "probs": effnet_prob.tolist(),
        "pred": int(np.argmax(effnet_prob)),
    }
    e2_preds[pid] = {
        "true_id": true_id,
        "probs": e2_prob.tolist(),
        "pred": int(np.argmax(e2_prob)),
    }
    combined_preds[pid] = {
        "true_id": true_id,
        "probs": combined_prob.tolist(),
        "pred": int(np.argmax(combined_prob)),
    }

# ----- Save CSVs -----
def write_csv(pred_dict, filename):
    out_path = results_dir / filename
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "true_label", "predicted_label", "probabilities"])
        writer.writeheader()
        for pid, d in pred_dict.items():
            writer.writerow({
                "participant_id": pid,
                "true_label": ID_TO_CLASS[d["true_id"]],
                "predicted_label": ID_TO_CLASS[d["pred"]],
                "probabilities": json.dumps([round(float(p), 6) for p in d["probs"]]),
            })
    print(f"[+] CSV saved: {out_path}")

write_csv(effnet_preds, "effnet_predictions.csv")
write_csv(e2_preds, "e2_predictions.csv")
write_csv(combined_preds, "combined_predictions.csv")

# ----- Metric computation -----
def compute_metrics(pred_dict):
    y_true = [d["true_id"] for d in pred_dict.values()]
    y_pred = [d["pred"] for d in pred_dict.values()]
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(ID_TO_CLASS.keys()))
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "confusion_matrix": cm.tolist(),
    }

metrics = {
    "effnet": compute_metrics(effnet_preds),
    "e2": compute_metrics(e2_preds),
    "combined": compute_metrics(combined_preds),
}

metrics_path = results_dir / "metrics.json"
with metrics_path.open("w") as f:
    json.dump(metrics, f, indent=2)
print(f"[+] Metrics JSON saved: {metrics_path}")

# ----- Majority-class baseline -----
train_labels = [EFFNET_CLASS_TO_ID[l] for l in split_df[split_df["split"].str.lower() == "train"]["blood_group"]]
if train_labels:
    majority_class = max(set(train_labels), key=train_labels.count)
    baseline_acc = sum(1 for d in effnet_preds.values() if d["true_id"] == majority_class) / len(effnet_preds)
    print(f"Majority-class ({ID_TO_CLASS[majority_class]}) baseline accuracy on test set: {baseline_acc:.2%}")
else:
    majority_class = 0
    baseline_acc = 0.0

# ----- Confusion matrix visualisations -----
def plot_cm(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID_TO_CLASS[i] for i in range(4)],
        yticklabels=[ID_TO_CLASS[i] for i in range(4)],
        cbar=False
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = results_dir / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix saved: {out}")

plot_cm(metrics["effnet"]["confusion_matrix"], "EfficientNet-B0 Test Confusion Matrix", "effnet_confusion.png")
plot_cm(metrics["e2"]["confusion_matrix"], "E2 Fusion Test Confusion Matrix", "e2_confusion.png")
plot_cm(metrics["combined"]["confusion_matrix"], "Combined Ensemble Test Confusion Matrix", "combined_confusion.png")

# ----- Summary report -----
summary_md = results_dir / "summary_report.md"
with summary_md.open("w") as f:
    f.write("# Model Comparison Summary\n\n")
    f.write(f"**Evaluation timestamp:** {datetime.now().isoformat()}\n\n")
    f.write("## Metrics (Test Set)\n\n")
    for name, m in metrics.items():
        f.write(f"### {name.capitalize()}\n")
        f.write(f"- Accuracy: {m['accuracy']:.2%}\n")
        f.write(f"- Macro Precision: {m['macro_precision']:.4f}\n")
        f.write(f"- Macro Recall: {m['macro_recall']:.4f}\n")
        f.write(f"- Macro F1: {m['macro_f1']:.4f}\n\n")
    f.write(f"**Majority-class baseline (class {ID_TO_CLASS[majority_class]}) accuracy:** {baseline_acc:.2%}\n\n")
    f.write("---\n\n")
    f.write("*All file hashes recorded for reproducibility.*\n")
print(f"[+] Summary report written to {summary_md}")

# ----- Record model checkpoint hashes -----
hash_record = {
    "effnet_checkpoint_sha256": sha256_of_file(EFFNET_CKPT),
    "e2_checkpoints_sha256": [sha256_of_file(p) for p in E2_CHECKPOINTS],
    "evaluation_timestamp": datetime.now().isoformat(),
}
hash_path = results_dir / "checkpoint_hashes.json"
with hash_path.open("w") as f:
    json.dump(hash_record, f, indent=2)
print(f"[+] Checkpoint hashes saved: {hash_path}")

print(f"\nEvaluation complete. All results are stored under {results_dir}")
