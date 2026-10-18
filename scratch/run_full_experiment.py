#!/usr/bin/env python3
"""
Full Execution Script for Phases E, F, G, and H:
Fair Comparison of EfficientNet-B0, E2 Fusion, and Equal-Weight Ensemble
across Development (80), Locked Test (21), and Descriptive (26) Cohorts.
"""

import os
import sys
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "reports" / "efficientnet_e2_ensemble_experiment"
EXP_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# Paths
EFFNET_CKPT = PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras"
E2_CKPT_DIR = PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2"
E2_CHECKPOINTS = [E2_CKPT_DIR / f"fold_{i}_best.weights.h5" for i in range(1, 6)]
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
GT_CSV = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"
FUSION_NPZ = PROJECT_ROOT / "reports" / "model_fusion" / "all_models_participant_probs.npz"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("STARTING FULL EXPERIMENT PIPELINE (PHASES E, F, G, H)")
print("=" * 80)

# -------------------------------------------------------------
# 1. VERIFY HASHES & INTEGRITY BEFORE RUN
# -------------------------------------------------------------
with open(EXP_DIR / "checkpoint_hashes_before_after.json") as f:
    hashes_doc = json.load(f)
initial_hashes = hashes_doc["before_experiment"]

current_hashes = {
    "effnet_checkpoint": sha256_of_file(EFFNET_CKPT),
    "e2_fold1_weights": sha256_of_file(E2_CHECKPOINTS[0]),
    "e2_fold2_weights": sha256_of_file(E2_CHECKPOINTS[1]),
    "e2_fold3_weights": sha256_of_file(E2_CHECKPOINTS[2]),
    "e2_fold4_weights": sha256_of_file(E2_CHECKPOINTS[3]),
    "e2_fold5_weights": sha256_of_file(E2_CHECKPOINTS[4]),
    "participant_split_csv": sha256_of_file(SPLIT_CSV),
    "mfs100_ground_truth_csv": sha256_of_file(GT_CSV)
}

for k, curr_h in current_hashes.items():
    init_h = initial_hashes[k]["sha256"]
    assert curr_h == init_h, f"FATAL: Checkpoint or manifest {k} hash mismatch! Aborting."

print("[PASS] All model checkpoints and manifests strictly verified against initial hashes.")

# -------------------------------------------------------------
# 2. LOAD MODELS
# -------------------------------------------------------------
print("Loading EfficientNet-B0...")
effnet_model = tf.keras.models.load_model(str(EFFNET_CKPT))

def build_E2():
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

print("Building E2 architecture...")
e2_model = build_E2()

def predict_effnet_participant(scan_paths):
    """Average 3 scan probability vectors for EfficientNet-B0 (224x224 RGB)."""
    assert len(scan_paths) == 3, f"Expected 3 scans, got {len(scan_paths)}"
    probs = []
    for sp in scan_paths:
        img = Image.open(sp).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = effnet_preprocess(np.array(img, dtype=np.float32))[np.newaxis, ...]
        p = effnet_model.predict(arr, verbose=0)[0]
        probs.append(p)
    return np.mean(probs, axis=0) # already in canonical order [A+, AB+, B+, O+]

def predict_e2_participant(scan_paths):
    """Run E2 5-fold ensemble on stacked 3 scans (112x112 transform_c5) and remap to canonical."""
    assert len(scan_paths) == 3, f"Expected 3 scans, got {len(scan_paths)}"
    scans_c5 = [transform_c5(str(p))[0] for p in scan_paths]
    tensor = np.stack(scans_c5, axis=0)[np.newaxis, ...] # (1, 3, 112, 112, 3)
    fold_preds = []
    for cp in E2_CHECKPOINTS:
        e2_model.load_weights(str(cp))
        clf, _ = e2_model.predict(tensor, verbose=0)
        fold_preds.append(clf[0])
    p_native = np.mean(fold_preds, axis=0)
    # Native order: [0: A+, 1: B+, 2: AB+, 3: O+]
    # Remap to canonical: [A+, AB+, B+, O+]
    p_canonical = np.array([
        p_native[0], # A+
        p_native[2], # AB+
        p_native[1], # B+
        p_native[3], # O+
    ], dtype=np.float32)
    return p_canonical

def compute_metrics(y_true, y_pred, probs=None):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Calculate always-O+ baseline (O+ is index 3)
    always_o_baseline = sum(1 for yt in y_true if yt == 3) / len(y_true)
    
    per_class = {}
    for idx, c in enumerate(CANONICAL_CLASSES):
        per_class[c] = {
            "support": int(support_class[idx]),
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx])
        }
        
    return {
        "num_participants": len(y_true),
        "num_correct": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "always_o_plus_baseline": float(always_o_baseline),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "true_class_distribution": {c: int(sum(1 for yt in y_true if yt == idx)) for idx, c in enumerate(CANONICAL_CLASSES)},
        "predicted_class_distribution": {c: int(sum(1 for yp in y_pred if yp == idx)) for idx, c in enumerate(CANONICAL_CLASSES)}
    }

def plot_and_save_cm(cm, title, out_path):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CANONICAL_CLASSES, yticklabels=CANONICAL_CLASSES, cbar=False
    )
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------------------------------------------------
# 3. PHASE E — DEVELOPMENT COHORT (80 PARTICIPANTS)
# -------------------------------------------------------------
print("\n--- RUNNING PHASE E: DEVELOPMENT COHORT (80 PARTICIPANTS) ---")
df_split = pd.read_csv(SPLIT_CSV)
dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()

# Group by participant
dev_participants = []
for pid, grp in dev_df.groupby("participant_id"):
    assert len(grp) == 3
    true_label = grp.iloc[0]["blood_group"]
    scans = [Path(p) for p in grp["scan_path"]]
    for sp in scans:
        assert sp.is_file(), f"Missing file: {sp}"
    dev_participants.append({
        "participant_id": pid,
        "true_label": true_label,
        "true_idx": CLASS_TO_IDX[true_label],
        "scans": scans
    })

dev_participants = sorted(dev_participants, key=lambda x: x["participant_id"])
print(f"Loaded {len(dev_participants)} development participants.")

# For E2, load verified OOF predictions from all_models_participant_probs.npz
npz_data = np.load(FUSION_NPZ)
e2_oof_raw = npz_data["E2 Ensemble (OOF)"] # (80, 4) in E2 native order: [A+, B+, AB+, O+]
assert len(e2_oof_raw) == 80

dev_eff_preds, dev_e2_preds, dev_comb_preds = [], [], []

for i, p_info in enumerate(dev_participants):
    pid = p_info["participant_id"]
    true_idx = p_info["true_idx"]
    scans = p_info["scans"]
    
    # E2 canonical OOF prob
    e2_nat = e2_oof_raw[i]
    p_e2 = np.array([e2_nat[0], e2_nat[2], e2_nat[1], e2_nat[3]], dtype=np.float32)
    # Ensure exact sum=1
    p_e2 = p_e2 / p_e2.sum()
    
    # EfficientNet-B0 zero-shot transfer prediction
    p_eff = predict_effnet_participant(scans)
    p_eff = p_eff / p_eff.sum()
    
    # Equal-weight ensemble
    p_comb = 0.5 * p_eff + 0.5 * p_e2
    
    pred_eff = int(np.argmax(p_eff))
    pred_e2 = int(np.argmax(p_e2))
    pred_comb = int(np.argmax(p_comb))
    
    dev_eff_preds.append({"participant_id": pid, "cohort": "DEVELOPMENT_80", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_eff, "pred_label": CANONICAL_CLASSES[pred_eff], "probs": p_eff.tolist()})
    dev_e2_preds.append({"participant_id": pid, "cohort": "DEVELOPMENT_80", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_e2, "pred_label": CANONICAL_CLASSES[pred_e2], "probs": p_e2.tolist()})
    dev_comb_preds.append({"participant_id": pid, "cohort": "DEVELOPMENT_80", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_comb, "pred_label": CANONICAL_CLASSES[pred_comb], "probs": p_comb.tolist()})

dev_y_true = [x["true_idx"] for x in dev_eff_preds]
dev_eff_metrics = compute_metrics(dev_y_true, [x["pred_idx"] for x in dev_eff_preds])
dev_e2_metrics = compute_metrics(dev_y_true, [x["pred_idx"] for x in dev_e2_preds])
dev_comb_metrics = compute_metrics(dev_y_true, [x["pred_idx"] for x in dev_comb_preds])

dev_metrics_bundle = {
    "cohort": "DEVELOPMENT_80",
    "description": "80 MFS100 development participants (64 TRAIN + 16 VALIDATION). E2 is 5-fold OOF; EfficientNet-B0 is zero-shot transfer.",
    "efficientnet_b0": dev_eff_metrics,
    "e2_fusion_oof": dev_e2_metrics,
    "equal_weight_ensemble": dev_comb_metrics
}

with open(EXP_DIR / "development_metrics.json", "w") as f:
    json.dump(dev_metrics_bundle, f, indent=2)

plot_and_save_cm(dev_eff_metrics["confusion_matrix"], "Dev: EfficientNet-B0 (Zero-Shot)", EXP_DIR / "dev_effnet_confusion.png")
plot_and_save_cm(dev_e2_metrics["confusion_matrix"], "Dev: E2 Fusion (5-Fold OOF)", EXP_DIR / "dev_e2_confusion.png")
plot_and_save_cm(dev_comb_metrics["confusion_matrix"], "Dev: Equal-Weight Ensemble", EXP_DIR / "dev_ensemble_confusion.png")

print(f"Dev Metrics: EffNet={dev_eff_metrics['accuracy']:.2%}, E2(OOF)={dev_e2_metrics['accuracy']:.2%}, Ensemble={dev_comb_metrics['accuracy']:.2%}, Always-O+={dev_eff_metrics['always_o_plus_baseline']:.2%}")

# -------------------------------------------------------------
# 4. PHASE F — LOCKED TEST EVALUATION (21 PARTICIPANTS)
# -------------------------------------------------------------
print("\n--- RUNNING PHASE F: LOCKED TEST EVALUATION (21 PARTICIPANTS) ---")
test_df = df_split[df_split["split"] == "TEST"].copy()
test_participants = []
for pid, grp in test_df.groupby("participant_id"):
    assert len(grp) == 3
    true_label = grp.iloc[0]["blood_group"]
    scans = [Path(p) for p in grp["scan_path"]]
    for sp in scans:
        assert sp.is_file(), f"Missing file: {sp}"
    test_participants.append({
        "participant_id": pid,
        "true_label": true_label,
        "true_idx": CLASS_TO_IDX[true_label],
        "scans": scans
    })

test_participants = sorted(test_participants, key=lambda x: x["participant_id"])
print(f"Loaded {len(test_participants)} locked test participants.")

test_eff_preds, test_e2_preds, test_comb_preds = [], [], []

for p_info in test_participants:
    pid = p_info["participant_id"]
    true_idx = p_info["true_idx"]
    scans = p_info["scans"]
    
    p_eff = predict_effnet_participant(scans)
    p_eff = p_eff / p_eff.sum()
    
    p_e2 = predict_e2_participant(scans)
    p_e2 = p_e2 / p_e2.sum()
    
    p_comb = 0.5 * p_eff + 0.5 * p_e2
    
    pred_eff = int(np.argmax(p_eff))
    pred_e2 = int(np.argmax(p_e2))
    pred_comb = int(np.argmax(p_comb))
    
    test_eff_preds.append({"participant_id": pid, "cohort": "LOCKED_TEST_21", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_eff, "pred_label": CANONICAL_CLASSES[pred_eff], "probs": p_eff.tolist()})
    test_e2_preds.append({"participant_id": pid, "cohort": "LOCKED_TEST_21", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_e2, "pred_label": CANONICAL_CLASSES[pred_e2], "probs": p_e2.tolist()})
    test_comb_preds.append({"participant_id": pid, "cohort": "LOCKED_TEST_21", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_comb, "pred_label": CANONICAL_CLASSES[pred_comb], "probs": p_comb.tolist()})

test_y_true = [x["true_idx"] for x in test_eff_preds]
test_eff_metrics = compute_metrics(test_y_true, [x["pred_idx"] for x in test_eff_preds])
test_e2_metrics = compute_metrics(test_y_true, [x["pred_idx"] for x in test_e2_preds])
test_comb_metrics = compute_metrics(test_y_true, [x["pred_idx"] for x in test_comb_preds])

test_metrics_bundle = {
    "cohort": "LOCKED_TEST_21",
    "description": "21 quarantined MFS100 test participants. Single un-tuned final evaluation.",
    "efficientnet_b0": test_eff_metrics,
    "e2_fusion": test_e2_metrics,
    "equal_weight_ensemble": test_comb_metrics
}

with open(EXP_DIR / "locked_test_metrics.json", "w") as f:
    json.dump(test_metrics_bundle, f, indent=2)

plot_and_save_cm(test_eff_metrics["confusion_matrix"], "Test: EfficientNet-B0", EXP_DIR / "test_effnet_confusion.png")
plot_and_save_cm(test_e2_metrics["confusion_matrix"], "Test: E2 Fusion", EXP_DIR / "test_e2_confusion.png")
plot_and_save_cm(test_comb_metrics["confusion_matrix"], "Test: Equal-Weight Ensemble", EXP_DIR / "test_ensemble_confusion.png")

print(f"Locked Test Metrics: EffNet={test_eff_metrics['accuracy']:.2%} ({test_eff_metrics['num_correct']}/21), E2={test_e2_metrics['accuracy']:.2%} ({test_e2_metrics['num_correct']}/21), Ensemble={test_comb_metrics['accuracy']:.2%} ({test_comb_metrics['num_correct']}/21), Always-O+={test_eff_metrics['always_o_plus_baseline']:.2%}")

# -------------------------------------------------------------
# 5. PHASE F — SEPARATE 26-PARTICIPANT COHORT (DESCRIPTIVE)
# -------------------------------------------------------------
print("\n--- RUNNING PHASE F: SEPARATE 26-PARTICIPANT COHORT (DESCRIPTIVE) ---")
df_gt = pd.read_csv(GT_CSV)
cohort26_participants = []
for pid, grp in df_gt.groupby("subject_id"):
    assert len(grp) == 3
    true_label = grp.iloc[0]["confirmed_blood_group"]
    scans = [PROJECT_ROOT / "data" / "captured" / f for f in grp["capture_filename"]]
    for sp in scans:
        assert sp.is_file(), f"Missing file: {sp}"
    cohort26_participants.append({
        "participant_id": pid,
        "true_label": true_label,
        "true_idx": CLASS_TO_IDX[true_label],
        "scans": scans
    })

cohort26_participants = sorted(cohort26_participants, key=lambda x: x["participant_id"])
print(f"Loaded {len(cohort26_participants)} separate cohort participants.")

c26_eff_preds, c26_e2_preds, c26_comb_preds = [], [], []

for p_info in cohort26_participants:
    pid = p_info["participant_id"]
    true_idx = p_info["true_idx"]
    scans = p_info["scans"]
    
    p_eff = predict_effnet_participant(scans)
    p_eff = p_eff / p_eff.sum()
    
    p_e2 = predict_e2_participant(scans)
    p_e2 = p_e2 / p_e2.sum()
    
    p_comb = 0.5 * p_eff + 0.5 * p_e2
    
    pred_eff = int(np.argmax(p_eff))
    pred_e2 = int(np.argmax(p_e2))
    pred_comb = int(np.argmax(p_comb))
    
    c26_eff_preds.append({"participant_id": pid, "cohort": "SEPARATE_26", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_eff, "pred_label": CANONICAL_CLASSES[pred_eff], "probs": p_eff.tolist()})
    c26_e2_preds.append({"participant_id": pid, "cohort": "SEPARATE_26", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_e2, "pred_label": CANONICAL_CLASSES[pred_e2], "probs": p_e2.tolist()})
    c26_comb_preds.append({"participant_id": pid, "cohort": "SEPARATE_26", "true_label": p_info["true_label"], "true_idx": true_idx, "pred_idx": pred_comb, "pred_label": CANONICAL_CLASSES[pred_comb], "probs": p_comb.tolist()})

c26_y_true = [x["true_idx"] for x in c26_eff_preds]
c26_eff_metrics = compute_metrics(c26_y_true, [x["pred_idx"] for x in c26_eff_preds])
c26_e2_metrics = compute_metrics(c26_y_true, [x["pred_idx"] for x in c26_e2_preds])
c26_comb_metrics = compute_metrics(c26_y_true, [x["pred_idx"] for x in c26_comb_preds])

c26_metrics_bundle = {
    "cohort": "SEPARATE_26",
    "description": "26 MFS100 participants with hospital serological confirmation (S001-S026). Descriptive evaluation of frozen models.",
    "efficientnet_b0": c26_eff_metrics,
    "e2_fusion": c26_e2_metrics,
    "equal_weight_ensemble": c26_comb_metrics
}

with open(EXP_DIR / "separate_26_cohort_metrics.json", "w") as f:
    json.dump(c26_metrics_bundle, f, indent=2)

plot_and_save_cm(c26_eff_metrics["confusion_matrix"], "26-Cohort: EfficientNet-B0", EXP_DIR / "cohort26_effnet_confusion.png")
plot_and_save_cm(c26_e2_metrics["confusion_matrix"], "26-Cohort: E2 Fusion", EXP_DIR / "cohort26_e2_confusion.png")
plot_and_save_cm(c26_comb_metrics["confusion_matrix"], "26-Cohort: Equal-Weight Ensemble", EXP_DIR / "cohort26_ensemble_confusion.png")

print(f"26-Cohort Metrics: EffNet={c26_eff_metrics['accuracy']:.2%} ({c26_eff_metrics['num_correct']}/26), E2={c26_e2_metrics['accuracy']:.2%} ({c26_e2_metrics['num_correct']}/26), Ensemble={c26_comb_metrics['accuracy']:.2%} ({c26_comb_metrics['num_correct']}/26), Always-O+={c26_eff_metrics['always_o_plus_baseline']:.2%}")

# -------------------------------------------------------------
# 6. SAVE COMBINED PREDICTION CSVS & DISAGREEMENT ANALYSIS
# -------------------------------------------------------------
print("\n--- SAVING CSV PREDICTION TABLES & DISAGREEMENT ANALYSIS ---")
all_eff_preds = dev_eff_preds + test_eff_preds + c26_eff_preds
all_e2_preds = dev_e2_preds + test_e2_preds + c26_e2_preds
all_comb_preds = dev_comb_preds + test_comb_preds + c26_comb_preds

def write_prediction_csv(preds, filename):
    out_p = EXP_DIR / filename
    with open(out_p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "cohort", "true_label", "predicted_label", "probabilities"])
        writer.writeheader()
        for r in preds:
            writer.writerow({
                "participant_id": r["participant_id"],
                "cohort": r["cohort"],
                "true_label": r["true_label"],
                "predicted_label": r["pred_label"],
                "probabilities": json.dumps([round(float(p), 6) for p in r["probs"]])
            })
    print(f"Saved: {out_p}")

write_prediction_csv(all_eff_preds, "efficientnet_predictions.csv")
write_prediction_csv(all_e2_preds, "e2_predictions.csv")
write_prediction_csv(all_comb_preds, "equal_weight_ensemble_predictions.csv")

# Disagreement Analysis Table
disagreement_rows = []
for i in range(len(all_eff_preds)):
    eff_r = all_eff_preds[i]
    e2_r = all_e2_preds[i]
    comb_r = all_comb_preds[i]
    assert eff_r["participant_id"] == e2_r["participant_id"] == comb_r["participant_id"]
    
    true_l = eff_r["true_label"]
    p_eff_l = eff_r["pred_label"]
    p_e2_l = e2_r["pred_label"]
    p_comb_l = comb_r["pred_label"]
    
    eff_correct = (p_eff_l == true_l)
    e2_correct = (p_e2_l == true_l)
    comb_correct = (p_comb_l == true_l)
    
    disagreement_rows.append({
        "participant_id": eff_r["participant_id"],
        "cohort": eff_r["cohort"],
        "true_label": true_l,
        "effnet_prediction": p_eff_l,
        "e2_prediction": p_e2_l,
        "ensemble_prediction": p_comb_l,
        "effnet_correct": eff_correct,
        "e2_correct": e2_correct,
        "ensemble_correct": comb_correct,
        "models_agree": (p_eff_l == p_e2_l),
        "ensemble_corrects_effnet": (not eff_correct and comb_correct),
        "ensemble_corrects_e2": (not e2_correct and comb_correct),
        "ensemble_introduces_error": ((eff_correct or e2_correct) and not comb_correct)
    })

df_disagreement = pd.DataFrame(disagreement_rows)
df_disagreement.to_csv(EXP_DIR / "participant_disagreement_analysis.csv", index=False)
print(f"Saved: {EXP_DIR / 'participant_disagreement_analysis.csv'}")

# Master metrics.json
master_metrics = {
    "evaluation_timestamp": datetime.now().isoformat(),
    "development_cohort_80": dev_metrics_bundle,
    "locked_test_cohort_21": test_metrics_bundle,
    "separate_cohort_26": c26_metrics_bundle
}
with open(EXP_DIR / "metrics.json", "w") as f:
    json.dump(master_metrics, f, indent=2)
print(f"Saved: {EXP_DIR / 'metrics.json'}")

# -------------------------------------------------------------
# 7. PHASE G — INDEPENDENT VERIFICATION OF SAVED CSVS & HASHES
# -------------------------------------------------------------
print("\n--- PHASE G: RUNNING INDEPENDENT MATHEMATICAL INTEGRITY VERIFICATION ---")
verification_checks = {}

# Check 1: CSV row counts
df_p_eff = pd.read_csv(EXP_DIR / "efficientnet_predictions.csv")
df_p_e2 = pd.read_csv(EXP_DIR / "e2_predictions.csv")
df_p_comb = pd.read_csv(EXP_DIR / "equal_weight_ensemble_predictions.csv")

total_expected_rows = 80 + 21 + 26 # 127
verification_checks["check1_row_counts_127_each"] = (len(df_p_eff) == total_expected_rows and len(df_p_e2) == total_expected_rows and len(df_p_comb) == total_expected_rows)

# Check 2: Matching participant IDs and cohort labels across all three CSVs
id_match = (df_p_eff["participant_id"].tolist() == df_p_e2["participant_id"].tolist() == df_p_comb["participant_id"].tolist())
cohort_match = (df_p_eff["cohort"].tolist() == df_p_e2["cohort"].tolist() == df_p_comb["cohort"].tolist())
verification_checks["check2_matching_ids_and_cohorts"] = (id_match and cohort_match)

# Check 3: Finite, non-negative, and sum-to-1 probabilities
prob_math_valid = True
for df in [df_p_eff, df_p_e2, df_p_comb]:
    for p_str in df["probabilities"]:
        arr = np.array(json.loads(p_str))
        if len(arr) != 4 or np.any(np.isnan(arr)) or np.any(arr < 0) or abs(arr.sum() - 1.0) > 1e-4:
            prob_math_valid = False
verification_checks["check3_probabilities_finite_nonneg_sum1"] = prob_math_valid

# Check 4: Exact equal-weight arithmetic mean formula
mean_formula_valid = True
for i in range(len(df_p_eff)):
    peff = np.array(json.loads(df_p_eff.iloc[i]["probabilities"]))
    pe2 = np.array(json.loads(df_p_e2.iloc[i]["probabilities"]))
    pcomb = np.array(json.loads(df_p_comb.iloc[i]["probabilities"]))
    expected = (peff + pe2) / 2.0
    if np.max(np.abs(expected - pcomb)) > 1e-4:
        mean_formula_valid = False
verification_checks["check4_exact_equal_weight_formula"] = mean_formula_valid

# Check 5: Recompute metrics directly from CSVs and verify match with saved metrics.json
recalc_match = True
for cohort_key, sub_df in [("DEVELOPMENT_80", df_p_comb[df_p_comb["cohort"] == "DEVELOPMENT_80"]),
                           ("LOCKED_TEST_21", df_p_comb[df_p_comb["cohort"] == "LOCKED_TEST_21"]),
                           ("SEPARATE_26", df_p_comb[df_p_comb["cohort"] == "SEPARATE_26"])]:
    y_t = [CLASS_TO_IDX[c] for c in sub_df["true_label"]]
    y_p = [CLASS_TO_IDX[c] for c in sub_df["predicted_label"]]
    recalc_acc = accuracy_score(y_t, y_p)
    bundle_name = "development_cohort_80" if cohort_key == "DEVELOPMENT_80" else ("locked_test_cohort_21" if cohort_key == "LOCKED_TEST_21" else "separate_cohort_26")
    saved_acc = master_metrics[bundle_name]["equal_weight_ensemble"]["accuracy"]
    if abs(recalc_acc - saved_acc) > 1e-5:
        recalc_match = False
verification_checks["check5_recalculated_metrics_match_saved"] = recalc_match

# Check 6: Post-experiment SHA-256 hash verification
post_hashes = {
    "effnet_checkpoint": sha256_of_file(EFFNET_CKPT),
    "e2_fold1_weights": sha256_of_file(E2_CHECKPOINTS[0]),
    "e2_fold2_weights": sha256_of_file(E2_CHECKPOINTS[1]),
    "e2_fold3_weights": sha256_of_file(E2_CHECKPOINTS[2]),
    "e2_fold4_weights": sha256_of_file(E2_CHECKPOINTS[3]),
    "e2_fold5_weights": sha256_of_file(E2_CHECKPOINTS[4]),
    "participant_split_csv": sha256_of_file(SPLIT_CSV),
    "mfs100_ground_truth_csv": sha256_of_file(GT_CSV)
}
post_hash_match = True
for k, post_h in post_hashes.items():
    if post_h != initial_hashes[k]["sha256"]:
        post_hash_match = False
verification_checks["check6_checkpoints_and_manifests_strictly_unchanged"] = post_hash_match

# Save before-after hash file
hashes_doc["after_experiment"] = {
    k: {"path": initial_hashes[k]["path"], "size_bytes": initial_hashes[k]["size_bytes"], "sha256": post_hashes[k]}
    for k in post_hashes
}
hashes_doc["hash_verification_status"] = "ALL_HASHES_MATCH_EXACTLY" if post_hash_match else "HASH_MISMATCH"
with open(EXP_DIR / "checkpoint_hashes_before_after.json", "w") as f:
    json.dump(hashes_doc, f, indent=2)

print("Integrity verification results:")
for chk, res in verification_checks.items():
    print(f"  {chk}: {'PASS' if res else 'FAIL'}")

assert all(verification_checks.values()), "One or more Phase G integrity checks failed!"
print("\n[ALL PHASE G INTEGRITY CHECKS PASSED]")
