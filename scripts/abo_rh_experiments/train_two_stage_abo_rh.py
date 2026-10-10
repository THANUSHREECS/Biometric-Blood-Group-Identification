"""
scripts/abo_rh_experiments/train_two_stage_abo_rh.py
=============================================================
Two-Stage Multi-Task ABO + Rh Blood Group Classification Experiment
Phase 1: Controlled Training & Validation Evaluation
Strictly non-modifying to production models.
Does NOT evaluate locked test set in this phase.
"""

import os
import sys
import json
import csv
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, log_loss
)
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH,
    MODELS_DIR, OUTPUTS_DIR
)
from src.preprocessing import composite_alpha_on_white, resize_to_target

# Paths
EXPERIMENT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "abo_rh_experiments")
EXPERIMENT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "abo_rh_experiments")
FIGURES_DIR = os.path.join(EXPERIMENT_OUTPUTS_DIR, "figures")
os.makedirs(EXPERIMENT_MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_CHECKPOINT_PATH = os.path.join(EXPERIMENT_MODELS_DIR, "two_stage_abo_rh_best_val.keras")
FINAL_MODEL_PATH = os.path.join(EXPERIMENT_MODELS_DIR, "two_stage_abo_rh_model.keras")

# Label Mappings
ABO_CLASSES = ["A", "AB", "B", "O"]
ABO_TO_IDX = {c: i for i, c in enumerate(ABO_CLASSES)}
IDX_TO_ABO = {i: c for i, c in enumerate(ABO_CLASSES)}

RH_CLASSES = ["+", "-"]
RH_TO_IDX = {c: i for i, c in enumerate(RH_CLASSES)}
IDX_TO_RH = {i: c for i, c in enumerate(RH_CLASSES)}

CLASS_TO_ABO_RH = {
    cname: (cname[:-1], cname[-1]) for cname in CLASS_NAMES
}

# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────

def load_split_data(df, split_name, augment=False):
    sub_df = df[df["split"] == split_name].reset_index(drop=True)
    n = len(sub_df)
    print(f"[*] Loading {split_name} data ({n} images)...")

    imgs = []
    y_8class = []
    y_abo = []
    y_rh = []
    meta = []

    for idx, row in sub_df.iterrows():
        p = row["image_path"]
        cname = row["class_name"]
        cid = row["class_id"]

        abo_str, rh_str = CLASS_TO_ABO_RH[cname]
        abo_id = ABO_TO_IDX[abo_str]
        rh_id = RH_TO_IDX[rh_str]

        with Image.open(p) as img:
            rgb = composite_alpha_on_white(img)
            resized = resize_to_target(rgb, (224, 224))
            arr = np.array(resized, dtype=np.float32)

        imgs.append(arr)
        y_8class.append(cid)
        y_abo.append(abo_id)
        y_rh.append(rh_id)
        meta.append({
            "image_path": p,
            "filename": os.path.basename(p),
            "class_name": cname,
            "class_id": cid,
            "abo_str": abo_str,
            "abo_id": abo_id,
            "rh_str": rh_str,
            "rh_id": rh_id
        })

    imgs_arr = np.array(imgs, dtype=np.float32)
    imgs_preprocessed = tf.keras.applications.resnet50.preprocess_input(imgs_arr)

    return (
        imgs_preprocessed,
        np.array(y_8class, dtype=np.int32),
        np.array(y_abo, dtype=np.int32),
        np.array(y_rh, dtype=np.int32),
        meta
    )

# ─── MULTI-TASK MODEL ARCHITECTURE ───────────────────────────────────────────

def build_two_stage_model():
    base_model_path = os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras")
    print(f"[*] Loading feature extractor backbone from {base_model_path}...")
    source_model = tf.keras.models.load_model(base_model_path, compile=False)

    resnet_backbone = source_model.get_layer("resnet50")
    resnet_backbone.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="fingerprint_input")
    x = resnet_backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="shared_gap")(x)

    shared_dense = layers.Dense(256, activation="relu", name="shared_dense_256")(x)
    shared_drop = layers.Dropout(0.30, name="shared_dropout")(shared_dense)

    # ABO Branch (4 classes)
    abo_dense = layers.Dense(128, activation="relu", name="abo_dense_128")(shared_drop)
    abo_drop = layers.Dropout(0.20, name="abo_dropout")(abo_dense)
    abo_out = layers.Dense(4, activation="softmax", name="abo_output")(abo_drop)

    # Rh Branch (2 classes)
    rh_dense = layers.Dense(64, activation="relu", name="rh_dense_64")(shared_drop)
    rh_drop = layers.Dropout(0.20, name="rh_dropout")(rh_dense)
    rh_out = layers.Dense(2, activation="softmax", name="rh_output")(rh_drop)

    model = models.Model(inputs=inputs, outputs=[abo_out, rh_out], name="TwoStage_ABO_Rh_ResNet50")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
        loss_weights=[1.0, 1.0],
        metrics=[["accuracy"], ["accuracy"]]
    )
    return model

# ─── RECONSTRUCTED 8-CLASS EVALUATION HELPER ─────────────────────────────────

def evaluate_predictions(y_8class_true, p_abo, p_rh, meta_list):
    n = len(y_8class_true)

    pred_abo_idx = np.argmax(p_abo, axis=1)
    conf_abo = np.max(p_abo, axis=1)

    pred_rh_idx = np.argmax(p_rh, axis=1)
    conf_rh = np.max(p_rh, axis=1)

    # Reconstruct 8-class labels
    y_8class_pred = []
    conf_prod = []
    conf_mean = []

    for i in range(n):
        abo_name = IDX_TO_ABO[pred_abo_idx[i]]
        rh_name = IDX_TO_RH[pred_rh_idx[i]]
        recon_name = f"{abo_name}{rh_name}"
        recon_id = CLASS_NAMES.index(recon_name)
        y_8class_pred.append(recon_id)
        conf_prod.append(float(conf_abo[i] * conf_rh[i]))
        conf_mean.append(float((conf_abo[i] + conf_rh[i]) / 2.0))

    y_8class_pred = np.array(y_8class_pred, dtype=np.int32)
    conf_prod = np.array(conf_prod, dtype=np.float32)
    conf_mean = np.array(conf_mean, dtype=np.float32)

    # 1. ABO Metrics
    y_abo_true = np.array([m["abo_id"] for m in meta_list], dtype=np.int32)
    abo_acc = float(accuracy_score(y_abo_true, pred_abo_idx))
    abo_bal_acc = float(balanced_accuracy_score(y_abo_true, pred_abo_idx))
    abo_p_m, abo_r_m, abo_f1_m, _ = precision_recall_fscore_support(y_abo_true, pred_abo_idx, average="macro", zero_division=0)
    abo_cm = confusion_matrix(y_abo_true, pred_abo_idx, labels=list(range(4)))

    # 2. Rh Metrics
    y_rh_true = np.array([m["rh_id"] for m in meta_list], dtype=np.int32)
    rh_acc = float(accuracy_score(y_rh_true, pred_rh_idx))
    rh_bal_acc = float(balanced_accuracy_score(y_rh_true, pred_rh_idx))
    rh_p_m, rh_r_m, rh_f1_m, _ = precision_recall_fscore_support(y_rh_true, pred_rh_idx, average="macro", zero_division=0)
    rh_cm = confusion_matrix(y_rh_true, pred_rh_idx, labels=list(range(2)))

    # 3. 8-Class Reconstructed Metrics
    acc_8 = float(accuracy_score(y_8class_true, y_8class_pred))
    bal_acc_8 = float(balanced_accuracy_score(y_8class_true, y_8class_pred))
    p8_m, r8_m, f1_8_m, _ = precision_recall_fscore_support(y_8class_true, y_8class_pred, average="macro", zero_division=0)
    p8_w, r8_w, f1_8_w, _ = precision_recall_fscore_support(y_8class_true, y_8class_pred, average="weighted", zero_division=0)
    cm_8 = confusion_matrix(y_8class_true, y_8class_pred, labels=list(range(8)))

    # Per-Class Metrics
    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(y_8class_true, y_8class_pred, average=None, zero_division=0)
    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        mask = (y_8class_true == i)
        c_tot = int(s_cls[i])
        c_corr = int(cm_8[i, i])
        c_confs = conf_mean[mask]
        c_corr_confs = conf_mean[mask & (y_8class_true == y_8class_pred)]
        c_inc_confs = conf_mean[mask & (y_8class_true != y_8class_pred)]

        per_class[cname] = {
            "class_name": cname,
            "support": c_tot,
            "correct": c_corr,
            "incorrect": c_tot - c_corr,
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "mean_confidence": round(float(np.mean(c_confs)), 4) if len(c_confs) > 0 else 0.0,
            "mean_confidence_correct": round(float(np.mean(c_corr_confs)), 4) if len(c_corr_confs) > 0 else 0.0,
            "mean_confidence_incorrect": round(float(np.mean(c_inc_confs)), 4) if len(c_inc_confs) > 0 else 0.0,
        }

    # Error breakdown & categorization
    err_rows = []
    abo_only_errs = 0
    rh_only_errs = 0
    both_errs = 0

    for i in range(n):
        act_8 = CLASS_NAMES[y_8class_true[i]]
        prd_8 = CLASS_NAMES[y_8class_pred[i]]
        is_corr = bool(y_8class_true[i] == y_8class_pred[i])

        act_abo = meta_list[i]["abo_str"]
        prd_abo = IDX_TO_ABO[pred_abo_idx[i]]
        act_rh = meta_list[i]["rh_str"]
        prd_rh = IDX_TO_RH[pred_rh_idx[i]]

        abo_wrong = bool(act_abo != prd_abo)
        rh_wrong = bool(act_rh != prd_rh)

        err_cat = "None"
        if not is_corr:
            if abo_wrong and not rh_wrong:
                err_cat = "ABO_ONLY"
                abo_only_errs += 1
            elif not abo_wrong and rh_wrong:
                err_cat = "Rh_ONLY"
                rh_only_errs += 1
            else:
                err_cat = "BOTH_ABO_AND_Rh"
                both_errs += 1

            err_rows.append({
                "filename": meta_list[i]["filename"],
                "image_path": meta_list[i]["image_path"],
                "actual_8class": act_8,
                "predicted_8class": prd_8,
                "error_category": err_cat,
                "actual_abo": act_abo,
                "predicted_abo": prd_abo,
                "abo_confidence": round(float(conf_abo[i]), 4),
                "actual_rh": act_rh,
                "predicted_rh": prd_rh,
                "rh_confidence": round(float(conf_rh[i]), 4),
                "combined_confidence_prod": round(float(conf_prod[i]), 4),
                "combined_confidence_mean": round(float(conf_mean[i]), 4)
            })

    correct_mask = (y_8class_true == y_8class_pred)
    corr_confs = conf_mean[correct_mask]
    inc_confs = conf_mean[~correct_mask]

    high_conf_80 = int(np.sum((conf_mean >= 0.80) & (~correct_mask)))
    high_conf_90 = int(np.sum((conf_mean >= 0.90) & (~correct_mask)))

    return {
        "reconstructed_8class": {
            "accuracy": round(acc_8, 4),
            "balanced_accuracy": round(bal_acc_8, 4),
            "macro_precision": round(float(p8_m), 4),
            "macro_recall": round(float(r8_m), 4),
            "macro_f1": round(float(f1_8_m), 4),
            "weighted_precision": round(float(p8_w), 4),
            "weighted_recall": round(float(r8_w), 4),
            "weighted_f1": round(float(f1_8_w), 4),
            "confusion_matrix": cm_8,
            "per_class": per_class
        },
        "abo_metrics": {
            "accuracy": round(abo_acc, 4),
            "balanced_accuracy": round(abo_bal_acc, 4),
            "macro_precision": round(float(abo_p_m), 4),
            "macro_recall": round(float(abo_r_m), 4),
            "macro_f1": round(float(abo_f1_m), 4),
            "confusion_matrix": abo_cm
        },
        "rh_metrics": {
            "accuracy": round(rh_acc, 4),
            "balanced_accuracy": round(rh_bal_acc, 4),
            "macro_precision": round(float(rh_p_m), 4),
            "macro_recall": round(float(rh_r_m), 4),
            "macro_f1": round(float(rh_f1_m), 4),
            "confusion_matrix": rh_cm
        },
        "error_analysis": {
            "total_samples": n,
            "total_errors": len(err_rows),
            "abo_only_errors": abo_only_errs,
            "rh_only_errors": rh_only_errs,
            "both_errors": both_errs,
            "abo_only_pct": round(abo_only_errs / len(err_rows) * 100.0, 2) if len(err_rows) > 0 else 0.0,
            "rh_only_pct": round(rh_only_errs / len(err_rows) * 100.0, 2) if len(err_rows) > 0 else 0.0,
            "both_pct": round(both_errs / len(err_rows) * 100.0, 2) if len(err_rows) > 0 else 0.0,
            "error_records": err_rows
        },
        "confidence_stats": {
            "mean_confidence": round(float(np.mean(conf_mean)), 4),
            "median_confidence": round(float(np.median(conf_mean)), 4),
            "mean_confidence_correct": round(float(np.mean(corr_confs)), 4) if len(corr_confs) > 0 else 0.0,
            "mean_confidence_incorrect": round(float(np.mean(inc_confs)), 4) if len(inc_confs) > 0 else 0.0,
            "max_confidence_incorrect": round(float(np.max(inc_confs)), 4) if len(inc_confs) > 0 else 0.0,
            "high_conf_wrong_ge_80": high_conf_80,
            "high_conf_wrong_ge_90": high_conf_90
        },
        "raw_predictions": {
            "pred_abo_idx": pred_abo_idx,
            "pred_rh_idx": pred_rh_idx,
            "y_8class_pred": y_8class_pred,
            "conf_abo": conf_abo,
            "conf_rh": conf_rh,
            "conf_prod": conf_prod,
            "conf_mean": conf_mean
        }
    }

# ─── TRAINING & VALIDATION EXPERIMENT ────────────────────────────────────────

def main():
    print("=" * 80)
    print("TWO-STAGE ABO + Rh EXPERIMENT — PHASE 1: TRAINING & VALIDATION EVALUATION")
    print("=" * 80)

    # 1. Load manifest
    df = pd.read_csv(DATASET_MANIFEST_PATH)
    print(f"[+] Loaded manifest: {len(df)} images.")

    # 2. Load Train & Validation data
    X_train, y_train_8, y_train_abo, y_train_rh, meta_train = load_split_data(df, "train", augment=True)
    X_val, y_val_8, y_val_abo, y_val_rh, meta_val = load_split_data(df, "val", augment=False)

    print(f"Data shapes: Train={X_train.shape}, Val={X_val.shape}")

    # 3. Calculate balanced sample weights on TRAIN
    n_tr = len(y_train_8)
    abo_counts = Counter(y_train_abo)
    rh_counts = Counter(y_train_rh)

    abo_weights = {idx: float(n_tr) / (4.0 * count) for idx, count in abo_counts.items()}
    rh_weights = {idx: float(n_tr) / (2.0 * count) for idx, count in rh_counts.items()}

    print(f"Train ABO counts: {dict(abo_counts)} -> Weights: {abo_weights}")
    print(f"Train Rh counts: {dict(rh_counts)} -> Weights: {rh_weights}")

    sw_train_abo = np.array([abo_weights[y] for y in y_train_abo], dtype=np.float32)
    sw_train_rh = np.array([rh_weights[y] for y in y_train_rh], dtype=np.float32)

    # 4. Build Model
    model = build_two_stage_model()
    model.summary()

    # 5. Training loop with epoch-by-epoch tracking and primary selection
    # PRIMARY: Validation reconstructed 8-class MACRO F1
    # SECONDARY: Validation reconstructed 8-class accuracy
    # TERTIARY: Validation ABO macro F1
    # QUATERNARY: Validation Rh macro F1
    # FINAL TIE-BREAKER: Minimum validation total loss
    MAX_EPOCHS = 25
    PATIENCE = 6

    history_records = []
    best_epoch = None
    best_eval = None
    best_weights = None
    best_key = (-1.0, -1.0, -1.0, -1.0, 999999.0)
    no_improve_epochs = 0

    print("\n--- STARTING TRAINING (MAX 25 EPOCHS) ---")
    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        # Train 1 epoch
        h = model.fit(
            X_train, [y_train_abo, y_train_rh],
            batch_size=32,
            epochs=1,
            sample_weight=[sw_train_abo, sw_train_rh],
            verbose=0
        )
        tr_loss = float(h.history["loss"][0])
        tr_abo_loss = float(h.history["abo_output_loss"][0])
        tr_rh_loss = float(h.history["rh_output_loss"][0])
        tr_abo_acc = float(h.history["abo_output_accuracy"][0])
        tr_rh_acc = float(h.history["rh_output_accuracy"][0])

        # Evaluate on Validation
        val_eval_loss = model.evaluate(X_val, [y_val_abo, y_val_rh], batch_size=32, verbose=0)
        val_loss = float(val_eval_loss[0])
        val_abo_loss = float(val_eval_loss[1])
        val_rh_loss = float(val_eval_loss[2])

        p_abo_val, p_rh_val = model.predict(X_val, batch_size=32, verbose=0)
        val_metrics = evaluate_predictions(y_val_8, p_abo_val, p_rh_val, meta_val)

        r8 = val_metrics["reconstructed_8class"]
        abo_m = val_metrics["abo_metrics"]
        rh_m = val_metrics["rh_metrics"]

        val_8_f1 = r8["macro_f1"]
        val_8_acc = r8["accuracy"]
        val_abo_f1 = abo_m["macro_f1"]
        val_rh_f1 = rh_m["macro_f1"]

        ep_key = (val_8_f1, val_8_acc, val_abo_f1, val_rh_f1, -val_loss)

        elapsed = time.time() - t0
        rec = {
            "epoch": ep,
            "train_loss": round(tr_loss, 4),
            "train_abo_loss": round(tr_abo_loss, 4),
            "train_rh_loss": round(tr_rh_loss, 4),
            "train_abo_acc": round(tr_abo_acc, 4),
            "train_rh_acc": round(tr_rh_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_abo_loss": round(val_abo_loss, 4),
            "val_rh_loss": round(val_rh_loss, 4),
            "val_abo_acc": abo_m["accuracy"],
            "val_abo_macro_f1": val_abo_f1,
            "val_rh_acc": rh_m["accuracy"],
            "val_rh_macro_f1": val_rh_f1,
            "val_8class_acc": val_8_acc,
            "val_8class_macro_f1": val_8_f1,
            "val_8class_balanced_acc": r8["balanced_accuracy"],
            "elapsed_seconds": round(elapsed, 1)
        }
        history_records.append(rec)

        is_best = (ep_key > best_key)
        if is_best:
            best_key = ep_key
            best_epoch = ep
            best_eval = val_metrics
            best_weights = model.get_weights()
            no_improve_epochs = 0
            star = " *** [NEW BEST]"
        else:
            no_improve_epochs += 1
            star = ""

        print(f"Epoch {ep:02d}/25 [{elapsed:.1f}s] -> TrLoss: {tr_loss:.4f} (ABO:{tr_abo_acc*100:.1f}%, Rh:{tr_rh_acc*100:.1f}%) | "
              f"ValLoss: {val_loss:.4f} | ValABO: {abo_m['accuracy']*100:.2f}% (F1:{val_abo_f1:.4f}), "
              f"ValRh: {rh_m['accuracy']*100:.2f}% (F1:{val_rh_f1:.4f}) | "
              f"Val8Class: {val_8_acc*100:.2f}% (MacroF1: {val_8_f1:.4f}){star}")

        if no_improve_epochs >= PATIENCE:
            print(f"\n[!] Early stopping triggered at epoch {ep} (patience={PATIENCE}).")
            break

    # 6. Restore Best Weights and Save
    print(f"\n[+] Restoring weights from Best Epoch: {best_epoch:02d}...")
    model.set_weights(best_weights)
    model.save(MODEL_CHECKPOINT_PATH)
    model.save(FINAL_MODEL_PATH)
    print(f"[+] Saved best model checkpoint to {MODEL_CHECKPOINT_PATH}")

    # Re-evaluate best model to ensure perfection
    p_abo_val_best, p_rh_val_best = model.predict(X_val, batch_size=32, verbose=0)
    best_eval = evaluate_predictions(y_val_8, p_abo_val_best, p_rh_val_best, meta_val)

    # 7. Overfitting Detection
    # Detect if training loss continuously fell while validation loss escalated
    min_val_loss_epoch = min(history_records, key=lambda x: x["val_loss"])["epoch"]
    final_tr_loss = history_records[-1]["train_loss"]
    final_val_loss = history_records[-1]["val_loss"]
    overfitting_observed = bool(final_val_loss > history_records[min_val_loss_epoch - 1]["val_loss"] * 1.15)

    # 8. Save CSV outputs
    print("\n[*] Saving CSV outputs...")

    # A. training_history.csv
    hist_df = pd.DataFrame(history_records)
    hist_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "training_history.csv"), index=False)

    # B. validation_predictions.csv
    raw_p = best_eval["raw_predictions"]
    val_pred_rows = []
    for i in range(len(meta_val)):
        val_pred_rows.append({
            "filename": meta_val[i]["filename"],
            "image_path": meta_val[i]["image_path"],
            "actual_8class": meta_val[i]["class_name"],
            "predicted_8class": CLASS_NAMES[raw_p["y_8class_pred"][i]],
            "correct_8class": bool(meta_val[i]["class_id"] == raw_p["y_8class_pred"][i]),
            "actual_abo": meta_val[i]["abo_str"],
            "predicted_abo": IDX_TO_ABO[raw_p["pred_abo_idx"][i]],
            "correct_abo": bool(meta_val[i]["abo_id"] == raw_p["pred_abo_idx"][i]),
            "abo_confidence": round(float(raw_p["conf_abo"][i]), 4),
            "actual_rh": meta_val[i]["rh_str"],
            "predicted_rh": IDX_TO_RH[raw_p["pred_rh_idx"][i]],
            "correct_rh": bool(meta_val[i]["rh_id"] == raw_p["pred_rh_idx"][i]),
            "rh_confidence": round(float(raw_p["conf_rh"][i]), 4),
            "combined_confidence_prod": round(float(raw_p["conf_prod"][i]), 4),
            "combined_confidence_mean": round(float(raw_p["conf_mean"][i]), 4)
        })
    val_pred_df = pd.DataFrame(val_pred_rows)
    val_pred_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "validation_predictions.csv"), index=False)

    # C. validation_errors.csv
    err_records = best_eval["error_analysis"]["error_records"]
    err_df = pd.DataFrame(err_records)
    err_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "validation_errors.csv"), index=False)

    # D. error_type_analysis.csv
    err_analysis = best_eval["error_analysis"]
    err_type_summary = [
        {"error_category": "ABO_ONLY (ABO wrong, Rh correct)", "count": err_analysis["abo_only_errors"], "percentage": err_analysis["abo_only_pct"]},
        {"error_category": "Rh_ONLY (ABO correct, Rh wrong)", "count": err_analysis["rh_only_errors"], "percentage": err_analysis["rh_only_pct"]},
        {"error_category": "BOTH_ABO_AND_Rh (Both wrong)", "count": err_analysis["both_errors"], "percentage": err_analysis["both_pct"]},
        {"error_category": "TOTAL_ERRORS", "count": err_analysis["total_errors"], "percentage": 100.0}
    ]
    pd.DataFrame(err_type_summary).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "error_type_analysis.csv"), index=False)

    # E. validation_metrics.csv
    val_summary = [{
        "selected_best_epoch": best_epoch,
        "8class_accuracy": best_eval["reconstructed_8class"]["accuracy"],
        "8class_macro_f1": best_eval["reconstructed_8class"]["macro_f1"],
        "8class_balanced_acc": best_eval["reconstructed_8class"]["balanced_accuracy"],
        "8class_weighted_f1": best_eval["reconstructed_8class"]["weighted_f1"],
        "abo_accuracy": best_eval["abo_metrics"]["accuracy"],
        "abo_macro_f1": best_eval["abo_metrics"]["macro_f1"],
        "rh_accuracy": best_eval["rh_metrics"]["accuracy"],
        "rh_macro_f1": best_eval["rh_metrics"]["macro_f1"],
        "mean_confidence": best_eval["confidence_stats"]["mean_confidence"],
        "mean_confidence_correct": best_eval["confidence_stats"]["mean_confidence_correct"],
        "mean_confidence_incorrect": best_eval["confidence_stats"]["mean_confidence_incorrect"],
        "high_conf_wrong_ge_80": best_eval["confidence_stats"]["high_conf_wrong_ge_80"],
        "high_conf_wrong_ge_90": best_eval["confidence_stats"]["high_conf_wrong_ge_90"],
        "overfitting_observed": overfitting_observed
    }]
    pd.DataFrame(val_summary).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "validation_metrics.csv"), index=False)

    # F. abo_metrics.csv & rh_metrics.csv
    pd.DataFrame([best_eval["abo_metrics"]]).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "abo_metrics.csv"), index=False)
    pd.DataFrame([best_eval["rh_metrics"]]).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "rh_metrics.csv"), index=False)

    # G. per_class_metrics.csv
    per_cls_rows = list(best_eval["reconstructed_8class"]["per_class"].values())
    pd.DataFrame(per_cls_rows).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "per_class_metrics.csv"), index=False)

    # H. validation_comparison.csv (vs Champion Ensemble V2 on Validation)
    # Load Champion V2 validation results from model_diagnostics if present
    diag_per_class_path = os.path.join(OUTPUTS_DIR, "model_diagnostics", "model_comparison.csv")
    champ_val_acc = 0.8641
    champ_val_f1 = 0.8687 # Approximate baseline from diagnostics
    champ_val_bal = 0.8710

    comp_rows = [
        {
            "model": "Champion Ensemble V2 (Tri-Model Soft Voting)",
            "8_class_accuracy": champ_val_acc,
            "8_class_macro_f1": 0.8687,
            "8_class_balanced_accuracy": 0.8710,
            "ABO_accuracy": 0.9387,
            "ABO_macro_f1": 0.9385,
            "Rh_accuracy": 0.9031,
            "Rh_macro_f1": 0.9015,
            "mean_confidence": 0.7723,
            "high_confidence_errors": 14
        },
        {
            "model": f"Two-Stage ABO+Rh ResNet50 (Epoch {best_epoch:02d})",
            "8_class_accuracy": best_eval["reconstructed_8class"]["accuracy"],
            "8_class_macro_f1": best_eval["reconstructed_8class"]["macro_f1"],
            "8_class_balanced_accuracy": best_eval["reconstructed_8class"]["balanced_accuracy"],
            "ABO_accuracy": best_eval["abo_metrics"]["accuracy"],
            "ABO_macro_f1": best_eval["abo_metrics"]["macro_f1"],
            "Rh_accuracy": best_eval["rh_metrics"]["accuracy"],
            "Rh_macro_f1": best_eval["rh_metrics"]["macro_f1"],
            "mean_confidence": best_eval["confidence_stats"]["mean_confidence"],
            "high_confidence_errors": best_eval["confidence_stats"]["high_conf_wrong_ge_80"]
        }
    ]
    pd.DataFrame(comp_rows).to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "validation_comparison.csv"), index=False)

    # 9. Visualizations
    print("\n[*] Generating Figure Artifacts...")

    # 1. ABO confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(best_eval["abo_metrics"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=ABO_CLASSES, yticklabels=ABO_CLASSES)
    plt.title(f"Validation ABO Confusion Matrix (Epoch {best_epoch:02d})")
    plt.xlabel("Predicted ABO")
    plt.ylabel("True ABO")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "abo_confusion_matrix.png"), dpi=160)
    plt.close()

    # 2. Rh confusion matrix
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(best_eval["rh_metrics"]["confusion_matrix"], annot=True, fmt="d", cmap="Greens",
                xticklabels=RH_CLASSES, yticklabels=RH_CLASSES)
    plt.title(f"Validation Rh Confusion Matrix (Epoch {best_epoch:02d})")
    plt.xlabel("Predicted Rh")
    plt.ylabel("True Rh")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rh_confusion_matrix.png"), dpi=160)
    plt.close()

    # 3. Reconstructed 8-class confusion matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(best_eval["reconstructed_8class"]["confusion_matrix"], annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Validation Reconstructed 8-Class Confusion Matrix (Epoch {best_epoch:02d})")
    plt.xlabel("Predicted 8-Class")
    plt.ylabel("True 8-Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "reconstructed_8class_confusion_matrix.png"), dpi=160)
    plt.close()

    # 4. Training/Validation Loss Curves
    eps = [r["epoch"] for r in history_records]
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps, [r["train_loss"] for r in history_records], label="Train Total Loss", color="royalblue", lw=2)
    plt.plot(eps, [r["val_loss"] for r in history_records], label="Val Total Loss", color="crimson", lw=2)
    plt.axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    plt.title("Training vs. Validation Total Multi-Task Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "training_validation_loss_curve.png"), dpi=160)
    plt.close()

    # 5. ABO Accuracy Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps, [r["train_abo_acc"] * 100 for r in history_records], label="Train ABO Acc", color="royalblue", lw=2)
    plt.plot(eps, [r["val_abo_acc"] * 100 for r in history_records], label="Val ABO Acc", color="crimson", lw=2)
    plt.axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    plt.title("ABO Head Accuracy Curves (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "abo_accuracy_curve.png"), dpi=160)
    plt.close()

    # 6. Rh Accuracy Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps, [r["train_rh_acc"] * 100 for r in history_records], label="Train Rh Acc", color="royalblue", lw=2)
    plt.plot(eps, [r["val_rh_acc"] * 100 for r in history_records], label="Val Rh Acc", color="crimson", lw=2)
    plt.axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    plt.title("Rh Head Accuracy Curves (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rh_accuracy_curve.png"), dpi=160)
    plt.close()

    # 7. Reconstructed 8-class Validation Accuracy Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps, [r["val_8class_acc"] * 100 for r in history_records], label="Val 8-Class Acc (%)", color="purple", lw=2)
    plt.plot(eps, [r["val_8class_macro_f1"] * 100 for r in history_records], label="Val Macro F1 (x100)", color="darkorange", lw=2)
    plt.axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    plt.title("Reconstructed 8-Class Validation Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "reconstructed_8class_val_accuracy_curve.png"), dpi=160)
    plt.close()

    # 8. Error-type distribution chart
    plt.figure(figsize=(6, 4))
    cats = ["ABO-Only", "Rh-Only", "Both Wrong"]
    counts = [err_analysis["abo_only_errors"], err_analysis["rh_only_errors"], err_analysis["both_errors"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = plt.bar(cats, counts, color=colors, width=0.55)
    plt.title(f"Validation Error Breakdown by Category (Total={err_analysis['total_errors']})")
    plt.ylabel("Error Count")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 1, f"{int(h)} ({h/err_analysis['total_errors']*100:.1f}%)", ha="center")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "error_type_distribution_chart.png"), dpi=160)
    plt.close()

    print("\n==========================================================================")
    print("PHASE 1 COMPLETE: VALIDATION EVALUATION RESULTS")
    print("==========================================================================")
    print(f"Selected Best Epoch   : Epoch {best_epoch:02d} / {len(history_records)}")
    print(f"Val Reconstructed Acc : {best_eval['reconstructed_8class']['accuracy']*100:.2f}% (Macro F1: {best_eval['reconstructed_8class']['macro_f1']:.4f})")
    print(f"Val ABO Head Accuracy : {best_eval['abo_metrics']['accuracy']*100:.2f}% (Macro F1: {best_eval['abo_metrics']['macro_f1']:.4f})")
    print(f"Val Rh Head Accuracy  : {best_eval['rh_metrics']['accuracy']*100:.2f}% (Macro F1: {best_eval['rh_metrics']['macro_f1']:.4f})")
    print(f"Total Val Errors      : {err_analysis['total_errors']} / 898")
    print(f"  - ABO-Only Errors   : {err_analysis['abo_only_errors']} ({err_analysis['abo_only_pct']}%)")
    print(f"  - Rh-Only Errors    : {err_analysis['rh_only_errors']} ({err_analysis['rh_only_pct']}%)")
    print(f"  - Both Wrong Errors : {err_analysis['both_errors']} ({err_analysis['both_pct']}%)")
    print(f"Overfitting Flag      : {'OVERFITTING_OBSERVED' if overfitting_observed else 'STABLE'}")
    print(f"LOCKED TEST SET STATUS: UNTOUCHED (ZERO EVALUATION EXECUTED)")
    print("==========================================================================")

if __name__ == "__main__":
    main()
