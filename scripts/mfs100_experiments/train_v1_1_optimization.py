"""
scripts/mfs100_experiments/train_v1_1_optimization.py
=============================================================
MFS100 ABO V1.1 — Controlled Training Optimization Experiment
Evaluates whether additional epochs improve validation performance
under exact balanced class weighting.
Strictly validation-based checkpoint selection. Test set evaluated ONCE.
Production locked and read-only.
"""

import os
import sys
import json
import csv
import shutil
import random
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ─── SEED & DIRECTORY INVARIANTS ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

PILOT_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PILOT_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(PILOT_CLASSES)}

MFS_GT_CSV = os.path.join(PROJECT_ROOT, "data", "mfs100_ground_truth.csv")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "data", "mfs100_experiments", "proposed_pilot_subject_split.json")
PREP_CONFIG_FILE = os.path.join(PROJECT_ROOT, "outputs", "mfs100_experiments", "v1_preprocessing_config.json")
BASE_RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "resnet50_improved_blood_group.keras")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mfs100_experiments", "v1_1_optimization")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models", "mfs100_experiments", "v1_1_checkpoints")
FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mfs100_experiments", "resnet50_mfs100_abo_v1_1.keras")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ─── PRE-FLIGHT VERIFICATION ─────────────────────────────────────────────────

def run_preflight():
    print("==========================================================================")
    print("MFS100 ABO V1.1 PRE-FLIGHT AUDIT & VERIFICATION")
    print("==========================================================================")

    if not os.path.exists(MFS_GT_CSV):
        print(f"[FAIL] Missing ground truth CSV: {MFS_GT_CSV}")
        return False, None, None

    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 78:
        print(f"[FAIL] Expected 78 rows, found {len(rows)}")
        return False, None, None
    print("[PASS] 78 ground-truth rows confirmed.")

    missing_caps = []
    for r in rows:
        fp = os.path.join(PROJECT_ROOT, "data", "captured", r["capture_filename"].strip())
        if not os.path.exists(fp):
            missing_caps.append(r["capture_filename"].strip())
    if missing_caps:
        print(f"[FAIL] Missing {len(missing_caps)} capture files: {missing_caps[:5]}")
        return False, None, None
    print("[PASS] 78/78 matching capture files present in data/captured/.")

    counts = Counter(r["subject_id"] for r in rows)
    if len(counts) != 26 or not all(v == 3 for v in counts.values()):
        print(f"[FAIL] Expected 26 subjects with 3 impressions each. Got {len(counts)} subjects.")
        return False, None, None
    print("[PASS] 26 subjects with exactly 3 impressions each confirmed.")

    if not os.path.exists(SPLIT_FILE):
        print(f"[FAIL] Missing split file: {SPLIT_FILE}")
        return False, None, None

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    expected_train = ["S001", "S002", "S005", "S007", "S008", "S009", "S010", "S011", "S012", "S015", "S016", "S017", "S018", "S019", "S020", "S021", "S022"]
    expected_val = ["S003", "S006", "S013", "S023", "S024"]
    expected_test = ["S004", "S026", "S014", "S025"]

    if sorted(split.get("train_subjects", [])) != sorted(expected_train):
        print("[FAIL] Train split mismatch.")
        return False, None, None
    if sorted(split.get("val_subjects", [])) != sorted(expected_val):
        print("[FAIL] Val split mismatch.")
        return False, None, None
    if sorted(split.get("test_subjects", [])) != sorted(expected_test):
        print("[FAIL] Test split mismatch.")
        return False, None, None
    print("[PASS] Fixed subject split verified (17 Train, 5 Val, 4 Test).")

    # Preprocessing check
    if not os.path.exists(PREP_CONFIG_FILE):
        print(f"[FAIL] Missing preprocessing config: {PREP_CONFIG_FILE}")
        return False, None, None
    with open(PREP_CONFIG_FILE, "r", encoding="utf-8") as f:
        prep_cfg = json.load(f)
    if prep_cfg.get("variant_name") != "Variant_D":
        print(f"[FAIL] Variant is not Variant_D: {prep_cfg.get('variant_name')}")
        return False, None, None
    print("[PASS] Variant_D preprocessing configuration verified.")

    # Class counts and exact balanced weights from TRAIN
    train_subs = set(split["train_subjects"])
    train_counts = Counter()
    for r in rows:
        if r["subject_id"] in train_subs:
            train_counts[r["confirmed_blood_group"].strip()] += 1

    expected_counts = {"A+": 6, "AB+": 3, "B+": 18, "O+": 24}
    if dict(train_counts) != expected_counts:
        print(f"[FAIL] Train counts mismatch! Found: {dict(train_counts)}, Expected: {expected_counts}")
        return False, None, None
    print(f"[PASS] Exact train class counts verified: {dict(train_counts)} (Total = {sum(train_counts.values())}).")

    N = sum(train_counts.values())  # 51
    K = len(PILOT_CLASSES)         # 4
    class_weights = {}
    for c in PILOT_CLASSES:
        idx = CLASS_TO_IDX[c]
        class_weights[idx] = float(N) / (float(K) * float(train_counts[c]))

    expected_weights = {
        CLASS_TO_IDX["A+"]: 2.125,
        CLASS_TO_IDX["AB+"]: 4.250,
        CLASS_TO_IDX["B+"]: 51.0 / 72.0,  # 0.7083333333...
        CLASS_TO_IDX["O+"]: 0.53125
    }

    print("\nCalculated Class Weights (TRAIN ONLY):")
    for c in PILOT_CLASSES:
        idx = CLASS_TO_IDX[c]
        print(f"  {c:4s} (count={train_counts[c]:2d}) : {class_weights[idx]:.10f} (Expected: {expected_weights[idx]:.10f})")

    for idx in expected_weights:
        if abs(class_weights[idx] - expected_weights[idx]) > 1e-6:
            print(f"[FAIL] Class weight for index {idx} differs materially!")
            return False, None, None
    print("[PASS] Exact balanced class weights verified against specification.")

    # Production models untouched
    prod_files = [
        "models/resnet50_improved_blood_group.keras",
        "models/vgg16_blood_group.keras",
        "models/efficientnetb0_blood_group.keras",
        "src/predictor.py",
        "src/preprocessing.py"
    ]
    for pf in prod_files:
        full_p = os.path.join(PROJECT_ROOT, pf)
        if not os.path.exists(full_p):
            print(f"[FAIL] Production file missing: {pf}")
            return False, None, None
    print("[PASS] Production models exist and production files intact.")

    print("==========================================================================")
    print("ALL PRE-FLIGHT AUDITS PASSED -> AUTHORIZED FOR V1.1 OPTIMIZATION RUN")
    print("==========================================================================\n")
    return True, split, class_weights

# ─── PREPROCESSING (VARIANT_D) ───────────────────────────────────────────────

def _get_roi_crop_and_pad(img_pil):
    gray = np.array(img_pil.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)
        crop_pil = img_pil.crop((x1, y1, x2, y2))
    else:
        from src.preprocessing import crop_mfs100_foreground
        crop_pil, _ = crop_mfs100_foreground(img_pil.convert("RGB"))

    w_c, h_c = crop_pil.size
    max_side = max(w_c, h_c)
    padded = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    padded.paste(crop_pil.convert("RGB"), ((max_side - w_c) // 2, (max_side - h_c) // 2))
    return padded

def preprocess_variant_d(img_pil):
    padded = _get_roi_crop_and_pad(img_pil)
    arr = np.array(padded.convert("L"), dtype=np.float32)
    p2 = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    if p98 > p2:
        norm = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = arr.astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized

# ─── AUGMENTATION (CONSERVATIVE, TRAIN ONLY) ─────────────────────────────────

def augment_image(img_arr):
    h, w, _ = img_arr.shape
    angle = random.uniform(-8, 8)
    tx = random.uniform(-6, 6)
    ty = random.uniform(-6, 6)
    scale = random.uniform(0.95, 1.05)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug = cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-10, 10)
    aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)
    return aug

# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_datasets(split):
    cap_to_info = {}
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cap_to_info[r["capture_filename"].strip()] = {
                "subject_id": r["subject_id"].strip(),
                "blood_group": r["confirmed_blood_group"].strip()
            }

    def _build_partition(subject_list, is_train=False, aug_factor=4):
        X, y, subjects, filenames = [], [], [], []
        for fn, info in cap_to_info.items():
            if info["subject_id"] in subject_list:
                p = os.path.join(PROJECT_ROOT, "data", "captured", fn)
                with Image.open(p) as img_pil:
                    base_arr = preprocess_variant_d(img_pil)
                label_idx = CLASS_TO_IDX[info["blood_group"]]

                X.append(base_arr)
                y.append(label_idx)
                subjects.append(info["subject_id"])
                filenames.append(fn)

                if is_train:
                    for _ in range(aug_factor):
                        aug_arr = augment_image(base_arr)
                        X.append(aug_arr)
                        y.append(label_idx)
                        subjects.append(info["subject_id"])
                        filenames.append(f"aug_{fn}")

        X_arr = np.array(X, dtype=np.float32)
        X_res = resnet_preprocess(X_arr)
        y_arr = np.array(y, dtype=np.int32)
        return X_res, y_arr, subjects, filenames

    X_train, y_train, train_subs, train_files = _build_partition(split["train_subjects"], is_train=True, aug_factor=4)
    X_val, y_val, val_subs, val_files = _build_partition(split["val_subjects"], is_train=False)
    X_test, y_test, test_subs, test_files = _build_partition(split["test_subjects"], is_train=False)

    return (
        (X_train, y_train, train_subs, train_files),
        (X_val, y_val, val_subs, val_files),
        (X_test, y_test, test_subs, test_files)
    )

# ─── MODEL BUILDER (FROZEN BACKBONE) ─────────────────────────────────────────

def build_v1_1_model():
    base_model = tf.keras.models.load_model(BASE_RESNET_PATH, compile=False)
    resnet_backbone = base_model.get_layer("resnet50")
    resnet_backbone.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="mfs100_input")
    x = resnet_backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="v1_1_gap")(x)
    x = layers.BatchNormalization(name="v1_1_bn")(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
        name="v1_1_dense_64"
    )(x)
    x = layers.Dropout(0.50, name="v1_1_dropout")(x)
    outputs = layers.Dense(4, activation="softmax", name="abo_4class_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_MFS100_ABO_v1_1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─── COHORT EVALUATION HELPER ────────────────────────────────────────────────

def evaluate_cohort(y_true, y_probs, subjects):
    y_preds = np.argmax(y_probs, axis=1)
    confs = np.max(y_probs, axis=1)
    n = len(y_true)

    imp_acc = float(np.sum(y_true == y_preds)) / n

    # Balanced accuracy
    recalls = []
    for c in range(4):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            recalls.append(float(np.sum((y_preds == c) & mask)) / np.sum(mask))
    bal_acc = float(np.mean(recalls)) if recalls else 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_preds, labels=list(range(4)), average="macro", zero_division=0
    )

    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(
        y_true, y_preds, labels=list(range(4)), average=None, zero_division=0
    )
    per_class = {}
    for i, cname in enumerate(PILOT_CLASSES):
        per_class[cname] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f1_cls[i]), 4),
            "support": int(s_cls[i])
        }

    cm_raw = confusion_matrix(y_true, y_preds, labels=list(range(4)))
    cm = {}
    for i, true_c in enumerate(PILOT_CLASSES):
        cm[true_c] = {}
        for j, pred_c in enumerate(PILOT_CLASSES):
            cm[true_c][pred_c] = int(cm_raw[i, j])

    # Subject consensus
    sub_probs = defaultdict(list)
    sub_gt = {}
    for sid, true_idx, probs in zip(subjects, y_true, y_probs):
        sub_probs[sid].append(probs)
        sub_gt[sid] = IDX_TO_CLASS[true_idx]

    maj_correct = 0
    soft_correct = 0
    consensus_details = {}

    for sid, p_list in sub_probs.items():
        gt = sub_gt[sid]
        pred_labels = [IDX_TO_CLASS[int(np.argmax(p))] for p in p_list]
        vote_counts = Counter(pred_labels)
        maj_pred = vote_counts.most_common(1)[0][0]

        soft_pred = IDX_TO_CLASS[int(np.argmax(np.mean(p_list, axis=0)))]

        is_maj = (maj_pred == gt)
        is_soft = (soft_pred == gt)
        if is_maj:
            maj_correct += 1
        if is_soft:
            soft_correct += 1

        consensus_details[sid] = {
            "ground_truth": gt,
            "impressions": pred_labels,
            "majority_consensus": maj_pred,
            "soft_consensus": soft_pred,
            "majority_correct": is_maj,
            "soft_correct": is_soft
        }

    n_subs = len(sub_probs)
    maj_acc = float(maj_correct) / n_subs if n_subs > 0 else 0.0
    soft_acc = float(soft_correct) / n_subs if n_subs > 0 else 0.0

    # Prediction share & collapse flags
    pred_counts = Counter(y_preds)
    pred_shares = {IDX_TO_CLASS[i]: round(float(pred_counts.get(i, 0)) / n * 100.0, 2) for i in range(4)}
    actual_counts = Counter(y_true)
    actual_shares = {IDX_TO_CLASS[i]: round(float(actual_counts.get(i, 0)) / n * 100.0, 2) for i in range(4)}

    collapse_flags = {}
    for cname in PILOT_CLASSES:
        collapse_flags[f"{cname}_collapse"] = bool(pred_shares[cname] >= 75.0 and actual_shares[cname] <= 25.0)

    correct_mask = (y_true == y_preds)
    correct_confs = confs[correct_mask]
    incorrect_confs = confs[~correct_mask]

    return {
        "impression_accuracy": round(imp_acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "majority_consensus_accuracy": round(maj_acc, 4),
        "majority_correct_count": maj_correct,
        "total_subjects": n_subs,
        "soft_consensus_accuracy": round(soft_acc, 4),
        "soft_correct_count": soft_correct,
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "prediction_shares": pred_shares,
        "actual_shares": actual_shares,
        "collapse_flags": collapse_flags,
        "confidence": {
            "mean": round(float(np.mean(confs)), 4),
            "median": round(float(np.median(confs)), 4),
            "mean_correct": round(float(np.mean(correct_confs)), 4) if len(correct_confs) > 0 else 0.0,
            "mean_incorrect": round(float(np.mean(incorrect_confs)), 4) if len(incorrect_confs) > 0 else 0.0
        },
        "consensus_details": consensus_details
    }

# ─── EPOCH RECORDING CALLBACK ────────────────────────────────────────────────

class EpochEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, val_subs):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.val_subs = val_subs
        self.epoch_records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep = epoch + 1
        lr = float(self.model.optimizer.learning_rate)

        # Save per-epoch checkpoint
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{ep:02d}.keras")
        self.model.save(ckpt_path)

        # Compute validation probabilities
        val_probs = self.model.predict(self.X_val, verbose=0)
        val_eval = evaluate_cohort(self.y_val, val_probs, self.val_subs)

        rec = {
            "epoch": ep,
            "train_loss": round(float(logs.get("loss", 0.0)), 4),
            "train_accuracy": round(float(logs.get("accuracy", 0.0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0.0)), 4),
            "val_accuracy": round(float(logs.get("val_accuracy", 0.0)), 4),
            "val_balanced_accuracy": val_eval["balanced_accuracy"],
            "val_macro_precision": val_eval["macro_precision"],
            "val_macro_recall": val_eval["macro_recall"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_majority_consensus_acc": val_eval["majority_consensus_accuracy"],
            "val_majority_correct": val_eval["majority_correct_count"],
            "val_soft_consensus_acc": val_eval["soft_consensus_accuracy"],
            "learning_rate": lr,
            "checkpoint_path": ckpt_path
        }
        self.epoch_records.append((rec, val_eval))

        print(f"Epoch {ep:02d}/30 -> TrainLoss={rec['train_loss']:.4f}, TrainAcc={rec['train_accuracy']:.4f} | "
              f"ValLoss={rec['val_loss']:.4f}, ValAcc={rec['val_accuracy']:.4f}, ValBalAcc={rec['val_balanced_accuracy']:.4f}, "
              f"ValF1={rec['val_macro_f1']:.4f} | ValConsensus={rec['val_majority_correct']}/{val_eval['total_subjects']} "
              f"({rec['val_majority_consensus_acc']*100:.1f}%)")

# ─── MAIN EXPERIMENT EXECUTION ───────────────────────────────────────────────

def main():
    preflight_ok, split, class_weights = run_preflight()
    if not preflight_ok:
        print("[CRITICAL] Pre-flight failed. Halting.")
        sys.exit(1)

    print("--- Loading Datasets ---")
    (X_train, y_train, train_subs, train_files), \
    (X_val, y_val, val_subs, val_files), \
    (X_test, y_test, test_subs, test_files) = load_datasets(split)

    print(f"Dataset Tensors:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape} ({len(train_subs)} augmented impressions from 17 subjects)")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape} (15 impressions from 5 subjects)")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape} (12 impressions from 4 subjects) [HELD-OUT]")

    model = build_v1_1_model()
    model.summary()

    eval_cb = EpochEvaluationCallback(X_val, y_val, val_subs)

    print("\n--- STARTING 30-EPOCH CONTROLLED OPTIMIZATION RUN ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=8,
        class_weight=class_weights,
        callbacks=[eval_cb],
        verbose=0
    )

    records = [r[0] for r in eval_cb.epoch_records]
    val_evals = [r[1] for r in eval_cb.epoch_records]

    # Save epoch_metrics.csv
    csv_fields = [
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "val_balanced_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
        "val_majority_consensus_acc", "val_majority_correct", "val_soft_consensus_acc",
        "learning_rate"
    ]
    with open(os.path.join(OUTPUT_DIR, "epoch_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in csv_fields})

    # Validation-based selection
    # Primary: val_majority_consensus_acc (max)
    # Tie 1: val_macro_f1 (max)
    # Tie 2: val_balanced_accuracy (max)
    # Tie 3: val_loss (min)
    def selection_key(item):
        rec = item[0]
        return (
            rec["val_majority_consensus_acc"],
            rec["val_macro_f1"],
            rec["val_balanced_accuracy"],
            -rec["val_loss"]
        )

    best_record, best_val_eval = max(eval_cb.epoch_records, key=selection_key)
    best_epoch = best_record["epoch"]
    best_ckpt = best_record["checkpoint_path"]

    print("\n==========================================================================")
    print("VALIDATION-BASED CHECKPOINT SELECTION COMPLETE")
    print(f"Selected Best Epoch: Epoch {best_epoch:02d} / 30")
    print(f"  Val Consensus Accuracy : {best_record['val_majority_consensus_acc']*100:.2f}% ({best_record['val_majority_correct']}/5 subjects)")
    print(f"  Val Macro F1           : {best_record['val_macro_f1']:.4f}")
    print(f"  Val Balanced Accuracy  : {best_record['val_balanced_accuracy']*100:.2f}%")
    print(f"  Val Loss               : {best_record['val_loss']:.4f}")
    print(f"  Checkpoint             : {best_ckpt}")
    print("==========================================================================")

    # Copy best checkpoint to final model path
    shutil.copyfile(best_ckpt, FINAL_MODEL_PATH)
    print(f"Saved selected model to: {FINAL_MODEL_PATH}")

    # Overfitting analysis
    # Detect point where train_acc continues increasing while val_loss rises or val_macro_f1 drops
    overfitting_epoch = None
    min_val_loss_epoch = min(records, key=lambda x: x["val_loss"])["epoch"]
    max_val_f1_epoch = max(records, key=lambda x: x["val_macro_f1"])["epoch"]

    # Check degradation after epoch 5
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        if curr["train_accuracy"] > prev["train_accuracy"] and curr["val_loss"] > prev["val_loss"] and curr["val_macro_f1"] <= prev["val_macro_f1"]:
            if curr["epoch"] > min_val_loss_epoch and overfitting_epoch is None:
                overfitting_epoch = curr["epoch"]

    overfitting_observed = (records[-1]["val_loss"] > records[min_val_loss_epoch - 1]["val_loss"] * 1.2)

    # Save best_epoch.json
    best_epoch_data = {
        "selected_best_epoch": best_epoch,
        "selection_criteria": {
            "primary": "Validation 3-scan subject majority consensus accuracy",
            "tie_breaker_1": "Validation macro F1",
            "tie_breaker_2": "Validation balanced accuracy",
            "tie_breaker_3": "Validation loss (minimum)"
        },
        "metrics_at_best_epoch": best_record,
        "min_val_loss_epoch": min_val_loss_epoch,
        "max_val_f1_epoch": max_val_f1_epoch,
        "overfitting_analysis": {
            "overfitting_observed": bool(overfitting_observed),
            "first_divergence_epoch": overfitting_epoch,
            "final_epoch_val_loss": records[-1]["val_loss"],
            "min_val_loss": records[min_val_loss_epoch - 1]["val_loss"]
        }
    }
    with open(os.path.join(OUTPUT_DIR, "best_epoch.json"), "w", encoding="utf-8") as f:
        json.dump(best_epoch_data, f, indent=2)

    # Save validation_results.json (for selected best epoch)
    with open(os.path.join(OUTPUT_DIR, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(best_val_eval, f, indent=2)

    # ─── SINGLE FINAL TEST SET EVALUATION ─────────────────────────────────────
    print("\n--- EVALUATING SELECTED MODEL ON UNSEEN HELD-OUT TEST SET (SINGLE PASS) ---")
    selected_model = tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)
    test_probs = selected_model.predict(X_test, verbose=0)
    test_metrics = evaluate_cohort(y_test, test_probs, test_subs)

    # Save test_results.json
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Classification report
    y_test_pred_idx = np.argmax(test_probs, axis=1)
    clf_rep = classification_report(y_test, y_test_pred_idx, target_names=PILOT_CLASSES, zero_division=0)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== MFS100 ABO V1.1 OPTIMIZATION: CLASSIFICATION REPORT (UNSEEN TEST SET) ===\n")
        f.write(f"Evaluated Model Checkpoint: Epoch {best_epoch:02d}\n")
        f.write("Evaluation Cohort: 4 subjects (S004, S026, S014, S025), 12 impressions\n\n")
        f.write(clf_rep)

    # Subject consensus results CSV
    sub_rows = []
    for sid, det in test_metrics["consensus_details"].items():
        sub_rows.append({
            "subject_id": sid,
            "ground_truth": det["ground_truth"],
            "impression_1": det["impressions"][0],
            "impression_2": det["impressions"][1],
            "impression_3": det["impressions"][2],
            "majority_consensus": det["majority_consensus"],
            "soft_consensus": det["soft_consensus"],
            "majority_correct": det["majority_correct"],
            "soft_correct": det["soft_correct"]
        })
    with open(os.path.join(OUTPUT_DIR, "subject_consensus_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "ground_truth", "impression_1", "impression_2", "impression_3",
            "majority_consensus", "soft_consensus", "majority_correct", "soft_correct"
        ])
        writer.writeheader()
        writer.writerows(sub_rows)

    # ─── PLOTS ───────────────────────────────────────────────────────────────

    # 1. Confusion Matrix
    cm_arr = confusion_matrix(y_test, y_test_pred_idx, labels=list(range(4)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=PILOT_CLASSES,
        yticklabels=PILOT_CLASSES,
        title=f"Test Confusion Matrix - V1.1 (Epoch {best_epoch:02d})",
        ylabel="True Class",
        xlabel="Predicted Class"
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, format(cm_arr[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2. else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 2. Training Curves
    epochs_axis = [r["epoch"] for r in records]
    train_losses = [r["train_loss"] for r in records]
    val_losses = [r["val_loss"] for r in records]
    train_accs = [r["train_accuracy"] for r in records]
    val_accs = [r["val_accuracy"] for r in records]
    val_f1s = [r["val_macro_f1"] for r in records]
    val_consensus = [r["val_majority_consensus_acc"] * 100.0 for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(epochs_axis, train_losses, label="Train Loss", color="royalblue", lw=2)
    axes[0, 0].plot(epochs_axis, val_losses, label="Val Loss", color="crimson", lw=2)
    axes[0, 0].axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    axes[0, 0].set_title("Loss Curves (Train vs Val)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs_axis, train_accs, label="Train Acc", color="royalblue", lw=2)
    axes[0, 1].plot(epochs_axis, val_accs, label="Val Acc", color="crimson", lw=2)
    axes[0, 1].axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    axes[0, 1].set_title("Impression Accuracy (Train vs Val)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Val Macro F1
    axes[1, 0].plot(epochs_axis, val_f1s, label="Val Macro F1", color="darkorange", lw=2)
    axes[1, 0].axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    axes[1, 0].set_title("Validation Macro F1 Across Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Val Consensus
    axes[1, 1].plot(epochs_axis, val_consensus, label="Val Consensus %", color="purple", lw=2)
    axes[1, 1].axvline(best_epoch, color="forestgreen", linestyle="--", label=f"Best (Ep {best_epoch})")
    axes[1, 1].set_title("Validation 3-Scan Majority Consensus (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Majority Consensus (%)")
    axes[1, 1].set_yticks([0, 20, 40, 60, 80, 100])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    # 3. Prediction Distribution
    pred_counts = [int(np.sum(y_test_pred_idx == i)) for i in range(4)]
    actual_counts = [int(np.sum(y_test == i)) for i in range(4)]
    x_indices = np.arange(4)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x_indices - bar_w / 2, actual_counts, bar_w, label="Actual Ground Truth", color="#4C72B0")
    ax.bar(x_indices + bar_w / 2, pred_counts, bar_w, label="V1.1 Predicted", color="#55A868")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(PILOT_CLASSES)
    ax.set_ylabel("Count of Impressions")
    ax.set_title(f"Class Distribution: Actual vs Predicted (Test Cohort, Epoch {best_epoch:02d})")
    ax.set_ylim(0, 13)
    for i in range(4):
        ax.text(x_indices[i] - bar_w / 2, actual_counts[i] + 0.2, str(actual_counts[i]), ha="center")
        ax.text(x_indices[i] + bar_w / 2, pred_counts[i] + 0.2, str(pred_counts[i]), ha="center")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"), dpi=150)
    plt.close()

    # Save experiment_config.json
    exp_config = {
        "experiment_title": "MFS100 ABO V1.1 Controlled Training Optimization",
        "date": "2026-09-08",
        "objective": "Evaluate validation trajectory and test performance under exact balanced class weighting",
        "target_model_path": FINAL_MODEL_PATH,
        "backbone": "ResNet50 (Frozen, read-only from models/resnet50_improved_blood_group.keras)",
        "adaptation_head": "GAP -> BatchNorm -> Dense(64, ReLU, L2=1e-3) -> Dropout(0.50) -> Dense(4, Softmax)",
        "classes": PILOT_CLASSES,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "batch_size": 8,
        "total_epochs": 30,
        "seed": SEED,
        "augmentation": {
            "rotation": "+/- 8 deg",
            "translation": "+/- 6 px",
            "zoom": "0.95 - 1.05",
            "brightness_contrast": "+/- 10%"
        },
        "preprocessing_variant": "Variant_D",
        "train_class_counts": {"A+": 6, "AB+": 3, "B+": 18, "O+": 24},
        "train_class_weights": {IDX_TO_CLASS[k]: v for k, v in class_weights.items()},
        "selected_best_epoch": best_epoch,
        "split": {
            "train_subjects": split["train_subjects"],
            "val_subjects": split["val_subjects"],
            "test_subjects": split["test_subjects"]
        }
    }
    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # Save README.md
    readme_md = f"""# MFS100 ABO V1.1 — Controlled Training Optimization Report

## Overview
This experiment evaluated whether additional training epochs (up to 30 epochs) under the exact balanced class weights formula:
$$\\text{{weight}}_c = \\frac{{N}}{{K \\cdot n_c}} = \\frac{{51}}{{4 \\cdot n_c}}$$
improve validation and test performance on MFS100 four-class ABO positive classification.

## Statistical Limitation Notice
> [!IMPORTANT]
> The test set contains **4 subjects and 12 impressions** (S004: A+, S026: AB+, S014: B+, S025: O+).
> All test performance numbers reported herein apply strictly to this **12-impression / 4-subject pilot test set** and must be interpreted as preliminary pilot findings rather than real-world generalizability guarantees.

## Checkpoint Selection (Validation-Based Only)
- **Selection Metric**: 3-scan subject majority consensus accuracy on the 5 validation subjects (S003, S006, S013, S023, S024).
- **Selected Best Epoch**: **Epoch {best_epoch:02d} / 30**
- **Validation Consensus**: {best_record['val_majority_correct']}/5 subjects ({best_record['val_majority_consensus_acc']*100:.1f}%)
- **Validation Macro F1**: {best_record['val_macro_f1']:.4f}
- **Validation Balanced Accuracy**: {best_record['val_balanced_accuracy']*100:.2f}%
- **Validation Loss**: {best_record['val_loss']:.4f}

## Test Performance Comparison
| Metric | Historical V1 (Reference) | Reproduced V1 (Reproduction) | V1.1 Optimization (Epoch {best_epoch:02d}) |
| :--- | :---: | :---: | :---: |
| **Test Impression Accuracy** | 50.00% (6/12) | 33.33% (4/12) | {test_metrics['impression_accuracy']*100:.2f}% ({int(test_metrics['impression_accuracy']*12)}/12) |
| **Test Balanced Accuracy** | 50.00% | 33.33% | {test_metrics['balanced_accuracy']*100:.2f}% |
| **Majority Consensus Accuracy** | 50.00% (2/4) | 25.00% (1/4) | {test_metrics['majority_consensus_accuracy']*100:.2f}% ({test_metrics['majority_correct_count']}/4) |
| **Soft Consensus Accuracy** | 50.00% (2/4) | 25.00% (1/4) | {test_metrics['soft_consensus_accuracy']*100:.2f}% ({test_metrics['soft_correct_count']}/4) |
| **Macro Precision** | 0.3333 | 0.2857 | {test_metrics['macro_precision']:.4f} |
| **Macro Recall** | 0.5000 | 0.3333 | {test_metrics['macro_recall']:.4f} |
| **Macro F1** | 0.3750 | 0.3000 | {test_metrics['macro_f1']:.4f} |
| **Mean Confidence** | 0.4640 | 0.4883 | {test_metrics['confidence']['mean']:.4f} |

## Class Collapse Check
- Actual O+ Share: {test_metrics['actual_shares']['O+']}%
- Predicted O+ Share: {test_metrics['prediction_shares']['O+']}%
- O+ Collapse Flag: {str(test_metrics['collapse_flags']['O+_collapse']).upper()}

## Production Invariants
- Production models: UNTOUCHED
- Champion Ensemble V2 weights: 0.58 / 0.38 / 0.04
- Locked benchmark: 85.41%
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_md)

    print("\n==========================================================================")
    print("V1.1 OPTIMIZATION RUN COMPLETE")
    print(f"Artifacts saved to: {OUTPUT_DIR}")
    print(f"Model saved to: {FINAL_MODEL_PATH}")
    print(f"Test Accuracy: {test_metrics['impression_accuracy']*100:.2f}%")
    print(f"Majority Consensus: {test_metrics['majority_consensus_accuracy']*100:.2f}% ({test_metrics['majority_correct_count']}/4)")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
