"""
scripts/mfs100_experiments/run_mfs100_abo_adaptation_pilot.py
=============================================================
Authoritative orchestrator for the MFS100 Four-Class ABO-Positive Adaptation Pilot.

Protocol:
  1. Load approved subject-level split:
     - TRAIN (17 subjects, 51 impressions)
     - VALIDATION (5 subjects, 15 impressions)
     - UNSEEN TEST (4 subjects, 12 impressions)
  2. Implement Preprocessing Variants (A, B, C, D, E).
  3. Preprocessing Variant Selection on VALIDATION subjects only:
     - Select best variant based on validation cross-entropy / accuracy.
  4. Train Experimental Model resnet50_mfs100_abo_v1.keras:
     - Pretrained ResNet50 backbone (100% frozen)
     - GAP -> Dense(64, ReLU, L2=1e-3) -> Dropout(0.5) -> Dense(4, Softmax)
     - Target classes: A+, AB+, B+, O+
     - Conservative augmentation on training set only (rotation, translation, zoom, contrast)
     - EarlyStopping + ReduceLROnPlateau
  5. Baseline Evaluation on the UNSEEN TEST subjects:
     - Champion Ensemble V2 (re-normalized 4-class output)
  6. Adapted Model Evaluation on the UNSEEN TEST subjects:
     - Impression-level accuracy & Balanced Accuracy
     - 3-Impression Majority-Vote Consensus
     - 3-Impression Average-Probability Consensus
     - Precision, Recall, F1 (Macro & Per-Class)
     - Confusion Matrix
     - O+ Prediction Bias Analysis
     - Confidence Calibration Analysis
  7. Persist all outputs to outputs/mfs100_experiments/ and models/mfs100_experiments/
"""

import os
import sys
import json
import csv
import random
from collections import Counter, defaultdict

import numpy as np
import cv2
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import crop_mfs100_foreground

PILOT_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PILOT_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(PILOT_CLASSES)}

MFS_GT_CSV = os.path.join(PROJECT_ROOT, "data", "mfs100_ground_truth.csv")
SPLIT_FILE = os.path.join(PROJECT_ROOT, "data", "mfs100_experiments", "proposed_pilot_subject_split.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mfs100_experiments")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "mfs100_experiments")
BASE_RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "resnet50_improved_blood_group.keras")
TARGET_V1_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_mfs100_abo_v1.keras")

# ─── PREPROCESSING VARIANTS ───────────────────────────────────────────────────

def preprocess_variant_a(img_pil):
    """Variant A: Current production MFS100 preprocessing (crop bbox + bilinear resize 224x224)."""
    cropped, _ = crop_mfs100_foreground(img_pil.convert("RGB"))
    resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)

def _get_roi_crop_and_pad(img_pil):
    """Helper: Foreground/ROI crop + aspect ratio square padding."""
    gray = np.array(img_pil.convert("L"))
    # Otsu thresholding for fingerprint contact area
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find bounding box
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Pad slightly
        pad_x, pad_y = int(w * 0.05), int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)
        crop_pil = img_pil.crop((x1, y1, x2, y2))
    else:
        crop_pil, _ = crop_mfs100_foreground(img_pil.convert("RGB"))

    # Square pad preserving aspect ratio
    w_c, h_c = crop_pil.size
    max_side = max(w_c, h_c)
    padded = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    padded.paste(crop_pil.convert("RGB"), ((max_side - w_c) // 2, (max_side - h_c) // 2))
    return padded

def preprocess_variant_b(img_pil):
    """Variant B: ROI crop + padding + resize 224x224."""
    padded = _get_roi_crop_and_pad(img_pil)
    resized = padded.resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)

def preprocess_variant_c(img_pil):
    """Variant C: ROI crop + padding + CLAHE + resize 224x224."""
    padded = _get_roi_crop_and_pad(img_pil)
    gray = np.array(padded.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    rgb = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized

def preprocess_variant_d(img_pil):
    """Variant D: ROI crop + padding + controlled contrast normalization (p2 to p98) + resize."""
    padded = _get_roi_crop_and_pad(img_pil)
    gray = np.array(padded.convert("L"), dtype=np.float32)
    p2, p98 = np.percentile(gray, [2.0, 98.0])
    if p98 > p2:
        norm = np.clip((gray - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = gray.astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized

def preprocess_variant_e(img_pil):
    """Variant E: ROI crop + padding + CLAHE + controlled contrast normalization + resize."""
    padded = _get_roi_crop_and_pad(img_pil)
    gray = np.array(padded.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray).astype(np.float32)
    p2, p98 = np.percentile(cl, [2.0, 98.0])
    if p98 > p2:
        norm = np.clip((cl - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = cl.astype(np.uint8)
    rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return resized

PREPROCESSORS = {
    "Variant_A": preprocess_variant_a,
    "Variant_B": preprocess_variant_b,
    "Variant_C": preprocess_variant_c,
    "Variant_D": preprocess_variant_d,
    "Variant_E": preprocess_variant_e,
}

# ─── CONSERVATIVE AUGMENTATION (TRAIN ONLY) ──────────────────────────────────

def augment_image(img_arr):
    """
    Fingerprint-preserving augmentation:
    - Small rotation (+/- 8 deg)
    - Small translation (+/- 6 pixels)
    - Slight zoom (0.95 to 1.05)
    - Mild brightness adjustment (+/- 10%)
    - Mild contrast adjustment (+/- 10%)
    """
    h, w, _ = img_arr.shape
    # Rotation
    angle = random.uniform(-8, 8)
    # Translation
    tx = random.uniform(-6, 6)
    ty = random.uniform(-6, 6)
    # Zoom
    scale = random.uniform(0.95, 1.05)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    aug = cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Brightness & Contrast
    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-10, 10)
    aug = np.clip(alpha * aug + beta, 0, 255).astype(np.uint8)
    return aug

# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_split_data(split_manifest, preprocessor_fn, augment_train=True, aug_factor=4):
    """
    Loads images for train, val, test according to the preprocessor function.
    Augmentation is applied only to training images.
    """
    # Map captures to labels
    cap_to_label = {}
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cap_to_label[r["capture_filename"].strip()] = r["confirmed_blood_group"].strip()

    def _process_partition(captures, is_train=False):
        X, y, meta = [], [], []
        for fn in captures:
            p = os.path.join(PROJECT_ROOT, "data", "captured", fn)
            bg = cap_to_label[fn]
            label_idx = CLASS_TO_IDX[bg]
            with Image.open(p) as img_pil:
                base_arr = preprocessor_fn(img_pil)

            X.append(base_arr)
            y.append(label_idx)
            meta.append({"filename": fn, "label": bg})

            if is_train and augment_train:
                for _ in range(aug_factor):
                    aug_arr = augment_image(base_arr)
                    X.append(aug_arr)
                    y.append(label_idx)
                    meta.append({"filename": f"aug_{fn}", "label": bg})

        X_arr = np.array(X, dtype=np.float32)
        X_res = resnet_preprocess(X_arr)
        y_arr = np.array(y, dtype=np.int32)
        return X_res, y_arr, meta

    # Train captures
    train_caps = []
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subject_id"] in split_manifest["train_subjects"]:
                train_caps.append(r["capture_filename"])

    val_caps = []
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subject_id"] in split_manifest["val_subjects"]:
                val_caps.append(r["capture_filename"])

    test_caps = []
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subject_id"] in split_manifest["test_subjects"]:
                test_caps.append(r["capture_filename"])

    X_train, y_train, meta_train = _process_partition(train_caps, is_train=True)
    X_val, y_val, meta_val = _process_partition(val_caps, is_train=False)
    X_test, y_test, meta_test = _process_partition(test_caps, is_train=False)

    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)

# ─── BUILD ADAPTER MODEL ─────────────────────────────────────────────────────

def build_adapted_resnet50_v1():
    """
    Creates resnet50_mfs100_abo_v1 architecture:
    - Pretrained ResNet50 backbone (100% frozen)
    - GAP -> Dense(64, ReLU, L2=1e-3) -> Dropout(0.5) -> Dense(4, Softmax)
    """
    base_model = tf.keras.models.load_model(BASE_RESNET_PATH, compile=False)
    # Extract the resnet50 backbone
    resnet_backbone = base_model.get_layer("resnet50")
    resnet_backbone.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="mfs100_input")
    x = resnet_backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="adaptation_gap")(x)
    x = layers.BatchNormalization(name="adaptation_bn")(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-3),
        name="adaptation_dense_64"
    )(x)
    x = layers.Dropout(0.50, name="adaptation_dropout")(x)
    outputs = layers.Dense(4, activation="softmax", name="abo_4class_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50_MFS100_ABO_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─── EVALUATION HELPERS ──────────────────────────────────────────────────────

def evaluate_predictions(y_true_indices, y_prob_matrix, subject_ids):
    """
    Computes impression-level, consensus-level, and calibration metrics.
    """
    y_pred_indices = np.argmax(y_prob_matrix, axis=1)
    y_confs = np.max(y_prob_matrix, axis=1)

    classes = PILOT_CLASSES
    n_samples = len(y_true_indices)

    # 1. Impression Metrics
    correct_impressions = int(np.sum(y_true_indices == y_pred_indices))
    imp_acc = round(correct_impressions / n_samples, 4)

    # Confusion Matrix
    cm = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for yt, yp in zip(y_true_indices, y_pred_indices):
        cm[IDX_TO_CLASS[yt]][IDX_TO_CLASS[yp]] += 1

    # Per-Class Precision, Recall, F1
    per_class = {}
    recalls = []
    for i, c in enumerate(classes):
        tp = cm[c][c]
        fn = sum(cm[c][other] for other in classes if other != c)
        fp = sum(cm[other][c] for other in classes if other != c)
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        f1 = round((2 * prec * rec) / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        recalls.append(rec)

    balanced_acc = round(float(np.mean(recalls)), 4)
    macro_prec = round(float(np.mean([per_class[c]["precision"] for c in classes])), 4)
    macro_rec = round(float(np.mean(recalls)), 4)
    macro_f1 = round(float(np.mean([per_class[c]["f1"] for c in classes])), 4)

    # 2. 3-Impression Consensus Metrics
    sub_groups = defaultdict(list)
    for sid, yt, yp, probs in zip(subject_ids, y_true_indices, y_pred_indices, y_prob_matrix):
        sub_groups[sid].append({"true": yt, "pred": yp, "probs": probs})

    maj_correct = 0
    soft_correct = 0
    total_subjects = len(sub_groups)
    consensus_details = {}

    for sid, impressions in sub_groups.items():
        true_label = impressions[0]["true"]
        pred_list = [imp["pred"] for imp in impressions]
        prob_mean = np.mean([imp["probs"] for imp in impressions], axis=0)

        # Majority vote
        counts = Counter(pred_list)
        maj_vote = counts.most_common(1)[0][0]
        # Soft vote
        soft_vote = int(np.argmax(prob_mean))

        is_maj_correct = (maj_vote == true_label)
        is_soft_correct = (soft_vote == true_label)

        if is_maj_correct:
            maj_correct += 1
        if is_soft_correct:
            soft_correct += 1

        consensus_details[sid] = {
            "ground_truth": IDX_TO_CLASS[true_label],
            "impression_predictions": [IDX_TO_CLASS[p] for p in pred_list],
            "majority_consensus": IDX_TO_CLASS[maj_vote],
            "soft_consensus": IDX_TO_CLASS[soft_vote],
            "majority_correct": is_maj_correct,
            "soft_correct": is_soft_correct
        }

    maj_consensus_acc = round(maj_correct / total_subjects, 4)
    soft_consensus_acc = round(soft_correct / total_subjects, 4)

    # 3. O+ Prediction Collapse Analysis
    o_plus_preds = int(np.sum(y_pred_indices == CLASS_TO_IDX["O+"]))
    o_plus_pct = round((o_plus_preds / n_samples) * 100.0, 2)
    actual_o_plus = int(np.sum(y_true_indices == CLASS_TO_IDX["O+"]))
    actual_o_plus_pct = round((actual_o_plus / n_samples) * 100.0, 2)

    # 4. Confidence Analysis
    correct_confs = [float(c) for yt, yp, c in zip(y_true_indices, y_pred_indices, y_confs) if yt == yp]
    incorrect_confs = [float(c) for yt, yp, c in zip(y_true_indices, y_pred_indices, y_confs) if yt != yp]

    return {
        "impression_accuracy": imp_acc,
        "balanced_accuracy": balanced_acc,
        "majority_consensus_accuracy": maj_consensus_acc,
        "soft_consensus_accuracy": soft_consensus_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "o_plus_collapse_check": {
            "predicted_o_plus_count": o_plus_preds,
            "predicted_o_plus_percentage": o_plus_pct,
            "actual_o_plus_percentage": actual_o_plus_pct,
            "collapse_flag": o_plus_pct > 60.0 and actual_o_plus_pct <= 30.0
        },
        "confidence_analysis": {
            "mean_confidence": round(float(np.mean(y_confs)), 4),
            "median_confidence": round(float(np.median(y_confs)), 4),
            "mean_confidence_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else 0.0,
            "mean_confidence_incorrect": round(float(np.mean(incorrect_confs)), 4) if incorrect_confs else 0.0
        },
        "consensus_details_by_subject": consensus_details
    }

# ─── BASELINE EVALUATOR: CHAMPION ENSEMBLE V2 (RE-NORMALIZED) ─────────────────

def evaluate_champion_v2_baseline(test_captures, preprocessor_fn):
    """
    Evaluates Champion Ensemble V2 on unseen test captures without modifying model.
    Maps/aggregates 8-class outputs to the four pilot classes (A+, AB+, B+, O+) via
    sub-distribution re-normalization:
        P_4class(c) = P_8class(c) / sum_{k in {A+, AB+, B+, O+}} P_8class(k)
    """
    from src.predictor import BloodGroupPredictor
    predictor = BloodGroupPredictor()

    cap_to_sub = {}
    cap_to_label = {}
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r["capture_filename"].strip()
            cap_to_sub[fn] = r["subject_id"].strip()
            cap_to_label[fn] = r["confirmed_blood_group"].strip()

    y_true = []
    y_prob_4class = []
    subject_ids = []

    for fn in test_captures:
        p = os.path.join(PROJECT_ROOT, "data", "captured", fn)
        gt_bg = cap_to_label[fn]
        y_true.append(CLASS_TO_IDX[gt_bg])
        subject_ids.append(cap_to_sub[fn])

        # Predict using Champion Ensemble V2
        res = predictor.predict(p, source="mfs100")
        prob_dist_8class = res.get("probabilities", {})

        # Sub-distribution extraction and re-normalization
        sub_probs = [prob_dist_8class.get(c, 0.0) for c in PILOT_CLASSES]
        sum_probs = sum(sub_probs)
        if sum_probs > 0:
            norm_4class = [p / sum_probs for p in sub_probs]
        else:
            norm_4class = [0.25, 0.25, 0.25, 0.25]
        y_prob_4class.append(norm_4class)

    y_prob_matrix = np.array(y_prob_4class, dtype=np.float32)
    return evaluate_predictions(np.array(y_true), y_prob_matrix, subject_ids)

# ─── MAIN EXPERIMENT EXECUTION ───────────────────────────────────────────────

def run_experiment():
    print("==========================================================================")
    print("STARTING: MFS100 FOUR-CLASS ABO-POSITIVE ADAPTATION PILOT")
    print("==========================================================================")

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    # Compute Class Weights for Imbalanced Training Set (A+: 6 imgs, AB+: 3 imgs, B+: 18 imgs, O+: 24 imgs)
    class_counts = Counter()
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subject_id"] in split_manifest["train_subjects"]:
                class_counts[CLASS_TO_IDX[r["confirmed_blood_group"]]] += 1

    total_train = sum(class_counts.values())
    class_weights = {idx: round(total_train / (4.0 * count), 3) for idx, count in class_counts.items()}
    print(f"Train Class Distribution: {dict(class_counts)}")
    print(f"Calculated Class Weights: {class_weights}")

    # ── PHASE 1: PREPROCESSING VARIANT SELECTION (VALIDATION SUBJECTS ONLY) ────
    print("\n--- PHASE 1: PREPROCESSING VARIANT SELECTION ON VALIDATION SUBJECTS ---")
    variant_results = {}
    best_val_acc = -1.0
    best_variant_name = "Variant_C"

    # Screen each variant using a fast trial adapter
    for var_name, preproc in PREPROCESSORS.items():
        print(f"\nEvaluating Preprocessing Variant: {var_name}...")
        (X_tr, y_tr, _), (X_v, y_v, _), _ = load_split_data(
            split_manifest, preproc, augment_train=True, aug_factor=3
        )

        trial_model = build_adapted_resnet50_v1()
        history = trial_model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            epochs=15,
            batch_size=8,
            class_weight=class_weights,
            verbose=0
        )
        val_acc = max(history.history["val_accuracy"])
        final_val_loss = min(history.history["val_loss"])
        variant_results[var_name] = {
            "best_val_accuracy": round(float(val_acc), 4),
            "best_val_loss": round(float(final_val_loss), 4)
        }
        print(f"  {var_name} -> Best Val Acc: {val_acc:.4f} | Min Val Loss: {final_val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_variant_name = var_name

    print(f"\nSELECTED BEST PREPROCESSING VARIANT: {best_variant_name} (Val Acc: {best_val_acc:.4f})")
    selected_preprocessor = PREPROCESSORS[best_variant_name]

    # Save preprocessing selection summary
    with open(os.path.join(OUTPUT_DIR, "preprocessing_selection_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "variants_evaluated": variant_results,
            "selected_variant": best_variant_name,
            "criterion": "Validation subject accuracy & loss (zero test-set leakage)"
        }, f, indent=2)

    # ── PHASE 2: TRAIN FINAL ADAPTED MODEL (resnet50_mfs100_abo_v1) ───────────
    print(f"\n--- PHASE 2: TRAINING ADAPTATION MODEL resnet50_mfs100_abo_v1.keras ---")
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, meta_test) = load_split_data(
        split_manifest, selected_preprocessor, augment_train=True, aug_factor=4
    )
    print(f"Prepared Tensors: Train={X_train.shape}, Val={X_val.shape}, Unseen Test={X_test.shape}")

    adapter_model = build_adapted_resnet50_v1()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=TARGET_V1_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = adapter_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=8,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )

    training_log = {
        "epochs_completed": len(history.history["loss"]),
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        "final_val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
        "history": {
            "loss": [round(float(x), 4) for x in history.history["loss"]],
            "accuracy": [round(float(x), 4) for x in history.history["accuracy"]],
            "val_loss": [round(float(x), 4) for x in history.history["val_loss"]],
            "val_accuracy": [round(float(x), 4) for x in history.history["val_accuracy"]],
        }
    }
    with open(os.path.join(OUTPUT_DIR, "adaptation_v1_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    # ── PHASE 3: UNSEEN TEST EVALUATION: BASELINE VS. ADAPTED MODEL ───────────
    print("\n--- PHASE 3: EVALUATION ON UNSEEN TEST SUBJECTS (S004, S026, S014, S025) ---")

    test_captures = []
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subject_id"] in split_manifest["test_subjects"]:
                test_captures.append(r["capture_filename"])

    # 1. Baseline: Champion Ensemble V2
    print("Evaluating Baseline: Champion Ensemble V2 (4-Class Re-Normalized)...")
    baseline_metrics = evaluate_champion_v2_baseline(test_captures, selected_preprocessor)

    # 2. Adapted Model: resnet50_mfs100_abo_v1
    print("Evaluating Adapted Model: ResNet50 MFS100 ABO v1...")
    adapted_probs = adapter_model.predict(X_test)
    test_sub_ids = []
    cap_to_sub = {}
    with open(MFS_GT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cap_to_sub[r["capture_filename"].strip()] = r["subject_id"].strip()
    for m in meta_test:
        test_sub_ids.append(cap_to_sub[m["filename"]])

    adapted_metrics = evaluate_predictions(y_test, adapted_probs, test_sub_ids)

    # Compile Full Pilot Comparison Report
    final_report = {
        "experiment_title": "MFS100 Four-Class ABO-Positive Adaptation Pilot",
        "date": "2026-09-07",
        "authoritative_statement": (
            "The final system targets eight ABO/Rh blood-group classes. During the initial real-world "
            "MFS100 adaptation pilot, independently confirmed participants were available only for "
            "A+, AB+, B+, and O+. Therefore, the adaptation experiment was conducted as a four-class "
            "ABO-positive pilot. A-, AB-, B-, and O- were not fabricated or inferred and remain an "
            "important part of the final system, requiring future independently confirmed MFS100 samples."
        ),
        "pilot_classes": PILOT_CLASSES,
        "unrepresented_classes": ["A-", "AB-", "B-", "O-"],
        "unrepresented_status": "NOT EVALUATED — NO INDEPENDENTLY CONFIRMED REAL-WORLD MFS100 SAMPLES AVAILABLE",
        "cohort_summary": {
            "total_subjects": 26,
            "total_captures": 78,
            "train_subjects": split_manifest["train_subjects"],
            "val_subjects": split_manifest["val_subjects"],
            "test_subjects": split_manifest["test_subjects"]
        },
        "selected_preprocessing_variant": best_variant_name,
        "models_evaluated": {
            "baseline_champion_ensemble_v2": baseline_metrics,
            "adapted_resnet50_mfs100_abo_v1": adapted_metrics
        }
    }

    report_path = os.path.join(OUTPUT_DIR, "mfs100_adaptation_pilot_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\n==========================================================================")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"Results report saved to: {report_path}")
    print(f"Model saved to: {TARGET_V1_MODEL_PATH}")
    print("==========================================================================")
    return final_report

if __name__ == "__main__":
    run_experiment()
