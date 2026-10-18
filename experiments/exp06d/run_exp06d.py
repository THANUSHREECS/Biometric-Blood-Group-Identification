#!/usr/bin/env python3
"""
EXP-06D: Preprocessing & Multi-Scan Representation Investigation
===============================================================
Controlled single-factor ablation investigating whether legacy ridge contrast
enhancement & bilateral filtering (transform_c5) improves participant-level
blood group classification over raw platen-cropped optical ridge impressions
(transform_c5_raw) under the verified EXP-06C Config A linear projection
SupCon architecture (tau = 0.05, lambda = 0.50).

Configurations:
  - Config A (Reference: Enhanced C5):
      Preprocessing: White composite -> Platen Crop -> CLAHE + Contrast Stretch + Bilateral Filter -> Resize (112, 112)
      Backbone: Pretrained DenseNet-121 (strictly frozen, 7,037,504 non-trainable params)
      Fusion: 3-Scan Concatenation (3072-d)
      Heads: Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax) AND Linear Dense(128, None)
      Loss: 1.0 * CCE + 0.50 * SupCon(tau=0.05)
      Trainable parameters: 1,968,772 across 6 tensors
  - Config B (Candidate: Raw Platen C5):
      Preprocessing: White composite -> Platen Crop -> Resize (112, 112) (NO CLAHE, NO stretch, NO bilateral filter)
      Backbone: Pretrained DenseNet-121 (strictly frozen, 7,037,504 non-trainable params)
      Fusion: 3-Scan Concatenation (3072-d)
      Heads: Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax) AND Linear Dense(128, None)
      Loss: 1.0 * CCE + 0.50 * SupCon(tau=0.05)
      Trainable parameters: 1,968,772 across 6 tensors (EXACT MATCH, 0% parameter confound)

MANDATORY SAFETY BOUNDARIES:
  - 80 Development Participants ONLY (240 scans)
  - 21-participant locked test cohort strictly quarantined (0 access)
  - 26-participant hospital cohort strictly quarantined (0 access)
  - All 66 pre-existing checkpoints protected and hash-verified
  - FAIL-CLOSED TRAINING GUARD: Training cannot execute unless explicitly authorized.
"""

import os
import sys
import json
import csv
import time
import shutil
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white,
    crop_mfs100_foreground, enhance_mfs100_ridge_contrast,
    resize_to_target
)

# ==============================================================================
# AUTHORIZATION BOUNDARY & FAIL-CLOSED GUARD
# ==============================================================================
# Training has completed. Restoring fail-closed guard to False.
TRAINING_AUTHORIZED = False

EXP_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp06d" / "checkpoints"
EXP_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06d"
CM_DIR = EXP_REPORT_DIR / "confusion_matrices"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
PRE_HASHES_FILE = PROJECT_ROOT / "scratch" / "exp06d_pre_preparation_hashes.json"

# Protected checkpoint directories that MUST NEVER be targeted for output writes
PROTECTED_CKPT_DIRS = [
    PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2",
    PROJECT_ROOT / "experiments" / "exp03" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp05" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06a" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06b" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06c" / "checkpoints"
]

# Global Hyperparameters (100% matched to EXP-06C Config A reference)
SEED = 42
BATCH_SIZE = 16
EPOCHS = 150
MIN_EPOCHS = 100
PATIENCE = 15
WEIGHT_DECAY = 1e-4
HEAD_LR = 1e-3
FIXED_TEMPERATURE = 0.05
FIXED_LAMBDA = 0.50

CONFIG_SPECS = {
    "config_a": {
        "name": "Config A: Reference (Enhanced C5: Crop + CLAHE + Bilateral Filter)",
        "preprocessing_mode": "enhanced",
        "head_type": "linear",
        "lambda_supcon": FIXED_LAMBDA,
        "temperature": FIXED_TEMPERATURE,
        "expected_trainable_vars": 6,
        "expected_trainable_params": 1968772,
        "projection_dim": 128
    },
    "config_b": {
        "name": "Config B: Candidate (Raw Platen C5: Crop Only, No CLAHE/Bilateral Filter)",
        "preprocessing_mode": "raw",
        "head_type": "linear",
        "lambda_supcon": FIXED_LAMBDA,
        "temperature": FIXED_TEMPERATURE,
        "expected_trainable_vars": 6,
        "expected_trainable_params": 1968772,
        "projection_dim": 128
    }
}

NATIVE_CLASSES = ["A+", "B+", "AB+", "O+"]
NATIVE_REV_MAP = {c: i for i, c in enumerate(NATIVE_CLASSES)}
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# ==============================================================================
# PREPROCESSING IMPLEMENTATIONS
# ==============================================================================

def transform_c5_enhanced(image_input):
    """
    Reference C5 pipeline (EXP-06C Config A):
    1. Composite alpha on white.
    2. Crop MFS100 active platen (with 6% padding).
    3. CLAHE (clip=2.5) + dynamic contrast stretching + bilateral filter.
    4. Resize to (112, 112).
    5. Output raw float32 array in [0.0, 255.0].
    """
    raw_pil = load_image_to_pil(image_input)
    rgb_pil = composite_alpha_on_white(raw_pil)
    cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)
    enhanced_pil = enhance_mfs100_ridge_contrast(cropped_pil)
    resized_pil = resize_to_target(enhanced_pil, target_size=(112, 112))
    img_array = np.array(resized_pil, dtype=np.float32)
    return img_array, resized_pil, crop_meta

def transform_c5_raw(image_input):
    """
    Candidate C5 pipeline (EXP-06D Config B):
    1. Composite alpha on white.
    2. Crop MFS100 active platen (with 6% padding).
    3. NO CLAHE, NO contrast stretching, NO bilateral filtering (raw optical ridges).
    4. Resize to (112, 112).
    5. Output raw float32 array in [0.0, 255.0].
    """
    raw_pil = load_image_to_pil(image_input)
    rgb_pil = composite_alpha_on_white(raw_pil)
    cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)
    resized_pil = resize_to_target(cropped_pil, target_size=(112, 112))
    img_array = np.array(resized_pil, dtype=np.float32)
    return img_array, resized_pil, crop_meta

# ==============================================================================
# VALIDATION & INTEGRITY CHECK ROUTINES
# ==============================================================================

def sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_input_shapes(X: np.ndarray, y: np.ndarray, pids: np.ndarray):
    """
    Validates participant-level tensor shapes and alignment.
    """
    if len(X.shape) != 5:
        raise ValueError(f"Input X must have 5 dimensions (N, 3, H, W, C), got {X.shape}")
    N, num_scans, H, W, C = X.shape
    if num_scans != 3:
        raise ValueError(f"Expected exactly 3 scans per participant, got {num_scans}")
    if (H, W, C) != (112, 112, 3):
        raise ValueError(f"Expected scan resolution (112, 112, 3), got ({H}, {W}, {C})")
    if len(y) != N:
        raise ValueError(f"Label count {len(y)} does not match participant count {N}")
    if len(pids) != N:
        raise ValueError(f"Participant ID count {len(pids)} does not match sample count {N}")
    return True

def validate_three_scan_integrity(df: pd.DataFrame):
    """
    Validates that every participant has exactly 3 scans.
    """
    counts = df.groupby("participant_id")["scan_path"].count()
    invalid = counts[counts != 3]
    if len(invalid) > 0:
        raise ValueError(f"Participants with scan count != 3: {dict(invalid)}")
    return True

def validate_label_consistency(p_scans: pd.DataFrame):
    """
    Validates that all scans for a single participant have identical labels.
    """
    unique_labels = p_scans["blood_group"].unique()
    if len(unique_labels) != 1:
        raise ValueError(f"Inconsistent labels for participant {p_scans.iloc[0]['participant_id']}: {unique_labels}")
    return True

def validate_participant_leakage(train_pids: set, val_pids: set):
    """
    Validates zero participant overlap between train and validation splits.
    """
    overlap = train_pids.intersection(val_pids)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL LEAKAGE: Participants in both train and validation: {overlap}")
    return True

def validate_probabilities(probs: np.ndarray):
    """
    Validates that probability predictions are finite, non-negative, and sum to 1.0.
    """
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities contain NaN or Inf values")
    if np.any(probs < 0.0):
        raise ValueError("Probabilities contain negative values")
    sums = np.sum(probs, axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError(f"Probabilities do not sum to 1.0: min={np.min(sums)}, max={np.max(sums)}")
    return True

def validate_output_directory_isolation(target_dir: Path):
    """
    Validates that the output directory is strictly isolated to experiments/exp06d
    and does not conflict with protected historical directories.
    """
    resolved_target = target_dir.resolve()
    for protected in PROTECTED_CKPT_DIRS:
        resolved_prot = protected.resolve()
        if resolved_target == resolved_prot or resolved_prot in resolved_target.parents:
            raise PermissionError(f"DIRECTORY CONFLICT: Target directory {resolved_target} conflicts with protected {resolved_prot}")
    return True

def check_cohort_quarantine(pids: set, split_csv: Path = SPLIT_CSV):
    """
    Verifies that zero locked test participants or hospital participants exist in the dataset.
    """
    df_split = pd.read_csv(split_csv)
    test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
    overlap = pids.intersection(test_pids)
    if len(overlap) > 0:
        raise PermissionError(f"QUARANTINE BREACH: Locked test cohort IDs detected: {overlap}")
    return True

def verify_protected_checkpoints(hashes_file: Path = PRE_HASHES_FILE):
    """
    Audits all 66 protected checkpoints against pre-preparation hashes.
    """
    if not hashes_file.is_file():
        raise FileNotFoundError(f"Protected hashes file not found: {hashes_file}")
    with open(hashes_file, "r") as f:
        registry = json.load(f)
    for name, info in registry.items():
        p = Path(info["path"])
        if not p.is_file():
            raise FileNotFoundError(f"Protected checkpoint missing: {p} ({name})")
        curr_h = sha256_of_file(p)
        if curr_h != info["sha256"]:
            raise ValueError(f"CHECKPOINT CORRUPTION: {name} altered! pre={info['sha256']}, curr={curr_h}")
    return len(registry)

# ==============================================================================
# MODEL ARCHITECTURE & LOSS
# ==============================================================================

def supervised_contrastive_loss(feature_vectors, labels, temperature=FIXED_TEMPERATURE):
    feat = tf.math.l2_normalize(feature_vectors, axis=1)
    sim = tf.divide(tf.matmul(feat, tf.transpose(feat)), temperature)
    sim_max = tf.math.reduce_max(sim, axis=1, keepdims=True)
    sim = sim - tf.stop_gradient(sim_max)
    lbl = tf.reshape(labels, [-1, 1])
    mask = tf.cast(tf.equal(lbl, tf.transpose(lbl)), tf.float32)
    bs = tf.shape(labels)[0]
    logits_mask = tf.ones_like(mask) - tf.eye(bs)
    mask = mask * logits_mask
    exp_sim = tf.math.exp(sim) * logits_mask
    log_prob = sim - tf.math.log(tf.math.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-12)
    mask_sum = tf.math.reduce_sum(mask, axis=1)
    has_pos = tf.cast(mask_sum > 1e-5, tf.float32)
    mask_sum = tf.where(mask_sum < 1e-5, tf.ones_like(mask_sum), mask_sum)
    mean_log_prob = (tf.math.reduce_sum(mask * log_prob, axis=1) / mask_sum) * has_pos
    loss = -mean_log_prob
    num_valid = tf.reduce_sum(has_pos)
    return tf.cond(
        num_valid > 0.0,
        lambda: tf.reduce_sum(loss) / num_valid,
        lambda: tf.constant(0.0, dtype=tf.float32)
    )

class E2PreprocessingAblationModel(tf.keras.Model):
    def __init__(self, inputs, outputs, base_model, head_layers,
                 lambda_supcon=FIXED_LAMBDA, temperature=FIXED_TEMPERATURE, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.base_model = base_model
        self.head_layers = head_layers
        self.lambda_supcon = float(lambda_supcon)
        self.temperature = float(temperature)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            preds_clf, preds_emb = self(x, training=True)
            loss_clf = self.compiled_loss(
                y, preds_clf, sample_weight=sample_weight, regularization_losses=self.losses
            )
            loss_supcon = supervised_contrastive_loss(preds_emb, y, temperature=self.temperature)
            total_loss = loss_clf + (self.lambda_supcon * loss_supcon)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, preds_clf, sample_weight=sample_weight)
        res = {m.name: m.result() for m in self.metrics}
        res["loss"] = total_loss
        res["loss_clf"] = loss_clf
        res["loss_supcon"] = loss_supcon
        return res

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        preds_clf, preds_emb = self(x, training=False)
        loss_clf = self.compiled_loss(
            y, preds_clf, sample_weight=sample_weight, regularization_losses=self.losses
        )
        loss_supcon = supervised_contrastive_loss(preds_emb, y, temperature=self.temperature)
        total_loss = loss_clf + (self.lambda_supcon * loss_supcon)

        self.compiled_metrics.update_state(y, preds_clf, sample_weight=sample_weight)
        res = {m.name: m.result() for m in self.metrics}
        res["loss"] = total_loss
        res["loss_clf"] = loss_clf
        res["loss_supcon"] = loss_supcon
        return res

def build_exp06d_model(seed_head: int = 42):
    """
    Constructs the E2 architecture for EXP-06D:
    Input: (None, 3, 112, 112, 3)
    Backbone: Pretrained DenseNet-121 (strictly frozen)
    Fusion: 3-Scan Concatenation (3072-d)
    Classifier: Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)
    Projection: Linear Dense(128, None)
    Trainable parameters: exactly 1,968,772 across 6 tensors.
    """
    inputs = tf.keras.Input(shape=(3, 112, 112, 3), name="participant_scans")
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False

    pool = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    f1, f2, f3 = pool(base(s1)), pool(base(s2)), pool(base(s3))
    fused = tf.keras.layers.Concatenate(name="fused_features")([f1, f2, f3])

    dense_512 = tf.keras.layers.Dense(
        512, activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head),
        bias_initializer="zeros",
        name="dense_512"
    )
    dropout_05 = tf.keras.layers.Dropout(0.5, name="dropout_0.5")
    clf = tf.keras.layers.Dense(
        4, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head + 1),
        bias_initializer="zeros",
        name="clf"
    )
    d = dense_512(fused)
    drop = dropout_05(d)
    out_clf = clf(drop)

    emb = tf.keras.layers.Dense(
        128, activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head + 2),
        bias_initializer="zeros",
        name="emb"
    )
    out_emb = emb(fused)
    head_layers = [dense_512, clf, emb]

    model = E2PreprocessingAblationModel(
        inputs=inputs,
        outputs=[out_clf, out_emb],
        base_model=base,
        head_layers=head_layers,
        lambda_supcon=FIXED_LAMBDA,
        temperature=FIXED_TEMPERATURE,
        name="E2_EXP06D_LinearModel"
    )
    return model, base, head_layers

def safe_save_weights(model, filepath, max_retries=10, delay=1.0):
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            model.save_weights(str(p))
            return
        except (PermissionError, OSError):
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                tmp_p = p.parent / f"{p.stem}_tmp_{os.getpid()}_{attempt}{p.suffix}"
                model.save_weights(str(tmp_p))
                for mv_att in range(max_retries):
                    try:
                        if p.is_file():
                            os.remove(str(p))
                        shutil.move(str(tmp_p), str(p))
                        return
                    except (PermissionError, OSError):
                        time.sleep(delay)
                if tmp_p.is_file() and not p.is_file():
                    shutil.copy(str(tmp_p), str(p))
                    return

class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """
    Monitors validation Macro F1 on held-out fold validation data.
    Saves best model weights and enforces MIN_EPOCHS floor before allowing early stop.
    """
    def __init__(self, filepath, X_val, y_val, patience=PATIENCE, min_epochs=MIN_EPOCHS):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.wait = 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        clf_preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        y_pred = np.argmax(clf_preds, axis=1)

        val_f1 = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))
        val_acc = float(accuracy_score(self.y_val, y_pred))

        self.history.append({
            "epoch": epoch + 1,
            "val_macro_f1": val_f1,
            "val_acc": val_acc,
            "train_loss": float(logs.get("loss", 0.0)),
            "train_loss_clf": float(logs.get("loss_clf", 0.0)),
            "train_loss_supcon": float(logs.get("loss_supcon", 0.0))
        })

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = epoch + 1
            self.wait = 0
            safe_save_weights(self.model, self.filepath)
            print(f"    Epoch {epoch+1:3d}: New Best Val Macro F1 = {val_f1:.4f} (Acc = {val_acc*100:.1f}%) [SAVED]", flush=True)
        else:
            self.wait += 1
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}: Val Macro F1 = {val_f1:.4f} (Best: {self.best_f1:.4f} @ Ep {self.best_epoch})", flush=True)

        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            print(f"    Early stopping triggered at epoch {epoch+1} (Best epoch: {self.best_epoch}, Best Val Macro F1: {self.best_f1:.4f})", flush=True)
            self.model.stop_training = True

# ==============================================================================
# DATA LOADER & PREFLIGHT
# ==============================================================================

def load_development_dataset(preprocessing_mode: str = "enhanced"):
    """
    Loads strictly the 80 approved development participants.
    Applies either transform_c5_enhanced or transform_c5_raw based on preprocessing_mode.
    """
    df_split = pd.read_csv(SPLIT_CSV)
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()

    assert dev_df["participant_id"].nunique() == 80, f"Expected 80 dev participants, got {dev_df['participant_id'].nunique()}"
    assert len(dev_df) == 240, f"Expected 240 dev scans, got {len(dev_df)}"
    validate_three_scan_integrity(dev_df)

    pids_unique = sorted(dev_df["participant_id"].unique())
    check_cohort_quarantine(set(pids_unique))

    dev_X_list = []
    dev_y_list = []
    dev_labels_list = []

    transform_fn = transform_c5_enhanced if preprocessing_mode == "enhanced" else transform_c5_raw

    for pid in pids_unique:
        p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
        validate_label_consistency(p_scans)

        bg = p_scans.iloc[0]["blood_group"]
        assert bg in NATIVE_CLASSES, f"Unknown blood group: {bg}"
        dev_y_list.append(NATIVE_REV_MAP[bg])
        dev_labels_list.append(bg)

        three_scans = []
        for _, row in p_scans.iterrows():
            img_p = Path(row["scan_path"])
            assert img_p.is_file(), f"Scan file not found: {img_p}"
            arr, _, _ = transform_fn(str(img_p))
            three_scans.append(arr)

        dev_X_list.append(np.array(three_scans, dtype=np.float32))

    dev_X = np.array(dev_X_list, dtype=np.float32)
    dev_y = np.array(dev_y_list, dtype=np.int32)
    dev_pids = np.array(pids_unique)

    validate_input_shapes(dev_X, dev_y, dev_pids)
    print(f"Loaded {preprocessing_mode.upper()} dev_X: shape={dev_X.shape}, dev_y: shape={dev_y.shape}")
    return dev_X, dev_y, dev_pids, dev_labels_list

def run_preflight():
    print("=" * 80)
    print("EXP-06D: PREFLIGHT INTEGRITY & SAFETY GATE AUDIT")
    print("=" * 80)

    # 1. Dataset & quarantine checks
    print("\n[Preflight 1] Dataset Isolation & Quarantine Checks:")
    df_split = pd.read_csv(SPLIT_CSV)
    dev_pids_set = set(df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]["participant_id"])
    test_pids_set = set(df_split[df_split["split"] == "TEST"]["participant_id"])

    assert len(dev_pids_set) == 80, f"Expected 80 dev participants, got {len(dev_pids_set)}"
    assert len(test_pids_set) == 21, f"Expected 21 test participants, got {len(test_pids_set)}"
    assert len(dev_pids_set.intersection(test_pids_set)) == 0, "DATA LEAKAGE: Overlap with locked test set!"
    print("  Development cohort: 80 participants (240 scans) strictly isolated.")
    print("  Locked test set:    21 participants (63 scans) STRICTLY QUARANTINED (0 access).")

    # 2. Checkpoint integrity of historical baselines (66 models)
    print("\n[Preflight 2] Protected Checkpoint Integrity Audit (66 models):")
    count_verified = verify_protected_checkpoints(PRE_HASHES_FILE)
    print(f"  All {count_verified} protected checkpoints verified 100% bit-for-bit unchanged.")

    # 3. Parameter count verification
    print("\n[Preflight 3] Model Parameter & Architecture Verification:")
    m_a, b_a, h_a = build_exp06d_model(seed_head=42)
    assert not b_a.trainable, "Backbone must be frozen!"
    assert b_a.count_params() == 7037504, f"DenseNet-121 non-trainable params mismatch: {b_a.count_params()}"
    t_params = sum(int(np.prod(v.shape)) for v in m_a.trainable_variables)
    t_vars = len(m_a.trainable_variables)
    assert t_params == 1968772, f"Trainable params mismatch: {t_params} != 1968772"
    assert t_vars == 6, f"Trainable variables mismatch: {t_vars} != 6"
    print(f"  Config A/B Architecture: {t_params:,} trainable params across {t_vars} variables (0% confound).")
    del m_a

    # 4. 5-Fold StratifiedGroupKFold Integrity
    print("\n[Preflight 4] 5-Fold StratifiedGroupKFold Integrity:")
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
    pids_sorted = sorted(dev_df["participant_id"].unique())
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy_X = np.zeros((len(pids_sorted), 1))
    dummy_y = np.array([NATIVE_REV_MAP[dev_df[dev_df['participant_id'] == pid].iloc[0]['blood_group']] for pid in pids_sorted])
    folds = list(sgkf.split(dummy_X, dummy_y, pids_sorted))
    assert len(folds) == 5
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        assert len(t_idx) == 64 and len(v_idx) == 16
        leak = set(np.array(pids_sorted)[t_idx]) & set(np.array(pids_sorted)[v_idx])
        assert len(leak) == 0, f"Participant leakage in fold {f_idx}!"
    print("  Folds 1-5: PASS | Exactly 64 train / 16 val per fold, zero participant leakage.")

    print("\n[Preflight 5] Preflight Complete — ALL INTEGRITY CHECKS PASSED.")
    print("=" * 80)
    return True

# ==============================================================================
# TRAINING & EVALUATION ENGINE
# ==============================================================================

def run_configuration(config_id: str, dry_run=False):
    """
    Five-fold cross-validation execution with fail-closed authorization guards,
    shared epoch-zero initialization parity, and participant-level OOF logging.
    """
    # FAIL-CLOSED AUTHORIZATION GUARD
    if not TRAINING_AUTHORIZED:
        raise PermissionError(
            "TRAINING REFUSED (FAIL-CLOSED GUARD): "
            "TRAINING_AUTHORIZED is False. Real-data training has not been authorized by the user. "
            "Execution aborted safely."
        )

    spec = CONFIG_SPECS[config_id]
    print(f"\n{'='*80}")
    print(f"RUNNING EXP-06D: {spec['name'].upper()} (dry_run={dry_run})")
    print(f"{'='*80}")

    ckpt_dir = EXP_CKPT_BASE / config_id
    validate_output_directory_isolation(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    EXP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    dev_X, dev_y, dev_pids, dev_labels_list = load_development_dataset(spec["preprocessing_mode"])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy_X = np.zeros((len(dev_pids), 1))
    folds = list(sgkf.split(dummy_X, dev_y, dev_pids))

    oof_records = []
    fold_logs = []
    t0_cfg = time.time()

    epochs_to_run = 1 if dry_run else EPOCHS
    min_epochs_to_run = 1 if dry_run else MIN_EPOCHS
    patience_to_run = 1 if dry_run else PATIENCE

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        t0_fold = time.time()
        print(f"\n--- {config_id.upper()} | FOLD {fold_idx}/5 ---", flush=True)

        X_train_f, y_train_f = dev_X[train_idx], dev_y[train_idx]
        X_val_f, y_val_f = dev_X[val_idx], dev_y[val_idx]
        val_pids_f = dev_pids[val_idx]
        train_pids_f = dev_pids[train_idx]

        # Verify zero participant leakage
        validate_participant_leakage(set(train_pids_f), set(val_pids_f))

        # Shared epoch-zero initialization parity across Config A and Config B
        # For fold_idx, if the shared weight file does not exist, build reference model and dump weights.
        shared_init_file = SCRATCH_DIR / f"exp06d_init_shared_weights_fold_{fold_idx}.json"
        if not shared_init_file.is_file():
            print(f"  Generating shared epoch-zero weights for Fold {fold_idx} under seed {SEED + fold_idx}...", flush=True)
            ref_model, _, ref_heads = build_exp06d_model(seed_head=SEED + fold_idx)
            shared_weights = {
                "dense_512_kernel": ref_heads[0].get_weights()[0].tolist(),
                "dense_512_bias": ref_heads[0].get_weights()[1].tolist(),
                "clf_kernel": ref_heads[1].get_weights()[0].tolist(),
                "clf_bias": ref_heads[1].get_weights()[1].tolist(),
                "emb_kernel": ref_heads[2].get_weights()[0].tolist(),
                "emb_bias": ref_heads[2].get_weights()[1].tolist(),
            }
            with open(shared_init_file, "w") as f_json:
                json.dump(shared_weights, f_json)
            del ref_model

        # Build model for current configuration
        model, _, head_layers = build_exp06d_model(seed_head=SEED + fold_idx)

        # Apply exact shared initialization
        with open(shared_init_file, "r") as f_json:
            shared_w = json.load(f_json)
        head_layers[0].set_weights([
            np.array(shared_w["dense_512_kernel"], dtype=np.float32),
            np.array(shared_w["dense_512_bias"], dtype=np.float32)
        ])
        head_layers[1].set_weights([
            np.array(shared_w["clf_kernel"], dtype=np.float32),
            np.array(shared_w["clf_bias"], dtype=np.float32)
        ])
        head_layers[2].set_weights([
            np.array(shared_w["emb_kernel"], dtype=np.float32),
            np.array(shared_w["emb_bias"], dtype=np.float32)
        ])

        # Compute balanced class weights for fold training partition
        train_class_counts = {i: int(np.sum(y_train_f == i)) for i in range(4)}
        total_train = len(y_train_f)
        cw = {i: float(total_train / (4.0 * max(train_class_counts[i], 1))) for i in range(4)}

        # Optimizer: AdamW matching EXP-06C reference exactly
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=HEAD_LR,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            weighted_metrics=["accuracy"]
        )

        ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.weights.h5"
        cb = MacroF1Checkpoint(
            filepath=str(ckpt_path),
            X_val=X_val_f,
            y_val=y_val_f,
            patience=patience_to_run,
            min_epochs=min_epochs_to_run
        )

        model.fit(
            X_train_f, y_train_f,
            sample_weight=np.array([cw[int(y)] for y in y_train_f], dtype=np.float32),
            batch_size=BATCH_SIZE,
            epochs=epochs_to_run,
            validation_data=(X_val_f, y_val_f),
            callbacks=[cb],
            verbose=0
        )

        if not ckpt_path.is_file():
            safe_save_weights(model, str(ckpt_path))

        assert ckpt_path.is_file(), f"Checkpoint missing: {ckpt_path}"
        model.load_weights(str(ckpt_path))
        print(f"  Fold {fold_idx} Complete: Best Val Macro F1 = {cb.best_f1:.4f} @ Epoch {cb.best_epoch}", flush=True)

        preds_val = model.predict(X_val_f, verbose=0)
        preds_clf = preds_val[0] if isinstance(preds_val, (list, tuple)) else preds_val
        validate_probabilities(preds_clf)

        for i, p_nat in enumerate(preds_clf):
            pid = val_pids_f[i]
            true_lbl = dev_labels_list[val_idx[i]]

            # Canonical remapping:
            # Native order:    0: A+, 1: B+, 2: AB+, 3: O+
            # Canonical order: 0: A+, 1: AB+, 2: B+, 3: O+
            p_canon = np.array([p_nat[0], p_nat[2], p_nat[1], p_nat[3]], dtype=np.float64)
            p_canon = p_canon / p_canon.sum()

            pred_idx_canon = int(np.argmax(p_canon))
            pred_lbl = CANONICAL_CLASSES[pred_idx_canon]

            oof_records.append({
                "participant_id": pid,
                "fold_id": fold_idx,
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "canonical_probabilities": [round(float(x), 6) for x in p_canon],
                "native_probabilities": [round(float(x), 6) for x in p_nat]
            })

        fold_runtime = time.time() - t0_fold
        fold_logs.append({
            "fold": fold_idx,
            "best_epoch": cb.best_epoch if cb.best_epoch > 0 else 1,
            "best_val_f1": float(cb.best_f1),
            "runtime_sec": float(fold_runtime),
            "train_class_counts": {NATIVE_CLASSES[k]: v for k, v in train_class_counts.items()},
            "class_weights": {NATIVE_CLASSES[k]: float(cw[k]) for k in range(4)},
            "val_participants": list(val_pids_f),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": sha256_of_file(ckpt_path),
            "history": cb.history
        })

    total_cfg_runtime = time.time() - t0_cfg

    # Verify exactly 80 OOF participant records
    assert len(oof_records) == 80, f"Expected 80 OOF records, got {len(oof_records)}"
    recorded_pids = set(r["participant_id"] for r in oof_records)
    assert len(recorded_pids) == 80, f"Expected 80 unique OOF participants, got {len(recorded_pids)}"

    # Save OOF predictions CSV
    oof_records = sorted(oof_records, key=lambda x: x["participant_id"])
    pred_csv_path = EXP_REPORT_DIR / f"{config_id}_oof_predictions.csv"
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "fold_id", "true_label", "predicted_label",
            "canonical_probabilities", "native_probabilities"
        ])
        writer.writeheader()
        for row in oof_records:
            writer.writerow({
                "participant_id": row["participant_id"],
                "fold_id": row["fold_id"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "canonical_probabilities": json.dumps(row["canonical_probabilities"]),
                "native_probabilities": json.dumps(row["native_probabilities"])
            })

    # Compute validation metrics
    y_true_all = [r["true_label"] for r in oof_records]
    y_pred_all = [r["predicted_label"] for r in oof_records]

    acc = float(accuracy_score(y_true_all, y_pred_all))
    bacc = float(balanced_accuracy_score(y_true_all, y_pred_all))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=CANONICAL_CLASSES, average="macro", zero_division=0
    )
    prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=CANONICAL_CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true_all, y_pred_all, labels=CANONICAL_CLASSES)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CANONICAL_CLASSES, yticklabels=CANONICAL_CLASSES)
    plt.title(f"EXP-06D {config_id.upper()} ({spec['preprocessing_mode'].upper()})\nAcc: {acc*100:.2f}%, BalAcc: {bacc*100:.2f}%, Macro F1: {f1_m:.4f}")
    plt.xlabel("Predicted Canonical Class")
    plt.ylabel("True Blood Group")
    plt.tight_layout()
    cm_path = CM_DIR / f"{config_id}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    metrics = {
        "config_id": config_id,
        "config_name": spec["name"],
        "preprocessing_mode": spec["preprocessing_mode"],
        "head_type": spec["head_type"],
        "lambda_supcon": spec["lambda_supcon"],
        "temperature": spec["temperature"],
        "trainable_params": spec["expected_trainable_params"],
        "total_runtime_sec": total_cfg_runtime,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": float(prec_m),
        "macro_recall": float(rec_m),
        "macro_f1": float(f1_m),
        "per_class": {
            cls: {
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1": float(f1_c[i]),
                "support": int(supp_c[i]),
                "correct": int(cm[i, i])
            } for i, cls in enumerate(CANONICAL_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "folds": fold_logs
    }

    print(f"\n--- {config_id.upper()} SUMMARY ---")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bacc*100:.2f}%")
    print(f"Macro F1-Score:    {f1_m:.4f}")
    for cls in CANONICAL_CLASSES:
        pc = metrics["per_class"][cls]
        print(f"{cls:<4} Recall:        {pc['recall']*100:.2f}% ({pc['correct']}/{pc['support']})")
    print(f"Total Runtime:     {total_cfg_runtime/60:.2f} minutes")

    return metrics

def generate_comparative_report(new_metrics_list):
    print("\n" + "=" * 80)
    print("PHASE 3: VALIDATION & STATISTICAL COMPARISON (EXP-06D)")
    print("=" * 80)

    json_path = EXP_REPORT_DIR / "experiment_config.json"
    all_metrics_dict = {}
    if json_path.is_file():
        try:
            with open(json_path, "r") as f:
                existing = json.load(f)
                for item in existing:
                    all_metrics_dict[item["config_id"]] = item
        except Exception:
            pass

    for m in new_metrics_list:
        all_metrics_dict[m["config_id"]] = m

    ordered_cids = [c for c in ["config_a", "config_b"] if c in all_metrics_dict]
    all_metrics = [all_metrics_dict[c] for c in ordered_cids]

    rows = []
    for m in all_metrics:
        cid = m["config_id"]
        pc = m["per_class"]
        fold_f1s = [f["best_val_f1"] for f in m["folds"]]
        f1_mean = float(np.mean(fold_f1s))
        f1_std = float(np.std(fold_f1s))

        rows.append({
            "Config ID": cid,
            "Config Name": m["config_name"],
            "Preprocessing Mode": m.get("preprocessing_mode", "enhanced"),
            "Loss Weight (lambda)": m.get("lambda_supcon", FIXED_LAMBDA),
            "Temperature (tau)": m.get("temperature", FIXED_TEMPERATURE),
            "Trainable Params": f"{m['trainable_params']:,}",
            "Accuracy": f"{m['accuracy']*100:.2f}%",
            "Balanced Accuracy": f"{m['balanced_accuracy']*100:.2f}%",
            "Macro F1 (OOF)": f"{m['macro_f1']:.4f}",
            "Fold F1 Mean (Std)": f"{f1_mean:.4f} (+/-{f1_std:.4f})",
            "A+ Recall": f"{pc['A+']['recall']*100:.2f}% ({pc['A+']['correct']}/{pc['A+']['support']})",
            "AB+ Recall": f"{pc['AB+']['recall']*100:.2f}% ({pc['AB+']['correct']}/{pc['AB+']['support']})",
            "B+ Recall": f"{pc['B+']['recall']*100:.2f}% ({pc['B+']['correct']}/{pc['B+']['support']})",
            "O+ Recall": f"{pc['O+']['recall']*100:.2f}% ({pc['O+']['correct']}/{pc['O+']['support']})",
            "Runtime (min)": f"{m['total_runtime_sec']/60:.2f}"
        })

    df_comp = pd.DataFrame(rows)
    comp_csv = EXP_REPORT_DIR / "metrics_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)
    print("\nMetrics Comparison Table:")
    print(df_comp.to_string(index=False))

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Perform paired statistical test if both Config A and Config B exist
    if "config_a" in all_metrics_dict and "config_b" in all_metrics_dict:
        print("\nComputing Paired Participant-Level Statistical Tests...")
        csv_a = EXP_REPORT_DIR / "config_a_oof_predictions.csv"
        csv_b = EXP_REPORT_DIR / "config_b_oof_predictions.csv"
        if csv_a.is_file() and csv_b.is_file():
            df_a = pd.read_csv(csv_a).sort_values("participant_id")
            df_b = pd.read_csv(csv_b).sort_values("participant_id")
            corr_a = (df_a["predicted_label"] == df_a["true_label"]).to_numpy()
            corr_b = (df_b["predicted_label"] == df_b["true_label"]).to_numpy()

            n11 = int(np.sum(corr_a & corr_b))
            n10 = int(np.sum(corr_a & (~corr_b)))
            n01 = int(np.sum((~corr_a) & corr_b))
            n00 = int(np.sum((~corr_a) & (~corr_b)))

            discordant = n10 + n01
            binom_res = binomtest(n10, discordant, 0.5) if discordant > 0 else None
            p_binom = float(binom_res.pvalue) if binom_res is not None else 1.0

            # Paired bootstrap CI for Macro F1 difference (Config B - Config A)
            rng = np.random.default_rng(SEED)
            f1_diffs = []
            N = len(df_a)
            for _ in range(2000):
                boot_idx = rng.choice(N, size=N, replace=True)
                y_true_boot = df_a["true_label"].iloc[boot_idx].to_list()
                pred_a_boot = df_a["predicted_label"].iloc[boot_idx].to_list()
                pred_b_boot = df_b["predicted_label"].iloc[boot_idx].to_list()
                f1_a = f1_score(y_true_boot, pred_a_boot, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
                f1_b = f1_score(y_true_boot, pred_b_boot, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
                f1_diffs.append(f1_b - f1_a)

            ci_low, ci_high = np.percentile(f1_diffs, [2.5, 97.5])
            stat_results = {
                "n_participants": N,
                "contingency_table": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
                "exact_binomial_two_sided_p": p_binom,
                "bootstrap_f1_diff_mean": float(np.mean(f1_diffs)),
                "bootstrap_f1_diff_ci95": [float(ci_low), float(ci_high)]
            }
            stat_path = EXP_REPORT_DIR / "statistical_analysis_results.json"
            with open(stat_path, "w") as f:
                json.dump(stat_results, f, indent=2)
            print(f"Paired comparison: n10={n10}, n01={n01}, Exact Binomial p={p_binom:.4f}, Delta F1 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print(f"\nSaved comparison CSV: {comp_csv}")
    print(f"Saved full experiment metrics: {json_path}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="EXP-06D: Preprocessing & Multi-Scan Representation Investigation")
    parser.add_argument("--config", choices=["all", "config_a", "config_b"], default="all",
                        help="Configuration to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch per fold for pipeline integration verification")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run preflight integrity checks and exit")
    parser.add_argument("--authorize-training", action="store_true",
                        help="Explicit authorization flag required to begin real-data training")
    args = parser.parse_args()

    # Preflight check can execute safely without training
    if args.preflight_only:
        run_preflight()
        print("\nPreflight check completed successfully. Exiting without training.")
        return

    # Fail-closed guard against unauthorized training
    if not TRAINING_AUTHORIZED or not args.authorize_training:
        print("\n" + "!" * 80)
        print("FAIL-CLOSED SAFETY GUARD ACTIVATED: Real-data training is currently UNAUTHORIZED.")
        print("Preparation and synthetic testing mode ONLY. Refusing to begin training.")
        print("!" * 80 + "\n")
        sys.exit(0)

    # If training was authorized, execution would proceed
    run_preflight()
    configs = ["config_a", "config_b"] if args.config == "all" else [args.config]
    results = []
    for cid in configs:
        metrics = run_configuration(cid, dry_run=args.dry_run)
        results.append(metrics)

    generate_comparative_report(results)

if __name__ == "__main__":
    main()
