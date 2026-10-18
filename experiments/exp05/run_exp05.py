#!/usr/bin/env python3
"""
experiments/exp05/run_exp05.py
==============================
EXP-05: Controlled Three-Scan Feature Fusion for Fingerprint Blood Group Classification.

Research Question:
Can a lightweight feature-fusion module (Gated Feature Fusion vs. Cross-Scan Multi-Head
Self-Attention) improve participant-level blood group classification compared with the
existing concatenation baseline, while keeping the pretrained DenseNet-121 backbone frozen?

Three Experimental Configurations:
- Config A: Concatenation Baseline (Reference)
            Three 1024-d scan embeddings concatenated -> 3072-d representation.
            Heads trained at LR = 1e-3, DenseNet-121 strictly frozen.
            1,968,772 trainable parameters (6 trainable variables).
- Config B: Gated Feature Fusion
            Lightweight projection neck computes scan-importance gate weights (alpha_1, alpha_2, alpha_3).
            Normalized with Softmax (sum=1, non-negative).
            Fused representation: F_fused = sum(alpha_i * F_i) in R^1024.
            789,381 trainable parameters (10 trainable variables: 4 neck + 6 head).
- Config C: Cross-Scan Multi-Head Self-Attention
            Three 1024-d scan embeddings projected to 256-d -> stacked into sequence (B, 3, 256).
            Multi-Head Attention (4 heads, key_dim=64) + Residual + LayerNorm.
            Position-wise FFN (256 -> 512 -> 256) + Residual + LayerNorm.
            Flattened attended sequence -> 768-d representation.
            1,283,716 trainable parameters (24 trainable variables: 18 neck + 6 head).

Strict Protocols:
- 80 Development participants only (64 TRAIN, 16 VALIDATION per fold from participant_split.csv).
- 5-Fold StratifiedGroupKFold cross-validation (participants grouped, zero leakage).
- 21-participant locked test cohort and 26-participant hospital cohort are STRICTLY QUARANTINED.
- Preprocessing: C5 without data augmentation.
- Loss: 1.0 * Sparse Categorical Cross-Entropy + 0.1 * Supervised Contrastive Loss (tau=0.07).
- Sample weighting: w_i = N / (4 * N_c) applied exclusively to classification loss.
- Checkpoint selection: Best validation Macro F1 on held-out fold.
- Duration: EPOCHS=150, MIN_EPOCHS=100, PATIENCE=15.
- Shared initialization: clf layer (512, 4) initialized identically across Configs A, B, and C per fold.
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
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5

EXP_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp05" / "checkpoints"
EXP_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp05"
CM_DIR = EXP_REPORT_DIR / "confusion_matrices"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

EXP_CKPT_BASE.mkdir(parents=True, exist_ok=True)
EXP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

# Protected Checkpoints for Pre/Post Integrity Verification
ORIG_E2_DIR = PROJECT_ROOT / "models" / "exp6_folds"
ORIG_CHECKPOINTS = [ORIG_E2_DIR / f"model_fold_{i}.keras" for i in range(1, 6)]
EXP03_CHECKPOINTS = [PROJECT_ROOT / "experiments" / "exp03" / "checkpoints" / "arm1" / f"arm1_best_model_fold{i}.keras" for i in range(1, 6)]
PROD_MODEL = PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras"

# Global Hyperparameters
SEED = 42
BATCH_SIZE = 16
EPOCHS = 150
MIN_EPOCHS = 100
PATIENCE = 15
WEIGHT_DECAY = 1e-4
HEAD_LR = 1e-3
LAMBDA_SUPCON = 0.1
SUPCON_TEMP = 0.07

CONFIG_SPECS = {
    "config_a": {
        "name": "Config A: Concatenation Baseline",
        "fusion_type": "concatenation",
        "fused_dim": 3072,
        "expected_trainable_vars": 6,
        "expected_trainable_params": 1968772
    },
    "config_b": {
        "name": "Config B: Gated Feature Fusion",
        "fusion_type": "gated",
        "fused_dim": 1024,
        "expected_trainable_vars": 10,
        "expected_trainable_params": 789381
    },
    "config_c": {
        "name": "Config C: Cross-Scan Multi-Head Self-Attention",
        "fusion_type": "attention",
        "fused_dim": 768,
        "expected_trainable_vars": 24,
        "expected_trainable_params": 1283716
    }
}

NATIVE_CLASSES = ["A+", "B+", "AB+", "O+"]
NATIVE_REV_MAP = {c: i for i, c in enumerate(NATIVE_CLASSES)}
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------------------------------------------------------
# 1. LOSS & MODEL ARCHITECTURE
# -------------------------------------------------------------------------
def supervised_contrastive_loss(labels, feature_vectors, temperature=SUPCON_TEMP):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Audited implementation matching experiment6, EXP-03, and EXP-04.
    """
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

class GatedScanFusion(tf.keras.layers.Layer):
    """
    Config B: Gated Feature Fusion Module.
    Computes scan-level scalar logits, normalizes with Softmax, and forms a convex sum.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate_dense = tf.keras.layers.Dense(128, activation="relu", name="gate_dense1")
        self.gate_logit = tf.keras.layers.Dense(1, activation=None, name="gate_logit")
        self.softmax = tf.keras.layers.Softmax(axis=1, name="gate_softmax")

    def call(self, inputs, return_gate_weights=False):
        stacked = tf.stack(inputs, axis=1) # (B, 3, 1024)
        h = self.gate_dense(stacked) # (B, 3, 128)
        logits = tf.squeeze(self.gate_logit(h), axis=-1) # (B, 3)
        weights = self.softmax(logits) # (B, 3)
        expanded_weights = tf.expand_dims(weights, axis=-1) # (B, 3, 1)
        fused = tf.reduce_sum(stacked * expanded_weights, axis=1) # (B, 1024)
        if return_gate_weights:
            return fused, weights
        return fused

    def get_config(self):
        return super().get_config()

class CrossScanAttentionFusion(tf.keras.layers.Layer):
    """
    Config C: Cross-Scan Multi-Head Self-Attention Fusion Module.
    Projects 3 scan embeddings to 256-d, performs MHA with residual & LayerNorm,
    applies position-wise FFN with residual & LayerNorm, and flattens to 768-d.
    """
    def __init__(self, num_heads=4, key_dim=64, ffn_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ffn_dim = ffn_dim
        self.proj = tf.keras.layers.Dense(256, activation=None, name="attn_input_proj")
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name="cross_scan_mha")
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="attn_ln1")
        self.ffn1 = tf.keras.layers.Dense(ffn_dim, activation="relu", name="ffn_dense1")
        self.ffn2 = tf.keras.layers.Dense(256, activation=None, name="ffn_dense2")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="attn_ln2")
        self.flatten = tf.keras.layers.Flatten(name="attn_flatten")

    def call(self, inputs):
        stacked = tf.stack(inputs, axis=1) # (B, 3, 1024)
        x = self.proj(stacked) # (B, 3, 256)
        attn_out = self.mha(query=x, key=x, value=x) # (B, 3, 256)
        x = self.ln1(x + attn_out) # (B, 3, 256)
        ffn_out = self.ffn2(self.ffn1(x)) # (B, 3, 256)
        x = self.ln2(x + ffn_out) # (B, 3, 256)
        fused = self.flatten(x) # (B, 768)
        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ffn_dim": self.ffn_dim
        })
        return config

class E2FusionModel(tf.keras.Model):
    """
    EXP-05 Model wrapper with custom multi-task train_step.
    Applies inverse-frequency sample weights exclusively to classification loss.
    """
    def __init__(self, inputs, outputs, base_model, fusion_layer, head_layers, config_id, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.base_model = base_model
        self.fusion_layer = fusion_layer
        self.head_layers = head_layers
        self.config_id = config_id

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        y_clf = y[0]
        y_emb = y[1]
        sw_clf = sample_weight[0] if sample_weight is not None else None

        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            pred_clf, pred_emb = preds[0], preds[1]

            # 1. Classification loss with sample weights
            loss_c_raw = tf.keras.losses.sparse_categorical_crossentropy(y_clf, pred_clf)
            if sw_clf is not None:
                loss_clf = tf.reduce_mean(loss_c_raw * tf.cast(sw_clf, tf.float32))
            else:
                loss_clf = tf.reduce_mean(loss_c_raw)

            # 2. Supervised Contrastive loss on embeddings (unweighted)
            loss_supcon = supervised_contrastive_loss(y_emb, pred_emb, temperature=SUPCON_TEMP)

            # 3. Total Multi-Task Loss
            total_loss = 1.0 * loss_clf + LAMBDA_SUPCON * loss_supcon

        # Update trainable parameters (head + fusion neck; backbone is frozen)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "clf_loss": loss_clf,
            "supcon_loss": loss_supcon
        }

def build_exp05_model(config_id: str, seed_clf: int = 42):
    """
    Constructs the E2 3-scan feature fusion model for EXP-05.
    DenseNet-121 is strictly frozen.
    """
    inputs = tf.keras.Input(shape=(3, 112, 112, 3), name="participant_scans")
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    # Pretrained ImageNet DenseNet-121 (strictly frozen)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False

    pool = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    f1, f2, f3 = pool(base(s1)), pool(base(s2)), pool(base(s3))

    if config_id == "config_a":
        fused = tf.keras.layers.Concatenate(name="fused_features")([f1, f2, f3])
        fusion_layer = None
    elif config_id == "config_b":
        fusion_layer = GatedScanFusion(name="gated_fusion")
        fused = fusion_layer([f1, f2, f3])
    elif config_id == "config_c":
        fusion_layer = CrossScanAttentionFusion(name="attention_fusion")
        fused = fusion_layer([f1, f2, f3])
    else:
        raise ValueError(f"Unknown config_id: {config_id}")

    dense_512 = tf.keras.layers.Dense(512, activation="relu", name="dense_512")
    dropout_05 = tf.keras.layers.Dropout(0.5, name="dropout_0.5")

    # Seeded clf layer for shared initial weights across configurations
    clf_kernel_init = tf.keras.initializers.GlorotUniform(seed=seed_clf)
    clf = tf.keras.layers.Dense(4, activation="softmax", kernel_initializer=clf_kernel_init, bias_initializer="zeros", name="clf")
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")

    d = dense_512(fused)
    drop = dropout_05(d)
    out_clf = clf(drop)
    out_emb = emb(fused)

    head_layers = [dense_512, clf, emb]
    model = E2FusionModel(
        inputs=inputs,
        outputs=[out_clf, out_emb],
        base_model=base,
        fusion_layer=fusion_layer,
        head_layers=head_layers,
        config_id=config_id,
        name=f"E2_EXP05_{config_id}"
    )
    return model, base, fusion_layer, head_layers

def safe_save_weights(model, filepath, max_retries=10, delay=1.0):
    p = Path(filepath)
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
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(self.y_val, y_pred)

        self.history.append({
            "epoch": epoch + 1,
            "train_loss": float(logs.get("loss", 0.0)) if logs else 0.0,
            "val_f1": float(macro_f1),
            "val_acc": float(acc)
        })

        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.best_epoch = epoch + 1
            self.wait = 0
            safe_save_weights(self.model, self.filepath)
            print(f"    Epoch {epoch+1:3d}: New Best Val Macro F1 = {macro_f1:.4f} (Acc = {acc*100:.1f}%) [SAVED]", flush=True)
        else:
            self.wait += 1
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}: Val Macro F1 = {macro_f1:.4f} (Best: {self.best_f1:.4f} @ Ep {self.best_epoch})", flush=True)

        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            print(f"    Early stopping triggered at epoch {epoch+1} (Best epoch: {self.best_epoch}, Best Val Macro F1: {self.best_f1:.4f})", flush=True)
            self.model.stop_training = True

# -------------------------------------------------------------------------
# 2. DATASET LOADING & QUARANTINE ASSERTIONS
# -------------------------------------------------------------------------
def load_development_dataset():
    print("\n[Data Loader] Loading 80 Development Participants from participant_split.csv...")
    df_split = pd.read_csv(SPLIT_CSV)
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()

    test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
    dev_pids_check = set(dev_df["participant_id"].unique())
    assert len(test_pids) == 21, f"Expected 21 test subjects, found {len(test_pids)}"
    assert len(dev_pids_check) == 80, f"Expected 80 dev subjects, found {len(dev_pids_check)}"
    assert len(test_pids & dev_pids_check) == 0, "CRITICAL ERROR: Locked test participants leak into dev cohort!"

    grouped = dev_df.groupby("participant_id")
    dev_pids_list = []
    dev_labels_list = []
    dev_y_native = []
    dev_X_list = []

    for pid, group in grouped:
        if len(group) != 3:
            raise ValueError(f"Participant {pid} has {len(group)} scans; expected exactly 3.")
        scans_c5 = []
        for _, row in group.iterrows():
            p = Path(row["scan_path"])
            if not p.is_file():
                raise FileNotFoundError(f"Missing scan file: {p}")
            arr, _, _ = transform_c5(str(p))
            scans_c5.append(arr)
        dev_X_list.append(scans_c5)
        lbl = group.iloc[0]["blood_group"]
        dev_labels_list.append(lbl)
        dev_y_native.append(NATIVE_REV_MAP[lbl])
        dev_pids_list.append(pid)

    dev_X = np.array(dev_X_list, dtype=np.float32)  # (80, 3, 112, 112, 3)
    dev_y = np.array(dev_y_native, dtype=np.int32)   # (80,)
    dev_pids = np.array(dev_pids_list)

    print(f"Loaded dev_X: shape={dev_X.shape}, dev_y: shape={dev_y.shape}")
    print(f"Total participants: {len(dev_pids)}")
    print(f"Class distribution: {pd.Series(dev_labels_list).value_counts().to_dict()}")
    return dev_X, dev_y, dev_pids, dev_labels_list

# -------------------------------------------------------------------------
# 3. PREFLIGHT VERIFICATION
# -------------------------------------------------------------------------
def run_preflight(dev_X, dev_y, dev_pids):
    print("\n" + "=" * 80)
    print("PHASE 1: FINAL PREFLIGHT VERIFICATION")
    print("=" * 80)

    # 1. Baseline checkpoint hash check
    print("\n[Preflight 1] Baseline Checkpoint SHA-256 Hashes:")
    for p in ORIG_CHECKPOINTS:
        if p.is_file():
            print(f"  Orig E2:  {p.name} -> {sha256_of_file(p)[:16]}...")
    for p in EXP03_CHECKPOINTS:
        if p.is_file():
            print(f"  EXP-03:   {p.name} -> {sha256_of_file(p)[:16]}...")
    if PROD_MODEL.is_file():
        print(f"  Prod:     {PROD_MODEL.name} -> {sha256_of_file(PROD_MODEL)[:16]}...")

    # 2. Variable partition verification
    print("\n[Preflight 2] Model Parameter and Trainable Variable Verification:")
    for cid, spec in CONFIG_SPECS.items():
        m, b, fl, hl = build_exp05_model(cid, seed_clf=42)
        all_vars = m.trainable_variables
        tot_params = m.count_params()
        trainable_params = sum(v.numpy().size for v in all_vars)
        frozen_bb_params = b.count_params()

        assert b.trainable is False, f"[{cid}] Backbone is not frozen!"
        assert frozen_bb_params == 7037504, f"[{cid}] Frozen backbone param mismatch!"
        assert len(all_vars) == spec["expected_trainable_vars"], f"[{cid}] Expected {spec['expected_trainable_vars']} vars, got {len(all_vars)}"
        assert trainable_params == spec["expected_trainable_params"], f"[{cid}] Expected {spec['expected_trainable_params']} params, got {trainable_params}"
        print(f"  {cid.upper()}: PASS | trainable_vars={len(all_vars)}, trainable_params={trainable_params:,}, frozen_bb=7,037,504")

    # 3. Folds and participant split verification
    print("\n[Preflight 3] 5-Fold StratifiedGroupKFold Integrity:")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy_X = np.zeros((len(dev_pids), 1))
    folds = list(sgkf.split(dummy_X, dev_y, dev_pids))
    assert len(folds) == 5
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        assert len(t_idx) == 64 and len(v_idx) == 16
        leak = set(dev_pids[t_idx]) & set(dev_pids[v_idx])
        assert len(leak) == 0, f"Participant leakage in fold {f_idx}!"
    print("  Folds 1-5: PASS | Exactly 64 train / 16 val per fold, zero participant leakage.")

    print("\n[Preflight 4] Preflight Complete — ALL INTEGRITY CHECKS PASSED.")
    print("=" * 80)
    return True

# -------------------------------------------------------------------------
# 4. TRAINING EXECUTION ACROSS CONFIGURATIONS
# -------------------------------------------------------------------------
def run_configuration(config_id: str, dev_X, dev_y, dev_pids, dev_labels_list, dry_run=False):
    spec = CONFIG_SPECS[config_id]
    print(f"\n{'='*80}")
    print(f"RUNNING EXP-05: {spec['name'].upper()} (dry_run={dry_run})")
    print(f"{'='*80}")

    t0_config = time.time()
    ckpt_dir = EXP_CKPT_BASE / config_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy_X = np.zeros((len(dev_pids), 1))
    folds = list(sgkf.split(dummy_X, dev_y, dev_pids))

    oof_records = []
    fold_logs = []

    epochs_to_run = 1 if dry_run else EPOCHS
    min_epochs_to_run = 1 if dry_run else MIN_EPOCHS
    patience_to_run = 1 if dry_run else PATIENCE

    for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
        t0_fold = time.time()
        print(f"\n--- {config_id.upper()} | FOLD {fold_idx}/5 ---", flush=True)
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)

        train_pids_f = dev_pids[train_idx]
        val_pids_f = dev_pids[val_idx]
        X_train_f, y_train_f = dev_X[train_idx], dev_y[train_idx]
        X_val_f, y_val_f = dev_X[val_idx], dev_y[val_idx]

        train_class_counts = {i: int(sum(y_train_f == i)) for i in range(4)}
        n_train = len(y_train_f)
        cw = {i: n_train / (4.0 * train_class_counts[i]) for i in range(4)}
        sw = np.array([cw[y] for y in y_train_f], dtype=np.float32)

        # ---------------------------------------------------------------------
        # Shared Ephemeral Classification Head Initialization
        # ---------------------------------------------------------------------
        init_clf_file = SCRATCH_DIR / f"exp05_init_fold_{fold_idx}_clf.npz"
        if not init_clf_file.is_file():
            print(f"  Generating shared epoch-zero clf weights for Fold {fold_idx} under seed {SEED + fold_idx}...")
            ref_m, _, _, ref_hl = build_exp05_model("config_a", seed_clf=SEED + fold_idx)
            clf_weights = ref_m.get_layer("clf").get_weights()
            np.savez(str(init_clf_file), kernel=clf_weights[0], bias=clf_weights[1])
            del ref_m, ref_hl

        # Build model for current configuration
        model, base, fusion_layer, head_layers = build_exp05_model(config_id, seed_clf=SEED + fold_idx)

        # Load shared initial clf weights
        with np.load(str(init_clf_file)) as loaded_npz:
            model.get_layer("clf").set_weights([loaded_npz["kernel"], loaded_npz["bias"]])

        # Optimizer (AdamW at LR = 1e-3, weight decay = 1e-4)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY)
        model.compile(optimizer=optimizer)

        ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.weights.h5"
        cb = MacroF1Checkpoint(
            str(ckpt_path), X_val_f, y_val_f,
            patience=patience_to_run,
            min_epochs=min_epochs_to_run
        )

        model.fit(
            X_train_f, [y_train_f, y_train_f],
            sample_weight=[sw, np.ones_like(sw)],
            batch_size=BATCH_SIZE,
            epochs=epochs_to_run,
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

        for i, p_nat in enumerate(preds_clf):
            pid = val_pids_f[i]
            true_lbl = dev_labels_list[val_idx[i]]

            # Canonical remapping:
            # native indices: 0: A+, 1: B+, 2: AB+, 3: O+
            # canonical indices: 0: A+, 1: AB+, 2: B+, 3: O+
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

    total_config_runtime = time.time() - t0_config

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
    plt.title(f"EXP-05 {config_id.upper()} Confusion Matrix\nAcc: {acc*100:.2f}%, BalAcc: {bacc*100:.2f}%, Macro F1: {f1_m:.4f}")
    plt.xlabel("Predicted Canonical Class")
    plt.ylabel("True Blood Group")
    plt.tight_layout()
    cm_path = CM_DIR / f"{config_id}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    metrics = {
        "config_id": config_id,
        "config_name": spec["name"],
        "trainable_params": spec["expected_trainable_params"],
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": float(prec_m),
        "macro_recall": float(rec_m),
        "macro_f1": float(f1_m),
        "per_class": {
            CANONICAL_CLASSES[i]: {
                "precision": float(prec_c[i]),
                "recall": float(rec_c[i]),
                "f1_score": float(f1_c[i]),
                "support": int(supp_c[i]),
                "correct": int(cm[i, i])
            } for i in range(4)
        },
        "confusion_matrix": cm.tolist(),
        "total_runtime_sec": float(total_config_runtime),
        "folds": fold_logs
    }

    print(f"\n--- {config_id.upper()} SUMMARY ---")
    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bacc*100:.2f}%")
    print(f"Macro F1-Score:    {f1_m:.4f}")
    print(f"A+  Recall:        {rec_c[0]*100:.2f}% ({int(cm[0, 0])}/{int(supp_c[0])})")
    print(f"AB+ Recall:        {rec_c[1]*100:.2f}% ({int(cm[1, 1])}/{int(supp_c[1])})")
    print(f"B+  Recall:        {rec_c[2]*100:.2f}% ({int(cm[2, 2])}/{int(supp_c[2])})")
    print(f"O+  Recall:        {rec_c[3]*100:.2f}% ({int(cm[3, 3])}/{int(supp_c[3])})")
    print(f"Total Runtime:     {total_config_runtime/60:.2f} minutes")
    return metrics

# -------------------------------------------------------------------------
# 5. STATISTICAL COMPARISON & SUMMARY REPORT
# -------------------------------------------------------------------------
def generate_comparative_report(new_metrics_list):
    print("\n" + "=" * 80)
    print("PHASE 3: VALIDATION & STATISTICAL COMPARISON")
    print("=" * 80)

    json_path = EXP_REPORT_DIR / "experiment_config.json"
    all_metrics_dict = {}
    if json_path.is_file():
        try:
            with open(json_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    for item in existing:
                        all_metrics_dict[item["config_id"]] = item
        except Exception:
            pass

    for m in new_metrics_list:
        all_metrics_dict[m["config_id"]] = m

    ordered_cids = [c for c in ["config_a", "config_b", "config_c"] if c in all_metrics_dict]
    all_metrics = [all_metrics_dict[c] for c in ordered_cids]

    rows = []
    for m in all_metrics:
        cid = m["config_id"]
        pc = m["per_class"]
        # Fold standard deviation of macro F1
        f1_vals = [f["best_val_f1"] for f in m["folds"]]
        f1_mean = np.mean(f1_vals)
        f1_std = np.std(f1_vals)
        rows.append({
            "Config ID": cid,
            "Configuration Name": m["config_name"],
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

    # Save full JSON log
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Clean ephemeral scratch files only if all 3 configurations are completed
    if len(all_metrics) == 3:
        for p in SCRATCH_DIR.glob("exp05_init_fold_*.npz"):
            try:
                p.unlink()
            except Exception:
                pass

    print(f"\nSaved comparison CSV: {comp_csv}")
    print(f"Saved full experiment metrics: {json_path}")
    print("=" * 80)

# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EXP-05: Controlled Three-Scan Feature Fusion")
    parser.add_argument("--config", choices=["all", "config_a", "config_b", "config_c"], default="all",
                        help="Configuration to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 epoch per fold for pipeline integration verification")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run preflight integrity checks and exit")
    args = parser.parse_args()

    dev_X, dev_y, dev_pids, dev_labels = load_development_dataset()
    run_preflight(dev_X, dev_y, dev_pids)

    if args.preflight_only:
        print("\nPreflight check requested only. Exiting with status 0.")
        return

    configs_to_run = ["config_a", "config_b", "config_c"] if args.config == "all" else [args.config]
    results = []
    for cid in configs_to_run:
        metrics = run_configuration(cid, dev_X, dev_y, dev_pids, dev_labels, dry_run=args.dry_run)
        results.append(metrics)

    generate_comparative_report(results)

if __name__ == "__main__":
    main()
