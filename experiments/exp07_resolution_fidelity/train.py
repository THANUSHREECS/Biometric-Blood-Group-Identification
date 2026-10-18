#!/usr/bin/env python3
"""
EXP-07-RES-FIDELITY: Training Pipeline & Architecture Definition
==============================================================
Implements the controlled comparison of 112x112 vs 224x224 input resolution
under leakage-free validation protocols (Fixed-Duration & Nested Inner-Val).

Strict Safeguards:
  - Fail-closed authorization guard: TRAINING_AUTHORIZED must be True.
  - Zero outer-fold monitoring during training.
  - Zero test cohort or hospital cohort access.
  - Exactly matched head parameters: 1,968,772 across 6 tensors.
  - Strictly frozen DenseNet-121 backbone: 7,037,504 non-trainable parameters.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Ensure project imports
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white,
    crop_mfs100_foreground, resize_to_target
)

from experiments.exp07_resolution_fidelity.config import (
    EXP_DIR, CKPT_DIR, REPORT_DIR, SPLIT_CSV, HOSPITAL_DIR,
    PROTECTED_CKPT_DIRS, TRAINING_AUTHORIZED,
    NATIVE_CLASSES, NATIVE_REV_MAP, SEED, BATCH_SIZE,
    N_OUTER_FOLDS, FIXED_EPOCHS, HEAD_LR, WEIGHT_DECAY,
    FIXED_TEMPERATURE, FIXED_LAMBDA, CONDITIONS, MODEL_SPEC
)

def sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def supervised_contrastive_loss(feature_vectors, labels, temperature=FIXED_TEMPERATURE):
    """
    Supervised Contrastive Loss with fixed temperature (tau = 0.05).
    Operates on L2-normalized participant embeddings.
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

class E2ResolutionModel(tf.keras.Model):
    """
    E2 Multi-Instance Three-Scan Model supporting variable input spatial resolution.
    Backbone is strictly frozen.
    """
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

def build_e2_model(target_resolution=(112, 112), seed_head=SEED):
    """
    Constructs the E2 model for target resolution (112, 112) or (224, 224).
    DenseNet-121 backbone is strictly frozen.
    Classification and projection heads match EXP-06C Config A linear reference exactly.
    """
    h, w = target_resolution
    inputs = tf.keras.Input(shape=(3, h, w, 3), name=f"scans_{h}x{w}")
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    # Pretrained ImageNet DenseNet-121 (strictly frozen)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(h, w, 3))
    base.trainable = False

    pool = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    f1, f2, f3 = pool(base(s1)), pool(base(s2)), pool(base(s3))
    fused = tf.keras.layers.Concatenate(name="fused_features")([f1, f2, f3])

    # Shared classifier branch
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

    # Linear projection head for Supervised Contrastive Loss
    emb = tf.keras.layers.Dense(
        128, activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head + 2),
        bias_initializer="zeros",
        name="emb"
    )
    out_emb = emb(fused)

    head_layers = [dense_512, clf, emb]
    model_name = f"E2_EXP07_{h}x{w}"

    model = E2ResolutionModel(
        inputs=inputs,
        outputs=[out_clf, out_emb],
        base_model=base,
        head_layers=head_layers,
        lambda_supcon=FIXED_LAMBDA,
        temperature=FIXED_TEMPERATURE,
        name=model_name
    )
    return model, base, head_layers

def preprocess_scan_fidelity(image_path, target_resolution=(112, 112)):
    """
    Minimal, non-destructive preprocessing pipeline:
      1. Composite alpha transparency onto solid white background.
      2. Platen foreground crop (removes empty scanner borders).
      3. High-quality Lanczos resize to target_resolution.
      4. Raw float32 array in [0, 255] (NO CLAHE, NO bilateral filtering).
    """
    raw_pil = load_image_to_pil(image_path)
    rgb_pil = composite_alpha_on_white(raw_pil)
    cropped_pil, _ = crop_mfs100_foreground(rgb_pil)
    resized_pil = resize_to_target(cropped_pil, target_size=target_resolution)
    img_array = np.array(resized_pil, dtype=np.float32)
    return img_array

def load_development_cohort(target_resolution=(112, 112)):
    """
    Loads strictly the 80 approved development participants.
    Quarantine safeguards prevent any access to locked test or hospital data.
    """
    assert SPLIT_CSV.is_file(), f"Split CSV not found: {SPLIT_CSV}"
    df_split = pd.read_csv(SPLIT_CSV)

    # Quarantine Guard Assertions
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
    test_df = df_split[df_split["split"] == "TEST"].copy()

    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    assert len(dev_pids) == 80, f"Expected 80 dev participants, got {len(dev_pids)}"
    assert len(test_pids) == 21, f"Expected 21 test participants, got {len(test_pids)}"
    assert len(dev_pids.intersection(test_pids)) == 0, "DATA LEAKAGE: Overlap with locked test set!"

    pids_unique = sorted(dev_pids)
    X_list = []
    y_list = []

    print(f"Loading Dev-80 cohort at resolution {target_resolution} (minimal preprocessing)...")
    for pid in pids_unique:
        p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
        assert len(p_scans) == 3, f"Participant {pid} does not have exactly 3 scans!"

        bg = p_scans.iloc[0]["blood_group"]
        assert bg in NATIVE_CLASSES, f"Unknown blood group: {bg}"
        y_list.append(NATIVE_REV_MAP[bg])

        three_scans = []
        for _, row in p_scans.iterrows():
            img_p = Path(row["scan_path"])
            assert img_p.is_file(), f"Scan file not found: {img_p}"
            # Assert path is NOT inside test or hospital
            assert "hospital" not in str(img_p).lower(), "QUARANTINE BREACH: Hospital scan in dev loader!"
            arr = preprocess_scan_fidelity(img_p, target_resolution=target_resolution)
            three_scans.append(arr)

        X_list.append(np.array(three_scans, dtype=np.float32))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    pids = np.array(pids_unique)

    print(f"Loaded cohort: X shape={X.shape}, y shape={y.shape}, participants={len(pids)}")
    return X, y, pids

def train_condition(condition_key: str, evaluation_regime: str = "fixed_duration"):
    """
    Executes 5-fold cross-validation for a specific condition under the specified evaluation regime.
    
    Regimes:
      - 'fixed_duration': 80 fixed epochs. Outer validation fold is NOT monitored during training.
                          Final epoch weights evaluated on outer fold. Leakage = 0.
      - 'nested_inner': 64 train participants split into 48 inner-train and 16 inner-val.
                        Best checkpoint selected on inner-val. Evaluated once on outer fold.
    """
    # FAIL-CLOSED AUTHORIZATION CHECK
    if not TRAINING_AUTHORIZED:
        raise PermissionError(
            "\n[SAFETY BLOCK] TRAINING_AUTHORIZED is False.\n"
            "Execution on real participant data is locked. Run with synthetic test first.\n"
            "Authorization required from senior research supervisor."
        )

    spec = CONDITIONS[condition_key]
    res = spec["input_resolution"]
    print("\n" + "=" * 80)
    print(f"EXP-07 TRAINING: {spec['name']}")
    print(f"Regime: {evaluation_regime.upper()} | Resolution: {res} | Folds: {N_OUTER_FOLDS}")
    print("=" * 80)

    X, y, pids = load_development_cohort(target_resolution=res)
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(np.zeros((len(pids), 1)), y, pids))

    out_dir = CKPT_DIR / condition_key
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_predictions = np.zeros((len(pids), 4), dtype=np.float32)

    for f_idx, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\n--- Fold {f_idx}/{N_OUTER_FOLDS} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Shared head seed per fold to guarantee identical initializations
        model, base, heads = build_e2_model(target_resolution=res, seed_head=SEED + f_idx)
        # Learning rate schedule with CosineDecay over 80 epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=HEAD_LR,
            decay_steps=FIXED_EPOCHS * (len(train_idx) // BATCH_SIZE),
            alpha=1e-3
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        if evaluation_regime == "fixed_duration":
            # ZERO outer validation callback or monitoring
            print(f"  Training for fixed {FIXED_EPOCHS} epochs (Zero outer-fold feedback)...")
            model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=FIXED_EPOCHS,
                verbose=0
            )
            # Evaluate strictly at final epoch
            preds, _ = model.predict(X_val, verbose=0)
            oof_predictions[val_idx] = preds

        elif evaluation_regime == "nested_inner":
            # Inner 4-fold split (48 inner-train, 16 inner-val)
            inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + f_idx)
            inner_folds = list(inner_sgkf.split(np.zeros((len(train_idx), 1)), y_train, pids[train_idx]))
            it_idx, iv_idx = inner_folds[0]

            X_it, y_it = X_train[it_idx], y_train[it_idx]
            X_iv, y_iv = X_train[iv_idx], y_train[iv_idx]

            print(f"  Inner-split: 48 Inner-Train / 16 Inner-Val. Monitoring inner-val Macro F1...")
            # Custom inner-callback (safely initialized)
            best_inner_f1 = -1.0
            best_inner_weights = model.get_weights()

            for ep in range(1, FIXED_EPOCHS + 1):
                model.fit(X_it, y_it, batch_size=BATCH_SIZE, epochs=1, verbose=0)
                iv_preds, _ = model.predict(X_iv, verbose=0)
                iv_pred_cls = np.argmax(iv_preds, axis=1)
                iv_f1 = f1_score(y_iv, iv_pred_cls, average="macro", zero_division=0)
                if iv_f1 > best_inner_f1:
                    best_inner_f1 = iv_f1
                    best_inner_weights = model.get_weights()

            model.set_weights(best_inner_weights)
            preds, _ = model.predict(X_val, verbose=0)
            oof_predictions[val_idx] = preds

        val_acc = accuracy_score(y_val, np.argmax(oof_predictions[val_idx], axis=1))
        val_f1 = f1_score(y_val, np.argmax(oof_predictions[val_idx], axis=1), average="macro", zero_division=0)
        print(f"  Fold {f_idx} Final Out-of-Fold Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f}")

    oof_acc = accuracy_score(y, np.argmax(oof_predictions, axis=1))
    oof_bal_acc = balanced_accuracy_score(y, np.argmax(oof_predictions, axis=1))
    oof_f1 = f1_score(y, np.argmax(oof_predictions, axis=1), average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(f"OVERALL OUT-OF-FOLD RESULTS ({condition_key}):")
    print(f"  Accuracy:         {oof_acc*100:.2f}% ({int(oof_acc*len(y))}/{len(y)})")
    print(f"  Balanced Accuracy:{oof_bal_acc*100:.2f}%")
    print(f"  Macro F1:         {oof_f1:.4f}")
    print("=" * 80)

    # Save OOF predictions
    np.save(out_dir / "oof_predictions.npy", oof_predictions)
    return oof_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-07 Training Pipeline")
    parser.add_argument("--condition", choices=["arm_a_112", "arm_b_224", "all"], default="all")
    parser.add_argument("--regime", choices=["fixed_duration", "nested_inner"], default="fixed_duration")
    args = parser.parse_args()

    try:
        if args.condition == "all":
            train_condition("arm_a_112", evaluation_regime=args.regime)
            train_condition("arm_b_224", evaluation_regime=args.regime)
        else:
            train_condition(args.condition, evaluation_regime=args.regime)
    except PermissionError as e:
        print(e)
        sys.exit(0)
