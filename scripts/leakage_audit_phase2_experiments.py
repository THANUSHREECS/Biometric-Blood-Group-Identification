import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cached dev features
npz = np.load(OUT_DIR / "dev_densenet121_features.npz")
X = npz["features"] # (80, 3072)
y = npz["labels"]   # (80,)
folds = npz["folds"] # (80,)
pids = npz["pids"]   # (80,)

print(f"Features: {X.shape}, Labels: {y.shape}, Folds: {folds.shape}")

# Hyperparameters matching EXP-06C Config A
FIXED_TEMPERATURE = 0.05
FIXED_LAMBDA = 0.50
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EPOCHS = 150
MIN_EPOCHS = 100
PATIENCE = 15

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

class HeadModel(tf.keras.Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    @tf.function
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
            loss_supcon = supervised_contrastive_loss(preds_emb, y, temperature=FIXED_TEMPERATURE)
            total_loss = loss_clf + (FIXED_LAMBDA * loss_supcon)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, preds_clf, sample_weight=sample_weight)
        res = {m.name: m.result() for m in self.metrics}
        res["loss"] = total_loss
        return res

def build_head(seed_head=42):
    inp = tf.keras.Input(shape=(3072,), name="fused_features")
    dense_512 = tf.keras.layers.Dense(
        512, activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head),
        bias_initializer="zeros", name="dense_512"
    )
    dropout_05 = tf.keras.layers.Dropout(0.5, name="dropout_0.5")
    clf = tf.keras.layers.Dense(
        4, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head + 1),
        bias_initializer="zeros", name="clf"
    )
    emb = tf.keras.layers.Dense(
        128, activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_head + 2),
        bias_initializer="zeros", name="emb"
    )
    d = dense_512(inp)
    drop = dropout_05(d)
    out_clf = clf(drop)
    out_emb = emb(inp)
    return HeadModel(inputs=inp, outputs=[out_clf, out_emb])

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "balanced_accuracy": float(bal_acc), "macro_f1": float(mf1)}

# =========================================================================
# PROTOCOL 1: ORIGINAL HELD-OUT-FOLD SELECTION RULE (Rule i)
# =========================================================================
def run_cv_heldout_rule(y_target, seed=42):
    oof_preds = np.zeros(len(y_target), dtype=int)
    per_fold_metrics = {}
    
    for fold in range(1, 6):
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y_target[train_mask]
        X_val, y_val = X[val_mask], y_target[val_mask]
        
        counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
        cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
        sample_weight = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
        
        model = build_head(seed_head=seed + fold * 10)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            weighted_metrics=["accuracy"]
        )
        
        best_f1 = -1.0
        best_weights = None
        best_epoch = 0
        wait = 0
        
        # Training loop
        n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))
        rng = np.random.RandomState(seed + fold)
        
        for ep in range(EPOCHS):
            # Shuffle training batches
            perm = rng.permutation(len(X_tr))
            for b in range(n_batches):
                b_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                model.train_on_batch(X_tr[b_idx], y_tr[b_idx], sample_weight=sample_weight[b_idx])
                
            # Monitor held-out fold
            val_preds = model(X_val, training=False)[0].numpy()
            y_pred_val = np.argmax(val_preds, axis=1)
            val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = ep + 1
                best_weights = [w.numpy() for w in model.weights]
                wait = 0
            else:
                wait += 1
                
            if (ep + 1) >= MIN_EPOCHS and wait >= PATIENCE:
                break
                
        # Restore best weights and record OOF
        for w_target, w_src in zip(model.weights, best_weights):
            w_target.assign(w_src)
        final_preds = np.argmax(model(X_val, training=False)[0].numpy(), axis=1)
        oof_preds[val_mask] = final_preds
        per_fold_metrics[f"fold_{fold}"] = evaluate_predictions(y_val, final_preds)
        per_fold_metrics[f"fold_{fold}"]["best_epoch"] = int(best_epoch)
        
    overall = evaluate_predictions(y_target, oof_preds)
    return overall, per_fold_metrics, oof_preds

# =========================================================================
# PROTOCOL 2: CLEAN RULE A (Inner validation split on training fold only)
# =========================================================================
def run_cv_inner_val_rule(y_target, seed=42):
    oof_preds = np.zeros(len(y_target), dtype=int)
    per_fold_metrics = {}
    
    for fold in range(1, 6):
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_outer_tr, y_outer_tr = X[train_mask], y_target[train_mask]
        X_outer_val, y_outer_val = X[val_mask], y_target[val_mask]
        
        # Inner 80/20 stratified split of training participants ONLY
        # outer train has 64 participants -> inner train: ~51, inner val: ~13
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
        inner_tr_idx, inner_val_idx = next(skf_inner.split(X_outer_tr, y_outer_tr))
        
        X_in_tr, y_in_tr = X_outer_tr[inner_tr_idx], y_outer_tr[inner_tr_idx]
        X_in_val, y_in_val = X_outer_tr[inner_val_idx], y_outer_tr[inner_val_idx]
        
        counts = {i: int(np.sum(y_in_tr == i)) for i in range(4)}
        cw = {i: float(len(y_in_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
        sample_weight = np.array([cw[int(lbl)] for lbl in y_in_tr], dtype=np.float32)
        
        model = build_head(seed_head=seed + fold * 10)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            weighted_metrics=["accuracy"]
        )
        
        best_f1 = -1.0
        best_weights = None
        best_epoch = 0
        wait = 0
        
        n_batches = int(np.ceil(len(X_in_tr) / BATCH_SIZE))
        rng = np.random.RandomState(seed + fold)
        
        for ep in range(EPOCHS):
            perm = rng.permutation(len(X_in_tr))
            for b in range(n_batches):
                b_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                model.train_on_batch(X_in_tr[b_idx], y_in_tr[b_idx], sample_weight=sample_weight[b_idx])
                
            # Monitored on INNER validation set only (never outer held-out fold!)
            in_val_preds = model(X_in_val, training=False)[0].numpy()
            y_pred_in_val = np.argmax(in_val_preds, axis=1)
            in_val_f1 = f1_score(y_in_val, y_pred_in_val, average="macro", zero_division=0)
            
            if in_val_f1 > best_f1:
                best_f1 = in_val_f1
                best_epoch = ep + 1
                best_weights = [w.numpy() for w in model.weights]
                wait = 0
            else:
                wait += 1
                
            if (ep + 1) >= MIN_EPOCHS and wait >= PATIENCE:
                break
                
        # Evaluate chosen checkpoint strictly once on outer held-out fold
        for w_target, w_src in zip(model.weights, best_weights):
            w_target.assign(w_src)
        final_preds = np.argmax(model(X_outer_val, training=False)[0].numpy(), axis=1)
        oof_preds[val_mask] = final_preds
        per_fold_metrics[f"fold_{fold}"] = evaluate_predictions(y_outer_val, final_preds)
        per_fold_metrics[f"fold_{fold}"]["best_inner_epoch"] = int(best_epoch)
        per_fold_metrics[f"fold_{fold}"]["best_inner_f1"] = float(best_f1)
        
    overall = evaluate_predictions(y_target, oof_preds)
    return overall, per_fold_metrics, oof_preds

# =========================================================================
# PROTOCOL 3: CLEAN RULE B (Fixed epoch count = 100 declared in advance)
# =========================================================================
def run_cv_fixed_epochs_rule(y_target, seed=42, fixed_epochs=100):
    oof_preds = np.zeros(len(y_target), dtype=int)
    per_fold_metrics = {}
    
    for fold in range(1, 6):
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y_target[train_mask]
        X_val, y_val = X[val_mask], y_target[val_mask]
        
        counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
        cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
        sample_weight = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
        
        model = build_head(seed_head=seed + fold * 10)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            weighted_metrics=["accuracy"]
        )
        
        n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))
        rng = np.random.RandomState(seed + fold)
        
        for ep in range(fixed_epochs):
            perm = rng.permutation(len(X_tr))
            for b in range(n_batches):
                b_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                model.train_on_batch(X_tr[b_idx], y_tr[b_idx], sample_weight=sample_weight[b_idx])
                
        # Evaluate strictly at fixed_epochs (no checkpoint selection)
        final_preds = np.argmax(model(X_val, training=False)[0].numpy(), axis=1)
        oof_preds[val_mask] = final_preds
        per_fold_metrics[f"fold_{fold}"] = evaluate_predictions(y_val, final_preds)
        
    overall = evaluate_predictions(y_target, oof_preds)
    return overall, per_fold_metrics, oof_preds

# =========================================================================
# EXECUTE SEEDS FOR ALL THREE PROTOCOLS
# =========================================================================
seeds = [42, 43, 44]
results = {
    "heldout_selection_rule_i": {},
    "clean_rule_a_inner_val": {},
    "clean_rule_b_fixed_epochs": {}
}

print("\n" + "="*70)
print("RUNNING PROTOCOL (i): SANITY CHECK (ORIGINAL HELD-OUT SELECTION)")
print("="*70)
for s in seeds:
    t0 = time.time()
    ov, pf, _ = run_cv_heldout_rule(y, seed=s)
    results["heldout_selection_rule_i"][f"seed_{s}"] = {"overall": ov, "per_fold": pf}
    print(f"Seed {s}: Acc = {ov['accuracy']*100:.2f}%, BalAcc = {ov['balanced_accuracy']:.4f}, MacroF1 = {ov['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

print("\n" + "="*70)
print("RUNNING CLEAN RULE (a): INNER STRATIFIED SPLIT ON TRAINING FOLD")
print("="*70)
for s in seeds:
    t0 = time.time()
    ov, pf, _ = run_cv_inner_val_rule(y, seed=s)
    results["clean_rule_a_inner_val"][f"seed_{s}"] = {"overall": ov, "per_fold": pf}
    print(f"Seed {s}: Acc = {ov['accuracy']*100:.2f}%, BalAcc = {ov['balanced_accuracy']:.4f}, MacroF1 = {ov['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

print("\n" + "="*70)
print("RUNNING CLEAN RULE (b): FIXED EPOCH COUNT (100 EPOCHS)")
print("="*70)
for s in seeds:
    t0 = time.time()
    ov, pf, _ = run_cv_fixed_epochs_rule(y, seed=s, fixed_epochs=100)
    results["clean_rule_b_fixed_epochs"][f"seed_{s}"] = {"overall": ov, "per_fold": pf}
    print(f"Seed {s}: Acc = {ov['accuracy']*100:.2f}%, BalAcc = {ov['balanced_accuracy']:.4f}, MacroF1 = {ov['macro_f1']:.4f} ({time.time()-t0:.1f}s)")

# Save Phase 2 preliminary results
with open(OUT_DIR / "phase2_selection_rules_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved comparison to {OUT_DIR / 'phase2_selection_rules_comparison.json'}")
