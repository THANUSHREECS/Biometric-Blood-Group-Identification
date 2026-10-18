import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cached dev features
npz = np.load(OUT_DIR / "dev_densenet121_features.npz")
X = npz["features"] # (80, 3072)
y_true = npz["labels"]   # (80,)
folds = npz["folds"] # (80,)
pids = npz["pids"]   # (80,)

print(f"Features: {X.shape}, True Labels: {y_true.shape}, Folds: {folds.shape}")

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

def run_one_repeat_rule_i(y_permuted, repeat_seed):
    oof_preds = np.zeros(len(y_permuted), dtype=int)
    
    for fold in range(1, 6):
        val_mask = (folds == fold)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y_permuted[train_mask]
        X_val, y_val = X[val_mask], y_permuted[val_mask]
        
        counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
        cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
        sample_weight = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
        
        model = build_head(seed_head=repeat_seed * 100 + fold)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            weighted_metrics=["accuracy"]
        )
        
        best_f1 = -1.0
        best_weights = None
        wait = 0
        
        n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))
        rng = np.random.RandomState(repeat_seed * 10 + fold)
        
        for ep in range(EPOCHS):
            perm = rng.permutation(len(X_tr))
            for b in range(n_batches):
                b_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                model.train_on_batch(X_tr[b_idx], y_tr[b_idx], sample_weight=sample_weight[b_idx])
                
            # Monitored on held-out fold exactly as Rule (i)
            val_preds = model(X_val, training=False)[0].numpy()
            y_pred_val = np.argmax(val_preds, axis=1)
            val_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_weights = [w.numpy() for w in model.weights]
                wait = 0
            else:
                wait += 1
                
            if (ep + 1) >= MIN_EPOCHS and wait >= PATIENCE:
                break
                
        for w_target, w_src in zip(model.weights, best_weights):
            w_target.assign(w_src)
        final_preds = np.argmax(model(X_val, training=False)[0].numpy(), axis=1)
        oof_preds[val_mask] = final_preds
        
    acc = float(accuracy_score(y_permuted, oof_preds))
    bal_acc = float(balanced_accuracy_score(y_permuted, oof_preds))
    mf1 = float(f1_score(y_permuted, oof_preds, average="macro", zero_division=0))
    return acc, bal_acc, mf1

TOTAL_REPEATS = 100
CSV_OUT = OUT_DIR / "selection_inclusive_null_results.csv"
JSON_OUT = OUT_DIR / "selection_inclusive_null_summary.json"

print(f"Starting Selection-Inclusive Null: {TOTAL_REPEATS} repeats of Rule (i)")
print(f"Partial results will be checkpointed to: {CSV_OUT}")

null_records = []
t_start_all = time.time()

# Master permutation RNG
master_rng = np.random.RandomState(1337)

for r in range(1, TOTAL_REPEATS + 1):
    t_rep_start = time.time()
    rep_seed = int(master_rng.randint(1, 1000000))
    
    # Participant-level label permutation
    # (permute labels across the 80 participants)
    permuted_y = master_rng.permutation(y_true)
    
    acc, bal_acc, mf1 = run_one_repeat_rule_i(permuted_y, repeat_seed=rep_seed)
    t_rep = time.time() - t_rep_start
    
    record = {
        "repeat": r,
        "seed": rep_seed,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": mf1,
        "runtime_sec": round(t_rep, 2)
    }
    null_records.append(record)
    
    elapsed = time.time() - t_start_all
    avg_sec = elapsed / r
    rem_sec = avg_sec * (TOTAL_REPEATS - r)
    print(f"Repeat {r:3d}/{TOTAL_REPEATS}: Acc = {acc*100:5.2f}%, BalAcc = {bal_acc:.4f}, MacroF1 = {mf1:.4f} ({t_rep:.1f}s | ETA: {rem_sec/60:.1f}m)", flush=True)
    
    # Checkpoint every 10 repeats
    if r % 10 == 0 or r == TOTAL_REPEATS:
        df_null = pd.DataFrame(null_records)
        df_null.to_csv(CSV_OUT, index=False)
        print(f"  --> Checkpointed {r} repeats to {CSV_OUT}", flush=True)

df_null = pd.DataFrame(null_records)
df_null.to_csv(CSV_OUT, index=False)

# Compute null summary statistics
null_acc = df_null["accuracy"].values
null_bal = df_null["balanced_accuracy"].values
null_f1 = df_null["macro_f1"].values

summary = {
    "total_repeats": TOTAL_REPEATS,
    "accuracy": {
        "mean": float(np.mean(null_acc)),
        "std": float(np.std(null_acc)),
        "p95": float(np.percentile(null_acc, 95)),
        "p99": float(np.percentile(null_acc, 99)),
        "min": float(np.min(null_acc)),
        "max": float(np.max(null_acc))
    },
    "balanced_accuracy": {
        "mean": float(np.mean(null_bal)),
        "std": float(np.std(null_bal)),
        "p95": float(np.percentile(null_bal, 95)),
        "p99": float(np.percentile(null_bal, 99)),
        "min": float(np.min(null_bal)),
        "max": float(np.max(null_bal))
    },
    "macro_f1": {
        "mean": float(np.mean(null_f1)),
        "std": float(np.std(null_f1)),
        "p95": float(np.percentile(null_f1, 95)),
        "p99": float(np.percentile(null_f1, 99)),
        "min": float(np.min(null_f1)),
        "max": float(np.max(null_f1))
    }
}

# Empirical p-values with +1 correction: p = (sum(null >= obs) + 1) / (N + 1)
# Benchmark targets from Phase 2 comparison:
targets = {
    "rule_i_three_seed_mean": {"acc": 0.5083, "mf1": 0.5058, "bal": 0.5012},
    "rule_i_seed_42": {"acc": 0.5375, "mf1": 0.5218, "bal": 0.5183},
    "clean_rule_a_three_seed_mean": {"acc": 0.3167, "mf1": 0.2841, "bal": 0.2888},
    "clean_rule_b_three_seed_mean": {"acc": 0.3625, "mf1": 0.3278, "bal": 0.3210}
}

p_values = {}
for name, tgt in targets.items():
    p_acc = float((np.sum(null_acc >= tgt["acc"]) + 1) / (TOTAL_REPEATS + 1))
    p_bal = float((np.sum(null_bal >= tgt["bal"]) + 1) / (TOTAL_REPEATS + 1))
    p_f1 = float((np.sum(null_f1 >= tgt["mf1"]) + 1) / (TOTAL_REPEATS + 1))
    p_values[name] = {
        "target": tgt,
        "empirical_p_accuracy": p_acc,
        "empirical_p_balanced_acc": p_bal,
        "empirical_p_macro_f1": p_f1
    }

summary["empirical_p_values"] = p_values

with open(JSON_OUT, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("SELECTION-INCLUSIVE NULL DISTRIBUTION COMPLETE")
print("="*70)
print(f"Null Accuracy:  Mean = {summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}% | 95th = {summary['accuracy']['p95']*100:.2f}%, 99th = {summary['accuracy']['p99']*100:.2f}%")
print(f"Null Macro F1:  Mean = {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f} | 95th = {summary['macro_f1']['p95']:.4f}, 99th = {summary['macro_f1']['p99']:.4f}")
print("\nEmpirical p-values (+1 correction):")
for name, p_res in p_values.items():
    print(f"  {name:<30}: p(Acc) = {p_res['empirical_p_accuracy']:.4f}, p(F1) = {p_res['empirical_p_macro_f1']:.4f}")
