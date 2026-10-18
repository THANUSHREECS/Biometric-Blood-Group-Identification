# PROVENANCE NOTE:
# This script is a verified copy of scratch/extract_seed42_participant_preds.py,
# preserved in scripts/ for full audit reproduction and provenance.
# Do not modify or delete the scratch original.

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

# Load cached dev features
npz = np.load("experiments/leakage_audit/dev_densenet121_features.npz")
X = npz["features"] # (80, 3072)
y_true = npz["labels"]   # (80,)
folds = npz["folds"] # (80,)
pids = npz["pids"]   # (80,)
classes = ["A+", "B+", "AB+", "O+"]

# Load historical OOF
hist_df = pd.read_csv("reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv")

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

print("Running single seed-42 replication under Rule (i)...")
oof_preds = np.zeros(len(y_true), dtype=int)
seed = 42

for fold in range(1, 6):
    val_mask = (folds == fold)
    train_mask = ~val_mask
    
    X_tr, y_tr = X[train_mask], y_true[train_mask]
    X_val, y_val = X[val_mask], y_true[val_mask]
    
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
    wait = 0
    
    n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))
    rng = np.random.RandomState(seed + fold)
    
    for ep in range(EPOCHS):
        perm = rng.permutation(len(X_tr))
        for b in range(n_batches):
            b_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            model.train_on_batch(X_tr[b_idx], y_tr[b_idx], sample_weight=sample_weight[b_idx])
            
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

# Save predictions
df_out = pd.DataFrame({
    "participant_id": pids,
    "fold_id": folds,
    "true_label": [classes[i] for i in y_true],
    "rerun_pred_label": [classes[i] for i in oof_preds]
})
df_out.to_csv("experiments/leakage_audit/seed42_rerun_predictions.csv", index=False)

# Compare with historical
merged = pd.merge(df_out, hist_df[["participant_id", "predicted_label"]], on="participant_id")
merged.rename(columns={"predicted_label": "historical_pred_label"}, inplace=True)

diff_count = int(np.sum(merged["rerun_pred_label"] != merged["historical_pred_label"]))
same_count = int(np.sum(merged["rerun_pred_label"] == merged["historical_pred_label"]))

print(f"\nPOOLED METRICS:")
print(f"  Historical Accuracy: {accuracy_score(merged['true_label'], merged['historical_pred_label'])*100:.2f}%")
print(f"  Rerun Accuracy:      {accuracy_score(merged['true_label'], merged['rerun_pred_label'])*100:.2f}%")
print(f"  Historical Macro F1: {f1_score(merged['true_label'], merged['historical_pred_label'], average='macro'):.4f}")
print(f"  Rerun Macro F1:      {f1_score(merged['true_label'], merged['rerun_pred_label'], average='macro'):.4f}")

print(f"\nPARTICIPANT-LEVEL CONCORDANCE:")
print(f"  Total Participants:       80")
print(f"  Matching Predictions:     {same_count} / 80 ({same_count/80*100:.2f}%)")
print(f"  Differing Predictions:    {diff_count} / 80 ({diff_count/80*100:.2f}%)")

print("\nPER-FOLD BREAKDOWN:")
print(f"{'Fold':<6} | {'Hist Acc':<10} | {'Rerun Acc':<10} | {'Hist F1':<10} | {'Rerun F1':<10} | {'Differing':<10}")
print("-" * 65)
for f in range(1, 6):
    sub = merged[merged["fold_id"] == f]
    h_acc = accuracy_score(sub["true_label"], sub["historical_pred_label"]) * 100
    r_acc = accuracy_score(sub["true_label"], sub["rerun_pred_label"]) * 100
    h_f1 = f1_score(sub["true_label"], sub["historical_pred_label"], average="macro", zero_division=0)
    r_f1 = f1_score(sub["true_label"], sub["rerun_pred_label"], average="macro", zero_division=0)
    f_diff = int(np.sum(sub["rerun_pred_label"] != sub["historical_pred_label"]))
    print(f"Fold {f:<2} | {h_acc:6.2f}%   | {r_acc:6.2f}%   | {h_f1:6.4f}   | {r_f1:6.4f}   | {f_diff}/16")

