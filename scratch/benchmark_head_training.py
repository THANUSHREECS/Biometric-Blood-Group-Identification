import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Load cached features
data = np.load("experiments/leakage_audit/dev_densenet121_features.npz")
X = data["features"] # (80, 3072)
y = data["labels"]   # (80,)
folds = data["folds"] # (80,)

print(f"Loaded X: {X.shape}, y: {y.shape}, folds: {folds.shape}")

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

    @tf.function
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
        loss_supcon = supervised_contrastive_loss(preds_emb, y, temperature=FIXED_TEMPERATURE)
        total_loss = loss_clf + (FIXED_LAMBDA * loss_supcon)
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

class QuickMacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, patience=PATIENCE, min_epochs=MIN_EPOCHS):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_f1 = -1.0
        self.best_weights = None
        self.wait = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model(self.X_val, training=False)
        clf_preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        y_pred = np.argmax(clf_preds.numpy(), axis=1)
        val_f1 = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = epoch + 1
            self.wait = 0
            self.best_weights = [w.numpy() for w in self.model.weights]
        else:
            self.wait += 1
            
        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            self.model.stop_training = True

# Benchmark 1 full 5-fold CV run
print("\nBenchmarking 1 full 5-fold CV run on cached features...")
t_start = time.time()
oof_preds = np.zeros(len(y), dtype=int)

for fold in range(1, 6):
    val_mask = (folds == fold)
    train_mask = ~val_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    # Class weights
    counts = {i: int(np.sum(y_train == i)) for i in range(4)}
    cw = {i: float(len(y_train) / (4.0 * max(counts[i], 1))) for i in range(4)}
    sample_weight = np.array([cw[int(lbl)] for lbl in y_train], dtype=np.float32)
    
    model = build_head(seed_head=42 + fold)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        weighted_metrics=["accuracy"]
    )
    
    cb = QuickMacroF1Checkpoint(X_val, y_val, patience=PATIENCE, min_epochs=MIN_EPOCHS)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[cb],
        verbose=0
    )
    
    # Restore best weights in memory
    for w_target, w_src in zip(model.weights, cb.best_weights):
        w_target.assign(w_src)
        
    preds = model(X_val, training=False)[0].numpy()
    oof_preds[val_mask] = np.argmax(preds, axis=1)

t_end = time.time()
acc = accuracy_score(y, oof_preds)
bal_acc = balanced_accuracy_score(y, oof_preds)
mf1 = f1_score(y, oof_preds, average="macro")

print(f"\n1 Full 5-Fold Repeat completed in {t_end - t_start:.2f} seconds!")
print(f"Re-run Accuracy: {acc*100:.2f}% (Historical EXP-06C: 53.75%)")
print(f"Re-run Balanced Accuracy: {bal_acc:.4f}")
print(f"Re-run Macro F1: {mf1:.4f}")
