import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Load features
eff_b0 = np.load("experiments/leakage_audit/exp1_efficientnetb0_embeddings.npy") # (2769, 1280)
manifest = pd.read_csv("outputs/dataset_manifest.csv")
pos = manifest[manifest["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy().reset_index(drop=True)

classes = ["A+", "B+", "AB+", "O+"]
c2idx = {c: i for i, c in enumerate(classes)}
y = pos["class_name"].map(c2idx).values

print(f"EfficientNet-B0 features: {eff_b0.shape}, Labels: {y.shape}")

FIXED_EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3

def build_clf_head(seed=42):
    inp = tf.keras.Input(shape=(1280,), name="effb0_features")
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer="zeros",
        name="dense_256"
    )(inp)
    x = tf.keras.layers.Dropout(0.30, name="dropout_0.30")(x)
    out = tf.keras.layers.Dense(
        4, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        bias_initializer="zeros",
        name="out_4"
    )(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="Historical_Head")
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# 1. Benchmark Fixed Epochs (50 epochs) across 5 folds
print("\nBenchmarking 1 full 5-fold CV run: Arm 1 (Fixed 50 Epochs)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
t0 = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(eff_b0, y)):
    X_tr, y_tr = eff_b0[tr_idx], y[tr_idx]
    X_val, y_val = eff_b0[val_idx], y[val_idx]
    
    # Class weights
    counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
    cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
    sw = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
    
    model = build_clf_head(seed=42 + fold)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        batch_size=BATCH_SIZE,
        epochs=FIXED_EPOCHS,
        verbose=0
    )

t_arm1 = time.time() - t0
print(f"Arm 1 (Fixed 50 Epochs, 5 folds) completed in {t_arm1:.2f} seconds ({t_arm1/5:.2f}s per fold)")

# 2. Benchmark Inner Stratified Validation Split across 5 folds
print("\nBenchmarking 1 full 5-fold CV run: Arm 2 (Inner Validation Split early stopping)...")
t0 = time.time()

for fold, (tr_idx, val_idx) in enumerate(skf.split(eff_b0, y)):
    X_outer_tr, y_outer_tr = eff_b0[tr_idx], y[tr_idx]
    X_outer_val, y_outer_val = eff_b0[val_idx], y[val_idx]
    
    # Inner 80/20 stratified split of training images only
    skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + fold)
    in_tr_idx, in_val_idx = next(skf_in.split(X_outer_tr, y_outer_tr))
    
    X_in_tr, y_in_tr = X_outer_tr[in_tr_idx], y_outer_tr[in_tr_idx]
    X_in_val, y_in_val = X_outer_tr[in_val_idx], y_outer_tr[in_val_idx]
    
    counts = {i: int(np.sum(y_in_tr == i)) for i in range(4)}
    cw = {i: float(len(y_in_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
    sw = np.array([cw[int(lbl)] for lbl in y_in_tr], dtype=np.float32)
    
    model = build_clf_head(seed=42 + fold)
    # Early stopping on inner validation loss
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    model.fit(
        X_in_tr, y_in_tr,
        sample_weight=sw,
        batch_size=BATCH_SIZE,
        epochs=80,
        validation_data=(X_in_val, y_in_val),
        callbacks=[es],
        verbose=0
    )

t_arm2 = time.time() - t0
print(f"Arm 2 (Inner Val Split, 5 folds) completed in {t_arm2:.2f} seconds ({t_arm2/5:.2f}s per fold)")
