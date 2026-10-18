import time
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5

SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
df_split = pd.read_csv(SPLIT_CSV)
dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()

# Pre-load all 80 development participants
print("Preloading development scans for timing check...")
grouped = dev_df.groupby("participant_id")
dev_X = []
dev_y = []
classes = ["A+", "B+", "AB+", "O+"]
rev_class_mapping = {c: i for i, c in enumerate(classes)}

for pid, group in grouped:
    scans = [transform_c5(r["scan_path"])[0] for _, r in group.iterrows()]
    dev_X.append(scans)
    dev_y.append(rev_class_mapping[group.iloc[0]["blood_group"]])

dev_X = np.array(dev_X)
dev_y = np.array(dev_y)
print("dev_X loaded:", dev_X.shape)

# Build E2 architecture
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

inputs = tf.keras.Input(shape=(3, 112, 112, 3))
s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
a1, a2, a3 = data_augmentation(s1), data_augmentation(s2), data_augmentation(s3)
base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
base.trainable = False
pool = tf.keras.layers.GlobalAveragePooling2D()
f1, f2, f3 = pool(base(a1)), pool(base(a2)), pool(base(a3))
concat = tf.keras.layers.Concatenate()([f1, f2, f3])
d = tf.keras.layers.Dense(512, activation="relu")(concat)
drop = tf.keras.layers.Dropout(0.5)(d)
clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
model = tf.keras.Model(inputs, [clf, emb])

def supcon_loss(labels, feature_vectors, temperature=0.07):
    feat = tf.math.l2_normalize(feature_vectors, axis=1)
    sim = tf.matmul(feat, feat, transpose_b=True) / temperature
    sim_max = tf.reduce_max(sim, axis=1, keepdims=True)
    sim = sim - tf.stop_gradient(sim_max)
    lbl = tf.reshape(labels, [-1, 1])
    mask = tf.cast(tf.equal(lbl, tf.transpose(lbl)), tf.float32)
    bs = tf.shape(labels)[0]
    logits_mask = tf.ones_like(mask) - tf.eye(bs)
    mask = mask * logits_mask
    exp_sim = tf.exp(sim) * logits_mask
    log_prob = sim - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-12)
    mask_sum = tf.reduce_sum(mask, axis=1)
    mask_sum = tf.where(mask_sum < 1e-5, tf.ones_like(mask_sum), mask_sum)
    return -tf.reduce_mean(tf.reduce_sum(mask * log_prob, axis=1) / mask_sum)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=["sparse_categorical_crossentropy", supcon_loss],
    loss_weights=[1.0, 0.1]
)

X_train, y_train = dev_X[:64], dev_y[:64]
t0 = time.time()
model.fit(X_train, [y_train, y_train], batch_size=16, epochs=2, verbose=1)
t1 = time.time()
print(f"2 epochs on 64 participants took: {t1 - t0:.2f} seconds ({(t1 - t0)/2:.2f} s/epoch)")
