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

# Pre-load just 16 participants to time 1 epoch
sample_pids = dev_df["participant_id"].unique()[:16]
X_sample = []
y_sample = []
for pid in sample_pids:
    sub = dev_df[dev_df["participant_id"] == pid]
    scans = [transform_c5(r["scan_path"])[0] for _, r in sub.iterrows()]
    X_sample.append(scans)
    y_sample.append(0)

X_sample = np.array(X_sample)
y_sample = np.array(y_sample)
print("X_sample shape:", X_sample.shape)

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

model.compile(optimizer="adam", loss=["sparse_categorical_crossentropy", lambda y, p: tf.constant(0.0)])
t0 = time.time()
model.fit(X_sample, [y_sample, y_sample], batch_size=8, epochs=3, verbose=0)
t1 = time.time()
print(f"3 epochs on 16 participants took: {t1 - t0:.2f} seconds ({ (t1 - t0)/3:.2f} s/epoch)")
