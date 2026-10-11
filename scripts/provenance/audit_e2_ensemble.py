"""Provenance copy of scratch/audit_e2_ensemble.py
Original path: scratch/audit_e2_ensemble.py
Original timestamp: 2026-10-09 21:40:27
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
dev_df = pd.read_csv(split_csv_path)
dev_df = dev_df[dev_df["split"].isin(["TRAIN", "VALIDATION"])]

IMAGE_SIZE = (112, 112)
classes = ["A+", "B+", "AB+", "O+"]
rev_class_mapping = {c: i for i, c in enumerate(classes)}
class_mapping = {i: c for i, c in enumerate(classes)}

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=IMAGE_SIZE)
    return np.array(resized, dtype=np.float32)

print("Loading dataset...")
dev_X, dev_y, dev_pids_list = [], [], []
for pid, group in dev_df.groupby("participant_id"):
    scans = [transform_c5(row["scan_path"]) for _, row in group.iterrows()]
    dev_X.append(scans)
    dev_y.append(rev_class_mapping[group.iloc[0]["blood_group"]])
    dev_pids_list.append(pid)
    
dev_X = np.array(dev_X)
dev_y = np.array(dev_y)
dev_pids_list = np.array(dev_pids_list)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_E2():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    
    # Passing training=False to avoid arbitrary augmentation during inference 
    aug1, aug2, aug3 = data_augmentation(scan1, training=False), data_augmentation(scan2, training=False), data_augmentation(scan3, training=False)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    feat1, feat2, feat3 = pool(base(aug1)), pool(base(aug2)), pool(base(aug3))
    concat = tf.keras.layers.Concatenate()([feat1, feat2, feat3])
    
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(sgkf.split(dev_X, dev_y, dev_pids_list))

model = build_E2()
print("E2 Model Built. Output shapes:", [out.shape for out in model.outputs])

for fold, (train_idx, val_idx) in enumerate(fold_indices):
    ckpt_path = os.path.join(project_root, f"models/phase3_fusion_metric_learning/E2/fold_{fold+1}_best.weights.h5")
    model.load_weights(ckpt_path)
    X_val, y_val = dev_X[val_idx], dev_y[val_idx]
    
    # We only care about the clf output [0]
    preds = model.predict(X_val, verbose=0)[0]
    y_pred = np.argmax(preds, axis=1)
    
    y_true_p = [class_mapping[y] for y in y_val]
    y_pred_p = [class_mapping[y] for y in y_pred]
    
    acc = accuracy_score(y_true_p, y_pred_p)
    macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
    rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
    
    print(f"FOLD {fold+1} RESULTS:")
    print(f"Val Macro F1: {macro_f1:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val Participants: {len(X_val)}")
    print(f"Recall -> A+: {rep.get('A+', {}).get('recall', 0):.2f}, B+: {rep.get('B+', {}).get('recall', 0):.2f}, AB+: {rep.get('AB+', {}).get('recall', 0):.2f}, O+: {rep.get('O+', {}).get('recall', 0):.2f}")
    print(f"Checkpoint Path: {ckpt_path}\n")
