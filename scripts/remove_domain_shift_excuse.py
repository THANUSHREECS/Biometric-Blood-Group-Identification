import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2

def load_data(df, image_col):
    images = []
    valid = []
    for i, row in df.iterrows():
        p = row[image_col]
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            # Preprocess for DenseNet
            img = tf.keras.applications.densenet.preprocess_input(img.astype(np.float32))
            images.append(img)
            valid.append(i)
    return np.array(images), df.iloc[valid].reset_index(drop=True)

print("Loading models...")
# 1. Dataset-trained DenseNet121 backbone
trained_model = tf.keras.models.load_model("models/densenet121_blood_group.keras", compile=False)
feat_extractor_dataset = tf.keras.Model(inputs=trained_model.input, outputs=trained_model.get_layer("global_avg_pool").output)

# 2. ImageNet DenseNet121 backbone
base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feat_extractor_imagenet = tf.keras.Model(inputs=base.input, outputs=base.output)

print("\n--- LIVE VALIDATION (PARTICIPANT-LEVEL) ---")
part_df = pd.read_csv("data/mfs100_final/metadata/participants.csv")
log_df = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
live_df = pd.merge(log_df, part_df, on="participant_id", how="left")
live_df = live_df[live_df["blood_group"].isin(["A+", "B+", "AB+", "O+"])]

print("Extracting live features...")
live_imgs, live_df = load_data(live_df, "image_path")
feats_ds_live = feat_extractor_dataset.predict(live_imgs, batch_size=32)
feats_in_live = feat_extractor_imagenet.predict(live_imgs, batch_size=32)

y_live = live_df["blood_group"].values
groups_live = live_df["participant_id"].values

# Aggregate to participant level
unique_parts = np.unique(groups_live)
X_ds_part = []
X_in_part = []
y_part = []
for p in unique_parts:
    idx = groups_live == p
    X_ds_part.append(np.mean(feats_ds_live[idx], axis=0))
    X_in_part.append(np.mean(feats_in_live[idx], axis=0))
    y_part.append(y_live[idx][0])

X_ds_part = np.array(X_ds_part)
X_in_part = np.array(X_in_part)
y_part = np.array(y_part)

def evaluate_features(X, y, groups=None):
    gkf = GroupKFold(n_splits=5)
    accs = []
    if groups is None:
        groups = np.arange(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    
    # Permutation test
    np.random.seed(42)
    perm_accs = []
    for _ in range(100):
        acc_p = []
        shuffled_y = y.copy()
        np.random.shuffle(shuffled_y)
        for train_idx, test_idx in gkf.split(X, shuffled_y, groups=groups):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[train_idx], shuffled_y[train_idx])
            acc_p.append(accuracy_score(shuffled_y[test_idx], clf.predict(X[test_idx])))
        perm_accs.append(np.mean(acc_p))
    p_val = np.mean(np.array(perm_accs) >= np.mean(accs))
    
    return np.mean(accs), p_val

print("Evaluating Dataset Backbone Head on Live Data...")
acc_ds, pval_ds = evaluate_features(X_ds_part, y_part)
print(f"Dataset Backbone Accuracy: {acc_ds*100:.2f}%, Permutation p-val: {pval_ds}")

print("Evaluating ImageNet Backbone Head on Live Data...")
acc_in, pval_in = evaluate_features(X_in_part, y_part)
print(f"ImageNet Backbone Accuracy: {acc_in*100:.2f}%, Permutation p-val: {pval_in}")


print("\n--- DATASET EVALUATION (PSEUDO-SUBJECT GROUPS) ---")
manifest_path = "outputs/dataset_manifest.csv"
ds_df = pd.read_csv(manifest_path)
ds_df = ds_df[(ds_df["split"] == "test") & (ds_df["class_name"].isin(["A+", "B+", "AB+", "O+"]))].reset_index(drop=True)

print("Extracting dataset features (Test set only for speed)...")
ds_imgs, ds_df = load_data(ds_df, "image_path")
feats_ds_ds = feat_extractor_dataset.predict(ds_imgs, batch_size=32)

y_ds = ds_df["class_name"].values

# Quick ORB group assignment for test set
print("Computing pseudo-subjects on test set...")
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
descs = []
for p in ds_df["image_path"]:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        _, des = orb.detectAndCompute(img, None)
        descs.append(des)
    else: descs.append(None)

import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(len(ds_df)))
for i in range(len(ds_df)):
    if descs[i] is None: continue
    for j in range(i+1, min(i+10, len(ds_df))): # approximate sequential duplicates
        if descs[j] is None: continue
        matches = bf.match(descs[i], descs[j])
        if len(matches) > 40:
            G.add_edge(i, j)

groups_ds = np.zeros(len(ds_df), dtype=int)
for group_id, component in enumerate(nx.connected_components(G)):
    for node in component:
        groups_ds[node] = group_id

print("Evaluating Dataset Backbone Head on Pseudo-Subject Dataset...")
acc_ds_pseudo, pval_ds_pseudo = evaluate_features(feats_ds_ds, y_ds, groups=groups_ds)
print(f"Pseudo-Subject Dataset Accuracy: {acc_ds_pseudo*100:.2f}%, Permutation p-val: {pval_ds_pseudo}")
