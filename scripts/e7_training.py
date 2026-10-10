import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.applications.densenet import preprocess_input

print("Starting Task 1: E7 Group-Aware Training")

# 1. Load Manifest
manifest_path = "outputs/dataset_manifest.csv"
df = pd.read_csv(manifest_path)
df = df[df["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy()
df["cluster_id"] = df["image_path"].str.extract(r'cluster_(\d+)_').astype(float)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 2. Extract Features using Frozen DenseNet121
base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", pooling="avg")

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_bmp(img, channels=3) # Or decode_image
    img = tf.image.resize(img, (112, 112))
    return preprocess_input(img)

# Subsample to make extraction fast for this demo (100 images per class)
df_sub = df.groupby('class_name').head(100).reset_index(drop=True)
paths = df_sub["image_path"].values
labels = df_sub["class_name"].values
groups = df_sub["cluster_id"].values

print(f"Extracting features for {len(paths)} images...")
dataset = tf.data.Dataset.from_tensor_slices(paths).map(load_img).batch(32)
features = base_model.predict(dataset, verbose=1)

# 3. GroupKFold Training
gkf = GroupKFold(n_splits=4)
accs = []
f1s = []

os.makedirs("results/E7", exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(gkf.split(features, labels, groups)):
    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # We must handle the fact that the model might not predict the held-out class at all
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    
    accs.append(acc)
    f1s.append(f1)
    
    print(f"Fold {fold+1}: Held out groups {np.unique(groups[test_idx])} (Class {np.unique(y_test)}) -> Acc: {acc:.4f}, Macro F1: {f1:.4f}")

import pickle
with open("results/E7/e7_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print(f"Mean Accuracy: {np.mean(accs):.4f}")
print(f"Mean Macro F1: {np.mean(f1s):.4f}")
print("Model saved to results/E7/e7_model.pkl")

# Permutation test at cluster level
print("Running cluster-level permutation test...")
np.random.seed(42)
perm_accs = []
unique_groups = np.unique(groups)
for _ in range(100):
    shuffled_classes = np.random.permutation(["A+", "B+", "AB+", "O+"])
    group_to_class = {g: c for g, c in zip(unique_groups, shuffled_classes)}
    shuffled_labels = np.array([group_to_class[g] for g in groups])
    
    # Train on shuffled
    clf_perm = LogisticRegression(max_iter=1000)
    # We just do a standard train-test split for the permutation since GroupKFold on 4 clusters is degenerate
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(features, shuffled_labels, test_size=0.25)
    clf_perm.fit(X_tr, y_tr)
    perm_accs.append(accuracy_score(y_te, clf_perm.predict(X_te)))

print(f"Permutation mean accuracy: {np.mean(perm_accs):.4f}")
