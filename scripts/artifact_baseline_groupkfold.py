import os
import sys
import numpy as np
import pandas as pd
import cv2
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

print("Artifact Baseline with Pseudo-Subject Split")

manifest_path = "outputs/dataset_manifest.csv"
df = pd.read_csv(manifest_path)
df = df[df["class_name"].isin(["A+", "B+", "AB+", "O+"])].reset_index(drop=True)

# Rebuild pseudo-subjects
features = []
valid_indices = []
for i, path in enumerate(df["image_path"]):
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            features.append(cv2.resize(img, (64, 64)).flatten())
            valid_indices.append(i)

df = df.iloc[valid_indices].reset_index(drop=True)
features = np.array(features, dtype=np.float32)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
descriptors = []
for path in df["image_path"]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    _, des = orb.detectAndCompute(img, None)
    descriptors.append(des)

G = nx.Graph()
G.add_nodes_from(range(len(df)))
match_threshold = 40
for i in range(len(df)):
    dists = np.linalg.norm(features - features[i], axis=1)
    top20 = np.argsort(dists)[1:21]
    des1 = descriptors[i]
    if des1 is None: continue
    for j in top20:
        if i >= j: continue
        des2 = descriptors[j]
        if des2 is None: continue
        matches = bf.match(des1, des2)
        if len(matches) > match_threshold:
            G.add_edge(i, j)

groups = np.zeros(len(df), dtype=int)
for group_id, component in enumerate(nx.connected_components(G)):
    for node in component:
        groups[node] = group_id

# Extract Artifact features
X_meta = []
for path in df["image_path"]:
    size = os.path.getsize(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mean_i = np.mean(img)
    std_i = np.std(img)
    border = np.concatenate([img[0,:], img[-1,:], img[:,0], img[:,-1]])
    X_meta.append([size, mean_i, std_i, np.mean(border), np.std(border)])

X_meta = np.array(X_meta)
y = df["class_name"].astype('category').cat.codes.values

majority_class = np.argmax(np.bincount(y))
majority_acc = np.mean(y == majority_class)

gkf = GroupKFold(n_splits=5)
accs = []
for train_idx, test_idx in gkf.split(X_meta, y, groups=groups):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_meta[train_idx], y[train_idx])
    preds = clf.predict(X_meta[test_idx])
    accs.append(accuracy_score(y[test_idx], preds))

mean_acc = np.mean(accs)
std_acc = np.std(accs)
diff = mean_acc - majority_acc
diff_ci_lower = diff - 1.96 * (std_acc / np.sqrt(5))
diff_ci_upper = diff + 1.96 * (std_acc / np.sqrt(5))

print(f"Majority Baseline: {majority_acc*100:.2f}%")
print(f"Artifact Classifier (GroupKFold): {mean_acc*100:.2f}%")
print(f"Difference from Baseline: {diff*100:.2f}%")
print(f"95% CI of Difference: [{diff_ci_lower*100:.2f}%, {diff_ci_upper*100:.2f}%]")
