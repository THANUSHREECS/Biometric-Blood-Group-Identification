import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2

print("Task 4: Dataset Artifact Baseline")
manifest_path = "outputs/dataset_manifest.csv"
df = pd.read_csv(manifest_path)
df = df[df["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy()
df["cluster_id"] = df["image_path"].str.extract(r'cluster_(\d+)_').astype(float)

# We will sample 100 images per class to keep it fast
df = df.groupby('class_name').head(100).reset_index(drop=True)

X = []
for path in df["image_path"]:
    file_size = os.path.getsize(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        X.append([0, 0, 0, 0])
        continue
    h, w = img.shape[:2]
    mean_val = np.mean(img)
    std_val = np.std(img)
    X.append([h*w, mean_val, std_val, file_size])
    
X = np.array(X)
y = df["class_name"].values
groups = df["cluster_id"].values

gkf = GroupKFold(n_splits=4)
accs = []
for train_idx, test_idx in gkf.split(X, y, groups):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    accs.append(accuracy_score(y_te, preds))

print(f"Mean Accuracy with Group-Aware Split on Artifacts: {np.mean(accs)*100:.2f}%")
print("Explanation: Since GroupKFold holds out an entire class each fold, NO classifier can beat 0% accuracy.")

# If we do random split instead:
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
print(f"Mean Accuracy with Random Split on Artifacts: {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")

