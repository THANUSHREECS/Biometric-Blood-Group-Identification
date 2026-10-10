import pandas as pd
import numpy as np
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import struct

# Load dataset manifest
df = pd.read_csv("outputs/dataset_manifest.csv")
# Wait, dataset_manifest.csv doesn't specify train/test. We need the 4-class split.
# Let's find the split CSV.
# We know the random split is 4189 train / 415 test.
# "outputs/dataset_manifest.csv" has a "split" column.
# Let's verify lengths.
train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

def extract_features(path):
    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    
    # Read BMP header (bytes 18-21: width, 22-25: height, 28-29: bit depth)
    try:
        with open(path, "rb") as f:
            header = f.read(54)
            width = struct.unpack("<i", header[18:22])[0]
            height = struct.unpack("<i", header[22:26])[0]
            bpp = struct.unpack("<H", header[28:30])[0]
    except Exception:
        width, height, bpp = 0, 0, 0

    meta = [size, mtime, width, height, bpp]
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return meta, [0,0], [0,0]
        
    h, w = img.shape
    # corners 10x10
    c1, c2, c3, c4 = img[:10,:10], img[:10,w-10:], img[h-10:,:10], img[h-10:,w-10:]
    corners = np.concatenate([c1.flatten(), c2.flatten(), c3.flatten(), c4.flatten()])
    border_stats = [corners.mean(), corners.std()]
    
    # ridge area (center)
    center = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    ridge_stats = [center.mean(), center.std()]
    
    return meta, border_stats, ridge_stats

X_meta_tr, X_border_tr, X_ridge_tr, y_tr = [], [], [], []
X_meta_te, X_border_te, X_ridge_te, y_te = [], [], [], []

for _, row in train_df.iterrows():
    p = row["image_path"]
    m, b, r = extract_features(p)
    X_meta_tr.append(m)
    X_border_tr.append(b)
    X_ridge_tr.append(r)
    y_tr.append(row["class_name"])
    
for _, row in test_df.iterrows():
    p = row["image_path"]
    m, b, r = extract_features(p)
    X_meta_te.append(m)
    X_border_te.append(b)
    X_ridge_te.append(r)
    y_te.append(row["class_name"])

X_meta_tr, X_border_tr, X_ridge_tr = np.array(X_meta_tr), np.array(X_border_tr), np.array(X_ridge_tr)
X_meta_te, X_border_te, X_ridge_te = np.array(X_meta_te), np.array(X_border_te), np.array(X_ridge_te)
y_tr, y_te = np.array(y_tr), np.array(y_te)

def eval_with_ci(X_tr, y_tr, X_te, y_te, name):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    
    # Bootstrap CI
    np.random.seed(42)
    boot_accs = []
    n = len(y_te)
    for _ in range(1000):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        boot_accs.append(accuracy_score(y_te[idx], preds[idx]))
    ci_lower, ci_upper = np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

eval_with_ci(X_border_tr, y_tr, X_border_te, y_te, "(a) Border/corner-pixel stats only")
eval_with_ci(X_meta_tr, y_tr, X_meta_te, y_te, "(b) BMP header + file size + timestamp only")
eval_with_ci(X_ridge_tr, y_tr, X_ridge_te, y_te, "(c) Ridge-area stats only")
