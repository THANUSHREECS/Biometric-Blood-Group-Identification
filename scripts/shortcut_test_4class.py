import pandas as pd
import numpy as np
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import struct

# Load dataset manifest
df = pd.read_csv("outputs/dataset_manifest.csv")
classes = ['A+', 'B+', 'AB+', 'O+']
df = df[df['class_name'].isin(classes)]

test_df = pd.read_csv("outputs/abo_positive_4class/test_predictions.csv")
test_paths = set(test_df['image_path'].tolist())

train_df = df[~df['image_path'].isin(test_paths)]
real_test_df = df[df['image_path'].isin(test_paths)]

print(f"Train size: {len(train_df)}, Test size: {len(real_test_df)}")
counts = real_test_df['class_name'].value_counts()
print(f"Test Class Counts:\n{counts}")
print(f"Majority Baseline: {counts.iloc[0]/len(real_test_df)*100:.2f}%")

def extract_features(path):
    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    
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
        return meta, [0,0], [0,0], [0,0]
        
    h, w = img.shape
    c1, c2, c3, c4 = img[:10,:10], img[:10,w-10:], img[h-10:,:10], img[h-10:,w-10:]
    corners = np.concatenate([c1.flatten(), c2.flatten(), c3.flatten(), c4.flatten()])
    border_stats = [corners.mean(), corners.std()]
    
    center = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    ridge_stats = [center.mean(), center.std()]
    
    intensity_stats = [img.mean(), img.std(), np.percentile(img, 5), np.percentile(img, 95)]
    
    return meta, border_stats, ridge_stats, intensity_stats

X_meta_tr, X_border_tr, X_ridge_tr, X_int_tr, y_tr = [], [], [], [], []
X_meta_te, X_border_te, X_ridge_te, X_int_te, y_te = [], [], [], [], []

for _, row in train_df.iterrows():
    p = row["image_path"]
    if not os.path.exists(p): continue
    m, b, r, i = extract_features(p)
    X_meta_tr.append(m); X_border_tr.append(b); X_ridge_tr.append(r); X_int_tr.append(i); y_tr.append(row["class_name"])
    
for _, row in real_test_df.iterrows():
    p = row["image_path"]
    if not os.path.exists(p): continue
    m, b, r, i = extract_features(p)
    X_meta_te.append(m); X_border_te.append(b); X_ridge_te.append(r); X_int_te.append(i); y_te.append(row["class_name"])

def eval_with_ci(X_tr, y_tr, X_te, y_te, name):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    y_te = np.array(y_te)
    acc = accuracy_score(y_te, preds)
    
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
eval_with_ci(X_int_tr, y_tr, X_int_te, y_te, "(d) Intensity/contrast")
