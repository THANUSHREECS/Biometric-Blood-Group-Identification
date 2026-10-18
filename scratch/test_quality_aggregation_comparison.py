#!/usr/bin/env python3
"""
Controlled Comparison Prototype for EXP-10:
Condition A: Unweighted Three-Scan Average
Condition B: Quality-Weighted Average (weights proportional to Q)
Condition C: Best Single Scan (argmax Q)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

split_df = pd.read_csv(SPLIT_CSV)
dev_df = split_df[split_df["split"].isin(["TRAIN", "VALIDATION"])].copy()

dev_pids = sorted(dev_df["participant_id"].unique())
assert len(dev_pids) == 80

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

X_unweighted = []
X_weighted = []
X_best = []
y = []
pids = []

for pid in dev_pids:
    p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
    assert len(p_scans) == 3

    scan_vecs = []
    scan_qs = []

    for _, row in p_scans.iterrows():
        img = cv2.imread(row["scan_path"], cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        mask = (img < 238).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        fg_pixels = np.sum(mask > 0)
        fg_cov = fg_pixels / (h * w)

        sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        g_xx = cv2.filter2D(sobel_x**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
        g_yy = cv2.filter2D(sobel_y**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
        g_xy = cv2.filter2D(sobel_x * sobel_y, -1, np.ones((16, 16), dtype=np.float32) / 256.0)

        v_x = 2.0 * g_xy
        v_y = g_xx - g_yy
        theta = 0.5 * np.arctan2(v_x, v_y)
        coherence = np.sqrt(v_x**2 + v_y**2) / (g_xx + g_yy + 1e-6)

        mean_filt = cv2.filter2D(img.astype(np.float32), -1, np.ones((16, 16), dtype=np.float32) / 256.0)
        sq_filt = cv2.filter2D(img.astype(np.float32)**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
        local_std = np.sqrt(np.maximum(0, sq_filt - mean_filt**2))

        unusable_mask = (mask > 0) & ((coherence < 0.25) | (local_std < 12.0))
        unusable_prop = float(np.sum(unusable_mask) / fg_pixels) if fg_pixels > 0 else 1.0

        mean_coh = float(np.mean(coherence[mask > 0])) if fg_pixels > 0 else 0.0
        mean_con = float(np.mean(local_std[mask > 0])) if fg_pixels > 0 else 0.0

        q = float(fg_cov * mean_coh * (1.0 - unusable_prop) * min(1.0, mean_con / 40.0))
        q = max(0.001, q)

        theta_grid = theta[::16, ::16]
        coherence_grid = coherence[::16, ::16]
        mask_grid = cv2.resize(mask, (theta_grid.shape[1], theta_grid.shape[0]), interpolation=cv2.INTER_NEAREST)
        fg_ind = (mask_grid > 0).astype(np.float32)

        u = (np.cos(2.0 * theta_grid) * coherence_grid * fg_ind).flatten()
        v = (np.sin(2.0 * theta_grid) * coherence_grid * fg_ind).flatten()
        vec = np.concatenate([u, v])

        scan_vecs.append(vec)
        scan_qs.append(q)

    # Condition A: Unweighted mean
    vec_unweighted = np.mean(scan_vecs, axis=0)

    # Condition B: Quality-weighted mean
    weights = np.array(scan_qs) / np.sum(scan_qs)
    vec_weighted = np.sum([w * v for w, v in zip(weights, scan_vecs)], axis=0)

    # Condition C: Best scan
    best_idx = np.argmax(scan_qs)
    vec_best = scan_vecs[best_idx]

    X_unweighted.append(vec_unweighted)
    X_weighted.append(vec_weighted)
    X_best.append(vec_best)

    bg = p_scans.iloc[0]["blood_group"]
    y.append(CANONICAL_REV_MAP[bg])
    pids.append(pid)

X_unweighted = np.array(X_unweighted, dtype=np.float32)
X_weighted = np.array(X_weighted, dtype=np.float32)
X_best = np.array(X_best, dtype=np.float32)
y = np.array(y, dtype=int)

print(f"Data shapes: X_unweighted={X_unweighted.shape}, X_weighted={X_weighted.shape}, X_best={X_best.shape}")

# Evaluate across models using identical 5-fold CV splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "RidgeClassifier": RidgeClassifier(alpha=10.0, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
}

print("\n" + "=" * 80)
print(f"{'Model':20s} | {'Aggregation Condition':28s} | {'Accuracy':10s} | {'Bal Acc':10s} | {'Macro F1':10s} | {'Kappa':8s}")
print("=" * 80)

for m_name, clf in models.items():
    for cond_name, X_mat in [
        ("A: Unweighted Mean", X_unweighted),
        ("B: Quality-Weighted Mean", X_weighted),
        ("C: Best Single Scan", X_best)
    ]:
        oof = np.zeros(len(y), dtype=int)
        for t, v in skf.split(X_mat, y):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_mat[t], y[t])
            oof[v] = pipe.predict(X_mat[v])

        acc = accuracy_score(y, oof)
        bal = balanced_accuracy_score(y, oof)
        f1 = f1_score(y, oof, average="macro", zero_division=0)
        kap = cohen_kappa_score(y, oof)
        print(f"{m_name:20s} | {cond_name:28s} | {acc*100:6.2f}%    | {bal*100:6.2f}%    | {f1:8.4f}   | {kap:6.4f}")
print("=" * 80)
