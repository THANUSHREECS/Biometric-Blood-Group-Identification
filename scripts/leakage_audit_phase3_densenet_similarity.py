import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*75)
print("PHASE 3 STEP (a): DENSENET-121 MAX-SIMILARITY DISTRIBUTION (ORIGINAL DATASET)")
print("="*75)

# Load positive dataset manifest
manifest = pd.read_csv(PROJECT_ROOT / "outputs" / "dataset_manifest.csv")
pos_df = manifest[manifest["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy().reset_index(drop=True)
print(f"Total positive images: {len(pos_df)}")
print(f"Split breakdown:\n{pos_df['split'].value_counts().to_string()}")

# Check if features are already extracted and cached
cache_path = OUT_DIR / "original_dataset_densenet121_features.npz"
if cache_path.is_file():
    print(f"\nLoading pre-cached DenseNet-121 features from {cache_path}...")
    npz = np.load(cache_path)
    features = npz["features"]
    labels = npz["labels"]
    splits = npz["splits"]
    paths = npz["paths"]
else:
    print("\nInstantiating pre-trained DenseNet-121 backbone (ImageNet weights, GAP)...")
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    base.trainable = False
    gap = tf.keras.layers.GlobalAveragePooling2D()

    # Preprocess images: standard 128x128 RGB normalization as used in historical dataset
    print(f"Extracting DenseNet-121 features for {len(pos_df)} images...")
    t0 = time.time()
    feats_list = []
    
    # Process in batches of 64
    batch_size = 64
    n_batches = int(np.ceil(len(pos_df) / batch_size))
    
    for b in range(n_batches):
        b_df = pos_df.iloc[b*batch_size : (b+1)*batch_size]
        batch_imgs = []
        for _, row in b_df.iterrows():
            img_path = str(row["image_path"])
            # Load and decode image
            raw = tf.io.read_file(img_path)
            img = tf.image.decode_bmp(raw)
            if tf.shape(img)[-1] == 1:
                img = tf.image.grayscale_to_rgb(img)
            img = tf.image.resize(img, [128, 128])
            img = tf.keras.applications.densenet.preprocess_input(img)
            batch_imgs.append(img)
            
        batch_tensor = tf.stack(batch_imgs)
        batch_feats = gap(base(batch_tensor)).numpy()
        feats_list.append(batch_feats)
        
        if (b + 1) % 10 == 0 or b == n_batches - 1:
            print(f"  Processed batch {b+1}/{n_batches} ({min((b+1)*batch_size, len(pos_df))}/{len(pos_df)} images, {time.time()-t0:.1f}s)")

    features = np.vstack(feats_list).astype(np.float32)
    labels = pos_df["class_name"].values
    splits = pos_df["split"].values
    paths = pos_df["image_path"].values

    print(f"Extraction complete in {time.time()-t0:.1f}s. Saving to {cache_path}...")
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        splits=splits,
        paths=paths
    )

# Normalize features to unit sphere for exact cosine similarity
norms = np.linalg.norm(features, axis=1, keepdims=True)
features_norm = features / np.clip(norms, 1e-12, None)

train_mask = (splits == "train")
val_mask = (splits == "val")
test_mask = (splits == "test")
eval_mask = (splits == "val") | (splits == "test")

feats_train = features_norm[train_mask]
labels_train = labels[train_mask]

feats_test = features_norm[test_mask]
labels_test = labels[test_mask]

feats_val = features_norm[val_mask]
labels_val = labels[val_mask]

feats_eval = features_norm[eval_mask]
labels_eval = labels[eval_mask]

print(f"\nFeature subsets:")
print(f"  Train: {len(feats_train)} images")
print(f"  Val:   {len(feats_val)} images")
print(f"  Test:  {len(feats_test)} images")

# -------------------------------------------------------------------------
# 1. Cosine similarity: TEST -> TRAIN
# -------------------------------------------------------------------------
print("\n" + "="*75)
print("1. LEGACY TEST -> TRAIN MAX-SIMILARITY DISTRIBUTION (N=415 Test Images)")
print("="*75)

# Full matrix (415 x 1938)
sim_matrix_test_train = np.dot(feats_test, feats_train.T)
max_sim_test_to_train = np.max(sim_matrix_test_train, axis=1)

# Intra-class only: max similarity to training image of the SAME class
same_class_mask_test_train = (labels_test[:, None] == labels_train[None, :])
sim_matrix_same_class = np.where(same_class_mask_test_train, sim_matrix_test_train, -1.0)
max_sim_test_to_train_same_class = np.max(sim_matrix_same_class, axis=1)

def print_distribution_summary(sims, title):
    print(f"\n--- {title} ---")
    print(f"Count:  {len(sims)}")
    print(f"Mean:   {np.mean(sims):.4f}")
    print(f"Std:    {np.std(sims):.4f}")
    print(f"Min:    {np.min(sims):.4f}")
    print(f"P25:    {np.percentile(sims, 25):.4f}")
    print(f"P50:    {np.percentile(sims, 50):.4f} (Median)")
    print(f"P75:    {np.percentile(sims, 75):.4f}")
    print(f"P90:    {np.percentile(sims, 90):.4f}")
    print(f"P95:    {np.percentile(sims, 95):.4f}")
    print(f"P99:    {np.percentile(sims, 99):.4f}")
    print(f"Max:    {np.max(sims):.4f}")
    
    thresholds = [0.99, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80]
    print("\nThreshold Exceedance Counts:")
    for t in thresholds:
        cnt = np.sum(sims >= t)
        pct = (cnt / len(sims)) * 100
        print(f"  >= {t:.2f}: {cnt:4d} / {len(sims)} ({pct:5.2f}%)")

print_distribution_summary(max_sim_test_to_train, "TEST -> TRAIN (Any Class)")
print_distribution_summary(max_sim_test_to_train_same_class, "TEST -> TRAIN (Same Class Only)")

# -------------------------------------------------------------------------
# 2. Cosine similarity: TRAIN -> TEST (and TRAIN -> EVAL)
# -------------------------------------------------------------------------
print("\n" + "="*75)
print("2. TRAINING IMAGES MAX-SIMILARITY TO TEST / EVAL (N=1938 Train Images)")
print("="*75)

# For each training image, what is its maximum similarity to ANY test image?
max_sim_train_to_test = np.max(sim_matrix_test_train, axis=0) # shape (1938,)

# For each training image, what is its maximum similarity to ANY evaluation image (val + test)?
sim_matrix_eval_train = np.dot(feats_eval, feats_train.T) # shape (831, 1938)
max_sim_train_to_eval = np.max(sim_matrix_eval_train, axis=0) # shape (1938,)

# Same-class max similarity from train to test
max_sim_train_to_test_same_class = np.max(sim_matrix_same_class, axis=0)

print_distribution_summary(max_sim_train_to_test, "TRAIN -> TEST (Any Class)")
print_distribution_summary(max_sim_train_to_test_same_class, "TRAIN -> TEST (Same Class Only)")
print_distribution_summary(max_sim_train_to_eval, "TRAIN -> (VAL + TEST) (Any Class)")

# -------------------------------------------------------------------------
# 3. Fraction of Training Images Removed at Candidate Purge Thresholds
# -------------------------------------------------------------------------
print("\n" + "="*75)
print("3. CANDIDATE PURGE THRESHOLDS (TRAIN IMAGES PURGED VS TEST SET)")
print("="*75)
print(f"{'Threshold (tau)':<16} | {'Train Purged (Count)':<20} | {'Train Purged (%)':<16} | {'Remaining Train':<16}")
print("-" * 75)
for tau in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.85, 0.80]:
    purged_cnt = int(np.sum(max_sim_train_to_test >= tau))
    purged_pct = (purged_cnt / len(feats_train)) * 100
    rem = len(feats_train) - purged_cnt
    print(f"tau >= {tau:4.2f}        | {purged_cnt:4d} / {len(feats_train)}          | {purged_pct:5.2f}%          | {rem:4d}")

# Save distribution summary to JSON
dist_summary = {
    "test_to_train_any_class": {
        "mean": float(np.mean(max_sim_test_to_train)),
        "std": float(np.std(max_sim_test_to_train)),
        "median": float(np.median(max_sim_test_to_train)),
        "p90": float(np.percentile(max_sim_test_to_train, 90)),
        "p95": float(np.percentile(max_sim_test_to_train, 95)),
        "p99": float(np.percentile(max_sim_test_to_train, 99)),
        "exceedance": {str(t): int(np.sum(max_sim_test_to_train >= t)) for t in [0.99, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80]}
    },
    "train_to_test_any_class": {
        "mean": float(np.mean(max_sim_train_to_test)),
        "std": float(np.std(max_sim_train_to_test)),
        "median": float(np.median(max_sim_train_to_test)),
        "p90": float(np.percentile(max_sim_train_to_test, 90)),
        "p95": float(np.percentile(max_sim_train_to_test, 95)),
        "p99": float(np.percentile(max_sim_train_to_test, 99)),
        "exceedance": {str(t): int(np.sum(max_sim_train_to_test >= t)) for t in [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.85]}
    }
}

import json
with open(OUT_DIR / "phase3_densenet121_similarity_distribution.json", "w") as f:
    json.dump(dist_summary, f, indent=2)

print(f"\nSaved distribution summary to {OUT_DIR / 'phase3_densenet121_similarity_distribution.json'}")
