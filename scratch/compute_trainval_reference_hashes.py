"""
scratch/compute_trainval_reference_hashes.py
Computes and serializes perceptual hash duplicate metrics of the 415 test images
against the combined Train+Val reference set (N = 2,354 images).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import imagehash

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
csv_path = EXP_DIR / "exp1_positive_images_hashes.csv"

df = pd.read_csv(csv_path)

trainval_mask = df["split"].isin(["train", "val"])
test_mask = df["split"] == "test"

trainval_df = df[trainval_mask].reset_index(drop=True)
test_df = df[test_mask].reset_index(drop=True)

print(f"Train+Val count: {len(trainval_df)}")
print(f"Test count: {len(test_df)}")

# Convert hex strings back to imagehash objects
trainval_phash = [imagehash.hex_to_hash(h) for h in trainval_df["phash"]]
test_phash = [imagehash.hex_to_hash(h) for h in test_df["phash"]]

trainval_dhash = [imagehash.hex_to_hash(h) for h in trainval_df["dhash"]]
test_dhash = [imagehash.hex_to_hash(h) for h in test_df["dhash"]]

HAMMING_THRESHOLDS = [2, 5, 10]
n_test = len(test_df)

min_phash_dist = []
min_dhash_dist = []

for t_idx in range(n_test):
    tp = test_phash[t_idx]
    td = test_dhash[t_idx]
    
    p_dists = [tp - ref_p for ref_p in trainval_phash]
    d_dists = [td - ref_d for ref_d in trainval_dhash]
    
    min_phash_dist.append(min(p_dists))
    min_dhash_dist.append(min(d_dists))

min_phash_dist = np.array(min_phash_dist)
min_dhash_dist = np.array(min_dhash_dist)

phash_counts = {}
dhash_counts = {}

for th in HAMMING_THRESHOLDS:
    p_cnt = int(np.sum(min_phash_dist <= th))
    d_cnt = int(np.sum(min_dhash_dist <= th))
    phash_counts[f"hamming_le_{th}"] = {
        "count": p_cnt,
        "percentage": float(p_cnt / n_test * 100)
    }
    dhash_counts[f"hamming_le_{th}"] = {
        "count": d_cnt,
        "percentage": float(d_cnt / n_test * 100)
    }

results = {
    "reference_set": "train_plus_val",
    "reference_sample_count": len(trainval_df),
    "test_sample_count": n_test,
    "hamming_thresholds_pre_declared": HAMMING_THRESHOLDS,
    "phash_min_dist_summary": {
        "mean": float(np.mean(min_phash_dist)),
        "std": float(np.std(min_phash_dist)),
        "min": int(np.min(min_phash_dist)),
        "max": int(np.max(min_phash_dist))
    },
    "dhash_min_dist_summary": {
        "mean": float(np.mean(min_dhash_dist)),
        "std": float(np.std(min_dhash_dist)),
        "min": int(np.min(min_dhash_dist)),
        "max": int(np.max(min_dhash_dist))
    },
    "phash_near_duplicate_counts": phash_counts,
    "dhash_near_duplicate_counts": dhash_counts
}

out_file = EXP_DIR / "exp1_perceptual_hashes_trainval_reference.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved train+val reference results to {out_file}")
print("pHash counts:", phash_counts)
print("dHash counts:", dhash_counts)
