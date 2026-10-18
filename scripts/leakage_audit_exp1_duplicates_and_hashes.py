"""
scripts/leakage_audit_exp1_duplicates_and_hashes.py
Experiment 1: Part 1 - Exact Duplicates and Perceptual Hashing Audit.

Audits:
1. Exact SHA-256 duplicate image detection across the legacy 4-class positive dataset.
2. Cross-split duplicate leakage across train, val, and test splits in outputs/dataset_manifest.csv.
3. pHash and dHash computation using imagehash.
4. Near-duplicate Hamming distance analysis between legacy test and train images under pre-declared thresholds:
   - Strict: Hamming <= 2
   - Moderate: Hamming <= 5
   - Permissive: Hamming <= 10
5. Outputs saved to JSON and CSV in experiments/leakage_audit/.
"""

import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

OUT_DIR = Path("experiments/leakage_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_file = Path("outputs/dataset_manifest.csv")
assert manifest_file.exists(), f"Missing manifest: {manifest_file}"

df_man = pd.read_csv(manifest_file)
pos_classes = ["A+", "AB+", "B+", "O+"]
df_pos = df_man[df_man["class_name"].isin(pos_classes)].copy().reset_index(drop=True)

print(f"Total positive-class images in manifest: {len(df_pos)}")
print(f"Split distribution:\n{df_pos['split'].value_counts()}")

# Resolve local paths
def resolve_local_path(p_str):
    idx = p_str.find("data/processed")
    if idx != -1:
        local_p = Path(p_str[idx:])
    else:
        local_p = Path(p_str)
    if not local_p.exists():
        # Try finding in data/processed by filename
        fn = Path(p_str).name
        cname = p_str.split("/")[-2] if "/" in p_str else p_str.split("\\")[-2]
        local_p = Path("data/processed") / cname / fn
    assert local_p.exists(), f"File not found: {local_p}"
    return local_p

df_pos["local_path"] = df_pos["image_path"].apply(resolve_local_path)

# 1. Exact SHA-256 duplicates
print("\n--- Computing SHA-256 Hashes ---")
sha256_to_indices = defaultdict(list)
for idx, row in df_pos.iterrows():
    h = hashlib.sha256(row["local_path"].read_bytes()).hexdigest()
    sha256_to_indices[h].append(idx)

exact_dups = {h: idxs for h, idxs in sha256_to_indices.items() if len(idxs) > 1}
print(f"Unique SHA-256 hashes: {len(sha256_to_indices)}")
print(f"Number of exact duplicate hash groups: {len(exact_dups)}")

cross_split_exact = []
for h, idxs in exact_dups.items():
    splits = df_pos.loc[idxs, "split"].unique()
    if len(splits) > 1:
        cross_split_exact.append({
            "hash": h,
            "indices": idxs,
            "splits": list(splits),
            "files": df_pos.loc[idxs, "local_path"].astype(str).tolist()
        })

print(f"Exact duplicate groups crossing splits: {len(cross_split_exact)}")

# Also check duplicates against data/quarantine/duplicates
quarantine_dir = Path("data/quarantine/duplicates")
quarantine_dups = []
if quarantine_dir.exists():
    for qf in quarantine_dir.glob("*.BMP"):
        qh = hashlib.sha256(qf.read_bytes()).hexdigest()
        if qh in sha256_to_indices:
            quarantine_dups.append((qf.name, [str(df_pos.loc[i, "local_path"]) for i in sha256_to_indices[qh]]))
print(f"Quarantined files matching active manifest hashes: {len(quarantine_dups)}")

# 2. Perceptual Hashes (pHash and dHash)
print("\n--- Computing Perceptual Hashes (pHash & dHash) ---")
phashes = []
dhashes = []

for idx, row in df_pos.iterrows():
    with Image.open(row["local_path"]) as img:
        p_h = imagehash.phash(img)
        d_h = imagehash.dhash(img)
        phashes.append(p_h)
        dhashes.append(d_h)

df_pos["phash"] = [str(h) for h in phashes]
df_pos["dhash"] = [str(h) for h in dhashes]

# Pre-declared Hamming distance thresholds
HAMMING_THRESHOLDS = [2, 5, 10]

train_mask = df_pos["split"] == "train"
test_mask = df_pos["split"] == "test"

train_phashes = [phashes[i] for i in np.where(train_mask)[0]]
test_phashes = [phashes[i] for i in np.where(test_mask)[0]]

train_dhashes = [dhashes[i] for i in np.where(train_mask)[0]]
test_dhashes = [dhashes[i] for i in np.where(test_mask)[0]]

n_test = len(test_phashes)
print(f"Comparing {n_test} test images against {len(train_phashes)} train images...")

min_phash_dist_to_train = []
min_dhash_dist_to_train = []

for t_idx in range(n_test):
    tp = test_phashes[t_idx]
    td = test_dhashes[t_idx]
    
    p_dists = [tp - tr_p for tr_p in train_phashes]
    d_dists = [td - tr_d for tr_d in train_dhashes]
    
    min_phash_dist_to_train.append(min(p_dists))
    min_dhash_dist_to_train.append(min(d_dists))

min_phash_dist_to_train = np.array(min_phash_dist_to_train)
min_dhash_dist_to_train = np.array(min_dhash_dist_to_train)

phash_leakage_counts = {}
dhash_leakage_counts = {}

for th in HAMMING_THRESHOLDS:
    p_cnt = int(np.sum(min_phash_dist_to_train <= th))
    d_cnt = int(np.sum(min_dhash_dist_to_train <= th))
    phash_leakage_counts[f"hamming_le_{th}"] = {
        "count": p_cnt,
        "percentage": float(p_cnt / n_test * 100)
    }
    dhash_leakage_counts[f"hamming_le_{th}"] = {
        "count": d_cnt,
        "percentage": float(d_cnt / n_test * 100)
    }
    print(f"pHash Hamming <= {th:2}: {p_cnt:3}/{n_test} test images ({p_cnt / n_test * 100:.2f}%) have near-duplicates in TRAIN")
    print(f"dHash Hamming <= {th:2}: {d_cnt:3}/{n_test} test images ({d_cnt / n_test * 100:.2f}%) have near-duplicates in TRAIN")

# Save outputs
dup_results = {
    "manifest_positive_count": len(df_pos),
    "exact_duplicates": {
        "unique_hashes": len(sha256_to_indices),
        "duplicate_groups": len(exact_dups),
        "cross_split_groups": len(cross_split_exact),
        "cross_split_details": cross_split_exact
    },
    "perceptual_hashes": {
        "hamming_thresholds_pre_declared": HAMMING_THRESHOLDS,
        "test_sample_count": n_test,
        "phash_min_dist_summary": {
            "mean": float(np.mean(min_phash_dist_to_train)),
            "std": float(np.std(min_phash_dist_to_train)),
            "min": int(np.min(min_phash_dist_to_train)),
            "max": int(np.max(min_phash_dist_to_train))
        },
        "dhash_min_dist_summary": {
            "mean": float(np.mean(min_dhash_dist_to_train)),
            "std": float(np.std(min_dhash_dist_to_train)),
            "min": int(np.min(min_dhash_dist_to_train)),
            "max": int(np.max(min_dhash_dist_to_train))
        },
        "phash_near_duplicate_counts": phash_leakage_counts,
        "dhash_near_duplicate_counts": dhash_leakage_counts
    }
}

json_path = OUT_DIR / "exp1_duplicates_and_perceptual_hashes.json"
with open(json_path, "w") as f:
    json.dump(dup_results, f, indent=2)

# Save image-level perceptual hash ledger
df_pos_hashes = df_pos[["image_path", "class_name", "split", "phash", "dhash"]].copy()
csv_path = OUT_DIR / "exp1_positive_images_hashes.csv"
df_pos_hashes.to_csv(csv_path, index=False)

print(f"\n[SAVED] Hash audit results saved to {json_path}")
print(f"[SAVED] Hash ledger saved to {csv_path}")
print("=== PART 1 COMPLETED SUCCESSFULLY ===")
