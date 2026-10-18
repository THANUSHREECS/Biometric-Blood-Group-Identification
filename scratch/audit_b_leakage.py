import hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import json

print("=== AUDIT B: LEAKAGE AND PARTICIPANT SPLIT VERIFICATION ===")

# 1. mfs100_final split check (participant_split.csv)
split_csv = Path("reports/phase3_participant_benchmark/participant_split.csv")
df_split = pd.read_csv(split_csv)
print(f"participant_split.csv rows: {len(df_split)}")

parts_by_split = df_split.groupby("split")["participant_id"].unique()
train_parts = set(parts_by_split["TRAIN"])
val_parts = set(parts_by_split["VALIDATION"])
test_parts = set(parts_by_split["TEST"])

print(f"Train participants: {len(train_parts)}")
print(f"Validation participants: {len(val_parts)}")
print(f"Test participants: {len(test_parts)}")
print(f"Total unique participants: {len(train_parts | val_parts | test_parts)}")

# Check ID overlap
train_val_overlap = train_parts & val_parts
train_test_overlap = train_parts & test_parts
val_test_overlap = val_parts & test_parts

print(f"Train/Val ID overlap: {train_val_overlap}")
print(f"Train/Test ID overlap: {train_test_overlap}")
print(f"Val/Test ID overlap: {val_test_overlap}")

# Check 3 scans per participant within split
scans_per_part_split = df_split.groupby(["participant_id", "split"]).size()
print(f"Participants split across multiple splits: {len(df_split.groupby('participant_id')['split'].nunique()[lambda x: x > 1])}")

# Check image hashes across splits in mfs100_final
hashes_by_split = defaultdict(dict)
all_mfs_hashes = {}
for idx, row in df_split.iterrows():
    fpath = Path(row["scan_path"])
    if not fpath.exists():
        # try relative path
        fpath = Path("data/mfs100_final/raw") / row["blood_group"] / row["participant_id"] / Path(row["scan_path"]).name
    h = hashlib.sha256(fpath.read_bytes()).hexdigest()
    hashes_by_split[row["split"]][h] = (row["participant_id"], fpath.name)
    all_mfs_hashes[h] = (row["participant_id"], row["split"], fpath.name)

print(f"Unique hashes in TRAIN: {len(hashes_by_split['TRAIN'])}")
print(f"Unique hashes in VALIDATION: {len(hashes_by_split['VALIDATION'])}")
print(f"Unique hashes in TEST: {len(hashes_by_split['TEST'])}")

h_tv = set(hashes_by_split["TRAIN"].keys()) & set(hashes_by_split["VALIDATION"].keys())
h_tt = set(hashes_by_split["TRAIN"].keys()) & set(hashes_by_split["TEST"].keys())
h_vt = set(hashes_by_split["VALIDATION"].keys()) & set(hashes_by_split["TEST"].keys())
print(f"Exact image hash overlap Train/Val: {len(h_tv)}")
print(f"Exact image hash overlap Train/Test: {len(h_tt)}")
print(f"Exact image hash overlap Val/Test: {len(h_vt)}")

# 2. Check outputs/dataset_manifest.csv (original 6000 dataset)
manifest_csv = Path("outputs/dataset_manifest.csv")
if manifest_csv.exists():
    df_man = pd.read_csv(manifest_csv)
    print(f"\noutputs/dataset_manifest.csv rows: {len(df_man)}")
    print(f"Columns: {list(df_man.columns)}")
    print(f"Splits in manifest: {df_man['split'].value_counts().to_dict()}")
    
    # Check if participant ID exists
    has_part_id = any("participant" in c.lower() or "subject" in c.lower() for c in df_man.columns)
    print(f"Participant/Subject column in original manifest: {has_part_id}")
    
    # Check filename patterns
    sample_files = df_man["image_path"].head(10).tolist()
    print("Sample file paths:", sample_files[:3])
    
    # Check duplicate hashes in original manifest across train/val/test
    man_hashes = defaultdict(dict)
    for idx, row in df_man.iterrows():
        p_str = row["image_path"]
        # extract relative path e.g. data/processed/...
        idx_p = p_str.find("data/processed")
        if idx_p != -1:
            local_p = Path(p_str[idx_p:])
        else:
            local_p = Path(p_str)
        if local_p.exists():
            h = hashlib.sha256(local_p.read_bytes()).hexdigest()
            man_hashes[row["split"]][h] = local_p.name
    
    print(f"Manifest processed hashes loaded: { {k: len(v) for k, v in man_hashes.items()} }")
    if "train" in man_hashes and "test" in man_hashes:
        cross_man = set(man_hashes["train"].keys()) & set(man_hashes["test"].keys())
        print(f"Original 6000 dataset Train/Test exact hash overlap: {len(cross_man)}")

# 3. Check hash overlap between mfs100_final and data/captured
cap_gt_csv = Path("data/mfs100_ground_truth.csv")
if cap_gt_csv.exists():
    df_cap = pd.read_csv(cap_gt_csv)
    print(f"\ndata/mfs100_ground_truth.csv rows: {len(df_cap)}")
    cap_hashes = {}
    for idx, row in df_cap.iterrows():
        fpath = Path("data/captured") / row["capture_filename"]
        if fpath.exists():
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            cap_hashes[h] = (row["subject_id"], row["capture_filename"])
    print(f"Captured cohort unique hashes: {len(cap_hashes)}")
    
    cross_cohort_exact = set(all_mfs_hashes.keys()) & set(cap_hashes.keys())
    print(f"Exact image hash overlap between mfs100_final and captured cohort: {len(cross_cohort_exact)}")
