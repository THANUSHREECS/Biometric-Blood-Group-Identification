import hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd

print("=== AUDIT B: LEAKAGE AND PARTICIPANT SPLIT VERIFICATION ===", flush=True)

# 1. mfs100_final split check (participant_split.csv)
df_split = pd.read_csv("reports/phase3_participant_benchmark/participant_split.csv")
print(f"participant_split.csv rows: {len(df_split)}", flush=True)

parts_by_split = df_split.groupby("split")["participant_id"].unique()
train_parts = set(parts_by_split["TRAIN"])
val_parts = set(parts_by_split["VALIDATION"])
test_parts = set(parts_by_split["TEST"])

print(f"Train participants: {len(train_parts)}", flush=True)
print(f"Validation participants: {len(val_parts)}", flush=True)
print(f"Test participants: {len(test_parts)}", flush=True)
print(f"Total unique participants: {len(train_parts | val_parts | test_parts)}", flush=True)

# Check ID overlap
train_val_overlap = train_parts & val_parts
train_test_overlap = train_parts & test_parts
val_test_overlap = val_parts & test_parts

print(f"Train/Val ID overlap: {train_val_overlap}", flush=True)
print(f"Train/Test ID overlap: {train_test_overlap}", flush=True)
print(f"Val/Test ID overlap: {val_test_overlap}", flush=True)

# Check 3 scans per participant within split
split_per_part = df_split.groupby('participant_id')['split'].nunique()
leaked_parts = split_per_part[split_per_part > 1]
print(f"Participants split across multiple partitions: {len(leaked_parts)}", flush=True)

# Check exact duplicate hashes across splits in mfs100_final (303 files)
hashes_by_split = defaultdict(dict)
all_mfs_hashes = {}
for idx, row in df_split.iterrows():
    p = Path(row["scan_path"])
    if not p.exists():
        p = Path("data/mfs100_final/raw") / row["blood_group"] / row["participant_id"] / p.name
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes_by_split[row["split"]][h] = (row["participant_id"], p.name)
    all_mfs_hashes[h] = (row["participant_id"], row["split"], p.name)

print(f"TRAIN hashes: {len(hashes_by_split['TRAIN'])}, VALIDATION hashes: {len(hashes_by_split['VALIDATION'])}, TEST hashes: {len(hashes_by_split['TEST'])}", flush=True)
h_tv = set(hashes_by_split["TRAIN"].keys()) & set(hashes_by_split["VALIDATION"].keys())
h_tt = set(hashes_by_split["TRAIN"].keys()) & set(hashes_by_split["TEST"].keys())
h_vt = set(hashes_by_split["VALIDATION"].keys()) & set(hashes_by_split["TEST"].keys())
print(f"Exact hash overlap Train/Val: {len(h_tv)}, Train/Test: {len(h_tt)}, Val/Test: {len(h_vt)}", flush=True)

# 2. Check 26-cohort overlap with mfs100_final
df_cap = pd.read_csv("data/mfs100_ground_truth.csv")
print(f"\ndata/mfs100_ground_truth.csv rows: {len(df_cap)}", flush=True)
cap_hashes = {}
for idx, row in df_cap.iterrows():
    p = Path("data/captured") / row["capture_filename"]
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        cap_hashes[h] = (row["subject_id"], row["capture_filename"])

cross_cohort_exact = set(all_mfs_hashes.keys()) & set(cap_hashes.keys())
print(f"Exact image hash overlap between mfs100_final and captured 26-cohort: {len(cross_cohort_exact)}", flush=True)

# 3. Check original 6000 dataset manifest
df_man = pd.read_csv("outputs/dataset_manifest.csv")
print(f"\noutputs/dataset_manifest.csv: {len(df_man)} rows", flush=True)
print(f"Columns: {list(df_man.columns)}", flush=True)
print(f"Splits: {df_man['split'].value_counts().to_dict()}", flush=True)
has_part_id = any("participant" in c.lower() or "subject" in c.lower() for c in df_man.columns)
print(f"Participant-level ID column present in 6000 dataset: {has_part_id}", flush=True)
sample_names = df_man["image_path"].head(5).apply(lambda x: Path(x).name).tolist()
print(f"Sample filename structure in 6000 dataset: {sample_names}", flush=True)
