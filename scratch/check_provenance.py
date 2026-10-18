import hashlib
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"

GT_CSV = DATA_ROOT / "mfs100_ground_truth.csv"
SPLIT_CSV = REPORTS_ROOT / "phase3_participant_benchmark" / "participant_split.csv"

df_gt = pd.read_csv(GT_CSV)
df_split = pd.read_csv(SPLIT_CSV)

print("GT Participants (S001-S026):", sorted(df_gt["subject_id"].unique()))
print("Split Participants (first 10):", sorted(df_split["participant_id"].unique())[:10])

# Compute file hashes of captured files in GT
gt_hashes = {}
for idx, row in df_gt.iterrows():
    fpath = DATA_ROOT / "captured" / row["capture_filename"]
    if fpath.is_file():
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()
        gt_hashes[h] = (row["subject_id"], row["capture_filename"])

print(f"Total GT images hashed: {len(gt_hashes)}")

# Check if any of these hashes exist in the 101 participant dataset or dataset_blood_group
match_count = 0
matches = []

# Scan participant_split scan_paths
for idx, row in df_split.iterrows():
    scan_p = Path(row["scan_path"])
    if not scan_p.is_absolute():
        scan_p = PROJECT_ROOT / scan_p
    if scan_p.is_file():
        h = hashlib.sha256(scan_p.read_bytes()).hexdigest()
        if h in gt_hashes:
            match_count += 1
            matches.append((row["participant_id"], str(scan_p), gt_hashes[h]))

print(f"Hash collision between GT 26 cohort and 101-participant split: {match_count}")
if matches:
    print("Matches found:", matches[:5])

# Also check timestamps and confirmation sources from GT CSV
print("\nConfirmation sources in GT CSV:")
print(df_gt["confirmation_source"].value_counts())
print("\nCapture timestamps in GT CSV (min, max):")
print(df_gt["timestamp"].min(), "to", df_gt["timestamp"].max())
print("\nNotes sample:")
print(df_gt[["subject_id", "notes"]].drop_duplicates().head(5))
