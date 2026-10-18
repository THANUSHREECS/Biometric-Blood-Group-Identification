import os
from pathlib import Path
import pandas as pd
import hashlib

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
GT_CSV = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"

df_split = pd.read_csv(SPLIT_CSV)
df_gt = pd.read_csv(GT_CSV)

print("=== PARTICIPANT SPLIT AUDIT ===")
split_counts = df_split.groupby("split")["participant_id"].nunique()
print("Unique participants per split:")
print(split_counts)

print("\nScans per split:")
print(df_split["split"].value_counts())

train_pids = set(df_split[df_split["split"] == "TRAIN"]["participant_id"].unique())
val_pids = set(df_split[df_split["split"] == "VALIDATION"]["participant_id"].unique())
test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
gt_pids = set(df_gt["subject_id"].unique())

print(f"\nTRAIN IDs ({len(train_pids)}):", sorted(list(train_pids))[:10], "...")
print(f"VALIDATION IDs ({len(val_pids)}):", sorted(list(val_pids)))
print(f"LOCKED TEST IDs ({len(test_pids)}):", sorted(list(test_pids)))
print(f"26 COHORT GT IDs ({len(gt_pids)}):", sorted(list(gt_pids)))

# Intersections
print("\n=== ID INTERSECTIONS ===")
print("TRAIN & VALIDATION:", train_pids & val_pids)
print("TRAIN & TEST:", train_pids & test_pids)
print("VAL & TEST:", val_pids & test_pids)
print("DEV (TRAIN+VAL) & TEST:", (train_pids | val_pids) & test_pids)
print("GT_26 & TRAIN:", gt_pids & train_pids)
print("GT_26 & VAL:", gt_pids & val_pids)
print("GT_26 & TEST:", gt_pids & test_pids)
print("GT_26 & ALL_101:", gt_pids & (train_pids | val_pids | test_pids))

# Physical file check for 101 participants
missing_split_scans = 0
for p in df_split["scan_path"]:
    if not Path(p).is_file():
        missing_split_scans += 1

print(f"\nPhysical scans missing in participant_split.csv: {missing_split_scans} / {len(df_split)}")

# Check ground-truth class distribution of TEST (21 participants)
test_df_participants = df_split[df_split["split"] == "TEST"].drop_duplicates(subset=["participant_id"])
print("\nLOCKED TEST (21 participants) Class Distribution:")
print(test_df_participants["blood_group"].value_counts())

# Check ground-truth class distribution of DEV (80 participants)
dev_df_participants = df_split[df_split["split"] != "TEST"].drop_duplicates(subset=["participant_id"])
print("\nDEVELOPMENT (80 participants) Class Distribution:")
print(dev_df_participants["blood_group"].value_counts())

# Check ground-truth class distribution of GT_26
gt_df_participants = df_gt.drop_duplicates(subset=["subject_id"])
print("\n26-COHORT Ground Truth Class Distribution:")
print(gt_df_participants["confirmed_blood_group"].value_counts())
