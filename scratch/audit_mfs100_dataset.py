import os
import sys
import pandas as pd
import hashlib

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
dataset_dir = os.path.join(project_root, "data", "mfs100_final")
split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")

df = pd.read_csv(split_csv_path)

# Verify counts
total_scans = len(df)
participants = df["participant_id"].unique()
total_participants = len(participants)

print(f"Total scans: {total_scans}")
print(f"Total participants: {total_participants}")

bg_counts = df.groupby("participant_id").first()["blood_group"].value_counts()
print(f"Class counts:\n{bg_counts}")

# Split counts
split_counts = df.groupby("participant_id").first()["split"].value_counts()
print(f"Split counts:\n{split_counts}")

# Scan counts per participant
scans_per_pid = df.groupby("participant_id").size()
if not all(scans_per_pid == 3):
    print("WARNING: Not all participants have 3 scans!")
else:
    print("All participants have exactly 3 scans.")

# Check for leakage
train_pids = set(df[df["split"]=="TRAIN"]["participant_id"])
val_pids = set(df[df["split"]=="VALIDATION"]["participant_id"])
test_pids = set(df[df["split"]=="TEST"]["participant_id"])

if train_pids.intersection(val_pids) or train_pids.intersection(test_pids) or val_pids.intersection(test_pids):
    print("WARNING: Leakage detected between splits!")
else:
    print("No participant leakage between splits.")

# Check for exact duplicates using md5
hashes = {}
duplicates = []
for i, row in df.iterrows():
    path = row["scan_path"]
    if not os.path.exists(path):
        print(f"WARNING: File missing: {path}")
        continue
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    if file_hash in hashes:
        duplicates.append((path, hashes[file_hash]))
    else:
        hashes[file_hash] = path

if duplicates:
    print(f"WARNING: Found {len(duplicates)} exact duplicates!")
else:
    print("No exact duplicate files found.")

# Verify files in directory match dataframe
actual_files = []
for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(('.png', '.bmp', '.jpg')):
            actual_files.append(os.path.join(root, f))
print(f"Files in dir: {len(actual_files)}")
if len(actual_files) != total_scans:
    print("WARNING: Directory file count does not match CSV!")

print("Audit complete.")
