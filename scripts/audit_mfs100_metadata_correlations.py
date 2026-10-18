# PROVENANCE NOTE:
# This script is a verified copy of scratch/audit_mfs100_metadata_correlations.py,
# preserved in scripts/ for full audit reproduction and provenance.
# Do not modify or delete the scratch original.

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, chi2_contingency
from datetime import datetime

split_csv = Path("reports/phase3_participant_benchmark/participant_split.csv")
df = pd.read_csv(split_csv)
dev_df = df[df["split"].isin(["TRAIN", "VALIDATION"])].copy()

n_dev_parts = dev_df["participant_id"].nunique()
print(f"Total dev rows: {len(dev_df)} (Participants: {n_dev_parts})")

# Extract file metadata
records = []
for idx, row in dev_df.iterrows():
    p = Path(row["scan_path"])
    if not p.exists():
        p = Path("data/mfs100_final/raw") / row["blood_group"] / row["participant_id"] / p.name
    stat = p.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime)
    ctime = datetime.fromtimestamp(stat.st_ctime)
    scan_num = int(p.stem.split("_")[-1]) if "_" in p.stem else 0
    records.append({
        "participant_id": row["participant_id"],
        "blood_group": row["blood_group"],
        "scan_num": scan_num,
        "mtime": stat.st_mtime,
        "ctime": stat.st_ctime,
        "size": stat.st_size
    })

meta_df = pd.DataFrame(records)

# Check participant-level mtime and ctime
part_df = meta_df.groupby("participant_id").agg({
    "blood_group": "first",
    "mtime": "mean",
    "ctime": "mean",
    "size": "mean"
}).reset_index()

# Extract numeric participant ID
part_df["pid_num"] = part_df["participant_id"].apply(
    lambda x: int("".join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
)

print("\n--- 1. Participant ID Number vs Blood Group ---")
groups_pid = [part_df[part_df["blood_group"] == bg]["pid_num"].values for bg in ["A+", "AB+", "B+", "O+"]]
f_pid, p_pid = f_oneway(*groups_pid)
h_pid, p_kruskal_pid = kruskal(*groups_pid)
print(f"  ANOVA on PID number: F = {f_pid:.4f}, p = {p_pid:.4f}")
print(f"  Kruskal-Wallis on PID number: H = {h_pid:.4f}, p = {p_kruskal_pid:.4f}")

for bg in ["A+", "AB+", "B+", "O+"]:
    vals = part_df[part_df["blood_group"] == bg]["pid_num"].values
    print(f"  {bg:4} (N={len(vals):2}): Mean PID = {np.mean(vals):6.1f} +/- {np.std(vals):5.1f}, Range = [{np.min(vals):3}, {np.max(vals):3}]")

print("\n--- 2. File Modification Timestamp (mtime) vs Blood Group ---")
groups_mtime = [part_df[part_df["blood_group"] == bg]["mtime"].values for bg in ["A+", "AB+", "B+", "O+"]]
f_mtime, p_mtime = f_oneway(*groups_mtime)
h_mtime, p_kruskal_mtime = kruskal(*groups_mtime)
print(f"  ANOVA on mtime: F = {f_mtime:.4f}, p = {p_mtime:.4f}")
print(f"  Kruskal-Wallis on mtime: H = {h_mtime:.4f}, p = {p_kruskal_mtime:.4f}")

print("\n--- 3. Scan Order Distribution Across Blood Groups ---")
scans_per_bg = meta_df.groupby(["blood_group", "scan_num"]).size().unstack(fill_value=0)
print(scans_per_bg)
chi2_scan, p_scan, _, _ = chi2_contingency(scans_per_bg)
print(f"  Chi-Square on scan order: Chi2 = {chi2_scan:.4f}, p = {p_scan:.4f}")

print("\n--- 4. Check data/mfs100_final metadata logs or session records ---")
meta_dir = Path("data/mfs100_final/metadata")
if meta_dir.exists():
    for f in meta_dir.iterdir():
        print(f"  Found metadata file: {f.name} ({f.stat().st_size} bytes)")
