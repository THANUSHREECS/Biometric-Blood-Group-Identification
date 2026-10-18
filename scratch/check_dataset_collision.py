import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
GT_CSV = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"
SOURCE_DIR = PROJECT_ROOT / "dataset_blood_group"

df_gt = pd.read_csv(GT_CSV)
gt_hashes = set()
for fname in df_gt["capture_filename"]:
    fpath = PROJECT_ROOT / "data" / "captured" / fname
    if fpath.is_file():
        gt_hashes.add(hashlib.sha256(fpath.read_bytes()).hexdigest())

source_files = list(SOURCE_DIR.glob("**/*.*"))
print(f"Checking {len(source_files)} files in dataset_blood_group against 78 GT image hashes...")
collisions = 0
for sf in source_files:
    if sf.is_file():
        h = hashlib.sha256(sf.read_bytes()).hexdigest()
        if h in gt_hashes:
            collisions += 1

print(f"Collisions with original dataset_blood_group: {collisions}")
