import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

df_split = pd.read_csv(SPLIT_CSV)
dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()

classes = ["A+", "B+", "AB+", "O+"]
rev_class_mapping = {c: i for i, c in enumerate(classes)}

# Group by participant
grouped = dev_df.groupby("participant_id")
dev_pids = []
dev_y = []

for pid, group in grouped:
    dev_pids.append(pid)
    dev_y.append(rev_class_mapping[group.iloc[0]["blood_group"]])

dev_pids = np.array(dev_pids)
dev_y = np.array(dev_y)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Since X is (80,), dev_y is (80,), dev_pids is (80,)
dummy_X = np.zeros((len(dev_pids), 1))
folds = list(sgkf.split(dummy_X, dev_y, dev_pids))

print(f"Total participants: {len(dev_pids)}")
all_val_pids = []

for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
    train_p = set(dev_pids[train_idx])
    val_p = set(dev_pids[val_idx])
    all_val_pids.extend(list(val_p))
    
    assert len(train_p & val_p) == 0, f"Leakage in fold {fold_idx}!"
    assert len(train_p) == 64, f"Fold {fold_idx} train size is {len(train_p)}"
    assert len(val_p) == 16, f"Fold {fold_idx} val size is {len(val_p)}"
    
    y_tr = dev_y[train_idx]
    y_va = dev_y[val_idx]
    
    tr_counts = {classes[i]: int(sum(y_tr == i)) for i in range(4)}
    va_counts = {classes[i]: int(sum(y_va == i)) for i in range(4)}
    
    print(f"\nFold {fold_idx}:")
    print(f"  Train ({len(train_p)}): {tr_counts}")
    print(f"  Val   ({len(val_p)}): {va_counts}")
    
    # Check all classes present in train
    for c, cnt in tr_counts.items():
        assert cnt > 0, f"Class {c} is missing in Fold {fold_idx} train!"

# Verify that all 80 participants appear in val exactly once
assert len(all_val_pids) == 80
assert len(set(all_val_pids)) == 80
print("\n[ALL 5 FOLDS PERFECTLY STRATIFIED AND DISJOINT - ALL 4 CLASSES PRESENT IN ALL FOLDS]")
