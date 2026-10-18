import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5

SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_CLASSES = ["A+", "B+", "AB+", "O+"]
NATIVE_REV_MAP = {c: i for i, c in enumerate(NATIVE_CLASSES)}

print("="*70)
print("CACHING FROZEN DENSENET-121 FEATURES FOR 80 DEV PARTICIPANTS")
print("="*70)

# Load development dataset (80 participants, 240 scans)
df_split = pd.read_csv(SPLIT_CSV)
dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
assert dev_df["participant_id"].nunique() == 80
assert len(dev_df) == 240

# Load EXP-06C fold assignments from historical OOF predictions to ensure 100% exact match
oof_hist = pd.read_csv(PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06c" / "config_a_oof_predictions.csv")
pid_to_fold = dict(zip(oof_hist["participant_id"], oof_hist["fold_id"]))

pids_unique = sorted(dev_df["participant_id"].unique())
dev_y_list = []
dev_labels_list = []
dev_folds_list = []
fused_features_list = []

# Build frozen DenseNet121 GAP model
print("Instantiating frozen DenseNet121 backbone...")
base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(112, 112, 3))
base.trainable = False
pool = tf.keras.layers.GlobalAveragePooling2D()

# Preprocess and extract features
print(f"Extracting features for {len(pids_unique)} participants (3 scans each)...")
t0 = time.time()
for idx, pid in enumerate(pids_unique):
    p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
    assert len(p_scans) == 3
    bg = p_scans.iloc[0]["blood_group"]
    dev_y_list.append(NATIVE_REV_MAP[bg])
    dev_labels_list.append(bg)
    dev_folds_list.append(pid_to_fold[pid])
    
    scans_batch = []
    for _, row in p_scans.iterrows():
        img_p = Path(row["scan_path"])
        arr, _, _ = transform_c5(str(img_p))
        scans_batch.append(arr)
    
    batch_arr = np.array(scans_batch, dtype=np.float32) # (3, 112, 112, 3)
    feats = pool(base(batch_arr)).numpy() # (3, 1024)
    fused = feats.flatten() # (3072,)
    fused_features_list.append(fused)
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(pids_unique)} participants ({time.time() - t0:.1f}s)")

features = np.array(fused_features_list, dtype=np.float32)
labels = np.array(dev_y_list, dtype=np.int32)
pids = np.array(pids_unique)
folds = np.array(dev_folds_list, dtype=np.int32)

print(f"\nExtraction complete in {time.time() - t0:.2f}s!")
print(f"Features shape: {features.shape}, dtype={features.dtype}")
print(f"Labels shape: {labels.shape}")
print(f"Folds distribution:\n{pd.Series(folds).value_counts().sort_index()}")

# Save to disk
cache_file = OUT_DIR / "dev_densenet121_features.npz"
np.savez_compressed(
    cache_file,
    features=features,
    labels=labels,
    pids=pids,
    folds=folds,
    classes=NATIVE_CLASSES
)
print(f"Saved cached features to {cache_file} ({os.path.getsize(cache_file) / 1024:.1f} KB)")
