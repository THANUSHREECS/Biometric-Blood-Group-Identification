import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. Record & verify checkpoint hashes
checkpoints = {
    "effnet_b0": PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras",
    "e2_fold1": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_1_best.weights.h5",
    "e2_fold2": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_2_best.weights.h5",
    "e2_fold3": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_3_best.weights.h5",
    "e2_fold4": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_4_best.weights.h5",
    "e2_fold5": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_5_best.weights.h5",
}

ckpt_hashes = {}
for name, p in checkpoints.items():
    ckpt_hashes[name] = {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": sha256_of_file(p)
    }

with open(OUT_DIR / "checkpoint_hashes.json", "w") as f:
    json.dump(ckpt_hashes, f, indent=2)

print("Checkpoint hashes recorded successfully.")

# 2. Inspect dataset_blood_group (Original ~6000 dataset)
orig_dir = PROJECT_ROOT / "dataset_blood_group"
orig_files = list(orig_dir.glob("**/*.*"))
orig_files = [f for f in orig_files if f.is_file() and f.suffix.lower() in [".bmp", ".jpg", ".jpeg", ".png"]]

print(f"Original dataset files count: {len(orig_files)}")
orig_classes = {}
orig_sizes = {}
orig_channels = {}
orig_hashes = set()
orig_duplicates = 0

for f in orig_files:
    # Determine class from parent or path
    parts = f.relative_to(orig_dir).parts
    cls_name = parts[0] if len(parts) > 0 else "unknown"
    orig_classes[cls_name] = orig_classes.get(cls_name, 0) + 1
    
    # Hash check
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in orig_hashes:
        orig_duplicates += 1
    else:
        orig_hashes.add(h)

# Sample 50 original images for size, channel, intensity stats
orig_sample = orig_files[:100]
orig_sample_stats = []
for f in orig_sample:
    try:
        with Image.open(f) as img:
            sz = img.size
            mode = img.mode
            arr = np.array(img)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            orig_sample_stats.append({
                "size": sz,
                "mode": mode,
                "mean": mean_val,
                "std": std_val
            })
    except Exception as e:
        pass

# 3. Inspect MFS100 101 dataset (data/mfs100_final)
mfs_dir = PROJECT_ROOT / "data" / "mfs100_final"
mfs_files = list(mfs_dir.glob("**/*.bmp"))
print(f"MFS100 final files count: {len(mfs_files)}")

mfs_hashes = set()
mfs_duplicates = 0
for f in mfs_files:
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in mfs_hashes:
        mfs_duplicates += 1
    else:
        mfs_hashes.add(h)

mfs_sample = mfs_files[:100]
mfs_sample_stats = []
for f in mfs_sample:
    try:
        with Image.open(f) as img:
            sz = img.size
            mode = img.mode
            arr = np.array(img)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            mfs_sample_stats.append({
                "size": sz,
                "mode": mode,
                "mean": mean_val,
                "std": std_val
            })
    except Exception as e:
        pass

# 4. Inspect data/captured (26 cohort)
cap_dir = PROJECT_ROOT / "data" / "captured"
cap_files = list(cap_dir.glob("*.bmp"))
print(f"Captured 26 cohort files count: {len(cap_files)}")

summary = {
    "original_dataset": {
        "total_images": len(orig_files),
        "unique_hashes": len(orig_hashes),
        "duplicates": orig_duplicates,
        "class_breakdown": orig_classes,
        "sample_size_modes": list(set([str(s["size"]) + "_" + s["mode"] for s in orig_sample_stats]))[:5],
        "mean_intensity": float(np.mean([s["mean"] for s in orig_sample_stats])),
        "mean_contrast_std": float(np.mean([s["std"] for s in orig_sample_stats]))
    },
    "mfs100_final": {
        "total_images": len(mfs_files),
        "unique_hashes": len(mfs_hashes),
        "duplicates": mfs_duplicates,
        "sample_size_modes": list(set([str(s["size"]) + "_" + s["mode"] for s in mfs_sample_stats]))[:5],
        "mean_intensity": float(np.mean([s["mean"] for s in mfs_sample_stats])),
        "mean_contrast_std": float(np.mean([s["std"] for s in mfs_sample_stats]))
    }
}

with open(OUT_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDataset summary JSON saved:")
print(json.dumps(summary, indent=2))
