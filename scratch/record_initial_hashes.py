import hashlib
import json
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "reports" / "efficientnet_e2_ensemble_experiment"
EXP_DIR.mkdir(parents=True, exist_ok=True)

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

files_to_hash = {
    "effnet_checkpoint": PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras",
    "e2_fold1_weights": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_1_best.weights.h5",
    "e2_fold2_weights": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_2_best.weights.h5",
    "e2_fold3_weights": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_3_best.weights.h5",
    "e2_fold4_weights": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_4_best.weights.h5",
    "e2_fold5_weights": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / "fold_5_best.weights.h5",
    "participant_split_csv": PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv",
    "mfs100_ground_truth_csv": PROJECT_ROOT / "data" / "mfs100_ground_truth.csv",
    "model_fusion_npz": PROJECT_ROOT / "reports" / "model_fusion" / "all_models_participant_probs.npz",
}

hashes = {}
for name, p in files_to_hash.items():
    if p.is_file():
        hashes[name] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_of_file(p)
        }
    else:
        hashes[name] = {"path": str(p), "status": "MISSING"}

with open(EXP_DIR / "checkpoint_hashes_before_after.json", "w") as f:
    json.dump({"before_experiment": hashes}, f, indent=2)

print("Recorded before-experiment SHA256 hashes successfully:")
for k, v in hashes.items():
    print(f"  {k}: {v.get('sha256', 'MISSING')[:16]}... ({v.get('size_bytes', 0)} bytes)")
