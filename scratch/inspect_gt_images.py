from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
GT_CSV = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"
df_gt = pd.read_csv(GT_CSV)

gt_stats = []
for fname in df_gt["capture_filename"]:
    fpath = PROJECT_ROOT / "data" / "captured" / fname
    with Image.open(fpath) as img:
        gt_stats.append({
            "size": img.size,
            "mode": img.mode,
            "mean": float(np.mean(np.array(img))),
            "std": float(np.std(np.array(img)))
        })

print("GT 26 cohort sample stats:")
print("Sizes:", set([s["size"] for s in gt_stats]))
print("Modes:", set([s["mode"] for s in gt_stats]))
print("Mean intensity:", np.mean([s["mean"] for s in gt_stats]))
print("Mean contrast:", np.mean([s["std"] for s in gt_stats]))
