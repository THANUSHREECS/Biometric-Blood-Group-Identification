import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import json

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

orig_dir = PROJECT_ROOT / "dataset_blood_group"
mfs_dir = PROJECT_ROOT / "data" / "mfs100_final"
cap_dir = PROJECT_ROOT / "data" / "captured"

orig_files = list(orig_dir.glob("**/*.*"))
orig_files = [f for f in orig_files if f.is_file() and f.suffix.lower() in [".bmp", ".jpg", ".png"]][:100]

mfs_files = list(mfs_dir.glob("**/*.bmp"))[:100]

gt_csv = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"
df_gt = pd.read_csv(gt_csv)
cap_files = [cap_dir / f for f in df_gt["capture_filename"][:100]]

def analyze_image_set(file_list, name):
    widths, heights, aspect_ratios = [], [], []
    means, stds = [], []
    sharpnesses = [] # Laplacian variance
    fg_coverages = []
    
    for f in file_list:
        try:
            with Image.open(f) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
                
                # Convert to grayscale
                gray = np.array(img.convert("L"), dtype=np.float32)
                means.append(np.mean(gray))
                stds.append(np.std(gray))
                
                # Sharpness via Laplacian variance
                lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
                sharpnesses.append(lap.var())
                
                # Foreground coverage: fraction of pixels not near pure white (>245) or near pure black (<10)
                # In original dataset, background is white/transparent (255)
                # In MFS100, background is bright/white sensor plate border
                fg_mask = (gray > 15) & (gray < 240)
                fg_coverages.append(np.mean(fg_mask))
        except Exception as e:
            pass
            
    return {
        "name": name,
        "count": len(widths),
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "mean_intensity": float(np.mean(means)),
        "std_intensity": float(np.mean(stds)),
        "mean_sharpness_laplacian_var": float(np.mean(sharpnesses)),
        "mean_foreground_coverage_pct": float(np.mean(fg_coverages) * 100)
    }

stat_orig = analyze_image_set(orig_files, "Original Dataset (dataset_blood_group)")
stat_mfs101 = analyze_image_set(mfs_files, "MFS100 Final (101 Cohort)")
stat_cap26 = analyze_image_set(cap_files, "MFS100 Captured (26 Cohort)")

stats_comparison = {
    "original": stat_orig,
    "mfs100_final_101": stat_mfs101,
    "mfs100_captured_26": stat_cap26
}

with open(OUT_DIR / "domain_comparison_stats.json", "w") as f:
    json.dump(stats_comparison, f, indent=2)

print("Domain shift stats computed:")
print(json.dumps(stats_comparison, indent=2))
