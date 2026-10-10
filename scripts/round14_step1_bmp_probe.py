"""
round14_step1_bmp_probe.py
Round 14, Step 1: Report mode, size, and alpha channel of BMPs in data/mfs100_final/raw.
Print code before output as required.
"""
import os
import sys
import numpy as np
from PIL import Image

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
raw_dir = os.path.join(project_root, "data", "mfs100_final", "raw")

bmp_files = []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if f.lower().endswith(".bmp"):
            bmp_files.append(os.path.join(root, f))

bmp_files = sorted(bmp_files)
print(f"Total BMP files found in {raw_dir}: {len(bmp_files)}")

# Sample first 10, last 5, and a midpoint for reporting
sample_indices = list(range(min(10, len(bmp_files)))) + [len(bmp_files)//2] + list(range(max(0, len(bmp_files)-5), len(bmp_files)))
sample_indices = sorted(set(sample_indices))

print("\n--- Sample BMP properties ---")
print(f"{'Index':>6} {'Mode':<8} {'Size':<14} {'Has Alpha':<10} {'Min':>6} {'Max':>6} {'Mean':>8}")
print("-" * 65)

modes = {}
sizes = {}
has_alpha_count = 0

for idx, bmp_path in enumerate(bmp_files):
    img = Image.open(bmp_path)
    mode = img.mode
    size = img.size
    has_alpha = mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info)

    modes[mode] = modes.get(mode, 0) + 1
    sizes[size] = sizes.get(size, 0) + 1
    if has_alpha:
        has_alpha_count += 1

    if idx in sample_indices:
        arr = np.array(img)
        print(f"{idx:>6} {mode:<8} {str(size):<14} {str(has_alpha):<10} {arr.min():>6} {arr.max():>6} {arr.mean():>8.2f}  {os.path.basename(bmp_path)}")

print("\n--- Summary ---")
print(f"Total BMPs:   {len(bmp_files)}")
print(f"Mode counts:  {modes}")
print(f"Size counts:  {sizes}")
print(f"Has alpha:    {has_alpha_count} / {len(bmp_files)}")
