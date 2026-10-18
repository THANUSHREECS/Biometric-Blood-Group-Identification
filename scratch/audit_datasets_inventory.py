import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import pandas as pd

DATA_DIR = Path("data")

print("=== DATA DIRECTORY INVENTORY ===")
for item in sorted(DATA_DIR.iterdir()):
    if item.is_dir():
        file_count = sum(1 for _ in item.rglob("*") if _.is_file())
        print(f"Dir: {item.name:25} | Files: {file_count}")
    else:
        print(f"File: {item.name:24} | Size: {item.stat().st_size:,} bytes")

def inspect_dataset(dir_path, name):
    print(f"\n--- Inspecting {name} ({dir_path}) ---")
    if not dir_path.exists():
        print("Path does not exist.")
        return
    
    files = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg"]]
    print(f"Total image files: {len(files)}")
    
    hashes = {}
    duplicates = []
    sizes = Counter()
    modes = Counter()
    formats = Counter()
    corrupted = []
    
    for f in files:
        try:
            data = f.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            if h in hashes:
                duplicates.append((f, hashes[h]))
            else:
                hashes[h] = f
            
            with Image.open(f) as img:
                sizes[img.size] += 1
                modes[img.mode] += 1
                formats[img.format] += 1
        except Exception as e:
            corrupted.append((f, str(e)))
            
    print(f"Unique SHA-256 hashes: {len(hashes)}")
    print(f"Duplicate image files: {len(duplicates)}")
    if duplicates:
        print(f"Sample duplicate: {duplicates[0][0].name} == {duplicates[0][1].name}")
    print(f"Image sizes: {dict(sizes.most_common(5))}")
    print(f"Image modes: {dict(modes.most_common(5))}")
    print(f"Image formats: {dict(formats.most_common(5))}")
    print(f"Corrupted images: {len(corrupted)}")
    return hashes, duplicates

h_final, dup_final = inspect_dataset(Path("data/mfs100_final"), "mfs100_final")
h_cap, dup_cap = inspect_dataset(Path("data/captured"), "captured")
if Path("data/raw").exists():
    inspect_dataset(Path("data/raw"), "raw")
if Path("data/processed").exists():
    inspect_dataset(Path("data/processed"), "processed")
if Path("data/quarantine").exists():
    inspect_dataset(Path("data/quarantine"), "quarantine")
