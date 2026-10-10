import os
import glob
import json
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(42)

def md5(fname):
    hash_md5 = hashlib.md5()
    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def main():
    print("Starting STEP 1: Dataset audit...")
    dataset_dir = "dataset_blood_group"
    classes = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
    
    all_files = []
    class_counts = defaultdict(int)
    corrupted = []
    
    # Audit info
    dimensions = set()
    channels = set()
    
    # Hashing
    hash_to_files = defaultdict(list)
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            path = os.path.join(cls_dir, f)
            if os.path.isfile(path):
                class_counts[cls] += 1
                all_files.append((path, cls))
                
                # Check readability and format
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        w, h = img.size
                        mode = img.mode
                        dimensions.add((w, h))
                        channels.add(mode)
                except Exception as e:
                    corrupted.append(path)
                    continue
                    
                h = md5(path)
                if h:
                    hash_to_files[h].append((path, cls))

    print(f"Total images found: {len(all_files)}")
    for cls in classes:
        print(f"  {cls}: {class_counts[cls]}")
    print(f"Dimensions found: {dimensions}")
    print(f"Channels found: {channels}")
    print(f"Corrupted files: {len(corrupted)}")
    
    exact_duplicates_groups = [v for k, v in hash_to_files.items() if len(v) > 1]
    print(f"Exact duplicate groups: {len(exact_duplicates_groups)}")
    
    cross_class_duplicates = 0
    for grp in exact_duplicates_groups:
        cls_set = set([x[1] for x in grp])
        if len(cls_set) > 1:
            cross_class_duplicates += 1
    
    print(f"Cross-class duplicate groups: {cross_class_duplicates}")
    
    # STEP 2: SPLIT
    print("Starting STEP 2: Creating reproducible 70/15/15 splits...")
    
    # Group by hash to ensure exact duplicates stay together
    # We will stratify by class. For groups with multiple files, we just use the class of the first file (since they shouldn't be cross-class, or if they are, it's messy, but we assume they aren't mostly)
    
    class_to_groups = defaultdict(list)
    for h, files in hash_to_files.items():
        # files is a list of (path, cls)
        # Check if any file in group is corrupted, if so, we should exclude them? Let's just include non-corrupted.
        valid_files = [f for f in files if f[0] not in corrupted]
        if not valid_files:
            continue
        primary_cls = valid_files[0][1]
        class_to_groups[primary_cls].append(valid_files)
        
    train_split = []
    val_split = []
    test_split = []
    
    for cls in classes:
        groups = class_to_groups[cls]
        # Sort groups deterministically
        groups.sort(key=lambda g: g[0][0])
        np.random.shuffle(groups)
        
        n_groups = len(groups)
        n_train = int(n_groups * 0.7)
        n_val = int(n_groups * 0.15)
        
        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train+n_val]
        test_groups = groups[n_train+n_val:]
        
        for g in train_groups:
            train_split.extend(g)
        for g in val_groups:
            val_split.extend(g)
        for g in test_groups:
            test_split.extend(g)
            
    os.makedirs("reports", exist_ok=True)
    
    pd.DataFrame(train_split, columns=["image_path", "true_label"]).to_csv("reports/day27_full_dataset_split_train.csv", index=False)
    pd.DataFrame(val_split, columns=["image_path", "true_label"]).to_csv("reports/day27_full_dataset_split_val.csv", index=False)
    pd.DataFrame(test_split, columns=["image_path", "true_label"]).to_csv("reports/day27_full_dataset_split_test.csv", index=False)
    
    manifest = {
        "train_size": len(train_split),
        "val_size": len(val_split),
        "test_size": len(test_split),
        "seed": 42
    }
    with open("reports/day27_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Split sizes: Train {len(train_split)}, Val {len(val_split)}, Test {len(test_split)}")
    print("Splits saved to reports/day27_full_dataset_split_*.csv")

if __name__ == "__main__":
    main()
