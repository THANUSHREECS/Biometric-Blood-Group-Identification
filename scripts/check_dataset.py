#!/usr/bin/env python3
"""
scripts/check_dataset.py
Validation script to check the integrity, formats, dimensions, and duplicates
in the fingerprint blood group dataset.
"""

import os
import sys
import hashlib
from collections import Counter, defaultdict
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import CLASS_NAMES, CLASS_TO_ID, RAW_DATA_DIR, SOURCE_DATA_DIR

def get_dataset_path():
    if os.path.exists(RAW_DATA_DIR) and len(os.listdir(RAW_DATA_DIR)) > 0:
        return RAW_DATA_DIR
    elif os.path.exists(SOURCE_DATA_DIR):
        return SOURCE_DATA_DIR
    else:
        raise FileNotFoundError(f"Neither {RAW_DATA_DIR} nor {SOURCE_DATA_DIR} was found!")

def check_dataset(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = get_dataset_path()

    print("=" * 60)
    print(f"CHECKING DATASET: {dataset_dir}")
    print("=" * 60)

    # 1. Check directory structure and class presence
    found_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Found {len(found_dirs)} class directories: {found_dirs}")

    missing_classes = [c for c in CLASS_NAMES if c not in found_dirs]
    unexpected_classes = [d for d in found_dirs if d not in CLASS_NAMES]

    if missing_classes:
        print(f"[-] ERROR: Missing target classes: {missing_classes}")
    if unexpected_classes:
        print(f"[-] WARNING: Unexpected subdirectories: {unexpected_classes}")

    # 2. Inspect every image file
    total_files = 0
    class_counts = Counter()
    format_counts = Counter()
    mode_counts = Counter()
    size_counts = Counter()
    corrupted_files = []
    zero_byte_files = []
    
    sha_map = defaultdict(list)
    sha_to_class = {}
    cross_class_dups = []

    for cls in CLASS_NAMES:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        files = sorted(os.listdir(cls_path))
        class_counts[cls] = len(files)
        total_files += len(files)

        for fname in files:
            fpath = os.path.join(cls_path, fname)
            
            # Check empty file
            size = os.path.getsize(fpath)
            if size == 0:
                zero_byte_files.append(fpath)
                corrupted_files.append((fpath, "Zero-byte file"))
                continue

            # Compute SHA-256
            hasher = hashlib.sha256()
            try:
                with open(fpath, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
                sha_map[file_hash].append(fpath)

                if file_hash in sha_to_class:
                    if sha_to_class[file_hash] != cls:
                        cross_class_dups.append((sha_to_class[file_hash], cls, fpath))
                else:
                    sha_to_class[file_hash] = cls
            except Exception as e:
                corrupted_files.append((fpath, f"Read error: {e}"))
                continue

            # PIL validation
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    format_counts[str(img.format)] += 1
                    mode_counts[str(img.mode)] += 1
                    size_counts[img.size] += 1
            except Exception as e:
                corrupted_files.append((fpath, f"Image error: {e}"))

    # Summary
    dup_clusters = {h: paths for h, paths in sha_map.items() if len(paths) > 1}
    redundant_dup_count = sum(len(paths) - 1 for paths in dup_clusters.values())

    print("\n--- DATASET INTEGRITY REPORT ---")
    print(f"Total images scanned      : {total_files}")
    print(f"Corrupted / Zero-byte files: {len(corrupted_files)}")
    print(f"Exact duplicate clusters  : {len(dup_clusters)}")
    print(f"Redundant duplicate files : {redundant_dup_count}")
    print(f"Cross-class duplicates    : {len(cross_class_dups)}")

    print("\nClass Distribution:")
    for cls in CLASS_NAMES:
        cnt = class_counts[cls]
        pct = (cnt / total_files * 100) if total_files > 0 else 0
        print(f"  {cls:5s} (ID {CLASS_TO_ID[cls]}): {cnt:5d} ({pct:5.2f}%)")

    print("\nFormat Distribution:")
    for fmt, cnt in format_counts.items():
        print(f"  {fmt}: {cnt}")

    print("\nMode Distribution:")
    for mode, cnt in mode_counts.items():
        print(f"  {mode}: {cnt}")

    print("\nDimension Breakdown:")
    for sz, cnt in size_counts.items():
        print(f"  {sz}: {cnt}")

    is_valid = (len(corrupted_files) == 0) and (len(missing_classes) == 0) and (len(cross_class_dups) == 0)
    print("\n" + "=" * 60)
    if is_valid:
        print("[+] CHECK RESULT: PASS - Dataset is structurally sound and verified.")
    else:
        print("[-] CHECK RESULT: FAIL - Issues detected in dataset.")
    print("=" * 60)

    return {
        "is_valid": is_valid,
        "total_files": total_files,
        "class_counts": dict(class_counts),
        "corrupted_count": len(corrupted_files),
        "duplicate_clusters": len(dup_clusters),
        "redundant_duplicates": redundant_dup_count,
        "cross_class_duplicates": len(cross_class_dups)
    }

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    res = check_dataset(target)
    sys.exit(0 if res["is_valid"] else 1)
