#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
Production dataset preparation script for Biometric Blood Group Identification:
1. Preserves raw dataset in data/raw/
2. Validates image integrity and handles RGBA alpha compositing on white background
3. Detects exact duplicates and quarantines redundant copies to data/quarantine/
4. Prepares cleaned images in data/processed/
5. Extracts sample images for each class to data/sample/
6. Generates stratified train (70%) / validation (15%) / test (15%) splits
7. Outputs dataset_manifest.csv and dataset_report.json
"""

import os
import sys
import shutil
import hashlib
import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    QUARANTINE_DATA_DIR,
    SAMPLE_IMAGE_DIR,
    SOURCE_DATA_DIR,
    OUTPUTS_DIR,
    DATASET_MANIFEST_PATH,
    DATASET_REPORT_PATH,
    BACKGROUND_COLOR
)

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def ensure_raw_data_preserved():
    """Ensures data/raw/ contains the untouched original dataset."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    raw_has_files = any(os.path.isdir(os.path.join(RAW_DATA_DIR, c)) and len(os.listdir(os.path.join(RAW_DATA_DIR, c))) > 0 for c in CLASS_NAMES if os.path.exists(os.path.join(RAW_DATA_DIR, c)))

    if raw_has_files:
        print(f"[+] Raw data directory already populated at: {RAW_DATA_DIR}")
        return RAW_DATA_DIR

    if not os.path.exists(SOURCE_DATA_DIR):
        raise FileNotFoundError(f"Source dataset not found at {SOURCE_DATA_DIR}")

    print(f"[*] Copying untouched raw dataset from {SOURCE_DATA_DIR} to {RAW_DATA_DIR}...")
    for cls in CLASS_NAMES:
        src_cls_dir = os.path.join(SOURCE_DATA_DIR, cls)
        dst_cls_dir = os.path.join(RAW_DATA_DIR, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        if os.path.isdir(src_cls_dir):
            for fname in os.listdir(src_cls_dir):
                src_file = os.path.join(src_cls_dir, fname)
                dst_file = os.path.join(dst_cls_dir, fname)
                shutil.copy2(src_file, dst_file)

    print(f"[+] Raw dataset preserved safely in {RAW_DATA_DIR}")
    return RAW_DATA_DIR

def clean_and_composite_image(img):
    """
    Composites any alpha transparency onto a clean white background (255, 255, 255)
    and converts to 3-channel RGB.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (*BACKGROUND_COLOR, 255))
        composite = Image.alpha_composite(background, alpha_img)
        return composite.convert("RGB")
    else:
        return img.convert("RGB")

def prepare_dataset():
    print("=" * 70)
    print("STEP 2: DATASET CLEANING, QUARANTINE & STRATIFIED PREPARATION")
    print("=" * 70)

    # 1. Ensure directories exist
    raw_dir = ensure_raw_data_preserved()
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(QUARANTINE_DATA_DIR, exist_ok=True)
    os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    quarantine_dup_dir = os.path.join(QUARANTINE_DATA_DIR, "duplicates")
    os.makedirs(quarantine_dup_dir, exist_ok=True)

    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, cls), exist_ok=True)

    # 2. Scan and audit all raw files
    print("\n[*] Auditing raw images, checking integrity and exact duplicates...")
    records = []
    sha_map = defaultdict(list)
    corrupted_records = []
    format_counts = Counter()
    size_counts = Counter()

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for fname in sorted(os.listdir(cls_dir)):
            raw_path = os.path.join(cls_dir, fname)
            if not os.path.isfile(raw_path):
                continue

            # Check file size
            fsize = os.path.getsize(raw_path)
            if fsize == 0:
                corrupted_records.append((raw_path, "Zero-byte file"))
                continue

            # Compute hash
            file_hash = compute_sha256(raw_path)

            # Validate with PIL
            try:
                with Image.open(raw_path) as img:
                    img.verify()
                with Image.open(raw_path) as img:
                    fmt = str(img.format)
                    mode = str(img.mode)
                    dims = img.size
                    format_counts[fmt] += 1
                    size_counts[str(dims)] += 1
            except Exception as e:
                corrupted_records.append((raw_path, str(e)))
                continue

            item = {
                "raw_path": raw_path,
                "filename": fname,
                "class_name": cls,
                "class_id": CLASS_TO_ID[cls],
                "sha256": file_hash,
                "original_format": fmt,
                "original_mode": mode,
                "original_size": dims,
                "file_bytes": fsize
            }
            records.append(item)
            sha_map[file_hash].append(item)

    total_raw_found = len(records) + len(corrupted_records)
    print(f"[+] Total raw files scanned: {total_raw_found}")
    print(f"[+] Valid image records: {len(records)}")
    print(f"[+] Corrupted files: {len(corrupted_records)}")

    # 3. Handle Duplicates and Quarantine Redundant Copies
    print("\n[*] Grouping exact duplicate clusters and quarantining redundant entries...")
    canonical_items = []
    quarantined_items = []
    duplicate_groups = {}

    for file_hash, group in sha_map.items():
        if len(group) == 1:
            canonical_items.append(group[0])
        else:
            # First occurrence is canonical
            canonical = group[0]
            canonical_items.append(canonical)
            duplicate_groups[file_hash] = {
                "canonical": canonical["raw_path"],
                "duplicates": [dup["raw_path"] for dup in group[1:]]
            }
            for dup in group[1:]:
                # Quarantine redundant file
                q_dest = os.path.join(quarantine_dup_dir, f"{dup['class_name']}_{dup['filename']}")
                shutil.copy2(dup["raw_path"], q_dest)
                dup["quarantine_path"] = q_dest
                quarantined_items.append(dup)

    print(f"[+] Canonical unique images retained: {len(canonical_items)}")
    print(f"[+] Redundant duplicate files quarantined: {len(quarantined_items)}")

    # Save quarantine manifest
    with open(os.path.join(QUARANTINE_DATA_DIR, "quarantine_manifest.json"), "w") as f:
        json.dump({
            "duplicate_groups_count": len(duplicate_groups),
            "total_quarantined": len(quarantined_items),
            "duplicate_groups": duplicate_groups
        }, f, indent=2)

    # 4. Clean and process canonical images into data/processed/
    print("\n[*] Processing images (alpha compositing on white background & RGB conversion)...")
    processed_records = []
    sample_extracted = set()

    for item in canonical_items:
        raw_p = item["raw_path"]
        cls = item["class_name"]
        fname = item["filename"]
        
        # Open and composite
        with Image.open(raw_p) as img:
            cleaned_img = clean_and_composite_image(img)
            
            # Save processed image as BMP
            processed_path = os.path.join(PROCESSED_DATA_DIR, cls, fname)
            cleaned_img.save(processed_path, format="BMP")

            # Extract 1 sample per class if not already done
            if cls not in sample_extracted:
                sample_path = os.path.join(SAMPLE_IMAGE_DIR, f"{cls}.bmp")
                cleaned_img.save(sample_path, format="BMP")
                sample_extracted.add(cls)

        item_processed = dict(item)
        item_processed["image_path"] = processed_path
        processed_records.append(item_processed)

    print(f"[+] Successfully processed and saved {len(processed_records)} clean images in {PROCESSED_DATA_DIR}")
    print(f"[+] Extracted authentic sample images for {len(sample_extracted)} classes in {SAMPLE_IMAGE_DIR}")

    # 5. Stratified Train / Validation / Test Splitting
    print("\n[*] Creating stratified splits: 70% Train, 15% Validation, 15% Test...")
    df = pd.DataFrame(processed_records)

    # Step 1: Split train (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["class_id"]
    )

    # Step 2: Split temp (30%) equally into val (15%) and test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["class_id"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Verify no overlap between splits
    train_set = set(train_df["sha256"])
    val_set = set(val_df["sha256"])
    test_set = set(test_df["sha256"])

    leakage_train_val = train_set.intersection(val_set)
    leakage_train_test = train_set.intersection(test_set)
    leakage_val_test = val_set.intersection(test_set)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError("CRITICAL: Data leakage detected between dataset splits!")
    print("[+] Verified: 0 cross-split leakage (all split hash intersections are empty).")

    # 6. Save Manifest CSV
    # Required columns: image_path,class_name,class_id,split
    out_manifest_df = manifest_df[["image_path", "class_name", "class_id", "split"]].copy()
    # Normalize paths to forward slashes for cross-platform consistency
    out_manifest_df["image_path"] = out_manifest_df["image_path"].apply(lambda p: p.replace("\\", "/"))
    out_manifest_df.to_csv(DATASET_MANIFEST_PATH, index=False)
    print(f"[+] Saved dataset manifest to: {DATASET_MANIFEST_PATH}")

    # 7. Generate Quality Report JSON
    class_dist = {cls: int((manifest_df["class_name"] == cls).sum()) for cls in CLASS_NAMES}
    train_dist = {cls: int((train_df["class_name"] == cls).sum()) for cls in CLASS_NAMES}
    val_dist = {cls: int((val_df["class_name"] == cls).sum()) for cls in CLASS_NAMES}
    test_dist = {cls: int((test_df["class_name"] == cls).sum()) for cls in CLASS_NAMES}

    report = {
        "original_total": total_raw_found,
        "clean_total": len(manifest_df),
        "removed_total": len(quarantined_items) + len(corrupted_records),
        "corrupted_count": len(corrupted_records),
        "duplicate_count": len(quarantined_items),
        "duplicate_clusters": len(duplicate_groups),
        "train_count": len(train_df),
        "validation_count": len(val_df),
        "test_count": len(test_df),
        "class_distribution": class_dist,
        "split_distribution": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist
        },
        "image_format_distribution": dict(format_counts),
        "image_size_statistics": dict(size_counts),
        "preprocessing_summary": {
            "alpha_compositing": "Composited on (255, 255, 255) white background",
            "channels": "3-channel RGB",
            "target_model_input_size": [224, 224, 3]
        }
    }

    with open(DATASET_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved dataset report to: {DATASET_REPORT_PATH}")

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Class':<8} | {'Class ID':<8} | {'Total Clean':<11} | {'Train (70%)':<11} | {'Val (15%)':<10} | {'Test (15%)':<10}")
    print("-" * 70)
    for cls in CLASS_NAMES:
        cid = CLASS_TO_ID[cls]
        tot = class_dist[cls]
        tr = train_dist[cls]
        va = val_dist[cls]
        te = test_dist[cls]
        print(f"{cls:<8} | {cid:<8} | {tot:<11} | {tr:<11} | {va:<10} | {te:<10}")
    print("-" * 70)
    print(f"{'TOTAL':<8} | {'-':<8} | {len(manifest_df):<11} | {len(train_df):<11} | {len(val_df):<10} | {len(test_df):<10}")
    print("=" * 70)
    print("[+] STEP 2 DATASET PREPARATION COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    prepare_dataset()
