"""
scripts/extract_original_dataset_raw_effb0_features.py
Transcription of the inline command that originally produced:
  experiments/leakage_audit/original_dataset_raw_effb0_features.npy

Purpose:
Extracts un-normalized raw Global Average Pooling (GAP) features (1,280-d) from
frozen ImageNet EfficientNet-B0 for all 2,769 positive-class images in the original
dataset (outputs/dataset_manifest.csv), preserving exact manifest row order.

Input:
- outputs/dataset_manifest.csv
- data/processed/ images

Output:
- experiments/leakage_audit/original_dataset_raw_effb0_features.npy (2,769 x 1,280, float32)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
EXP_DIR.mkdir(parents=True, exist_ok=True)

manifest_file = PROJECT_ROOT / "outputs" / "dataset_manifest.csv"
assert manifest_file.exists(), f"Missing manifest: {manifest_file}"

manifest = pd.read_csv(manifest_file)
pos = manifest[manifest["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy().reset_index(drop=True)
n_images = len(pos)
print(f"Extracting raw GAP EfficientNet-B0 features for {n_images} images...")

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(128, 128, 3)
)
base.trainable = False
gap = tf.keras.layers.GlobalAveragePooling2D()

batch_size = 64
n_batches = int(np.ceil(n_images / batch_size))
feats = []
t0 = time.time()

for b in range(n_batches):
    b_df = pos.iloc[b * batch_size : (b + 1) * batch_size]
    b_imgs = []
    for _, row in b_df.iterrows():
        p_str = str(row["image_path"])
        # Resolve path
        if not Path(p_str).exists():
            idx = p_str.find("data/processed")
            if idx != -1:
                p_str = str(PROJECT_ROOT / p_str[idx:])
        raw = tf.io.read_file(p_str)
        img = tf.image.decode_bmp(raw)
        if tf.shape(img)[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, [128, 128])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        b_imgs.append(img)
    batch_t = tf.stack(b_imgs)
    out = gap(base(batch_t)).numpy()  # RAW GAP, NO L2 NORMALIZATION
    feats.append(out)

raw_gap = np.vstack(feats).astype(np.float32)
elapsed = time.time() - t0
print(f"Extraction completed in {elapsed:.1f}s. Shape: {raw_gap.shape}")
norms = np.linalg.norm(raw_gap, axis=1)
print(f"Raw GAP Norms min={norms.min():.4f}, mean={norms.mean():.4f}, max={norms.max():.4f}")

import sys

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP_DIR / "original_dataset_raw_effb0_features.npy"
# Verify before overwrite or save
np.save(out_path, raw_gap)
print(f"Saved features to {out_path}")
