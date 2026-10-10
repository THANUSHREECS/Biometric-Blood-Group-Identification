"""
experiments/mfs100_domain_adaptation/preprocessing_variants.py
================================================================
Isolated experimental preprocessing module for MFS100 Domain Adaptation.
DOES NOT modify production src/preprocessing.py.

Variants:
  - Variant A: Full sensor image (316x354 -> 224x224 Lanczos)
  - Variant B: Foreground ROI (8-12% padding, aspect-ratio square pad, 224x224)
  - Variant C: ROI + Conservative CLAHE (clipLimit=2.0, tileGrid=(8,8))
  - Variant D: ROI + Conservative Ridge Enhancement (percentile stretch + bilateral filter)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

IMAGE_SIZE = (224, 224)

def preprocess_full_image(pil_img):
    """
    Variant A: Full sensor image.
    Native 316x354 image converted to RGB and resized directly to 224x224 via Lanczos.
    """
    rgb = pil_img.convert("RGB")
    resized = rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return resized

def _detect_foreground_bbox(pil_img, pad_pct=0.10, intensity_thresh=238, min_dim=80):
    """
    Detects the active fingerprint impression on the 316x354 platen with 8-12% adaptive padding.
    Returns: (x1, y1, x2, y2), is_valid
    """
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    # Check for blank / low-variance
    if float(np.std(gray)) < 5.0 or h < min_dim or w < min_dim:
        return (0, 0, w, h), False

    rm = np.mean(gray, axis=1)
    cm = np.mean(gray, axis=0)
    ar = np.where(rm < intensity_thresh)[0]
    ac = np.where(cm < intensity_thresh)[0]

    if len(ar) == 0 or len(ac) == 0:
        # Fallback to Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            x_min, x_max = x, x + bw
            y_min, y_max = y, y + bh
        else:
            return (0, 0, w, h), False
    else:
        y_min, y_max = int(ar[0]), int(ar[-1])
        x_min, x_max = int(ac[0]), int(ac[-1])

    if (y_max - y_min) < min_dim or (x_max - x_min) < min_dim:
        return (0, 0, w, h), False

    pad_x = int(w * pad_pct)
    pad_y = int(h * pad_pct)

    x1 = max(0, x_min - pad_x)
    x2 = min(w, x_max + 1 + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(h, y_max + 1 + pad_y)

    return (x1, y1, x2, y2), True

def _pad_to_square(pil_img, bg_color=(255, 255, 255)):
    """
    Pads a PIL Image to square dimensions preserving aspect ratio.
    """
    w, h = pil_img.size
    max_side = max(w, h)
    if w == h:
        return pil_img
    padded = Image.new("RGB", (max_side, max_side), bg_color)
    offset = ((max_side - w) // 2, (max_side - h) // 2)
    padded.paste(pil_img.convert("RGB"), offset)
    return padded

def preprocess_roi(pil_img, pad_pct=0.10):
    """
    Variant B: Foreground ROI.
    Detects fingerprint bounding box, adds 10% padding, pads to square, resizes to 224x224.
    """
    bbox, is_valid = _detect_foreground_bbox(pil_img, pad_pct=pad_pct)
    crop_pil = pil_img.convert("RGB").crop(bbox)
    square_pil = _pad_to_square(crop_pil)
    resized = square_pil.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return resized

def preprocess_roi_clahe(pil_img, pad_pct=0.10, clip_limit=2.0):
    """
    Variant C: ROI + Conservative CLAHE.
    Applies ROI crop + square pad, then conservative CLAHE local contrast equalization.
    """
    bbox, is_valid = _detect_foreground_bbox(pil_img, pad_pct=pad_pct)
    crop_pil = pil_img.convert("RGB").crop(bbox)
    square_pil = _pad_to_square(crop_pil)

    gray = np.array(square_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    enhanced_rgb = Image.fromarray(cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB))

    resized = enhanced_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return resized

def preprocess_roi_ridge_enhancement(pil_img, pad_pct=0.10):
    """
    Variant D: ROI + Conservative Ridge Enhancement.
    Applies ROI crop + square pad, then conservative percentile stretch + bilateral noise suppression.
    Avoids aggressive sharpening to prevent artifact generation.
    """
    bbox, is_valid = _detect_foreground_bbox(pil_img, pad_pct=pad_pct)
    crop_pil = pil_img.convert("RGB").crop(bbox)
    square_pil = _pad_to_square(crop_pil)

    gray = np.array(square_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    p_low = float(np.percentile(cl, 1.0))
    p_high = float(np.percentile(cl, 99.0))
    if p_high > p_low:
        stretched = np.clip((cl - p_low) / (p_high - p_low) * 230.0 + 15.0, 0, 255).astype(np.uint8)
    else:
        stretched = cl

    # Mild bilateral filter to preserve ridge edges while smoothing CMOS sensor noise
    denoised = cv2.bilateralFilter(stretched, d=3, sigmaColor=20, sigmaSpace=20)
    enhanced_rgb = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))

    resized = enhanced_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return resized

def generate_visual_comparison_chart(sample_image_path, out_path):
    """
    Generates preprocessing_comparison.png showing:
    1. Original 316x354 image
    2. Full-image resized (Variant A)
    3. Detected ROI bounding box on original
    4. ROI resized (Variant B)
    5. ROI + CLAHE (Variant C)
    6. ROI + Ridge enhancement (Variant D)
    """
    with Image.open(sample_image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

        # 1. Original
        orig = img_rgb

        # 2. Full resized
        full_resized = preprocess_full_image(img)

        # 3. Detected ROI bounding box
        bbox, _ = _detect_foreground_bbox(img, pad_pct=0.10)
        bbox_img = img_rgb.copy()
        draw = ImageDraw.Draw(bbox_img)
        draw.rectangle(bbox, outline="red", width=3)

        # 4. ROI resized
        roi_resized = preprocess_roi(img)

        # 5. ROI + CLAHE
        roi_clahe = preprocess_roi_clahe(img)

        # 6. ROI + Ridge enhancement
        roi_ridge = preprocess_roi_ridge_enhancement(img)

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))

    axes[0].imshow(orig)
    axes[0].set_title(f"1. Native MFS100\n({w}x{h})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(full_resized)
    axes[1].set_title("2. Full Resized\n(224x224, Variant A)", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(bbox_img)
    axes[2].set_title(f"3. Detected ROI BBox\n[{bbox[0]},{bbox[1]} to {bbox[2]},{bbox[3]}]", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(roi_resized)
    axes[3].set_title("4. ROI Resized\n(224x224, Variant B)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(roi_clahe)
    axes[4].set_title("5. ROI + CLAHE\n(224x224, Variant C)", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    axes[5].imshow(roi_ridge)
    axes[5].set_title("6. ROI + Ridge Enh.\n(224x224, Variant D)", fontsize=10, fontweight="bold")
    axes[5].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved visual preprocessing comparison: {out_path}")
