"""
src/preprocessing.py
Centralized, single-source-of-truth fingerprint preprocessing pipeline for:
- Model training & data generators
- Offline evaluation
- File upload prediction
- Physical / Mock Mantra MFS100 scanner capture

Baseline Pipeline:
1. Load image (from filepath, byte stream, PIL Image, or numpy array)
2. Handle alpha transparency: composite onto clean white background (255, 255, 255)
3. Convert to 3-channel RGB
4. High-quality resize to 224 x 224
5. Optional enhancement (disabled by default for baseline model)
6. ResNet50 preprocess_input (Caffe-style: BGR, zero-centered with ImageNet mean)
"""

import io
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf

from src.config import IMAGE_SIZE, BACKGROUND_COLOR

def load_image_to_pil(image_input):
    """
    Safely loads various image input formats into a PIL Image.
    Supports file path (str/Path), bytes, io.BytesIO, PIL.Image, or numpy array.
    """
    if isinstance(image_input, Image.Image):
        return image_input.copy()

    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        return Image.open(image_input)

    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input))

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        return Image.open(image_input)

    if isinstance(image_input, np.ndarray):
        # If float in [0, 1], scale to [0, 255]
        arr = image_input
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")

def composite_alpha_on_white(pil_img):
    """
    Composites any alpha transparency onto a solid white background (255, 255, 255)
    and converts to 3-channel RGB.
    Prevents artificial black borders caused by naive alpha dropping.
    """
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        alpha_img = pil_img.convert("RGBA")
        white_bg = Image.new("RGBA", alpha_img.size, (*BACKGROUND_COLOR, 255))
        composite = Image.alpha_composite(white_bg, alpha_img)
        return composite.convert("RGB")
    else:
        return pil_img.convert("RGB")

def resize_to_target(pil_img, target_size=IMAGE_SIZE):
    """
    Resizes image to target dimensions (224, 224) using high-quality Lanczos interpolation.
    """
    if pil_img.size == target_size:
        return pil_img
    return pil_img.resize(target_size, resample=Image.Resampling.LANCZOS)

def apply_fingerprint_enhancement(rgb_array, enabled=False):
    """
    Optional fingerprint contrast enhancement (CLAHE).
    By default disabled (enabled=False) to strictly adhere to the approved baseline model.
    """
    if not enabled:
        return rgb_array

    try:
        import cv2
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    except Exception:
        return rgb_array

def crop_mfs100_foreground(pil_img, pad_pct=0.06, min_dim=80, intensity_thresh=238):
    """
    Hardware-Domain Foreground Cropping for Mantra MFS100 optical scanner captures.
    Detects the active fingerprint ridge impression on the 316x354 sensor platen and
    crops away wide empty sensor glass margins before 224x224 resizing.

    Safeguards:
    1. Low-variance / blank platen guard: Returns original image unchanged if std < 5.
    2. Dimension floor: Prevents crops smaller than min_dim (80 px).
    3. Padding margin: Adds ~6% margin around detected bounds to prevent minutiae clipping.
    4. Margin presence check: Only crops if substantial empty border (> 5%) exists.

    Returns:
        cropped_pil: PIL.Image (cropped or original fallback)
        metadata: dict with execution details and bounding box
    """
    orig_size = pil_img.size
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape

    # Safeguard 1: Blank or near-uniform frame (e.g. no finger placed)
    if h < min_dim or w < min_dim or float(np.std(gray)) < 5.0:
        return pil_img, {
            "crop_applied": False,
            "reason": "low_variance_or_blank",
            "bbox": None,
            "orig_size": orig_size,
            "cropped_size": orig_size
        }

    rm = np.mean(gray, axis=1)
    cm = np.mean(gray, axis=0)

    ar = np.where(rm < intensity_thresh)[0]
    ac = np.where(cm < intensity_thresh)[0]

    # Safeguard 2: No active regions found
    if len(ar) == 0 or len(ac) == 0:
        return pil_img, {
            "crop_applied": False,
            "reason": "no_active_regions",
            "bbox": None,
            "orig_size": orig_size,
            "cropped_size": orig_size
        }

    y_min, y_max = int(ar[0]), int(ar[-1])
    x_min, x_max = int(ac[0]), int(ac[-1])

    # Safeguard 3: Crop too small
    if (y_max - y_min) < min_dim or (x_max - x_min) < min_dim:
        return pil_img, {
            "crop_applied": False,
            "reason": "crop_too_small",
            "bbox": (x_min, y_min, x_max, y_max),
            "orig_size": orig_size,
            "cropped_size": orig_size
        }

    # Safeguard 4: Check if empty margin actually exists
    total_margin_x = (x_min + (w - 1 - x_max)) / w
    total_margin_y = (y_min + (h - 1 - y_max)) / h
    if total_margin_x < 0.05 and total_margin_y < 0.05:
        return pil_img, {
            "crop_applied": False,
            "reason": "already_tightly_cropped",
            "bbox": (0, 0, w, h),
            "orig_size": orig_size,
            "cropped_size": orig_size
        }

    pad_x = int(w * pad_pct)
    pad_y = int(h * pad_pct)

    x1 = max(0, x_min - pad_x)
    x2 = min(w, x_max + 1 + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(h, y_max + 1 + pad_y)

    bbox = (x1, y1, x2, y2)
    cropped = pil_img.crop(bbox)
    return cropped, {
        "crop_applied": True,
        "reason": "success",
        "bbox": bbox,
        "orig_size": orig_size,
        "cropped_size": cropped.size
    }

def preprocess_fingerprint(image_input, target_size=IMAGE_SIZE, apply_enhancement=False, source="upload"):
    """
    Unified preprocessing function for single-model fallback.
    """
    raw_pil = load_image_to_pil(image_input)
    rgb_pil = composite_alpha_on_white(raw_pil)

    crop_meta = {"crop_applied": False, "reason": "source_upload", "bbox": None, "orig_size": rgb_pil.size, "cropped_size": rgb_pil.size}
    if source == "mfs100":
        cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)
    else:
        cropped_pil = rgb_pil

    resized_pil = resize_to_target(cropped_pil, target_size=target_size)
    img_array = np.array(resized_pil, dtype=np.uint8)

    if apply_enhancement:
        img_array = apply_fingerprint_enhancement(img_array, enabled=True)
        display_pil = Image.fromarray(img_array)
    else:
        display_pil = resized_pil

    float_array = img_array.astype(np.float32)
    batch_array = np.expand_dims(float_array, axis=0)
    preprocessed_batch = tf.keras.applications.resnet50.preprocess_input(batch_array)

    return preprocessed_batch, display_pil

def enhance_mfs100_ridge_contrast(pil_img):
    """
    Hardware-Domain Ridge Enhancement for Mantra MFS100 optical captures.
    Normalizes the low-contrast, washed-out optical sensor ridges (std ~40, min ~70)
    to match the high-contrast distribution of the training dataset (std ~95, deep black ridges min ~0).

    Steps:
    1. Grayscale conversion.
    2. Adaptive local contrast equalization via CLAHE (clipLimit=2.5, tileGrid=(8,8)).
    3. Dynamic percentile contrast stretching (p1 -> 15, p99 -> 245).
    4. Edge-preserving bilateral filtering to suppress CMOS sensor optical noise.
    5. Conversion back to 3-channel RGB PIL Image.
    """
    try:
        gray = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(gray)

        p_low = float(np.percentile(cl, 1.0))
        p_high = float(np.percentile(cl, 99.0))
        if p_high > p_low:
            stretched = np.clip((cl - p_low) / (p_high - p_low) * 230.0 + 15.0, 0, 255).astype(np.uint8)
        else:
            stretched = cl

        denoised = cv2.bilateralFilter(stretched, d=3, sigmaColor=25, sigmaSpace=25)
        return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))
    except Exception:
        return pil_img.convert("RGB")



def transform_c5(image_input):
    """
    E2 training-matched preprocessing pipeline (Round 14 fix).
    Steps: alpha composite -> crop MFS100 foreground -> CLAHE enhancement -> resize 112x112
    Returns raw float32 [0, 255] array -- NO preprocess_input, matching
    scratch/experiment6_fusion_supcon.py lines 45-51 and scratch/final_evaluation_e2.py lines 30-36.
    """
    raw_pil = load_image_to_pil(image_input)
    rgb_pil = composite_alpha_on_white(raw_pil)
    cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)
    enhanced_pil = enhance_mfs100_ridge_contrast(cropped_pil)
    resized_pil = resize_to_target(enhanced_pil, target_size=(112, 112))
    img_array = np.array(resized_pil, dtype=np.float32)   # raw [0, 255], no preprocess_input
    return img_array, resized_pil, crop_meta


def transform_c5_legacy_with_preprocess_input(image_input):
    """
    LEGACY / UNUSED. This was the deployed version before Round 14 fix.
    It incorrectly applied densenet.preprocess_input, shifting activations and
    causing ~85% A+ collapse on 21 locked test participants (5/21 = 23.81%).
    Kept for reference only. Do NOT use for E2 inference.
    """
    raw_pil = load_image_to_pil(image_input)
    rgb_pil = composite_alpha_on_white(raw_pil)
    cropped_pil, crop_meta = crop_mfs100_foreground(rgb_pil)
    enhanced_pil = enhance_mfs100_ridge_contrast(cropped_pil)
    resized_pil = resize_to_target(enhanced_pil, target_size=(112, 112))
    img_array = np.array(resized_pil, dtype=np.float32)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)
    return img_array, resized_pil, crop_meta


def load_and_preprocess_for_training(image_path, target_size=IMAGE_SIZE):
    """
    Fast loader for tf.data / training pipelines.
    Takes an image file path and returns preprocessed float32 array (224, 224, 3).
    """
    with Image.open(image_path) as img:
        rgb_img = composite_alpha_on_white(img)
        resized_img = resize_to_target(rgb_img, target_size=target_size)
        arr = np.array(resized_img, dtype=np.float32)
        return tf.keras.applications.resnet50.preprocess_input(arr)
