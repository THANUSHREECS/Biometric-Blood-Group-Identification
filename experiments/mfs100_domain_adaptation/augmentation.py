"""
experiments/mfs100_domain_adaptation/augmentation.py
======================================================
Isolated conservative data augmentation pipeline for MFS100 training images.
CRITICAL: Applied ONLY to training images. Never applied to validation or test sets.

Parameters:
  - Rotation: +/- 8 degrees
  - Translation: +/- 6%
  - Scaling / Zoom: +/- 5%
  - Brightness factor: [0.90, 1.10] (+/- 10%)
  - Contrast factor: [0.85, 1.15] (+/- 15%)
  - Gamma variation: [0.85, 1.15]
  - Gaussian blur: sigma in [0.0, 0.6]
  - Additive Gaussian noise: sigma in [0.0, 4.0]
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

AUGMENTATION_CONFIG = {
    "rotation_range_deg": (-8.0, 8.0),
    "translation_pct": (-0.06, 0.06),
    "scale_range": (0.95, 1.05),
    "brightness_range": (0.90, 1.10),
    "contrast_range": (0.85, 1.15),
    "gamma_range": (0.85, 1.15),
    "gaussian_blur_sigma_max": 0.6,
    "additive_noise_sigma_max": 4.0
}

def apply_conservative_mfs100_augmentation(pil_img):
    """
    Applies conservative stochastic transformations simulating optical acquisition variability.
    Takes a PIL Image and returns an augmented PIL Image of identical dimensions.
    """
    img = pil_img.copy()

    # 1. Random Rotation (-8 to +8 degrees)
    rot_deg = random.uniform(*AUGMENTATION_CONFIG["rotation_range_deg"])
    img = img.rotate(rot_deg, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))

    # 2. Random Translation (-6% to +6%) and Scaling (0.95 to 1.05)
    w, h = img.size
    scale = random.uniform(*AUGMENTATION_CONFIG["scale_range"])
    dx = int(w * random.uniform(*AUGMENTATION_CONFIG["translation_pct"]))
    dy = int(h * random.uniform(*AUGMENTATION_CONFIG["translation_pct"]))

    # Affine transform matrix
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    arr = np.array(img)
    arr_aff = cv2.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    img = Image.fromarray(arr_aff)

    # 3. Random Brightness (0.90 to 1.10)
    b_factor = random.uniform(*AUGMENTATION_CONFIG["brightness_range"])
    img = ImageEnhance.Brightness(img).enhance(b_factor)

    # 4. Random Contrast (0.85 to 1.15)
    c_factor = random.uniform(*AUGMENTATION_CONFIG["contrast_range"])
    img = ImageEnhance.Contrast(img).enhance(c_factor)

    # Convert to float for photometric gamma, blur, and noise
    arr = np.array(img, dtype=np.float32)

    # 5. Random Gamma Correction (0.85 to 1.15)
    gamma = random.uniform(*AUGMENTATION_CONFIG["gamma_range"])
    inv_gamma = 1.0 / gamma
    arr = 255.0 * ((arr / 255.0) ** inv_gamma)

    # 6. Mild Gaussian Blur (sigma <= 0.6 with 50% probability)
    if random.random() < 0.5:
        sigma = random.uniform(0.1, AUGMENTATION_CONFIG["gaussian_blur_sigma_max"])
        ksize = 3
        arr = cv2.GaussianBlur(arr, (ksize, ksize), sigma)

    # 7. Mild Additive Sensor Noise (sigma <= 4.0 with 50% probability)
    if random.random() < 0.5:
        noise_sigma = random.uniform(0.5, AUGMENTATION_CONFIG["additive_noise_sigma_max"])
        noise = np.random.normal(0, noise_sigma, arr.shape)
        arr = arr + noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
