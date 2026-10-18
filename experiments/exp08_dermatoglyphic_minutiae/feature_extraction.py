#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Handcrafted Biometric Feature Extraction
======================================================================
Extracts interpretable, biologically grounded dermatoglyphic features from
optical fingerprint images:
  1. Primary Pattern Type (Whorl, Left Loop, Right Loop, Arch)
  2. Sub-Core Ridge Density (Ridges per 25 mm² in standardized forensic ROI)
  3. Minutiae Extraction (Ridge Endings, Bifurcations, Total Count via Crossing Number)
  4. Core-to-Delta Physical Distance (in mm at 500 DPI)
  5. Image Quality & Ridge Coherence Metrics
"""

import sys
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import skimage.morphology as morph

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.config import (
    PIXELS_PER_MM, FEATURE_SETTINGS
)

def segment_foreground_platen(gray_img, thresh=238, min_area_pixels=5000):
    """
    Segments the active finger impression on the optical sensor platen.
    Returns:
      mask: uint8 binary mask (255 for active ridge region, 0 for empty glass)
      quality_ok: bool
    """
    # Active ridges are darker than platen glass background (~240-255)
    binary = (gray_img < thresh).astype(np.uint8) * 255

    # Morphological closing to fill inter-ridge valleys and sweat pores
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    if num_labels <= 1:
        return np.zeros_like(gray_img, dtype=np.uint8), False

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    if area < min_area_pixels:
        return np.zeros_like(gray_img, dtype=np.uint8), False

    mask = (labels == largest_idx).astype(np.uint8) * 255
    return mask, True

def compute_orientation_field(gray_img, mask, block_size=16):
    """
    Estimates the local ridge orientation field using gradient structure tensor.
    Returns:
      theta: 2D array of angles in radians [-pi/2, pi/2]
      coherence: 2D array of orientation consistency [0, 1]
    """
    h, w = gray_img.shape
    sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient covariance components
    g_xx = sobel_x ** 2
    g_yy = sobel_y ** 2
    g_xy = sobel_x * sobel_y

    # Block-wise averaging
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size * block_size)
    g_xx_b = cv2.filter2D(g_xx, -1, kernel)
    g_yy_b = cv2.filter2D(g_yy, -1, kernel)
    g_xy_b = cv2.filter2D(g_xy, -1, kernel)

    # Orientation angle
    v_x = 2.0 * g_xy_b
    v_y = g_xx_b - g_yy_b
    theta = 0.5 * np.arctan2(v_x, v_y)

    # Coherence (ridge flow clarity)
    denom = g_xx_b + g_yy_b + 1e-6
    coherence = np.sqrt(v_x ** 2 + v_y ** 2) / denom
    coherence = np.clip(coherence, 0.0, 1.0)
    coherence[mask == 0] = 0.0
    return theta, coherence

def detect_singular_points(theta, mask, block_size=16):
    """
    Detects Core and Delta singular points via Poincaré Index on orientation field.
    Returns:
      cores: list of (x, y) coordinates
      deltas: list of (x, y) coordinates
    """
    h, w = theta.shape
    cores = []
    deltas = []

    step = block_size
    mask_b = cv2.resize(mask, (w // step, h // step), interpolation=cv2.INTER_NEAREST)
    theta_b = theta[::step, ::step]
    bh, bw = theta_b.shape

    for r in range(1, bh - 1):
        for c in range(1, bw - 1):
            if mask_b[r, c] == 0:
                continue
            # Neighborhood around block (r, c): 8 neighbors counter-clockwise
            nbrs = [
                theta_b[r - 1, c - 1], theta_b[r - 1, c], theta_b[r - 1, c + 1],
                theta_b[r, c + 1], theta_b[r + 1, c + 1], theta_b[r + 1, c],
                theta_b[r + 1, c - 1], theta_b[r, c - 1]
            ]
            diffs = []
            for i in range(8):
                d = nbrs[(i + 1) % 8] - nbrs[i]
                # Wrap angle difference to [-pi/2, pi/2]
                while d > np.pi / 2.0:
                    d -= np.pi
                while d < -np.pi / 2.0:
                    d += np.pi
                diffs.append(d)

            poincare = np.sum(diffs) / np.pi
            if np.isclose(poincare, 1.0, atol=0.25):
                cores.append((c * step + step // 2, r * step + step // 2))
            elif np.isclose(poincare, -1.0, atol=0.25):
                deltas.append((c * step + step // 2, r * step + step // 2))

    # Cluster nearby points to avoid multiple detections of same landmark
    def cluster_points(pts, min_dist=30):
        if not pts:
            return []
        clustered = []
        for p in pts:
            if not any(np.hypot(p[0] - c[0], p[1] - c[1]) < min_dist for c in clustered):
                clustered.append(p)
        return clustered

    return cluster_points(cores), cluster_points(deltas)

def classify_primary_pattern(cores, deltas, theta, mask):
    """
    Classifies fingerprint pattern into: WHORL, LEFT_LOOP, RIGHT_LOOP, ARCH.
    Based on Henry/Galton forensic dermatoglyphic classification rules.
    """
    n_cores = len(cores)
    n_deltas = len(deltas)

    if n_cores >= 2 or (n_cores == 1 and n_deltas >= 2):
        return "WHORL", 0.85
    elif n_cores == 1 and n_deltas == 1:
        core_x = cores[0][0]
        delta_x = deltas[0][0]
        if delta_x > core_x:
            return "LEFT_LOOP", 0.80  # Delta on right, loops open to left
        else:
            return "RIGHT_LOOP", 0.80  # Delta on left, loops open to right
    elif n_cores == 1 and n_deltas == 0:
        # Infer loop direction from mean orientation in central zone
        mean_theta = np.mean(theta[mask > 0])
        if mean_theta > 0.15:
            return "LEFT_LOOP", 0.65
        elif mean_theta < -0.15:
            return "RIGHT_LOOP", 0.65
        else:
            return "ARCH", 0.60
    elif n_cores == 0:
        return "ARCH", 0.75
    else:
        return "WHORL", 0.50

def compute_subcore_ridge_density(gray_img, mask, cores, roi_size_px=98):
    """
    Calculates friction ridge density in a standardized 5 mm x 5 mm (roi_size_px)
    sub-core region (as standardized in forensic anthropology).
    Returns:
      ridges_per_25mm2: float
      density_valid: bool
    """
    h, w = gray_img.shape
    if cores:
        # Standard sub-core location: 2 mm (~40 px) below primary core
        cx, cy = cores[0]
        cy_sub = cy + int(2.0 * PIXELS_PER_MM)
    else:
        # Center of foreground mask
        pts = cv2.findNonZero(mask)
        if pts is None or len(pts) == 0:
            return np.nan, False
        pts_2d = pts.reshape(-1, 2)
        cx, cy_sub = int(np.mean(pts_2d[:, 0])), int(np.mean(pts_2d[:, 1]))

    half = roi_size_px // 2
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy_sub - half)
    y2 = min(h, cy_sub + half)

    roi = gray_img[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if roi.size == 0 or np.mean(roi_mask > 0) < 0.60:
        return np.nan, False

    # Detrend and compute 2D power spectrum to find dominant spatial frequency
    roi_norm = (roi.astype(np.float32) - np.mean(roi))
    dft = np.fft.fftshift(np.fft.fft2(roi_norm))
    mag = np.abs(dft)

    rh, rw = mag.shape
    cy_f, cx_f = rh // 2, rw // 2
    # Mask out DC component
    mag[cy_f - 2:cy_f + 3, cx_f - 2:cx_f + 3] = 0.0

    # Human ridge peak corresponds to wavelengths between 0.3 mm (6 px) and 0.7 mm (14 px)
    # At 5 mm ROI, peak radial frequency corresponds to ~7 to 16 cycles per 5 mm
    y_idx, x_idx = np.ogrid[:rh, :rw]
    rad = np.hypot(y_idx - cy_f, x_idx - cx_f)

    # Valid radial band in cycles per ROI: [5.0, 20.0]
    valid_band = (rad >= 5.0) & (rad <= 20.0)
    if not np.any(valid_band):
        return np.nan, False

    # Weighted peak radial frequency (ridges per 5 mm window = ridges per 25 mm²)
    peak_val = np.sum(rad[valid_band] * mag[valid_band]) / (np.sum(mag[valid_band]) + 1e-12)

    # Typical biological range: 8 to 16 ridges per 25 mm² (40 to 80 ridges/cm)
    ridges_per_25mm2 = float(np.clip(peak_val, 6.0, 22.0))
    return ridges_per_25mm2, True

def extract_minutiae_crossing_number(gray_img, mask, edge_margin=10):
    """
    Extracts ridge endings and bifurcations via Crossing Number (CN) on skeletonized ridges.
    Returns:
      endings: list of (x, y)
      bifurcations: list of (x, y)
    """
    # 1. Binarization (Otsu thresholding on contrast-enhanced image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[mask == 0] = 0

    # 2. Morphological skeletonization
    skeleton = morph.skeletonize(binary > 0).astype(np.uint8)

    # Erode mask to discard spurious minutiae along platen boundaries
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (edge_margin * 2, edge_margin * 2)))

    # 3. Crossing Number Algorithm on 8-neighborhood
    h, w = skeleton.shape
    endings = []
    bifurcations = []

    # Pad skeleton to prevent out-of-bounds indexing
    sk_pad = np.pad(skeleton, 1, mode="constant", constant_values=0)

    # 8-neighbors sequence: (r-1, c-1), (r-1, c), (r-1, c+1), (r, c+1),
    #                       (r+1, c+1), (r+1, c), (r+1, c-1), (r, c-1)
    pts = np.argwhere((skeleton == 1) & (eroded_mask > 0))
    for r, c in pts:
        p_r, p_c = r + 1, c + 1
        p = [
            sk_pad[p_r - 1, p_c - 1], sk_pad[p_r - 1, p_c], sk_pad[p_r - 1, p_c + 1],
            sk_pad[p_r, p_c + 1], sk_pad[p_r + 1, p_c + 1], sk_pad[p_r + 1, p_c],
            sk_pad[p_r + 1, p_c - 1], sk_pad[p_r, p_c - 1]
        ]
        # CN = 0.5 * sum(|p_i - p_{i+1}|)
        cn = 0.5 * sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8))
        if cn == 1.0:
            endings.append((c, r))
        elif cn == 3.0:
            bifurcations.append((c, r))

    return endings, bifurcations

def extract_features_from_scan(image_input):
    """
    Main extraction pipeline for a single fingerprint impression.
    Returns:
      feature_dict: dict of numerical and categorical biometric measurements
      diagnostics: dict of intermediate quality markers
    """
    # Safe load
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input if image_input.ndim == 2 else cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    # 1. Foreground segmentation
    mask, seg_ok = segment_foreground_platen(img)
    if not seg_ok:
        return {
            "valid_scan": False,
            "pattern_type": "UNKNOWN",
            "pattern_confidence": 0.0,
            "ridge_density_25mm2": np.nan,
            "ridge_endings_count": 0,
            "ridge_bifurcations_count": 0,
            "total_minutiae_count": 0,
            "minutiae_density_cm2": 0.0,
            "core_detected": 0,
            "delta_detected": 0,
            "core_delta_distance_mm": np.nan,
            "ridge_clarity_score": 0.0,
            "foreground_area_mm2": 0.0
        }, {}

    foreground_pixels = int(np.sum(mask > 0))
    foreground_area_mm2 = float(foreground_pixels / (PIXELS_PER_MM ** 2))

    # 2. Orientation field & coherence
    theta, coherence = compute_orientation_field(img, mask)
    mean_clarity = float(np.mean(coherence[mask > 0])) if foreground_pixels > 0 else 0.0

    # 3. Singular points & Pattern
    cores, deltas = detect_singular_points(theta, mask)
    pattern, pattern_conf = classify_primary_pattern(cores, deltas, theta, mask)

    # 4. Ridge density
    ridge_density, rd_valid = compute_subcore_ridge_density(img, mask, cores)

    # 5. Minutiae extraction
    endings, bifurcations = extract_minutiae_crossing_number(img, mask)
    total_minutiae = len(endings) + len(bifurcations)
    minutiae_density_cm2 = float(total_minutiae / (foreground_area_mm2 / 100.0)) if foreground_area_mm2 > 0 else 0.0

    # 6. Core-to-Delta distance
    if cores and deltas:
        cx, cy = cores[0]
        dx, dy = deltas[0]
        dist_px = float(np.hypot(cx - dx, cy - dy))
        core_delta_dist_mm = float(dist_px / PIXELS_PER_MM)
        core_det = 1
        delta_det = 1
    else:
        core_delta_dist_mm = np.nan
        core_det = 1 if cores else 0
        delta_det = 1 if deltas else 0

    features = {
        "valid_scan": True,
        "pattern_type": pattern,
        "pattern_confidence": float(pattern_conf),
        "ridge_density_25mm2": float(ridge_density),
        "ridge_endings_count": int(len(endings)),
        "ridge_bifurcations_count": int(len(bifurcations)),
        "total_minutiae_count": int(total_minutiae),
        "minutiae_density_cm2": float(minutiae_density_cm2),
        "core_detected": int(core_det),
        "delta_detected": int(delta_det),
        "core_delta_distance_mm": float(core_delta_dist_mm),
        "ridge_clarity_score": float(mean_clarity),
        "foreground_area_mm2": float(foreground_area_mm2)
    }

    diagnostics = {
        "cores": cores,
        "deltas": deltas,
        "foreground_pixels": foreground_pixels
    }
    return features, diagnostics
