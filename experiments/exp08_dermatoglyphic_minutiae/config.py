#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Configuration & Security Specifications
======================================================================
Defines feature extraction parameters, dataset splits, classifier pipelines,
and fail-closed security guards for the handcrafted dermatoglyphic benchmark.

Strict Safeguards:
  - 80 Development participants ONLY (240 scans).
  - 21-participant locked test cohort strictly quarantined (0 access).
  - 26-participant hospital cohort strictly quarantined (0 access).
  - FAIL-CLOSED TRAINING GUARD: TRAINING_AUTHORIZED must be True to execute real-data training.
  - Leak-free aggregation: 3 scans from one participant produce exactly ONE participant row.
  - Imputers and scalers fit strictly on training folds.
"""

import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp08_dermatoglyphic_minutiae"
DATA_DIR = EXP_DIR / "features"
RESULTS_DIR = EXP_DIR / "results"
REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement_audit"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
HOSPITAL_DIR = PROJECT_ROOT / "data" / "hospital_verified"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FAIL-CLOSED AUTHORIZATION GUARD
# Explicitly authorized by user for EXP-08 execution on Dev-80 cohort.
TRAINING_AUTHORIZED = True

# Class Definitions
NATIVE_CLASSES = ["A+", "B+", "AB+", "O+"]
NATIVE_REV_MAP = {c: i for i, c in enumerate(NATIVE_CLASSES)}

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Optical Sensor Physics (Mantra MFS100)
SENSOR_DPI = 500.0
PIXELS_PER_MM = SENSOR_DPI / 25.4  # ~19.685 pixels/mm
PLATTEN_DIMS = (316, 354)  # Width, Height in pixels

# Biometric Feature Extraction Settings
FEATURE_SETTINGS = {
    # 5 mm x 5 mm central sub-core measurement ROI
    "ridge_density_roi_mm": 5.0,
    "ridge_density_roi_pixels": int(5.0 * PIXELS_PER_MM),  # ~98 pixels
    "minutiae_edge_margin_pixels": 10,
    "block_size_orientation": 16,
    "pattern_classes": ["ARCH", "LEFT_LOOP", "RIGHT_LOOP", "WHORL", "UNKNOWN"]
}

# Cross-Validation & Modeling Hyperparameters
SEED = 42
N_OUTER_FOLDS = 5

MODELS = {
    "majority_baseline": {
        "name": "Majority-Class Zero-Intelligence Classifier",
        "type": "dummy"
    },
    "logistic_regression": {
        "name": "L2-Regularized Logistic Regression",
        "type": "linear",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 1000
    },
    "random_forest": {
        "name": "Random Forest Classifier",
        "type": "ensemble",
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_split": 4,
        "min_samples_leaf": 2
    }
}
