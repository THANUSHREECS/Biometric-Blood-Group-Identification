#!/usr/bin/env python3
"""
EXP-07-RES-FIDELITY: Configuration & Safety Guard Specification
==============================================================
Defines the controlled comparison between 112x112 and 224x224 input resolution
using the three-scan DenseNet-121 multi-instance architecture.

Strict Safeguards:
  - 80 Development participants ONLY (240 scans).
  - 21-participant locked test cohort strictly quarantined (ZERO access).
  - 26-participant hospital cohort strictly quarantined (ZERO access).
  - FAIL-CLOSED TRAINING GUARD: TRAINING_AUTHORIZED must be True to run on real data.
  - Outer-fold checkpoint selection completely abolished (zero outer-fold metrics monitoring).
"""

import os
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp07_resolution_fidelity"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp07"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
HOSPITAL_DIR = PROJECT_ROOT / "data" / "hospital_verified"

# Protected Checkpoints that must never be overwritten
PROTECTED_CKPT_DIRS = [
    PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2",
    PROJECT_ROOT / "experiments" / "exp03" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp05" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06a" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06b" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06c" / "checkpoints",
    PROJECT_ROOT / "experiments" / "exp06d" / "checkpoints",
]

# FAIL-CLOSED AUTHORIZATION GUARD
# Restored to False. Training for EXP-07 has completed.
TRAINING_AUTHORIZED = False

# Class Definitions
NATIVE_CLASSES = ["A+", "B+", "AB+", "O+"]
NATIVE_REV_MAP = {c: i for i, c in enumerate(NATIVE_CLASSES)}

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Global Hyperparameters
SEED = 42
BATCH_SIZE = 16
N_OUTER_FOLDS = 5
FIXED_EPOCHS = 80
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
FIXED_TEMPERATURE = 0.05
FIXED_LAMBDA = 0.50

# Experimental Conditions (Arms)
CONDITIONS = {
    "arm_a_112": {
        "name": "Arm A: 112x112 Input Resolution (Baseline Spatial Scale)",
        "input_resolution": (112, 112),
        "input_shape": (3, 112, 112, 3),
        "preprocessing_mode": "raw_crop_lanczos",
        "description": "Composite on white -> MFS100 platen crop -> Lanczos resize to 112x112 -> raw float32 [0, 255]."
    },
    "arm_b_224": {
        "name": "Arm B: 224x224 Input Resolution (High Fidelity Ridge Scale)",
        "input_resolution": (224, 224),
        "input_shape": (3, 224, 224, 3),
        "preprocessing_mode": "raw_crop_lanczos",
        "description": "Composite on white -> MFS100 platen crop -> Lanczos resize to 224x224 -> raw float32 [0, 255]."
    }
}

# Architecture & Parameter Specifications
MODEL_SPEC = {
    "backbone": "DenseNet121",
    "backbone_weights": "imagenet",
    "backbone_frozen": True,
    "expected_frozen_params": 7037504,
    "fusion_dim": 3072,  # 3 scans * 1024 features
    "classifier_dense_dim": 512,
    "dropout_rate": 0.5,
    "num_classes": 4,
    "projection_dim": 128,
    "expected_trainable_vars": 6,
    "expected_trainable_params": 1968772
}
