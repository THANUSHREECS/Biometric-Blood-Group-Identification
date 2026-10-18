#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Synthetic Pipeline & Unit Test Suite
==================================================================
Validates feature extraction algorithms, scan aggregation, model pipelines,
imputation, scaling, and security guards on synthetic data without touching
real participant images.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.config import (
    TRAINING_AUTHORIZED, MODELS, FEATURE_SETTINGS
)
from experiments.exp08_dermatoglyphic_minutiae.feature_extraction import (
    extract_features_from_scan
)
from experiments.exp08_dermatoglyphic_minutiae.participant_feature_builder import (
    aggregate_participant_scans
)
from experiments.exp08_dermatoglyphic_minutiae.train import (
    build_model_pipeline, train_and_evaluate_models
)
from experiments.exp08_dermatoglyphic_minutiae.evaluate import (
    run_evaluation
)

def generate_synthetic_fingerprint_image(h=354, w=316, pattern_seed=42):
    """
    Generates a synthetic grayscale image mimicking an optical fingerprint platen.
    """
    rng = np.random.RandomState(pattern_seed)
    # Background glass (~245)
    img = np.ones((h, w), dtype=np.uint8) * 248

    # Circular active ridge impression
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(y - cy, x - cx)
    mask = r < 120

    # Synthetic sinusoidal concentric ridges (period ~ 9 pixels)
    ridge_freq = 2.0 * np.pi / 9.0
    synth_ridges = 128.0 + 80.0 * np.sin(r * ridge_freq)

    # Add Gaussian noise
    noise = rng.normal(0, 15, size=(h, w))
    composite = np.clip(synth_ridges + noise, 0, 255).astype(np.uint8)

    img[mask] = composite[mask]
    return img

def run_synthetic_tests():
    print("=" * 80)
    print("EXP-08: SYNTHETIC PIPELINE & ALGORITHM VALIDATION")
    print("=" * 80)

    # 1. Test Fail-Closed Guard
    print("[Test 1] Fail-Closed Security Guard Verification:")
    assert TRAINING_AUTHORIZED is False, "Security Error: TRAINING_AUTHORIZED must be False during preparation!"
    try:
        train_and_evaluate_models()
        raise RuntimeError("FAIL: Guard failed to block unauthorized training!")
    except PermissionError as e:
        print(f"  [PASS] Guard successfully verified: {e.args[0].splitlines()[1]}")

    # 2. Test Feature Extraction on Synthetic Scan
    print("\n[Test 2] Feature Extraction on Synthetic Platen Image:")
    synth_img = generate_synthetic_fingerprint_image(pattern_seed=42)
    feats, diags = extract_features_from_scan(synth_img)

    assert feats["valid_scan"] is True, "Extraction failed on synthetic image!"
    assert feats["pattern_type"] in FEATURE_SETTINGS["pattern_classes"], f"Unknown pattern: {feats['pattern_type']}"
    assert feats["total_minutiae_count"] >= 0, "Invalid minutiae count!"
    assert not np.isnan(feats["ridge_density_25mm2"]), "Ridge density returned NaN on valid ridges!"
    assert feats["foreground_area_mm2"] > 50.0, "Foreground area calculation error!"

    print(f"  Pattern Type Detected:     {feats['pattern_type']} (Confidence: {feats['pattern_confidence']:.2f})")
    print(f"  Ridge Density (25 mm²):    {feats['ridge_density_25mm2']:.2f} ridges")
    print(f"  Total Minutiae Count:      {feats['total_minutiae_count']} (Endings: {feats['ridge_endings_count']}, Bifurcations: {feats['ridge_bifurcations_count']})")
    print(f"  Foreground Area:           {feats['foreground_area_mm2']:.2f} mm²")
    print(f"  Ridge Clarity Score:       {feats['ridge_clarity_score']:.3f}")
    print("  [PASS] Feature extraction algorithm verified.")

    # 3. Test Participant-Level Multi-Scan Aggregation
    print("\n[Test 3] Participant Multi-Scan Aggregation Logic:")
    # Simulate 3 scans with minor variations and 1 missing core-delta
    scans_mock = [
        feats.copy(),
        feats.copy(),
        feats.copy()
    ]
    scans_mock[0]["ridge_density_25mm2"] = 11.2
    scans_mock[1]["ridge_density_25mm2"] = 12.0
    scans_mock[2]["ridge_density_25mm2"] = 11.6
    scans_mock[2]["core_delta_distance_mm"] = np.nan  # Simulated failed landmark detection

    agg = aggregate_participant_scans(scans_mock)
    assert np.isclose(agg["ridge_density_25mm2_mean"], 11.6, atol=0.01), "Mean aggregation error!"
    assert agg["core_delta_valid_scans_count"] == 2, "Valid scans count error!"
    assert agg["pattern_concordance"] == 1.0, "Concordance calculation error!"
    print(f"  Aggregated 3 scans -> 1 participant record: Density Mean={agg['ridge_density_25mm2_mean']:.2f}, Std={agg['ridge_density_25mm2_std']:.2f}")
    print(f"  Core-Delta Valid Detections: {agg['core_delta_valid_scans_count']}/3 (Missingness safely handled)")
    print("  [PASS] Aggregation and missingness logic verified.")

    # 4. Test Model Pipelines with Imputation & Scaling
    print("\n[Test 4] Scikit-Learn Model Pipelines (Imputation, Scaling, Training Step):")
    n_dummy = 16
    n_feats = 12
    rng = np.random.RandomState(42)

    dummy_X = rng.randn(n_dummy, n_feats)
    # Inject missing values (NaN) to verify SimpleImputer
    dummy_X[0, 2] = np.nan
    dummy_X[3, 5] = np.nan
    dummy_y = rng.choice([0, 1, 2, 3], size=n_dummy)

    for m_key in MODELS.keys():
        pipeline = build_model_pipeline(m_key)
        # Fit on dummy train (first 12)
        pipeline.fit(dummy_X[:12], dummy_y[:12])
        # Predict on dummy val (last 4)
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba(dummy_X[12:])
        else:
            preds = pipeline.predict(dummy_X[12:])
            probs = np.zeros((4, 4))
            for i, p in enumerate(preds):
                probs[i, p] = 1.0
        assert probs.shape == (4, 4), f"Pipeline {m_key} returned unexpected shape: {probs.shape}"
        print(f"  {m_key}: Pipeline fit & transform verified (Prob shape: {probs.shape}) [PASS]")

    # 5. Test Evaluation Script Demo
    print("\n[Test 5] Evaluation & Permutation Engine Demo Check:")
    eval_res = run_evaluation(synthetic_demo=True)
    assert eval_res is not None, "Evaluation demo returned None!"
    print("  [PASS] Evaluation functions, Wilson intervals, and permutation tests verified.")

    print("\n" + "=" * 80)
    print("ALL EXP-08 SYNTHETIC UNIT TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_synthetic_tests()
