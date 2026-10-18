#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Participant-Level Feature Aggregation
===================================================================
Aggregates scan-level handcrafted features across 3 impressions per participant
into EXACTLY ONE feature vector per participant.
Prevents multi-scan data duplication / instance leakage.
"""

import sys
import json
import collections
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.config import (
    SPLIT_CSV, HOSPITAL_DIR, DATA_DIR, NATIVE_CLASSES, NATIVE_REV_MAP
)
from experiments.exp08_dermatoglyphic_minutiae.feature_extraction import (
    extract_features_from_scan
)

def aggregate_participant_scans(scan_features_list):
    """
    Combines 3 scan feature dictionaries into a single participant-level record.
    """
    assert len(scan_features_list) == 3, f"Expected 3 scans, got {len(scan_features_list)}"

    # 1. Numeric Feature Aggregation (Mean & Std)
    numeric_keys = [
        "ridge_density_25mm2", "ridge_endings_count", "ridge_bifurcations_count",
        "total_minutiae_count", "minutiae_density_cm2", "ridge_clarity_score",
        "foreground_area_mm2"
    ]
    agg_dict = {}

    for k in numeric_keys:
        vals = [f[k] for f in scan_features_list if not np.isnan(f.get(k, np.nan))]
        if vals:
            agg_dict[f"{k}_mean"] = float(np.mean(vals))
            agg_dict[f"{k}_std"] = float(np.std(vals))
        else:
            agg_dict[f"{k}_mean"] = np.nan
            agg_dict[f"{k}_std"] = np.nan

    # 2. Core-to-Delta Distance Aggregation (explicit missingness handling)
    cd_dists = [f["core_delta_distance_mm"] for f in scan_features_list if not np.isnan(f.get("core_delta_distance_mm", np.nan))]
    agg_dict["core_delta_distance_mm_mean"] = float(np.mean(cd_dists)) if cd_dists else np.nan
    agg_dict["core_delta_distance_mm_std"] = float(np.std(cd_dists)) if len(cd_dists) > 1 else 0.0
    agg_dict["core_delta_valid_scans_count"] = int(len(cd_dists))
    agg_dict["core_detection_rate"] = float(np.mean([f["core_detected"] for f in scan_features_list]))
    agg_dict["delta_detection_rate"] = float(np.mean([f["delta_detected"] for f in scan_features_list]))

    # 3. Categorical Pattern Aggregation (Majority Vote & Concordance)
    patterns = [f["pattern_type"] for f in scan_features_list]
    counts = collections.Counter(patterns)
    majority_pattern, count = counts.most_common(1)[0]

    agg_dict["majority_pattern"] = majority_pattern
    agg_dict["pattern_concordance"] = float(count / 3.0)  # 1.0 = perfect agreement, 0.67 = 2/3, 0.33 = discordant

    # One-hot encoded patterns
    for p_type in ["ARCH", "LEFT_LOOP", "RIGHT_LOOP", "WHORL"]:
        agg_dict[f"pattern_is_{p_type.lower()}"] = 1.0 if majority_pattern == p_type else 0.0

    return agg_dict

def build_development_feature_matrix(save_csv=True):
    """
    Extracts and aggregates features for all 80 Development participants.
    Quarantines test and hospital sets.
    """
    assert SPLIT_CSV.is_file(), f"Split manifest missing: {SPLIT_CSV}"
    df_split = pd.read_csv(SPLIT_CSV)

    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
    test_df = df_split[df_split["split"] == "TEST"].copy()

    dev_pids = sorted(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    # Quarantine Guard Assertions
    assert len(dev_pids) == 80, f"Expected 80 Dev participants, got {len(dev_pids)}"
    assert len(test_pids) == 21, f"Expected 21 Test participants, got {len(test_pids)}"
    assert len(set(dev_pids).intersection(test_pids)) == 0, "DATA LEAKAGE: Overlap with locked test set!"

    records = []
    print(f"Extracting handcrafted biometric features for {len(dev_pids)} Dev participants (240 scans)...")

    for idx, pid in enumerate(dev_pids, 1):
        p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
        assert len(p_scans) == 3, f"Participant {pid} does not have exactly 3 scans!"

        bg = p_scans.iloc[0]["blood_group"]
        assert bg in NATIVE_CLASSES, f"Unknown blood group: {bg}"

        scan_features = []
        for _, row in p_scans.iterrows():
            scan_p = Path(row["scan_path"])
            assert scan_p.is_file(), f"Scan file not found: {scan_p}"
            assert "hospital" not in str(scan_p).lower(), "QUARANTINE BREACH: Hospital scan in dev builder!"
            feats, _ = extract_features_from_scan(scan_p)
            scan_features.append(feats)

        participant_agg = aggregate_participant_scans(scan_features)
        participant_agg["participant_id"] = pid
        participant_agg["blood_group"] = bg
        participant_agg["target_label"] = NATIVE_REV_MAP[bg]
        records.append(participant_agg)

        if idx % 20 == 0 or idx == len(dev_pids):
            print(f"  Processed {idx}/{len(dev_pids)} participants...")

    feature_df = pd.DataFrame(records)

    # Reorder columns: ID, Label, Target, Features
    meta_cols = ["participant_id", "blood_group", "target_label"]
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    feature_df = feature_df[meta_cols + feat_cols]

    if save_csv:
        out_path = DATA_DIR / "dev80_handcrafted_features.csv"
        feature_df.to_csv(out_path, index=False)
        print(f"\nFeature matrix saved to: {out_path} ({feature_df.shape[0]} rows, {feature_df.shape[1]} columns)")

    return feature_df

if __name__ == "__main__":
    print("Testing participant feature aggregation logic...")
    # This is callable as a module or standalone test
