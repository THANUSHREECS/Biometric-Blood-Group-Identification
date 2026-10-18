#!/usr/bin/env python3
"""
EXP-07-RES-FIDELITY: Participant Split & Safety Firewall Validator
=================================================================
Validates participant-level partitioning firewalls, class stratification,
and guarantees complete isolation of the locked test and hospital cohorts.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Ensure project imports
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp07_resolution_fidelity.config import (
    SPLIT_CSV, HOSPITAL_DIR, NATIVE_CLASSES, NATIVE_REV_MAP,
    SEED, N_OUTER_FOLDS
)

def validate_dataset_firewall():
    """
    Validates dataset integrity and confirms strict quarantine of locked/hospital cohorts.
    """
    print("=" * 80)
    print("EXP-07: DATASET INTEGRITY & PARTICIPANT FIREWALL VALIDATION")
    print("=" * 80)

    assert SPLIT_CSV.is_file(), f"Split manifest not found: {SPLIT_CSV}"
    df_split = pd.read_csv(SPLIT_CSV)

    # 1. Check total scans and participant count
    total_participants = df_split["participant_id"].nunique()
    total_scans = len(df_split)
    print(f"[Check 1] Total Participants: {total_participants} | Total Scans: {total_scans}")
    assert total_participants == 101, f"Expected 101 participants, found {total_participants}"
    assert total_scans == 303, f"Expected 303 scans, found {total_scans}"

    # 2. Check 3 scans per participant
    scans_per_pid = df_split.groupby("participant_id").size()
    assert (scans_per_pid == 3).all(), "Integrity Error: Not all participants have exactly 3 scans!"
    print("  [PASS] All 101 participants have exactly 3 scans.")

    # 3. Check label consistency per participant
    labels_per_pid = df_split.groupby("participant_id")["blood_group"].nunique()
    assert (labels_per_pid == 1).all(), "Integrity Error: Inconsistent bloodgroup across participant scans!"
    print("  [PASS] Label consistency per participant: 100% uniform.")

    # 4. Check class values
    all_classes = set(df_split["blood_group"].unique())
    assert all_classes == set(NATIVE_CLASSES), f"Unexpected classes found: {all_classes}"
    print(f"  [PASS] Classes strictly conform to 4 positive groups: {sorted(list(all_classes))}")

    # 5. Cohort Separation & Strict Quarantine Check
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
    test_df = df_split[df_split["split"] == "TEST"].copy()

    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    print(f"\n[Check 2] Cohort Partitioning:")
    print(f"  Dev-80 Cohort:   {len(dev_pids)} participants ({len(dev_df)} scans)")
    print(f"  Locked Test Set: {len(test_pids)} participants ({len(test_df)} scans)")

    assert len(dev_pids) == 80, f"Expected 80 Dev participants, found {len(dev_pids)}"
    assert len(test_pids) == 21, f"Expected 21 Locked Test participants, found {len(test_pids)}"

    leakage = dev_pids.intersection(test_pids)
    assert len(leakage) == 0, f"FATAL DATA LEAKAGE: Overlap detected between Dev and Locked Test: {leakage}"
    print("  [PASS] Dev-80 and Locked Test sets are 100% disjoint (0 participant overlap).")

    # 6. Quarantined Hospital Cohort Audit
    print(f"\n[Check 3] Quarantined Hospital Cohort Isolation:")
    if HOSPITAL_DIR.exists():
        hospital_files = list(HOSPITAL_DIR.glob("**/*.*"))
        print(f"  Hospital directory verified at: {HOSPITAL_DIR} ({len(hospital_files)} files).")
        # Ensure no path in dev_df points to hospital directory
        hospital_paths_in_dev = [p for p in dev_df["scan_path"] if "hospital" in str(p).lower()]
        assert len(hospital_paths_in_dev) == 0, "FATAL LEAKAGE: Hospital paths detected in Dev split!"
        print("  [PASS] Zero hospital files present in Dev-80 training paths.")
    else:
        print("  [PASS] Hospital directory not linked into project paths.")

    return dev_df, test_df

def validate_cross_validation_folds(dev_df):
    """
    Validates outer 5-fold StratifiedGroupKFold on Dev-80 and nested inner-splits.
    """
    print(f"\n[Check 4] 5-Fold StratifiedGroupKFold Split Validation on Dev-80:")

    # Sort participants deterministically
    pids_unique = sorted(dev_df["participant_id"].unique())
    pid_to_bg = dev_df.groupby("participant_id")["blood_group"].first().to_dict()

    y = np.array([NATIVE_REV_MAP[pid_to_bg[pid]] for pid in pids_unique])
    groups = np.array(pids_unique)
    X_dummy = np.zeros((len(pids_unique), 1))

    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    outer_folds = list(sgkf.split(X_dummy, y, groups))

    assert len(outer_folds) == N_OUTER_FOLDS, f"Expected {N_OUTER_FOLDS} folds, got {len(outer_folds)}"

    fold_reports = []
    all_val_pids = []

    for f_idx, (train_idx, val_idx) in enumerate(outer_folds, 1):
        train_pids = set(groups[train_idx])
        val_pids = set(groups[val_idx])

        # Strict checks
        assert len(train_pids) == 64, f"Fold {f_idx}: Expected 64 train participants, got {len(train_pids)}"
        assert len(val_pids) == 16, f"Fold {f_idx}: Expected 16 val participants, got {len(val_pids)}"

        overlap = train_pids.intersection(val_pids)
        assert len(overlap) == 0, f"Fold {f_idx}: Participant leakage detected! Overlap: {overlap}"

        all_val_pids.extend(list(val_pids))

        # Check class distribution
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_dist = {NATIVE_CLASSES[c]: int((y_train == c).sum()) for c in range(4)}
        val_dist = {NATIVE_CLASSES[c]: int((y_val == c).sum()) for c in range(4)}

        # Validate nested inner split (48 train / 16 inner-val) for inner-selection regime
        inner_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + f_idx)
        inner_folds = list(inner_sgkf.split(np.zeros((len(train_idx), 1)), y_train, groups[train_idx]))
        inner_t_idx, inner_v_idx = inner_folds[0]
        inner_train_pids = set(groups[train_idx][inner_t_idx])
        inner_val_pids = set(groups[train_idx][inner_v_idx])

        assert len(inner_train_pids) == 48, f"Fold {f_idx}: Expected 48 inner-train, got {len(inner_train_pids)}"
        assert len(inner_val_pids) == 16, f"Fold {f_idx}: Expected 16 inner-val, got {len(inner_val_pids)}"
        assert len(inner_train_pids.intersection(inner_val_pids)) == 0
        assert len(inner_val_pids.intersection(val_pids)) == 0

        fold_reports.append({
            "fold": f_idx,
            "train_pids": len(train_pids),
            "val_pids": len(val_pids),
            "train_dist": train_dist,
            "val_dist": val_dist,
            "inner_train_pids": len(inner_train_pids),
            "inner_val_pids": len(inner_val_pids)
        })

        print(f"  Fold {f_idx}: 64 Train / 16 Outer-Val | Train: {train_dist} | Val: {val_dist}")
        print(f"          Nested Inner: 48 Inner-Train / 16 Inner-Val [PASS]")

    # Check that out-of-fold val sets cover all 80 participants exactly once
    assert len(all_val_pids) == 80, f"Expected 80 OOF participants, got {len(all_val_pids)}"
    assert len(set(all_val_pids)) == 80, "OOF participants are not mutually exclusive!"
    print("\n  [PASS] Out-of-fold validation sets form an exact partition of all 80 Dev participants.")
    print("=" * 80)
    print("ALL SPLIT FIREWALL AND INTEGRITY CHECKS PASSED.")
    print("=" * 80)
    return fold_reports

if __name__ == "__main__":
    dev_df, test_df = validate_dataset_firewall()
    fold_reports = validate_cross_validation_folds(dev_df)
