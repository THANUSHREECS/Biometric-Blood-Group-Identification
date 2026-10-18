#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Participant Split & Cohort Quarantine Validator
==============================================================================
Validates that only the 80 Dev participants are included in EXP-08,
confirms 5-fold stratification by participant ID, and guarantees zero leakage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.config import (
    SPLIT_CSV, HOSPITAL_DIR, NATIVE_CLASSES, NATIVE_REV_MAP, SEED, N_OUTER_FOLDS
)

def validate_participant_splits():
    print("=" * 80)
    print("EXP-08: PARTICIPANT SPLIT & SAFETY QUARANTINE VALIDATION")
    print("=" * 80)

    assert SPLIT_CSV.is_file(), f"Manifest missing: {SPLIT_CSV}"
    df_split = pd.read_csv(SPLIT_CSV)

    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
    test_df = df_split[df_split["split"] == "TEST"].copy()

    dev_pids = sorted(dev_df["participant_id"].unique())
    test_pids = sorted(test_df["participant_id"].unique())

    print(f"[Check 1] Cohort Sizes & Scans:")
    print(f"  Dev-80 Cohort:   {len(dev_pids)} participants ({len(dev_df)} scans)")
    print(f"  Locked Test Set: {len(test_pids)} participants ({len(test_df)} scans)")

    assert len(dev_pids) == 80, f"Expected 80 Dev participants, got {len(dev_pids)}"
    assert len(test_pids) == 21, f"Expected 21 Test participants, got {len(test_pids)}"
    assert len(dev_df) == 240, f"Expected 240 Dev scans, got {len(dev_df)}"

    # Zero overlap check
    overlap = set(dev_pids).intersection(set(test_pids))
    assert len(overlap) == 0, f"FATAL LEAKAGE: Overlap between Dev and Locked Test: {overlap}"
    print("  [PASS] Dev-80 and Locked Test cohorts are 100% disjoint.")

    # Hospital check
    if HOSPITAL_DIR.exists():
        hosp_paths = [p for p in dev_df["scan_path"] if "hospital" in str(p).lower()]
        assert len(hosp_paths) == 0, f"FATAL LEAKAGE: Hospital paths detected in Dev split!"
        print(f"  [PASS] Hospital cohort quarantined (0 references in Dev manifest).")

    # 5-Fold StratifiedGroupKFold Integrity
    print(f"\n[Check 2] 5-Fold Participant-Level Stratification:")
    pid_to_bg = dev_df.groupby("participant_id")["blood_group"].first().to_dict()
    y = np.array([NATIVE_REV_MAP[pid_to_bg[p]] for p in dev_pids])
    groups = np.array(dev_pids)

    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(np.zeros((len(dev_pids), 1)), y, groups))

    all_val_pids = []
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        t_pids = set(groups[t_idx])
        v_pids = set(groups[v_idx])

        assert len(t_pids) == 64, f"Fold {f_idx}: Expected 64 train, got {len(t_pids)}"
        assert len(v_pids) == 16, f"Fold {f_idx}: Expected 16 val, got {len(v_pids)}"
        assert len(t_pids.intersection(v_pids)) == 0, f"Fold {f_idx}: Participant leakage detected!"

        all_val_pids.extend(list(v_pids))

        dist_v = {NATIVE_CLASSES[c]: int((y[v_idx] == c).sum()) for c in range(4)}
        print(f"  Fold {f_idx}: 64 Train / 16 Val | Val Distribution: {dist_v} [PASS]")

    assert len(set(all_val_pids)) == 80, "Validation folds do not form exact partition of Dev-80!"
    print("\n  [PASS] Out-of-fold validation sets form an exact, mutually exclusive partition of Dev-80.")
    print("=" * 80)
    print("ALL SPLIT INTEGRITY CHECKS PASSED FOR EXP-08.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    validate_participant_splits()
