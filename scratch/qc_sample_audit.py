import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.feature_extraction import extract_features_from_scan
from experiments.exp08_dermatoglyphic_minutiae.participant_feature_builder import aggregate_participant_scans

df = pd.read_csv(PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv")
dev_df = df[df["split"].isin(["TRAIN", "VALIDATION"])].sort_values(["participant_id", "scan_path"])
sample_pids = sorted(dev_df["participant_id"].unique())[:5]

print("=== QUALITY CONTROL SAMPLE AUDIT (5 Participants, 15 Scans) ===")
for pid in sample_pids:
    p_scans = dev_df[dev_df["participant_id"] == pid]
    scans_feats = []
    print(f"\nParticipant: {pid}")
    for idx, row in p_scans.iterrows():
        p = Path(row["scan_path"])
        f, _ = extract_features_from_scan(p)
        scans_feats.append(f)
        cd_str = f"{f['core_delta_distance_mm']:.2f} mm" if not pd.isna(f['core_delta_distance_mm']) else "None"
        print(f"  Scan {p.name}: Valid={f['valid_scan']}, Pattern={f['pattern_type']:<10} (conf={f['pattern_confidence']:.2f}), "
              f"Density={f['ridge_density_25mm2']:.1f}, Minutiae={f['total_minutiae_count']:3d} "
              f"(End={f['ridge_endings_count']:2d}, Bif={f['ridge_bifurcations_count']:2d}), "
              f"Core={f['core_detected']}, Delta={f['delta_detected']}, CD_Dist={cd_str}")
    
    agg = aggregate_participant_scans(scans_feats)
    print(f"  --> Aggregated: Pattern={agg['majority_pattern']} (concordance={agg['pattern_concordance']:.2f}), "
          f"Density Mean={agg['ridge_density_25mm2_mean']:.1f} (std={agg['ridge_density_25mm2_std']:.2f}), "
          f"Minutiae Mean={agg['total_minutiae_count_mean']:.1f}, Valid CD Scans={agg['core_delta_valid_scans_count']}/3")
