import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "reports" / "efficientnet_e2_ensemble_experiment"
FUSION_DIR = PROJECT_ROOT / "reports" / "model_fusion"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

npz_path = FUSION_DIR / "all_models_participant_probs.npz"
npz_data = np.load(npz_path)

print("Arrays in npz:")
for k in npz_data.files:
    print(f"  {k}: shape={npz_data[k].shape}")

df_split = pd.read_csv(SPLIT_CSV)
dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])].copy()
dev_pids = sorted(dev_df["participant_id"].unique())
print(f"\nTotal dev participants in split CSV: {len(dev_pids)}")

# Check E2 OOF audit report for class distribution and accuracy
e2_oof = npz_data["E2 Ensemble (OOF)"]
effnet_dev = npz_data["EfficientNet-B0"]

# Check if there is a mapping of row index to participant ID
print(f"E2 OOF shape: {e2_oof.shape}, EffNet dev shape: {effnet_dev.shape}")

# Let's inspect leading_models_participant_predictions.csv
lead_csv = FUSION_DIR / "leading_models_participant_predictions.csv"
if lead_csv.is_file():
    df_lead = pd.read_csv(lead_csv)
    print("\nleading_models_participant_predictions.csv columns:")
    print(df_lead.columns)
    print(df_lead.head(5))
