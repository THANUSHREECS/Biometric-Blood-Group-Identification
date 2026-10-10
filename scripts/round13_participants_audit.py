import pandas as pd
import numpy as np

def main():
    split_csv = "reports/phase3_participant_benchmark/participant_split.csv"
    part_csv = "data/mfs100_final/metadata/participants.csv"
    
    print("="*70)
    print("STEP 1: WHO WERE E2's PARTICIPANTS?")
    print("="*70)
    
    df_split = pd.read_csv(split_csv)
    print(f"\n1. Column names of {split_csv}:")
    print(df_split.columns.tolist())
    
    print(f"\n2. First 5 rows of {split_csv}:")
    print(df_split.head(5).to_string())
    
    print("\n3. Unique participants per split:")
    split_part_counts = df_split.groupby("split")["participant_id"].nunique()
    print(split_part_counts)
    
    print("\nTotal scans per split:")
    print(df_split["split"].value_counts())
    
    print("\n4. Participant counts per split and per class:")
    # Dedup by participant_id to get participant-level class distribution
    part_level = df_split.drop_duplicates(subset=["participant_id"])
    ct = pd.crosstab(part_level["split"], part_level["blood_group"], margins=True)
    print(ct)
    
    # Also group by dev (TRAIN+VALIDATION) vs test (TEST)
    part_level["split_group"] = part_level["split"].apply(lambda s: "DEV (TRAIN+VAL)" if s in ["TRAIN", "VALIDATION"] else "TEST")
    print("\nParticipant counts per split group (DEV vs TEST) and per class:")
    ct_group = pd.crosstab(part_level["split_group"], part_level["blood_group"], margins=True)
    print(ct_group)
    
    print("\n5. Joining with data/mfs100_final/metadata/participants.csv:")
    df_part = pd.read_csv(part_csv)
    print(f"Total participants in {part_csv}: {len(df_part)}")
    print(f"Columns in {part_csv}: {df_part.columns.tolist()}")
    print("First 3 rows:")
    print(df_part.head(3).to_string())
    
    merged = pd.merge(df_part, part_level[["participant_id", "split", "split_group", "blood_group"]], on="participant_id", how="left", suffixes=("_meta", "_split"))
    print("\nMerge match summary:")
    print(merged["split_group"].value_counts(dropna=False))
    
    unmatched = merged[merged["split_group"].isna()]
    print(f"Unmatched participants: {len(unmatched)}")
    
    print("\n--- Summary Breakdown ---")
    n_dev = (merged["split_group"] == "DEV (TRAIN+VAL)").sum()
    n_test = (merged["split_group"] == "TEST").sum()
    print(f"Of the {len(df_part)} live-evaluation participants:")
    print(f"  - E2 Development participants (TRAIN + VALIDATION): {n_dev}")
    print(f"  - E2 Locked Test participants (TEST):               {n_test}")

if __name__ == "__main__":
    main()
