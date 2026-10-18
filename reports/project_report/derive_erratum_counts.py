import json
import pandas as pd

def derive_counts(artifact_map_path="reports/project_report/erratum_artifact_map.json"):
    # Load manually curated artifact map
    with open(artifact_map_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    summary_docs = set(meta["summary_docs"])
    checkpoint_map = meta["checkpoint_map"]
    
    df = pd.read_csv("reports/project_report/locked_cohort_usage.csv")
    
    scoring_rows = df[~df["artifact_path"].isin(summary_docs)].copy()
    scoring_rows["checkpoint"] = scoring_rows["artifact_path"].map(checkpoint_map)
    
    before_exp03 = scoring_rows[scoring_rows["available_before_exp03_06d"] == "yes"]
    after_exp03 = scoring_rows[scoring_rows["available_before_exp03_06d"] == "no"]
    
    atomic_checkpoints_before = set()
    for cp in before_exp03["checkpoint"].dropna():
        for part in cp.split(" + "):
            atomic_checkpoints_before.add(part.strip())
            
    print(f"Total prediction artifacts: {len(scoring_rows)}")
    print(f"Prediction artifacts before EXP-03: {len(before_exp03)}")
    print(f"Prediction artifacts after EXP-03: {len(after_exp03)} (one primary evaluation, one read-only re-analysis)")
    print(f"Distinct checkpoint paths before EXP-03 ({len(atomic_checkpoints_before)}):")
    for i, p in enumerate(sorted(atomic_checkpoints_before), 1):
        print(f"  {i:2d}. {p}")

if __name__ == "__main__":
    derive_counts()
