import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, chi2_contingency

parts = pd.read_csv("data/mfs100_final/metadata/participants.csv")
split_df = pd.read_csv("reports/phase3_participant_benchmark/participant_split.csv")
dev_ids = set(split_df[split_df["split"].isin(["TRAIN", "VALIDATION"])]["participant_id"])

dev_parts = parts[parts["participant_id"].isin(dev_ids)].copy()
print(f"Dev participants matched: {len(dev_parts)}")

# Parse collection date/time
dev_parts["dt"] = pd.to_datetime(dev_parts["collection_date"])
dev_parts["date"] = dev_parts["dt"].dt.date
dev_parts["hour"] = dev_parts["dt"].dt.hour
dev_parts["timestamp"] = dev_parts["dt"].astype("int64") // 10**9

print("\nCollection Date Distribution by Blood Group (Dev 80):")
date_ct = pd.crosstab(dev_parts["blood_group"], dev_parts["date"])
print(date_ct)

chi2, p_val, dof, _ = chi2_contingency(date_ct)
print(f"\nChi-Square on Collection Date vs Blood Group: Chi2 = {chi2:.4f}, p = {p_val:.4e} (dof={dof})")

groups_ts = [dev_parts[dev_parts["blood_group"] == bg]["timestamp"].values for bg in ["A+", "AB+", "B+", "O+"]]
f_ts, p_ts = f_oneway(*groups_ts)
h_ts, p_kts = kruskal(*groups_ts)
print(f"ANOVA on collection timestamp: F = {f_ts:.4f}, p = {p_ts:.4e}")
print(f"Kruskal-Wallis on collection timestamp: H = {h_ts:.4f}, p = {p_kts:.4e}")

for bg in ["A+", "AB+", "B+", "O+"]:
    sub = dev_parts[dev_parts["blood_group"] == bg]
    med = sub["dt"].median()
    print(f"  {bg:4} (N={len(sub):2}): Median Date/Time = {med}, Range = [{sub['dt'].min()} to {sub['dt'].max()}]")
