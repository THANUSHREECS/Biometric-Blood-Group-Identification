import json
import numpy as np
import pandas as pd

with open("experiments/leakage_audit/phase3_purged_cv_summary.json", "r") as f:
    d = json.load(f)

print("="*75)
print("PHASE 3 COMPLETE SUMMARY")
print("="*75)
print("Validity Gate:", d["validity_gate"])

print("\n--- PRIMARY ARM: FIXED 50 EPOCHS ---")
p_arm = d["primary_arm_fixed_epochs"]
print(f"0% Unpurged Baseline: {p_arm['0%_baseline']['mean']*100:.2f}% ± {p_arm['0%_baseline']['std']*100:.2f}%")
for lvl in ["10%_purge", "25%_purge", "50%_purge"]:
    entry = p_arm[lvl]
    pm, ps = entry["purged_mean"] * 100, entry["purged_std"] * 100
    rm, rs = entry["random_mean"] * 100, entry["random_std"] * 100
    dm, ds = entry["paired_delta_mean"] * 100, entry["paired_delta_std"] * 100
    print(f"{lvl:<10}: Purged = {pm:5.2f}% ± {ps:4.2f}% | Random-Drop = {rm:5.2f}% ± {rs:4.2f}% | Paired Delta = {dm:+5.2f}% ± {ds:4.2f}%")

print("\n--- SECONDARY ARM: INNER VALIDATION SPLIT ---")
s_arm = d["secondary_arm_inner_val"]
print(f"0% Unpurged Baseline: {s_arm['0%_baseline']['mean']*100:.2f}% ± {s_arm['0%_baseline']['std']*100:.2f}%")
for lvl in ["10%_purge", "25%_purge", "50%_purge"]:
    entry = s_arm[lvl]
    pm, ps = entry["purged_mean"] * 100, entry["purged_std"] * 100
    rm, rs = entry["random_mean"] * 100, entry["random_std"] * 100
    dm, ds = entry["paired_delta_mean"] * 100, entry["paired_delta_std"] * 100
    print(f"{lvl:<10}: Purged = {pm:5.2f}% ± {ps:4.2f}% | Random-Drop = {rm:5.2f}% ± {rs:4.2f}% | Paired Delta = {dm:+5.2f}% ± {ds:4.2f}%")

# Let's inspect the CSV for accuracy and macro F1, and final train accuracy
df = pd.read_csv("experiments/leakage_audit/phase3_purged_cv_results.csv")
print("\n--- OVERALL METRICS PER ARM AND PURGE FRACTION (MEAN OVER 3 SEEDS) ---")
agg = df.groupby(["arm", "condition", "purge_fraction"]).agg({
    "val_accuracy": ["mean", "std"],
    "val_balanced_accuracy": ["mean", "std"],
    "val_macro_f1": ["mean", "std"],
    "final_train_acc": ["mean", "std"]
}).reset_index()

print(agg.to_string())

# Per-fold paired differences table
print("\n--- PER-FOLD PAIRED DELTAS (PURGED - RANDOM-DROP) IN PRIMARY ARM ---")
p_arm_df = df[df["arm"] == "Arm1_FixedEpochs"]
for p_frac in [0.10, 0.25, 0.50]:
    p_sub = p_arm_df[(p_arm_df["condition"] == "similarity_purged") & (p_arm_df["purge_fraction"] == p_frac)]
    r_sub = p_arm_df[(p_arm_df["condition"] == "random_drop_control") & (p_arm_df["purge_fraction"] == p_frac)]
    merged = pd.merge(
        p_sub[["seed", "fold", "val_balanced_accuracy"]].rename(columns={"val_balanced_accuracy": "bal_p"}),
        r_sub[["seed", "fold", "val_balanced_accuracy"]].rename(columns={"val_balanced_accuracy": "bal_r"}),
        on=["seed", "fold"]
    )
    merged["delta"] = merged["bal_p"] - merged["bal_r"]
    print(f"\nPurge Level {int(p_frac*100)}%:")
    for _, row in merged.iterrows():
        print(f"  Seed {int(row['seed'])}, Fold {int(row['fold'])}: Purged={row['bal_p']*100:.2f}%, Random={row['bal_r']*100:.2f}%, Delta={row['delta']*100:+.2f}%")
