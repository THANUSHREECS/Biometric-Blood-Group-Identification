import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"

# --------------------------------------------------------------------
# 1. Phase 2 Selection Rules (Sample SD with ddof=1)
# --------------------------------------------------------------------
print("=== PHASE 2 SELECTION RULES (Sample SD ddof=1) ===")
with open(EXP_DIR / "phase2_selection_rules_comparison.json") as f:
    p2 = json.load(f)

for rule_name, rule_data in p2.items():
    accs = [s['overall']['accuracy'] for s in rule_data.values()]
    bals = [s['overall']['balanced_accuracy'] for s in rule_data.values()]
    f1s = [s['overall']['macro_f1'] for s in rule_data.values()]
    print(f"\nRule: {rule_name}")
    print(f"  Accuracy:          mean={np.mean(accs):.4f}, std={np.std(accs, ddof=1):.4f}, seeds={[round(x, 4) for x in accs]}")
    print(f"  Balanced Accuracy: mean={np.mean(bals):.4f}, std={np.std(bals, ddof=1):.4f}, seeds={[round(x, 4) for x in bals]}")
    print(f"  Macro F1:          mean={np.mean(f1s):.4f}, std={np.std(f1s, ddof=1):.4f}, seeds={[round(x, 4) for x in f1s]}")

# --------------------------------------------------------------------
# 2. Phase 2 Empirical p-values against Selection Null (N=100)
# --------------------------------------------------------------------
print("\n=== PHASE 2 EMPIRICAL P-VALUES AGAINST 100-REPEAT NULL ===")
null_df = pd.read_csv(EXP_DIR / "selection_inclusive_null_results.csv")
null_acc = null_df["accuracy"].values
null_bal = null_df["balanced_accuracy"].values
null_f1 = null_df["macro_f1"].values

def calc_p(null_vals, target):
    # p = (sum(null >= target) + 1) / (N + 1)
    return (np.sum(null_vals >= target) + 1) / (len(null_vals) + 1)

targets = {
    "Documented Historical Pooled": {"acc": 0.5375, "bal": 0.5347, "f1": 0.5487},
    "Rule (i) Seed 42 Rerun": {"acc": 0.5375, "bal": 0.5183, "f1": 0.5218},
    "Rule (i) 3-Seed Mean": {
        "acc": np.mean([s['overall']['accuracy'] for s in p2['heldout_selection_rule_i'].values()]),
        "bal": np.mean([s['overall']['balanced_accuracy'] for s in p2['heldout_selection_rule_i'].values()]),
        "f1": np.mean([s['overall']['macro_f1'] for s in p2['heldout_selection_rule_i'].values()])
    },
    "Clean Rule (a) Inner Val": {
        "acc": np.mean([s['overall']['accuracy'] for s in p2['clean_rule_a_inner_val'].values()]),
        "bal": np.mean([s['overall']['balanced_accuracy'] for s in p2['clean_rule_a_inner_val'].values()]),
        "f1": np.mean([s['overall']['macro_f1'] for s in p2['clean_rule_a_inner_val'].values()])
    },
    "Clean Rule (b) Fixed 100 Ep": {
        "acc": np.mean([s['overall']['accuracy'] for s in p2['clean_rule_b_fixed_epochs'].values()]),
        "bal": np.mean([s['overall']['balanced_accuracy'] for s in p2['clean_rule_b_fixed_epochs'].values()]),
        "f1": np.mean([s['overall']['macro_f1'] for s in p2['clean_rule_b_fixed_epochs'].values()])
    }
}

for name, t in targets.items():
    p_acc = calc_p(null_acc, t['acc'])
    p_bal = calc_p(null_bal, t['bal'])
    p_f1 = calc_p(null_f1, t['f1'])
    print(f"\n{name}:")
    print(f"  Target: Acc={t['acc']:.4f}, Bal={t['bal']:.4f}, F1={t['f1']:.4f}")
    print(f"  Empirical p-values: p(Acc)={p_acc:.4f}, p(Bal)={p_bal:.4f}, p(F1)={p_f1:.4f}")

# --------------------------------------------------------------------
# 3. Phase 3 Arm 1 Label-Shuffled Control
# --------------------------------------------------------------------
print("\n=== PHASE 3 ARM 1 LABEL SHUFFLED CONTROL ===")
p3_df = pd.read_csv(EXP_DIR / "phase3_purged_cv_results.csv")
shuff = p3_df[p3_df["condition"] == "0%_label_shuffled_control"]
if len(shuff) > 0:
    for seed, sgrp in shuff.groupby("seed"):
        print(f"  Seed {seed}: Acc={sgrp['val_accuracy'].mean():.4f}, Bal={sgrp['val_balanced_accuracy'].mean():.4f}, F1={sgrp['val_macro_f1'].mean():.4f}")
    print(f"  Overall Label Shuffled Mean Across Folds: Acc={shuff['val_accuracy'].mean():.4f} +/- {shuff['val_accuracy'].std(ddof=1):.4f}")
    print(f"  Overall Label Shuffled Mean Balanced:    Bal={shuff['val_balanced_accuracy'].mean():.4f} +/- {shuff['val_balanced_accuracy'].std(ddof=1):.4f}")
else:
    print("  Not run!")

# --------------------------------------------------------------------
# 4. Phase 3 Summary Numbers Verification
# --------------------------------------------------------------------
print("\n=== PHASE 3 SUMMARY JSON CHECK ===")
with open(EXP_DIR / "phase3_purged_cv_summary.json") as f:
    p3_sum = json.load(f)

for arm_key in ["primary_arm_fixed_epochs", "secondary_arm_inner_val"]:
    print(f"\nArm: {arm_key}")
    arm_data = p3_sum[arm_key]
    print(f"  0% baseline: mean={arm_data['0%_baseline']['mean']:.4f}, std={arm_data['0%_baseline']['std']:.4f}")
    for frac in ["10%_purge", "25%_purge", "50%_purge"]:
        fd = arm_data[frac]
        print(f"  {frac}:")
        print(f"    Purged: mean={fd['purged_mean']:.4f}, std={fd['purged_std']:.4f}")
        print(f"    Random: mean={fd['random_mean']:.4f}, std={fd['random_std']:.4f}")
        print(f"    Paired Delta: mean={fd['paired_delta_mean']:.4f}, std={fd['paired_delta_std']:.4f}")
