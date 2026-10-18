# PROVENANCE NOTE:
# This script is a verified copy of scratch/compare_seed42_historical.py,
# preserved in scripts/ for full audit reproduction and provenance.
# Do not modify or delete the scratch original.

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

hist_df = pd.read_csv("reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv")

with open("experiments/leakage_audit/phase2_selection_rules_comparison.json", "r") as f:
    res = json.load(f)

print("="*60)
print("PER-FOLD COMPARISON: HISTORICAL VS SEED-42 RERUN")
print("="*60)
s42_pf = res["heldout_selection_rule_i"]["seed_42"]["per_fold"]

print(f"{'Fold':<8} | {'Historical Acc':<15} | {'Rerun Acc':<15} | {'Historical F1':<15} | {'Rerun F1':<15}")
print("-" * 75)

for fold in range(1, 6):
    sub = hist_df[hist_df["fold_id"] == fold]
    h_acc = accuracy_score(sub["true_label"], sub["predicted_label"]) * 100
    h_f1 = f1_score(sub["true_label"], sub["predicted_label"], average="macro", zero_division=0)
    
    r_acc = s42_pf[f"fold_{fold}"]["accuracy"] * 100
    r_f1 = s42_pf[f"fold_{fold}"]["macro_f1"]
    
    print(f"Fold {fold:<3} | {h_acc:6.2f}%         | {r_acc:6.2f}%         | {h_f1:6.4f}          | {r_f1:6.4f}")

print("-" * 75)
ov_h_acc = accuracy_score(hist_df["true_label"], hist_df["predicted_label"]) * 100
ov_h_f1 = f1_score(hist_df["true_label"], hist_df["predicted_label"], average="macro")

ov_r_acc = res["heldout_selection_rule_i"]["seed_42"]["overall"]["accuracy"] * 100
ov_r_f1 = res["heldout_selection_rule_i"]["seed_42"]["overall"]["macro_f1"]

print(f"{'Pooled':<8} | {ov_h_acc:6.2f}%         | {ov_r_acc:6.2f}%         | {ov_h_f1:6.4f}          | {ov_r_f1:6.4f}")

# Now let's get the exact participant-by-participant predictions from the rerun
# We can re-evaluate or load them
import sys
sys.path.insert(0, ".")
from scripts.leakage_audit_phase2_experiments import run_cv_heldout_rule
npz = np.load("experiments/leakage_audit/dev_densenet121_features.npz")
y_true = npz["labels"]
pids = npz["pids"]
classes = ["A+", "B+", "AB+", "O+"]

_, _, rerun_preds = run_cv_heldout_rule(y_true, seed=42)
rerun_labels = [classes[i] for i in rerun_preds]

# Compare with historical predictions participant by participant
hist_map = dict(zip(hist_df["participant_id"], hist_df["predicted_label"]))
discordant = 0
matches = 0
for pid, r_lbl in zip(pids, rerun_labels):
    h_lbl = hist_map[pid]
    if r_lbl != h_lbl:
        discordant += 1
    else:
        matches += 1

print(f"\nParticipant-Level Prediction Comparison (N=80):")
print(f"  Matching Predictions:   {matches}/80 ({matches/80*100:.1f}%)")
print(f"  Differing Predictions:  {discordant}/80 ({discordant/80*100:.1f}%)")

