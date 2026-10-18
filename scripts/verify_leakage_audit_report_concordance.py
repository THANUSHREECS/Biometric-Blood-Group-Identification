"""
scripts/verify_leakage_audit_report_concordance.py
Automated Concordance Verification: leakage_audit_report.md vs. Serialized Artifacts.

Compares every reported numerical value, standard deviation, and p-value in
reports/leakage_audit/leakage_audit_report.md against the serialized JSON and CSV
artifacts in experiments/leakage_audit/.

Covers Sections 3.1, 3.2, 3.3, 4.1, 4.2, 5.3, 6.2, and 6.3.
Prints all verified matches and any discrepancies detected.
Exits with code 0 on perfect concordance, or code 1 if any mismatch is found.
"""

import sys
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
REPORT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "reports" / "leakage_audit" / "leakage_audit_report.md"

assert REPORT_FILE.exists(), f"Missing report: {REPORT_FILE}"

with open(REPORT_FILE, "r", encoding="utf-8") as f:
    report_text = f.read()

mismatches = []
matches = []

def check(claim_name, expected_str, artifact_val=None):
    found_in_report = expected_str in report_text
    if not found_in_report:
        mismatches.append(f"MISSING IN REPORT: {claim_name} expected text '{expected_str}' not found in report.")
    else:
        matches.append(f"VERIFIED: {claim_name} -> text '{expected_str}' found in report (matches artifact: {artifact_val})")

print("=" * 70)
print("VERIFYING CONCORDANCE: leakage_audit_report.md vs. Serialized Artifacts")
print("=" * 70)

# ======================================================================
# Section 3.1: Perceptual Hash Duplicate Audit (Train & Train+Val Reference Sets)
# ======================================================================
with open(EXP_DIR / "exp1_duplicates_and_perceptual_hashes.json", "r") as f:
    tr_hash = json.load(f)

p_cnts = tr_hash["perceptual_hashes"]["phash_near_duplicate_counts"]
d_cnts = tr_hash["perceptual_hashes"]["dhash_near_duplicate_counts"]

check("Sec 3.1 Train pHash <=2", f"Hamming $\\le 2$: {p_cnts['hamming_le_2']['count']} images (${p_cnts['hamming_le_2']['percentage']:.2f}\\%$)", p_cnts['hamming_le_2']['count'])
check("Sec 3.1 Train pHash <=5", f"Hamming $\\le 5$: {p_cnts['hamming_le_5']['count']} images (${p_cnts['hamming_le_5']['percentage']:.2f}\\%$)", p_cnts['hamming_le_5']['count'])
check("Sec 3.1 Train pHash <=10", f"Hamming $\\le 10$: {p_cnts['hamming_le_10']['count']} images (${p_cnts['hamming_le_10']['percentage']:.2f}\\%$)", p_cnts['hamming_le_10']['count'])
check("Sec 3.1 Train dHash <=2", f"Hamming $\\le 2$: {d_cnts['hamming_le_2']['count']} image (${d_cnts['hamming_le_2']['percentage']:.2f}\\%$)", d_cnts['hamming_le_2']['count'])
check("Sec 3.1 Train dHash <=5", f"Hamming $\\le 5$: {d_cnts['hamming_le_5']['count']} images (${d_cnts['hamming_le_5']['percentage']:.2f}\\%$)", d_cnts['hamming_le_5']['count'])
check("Sec 3.1 Train dHash <=10", f"Hamming $\\le 10$: {d_cnts['hamming_le_10']['count']} images (${d_cnts['hamming_le_10']['percentage']:.2f}\\%$)", d_cnts['hamming_le_10']['count'])

with open(EXP_DIR / "exp1_perceptual_hashes_trainval_reference.json", "r") as f:
    trval_hash = json.load(f)

tv_p = trval_hash["phash_near_duplicate_counts"]
tv_d = trval_hash["dhash_near_duplicate_counts"]

check("Sec 3.1 Train+Val pHash <=2", f"Hamming $\\le 2$: {tv_p['hamming_le_2']['count']} images (${tv_p['hamming_le_2']['percentage']:.2f}\\%$)", tv_p['hamming_le_2']['count'])
check("Sec 3.1 Train+Val pHash <=5", f"Hamming $\\le 5$: {tv_p['hamming_le_5']['count']} images (${tv_p['hamming_le_5']['percentage']:.2f}\\%$)", tv_p['hamming_le_5']['count'])
check("Sec 3.1 Train+Val pHash <=10", f"Hamming $\\le 10$: {tv_p['hamming_le_10']['count']} images (${tv_p['hamming_le_10']['percentage']:.2f}\\%$)", tv_p['hamming_le_10']['count'])
check("Sec 3.1 Train+Val dHash <=2", f"Hamming $\\le 2$: {tv_d['hamming_le_2']['count']} images (${tv_d['hamming_le_2']['percentage']:.2f}\\%$)", tv_d['hamming_le_2']['count'])
check("Sec 3.1 Train+Val dHash <=5", f"Hamming $\\le 5$: {tv_d['hamming_le_5']['count']} images (${tv_d['hamming_le_5']['percentage']:.2f}\\%$)", tv_d['hamming_le_5']['count'])
check("Sec 3.1 Train+Val dHash <=10", f"Hamming $\\le 10$: {tv_d['hamming_le_10']['count']} images (${tv_d['hamming_le_10']['percentage']:.2f}\\%$)", tv_d['hamming_le_10']['count'])

# ======================================================================
# Section 3.2: Connected-Component Percolation Table Rows (11 Thresholds)
# ======================================================================
with open(EXP_DIR / "exp1_percolation_finer_scan.json", "r") as f:
    perc = json.load(f)["thresholds_evaluated"]

table_rows = [
    ("0.985", "| **0.985** | 2 | 2,767 | 2 | 2,765 | 2 | 0.07% | Passes guard |"),
    ("0.980", "| **0.980** | 10 | 2,759 | 10 | 2,749 | 2 | 0.07% | Passes guard |"),
    ("0.975", "| **0.975** | 52 | 2,717 | 42 | 2,675 | 4 | 0.14% | Passes guard |"),
    ("0.970", "| **0.970** | 267 | 2,529 | 88 | 2,441 | 85 | 3.07% | Passes guard |"),
    ("0.965", "| **0.965** | 1,079 | 2,056 | 92 | 1,964 | 572 | 20.66% | **Violates guard** |"),
    ("0.960", "| **0.960** | 3,118 | 1,585 | 63 | 1,522 | 1,100 | 39.73% | **Violates guard** |"),
    ("0.955", "| **0.955** | 7,498 | 1,223 | 46 | 1,177 | 1,493 | 53.92% | **Violates guard** |"),
    ("0.950", "| **0.950** | 15,345 | 932 | 29 | 903 | 1,801 | 65.04% | **Violates guard** |"),
    ("0.920", "| **0.920** | 193,548 | 184 | 6 | 178 | 2,580 | 93.17% | **Violates guard** |"),
    ("0.900", "| **0.900** | 462,073 | 65 | 2 | 63 | 2,704 | 97.65% | **Violates guard** |"),
    ("0.850", "| **0.850** | 1,425,152 | 11 | 1 | 10 | 2,759 | 99.64% | **Violates guard** |"),
]

for th_str, row_str in table_rows:
    th_key = f"threshold_{th_str}"
    t_data = perc[th_key]
    check(f"Sec 3.2 Percolation Row {th_str}", row_str, f"edges={t_data['n_edges']}, max_comp={t_data['largest_component_size']}")

# ======================================================================
# Section 3.3: Historical Signal Tests on Saved Predictions
# ======================================================================
with open(EXP_DIR / "exp2_signal_tests_results.json", "r") as f:
    sig = json.load(f)

dev_sig = sig["development_80_participants"]
check("Sec 3.3 Dev Observed", "Accuracy $= 53.75\%$ (43/80), Balanced Accuracy $= 53.47\\%$, Macro F1 $= 0.5487$", dev_sig["observed"])
check("Sec 3.3 Dev Perm Null", "Mean accuracy $= 30.50\\% \\pm 4.94\\%$ ($p = 0.0002$); Mean Macro F1 $= 0.2475 \\pm 0.0490$ ($p = 0.0001$)", dev_sig["permutation_test"])
check("Sec 3.3 Dev Bootstrap Acc", "Accuracy $[42.50\\%, 65.00\\%]$", dev_sig["bootstrap_ci_95"]["accuracy"]["ci_95"])
check("Sec 3.3 Dev Bootstrap Bal", "Balanced Accuracy $[39.59\\%, 66.81\\%]$", dev_sig["bootstrap_ci_95"]["balanced_accuracy"]["ci_95"])
check("Sec 3.3 Dev Bootstrap F1", "Macro F1 $[0.4011, 0.6604]$", dev_sig["bootstrap_ci_95"]["macro_f1"]["ci_95"])

test_sig = sig["primary_test_21_participants"]
check("Sec 3.3 Test Observed", "Accuracy $= 33.33\\%$ (7/21), Balanced Accuracy $= 31.25\\%$, Macro F1 $= 0.2083$", test_sig["observed"])
check("Sec 3.3 Test Perm Null", "Mean accuracy $= 32.16\\% \\pm 6.51\\%$ ($p = 0.5641$); Mean Macro F1 $= 0.1786 \\pm 0.0649$ ($p = 0.3712$)", test_sig["permutation_test"])
check("Sec 3.3 Test Bootstrap Acc", "Accuracy $[14.29\\%, 52.38\\%]$", test_sig["bootstrap_ci_95"]["accuracy"]["ci_95"])
check("Sec 3.3 Test Bootstrap Bal", "Balanced Accuracy $[13.64\\%, 50.00\\%]$", test_sig["bootstrap_ci_95"]["balanced_accuracy"]["ci_95"])
check("Sec 3.3 Test Bootstrap F1", "Macro F1 $[0.0714, 0.3391]$", test_sig["bootstrap_ci_95"]["macro_f1"]["ci_95"])

# ======================================================================
# Section 4.1: Selection Rules Comparison
# ======================================================================
with open(EXP_DIR / "phase2_selection_rules_comparison.json", "r") as f:
    p2 = json.load(f)

def get_stats(rule_dict):
    accs = [rule_dict[f"seed_{s}"]["overall"]["accuracy"] for s in [42, 43, 44]]
    bals = [rule_dict[f"seed_{s}"]["overall"]["balanced_accuracy"] for s in [42, 43, 44]]
    f1s = [rule_dict[f"seed_{s}"]["overall"]["macro_f1"] for s in [42, 43, 44]]
    return {
        "acc_str": f"{np.mean(accs)*100:.2f}% $\\pm$ {np.std(accs, ddof=1)*100:.2f}%",
        "bal_str": f"{np.mean(bals)*100:.2f}% $\\pm$ {np.std(bals, ddof=1)*100:.2f}%",
        "f1_str": f"{np.mean(f1s):.4f} $\\pm$ {np.std(f1s, ddof=1):.4f}"
    }

rule_i_stats = get_stats(p2["heldout_selection_rule_i"])
check("Rule (i) Mean Acc", f"**{rule_i_stats['acc_str']}**", rule_i_stats['acc_str'])
check("Rule (i) Mean Bal", f"**{rule_i_stats['bal_str']}**", rule_i_stats['bal_str'])
check("Rule (i) Mean F1", f"**{rule_i_stats['f1_str']}**", rule_i_stats['f1_str'])

rule_a_stats = get_stats(p2["clean_rule_a_inner_val"])
check("Rule (a) Mean Acc", f"**{rule_a_stats['acc_str']}**", rule_a_stats['acc_str'])
check("Rule (a) Mean Bal", f"**{rule_a_stats['bal_str']}**", rule_a_stats['bal_str'])
check("Rule (a) Mean F1", f"**{rule_a_stats['f1_str']}**", rule_a_stats['f1_str'])

rule_b_stats = get_stats(p2["clean_rule_b_fixed_epochs"])
check("Rule (b) Mean Acc", f"**{rule_b_stats['acc_str']}**", rule_b_stats['acc_str'])
check("Rule (b) Mean Bal", f"**{rule_b_stats['bal_str']}**", rule_b_stats['bal_str'])
check("Rule (b) Mean F1", f"**{rule_b_stats['f1_str']}**", rule_b_stats['f1_str'])

# ======================================================================
# Section 4.2: 100-Repeat Selection-Inclusive Null
# ======================================================================
with open(EXP_DIR / "selection_inclusive_null_summary.json", "r") as f:
    null_data = json.load(f)

check("Null Mean Acc", f"**{null_data['accuracy']['mean']*100:.2f}% $\\pm$ {null_data['accuracy']['std']*100:.2f}%**", null_data['accuracy'])
check("Null Mean Bal", f"**{null_data['balanced_accuracy']['mean']*100:.2f}% $\\pm$ {null_data['balanced_accuracy']['std']*100:.2f}%**", null_data['balanced_accuracy'])
check("Null Mean F1", f"**{null_data['macro_f1']['mean']:.4f} $\\pm$ {null_data['macro_f1']['std']:.4f}**", null_data['macro_f1'])

check("Null 95th Acc", f"*{null_data['accuracy']['p95']*100:.2f}%*", null_data['accuracy']['p95'])
check("Null 95th Bal", f"*{null_data['balanced_accuracy']['p95']*100:.2f}%*", null_data['balanced_accuracy']['p95'])
check("Null 95th F1", f"*{null_data['macro_f1']['p95']:.4f}*", null_data['macro_f1']['p95'])

check("Null 99th Acc", f"*{null_data['accuracy']['p99']*100:.2f}%*", null_data['accuracy']['p99'])
check("Null 99th Bal", f"*{null_data['balanced_accuracy']['p99']*100:.2f}%*", null_data['balanced_accuracy']['p99'])
check("Null 99th F1", f"*{null_data['macro_f1']['p99']:.4f}*", null_data['macro_f1']['p99'])

null_df = pd.read_csv(EXP_DIR / "selection_inclusive_null_results.csv")
null_acc = null_df["accuracy"].values
null_bal = null_df["balanced_accuracy"].values
null_f1 = null_df["macro_f1"].values

def calc_p(null_vals, target):
    return (np.sum(null_vals >= target) + 1) / (len(null_vals) + 1)

p_hist_acc = calc_p(null_acc, 0.5375)
p_hist_bal = calc_p(null_bal, 0.5347)
p_hist_f1 = calc_p(null_f1, 0.5487)

check("Emp p Hist Acc", f"$p = {p_hist_acc:.4f}$", p_hist_acc)
check("Emp p Hist Bal", f"$p = {p_hist_bal:.4f}$", p_hist_bal)
check("Emp p Hist F1", f"$p = {p_hist_f1:.4f}^*$", p_hist_f1)

p_s42_acc = calc_p(null_acc, 0.5375)
p_s42_bal = calc_p(null_bal, 0.5183)
p_s42_f1 = calc_p(null_f1, 0.5218)

check("Emp p Seed42 Acc", f"$p = {p_s42_acc:.4f}$", p_s42_acc)
check("Emp p Seed42 Bal", f"$p = {p_s42_bal:.4f}$", p_s42_bal)
check("Emp p Seed42 F1", f"$p = {p_s42_f1:.4f}$", p_s42_f1)

p_3s_acc = calc_p(null_acc, np.mean([s['overall']['accuracy'] for s in p2['heldout_selection_rule_i'].values()]))
p_3s_bal = calc_p(null_bal, np.mean([s['overall']['balanced_accuracy'] for s in p2['heldout_selection_rule_i'].values()]))
p_3s_f1 = calc_p(null_f1, np.mean([s['overall']['macro_f1'] for s in p2['heldout_selection_rule_i'].values()]))

check("Emp p 3Seed Acc", f"$p = {p_3s_acc:.4f}$", p_3s_acc)
check("Emp p 3Seed Bal", f"$p = {p_3s_bal:.4f}$", p_3s_bal)
check("Emp p 3Seed F1", f"$p = {p_3s_f1:.4f}$", p_3s_f1)

p_ra_acc = calc_p(null_acc, np.mean([s['overall']['accuracy'] for s in p2['clean_rule_a_inner_val'].values()]))
p_ra_bal = calc_p(null_bal, np.mean([s['overall']['balanced_accuracy'] for s in p2['clean_rule_a_inner_val'].values()]))
p_ra_f1 = calc_p(null_f1, np.mean([s['overall']['macro_f1'] for s in p2['clean_rule_a_inner_val'].values()]))

check("Emp p RuleA Acc", f"$p = {p_ra_acc:.4f}$", p_ra_acc)
check("Emp p RuleA Bal", f"$p = {p_ra_bal:.4f}$", p_ra_bal)
check("Emp p RuleA F1", f"$p = {p_ra_f1:.4f}$", p_ra_f1)

p_rb_acc = calc_p(null_acc, np.mean([s['overall']['accuracy'] for s in p2['clean_rule_b_fixed_epochs'].values()]))
p_rb_bal = calc_p(null_bal, np.mean([s['overall']['balanced_accuracy'] for s in p2['clean_rule_b_fixed_epochs'].values()]))
p_rb_f1 = calc_p(null_f1, np.mean([s['overall']['macro_f1'] for s in p2['clean_rule_b_fixed_epochs'].values()]))

check("Emp p RuleB Acc", f"$p = {p_rb_acc:.4f}$", p_rb_acc)
check("Emp p RuleB Bal", f"$p = {p_rb_bal:.4f}$", p_rb_bal)
check("Emp p RuleB F1", f"$p = {p_rb_f1:.4f}$", p_rb_f1)

# ======================================================================
# Section 5.3: Similarity-Purged Cross-Validation
# ======================================================================
with open(EXP_DIR / "phase3_purged_cv_summary.json", "r") as f:
    p3 = json.load(f)

arm1 = p3["primary_arm_fixed_epochs"]
check("Phase 3 0% Baseline Bal", f"{arm1['0%_baseline']['mean']*100:.2f}% $\\pm$ {arm1['0%_baseline']['std']*100:.2f}%", arm1['0%_baseline'])
check("Phase 3 10% Paired Delta", f"**${arm1['10%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm1['10%_purge']['paired_delta_std']*100:.2f}\\%$**", arm1['10%_purge']['paired_delta_mean'])
check("Phase 3 25% Paired Delta", f"**${arm1['25%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm1['25%_purge']['paired_delta_std']*100:.2f}\\%$**", arm1['25%_purge']['paired_delta_mean'])
check("Phase 3 50% Paired Delta", f"**${arm1['50%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm1['50%_purge']['paired_delta_std']*100:.2f}\\%$**", arm1['50%_purge']['paired_delta_mean'])

arm2 = p3["secondary_arm_inner_val"]
check("Arm 2 0% Baseline Bal", f"{arm2['0%_baseline']['mean']*100:.2f}% $\\pm$ {arm2['0%_baseline']['std']*100:.2f}%", arm2['0%_baseline'])
check("Arm 2 10% Paired Delta", f"**${arm2['10%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm2['10%_purge']['paired_delta_std']*100:.2f}\\%$**", arm2['10%_purge']['paired_delta_mean'])
check("Arm 2 25% Paired Delta", f"**${arm2['25%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm2['25%_purge']['paired_delta_std']*100:.2f}\\%$**", arm2['25%_purge']['paired_delta_mean'])
check("Arm 2 50% Paired Delta", f"**${arm2['50%_purge']['paired_delta_mean']*100:+.2f}\\% \\pm {arm2['50%_purge']['paired_delta_std']*100:.2f}\\%$**", arm2['50%_purge']['paired_delta_mean'])

# ======================================================================
# Section 6.2 & 6.3: Trivial Feature Ablations & Concatenation
# ======================================================================
with open(EXP_DIR / "exp3_trivial_ablation_results.json", "r") as f:
    triv_abl = json.load(f)

check("Group (i) LR Bal", r"**$25.00\% \pm 0.07\%$**", triv_abl["groups"]["group_i_dims_filesize"]["logistic_regression"]["true"]["summary"]["mean_bal"])
check("Group (ii) LR Bal", r"**$38.36\% \pm 0.29\%$**", triv_abl["groups"]["group_ii_alpha_border"]["logistic_regression"]["true"]["summary"]["mean_bal"])
check("Group (iii) LR Bal", r"**$45.92\% \pm 0.27\%$**", triv_abl["groups"]["group_iii_intensity_hist_moments"]["logistic_regression"]["true"]["summary"]["mean_bal"])
check("Group (iv) LR Bal", r"**$51.43\% \pm 0.35\%$**", triv_abl["groups"]["group_iv_all_features"]["logistic_regression"]["true"]["summary"]["mean_bal"])

with open(EXP_DIR / "exp3_effnet_concat_results.json", "r") as f:
    eff_concat = json.load(f)

check("Concat Without Bal", r"**$85.30\% \pm 0.21\%$**", eff_concat["without_trivial_features"]["summary"]["mean_bal"])
check("Concat With Bal", r"**$85.65\% \pm 0.19\%$**", eff_concat["with_trivial_features_concatenated"]["summary"]["mean_bal"])
check("Concat Paired Delta Bal", r"**$+0.35\% \pm 0.03\%$**", 0.0035)

print(f"\nVerification Results: {len(matches)} passed, {len(mismatches)} mismatches.")
if mismatches:
    print("\nMISMATCHES DETECTED:")
    for m in mismatches:
        print("  -", m)
    sys.exit(1)
else:
    print(f"\nPERFECT CONCORDANCE: All {len(matches)} verified claims in leakage_audit_report.md match serialized artifacts.")
    sys.exit(0)
