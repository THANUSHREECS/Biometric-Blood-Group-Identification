import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_ROOT / "model_comparison_results"

GT_CSV = DATA_ROOT / "mfs100_ground_truth.csv"
EFFNET_PRED_CSV = RESULTS_DIR / "effnet_predictions.csv"
E2_PRED_CSV = RESULTS_DIR / "e2_predictions.csv"
COMBINED_PRED_CSV = RESULTS_DIR / "combined_predictions.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"

print("=" * 60)
print("AUDIT RUN START")
print("=" * 60)

# Check 1: 26 participants, 3 scans each, 78 scans total
df_gt = pd.read_csv(GT_CSV)
total_scans = len(df_gt)
unique_pids = df_gt["subject_id"].unique()
num_pids = len(unique_pids)
scans_per_pid = df_gt.groupby("subject_id").size()

print(f"Check 1: Total scans in CSV = {total_scans}, Unique participants = {num_pids}")
check1_passed = (total_scans == 78) and (num_pids == 26) and (scans_per_pid.min() == 3) and (scans_per_pid.max() == 3)
print(f"  Check 1 Passed: {check1_passed}")

# Check 2: Image existence and consistent confirmed_blood_group per participant
missing_files = []
inconsistent_labels = []

for pid, group in df_gt.groupby("subject_id"):
    labels = group["confirmed_blood_group"].unique()
    if len(labels) > 1:
        inconsistent_labels.append((pid, labels))
    for fname in group["capture_filename"]:
        fpath = DATA_ROOT / "captured" / fname
        if not fpath.is_file():
            missing_files.append(str(fpath))

print(f"Check 2: Missing files count = {len(missing_files)}, Inconsistent labels count = {len(inconsistent_labels)}")
check2_passed = (len(missing_files) == 0) and (len(inconsistent_labels) == 0)
print(f"  Check 2 Passed: {check2_passed}")

# Check 4 & 9: Verify prediction CSVs
CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def audit_pred_csv(csv_path, name):
    df = pd.read_csv(csv_path)
    count = len(df)
    unique_ids = df["participant_id"].nunique()
    print(f"Check 9 ({name}): Rows = {count}, Unique participants = {unique_ids}")
    
    # Check probabilities format and length
    probs_valid = True
    argmax_valid = True
    for _, row in df.iterrows():
        probs = json.loads(row["probabilities"])
        if len(probs) != 4:
            probs_valid = False
        pred = CLASSES[np.argmax(probs)]
        if pred != row["predicted_label"]:
            argmax_valid = False
            
    return df, (count == 26 and unique_ids == 26 and probs_valid and argmax_valid)

df_eff, check9_eff = audit_pred_csv(EFFNET_PRED_CSV, "EffNet")
df_e2, check9_e2 = audit_pred_csv(E2_PRED_CSV, "E2")
df_comb, check9_comb = audit_pred_csv(COMBINED_PRED_CSV, "Combined")

check9_passed = check9_eff and check9_e2 and check9_comb
print(f"  Check 9 Passed: {check9_passed}")

# Check 3: Participant-level aggregation verification
# Compare Combined probs with (EffNet probs + E2 probs)/2
comb_aligned = True
for pid in unique_pids:
    row_eff = df_eff[df_eff["participant_id"] == pid].iloc[0]
    row_e2 = df_e2[df_e2["participant_id"] == pid].iloc[0]
    row_comb = df_comb[df_comb["participant_id"] == pid].iloc[0]
    
    p_eff = np.array(json.loads(row_eff["probabilities"]))
    p_e2 = np.array(json.loads(row_e2["probabilities"]))
    p_comb = np.array(json.loads(row_comb["probabilities"]))
    
    expected_comb = (p_eff + p_e2) / 2.0
    if np.max(np.abs(expected_comb - p_comb)) > 1e-4:
        comb_aligned = False
        print(f"  Mismatch for {pid}: expected {expected_comb}, got {p_comb}")

print(f"Check 3: Combined ensemble equals exactly (EffNet + E2)/2: {comb_aligned}")

# Check 5 & 10: Recalculate metrics independently and compare with metrics.json
with open(METRICS_JSON, "r") as f:
    saved_metrics = json.load(f)

def calc_metrics(df):
    y_true = [CLASS_TO_IDX[c] for c in df["true_label"]]
    y_pred = [CLASS_TO_IDX[c] for c in df["predicted_label"]]
    
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    per_class_rec = {}
    for i, c in enumerate(CLASSES):
        # true positives / support
        support = sum(1 for yt in y_true if yt == i)
        tp = cm[i, i]
        per_class_rec[c] = tp / support if support > 0 else 0.0
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "per_class_recall": per_class_rec,
        "confusion_matrix": cm.tolist()
    }

recalc_eff = calc_metrics(df_eff)
recalc_e2 = calc_metrics(df_e2)
recalc_comb = calc_metrics(df_comb)

print("\n--- INDEPENDENTLY RECALCULATED METRICS ---")
for name, m in [("EffNet", recalc_eff), ("E2", recalc_e2), ("Combined", recalc_comb)]:
    print(f"\n[{name}]")
    print(f"  Accuracy:          {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
    print(f"  Macro Precision:   {m['macro_precision']:.4f}")
    print(f"  Macro Recall:      {m['macro_recall']:.4f}")
    print(f"  Macro F1:          {m['macro_f1']:.4f}")
    print(f"  Per-class Recall:  {m['per_class_recall']}")
    print(f"  Confusion Matrix:\n{np.array(m['confusion_matrix'])}")

# Check 10: Compare with metrics.json
check10_passed = True
for name, recalc, saved_key in [("EffNet", recalc_eff, "effnet"), ("E2", recalc_e2, "e2"), ("Combined", recalc_comb, "combined")]:
    sm = saved_metrics[saved_key]
    if abs(recalc["accuracy"] - sm["accuracy"]) > 1e-4:
        check10_passed = False
    if abs(recalc["macro_precision"] - sm["macro_precision"]) > 1e-4:
        check10_passed = False
    if abs(recalc["macro_recall"] - sm["macro_recall"]) > 1e-4:
        check10_passed = False
    if abs(recalc["macro_f1"] - sm["macro_f1"]) > 1e-4:
        check10_passed = False
    if recalc["confusion_matrix"] != sm["confusion_matrix"]:
        check10_passed = False

print(f"\nCheck 10: Generated metrics match independently recalculated metrics: {check10_passed}")

# Check 6: Always-O+ Baseline recalculation
# In df_gt:
labels_per_pid = df_gt.groupby("subject_id")["confirmed_blood_group"].first()
class_counts = labels_per_pid.value_counts()
print(f"\nCohort Class Distribution (26 participants):")
print(class_counts)
o_plus_count = class_counts.get("O+", 0)
baseline_o_plus = o_plus_count / len(labels_per_pid)
print(f"Always-O+ Baseline Accuracy = {o_plus_count}/26 = {baseline_o_plus:.4f} ({baseline_o_plus*100:.2f}%)")

# Save recalculation results to json
with open(PROJECT_ROOT / "scratch" / "audit_metrics_recalculated.json", "w") as f:
    json.dump({
        "effnet": recalc_eff,
        "e2": recalc_e2,
        "combined": recalc_comb,
        "always_o_plus_baseline": baseline_o_plus,
        "class_distribution": class_counts.to_dict()
    }, f, indent=2)

print("\nAudit calculations saved to scratch/audit_metrics_recalculated.json")
