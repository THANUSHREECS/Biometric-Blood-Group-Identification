#!/usr/bin/env python3
import json
import csv
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)
from scipy.stats import binomtest

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06d"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]

def run_audit():
    print("=" * 80)
    print("INDEPENDENT POST-EXPERIMENT AUDIT OF EXP-06D")
    print("=" * 80)

    # 1. Load OOF Predictions
    csv_a = EXP_DIR / "config_a_oof_predictions.csv"
    csv_b = EXP_DIR / "config_b_oof_predictions.csv"
    assert csv_a.is_file(), f"Missing {csv_a}"
    assert csv_b.is_file(), f"Missing {csv_b}"

    df_a = pd.read_csv(csv_a).sort_values("participant_id").reset_index(drop=True)
    df_b = pd.read_csv(csv_b).sort_values("participant_id").reset_index(drop=True)

    # Participant counts
    print(f"Config A rows: {len(df_a)}, unique participant_ids: {df_a['participant_id'].nunique()}")
    print(f"Config B rows: {len(df_b)}, unique participant_ids: {df_b['participant_id'].nunique()}")

    # Development cohort validation against manifest
    df_split = pd.read_csv(SPLIT_CSV)
    approved_dev_pids = sorted(df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]["participant_id"].unique())
    locked_test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())

    pids_a = df_a["participant_id"].tolist()
    pids_b = df_b["participant_id"].tolist()

    assert pids_a == approved_dev_pids, "Config A participant IDs mismatch approved dev manifest!"
    assert pids_b == approved_dev_pids, "Config B participant IDs mismatch approved dev manifest!"
    assert pids_a == pids_b, "Config A and Config B participant IDs do not match!"
    print("PASS: Both configurations contain exactly the 80 approved development participants in identical order.")

    # Quarantine validation
    leak_test_a = set(pids_a).intersection(locked_test_pids)
    leak_test_b = set(pids_b).intersection(locked_test_pids)
    assert len(leak_test_a) == 0, f"Config A leaked locked test cohort: {leak_test_a}"
    assert len(leak_test_b) == 0, f"Config B leaked locked test cohort: {leak_test_b}"
    print("PASS: Zero locked test cohort participants present in either configuration.")

    # True labels match
    lbl_match = (df_a["true_label"] == df_b["true_label"]).all()
    assert lbl_match, "True labels do not match between Config A and Config B!"
    print("PASS: True blood group labels match 100% between configurations.")

    # Fold assignment consistency
    folds_match = (df_a["fold_id"] == df_b["fold_id"]).all()
    assert folds_match, "Fold assignments do not match between configurations!"
    print("PASS: Fold assignments match 100% between Config A and Config B.")

    # Validate probability vectors
    for name, df in [("Config A", df_a), ("Config B", df_b)]:
        for idx, row in df.iterrows():
            p_canon = json.loads(row["canonical_probabilities"])
            assert len(p_canon) == 4, f"[{name}] Prob vector length != 4 at row {idx}"
            assert all(math.isfinite(x) for x in p_canon), f"[{name}] Non-finite prob at row {idx}"
            assert all(x >= 0.0 for x in p_canon), f"[{name}] Negative prob at row {idx}"
            assert abs(sum(p_canon) - 1.0) < 1e-4, f"[{name}] Prob sum != 1.0 at row {idx} (sum={sum(p_canon)})"
            
            # Check argmax matches predicted_label
            pred_idx = int(np.argmax(p_canon))
            expected_lbl = CANONICAL_CLASSES[pred_idx]
            assert expected_lbl == row["predicted_label"], f"[{name}] Label mismatch at row {idx}: {expected_lbl} != {row['predicted_label']}"
    print("PASS: All probability vectors are finite, non-negative, sum to 1.0, and argmax matches predicted_label.")

    # 2. Recompute Metrics
    recomputed = {}
    for name, cid, df in [("Config A (Enhanced C5)", "config_a", df_a), ("Config B (Raw Platen C5)", "config_b", df_b)]:
        y_true = df["true_label"].tolist()
        y_pred = df["predicted_label"].tolist()

        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )
        prec_c, rec_c, f1_c, supp_c = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

        fold_f1s = []
        for f in range(1, 6):
            df_f = df[df["fold_id"] == f]
            f_f1 = float(f1_score(df_f["true_label"], df_f["predicted_label"], labels=CANONICAL_CLASSES, average="macro", zero_division=0))
            fold_f1s.append(f_f1)

        f1_mean = float(np.mean(fold_f1s))
        f1_std = float(np.std(fold_f1s))

        recomputed[cid] = {
            "name": name,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_precision": float(prec_m),
            "macro_recall": float(rec_m),
            "macro_f1": float(f1_m),
            "fold_f1s": fold_f1s,
            "fold_f1_mean": f1_mean,
            "fold_f1_std": f1_std,
            "per_class": {
                c: {
                    "precision": float(prec_c[i]),
                    "recall": float(rec_c[i]),
                    "f1": float(f1_c[i]),
                    "support": int(supp_c[i]),
                    "correct": int(cm[i, i])
                } for i, c in enumerate(CANONICAL_CLASSES)
            },
            "confusion_matrix": cm.tolist()
        }

        print(f"\n--- {name} RECOMPUTED METRICS ---")
        print(f"Accuracy:          {acc*100:.2f}% ({int(np.trace(cm))}/80)")
        print(f"Balanced Accuracy: {bacc*100:.2f}%")
        print(f"Macro Precision:   {prec_m:.4f}")
        print(f"Macro Recall:      {rec_m:.4f}")
        print(f"Macro F1 (OOF):    {f1_m:.4f}")
        print(f"Fold F1 Mean:      {f1_mean:.4f} (+/- {f1_std:.4f})")
        print(f"Fold F1s:          {[round(x, 4) for x in fold_f1s]}")
        print(f"Confusion Matrix (Order: {CANONICAL_CLASSES}):\n{cm}")
        for c in CANONICAL_CLASSES:
            pc = recomputed[cid]["per_class"][c]
            print(f"  {c:<4} Prec: {pc['precision']*100:.2f}%, Rec: {pc['recall']*100:.2f}%, F1: {pc['f1']:.4f}, Correct: {pc['correct']}/{pc['support']}")

    # 3. Statistical Analysis Recomputation
    corr_a = (df_a["predicted_label"] == df_a["true_label"]).to_numpy()
    corr_b = (df_b["predicted_label"] == df_b["true_label"]).to_numpy()

    n11 = int(np.sum(corr_a & corr_b))
    n10 = int(np.sum(corr_a & (~corr_b)))
    n01 = int(np.sum((~corr_a) & corr_b))
    n00 = int(np.sum((~corr_a) & (~corr_b)))

    print("\n--- PAIRED CONTINGENCY TABLE ---")
    print(f"n11 (Both Correct):            {n11}")
    print(f"n10 (Config A only correct):    {n10}")
    print(f"n01 (Config B only correct):    {n01}")
    print(f"n00 (Both Wrong):              {n00}")
    print(f"Total Discordant (n10 + n01):  {n10 + n01}")

    b_res = binomtest(n10, n10 + n01, 0.5)
    p_exact = float(b_res.pvalue)
    print(f"Exact Binomial Two-Sided p-value: {p_exact:.6f}")

    # Paired Bootstrap CI (2000 iterations, seed=42)
    rng = np.random.default_rng(42)
    N = len(df_a)
    boot_diffs = []
    for _ in range(2000):
        b_idx = rng.choice(N, size=N, replace=True)
        yt_b = df_a["true_label"].iloc[b_idx].tolist()
        pa_b = df_a["predicted_label"].iloc[b_idx].tolist()
        pb_b = df_b["predicted_label"].iloc[b_idx].tolist()
        f1_a_boot = f1_score(yt_b, pa_b, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        f1_b_boot = f1_score(yt_b, pb_b, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        boot_diffs.append(f1_b_boot - f1_a_boot)

    ci95_low, ci95_high = np.percentile(boot_diffs, [2.5, 97.5])
    boot_mean = float(np.mean(boot_diffs))
    print(f"Paired Bootstrap Delta Macro F1 Mean: {boot_mean:.4f}")
    print(f"Paired Bootstrap 95% CI: [{ci95_low:.4f}, {ci95_high:.4f}]")

    # 4. Compare with Saved JSON & CSV
    with open(EXP_DIR / "statistical_analysis_results.json", "r") as f:
        saved_stats = json.load(f)

    assert saved_stats["contingency_table"]["n11"] == n11
    assert saved_stats["contingency_table"]["n10"] == n10
    assert saved_stats["contingency_table"]["n01"] == n01
    assert saved_stats["contingency_table"]["n00"] == n00
    assert math.isclose(saved_stats["exact_binomial_two_sided_p"], p_exact, abs_tol=1e-6)
    assert math.isclose(saved_stats["bootstrap_f1_diff_mean"], boot_mean, abs_tol=1e-4)
    assert math.isclose(saved_stats["bootstrap_f1_diff_ci95"][0], ci95_low, abs_tol=1e-4)
    assert math.isclose(saved_stats["bootstrap_f1_diff_ci95"][1], ci95_high, abs_tol=1e-4)
    print("PASS: Statistical analysis results match saved JSON 100% exactly.")

    # 5. Checkpoint Integrity Re-verification (66 models)
    pre_hashes_file = PROJECT_ROOT / "scratch" / "exp06d_pre_preparation_hashes.json"
    with open(pre_hashes_file, "r") as f:
        pre_hashes = json.load(f)

    mismatches = 0
    missing = 0
    for name, info in pre_hashes.items():
        p = Path(info["path"])
        if not p.is_file():
            missing += 1
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                h.update(chunk)
        if h.hexdigest() != info["sha256"]:
            mismatches += 1

    print(f"Protected Checkpoints Verified: {len(pre_hashes)} | Missing: {missing} | Mismatches: {mismatches}")
    assert missing == 0 and mismatches == 0, "CHECKPOINT CORRUPTION DETECTED!"

    # 6. Check TRAINING_AUTHORIZED flag in run_exp06d.py
    run_script = PROJECT_ROOT / "experiments" / "exp06d" / "run_exp06d.py"
    with open(run_script, "r") as f:
        content = f.read()
    assert "TRAINING_AUTHORIZED = False" in content, "TRAINING_AUTHORIZED is not False in run_exp06d.py!"
    print("PASS: TRAINING_AUTHORIZED is confirmed False in run_exp06d.py.")

    print("\nALL POST-EXPERIMENT AUDIT CHECKS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    run_audit()
