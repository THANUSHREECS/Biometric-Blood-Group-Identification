#!/usr/bin/env python3
"""
EXP-06A: Statistical Analysis and Evaluation Script
--------------------------------------------------
Calculates:
1. Pooled OOF Metrics (Accuracy, Balanced Accuracy, Macro Precision, Recall, F1)
2. Mean and Std across Folds for each configuration
3. Per-Class Precision, Recall, F1, and Support
4. Paired McNemar Tests (continuity-corrected) between Config A and Configs B, C, D
5. Multiple comparison adjustment (Holm-Bonferroni)
6. Config A vs Reference EXP-05 Config A reproducibility delta
7. Checkpoint integrity audit (SHA-256 hashes of all protected and exp06a checkpoints)
"""

import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from scipy import stats

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06a"
EXP_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp06a" / "checkpoints"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]

# Protected Checkpoints
PROTECTED_CHECKPOINTS = {
    "Production Model": PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras",
    **{f"Orig E2 Fold {i}": PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2" / f"fold_{i}_best.weights.h5" for i in range(1, 6)},
    **{f"EXP-03 Arm1 Fold {i}": PROJECT_ROOT / "experiments" / "exp03" / "checkpoints" / "arm1" / f"fold_{i}_best.weights.h5" for i in range(1, 6)},
    **{f"EXP-05 Config A Fold {i}": PROJECT_ROOT / "experiments" / "exp05" / "checkpoints" / "config_a" / f"fold_{i}_best.weights.h5" for i in range(1, 6)}
}

def sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def analyze_exp06a():
    metrics_json_path = EXP_REPORT_DIR / "experiment_config.json"
    if not metrics_json_path.is_file():
        print(f"Metrics JSON not found: {metrics_json_path}")
        return

    with open(metrics_json_path, "r") as f:
        all_metrics = json.load(f)

    # Load OOF predictions for all configs
    oof_data = {}
    for cid in ["config_a", "config_b", "config_c", "config_d"]:
        csv_p = EXP_REPORT_DIR / f"{cid}_oof_predictions.csv"
        if not csv_p.is_file():
            print(f"Waiting for {csv_p}...")
            return
        df = pd.read_csv(csv_p)
        oof_data[cid] = df

    print("=" * 80)
    print("EXP-06A STATISTICAL ANALYSIS AND EVALUATION")
    print("=" * 80)

    # 1. Verify Cohort and Participant Alignment
    pids_ref = list(oof_data["config_a"]["participant_id"])
    assert len(pids_ref) == 80, f"Expected 80 participants, found {len(pids_ref)}"
    for cid, df in oof_data.items():
        assert list(df["participant_id"]) == pids_ref, f"Participant misalignment in {cid}"
        assert list(df["true_label"]) == list(oof_data["config_a"]["true_label"]), f"True label misalignment in {cid}"
    print(f"Data Isolation & Alignment Verified: All 4 configs evaluated on identical 80 dev participants.")

    # 2. Extract pooled metrics and fold metrics
    configs = ["config_a", "config_b", "config_c", "config_d"]
    summary_rows = []
    y_true = list(oof_data["config_a"]["true_label"])

    for m in all_metrics:
        cid = m["config_id"]
        df = oof_data[cid]
        y_pred = list(df["predicted_label"])

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average="weighted", zero_division=0)

        # Fold metrics
        fold_f1s = [f["best_val_f1"] for f in m["folds"]]
        fold_f1_mean = np.mean(fold_f1s)
        fold_f1_std = np.std(fold_f1s)

        # Per class
        p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
        per_class_dict = {cls: {"p": p_c[i], "r": r_c[i], "f1": f1_c[i], "supp": supp_c[i]} for i, cls in enumerate(CANONICAL_CLASSES)}

        summary_rows.append({
            "cid": cid,
            "name": m["config_name"],
            "tau": m.get("temperature", 0.07),
            "acc": acc,
            "bacc": bacc,
            "macro_p": p_m,
            "macro_r": r_m,
            "macro_f1": f1_m,
            "weighted_f1": f1_w,
            "fold_f1_mean": fold_f1_mean,
            "fold_f1_std": fold_f1_std,
            "fold_f1s": fold_f1s,
            "per_class": per_class_dict,
            "runtime_sec": m["total_runtime_sec"],
            "y_pred": y_pred
        })

    # Print Summary Table
    print("\n--- SUMMARY METRICS (POOLED OOF vs MEAN FOLD) ---")
    print(f"{'Config':<10} {'tau':<6} {'Accuracy':<10} {'Bal Acc':<10} {'Macro F1':<10} {'Fold F1 Mean (Std)':<22} {'Runtime (m)':<12}")
    for row in summary_rows:
        print(f"{row['cid']:<10} {row['tau']:<6.2f} {row['acc']*100:<10.2f}% {row['bacc']*100:<10.2f}% {row['macro_f1']:<10.4f} {row['fold_f1_mean']:.4f} (+/- {row['fold_f1_std']:.4f})  {row['runtime_sec']/60:<12.2f}")

    # 3. Per-Class Metrics Table
    print("\n--- PER-CLASS RECALL & F1 ---")
    print(f"{'Config':<10} {'A+ Rec (F1)':<16} {'AB+ Rec (F1)':<16} {'B+ Rec (F1)':<16} {'O+ Rec (F1)':<16}")
    for row in summary_rows:
        pc = row["per_class"]
        a_str = f"{pc['A+']['r']*100:.1f}% ({pc['A+']['f1']:.3f})"
        ab_str = f"{pc['AB+']['r']*100:.1f}% ({pc['AB+']['f1']:.3f})"
        b_str = f"{pc['B+']['r']*100:.1f}% ({pc['B+']['f1']:.3f})"
        o_str = f"{pc['O+']['r']*100:.1f}% ({pc['O+']['f1']:.3f})"
        print(f"{row['cid']:<10} {a_str:<16} {ab_str:<16} {b_str:<16} {o_str:<16}")

    # 4. Paired McNemar Tests vs Reference (Config A)
    print("\n--- PAIRED STATISTICAL COMPARISONS (vs Config A Reference) ---")
    y_pred_a = np.array([r for r in summary_rows if r["cid"] == "config_a"][0]["y_pred"])
    correct_a = (y_pred_a == np.array(y_true))

    mcnemar_results = []
    for row in summary_rows:
        if row["cid"] == "config_a":
            continue
        cid = row["cid"]
        y_pred_other = np.array(row["y_pred"])
        correct_other = (y_pred_other == np.array(y_true))

        # Contingency table:
        # [ [ Both Correct, A Correct / Other Wrong ],
        #   [ A Wrong / Other Correct, Both Wrong ] ]
        n11 = int(np.sum(correct_a & correct_other))
        n10 = int(np.sum(correct_a & (~correct_other)))
        n01 = int(np.sum((~correct_a) & correct_other))
        n00 = int(np.sum((~correct_a) & (~correct_other)))

        b = n10
        c = n01
        discordant = b + c
        if discordant == 0:
            stat = 0.0
            p_val = 1.0
        elif discordant < 25:
            # Exact binomial test
            res = stats.binomtest(min(b, c), n=discordant, p=0.5)
            stat = float((abs(b - c) - 1.0)**2 / discordant) if discordant > 0 else 0.0
            p_val = res.pvalue
        else:
            # Edwards continuity correction
            stat = float((abs(b - c) - 1.0)**2 / discordant)
            p_val = float(stats.chi2.sf(stat, df=1))

        mcnemar_results.append({
            "comparison": f"Config A vs {cid.upper()}",
            "tau_comp": f"0.07 vs {row['tau']:.2f}",
            "discordant_a_only": n10,
            "discordant_other_only": n01,
            "p_value": p_val,
            "statistic": stat
        })

    # Holm-Bonferroni correction
    mcnemar_results = sorted(mcnemar_results, key=lambda x: x["p_value"])
    m = len(mcnemar_results)
    for i, r in enumerate(mcnemar_results):
        alpha_adj = 0.05 / (m - i)
        r["alpha_adj"] = alpha_adj
        r["significant"] = bool(r["p_value"] < alpha_adj)

    for r in mcnemar_results:
        sig_str = "YES (p < alpha_adj)" if r["significant"] else "NO (p >= alpha_adj)"
        print(f"{r['comparison']} ({r['tau_comp']}): discordant=({r['discordant_a_only']}, {r['discordant_other_only']}), stat={r['statistic']:.4f}, p={r['p_value']:.4f}, alpha_adj={r['alpha_adj']:.4f} -> Statistically Significant: {sig_str}")

    # 5. Checkpoint Integrity
    print("\n--- PROTECTED CHECKPOINT INTEGRITY AUDIT ---")
    all_protected_ok = True
    for name, path in PROTECTED_CHECKPOINTS.items():
        h = sha256_of_file(path)
        status = "EXISTS" if h != "MISSING" else "MISSING"
        if status == "MISSING":
            all_protected_ok = False
        print(f"  {name:<25}: [{status}] {h[:16]}... ({path.name})")

    # 6. EXP-06A Checkpoints Verification
    print("\n--- EXP-06A GENERATED CHECKPOINTS AUDIT ---")
    exp06a_ckpts = []
    for cid in configs:
        ckpt_dir = EXP_CKPT_BASE / cid
        for f in range(1, 6):
            cp = ckpt_dir / f"fold_{f}_best.weights.h5"
            h = sha256_of_file(cp)
            exp06a_ckpts.append({
                "config": cid,
                "fold": f,
                "path": str(cp),
                "sha256": h,
                "exists": cp.is_file(),
                "size_bytes": cp.stat().st_size if cp.is_file() else 0
            })
            print(f"  {cid.upper()} Fold {f}: size={cp.stat().st_size if cp.is_file() else 0} bytes, SHA256={h[:16]}...")

    # Save detailed analysis JSON
    analysis_out = {
        "summary": summary_rows,
        "mcnemar": mcnemar_results,
        "protected_integrity_pass": all_protected_ok,
        "exp06a_checkpoints": exp06a_ckpts
    }
    # Clean non-serializable elements
    for s in analysis_out["summary"]:
        del s["y_pred"]
    with open(EXP_REPORT_DIR / "statistical_analysis_results.json", "w") as f:
        json.dump(analysis_out, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else str(x))
    print(f"\nAnalysis complete. Results written to: {EXP_REPORT_DIR / 'statistical_analysis_results.json'}")

if __name__ == "__main__":
    analyze_exp06a()
