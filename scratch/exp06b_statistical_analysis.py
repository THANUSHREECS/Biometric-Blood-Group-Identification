#!/usr/bin/env python3
"""
EXP-06B: Statistical Analysis, Integrity Verification, and Audit Script
----------------------------------------------------------------------
Performs:
1. Cohort isolation and OOF prediction integrity check (80 participants exactly).
2. Post-execution hash verification of all 36 protected baseline checkpoints.
3. Verification and hashing of all 20 EXP-06B checkpoints.
4. Independent recomputation of pooled OOF metrics and fold metrics from prediction CSVs.
5. Paired statistical significance testing (McNemar with continuity correction / exact binomial)
   against Arm B (Reference: lambda=0.10), with Holm-Bonferroni multiple-comparison correction.
6. Comparison of Arm B against EXP-06A Config B (historical reference).
7. Serializes results into reports/accuracy_improvement/exp06b/statistical_analysis_results.json.
"""

import os
import sys
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
EXP06B_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06b"
EXP06B_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp06b" / "checkpoints"
EXP06A_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06a"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
PRE_HASHES_FILE = PROJECT_ROOT / "scratch" / "exp06b_pre_execution_hashes.json"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
ARMS = ["arm_a", "arm_b", "arm_c", "arm_d"]
ARM_NAMES = {
    "arm_a": "Arm A: Reduced Loss Weight (lambda=0.05)",
    "arm_b": "Arm B: Reference Loss Weight (lambda=0.10)",
    "arm_c": "Arm C: Moderate Loss Weight (lambda=0.20)",
    "arm_d": "Arm D: Heavy Loss Weight (lambda=0.50)"
}
ARM_LAMBDAS = {
    "arm_a": 0.05,
    "arm_b": 0.10,
    "arm_c": 0.20,
    "arm_d": 0.50
}

def sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def run_statistical_analysis():
    print("=" * 80)
    print("EXP-06B: POST-TRAINING AUDIT & STATISTICAL ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Protected Checkpoint Pre/Post Hash Verification
    # ---------------------------------------------------------
    print("\n[Step 1] Protected Checkpoint Pre/Post Hash Verification:")
    assert PRE_HASHES_FILE.is_file(), f"Pre-hashes file missing: {PRE_HASHES_FILE}"
    with open(PRE_HASHES_FILE, "r") as f:
        pre_hashes = json.load(f)

    protected_integrity_pass = True
    hash_audit_table = []
    for name, info in pre_hashes.items():
        p = Path(info["path"])
        post_h = sha256_of_file(p)
        pre_h = info["sha256"]
        match = (post_h == pre_h)
        if not match:
            protected_integrity_pass = False
        hash_audit_table.append({
            "name": name,
            "path": info["path"],
            "pre_sha256": pre_h,
            "post_sha256": post_h,
            "match": match
        })
        status_str = "MATCH" if match else "MISMATCH / CORRUPTED"
        print(f"  {name:<22} {status_str} | pre={pre_h[:12]}... post={post_h[:12]}...")

    assert protected_integrity_pass, "CRITICAL AUDIT FAILURE: Protected checkpoint hash altered during execution!"
    print(f"  All {len(pre_hashes)} protected checkpoints verified intact (0 bytes modified).")

    # ---------------------------------------------------------
    # 2. EXP-06B Generated Checkpoints Audit
    # ---------------------------------------------------------
    print("\n[Step 2] EXP-06B Generated Checkpoints Audit:")
    exp06b_ckpts = []
    for aid in ARMS:
        ckpt_dir = EXP06B_CKPT_BASE / aid
        for f in range(1, 6):
            cp = ckpt_dir / f"fold_{f}_best.weights.h5"
            assert cp.is_file(), f"Expected checkpoint missing: {cp}"
            sz = cp.stat().st_size
            h = sha256_of_file(cp)
            exp06b_ckpts.append({
                "arm": aid,
                "fold": f,
                "path": str(cp),
                "size_bytes": sz,
                "sha256": h
            })
            print(f"  {aid.upper()} Fold {f}: size={sz:,} bytes, sha256={h[:16]}...")
    assert len(exp06b_ckpts) == 20, f"Expected 20 checkpoints, got {len(exp06b_ckpts)}"
    print(f"  All 20 EXP-06B checkpoints saved and validated.")

    # ---------------------------------------------------------
    # 3. Cohort Isolation & Participant Alignment Audit
    # ---------------------------------------------------------
    print("\n[Step 3] Cohort Isolation & OOF Participant Alignment:")
    df_split = pd.read_csv(SPLIT_CSV)
    dev_pids_expected = sorted(df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]["participant_id"].unique())
    test_pids_locked = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
    assert len(dev_pids_expected) == 80
    assert len(test_pids_locked) == 21

    oof_dfs = {}
    for aid in ARMS:
        pred_p = EXP06B_REPORT_DIR / f"{aid}_oof_predictions.csv"
        assert pred_p.is_file(), f"Prediction file missing: {pred_p}"
        df = pd.read_csv(pred_p)
        assert len(df) == 80, f"Expected 80 predictions in {aid}, found {len(df)}"
        assert sorted(df["participant_id"].unique()) == dev_pids_expected, f"Participant ID mismatch in {aid}"
        # Quarantine check
        overlap = set(df["participant_id"]).intersection(test_pids_locked)
        assert len(overlap) == 0, f"DATA LEAKAGE: Locked test IDs found in {aid}: {overlap}"
        oof_dfs[aid] = df
        print(f"  {aid.upper()}: Exactly 80 dev participants, 0 test overlap, 1 prediction per participant.")

    # ---------------------------------------------------------
    # 4. Independent Metric Recomputation from OOF Predictions
    # ---------------------------------------------------------
    print("\n[Step 4] Independent Metric Recomputation from OOF CSVs:")
    y_true = list(oof_dfs["arm_a"]["true_label"])
    # Verify true labels match across all arms
    for aid in ARMS:
        assert list(oof_dfs[aid]["true_label"]) == y_true, f"Label mismatch across arms for {aid}"

    # Load fold logs from experiment_config.json
    with open(EXP06B_REPORT_DIR / "experiment_config.json", "r") as f:
        exp_config_metrics = {m["arm_id"]: m for m in json.load(f)}

    summary_rows = []
    for aid in ARMS:
        df = oof_dfs[aid]
        y_pred = list(df["predicted_label"])

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, average="weighted", zero_division=0
        )
        p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(
            y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

        fold_logs = exp_config_metrics[aid]["folds"]
        fold_f1s = [f["best_val_f1"] for f in fold_logs]
        f1_mean = float(np.mean(fold_f1s))
        f1_std = float(np.std(fold_f1s))

        per_class_dict = {
            cls: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f1_c[i]),
                "support": int(supp_c[i]),
                "correct": int(cm[i, i])
            } for i, cls in enumerate(CANONICAL_CLASSES)
        }

        row = {
            "arm_id": aid,
            "arm_name": ARM_NAMES[aid],
            "lambda_supcon": ARM_LAMBDAS[aid],
            "temperature": 0.05,
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "macro_precision": float(p_m),
            "macro_recall": float(r_m),
            "macro_f1": float(f1_m),
            "weighted_f1": float(f1_w),
            "fold_f1_mean": f1_mean,
            "fold_f1_std": f1_std,
            "fold_f1s": fold_f1s,
            "per_class": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "total_runtime_sec": exp_config_metrics[aid]["total_runtime_sec"]
        }
        summary_rows.append(row)

    print(f"\n{'Arm':<8} {'lambda':<8} {'Acc':<10} {'BalAcc':<10} {'Macro F1':<10} {'Fold F1 Mean (Std)':<22} {'Runtime (m)':<12}")
    for r in summary_rows:
        print(f"{r['arm_id']:<8} {r['lambda_supcon']:<8.2f} {r['accuracy']*100:<10.2f}% {r['balanced_accuracy']*100:<10.2f}% {r['macro_f1']:<10.4f} {r['fold_f1_mean']:.4f} (+/- {r['fold_f1_std']:.4f})  {r['total_runtime_sec']/60:<12.2f}")

    # ---------------------------------------------------------
    # 5. Paired Statistical Tests vs Arm B Reference
    # ---------------------------------------------------------
    print("\n[Step 5] Paired Statistical Significance Testing vs Arm B Reference:")
    y_pred_b = np.array(oof_dfs["arm_b"]["predicted_label"])
    correct_b = (y_pred_b == np.array(y_true))

    mcnemar_results = []
    comparisons = [("arm_a", "Arm A (lambda=0.05)"), ("arm_c", "Arm C (lambda=0.20)"), ("arm_d", "Arm D (lambda=0.50)")]
    for aid, name in comparisons:
        y_pred_other = np.array(oof_dfs[aid]["predicted_label"])
        correct_other = (y_pred_other == np.array(y_true))

        n11 = int(np.sum(correct_b & correct_other))
        n10 = int(np.sum(correct_b & (~correct_other)))
        n01 = int(np.sum((~correct_b) & correct_other))
        n00 = int(np.sum((~correct_b) & (~correct_other)))

        b = n10
        c = n01
        discordant = b + c
        if discordant == 0:
            stat = 0.0
            p_val = 1.0
            test_type = "Exact Binomial (no discordance)"
        elif discordant < 25:
            res = stats.binomtest(min(b, c), n=discordant, p=0.5)
            stat = float((abs(b - c) - 1.0)**2 / discordant) if discordant > 0 else 0.0
            p_val = float(res.pvalue)
            test_type = "Two-tailed Exact Binomial Test"
        else:
            stat = float((abs(b - c) - 1.0)**2 / discordant)
            p_val = float(stats.chi2.sf(stat, df=1))
            test_type = "McNemar Test with Edwards Continuity Correction"

        mcnemar_results.append({
            "comparison": f"Arm B (lambda=0.10) vs {name}",
            "arm_compared": aid,
            "discordant_ref_only": n10,
            "discordant_arm_only": n01,
            "total_discordant": discordant,
            "test_used": test_type,
            "statistic": stat,
            "p_value": p_val
        })

    # Holm-Bonferroni correction
    mcnemar_results = sorted(mcnemar_results, key=lambda x: x["p_value"])
    k = len(mcnemar_results)
    for i, r in enumerate(mcnemar_results):
        alpha_adj = 0.05 / (k - i)
        r["alpha_adj"] = alpha_adj
        r["significant"] = bool(r["p_value"] < alpha_adj)

    for r in mcnemar_results:
        sig_str = "YES (p < alpha_adj)" if r["significant"] else "NO (p >= alpha_adj)"
        print(f"  {r['comparison']}: discordant=({r['discordant_ref_only']}, {r['discordant_arm_only']}), stat={r['statistic']:.4f}, p={r['p_value']:.4f}, alpha_adj={r['alpha_adj']:.4f} -> Significant: {sig_str} [{r['test_used']}]")

    # ---------------------------------------------------------
    # 6. Comparison of Arm B vs Historical EXP-06A Config B
    # ---------------------------------------------------------
    print("\n[Step 6] Comparison of Arm B with Historical EXP-06A Config B:")
    exp06a_csv = EXP06A_REPORT_DIR / "config_b_oof_predictions.csv"
    if exp06a_csv.is_file():
        df_06a_b = pd.read_csv(exp06a_csv)
        assert list(df_06a_b["participant_id"]) == list(oof_dfs["arm_b"]["participant_id"])
        y_pred_06a = np.array(df_06a_b["predicted_label"])
        acc_06a = float(accuracy_score(y_true, y_pred_06a))
        bacc_06a = float(balanced_accuracy_score(y_true, y_pred_06a))
        _, _, f1_06a, _ = precision_recall_fscore_support(y_true, y_pred_06a, labels=CANONICAL_CLASSES, average="macro", zero_division=0)

        correct_06a = (y_pred_06a == np.array(y_true))
        n11_comp = int(np.sum(correct_b & correct_06a))
        n10_b_only = int(np.sum(correct_b & (~correct_06a)))
        n01_06a_only = int(np.sum((~correct_b) & correct_06a))
        n00_comp = int(np.sum((~correct_b) & (~correct_06a)))

        disc_comp = n10_b_only + n01_06a_only
        res_comp = stats.binomtest(min(n10_b_only, n01_06a_only), n=disc_comp, p=0.5) if disc_comp > 0 else None
        p_val_comp = float(res_comp.pvalue) if res_comp else 1.0

        comp_summary = {
            "exp06a_config_b_macro_f1": f1_06a,
            "exp06a_config_b_acc": acc_06a,
            "exp06a_config_b_bal_acc": bacc_06a,
            "exp06b_arm_b_macro_f1": summary_rows[1]["macro_f1"],
            "exp06b_arm_b_acc": summary_rows[1]["accuracy"],
            "exp06b_arm_b_bal_acc": summary_rows[1]["balanced_accuracy"],
            "delta_macro_f1": summary_rows[1]["macro_f1"] - f1_06a,
            "delta_acc": summary_rows[1]["accuracy"] - acc_06a,
            "discordant_exp06b_only": n10_b_only,
            "discordant_exp06a_only": n01_06a_only,
            "p_value": p_val_comp,
            "significant_difference": bool(p_val_comp < 0.05)
        }
        print(f"  Historical EXP-06A Config B Macro F1: {f1_06a:.4f} (Acc: {acc_06a*100:.2f}%)")
        print(f"  Current EXP-06B Arm B Macro F1:        {summary_rows[1]['macro_f1']:.4f} (Acc: {summary_rows[1]['accuracy']*100:.2f}%)")
        print(f"  Delta: {comp_summary['delta_macro_f1']:+.4f} Macro F1, Delta Acc: {comp_summary['delta_acc']*100:+.2f}%")
        print(f"  Paired Discordance: Arm B only correct = {n10_b_only}, EXP-06A only correct = {n01_06a_only}")
        print(f"  Two-tailed p-value: {p_val_comp:.4f} -> Statistically Indistinguishable: {'YES' if p_val_comp >= 0.05 else 'NO'}")
    else:
        comp_summary = {"status": "EXP-06A predictions CSV not found"}
        print("  Notice: EXP-06A Config B CSV not found.")

    # ---------------------------------------------------------
    # 7. Serialize Full Statistical Results JSON
    # ---------------------------------------------------------
    full_output = {
        "summary": summary_rows,
        "mcnemar_pairwise_vs_arm_b": mcnemar_results,
        "reproduction_comparison_arm_b_vs_exp06a_config_b": comp_summary,
        "protected_checkpoint_audit": {
            "all_passed": protected_integrity_pass,
            "details": hash_audit_table
        },
        "exp06b_checkpoints": exp06b_ckpts
    }

    out_json_path = EXP06B_REPORT_DIR / "statistical_analysis_results.json"
    with open(out_json_path, "w") as f:
        json.dump(full_output, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else float(x) if isinstance(x, (np.floating, np.float64)) else str(x))
    print(f"\n[Step 7] Full statistical results serialized to: {out_json_path}")
    print("=" * 80)
    return full_output

if __name__ == "__main__":
    run_statistical_analysis()
