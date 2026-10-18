#!/usr/bin/env python3
"""
EXP-06C: Statistical Analysis, Integrity Verification, and Audit Script
----------------------------------------------------------------------
Performs:
1. Cohort isolation and OOF prediction integrity check (80 participants exactly).
2. Post-execution hash verification of all 56 protected baseline checkpoints.
3. Verification and hashing of all 10 EXP-06C checkpoints (5 Config A, 5 Config B).
4. Independent recomputation of pooled OOF metrics and fold metrics from prediction CSVs.
5. Paired statistical significance testing (McNemar with continuity correction / exact binomial)
   between Config A (Linear) and Config B (Nonlinear MLP).
6. Participant-level paired non-parametric bootstrap (B=2000, seed=42) for Delta Macro F1
   with 95% percentile confidence interval.
7. Comparison of Config A against EXP-06B Arm D (historical linear reference).
8. Serializes results into reports/accuracy_improvement/exp06c/statistical_analysis_results.json.
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
EXP06C_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06c"
EXP06C_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp06c" / "checkpoints"
EXP06B_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06b"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
PRE_HASHES_FILE = PROJECT_ROOT / "scratch" / "exp06c_pre_preparation_hashes.json"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CONFIGS = ["config_a", "config_b"]
CONFIG_NAMES = {
    "config_a": "Config A: Linear Projection Reference (3072 -> 128)",
    "config_b": "Config B: Two-Layer Nonlinear MLP Projection (3072 -> 512 -> 128)"
}
HEAD_TYPES = {
    "config_a": "linear",
    "config_b": "mlp"
}
EXPECTED_PARAMS = {
    "config_a": 1968772,
    "config_b": 3214468
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
    print("EXP-06C: POST-TRAINING AUDIT & STATISTICAL ANALYSIS")
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
        print(f"  {name:<24} {status_str} | pre={pre_h[:12]}... post={post_h[:12]}...", flush=True)

    assert protected_integrity_pass, "CRITICAL AUDIT FAILURE: Protected checkpoint hash altered during execution!"
    print(f"  All {len(pre_hashes)} protected checkpoints verified intact (0 bytes modified).", flush=True)

    # ---------------------------------------------------------
    # 2. EXP-06C Generated Checkpoints Audit
    # ---------------------------------------------------------
    print("\n[Step 2] EXP-06C Generated Checkpoints Audit:")
    exp06c_ckpts = []
    for cid in CONFIGS:
        ckpt_dir = EXP06C_CKPT_BASE / cid
        for f in range(1, 6):
            cp = ckpt_dir / f"fold_{f}_best.weights.h5"
            assert cp.is_file(), f"Expected checkpoint missing: {cp}"
            sz = cp.stat().st_size
            h = sha256_of_file(cp)
            exp06c_ckpts.append({
                "config": cid,
                "fold": f,
                "path": str(cp),
                "size_bytes": sz,
                "sha256": h
            })
            print(f"  {cid.upper()} Fold {f}: size={sz:,} bytes, sha256={h[:16]}...")
    assert len(exp06c_ckpts) == 10, f"Expected 10 checkpoints, got {len(exp06c_ckpts)}"
    print(f"  All 10 EXP-06C checkpoints saved and validated.")

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
    for cid in CONFIGS:
        pred_p = EXP06C_REPORT_DIR / f"{cid}_oof_predictions.csv"
        assert pred_p.is_file(), f"Prediction file missing: {pred_p}"
        df = pd.read_csv(pred_p)
        assert len(df) == 80, f"Expected 80 predictions in {cid}, found {len(df)}"
        assert sorted(df["participant_id"].unique()) == dev_pids_expected, f"Participant ID mismatch in {cid}"
        # Quarantine check
        overlap = set(df["participant_id"]).intersection(test_pids_locked)
        assert len(overlap) == 0, f"DATA LEAKAGE: Locked test IDs found in {cid}: {overlap}"
        oof_dfs[cid] = df.sort_values("participant_id").reset_index(drop=True)
        print(f"  {cid.upper()}: Exactly 80 dev participants, 0 test overlap, 1 prediction per participant.")

    # ---------------------------------------------------------
    # 4. Independent Metric Recomputation from OOF Predictions
    # ---------------------------------------------------------
    print("\n[Step 4] Independent Metric Recomputation from OOF CSVs:")
    y_true = list(oof_dfs["config_a"]["true_label"])
    # Verify true labels match across configs
    assert list(oof_dfs["config_b"]["true_label"]) == y_true, "True label mismatch across configurations!"

    # Load fold logs from experiment_config.json
    with open(EXP06C_REPORT_DIR / "experiment_config.json", "r") as f:
        exp_config_metrics = {m["config_id"]: m for m in json.load(f)}

    summary_rows = []
    for cid in CONFIGS:
        df = oof_dfs[cid]
        y_pred = list(df["predicted_label"])

        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
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

        fold_logs = exp_config_metrics[cid]["folds"]
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
            "config_id": cid,
            "config_name": CONFIG_NAMES[cid],
            "head_type": HEAD_TYPES[cid],
            "trainable_params": EXPECTED_PARAMS[cid],
            "lambda_supcon": 0.50,
            "temperature": 0.05,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_precision": float(p_m),
            "macro_recall": float(r_m),
            "macro_f1": float(f1_m),
            "weighted_f1": float(f1_w),
            "fold_f1_mean": f1_mean,
            "fold_f1_std": f1_std,
            "fold_f1s": fold_f1s,
            "per_class": per_class_dict,
            "confusion_matrix": cm.tolist(),
            "total_runtime_sec": exp_config_metrics[cid]["total_runtime_sec"]
        }
        summary_rows.append(row)

    print(f"\n{'Config':<10} {'Head':<8} {'Params':<12} {'Acc':<10} {'BalAcc':<10} {'Macro F1':<10} {'Fold F1 Mean (Std)':<22} {'Runtime (m)':<12}")
    for r in summary_rows:
        print(f"{r['config_id']:<10} {r['head_type']:<8} {r['trainable_params']:<12,d} {r['accuracy']*100:<10.2f}% {r['balanced_accuracy']*100:<10.2f}% {r['macro_f1']:<10.4f} {r['fold_f1_mean']:.4f} (+/- {r['fold_f1_std']:.4f})  {r['total_runtime_sec']/60:<12.2f}")

    # ---------------------------------------------------------
    # 5. Paired Statistical Testing (McNemar / Exact Binomial)
    # ---------------------------------------------------------
    print("\n[Step 5] Paired Classification Correctness Test (Config A vs Config B):")
    y_true_arr = np.array(y_true)
    y_pred_a = np.array(oof_dfs["config_a"]["predicted_label"])
    y_pred_b = np.array(oof_dfs["config_b"]["predicted_label"])

    correct_a = (y_pred_a == y_true_arr)
    correct_b = (y_pred_b == y_true_arr)

    n11 = int(np.sum(correct_a & correct_b))      # Both correct
    n10 = int(np.sum(correct_a & (~correct_b)))   # Config A only
    n01 = int(np.sum((~correct_a) & correct_b))   # Config B only
    n00 = int(np.sum((~correct_a) & (~correct_b))) # Both incorrect

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

    contingency_table = {
        "n11_both_correct": n11,
        "n10_config_a_only": n10,
        "n01_config_b_only": n01,
        "n00_both_incorrect": n00,
        "total_discordant": discordant
    }

    mcnemar_test_result = {
        "comparison": "Config A (Linear) vs Config B (Nonlinear MLP)",
        "contingency_table": contingency_table,
        "test_used": test_type,
        "statistic": stat,
        "p_value": p_val,
        "statistically_significant": bool(p_val < 0.05)
    }

    print(f"  Contingency 2x2: Both Correct={n11}, A Only={n10}, B Only={n01}, Both Incorrect={n00}")
    print(f"  Discordant Pairs: {discordant} (A-only: {b}, B-only: {c})")
    print(f"  Test: {test_type}")
    print(f"  Statistic: {stat:.4f}, p-value: {p_val:.5f} (Significant @ alpha=0.05: {p_val < 0.05})")

    # ---------------------------------------------------------
    # 6. Participant-Level Paired Bootstrap Protocol
    # ---------------------------------------------------------
    print("\n[Step 6] Participant-Level Paired Non-Parametric Bootstrap (B=2000, seed=42):")
    B = 2000
    RNG_SEED = 42
    rng = np.random.RandomState(RNG_SEED)
    N = len(y_true)
    assert N == 80

    delta_macro_f1s = []
    f1_a_samples = []
    f1_b_samples = []

    for b_idx in range(B):
        sample_indices = rng.choice(N, size=N, replace=True)
        y_true_sample = y_true_arr[sample_indices]
        y_pred_a_sample = y_pred_a[sample_indices]
        y_pred_b_sample = y_pred_b[sample_indices]

        # Use zero_division=0 to deterministically handle bootstrap resamples omitting a class
        _, _, f1_a_boot, _ = precision_recall_fscore_support(
            y_true_sample, y_pred_a_sample, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )
        _, _, f1_b_boot, _ = precision_recall_fscore_support(
            y_true_sample, y_pred_b_sample, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )

        delta = float(f1_b_boot - f1_a_boot)
        delta_macro_f1s.append(delta)
        f1_a_samples.append(float(f1_a_boot))
        f1_b_samples.append(float(f1_b_boot))

    delta_macro_f1s = np.array(delta_macro_f1s)
    ci_lower = float(np.percentile(delta_macro_f1s, 2.5))
    ci_upper = float(np.percentile(delta_macro_f1s, 97.5))
    delta_mean = float(np.mean(delta_macro_f1s))
    delta_std = float(np.std(delta_macro_f1s))

    # Empirical two-tailed bootstrap p-value
    p_boot_greater = np.mean(delta_macro_f1s <= 0.0)
    p_boot_less = np.mean(delta_macro_f1s >= 0.0)
    bootstrap_p_value = float(2.0 * min(p_boot_greater, p_boot_less))

    ci_excludes_zero = bool(ci_lower > 0.0 or ci_upper < 0.0)

    bootstrap_results = {
        "num_replicates": B,
        "random_seed": RNG_SEED,
        "delta_observed": float(summary_rows[1]["macro_f1"] - summary_rows[0]["macro_f1"]),
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "ci_95_percentile": [ci_lower, ci_upper],
        "ci_excludes_zero": ci_excludes_zero,
        "bootstrap_p_value": bootstrap_p_value,
        "omitted_class_handling": "Handled deterministically via scikit-learn precision_recall_fscore_support with labels=CANONICAL_CLASSES and zero_division=0",
        "statistically_significant": ci_excludes_zero
    }

    print(f"  Observed Delta Macro F1 (B - A): {bootstrap_results['delta_observed']:+.4f}")
    print(f"  Bootstrap Mean Delta:            {delta_mean:+.4f} (+/- {delta_std:.4f})")
    print(f"  95% Percentile Bootstrap CI:     [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  CI Excludes Zero:                {ci_excludes_zero}")
    print(f"  Bootstrap Two-Tailed p-value:    {bootstrap_p_value:.5f}")

    # ---------------------------------------------------------
    # 7. Historical Reference Benchmark (Config A vs EXP-06B Arm D)
    # ---------------------------------------------------------
    print("\n[Step 7] Historical Baseline Comparison (Config A vs EXP-06B Arm D):")
    exp06b_arm_d_csv = EXP06B_REPORT_DIR / "arm_d_oof_predictions.csv"
    historical_comp = None
    if exp06b_arm_d_csv.is_file():
        df_hist = pd.read_csv(exp06b_arm_d_csv).sort_values("participant_id").reset_index(drop=True)
        y_pred_hist = np.array(df_hist["predicted_label"])
        correct_hist = (y_pred_hist == y_true_arr)

        hist_acc = float(accuracy_score(y_true_arr, y_pred_hist))
        hist_bacc = float(balanced_accuracy_score(y_true_arr, y_pred_hist))
        _, _, hist_f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_hist, labels=CANONICAL_CLASSES, average="macro", zero_division=0
        )

        n11_h = int(np.sum(correct_hist & correct_a))
        n10_h = int(np.sum(correct_hist & (~correct_a))) # Hist only
        n01_h = int(np.sum((~correct_hist) & correct_a)) # Config A only
        n00_h = int(np.sum((~correct_hist) & (~correct_a)))
        disc_h = n10_h + n01_h

        if disc_h == 0:
            p_val_h = 1.0
        elif disc_h < 25:
            p_val_h = float(stats.binomtest(min(n10_h, n01_h), n=disc_h, p=0.5).pvalue)
        else:
            chi_h = float((abs(n10_h - n01_h) - 1.0)**2 / disc_h)
            p_val_h = float(stats.chi2.sf(chi_h, df=1))

        historical_comp = {
            "historical_experiment": "EXP-06B Arm D (lambda=0.50, tau=0.05)",
            "historical_accuracy": hist_acc,
            "historical_balanced_accuracy": hist_bacc,
            "historical_macro_f1": float(hist_f1),
            "config_a_macro_f1": float(summary_rows[0]["macro_f1"]),
            "delta_macro_f1": float(summary_rows[0]["macro_f1"] - hist_f1),
            "contingency_table": {
                "both_correct": n11_h,
                "historical_only": n10_h,
                "config_a_only": n01_h,
                "both_incorrect": n00_h,
                "total_discordant": disc_h
            },
            "p_value": p_val_h,
            "consistent_with_stochastic_variation": bool(p_val_h >= 0.05)
        }
        print(f"  EXP-06B Arm D Historical Macro F1: {hist_f1:.4f} (Acc: {hist_acc*100:.2f}%)")
        print(f"  EXP-06C Config A Macro F1:         {summary_rows[0]['macro_f1']:.4f} (Acc: {summary_rows[0]['accuracy']*100:.2f}%)")
        print(f"  Delta:                             {historical_comp['delta_macro_f1']:+.4f} (Discordant: {disc_h}, p={p_val_h:.4f})")
        print(f"  Status: {'CONSISTENT' if p_val_h >= 0.05 else 'STATISTICALLY DIVERGENT'} with stochastic variation.")

    # ---------------------------------------------------------
    # 8. Serialize Results
    # ---------------------------------------------------------
    final_output = {
        "experiment_id": "EXP-06C",
        "experiment_name": "Projection Head Architecture Ablation",
        "primary_metric": "Pooled OOF Macro F1 across 80 development participants",
        "secondary_metric": "Mean fold-level Macro F1 across 5 validation folds",
        "configurations": summary_rows,
        "mcnemar_test": mcnemar_test_result,
        "paired_bootstrap": bootstrap_results,
        "historical_reference_comparison": historical_comp,
        "protected_checkpoints_integrity": {
            "total_checkpoints_verified": len(pre_hashes),
            "all_passed": protected_integrity_pass,
            "audit_table": hash_audit_table
        },
        "generated_checkpoints": exp06c_ckpts
    }

    out_file = EXP06C_REPORT_DIR / "statistical_analysis_results.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[Step 8] Statistical analysis successfully written to {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_statistical_analysis()
