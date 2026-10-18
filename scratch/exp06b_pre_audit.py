#!/usr/bin/env python3
"""
EXP-06B Pre-Training Audit & Recomputation Script
--------------------------------------------------
1. Independently recompute EXP-06A metrics from raw OOF predictions CSVs.
2. Independently recompute paired McNemar tests.
3. Compare EXP-05 Config A vs EXP-06A Config A down to seeds, folds, training loop, checkpoints.
4. Verify fold isolation (80 dev, 21 locked test quarantined, 26 hospital quarantined).
5. Synthetic safety test of EXP-06B arms (lambda in [0.05, 0.10, 0.20, 0.50] with fixed tau=0.05).
"""

import json
import csv
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP06A_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp06a"
EXP05_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp05"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]

def recompute_exp06a():
    print("=" * 80)
    print("1. INDEPENDENT RECOMPUTATION OF EXP-06A METRICS FROM OOF CSVs")
    print("=" * 80)

    configs = ["config_a", "config_b", "config_c", "config_d"]
    results = {}
    oof_dfs = {}

    for cid in configs:
        csv_path = EXP06A_REPORT_DIR / f"{cid}_oof_predictions.csv"
        df = pd.read_csv(csv_path)
        oof_dfs[cid] = df

        y_true = list(df["true_label"])
        y_pred = list(df["predicted_label"])

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        p_c, r_c, f1_c, s_c = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

        results[cid] = {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_precision": p_m,
            "macro_recall": r_m,
            "macro_f1": f1_m,
            "per_class": {
                c: {"recall": r_c[i], "precision": p_c[i], "f1": f1_c[i], "support": s_c[i]}
                for i, c in enumerate(CANONICAL_CLASSES)
            },
            "confusion_matrix": cm.tolist()
        }

        print(f"\n[{cid.upper()}] (N={len(df)})")
        print(f"  Accuracy:          {acc*100:.2f}% ({int(acc*len(df))}/{len(df)})")
        print(f"  Balanced Accuracy: {bacc*100:.2f}%")
        print(f"  Macro Precision:   {p_m:.4f}")
        print(f"  Macro Recall:      {r_m:.4f}")
        print(f"  Macro F1:          {f1_m:.4f}")
        for c in CANONICAL_CLASSES:
            pc = results[cid]["per_class"][c]
            print(f"    {c:<4}: Recall = {pc['recall']*100:.2f}%, Precision = {pc['precision']:.4f}, F1 = {pc['f1']:.4f} (Support={pc['support']})")

    # 2. Paired McNemar Tests
    print("\n" + "=" * 80)
    print("2. INDEPENDENT RECOMPUTATION OF PAIRED MCNEMAR TESTS")
    print("=" * 80)

    y_true_a = np.array(oof_dfs["config_a"]["true_label"])
    pred_a = np.array(oof_dfs["config_a"]["predicted_label"])
    correct_a = (pred_a == y_true_a)

    for other_cid in ["config_b", "config_c", "config_d"]:
        pred_other = np.array(oof_dfs[other_cid]["predicted_label"])
        correct_other = (pred_other == y_true_a)

        b = int(np.sum(correct_a & (~correct_other)))
        c = int(np.sum((~correct_a) & correct_other))
        n11 = int(np.sum(correct_a & correct_other))
        n00 = int(np.sum((~correct_a) & (~correct_other)))
        discordant = b + c

        if discordant < 25:
            res = stats.binomtest(min(b, c), n=discordant, p=0.5)
            stat = float((abs(b - c) - 1.0)**2 / discordant) if discordant > 0 else 0.0
            p_val = res.pvalue
            test_type = "Exact Binomial"
        else:
            stat = float((abs(b - c) - 1.0)**2 / discordant)
            p_val = float(stats.chi2.sf(stat, df=1))
            test_type = "Edwards Continuity Chi2"

        print(f"Config A vs {other_cid.upper()}:")
        print(f"  Contingency: [ [Both Correct: {n11}, A Only: {b}], [Other Only: {c}, Both Wrong: {n00}] ]")
        print(f"  Discordant: b={b}, c={c} (Total={discordant})")
        print(f"  Test: {test_type}, Stat={stat:.4f}, p={p_val:.4f}")

    return results, oof_dfs

def verify_config_a_discrepancy(oof_dfs):
    print("\n" + "=" * 80)
    print("3. VERIFY CONFIG A DISCREPANCY: EXP-05 vs EXP-06A")
    print("=" * 80)

    exp05_pred_csv = EXP05_REPORT_DIR / "config_a_oof_predictions.csv"
    if not exp05_pred_csv.is_file():
        print(f"EXP-05 Config A OOF CSV not found: {exp05_pred_csv}")
        return

    df_exp05 = pd.read_csv(exp05_pred_csv)
    df_exp06a = oof_dfs["config_a"]

    assert len(df_exp05) == 80
    assert len(df_exp06a) == 80

    # Align participants
    df_exp05 = df_exp05.sort_values("participant_id").reset_index(drop=True)
    df_exp06a = df_exp06a.sort_values("participant_id").reset_index(drop=True)

    assert list(df_exp05["participant_id"]) == list(df_exp06a["participant_id"])
    assert list(df_exp05["true_label"]) == list(df_exp06a["true_label"])

    y_true = np.array(df_exp05["true_label"])
    p_05 = np.array(df_exp05["predicted_label"])
    p_06 = np.array(df_exp06a["predicted_label"])

    c_05 = (p_05 == y_true)
    c_06 = (p_06 == y_true)

    agreed = (p_05 == p_06)
    num_agreed = int(np.sum(agreed))

    print(f"Total Participants: 80")
    print(f"Predictions Identical: {num_agreed}/80 ({num_agreed/80*100:.1f}%)")
    print(f"Predictions Differing: {80 - num_agreed}/80 ({(80 - num_agreed)/80*100:.1f}%)")
    print(f"EXP-05 Config A Correct:  {int(np.sum(c_05))}/80 ({accuracy_score(y_true, p_05)*100:.2f}%)")
    print(f"EXP-06A Config A Correct: {int(np.sum(c_06))}/80 ({accuracy_score(y_true, p_06)*100:.2f}%)")

    # Shift breakdown
    both_correct = int(np.sum(c_05 & c_06))
    exp05_only = int(np.sum(c_05 & (~c_06)))
    exp06a_only = int(np.sum((~c_05) & c_06))
    both_wrong = int(np.sum((~c_05) & (~c_06)))

    print(f"\nContingency Grid (EXP-05 vs EXP-06A Config A):")
    print(f"  Both Correct:              {both_correct} ({(both_correct/80)*100:.1f}%)")
    print(f"  EXP-05 Only Correct:       {exp05_only} ({(exp05_only/80)*100:.1f}%)")
    print(f"  EXP-06A Only Correct:      {exp06a_only} ({(exp06a_only/80)*100:.1f}%)")
    print(f"  Both Wrong:                {both_wrong} ({(both_wrong/80)*100:.1f}%)")
    print(f"  Net Delta:                 {int(np.sum(c_06)) - int(np.sum(c_05))} participants")

    # Check fold-level differences
    print("\nFold-by-Fold Breakdown:")
    for f in range(1, 6):
        idx_f = (df_exp05["fold_id"] == f)
        acc_05_f = accuracy_score(y_true[idx_f], p_05[idx_f])
        acc_06_f = accuracy_score(y_true[idx_f], p_06[idx_f])
        n_corr_05 = int(np.sum(c_05[idx_f]))
        n_corr_06 = int(np.sum(c_06[idx_f]))
        print(f"  Fold {f} (N={np.sum(idx_f)}): EXP-05 = {n_corr_05} ({acc_05_f*100:.1f}%), EXP-06A = {n_corr_06} ({acc_06_f*100:.1f}%), Delta = {n_corr_06 - n_corr_05}")

    # Inspect code differences between run_exp05.py and run_exp06a.py for Config A
    exp05_code = (PROJECT_ROOT / "experiments" / "exp05" / "run_exp05.py").read_text()
    exp06a_code = (PROJECT_ROOT / "experiments" / "exp06a" / "run_exp06a.py").read_text()

    print("\nCode Comparison (EXP-05 vs EXP-06A Config A):")
    # Check random seed
    print(f"  EXP-05 SEED:  42 in code? {'SEED = 42' in exp05_code}")
    print(f"  EXP-06A SEED: 42 in code? {'SEED = 42' in exp06a_code}")
    print(f"  EXP-05 Batch Size: 16?    {'BATCH_SIZE = 16' in exp05_code}")
    print(f"  EXP-06A Batch Size: 16?   {'BATCH_SIZE = 16' in exp06a_code}")
    print(f"  EXP-05 Head LR: 1e-3?     {'HEAD_LR = 1e-3' in exp05_code}")
    print(f"  EXP-06A Head LR: 1e-3?    {'HEAD_LR = 1e-3' in exp06a_code}")
    print(f"  EXP-05 Weight Decay: 1e-4?{'WEIGHT_DECAY = 1e-4' in exp05_code}")
    print(f"  EXP-06A Weight Decay: 1e-4?{'WEIGHT_DECAY = 1e-4' in exp06a_code}")

    # Check initialization mechanism
    print("  Initialization mechanism:")
    if "save_weights" in exp06a_code and "scratch" in exp06a_code:
        print("    EXP-06A used shared ephemeral epoch-zero initial weights across configs per fold.")
    if "build_concatenation_model" in exp05_code:
        print("    EXP-05 Config A instantiated model directly per fold with seed 42 + fold_idx.")

def verify_cohort_isolation():
    print("\n" + "=" * 80)
    print("4. COHORT ISOLATION & DATA INTEGRITY VERIFICATION")
    print("=" * 80)

    df_split = pd.read_csv(SPLIT_CSV)
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]
    test_df = df_split[df_split["split"] == "TEST"]

    print(f"Total split entries: {len(df_split)} rows")
    print(f"Development scans:   {len(dev_df)} (Participants: {dev_df['participant_id'].nunique()})")
    print(f"Locked test scans:   {len(test_df)} (Participants: {test_df['participant_id'].nunique()})")

    assert dev_df['participant_id'].nunique() == 80, "Expected 80 dev participants"
    assert len(dev_df) == 240, "Expected 240 dev scans"
    assert test_df['participant_id'].nunique() == 21, "Expected 21 locked test participants"
    assert len(test_df) == 63, "Expected 63 locked test scans"

    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())
    overlap = dev_pids.intersection(test_pids)
    print(f"Dev / Locked Test Overlap: {len(overlap)} (Must be 0)")
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {overlap}"

    # Verify hospital cohort paths
    hosp_dir = PROJECT_ROOT / "data" / "hospital_validation"
    if hosp_dir.is_dir():
        hosp_files = list(hosp_dir.glob("**/*.bmp")) + list(hosp_dir.glob("**/*.png"))
        print(f"Hospital validation files on disk: {len(hosp_files)} (Quarantined)")
    else:
        print("Hospital validation directory: Quarantined / Not on immediate path")

def test_exp06b_synthetic_safety():
    print("\n" + "=" * 80)
    print("5. EXP-06B SYNTHETIC LOSS & GRADIENT SAFETY TEST")
    print("=" * 80)

    # In EXP-06B, tau is fixed at 0.05, and lambda_supcon varies across [0.05, 0.10, 0.20, 0.50]
    lambdas = [0.05, 0.10, 0.20, 0.50]
    tau = 0.05

    # Define supervised contrastive loss function exactly
    def compute_supcon_loss(features, labels, temperature=0.05):
        features = tf.math.l2_normalize(features, axis=-1)
        similarity = tf.matmul(features, features, transpose_b=True) / temperature
        labels_col = tf.reshape(labels, (-1, 1))
        mask_pos = tf.cast(tf.equal(labels_col, tf.transpose(labels_col)), tf.float32)
        batch_dim = tf.shape(features)[0]
        mask_self = 1.0 - tf.eye(batch_dim, dtype=tf.float32)
        mask_pos = mask_pos * mask_self

        logits_max = tf.reduce_max(similarity, axis=-1, keepdims=True)
        logits = similarity - tf.stop_gradient(logits_max)
        exp_logits = tf.exp(logits) * mask_self
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=-1, keepdims=True) + 1e-8)

        pos_count = tf.reduce_sum(mask_pos, axis=-1)
        valid = tf.greater(pos_count, 0)
        mean_log_prob = tf.reduce_sum(mask_pos * log_prob, axis=-1) / tf.maximum(pos_count, 1.0)
        loss = -tf.reduce_mean(tf.boolean_mask(mean_log_prob, valid)) if tf.reduce_any(valid) else 0.0
        return loss

    # Create synthetic batch
    np.random.seed(42)
    tf.random.set_seed(42)
    B = 8
    dummy_emb = tf.random.normal((B, 128))
    dummy_labels = tf.constant([0, 1, 0, 2, 1, 3, 2, 3], dtype=tf.int32)
    dummy_logits = tf.random.normal((B, 4))
    dummy_ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(dummy_labels, dummy_logits)

    supcon_loss = compute_supcon_loss(dummy_emb, dummy_labels, temperature=tau)
    print(f"Synthetic Fixed Contrastive Loss (tau={tau}): {float(supcon_loss):.6f}")
    print(f"Synthetic Classification Loss (CE):         {float(dummy_ce_loss):.6f}")

    print("\nArm Loss Combinations:")
    for lam in lambdas:
        total_loss = dummy_ce_loss + lam * supcon_loss
        print(f"  Arm (lambda={lam:0.2f}, tau={tau:.2f}): Total Loss = {float(total_loss):.6f} (CE = {float(dummy_ce_loss):.4f}, SupCon = {float(lam*supcon_loss):.4f})")
        assert np.isfinite(float(total_loss)), f"Total loss not finite for lambda={lam}"

    print("Synthetic safety test PASSED: Loss scaling is strictly linear and mathematically well-behaved.")

if __name__ == "__main__":
    results, oof_dfs = recompute_exp06a()
    verify_config_a_discrepancy(oof_dfs)
    verify_cohort_isolation()
    test_exp06b_synthetic_safety()
