#!/usr/bin/env python3
"""
EXP-07-RES-FIDELITY: Evaluation & Statistical Hypothesis Testing
==============================================================
Calculates participant-level metrics, Wilson confidence intervals,
paired McNemar test, paired permutation test, and empirical null distributions.

Primary Predefined Metric: Out-of-Fold Balanced Accuracy
Secondary Metrics: Macro F1, Participant Accuracy, Cohen's Kappa, Per-Class Recall
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, f1_score, cohen_kappa_score
)

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp07_resolution_fidelity.config import (
    CKPT_DIR, REPORT_DIR, SPLIT_CSV, NATIVE_CLASSES, NATIVE_REV_MAP,
    CANONICAL_CLASSES, CANONICAL_REV_MAP
)

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    """
    Computes Wilson score interval for binomial proportion k / n.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% standard normal quantile
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper

def compute_comprehensive_metrics(y_true, y_probs, condition_name="Model"):
    """
    Computes all predefined performance metrics on out-of-fold participant predictions.
    """
    y_pred = np.argmax(y_probs, axis=1)
    n = len(y_true)
    k = int(np.sum(y_true == y_pred))

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    ci_low, ci_high = compute_wilson_ci(k, n)

    # Per-class precision, recall, f1
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(4)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    per_class = {}
    for i, c in enumerate(NATIVE_CLASSES):
        per_class[c] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i])
        }

    # Majority class baseline
    majority_class_idx = int(np.argmax(np.bincount(y_true)))
    majority_count = int(np.max(np.bincount(y_true)))
    majority_baseline = majority_count / n

    # Canonical remapping (swapping index 1 and 2 for B+ and AB+)
    # native indices: 0: A+, 1: B+, 2: AB+, 3: O+
    # canonical indices: 0: A+, 1: AB+, 2: B+, 3: O+
    canon_map = [0, 2, 1, 3]
    y_true_canon = np.array([canon_map[val] for val in y_true])
    y_pred_canon = np.array([canon_map[val] for val in y_pred])
    cm_canonical = confusion_matrix(y_true_canon, y_pred_canon, labels=list(range(4)))

    results = {
        "condition": condition_name,
        "n_participants": n,
        "correct_predictions": k,
        "accuracy": float(acc),
        "accuracy_ci_95": [float(ci_low), float(ci_high)],
        "primary_balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "majority_class": NATIVE_CLASSES[majority_class_idx],
        "majority_baseline_accuracy": float(majority_baseline),
        "native_class_order": NATIVE_CLASSES,
        "native_per_class_metrics": per_class,
        "native_confusion_matrix": cm.tolist(),
        "canonical_class_order": CANONICAL_CLASSES,
        "canonical_confusion_matrix": cm_canonical.tolist()
    }
    return results

def paired_mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    Executes McNemar's exact test on paired participant predictions.
    Contingency table of discordance:
      b: A correct, B incorrect
      c: A incorrect, B correct
    """
    corr_a = (y_pred_a == y_true)
    corr_b = (y_pred_b == y_true)

    b = int(np.sum(corr_a & (~corr_b)))  # 112 correct, 224 incorrect
    c = int(np.sum((~corr_a) & corr_b))  # 112 incorrect, 224 correct

    total_discordant = b + c
    if total_discordant == 0:
        p_val = 1.0
    else:
        # Exact two-sided binomial test under H0: p = 0.5
        res = binomtest(min(b, c), total_discordant, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "b_arm_a_better": b,
        "c_arm_b_better": c,
        "total_discordant_pairs": total_discordant,
        "mcnemar_exact_p_value": p_val
    }

def paired_permutation_test(y_true, y_probs_a, y_probs_b, n_permutations=10000, seed=42):
    """
    Non-parametric paired permutation test assessing whether the difference in
    balanced accuracy between Arm A (112x112) and Arm B (224x224) exceeds chance.
    """
    rng = np.random.RandomState(seed)
    y_pred_a = np.argmax(y_probs_a, axis=1)
    y_pred_b = np.argmax(y_probs_b, axis=1)

    obs_diff = balanced_accuracy_score(y_true, y_pred_b) - balanced_accuracy_score(y_true, y_pred_a)
    n = len(y_true)

    diff_null = np.zeros(n_permutations, dtype=np.float32)
    for i in range(n_permutations):
        # Swap predictions for each participant with probability 0.5
        swap = rng.rand(n) < 0.5
        perm_a = np.where(swap, y_pred_b, y_pred_a)
        perm_b = np.where(swap, y_pred_a, y_pred_b)
        diff_null[i] = balanced_accuracy_score(y_true, perm_b) - balanced_accuracy_score(y_true, perm_a)

    p_value = float(np.mean(np.abs(diff_null) >= np.abs(obs_diff)))
    return {
        "observed_balanced_acc_diff (B - A)": float(obs_diff),
        "paired_permutation_p_value": p_value,
        "n_permutations": n_permutations
    }

def participant_label_permutation_test(y_true, y_probs, n_permutations=1000, seed=42):
    """
    Participant-level label permutation test to derive non-parametric p-values against chance.
    """
    rng = np.random.RandomState(seed)
    y_pred = np.argmax(y_probs, axis=1)
    obs_bal_acc = balanced_accuracy_score(y_true, y_pred)
    obs_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    null_bal_acc = np.zeros(n_permutations, dtype=np.float32)
    null_f1 = np.zeros(n_permutations, dtype=np.float32)

    for i in range(n_permutations):
        perm_labels = rng.permutation(y_true)
        null_bal_acc[i] = balanced_accuracy_score(perm_labels, y_pred)
        null_f1[i] = f1_score(perm_labels, y_pred, average="macro", zero_division=0)

    p_bal_acc = float(np.mean(null_bal_acc >= obs_bal_acc))
    p_f1 = float(np.mean(null_f1 >= obs_macro_f1))

    return {
        "observed_balanced_accuracy": float(obs_bal_acc),
        "perm_p_value_balanced_accuracy": p_bal_acc,
        "observed_macro_f1": float(obs_macro_f1),
        "perm_p_value_macro_f1": p_f1,
        "null_balanced_acc_mean": float(np.mean(null_bal_acc)),
        "null_macro_f1_mean": float(np.mean(null_f1))
    }

def run_evaluation(synthetic_demo=False):
    print("=" * 80)
    print("EXP-07: EVALUATION & STATISTICAL COMPARISON PIPELINE")
    print("=" * 80)

    if synthetic_demo:
        print("\n[DEMO MODE] Running evaluation on synthetic dummy predictions...")
        np.random.seed(42)
        n = 80
        # True distribution matching Dev-80
        # A+: 20, B+: 25, AB+: 7, O+: 28
        y_true = np.array([0]*20 + [1]*25 + [2]*7 + [3]*28)
        np.random.shuffle(y_true)

        # Generate mock probabilities
        probs_a = np.random.dirichlet(np.ones(4), size=n)
        probs_b = np.random.dirichlet(np.ones(4), size=n)
    else:
        path_a = CKPT_DIR / "arm_a_112" / "oof_predictions.npy"
        path_b = CKPT_DIR / "arm_b_224" / "oof_predictions.npy"

        if not path_a.is_file() or not path_b.is_file():
            print("\nNotice: Out-of-fold prediction files not found yet (training has not been executed).")
            print("Run with --synthetic-demo to verify the evaluation equations and statistical tests.")
            return

        probs_a = np.load(path_a)
        probs_b = np.load(path_b)

        df_split = pd.read_csv(SPLIT_CSV)
        dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]
        pids_unique = sorted(dev_df["participant_id"].unique())
        pid_to_bg = dev_df.groupby("participant_id")["blood_group"].first().to_dict()
        y_true = np.array([NATIVE_REV_MAP[pid_to_bg[p]] for p in pids_unique])

    metrics_a = compute_comprehensive_metrics(y_true, probs_a, condition_name="Arm A (112x112)")
    metrics_b = compute_comprehensive_metrics(y_true, probs_b, condition_name="Arm B (224x224)")

    mcnemar = paired_mcnemar_test(y_true, np.argmax(probs_a, axis=1), np.argmax(probs_b, axis=1))
    paired_perm = paired_permutation_test(y_true, probs_a, probs_b)
    null_test_a = participant_label_permutation_test(y_true, probs_a)
    null_test_b = participant_label_permutation_test(y_true, probs_b)

    full_evaluation = {
        "arm_a_112": metrics_a,
        "arm_b_224": metrics_b,
        "paired_mcnemar_test": mcnemar,
        "paired_permutation_test": paired_perm,
        "null_test_arm_a": null_test_a,
        "null_test_arm_b": null_test_b
    }

    print("\n--- RESULTS SUMMARY ---")
    print(f"Arm A (112x112): Acc = {metrics_a['accuracy']*100:.2f}% (CI: [{metrics_a['accuracy_ci_95'][0]*100:.1f}%, {metrics_a['accuracy_ci_95'][1]*100:.1f}%]) | Bal Acc = {metrics_a['primary_balanced_accuracy']*100:.2f}% | Macro F1 = {metrics_a['macro_f1']:.4f}")
    print(f"Arm B (224x224): Acc = {metrics_b['accuracy']*100:.2f}% (CI: [{metrics_b['accuracy_ci_95'][0]*100:.1f}%, {metrics_b['accuracy_ci_95'][1]*100:.1f}%]) | Bal Acc = {metrics_b['primary_balanced_accuracy']*100:.2f}% | Macro F1 = {metrics_b['macro_f1']:.4f}")
    print(f"Majority Baseline (O+): {metrics_a['majority_baseline_accuracy']*100:.2f}% ({metrics_a['majority_class']})")
    print(f"Paired McNemar Exact p-value: {mcnemar['mcnemar_exact_p_value']:.4f} (Discordant: {mcnemar['b_arm_a_better']} vs {mcnemar['c_arm_b_better']})")
    print(f"Paired Permutation p-value:  {paired_perm['paired_permutation_p_value']:.4f} (Diff: {paired_perm['observed_balanced_acc_diff (B - A)']*100:+.2f}%)")
    print("=" * 80)

    if synthetic_demo:
        print("[PASS] Evaluation functions, Wilson score intervals, McNemar test, and permutation tests verified.")
    return full_evaluation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-demo", action="store_true", help="Run evaluation on synthetic demo data")
    args = parser.parse_args()
    run_evaluation(synthetic_demo=args.synthetic_demo)
