#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Comprehensive Evaluation & Statistical Testing
=============================================================================
Calculates participant-level out-of-fold balanced accuracy (primary metric),
macro F1, Wilson 95% confidence intervals, canonical confusion matrices,
and non-parametric permutation null distributions.
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

from experiments.exp08_dermatoglyphic_minutiae.config import (
    RESULTS_DIR, SPLIT_CSV, NATIVE_CLASSES, NATIVE_REV_MAP,
    CANONICAL_CLASSES, CANONICAL_REV_MAP, MODELS
)

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def evaluate_model_predictions(y_true, y_probs, model_name="Model"):
    y_pred = np.argmax(y_probs, axis=1)
    n = len(y_true)
    k = int(np.sum(y_true == y_pred))

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    ci_low, ci_high = compute_wilson_ci(k, n)

    # Per-class metrics in native order
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(4)), zero_division=0)
    cm_native = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    per_class = {}
    for i, c in enumerate(NATIVE_CLASSES):
        per_class[c] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i])
        }

    # Majority baseline
    counts = np.bincount(y_true)
    majority_class_idx = int(np.argmax(counts))
    majority_count = int(np.max(counts))
    majority_baseline = majority_count / n

    # Canonical remapping (swap index 1: B+ and 2: AB+)
    canon_map = [0, 2, 1, 3]
    y_true_canon = np.array([canon_map[val] for val in y_true])
    y_pred_canon = np.array([canon_map[val] for val in y_pred])
    cm_canonical = confusion_matrix(y_true_canon, y_pred_canon, labels=list(range(4)))

    macro_precision = float(np.mean(p))
    macro_recall = float(np.mean(r))

    return {
        "model_name": model_name,
        "n_participants": n,
        "correct_predictions": k,
        "accuracy": float(acc),
        "accuracy_ci_95": [float(ci_low), float(ci_high)],
        "primary_balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "cohen_kappa": float(kappa),
        "majority_class": NATIVE_CLASSES[majority_class_idx],
        "majority_baseline_accuracy": float(majority_baseline),
        "native_class_order": NATIVE_CLASSES,
        "native_per_class_metrics": per_class,
        "native_confusion_matrix": cm_native.tolist(),
        "canonical_class_order": CANONICAL_CLASSES,
        "canonical_confusion_matrix": cm_canonical.tolist()
    }

def participant_label_permutation_test(y_true, y_probs, n_permutations=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_pred = np.argmax(y_probs, axis=1)
    obs_bal = balanced_accuracy_score(y_true, y_pred)
    obs_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    null_bal = np.zeros(n_permutations, dtype=np.float32)
    null_f1 = np.zeros(n_permutations, dtype=np.float32)

    for i in range(n_permutations):
        perm_y = rng.permutation(y_true)
        null_bal[i] = balanced_accuracy_score(perm_y, y_pred)
        null_f1[i] = f1_score(perm_y, y_pred, average="macro", zero_division=0)

    p_bal = float(np.mean(null_bal >= obs_bal))
    p_f1 = float(np.mean(null_f1 >= obs_f1))

    return {
        "observed_balanced_accuracy": float(obs_bal),
        "perm_p_balanced_accuracy": p_bal,
        "observed_macro_f1": float(obs_f1),
        "perm_p_macro_f1": p_f1,
        "null_balanced_acc_mean": float(np.mean(null_bal)),
        "null_macro_f1_mean": float(np.mean(null_f1))
    }

def run_evaluation(synthetic_demo=False):
    print("=" * 80)
    print("EXP-08: DERMATOGLYPHIC FEATURE EVALUATION PIPELINE")
    print("=" * 80)

    if synthetic_demo:
        print("\n[DEMO MODE] Running evaluation on synthetic dummy predictions...")
        np.random.seed(42)
        n = 80
        y_true = np.array([0]*20 + [1]*25 + [2]*7 + [3]*28)
        np.random.shuffle(y_true)
        probs_dict = {
            m: np.random.dirichlet(np.ones(4), size=n) for m in MODELS.keys()
        }
    else:
        df_split = pd.read_csv(SPLIT_CSV)
        dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]
        pids = sorted(dev_df["participant_id"].unique())
        pid_to_bg = dev_df.groupby("participant_id")["blood_group"].first().to_dict()
        y_true = np.array([NATIVE_REV_MAP[pid_to_bg[p]] for p in pids])

        probs_dict = {}
        for m in MODELS.keys():
            p_file = RESULTS_DIR / f"{m}_oof_probs.npy"
            if not p_file.is_file():
                print(f"Notice: Prediction file not found: {p_file}. Run with --synthetic-demo first.")
                return None
            probs_dict[m] = np.load(p_file)

    eval_results = {}
    for m_key, probs in probs_dict.items():
        m_eval = evaluate_model_predictions(y_true, probs, model_name=MODELS[m_key]["name"])
        null_test = participant_label_permutation_test(y_true, probs)
        m_eval["permutation_null_test"] = null_test
        eval_results[m_key] = m_eval

        print(f"\nModel: {m_eval['model_name']}")
        print(f"  Accuracy:         {m_eval['accuracy']*100:.2f}% (95% CI: [{m_eval['accuracy_ci_95'][0]*100:.1f}%, {m_eval['accuracy_ci_95'][1]*100:.1f}%])")
        print(f"  Balanced Accuracy:{m_eval['primary_balanced_accuracy']*100:.2f}% (Permutation p = {null_test['perm_p_balanced_accuracy']:.4f})")
        print(f"  Macro F1:         {m_eval['macro_f1']:.4f} (Permutation p = {null_test['perm_p_macro_f1']:.4f})")

    # Save to json if not synthetic
    if not synthetic_demo:
        out_json = RESULTS_DIR / "exp08_evaluation_metrics.json"
        with open(out_json, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved evaluation metrics to: {out_json}")

    print("=" * 80)
    return eval_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args()
    run_evaluation(synthetic_demo=args.synthetic_demo)
