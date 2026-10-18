#!/usr/bin/env python3
"""
EXP-08: Full-Pipeline Cross-Validation Permutation Test
======================================================
Executes a rigorous participant-level permutation test of the complete
cross-validation and model-fitting procedures for EXP-08.

Methodological Safeguards:
  1. Uses only the authorized 80-participant development dataset.
  2. Uses the exact 24 numeric handcrafted features.
  3. Uses build_model_pipeline from train.py to guarantee identical hyperparameters:
     - L2 Logistic Regression (C=1.0, max_iter=1000, penalty="l2", class_weight="balanced")
     - Random Forest (n_estimators=100, max_depth=4, min_samples_split=4, min_samples_leaf=2, class_weight="balanced")
  4. For each of 1,000 permutations per model:
     - Participant labels are permuted across the 80 Dev participants (preserving marginal class counts).
     - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) re-stratifies folds under Y_perm.
     - Imputation and scaling are fit strictly on training folds.
     - Classifiers are refit strictly on training folds.
     - Exactly one out-of-fold prediction is generated per participant.
     - Out-of-fold balanced accuracy and macro F1 are calculated.
  5. Empirical p-values are computed with unbiased pseudo-counts:
     p = (1 + count(null >= observed)) / (B + 1)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.train import build_model_pipeline
from experiments.exp08_dermatoglyphic_minutiae.config import SEED, N_OUTER_FOLDS

EXP_DIR = PROJECT_ROOT / "experiments" / "exp08_dermatoglyphic_minutiae"
OUTPUT_DIR = EXP_DIR / "permutation_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 1000

def evaluate_cv(X, y, groups, model_key):
    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(y), 4), dtype=np.float32)
    
    for t_idx, v_idx in sgkf.split(X, y, groups):
        X_train, y_train = X[t_idx], y[t_idx]
        X_val, y_val = X[v_idx], y[v_idx]
        
        pipe = build_model_pipeline(model_key)
        pipe.fit(X_train, y_train)
        
        probs = pipe.predict_proba(X_val)
        if probs.shape[1] < 4:
            full_probs = np.zeros((len(v_idx), 4), dtype=np.float32)
            for col_idx, cls_label in enumerate(pipe.named_steps["clf"].classes_):
                full_probs[:, cls_label] = probs[:, col_idx]
            probs = full_probs
            
        oof_probs[v_idx] = probs
        
    preds = np.argmax(oof_probs, axis=1)
    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    return acc, bal_acc, macro_f1, preds

def main():
    print("=" * 80)
    print("EXP-08: FULL-PIPELINE PERMUTATION TEST (1,000 ITERATIONS PER MODEL)")
    print("=" * 80)
    
    # 1. Load data
    feat_csv = EXP_DIR / "features" / "dev80_handcrafted_features.csv"
    feat_names_json = EXP_DIR / "features" / "feature_names.json"
    
    assert feat_csv.is_file(), f"Feature file missing: {feat_csv}"
    assert feat_names_json.is_file(), f"Feature names missing: {feat_names_json}"
    
    df = pd.read_csv(feat_csv)
    with open(feat_names_json) as f:
        feature_cols = json.load(f)
        
    assert len(df) == 80, f"Expected 80 participants, got {len(df)}"
    assert len(feature_cols) == 24, f"Expected 24 numeric features, got {len(feature_cols)}"
    
    X = df[feature_cols].values
    y_true = df["target_label"].values
    groups = df["participant_id"].values
    
    print(f"Cohort size: {len(df)} Dev participants | Features: {len(feature_cols)} numeric")
    print(f"Class distribution: {dict(pd.Series(y_true).value_counts().sort_index())}")
    
    # 2. Pre-execution verification of observed scores
    print("\n--- Pre-Execution Verification: Observed Scores ---")
    observed_metrics = {}
    for m_key in ["logistic_regression", "random_forest"]:
        obs_acc, obs_bal, obs_f1, obs_preds = evaluate_cv(X, y_true, groups, m_key)
        observed_metrics[m_key] = {
            "accuracy": float(obs_acc),
            "balanced_accuracy": float(obs_bal),
            "macro_f1": float(obs_f1)
        }
        print(f"Observed {m_key}: Acc={obs_acc*100:.2f}% | BalAcc={obs_bal*100:.2f}% | MacroF1={obs_f1:.4f}")
        
    # Check exact agreement with saved EXP-08 final results
    assert abs(observed_metrics["logistic_regression"]["balanced_accuracy"] - 0.2115426) < 1e-4, "LogReg BalAcc mismatch!"
    assert abs(observed_metrics["random_forest"]["balanced_accuracy"] - 0.2705626) < 1e-4, "Random Forest BalAcc mismatch!"
    print("[PASS] Observed metrics verified 100% identical to EXP-08 final results.")
    
    # 3. Permutation Loop for each model
    results_summary = {}
    
    for m_key in ["logistic_regression", "random_forest"]:
        print(f"\n========================================================")
        print(f"Running 1,000 Full-Pipeline Permutations for {m_key}...")
        print(f"========================================================")
        
        obs_bal = observed_metrics[m_key]["balanced_accuracy"]
        obs_f1 = observed_metrics[m_key]["macro_f1"]
        obs_acc = observed_metrics[m_key]["accuracy"]
        
        null_records = []
        t0 = time.time()
        
        for b in range(1, N_PERMUTATIONS + 1):
            perm_rng = np.random.RandomState(SEED + b * 10007)
            y_perm = perm_rng.permutation(y_true)
            
            # Full 5-fold CV with re-stratified partitions under y_perm
            acc_b, bal_b, f1_b, preds_b = evaluate_cv(X, y_perm, groups, m_key)
            
            null_records.append({
                "permutation_id": b,
                "seed": int(SEED + b * 10007),
                "accuracy": float(acc_b),
                "balanced_accuracy": float(bal_b),
                "macro_f1": float(f1_b),
                "bal_acc_ge_observed": int(bal_b >= obs_bal),
                "macro_f1_ge_observed": int(f1_b >= obs_f1)
            })
            
            if b % 100 == 0 or b == N_PERMUTATIONS:
                elapsed = time.time() - t0
                rate = b / elapsed
                rem = (N_PERMUTATIONS - b) / rate if rate > 0 else 0
                print(f"  Iteration {b:4d}/{N_PERMUTATIONS} | Elapsed: {elapsed/60:.2f}m | Remaining: {rem/60:.2f}m")
                
        df_null = pd.DataFrame(null_records)
        
        # Save null distribution CSV
        csv_out = OUTPUT_DIR / f"{m_key}_null_distribution.csv"
        df_null.to_csv(csv_out, index=False)
        print(f"\nSaved null distribution to: {csv_out}")
        
        # Calculate statistics
        null_bal_arr = df_null["balanced_accuracy"].values
        null_f1_arr = df_null["macro_f1"].values
        
        count_bal_ge = int(np.sum(null_bal_arr >= obs_bal))
        count_f1_ge = int(np.sum(null_f1_arr >= obs_f1))
        
        p_val_bal = (1.0 + count_bal_ge) / (N_PERMUTATIONS + 1.0)
        p_val_f1 = (1.0 + count_f1_ge) / (N_PERMUTATIONS + 1.0)
        
        stats_m = {
            "model_key": m_key,
            "n_permutations": N_PERMUTATIONS,
            "observed_accuracy": float(obs_acc),
            "observed_balanced_accuracy": float(obs_bal),
            "observed_macro_f1": float(obs_f1),
            "null_balanced_accuracy_mean": float(np.mean(null_bal_arr)),
            "null_balanced_accuracy_std": float(np.std(null_bal_arr)),
            "null_balanced_accuracy_median": float(np.median(null_bal_arr)),
            "null_balanced_accuracy_p95": float(np.percentile(null_bal_arr, 95)),
            "null_balanced_accuracy_max": float(np.max(null_bal_arr)),
            "count_balanced_accuracy_ge_observed": count_bal_ge,
            "p_value_balanced_accuracy": float(p_val_bal),
            "null_macro_f1_mean": float(np.mean(null_f1_arr)),
            "null_macro_f1_std": float(np.std(null_f1_arr)),
            "null_macro_f1_p95": float(np.percentile(null_f1_arr, 95)),
            "count_macro_f1_ge_observed": count_f1_ge,
            "p_value_macro_f1": float(p_val_f1)
        }
        
        results_summary[m_key] = stats_m
        
        print(f"\n=== RESULTS FOR {m_key.upper()} ===")
        print(f"Observed Balanced Accuracy:  {obs_bal*100:.2f}%")
        print(f"Null Distribution (Mean+-Std): {stats_m['null_balanced_accuracy_mean']*100:.2f}% +- {stats_m['null_balanced_accuracy_std']*100:.2f}%")
        print(f"Null 95th Percentile:        {stats_m['null_balanced_accuracy_p95']*100:.2f}%")
        print(f"Null Permutations >= Obs:    {count_bal_ge}/{N_PERMUTATIONS}")
        print(f"Empirical p-value:           {p_val_bal:.4f}")
        print(f"Observed Macro F1:           {obs_f1:.4f} (Null mean: {stats_m['null_macro_f1_mean']:.4f}, p = {p_val_f1:.4f})")
        
    # Save machine-readable JSON
    json_out = OUTPUT_DIR / "exp08_full_pipeline_permutation_metrics.json"
    with open(json_out, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nAll permutation metrics saved to: {json_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
