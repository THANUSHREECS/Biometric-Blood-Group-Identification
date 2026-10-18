#!/usr/bin/env python3
"""
EXP-10: Multi-Scan Quality and Consistency Analysis
==================================================
Objective:
  Investigate whether scan-quality variation across the three impressions
  per participant contributes to poor predictive performance, and evaluate
  whether a prespecified quality-weighted aggregation strategy outperforms
  unweighted feature averaging.

Safety & Security Safeguards:
  - Uses ONLY the 80-participant development cohort (240 scans).
  - Strictly quarantines the 21-participant locked test cohort (0 access).
  - Strictly quarantines the 26-participant hospital cohort (0 access).
  - Participant-level evaluation: identical 5-fold Stratified K-Fold splits.
  - Zero data leakage: scalers fitted strictly within training folds.
  - Zero label usage in quality formulation.
"""

import os
import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "exp10_multi_scan_quality"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

EXP_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV_MAP = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

def run_experiment():
    start_time = time.time()
    print("=" * 75)
    print("EXP-10: MULTI-SCAN QUALITY AND CONSISTENCY ANALYSIS")
    print("=" * 75)

    # 1. Security & Cohort Guard
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Missing split CSV: {SPLIT_CSV}")

    split_df = pd.read_csv(SPLIT_CSV)
    dev_df = split_df[split_df["split"].isin(["TRAIN", "VALIDATION"])].copy()

    test_cohort = split_df[split_df["split"] == "TEST"]
    assert len(test_cohort["participant_id"].unique()) == 21, "Test cohort mismatch"
    assert len(dev_df["participant_id"].unique()) == 80, "Dev cohort mismatch"
    assert len(dev_df) == 240, f"Expected 240 scans, got {len(dev_df)}"

    dev_pids = sorted(dev_df["participant_id"].unique())
    test_pids = set(test_cohort["participant_id"].unique())
    assert set(dev_pids).isdisjoint(test_pids), "FATAL: Cohort overlap!"

    print("[PASS] Security Guard: 80 Development participants (240 scans) confirmed.")
    print("[PASS] Security Guard: Locked Test (21) & Hospital (26) strictly quarantined.")

    # 2. Phase 1 & 2: Per-Scan Quality & Multi-Scan Consistency Extraction
    print("\n[Phase 1] Auditing Scan Quality across 240 scans...")
    scan_audit_records = []
    participant_consistency_records = []

    X_unweighted_list = []
    X_weighted_list = []
    y_list = []
    participant_meta = []

    for pid in dev_pids:
        p_scans = dev_df[dev_df["participant_id"] == pid].sort_values("scan_path")
        assert len(p_scans) == 3, f"Participant {pid} must have 3 scans"

        bg = p_scans.iloc[0]["blood_group"]
        bg_idx = CANONICAL_REV_MAP[bg]

        scan_tensors = []
        scan_qs = []

        for scan_idx, (_, s_row) in enumerate(p_scans.iterrows()):
            scan_path = Path(s_row["scan_path"])
            if not scan_path.is_absolute():
                scan_path = PROJECT_ROOT / scan_path
            
            img = cv2.imread(str(scan_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to load scan: {scan_path}")

            h, w = img.shape

            # 1. Foreground mask (glass platen < 238)
            mask = (img < 238).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            fg_pixels = int(np.sum(mask > 0))
            fg_coverage = float(fg_pixels / (h * w))

            # 2. Gradient Structure Tensor for Coherence
            sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            g_xx = cv2.filter2D(sobel_x**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
            g_yy = cv2.filter2D(sobel_y**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
            g_xy = cv2.filter2D(sobel_x * sobel_y, -1, np.ones((16, 16), dtype=np.float32) / 256.0)

            v_x = 2.0 * g_xy
            v_y = g_xx - g_yy
            theta = 0.5 * np.arctan2(v_x, v_y)
            coherence = np.sqrt(v_x**2 + v_y**2) / (g_xx + g_yy + 1e-6)

            # 3. Local contrast / ridge clarity (local std in 16x16 window)
            mean_filt = cv2.filter2D(img.astype(np.float32), -1, np.ones((16, 16), dtype=np.float32) / 256.0)
            sq_filt = cv2.filter2D(img.astype(np.float32)**2, -1, np.ones((16, 16), dtype=np.float32) / 256.0)
            local_std = np.sqrt(np.maximum(0, sq_filt - mean_filt**2))

            # 4. Unusable region proportion
            unusable_mask = (mask > 0) & ((coherence < 0.25) | (local_std < 12.0))
            unusable_prop = float(np.sum(unusable_mask) / fg_pixels) if fg_pixels > 0 else 1.0

            mean_coh = float(np.mean(coherence[mask > 0])) if fg_pixels > 0 else 0.0
            mean_con = float(np.mean(local_std[mask > 0])) if fg_pixels > 0 else 0.0

            # Technical quality score Q (independent of labels):
            # Q = fg_coverage * mean_coh * (1 - unusable_prop) * min(1.0, mean_con / 40.0)
            q = float(fg_coverage * mean_coh * (1.0 - unusable_prop) * min(1.0, mean_con / 40.0))
            q = max(0.001, q)

            scan_qs.append(q)

            # 5. Continuous Orientation Tensor Representation (sampled on 16x16 grid)
            theta_grid = theta[::16, ::16]
            coherence_grid = coherence[::16, ::16]
            mask_grid = cv2.resize(mask, (theta_grid.shape[1], theta_grid.shape[0]), interpolation=cv2.INTER_NEAREST)
            fg_ind = (mask_grid > 0).astype(np.float32)

            u = (np.cos(2.0 * theta_grid) * coherence_grid * fg_ind).flatten()
            v = (np.sin(2.0 * theta_grid) * coherence_grid * fg_ind).flatten()
            tensor_vec = np.concatenate([u, v])
            scan_tensors.append(tensor_vec)

            scan_audit_records.append({
                "participant_id": pid,
                "scan_index": scan_idx + 1,
                "blood_group": bg,
                "fg_coverage": fg_coverage,
                "ridge_coherence": mean_coh,
                "ridge_contrast": mean_con,
                "unusable_proportion": unusable_prop,
                "quality_score": q
            })

        # Multi-scan consistency metrics for this participant
        def cos_sim(v1, v2):
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

        s12 = cos_sim(scan_tensors[0], scan_tensors[1])
        s13 = cos_sim(scan_tensors[0], scan_tensors[2])
        s23 = cos_sim(scan_tensors[1], scan_tensors[2])
        mean_pairwise_sim = (s12 + s13 + s23) / 3.0

        q_ratio = max(scan_qs) / min(scan_qs)
        participant_consistency_records.append({
            "participant_id": pid,
            "blood_group": bg,
            "mean_cosine_sim": mean_pairwise_sim,
            "q1": scan_qs[0],
            "q2": scan_qs[1],
            "q3": scan_qs[2],
            "q_min": min(scan_qs),
            "q_max": max(scan_qs),
            "q_ratio": q_ratio
        })

        # Condition A: Unweighted mean across 3 scans
        vec_unweighted = np.mean(scan_tensors, axis=0)

        # Condition B: Quality-weighted mean across 3 scans
        weights = np.array(scan_qs, dtype=np.float32) / np.sum(scan_qs)
        vec_weighted = np.sum([w * v for w, v in zip(weights, scan_tensors)], axis=0)

        X_unweighted_list.append(vec_unweighted)
        X_weighted_list.append(vec_weighted)
        y_list.append(bg_idx)
        participant_meta.append({"participant_id": pid, "blood_group": bg, "target": bg_idx})

    df_scans = pd.DataFrame(scan_audit_records)
    df_cons = pd.DataFrame(participant_consistency_records)

    # Save quality audit artifact
    audit_csv_path = EXP_DIR / "exp10_scan_quality_audit.csv"
    df_scans.to_csv(audit_csv_path, index=False)
    print(f"Saved Scan Quality Audit Artifact: {audit_csv_path}")

    # Summary of quality audit
    print("\n[Quality Metrics Distribution (240 scans)]")
    print(f"  Foreground Coverage:  {df_scans['fg_coverage'].mean():.4f} +/- {df_scans['fg_coverage'].std():.4f}")
    print(f"  Ridge Coherence:      {df_scans['ridge_coherence'].mean():.4f} +/- {df_scans['ridge_coherence'].std():.4f}")
    print(f"  Ridge Contrast:       {df_scans['ridge_contrast'].mean():.2f} +/- {df_scans['ridge_contrast'].std():.2f}")
    print(f"  Unusable Proportion:  {df_scans['unusable_proportion'].mean():.4f} +/- {df_scans['unusable_proportion'].std():.4f}")
    print(f"  Quality Score Q:      {df_scans['quality_score'].mean():.4f} +/- {df_scans['quality_score'].std():.4f}")

    print("\n[Multi-Scan Consistency (80 participants)]")
    print(f"  Intra-participant Cosine Sim: {df_cons['mean_cosine_sim'].mean():.4f} +/- {df_cons['mean_cosine_sim'].std():.4f}")
    print(f"  Quality Ratio (Max/Min):     {df_cons['q_ratio'].mean():.2f} +/- {df_cons['q_ratio'].std():.2f}")
    print(f"  Participants with Quality Ratio >= 1.5: {np.sum(df_cons['q_ratio'] >= 1.5)} / 80 ({np.mean(df_cons['q_ratio'] >= 1.5)*100:.1f}%)")

    # 3. Phase 3 & 4: Controlled Comparison
    X_A = np.array(X_unweighted_list, dtype=np.float32)
    X_B = np.array(X_weighted_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    meta_df = pd.DataFrame(participant_meta)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "majority_baseline": {
            "name": "Majority-Class Baseline",
            "type": "baseline"
        },
        "ridge_classifier": {
            "name": "Ridge Classifier (alpha=10.0, balanced)",
            "estimator": RidgeClassifier(alpha=10.0, class_weight="balanced")
        },
        "logistic_regression": {
            "name": "L2 Logistic Regression (C=0.1, balanced)",
            "estimator": LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
        },
        "random_forest": {
            "name": "Random Forest (n=100, depth=4, balanced)",
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=42)
        }
    }

    pred_dict = {
        "participant_id": meta_df["participant_id"],
        "blood_group": meta_df["blood_group"],
        "target": y
    }

    results = {}

    print("\n" + "=" * 75)
    print("PHASE 4: CONTROLLED COMPARISON RESULTS (5-FOLD PARTICIPANT CV)")
    print("=" * 75)

    majority_class_idx = CANONICAL_REV_MAP["O+"]

    for model_key, model_cfg in models.items():
        results[model_key] = {}

        for cond_name, X_mat in [("condition_A_unweighted", X_A), ("condition_B_quality_weighted", X_B)]:
            oof_preds = np.zeros(len(y), dtype=int)
            fold_metrics = []

            for fold_idx, (t_idx, v_idx) in enumerate(skf.split(X_mat, y)):
                X_train, y_train = X_mat[t_idx], y[t_idx]
                X_val, y_val = X_mat[v_idx], y[v_idx]

                if model_key == "majority_baseline":
                    fold_pred = np.full(len(v_idx), fill_value=majority_class_idx, dtype=int)
                else:
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", model_cfg["estimator"])
                    ])
                    pipeline.fit(X_train, y_train)
                    fold_pred = pipeline.predict(X_val)

                oof_preds[v_idx] = fold_pred

                fold_acc = accuracy_score(y_val, fold_pred)
                fold_bal = balanced_accuracy_score(y_val, fold_pred)
                fold_f1 = f1_score(y_val, fold_pred, average="macro", zero_division=0)
                fold_metrics.append({
                    "fold": fold_idx + 1,
                    "accuracy": fold_acc,
                    "balanced_accuracy": fold_bal,
                    "macro_f1": fold_f1
                })

            pred_dict[f"pred_{model_key}_{cond_name}"] = oof_preds
            pred_dict[f"label_{model_key}_{cond_name}"] = [CANONICAL_CLASSES[p] for p in oof_preds]

            acc = accuracy_score(y, oof_preds)
            bal = balanced_accuracy_score(y, oof_preds)
            macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y, oof_preds)
            cm = confusion_matrix(y, oof_preds, labels=[0, 1, 2, 3]).tolist()

            per_class_rec = {}
            for c_idx, c_name in enumerate(CANONICAL_CLASSES):
                mask_c = (y == c_idx)
                rec = float(np.mean(oof_preds[mask_c] == c_idx)) if np.sum(mask_c) > 0 else 0.0
                per_class_rec[c_name] = rec

            fold_accs = [fm["accuracy"] for fm in fold_metrics]
            fold_bals = [fm["balanced_accuracy"] for fm in fold_metrics]
            fold_f1s = [fm["macro_f1"] for fm in fold_metrics]

            results[model_key][cond_name] = {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal),
                "macro_f1": float(macro_f1),
                "cohen_kappa": float(kappa),
                "per_class_recall": per_class_rec,
                "confusion_matrix": cm,
                "fold_summary": {
                    "accuracy_mean": float(np.mean(fold_accs)),
                    "accuracy_std": float(np.std(fold_accs)),
                    "balanced_accuracy_mean": float(np.mean(fold_bals)),
                    "balanced_accuracy_std": float(np.std(fold_bals)),
                    "macro_f1_mean": float(np.mean(fold_f1s)),
                    "macro_f1_std": float(np.std(fold_f1s)),
                },
                "folds": fold_metrics
            }

        # Calculate difference (Condition B - Condition A)
        diff_acc = results[model_key]["condition_B_quality_weighted"]["accuracy"] - results[model_key]["condition_A_unweighted"]["accuracy"]
        diff_bal = results[model_key]["condition_B_quality_weighted"]["balanced_accuracy"] - results[model_key]["condition_A_unweighted"]["balanced_accuracy"]
        diff_f1 = results[model_key]["condition_B_quality_weighted"]["macro_f1"] - results[model_key]["condition_A_unweighted"]["macro_f1"]
        diff_kap = results[model_key]["condition_B_quality_weighted"]["cohen_kappa"] - results[model_key]["condition_A_unweighted"]["cohen_kappa"]

        results[model_key]["difference_B_minus_A"] = {
            "delta_accuracy": float(diff_acc),
            "delta_balanced_accuracy": float(diff_bal),
            "delta_macro_f1": float(diff_f1),
            "delta_cohen_kappa": float(diff_kap)
        }

        print(f"\nModel: {model_cfg['name']}")
        print(f"  Condition A (Unweighted):       Acc = {results[model_key]['condition_A_unweighted']['accuracy']*100:5.2f}% | BalAcc = {results[model_key]['condition_A_unweighted']['balanced_accuracy']*100:5.2f}% | F1 = {results[model_key]['condition_A_unweighted']['macro_f1']:.4f} | Kappa = {results[model_key]['condition_A_unweighted']['cohen_kappa']:.4f}")
        print(f"  Condition B (Quality-Weighted): Acc = {results[model_key]['condition_B_quality_weighted']['accuracy']*100:5.2f}% | BalAcc = {results[model_key]['condition_B_quality_weighted']['balanced_accuracy']*100:5.2f}% | F1 = {results[model_key]['condition_B_quality_weighted']['macro_f1']:.4f} | Kappa = {results[model_key]['condition_B_quality_weighted']['cohen_kappa']:.4f}")
        print(f"  Difference (B - A):             Delta Acc = {diff_acc*100:+5.2f}% | Delta BalAcc = {diff_bal*100:+5.2f}% | Delta F1 = {diff_f1:+6.4f} | Delta Kappa = {diff_kap:+6.4f}")
        print(f"  AB+ Recall (A vs B):            A: {results[model_key]['condition_A_unweighted']['per_class_recall']['AB+']*100:.1f}% vs B: {results[model_key]['condition_B_quality_weighted']['per_class_recall']['AB+']*100:.1f}%")

    # 4. Save Prediction CSV and Metrics JSON
    pred_df = pd.DataFrame(pred_dict)
    pred_csv_path = EXP_DIR / "exp10_participant_oof_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nSaved OOF Predictions: {pred_csv_path}")

    total_runtime = time.time() - start_time
    results["metadata"] = {
        "experiment": "EXP-10-MULTI-SCAN-QUALITY",
        "cohort": "Dev-80 (240 scans)",
        "canonical_classes": CANONICAL_CLASSES,
        "runtime_seconds": float(total_runtime),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_json_path = EXP_DIR / "exp10_evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Metrics JSON: {metrics_json_path}")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print("=" * 75)

if __name__ == "__main__":
    run_experiment()
