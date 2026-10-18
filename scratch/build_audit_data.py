#!/usr/bin/env python3
"""
Scratch script to verify metric recomputation across all experiments.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix
)

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CANONICAL_REV = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Verify Dev-80 split
split_df = pd.read_csv(PROJECT_ROOT / "reports/phase3_participant_benchmark/participant_split.csv")
dev_df = split_df[split_df["split"].isin(["TRAIN", "VALIDATION"])]
dev_pids = sorted(dev_df["participant_id"].unique())
pid_to_bg = dev_df.groupby("participant_id")["blood_group"].first().to_dict()

print(f"Dev-80 participants: {len(dev_pids)}")
assert len(dev_pids) == 80

# List of all experiment predictions to audit
exp_configs = [
    {
        "exp_id": "EXP-01",
        "name": "Class-Balanced Focal Loss",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp01/e2_focal_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("focal_loss", "predicted_label")],
        "reported": {"accuracy": 0.3625, "balanced_accuracy": 0.3282, "macro_f1": 0.3205}
    },
    {
        "exp_id": "EXP-02",
        "name": "Conservative Augmentation",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp02/e2_augmented_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("augmented_model", "predicted_label")],
        "reported": {"accuracy": 0.4750, "balanced_accuracy": 0.4360, "macro_f1": 0.4632}
    },
    {
        "exp_id": "EXP-03",
        "name": "Augmentation Ablation Arm 0 (Original E2 Aug)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp03/arm0_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("arm0", "predicted_label")],
        "reported": {"accuracy": 0.4875, "balanced_accuracy": 0.4256, "macro_f1": 0.4293}
    },
    {
        "exp_id": "EXP-03",
        "name": "Augmentation Ablation Arm 1 (Native Scans, No Aug)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp03/arm1_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("arm1_native", "predicted_label")],
        "reported": {"accuracy": 0.5375, "balanced_accuracy": 0.5284, "macro_f1": 0.5068}
    },
    {
        "exp_id": "EXP-03",
        "name": "Augmentation Ablation Arm 2 (Conservative Aug)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp03/arm2_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("arm2_conservative", "predicted_label")],
        "reported": {"accuracy": 0.3875, "balanced_accuracy": 0.4099, "macro_f1": 0.4220}
    },
    {
        "exp_id": "EXP-04",
        "name": "Fine-Tuning Depth Config A (Frozen Reference)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp04/config_a_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_a_frozen", "predicted_label")],
        "reported": {"accuracy": 0.5000, "balanced_accuracy": 0.4786, "macro_f1": 0.4781}
    },
    {
        "exp_id": "EXP-04",
        "name": "Fine-Tuning Depth Config B (DenseBlock 4 Trainable)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp04/config_b_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_b_db4", "predicted_label")],
        "reported": {"accuracy": 0.4125, "balanced_accuracy": 0.4297, "macro_f1": 0.4032}
    },
    {
        "exp_id": "EXP-04",
        "name": "Fine-Tuning Depth Config C (DenseBlocks 3+4 Trainable)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp04/config_c_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_c_db34", "predicted_label")],
        "reported": {"accuracy": 0.4750, "balanced_accuracy": 0.3904, "macro_f1": 0.3789}
    },
    {
        "exp_id": "EXP-05",
        "name": "Feature Fusion Config A (Concatenation 3072)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp05/config_a_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_a_concat", "predicted_label")],
        "reported": {"accuracy": 0.5250, "balanced_accuracy": 0.5184, "macro_f1": 0.5107}
    },
    {
        "exp_id": "EXP-05",
        "name": "Feature Fusion Config B (Gated Fusion)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp05/config_b_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_b_gated", "predicted_label")],
        "reported": {"accuracy": 0.5250, "balanced_accuracy": 0.5124, "macro_f1": 0.5023}
    },
    {
        "exp_id": "EXP-05",
        "name": "Feature Fusion Config C (Cross-Scan Self-Attention)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp05/config_c_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_c_attention", "predicted_label")],
        "reported": {"accuracy": 0.4500, "balanced_accuracy": 0.4809, "macro_f1": 0.4535}
    },
    {
        "exp_id": "EXP-06A",
        "name": "SupCon Temperature Config B (tau=0.05)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp06a/config_b_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_b_tau005", "predicted_label")],
        "reported": {"accuracy": 0.5250, "balanced_accuracy": 0.5518, "macro_f1": 0.5313}
    },
    {
        "exp_id": "EXP-06B",
        "name": "SupCon Loss Weight Arm D (lambda=0.50)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp06b/arm_d_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("arm_d_lam050", "predicted_label")],
        "reported": {"accuracy": 0.5250, "balanced_accuracy": 0.5146, "macro_f1": 0.4925}
    },
    {
        "exp_id": "EXP-06C",
        "name": "Head Architecture Config A (Linear Head)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_a_linear_head", "predicted_label")],
        "reported": {"accuracy": 0.5375, "balanced_accuracy": 0.5347, "macro_f1": 0.5487}
    },
    {
        "exp_id": "EXP-06C",
        "name": "Head Architecture Config B (Two-Layer MLP Head)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp06c/config_b_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_b_mlp_head", "predicted_label")],
        "reported": {"accuracy": 0.4625, "balanced_accuracy": 0.5212, "macro_f1": 0.4829}
    },
    {
        "exp_id": "EXP-06D",
        "name": "Preprocessing Config B (Raw Platen C5, No CLAHE)",
        "path": PROJECT_ROOT / "reports/accuracy_improvement/exp06d/config_b_oof_predictions.csv",
        "type": "csv_standard",
        "models": [("config_b_raw_platen", "predicted_label")],
        "reported": {"accuracy": 0.5250, "balanced_accuracy": 0.5007, "macro_f1": 0.5078}
    },
    {
        "exp_id": "EXP-07",
        "name": "Resolution Fidelity Arm A (112x112)",
        "path": PROJECT_ROOT / "experiments/exp07_resolution_fidelity/checkpoints/arm_a_112/oof_predictions.npy",
        "type": "npy_probs",
        "models": [("arm_a_112", 0)],
        "reported": {"accuracy": 0.3625, "balanced_accuracy": 0.2861, "macro_f1": 0.2800}
    },
    {
        "exp_id": "EXP-07",
        "name": "Resolution Fidelity Arm B (224x224)",
        "path": PROJECT_ROOT / "experiments/exp07_resolution_fidelity/checkpoints/arm_b_224/oof_predictions.npy",
        "type": "npy_probs",
        "models": [("arm_b_224", 0)],
        "reported": {"accuracy": 0.3375, "balanced_accuracy": 0.2615, "macro_f1": 0.2506}
    },
    {
        "exp_id": "EXP-08",
        "name": "Handcrafted Dermatoglyphic Minutiae",
        "path": PROJECT_ROOT / "experiments/exp08_dermatoglyphic_minutiae/results/exp08_participant_oof_predictions.csv",
        "type": "csv_exp08",
        "models": [
            ("majority_baseline", "majority_baseline_pred_blood_group"),
            ("logistic_regression", "logistic_regression_pred_blood_group"),
            ("random_forest", "random_forest_pred_blood_group")
        ],
        "reported": {
            "majority_baseline": {"accuracy": 0.3625, "balanced_accuracy": 0.2500, "macro_f1": 0.1330},
            "logistic_regression": {"accuracy": 0.3000, "balanced_accuracy": 0.2336, "macro_f1": 0.2241},
            "random_forest": {"accuracy": 0.3500, "balanced_accuracy": 0.2706, "macro_f1": 0.2603}
        }
    },
    {
        "exp_id": "EXP-09",
        "name": "Continuous Orientation Tensor Field",
        "path": PROJECT_ROOT / "experiments/exp09_orientation_tensor/exp09_participant_oof_predictions.csv",
        "type": "csv_exp09",
        "models": [
            ("majority_baseline", "pred_label_majority_baseline"),
            ("ridge_classifier", "pred_label_ridge_classifier"),
            ("logistic_regression", "pred_label_logistic_regression"),
            ("random_forest", "pred_label_random_forest")
        ],
        "reported": {
            "majority_baseline": {"accuracy": 0.3625, "balanced_accuracy": 0.2500, "macro_f1": 0.1330},
            "ridge_classifier": {"accuracy": 0.3625, "balanced_accuracy": 0.2920, "macro_f1": 0.2770},
            "logistic_regression": {"accuracy": 0.3500, "balanced_accuracy": 0.2897, "macro_f1": 0.2775},
            "random_forest": {"accuracy": 0.4000, "balanced_accuracy": 0.3137, "macro_f1": 0.3048}
        }
    },
    {
        "exp_id": "EXP-10",
        "name": "Multi-Scan Quality & Consistency",
        "path": PROJECT_ROOT / "experiments/exp10_multi_scan_quality/exp10_participant_oof_predictions.csv",
        "type": "csv_exp10",
        "models": [
            ("majority_baseline_A", "label_majority_baseline_condition_A_unweighted"),
            ("ridge_classifier_A", "label_ridge_classifier_condition_A_unweighted"),
            ("ridge_classifier_B", "label_ridge_classifier_condition_B_quality_weighted"),
            ("logistic_regression_A", "label_logistic_regression_condition_A_unweighted"),
            ("logistic_regression_B", "label_logistic_regression_condition_B_quality_weighted"),
            ("random_forest_A", "label_random_forest_condition_A_unweighted"),
            ("random_forest_B", "label_random_forest_condition_B_quality_weighted")
        ],
        "reported": {
            "majority_baseline_A": {"accuracy": 0.3625, "balanced_accuracy": 0.2500, "macro_f1": 0.1330},
            "ridge_classifier_A": {"accuracy": 0.3625, "balanced_accuracy": 0.2920, "macro_f1": 0.2770},
            "ridge_classifier_B": {"accuracy": 0.3625, "balanced_accuracy": 0.2875, "macro_f1": 0.2730},
            "logistic_regression_A": {"accuracy": 0.3500, "balanced_accuracy": 0.2897, "macro_f1": 0.2775},
            "logistic_regression_B": {"accuracy": 0.3375, "balanced_accuracy": 0.2765, "macro_f1": 0.2650},
            "random_forest_A": {"accuracy": 0.4000, "balanced_accuracy": 0.3137, "macro_f1": 0.3048},
            "random_forest_B": {"accuracy": 0.3375, "balanced_accuracy": 0.2615, "macro_f1": 0.2503}
        }
    }
]

print("Starting verification loop...")
for cfg in exp_configs:
    exp_id = cfg["exp_id"]
    p_path = cfg["path"]
    if not p_path.exists():
        print(f"[MISSING] {exp_id}: {p_path}")
        continue
    
    if cfg["type"] == "npy_probs":
        probs = np.load(p_path)
        y_true = np.array([pid_to_bg[p] for p in dev_pids])
        # In EXP-07 native order was ["A+", "B+", "AB+", "O+"]
        native_classes = ["A+", "B+", "AB+", "O+"]
        y_pred = [native_classes[i] for i in np.argmax(probs, axis=1)]
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rep = cfg["reported"]
        print(f"[{exp_id}] {cfg['name']}: Recalc Acc={acc*100:.2f}% (Rep={rep['accuracy']*100:.2f}%), BalAcc={bal_acc*100:.2f}% (Rep={rep['balanced_accuracy']*100:.2f}%), F1={f1:.4f} (Rep={rep['macro_f1']:.4f})")
    elif cfg["type"] in ["csv_standard", "csv_exp08", "csv_exp09", "csv_exp10"]:
        df = pd.read_csv(p_path)
        assert len(df) == 80, f"Expected 80 rows, got {len(df)}"
        assert df["participant_id"].nunique() == 80, "Non-unique PIDs!"
        
        for m_key, col in cfg["models"]:
            if col not in df.columns:
                print(f"Column {col} not found in {p_path}!")
                continue
            
            y_pred = df[col].values
            y_true = [pid_to_bg[p] for p in df["participant_id"]]
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
            # Check reported
            rep = cfg["reported"].get(m_key, cfg["reported"])
            diff_acc = abs(acc - rep["accuracy"])
            diff_bal = abs(bal_acc - rep["balanced_accuracy"])
            diff_f1 = abs(f1 - rep["macro_f1"])
            status = "PASS" if max(diff_acc, diff_bal, diff_f1) < 0.005 else "DIFF"
            print(f"[{exp_id} - {m_key}] {status}: Recalc Acc={acc*100:.2f}% (Rep={rep['accuracy']*100:.2f}%), BalAcc={bal_acc*100:.2f}% (Rep={rep['balanced_accuracy']*100:.2f}%), F1={f1:.4f} (Rep={rep['macro_f1']:.4f})")

print("All tested configurations loaded and verified!")
