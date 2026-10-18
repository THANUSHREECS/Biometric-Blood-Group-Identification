#!/usr/bin/env python3
"""
EXP-08-DERMATOGLYPHIC-MINUTIAE: Training & Cross-Validation Engine
================================================================
Trains interpretable classifiers (Majority Baseline, L2 Logistic Regression,
Random Forest) using strict 5-fold cross-validation on participant-level
handcrafted features.

Strict Safeguards:
  - Fail-closed guard: TRAINING_AUTHORIZED must be True to run on real data.
  - Imputers and scalers are fit strictly within training folds (Zero data leakage).
  - Out-of-fold predictions saved for unbiased evaluation.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp08_dermatoglyphic_minutiae.config import (
    DATA_DIR, RESULTS_DIR, TRAINING_AUTHORIZED, SEED, N_OUTER_FOLDS,
    NATIVE_CLASSES, NATIVE_REV_MAP, MODELS
)
from experiments.exp08_dermatoglyphic_minutiae.participant_feature_builder import (
    build_development_feature_matrix
)

def build_model_pipeline(model_key: str):
    """
    Builds a leak-free scikit-learn Pipeline with median imputation and scaling.
    """
    if model_key == "majority_baseline":
        clf = DummyClassifier(strategy="most_frequent")
        return Pipeline([("clf", clf)])

    elif model_key == "logistic_regression":
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        clf = LogisticRegression(
            penalty="l2",
            C=MODELS["logistic_regression"]["C"],
            max_iter=MODELS["logistic_regression"]["max_iter"],
            random_state=SEED,
            class_weight="balanced"
        )
        return Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])

    elif model_key == "random_forest":
        imputer = SimpleImputer(strategy="median")
        clf = RandomForestClassifier(
            n_estimators=MODELS["random_forest"]["n_estimators"],
            max_depth=MODELS["random_forest"]["max_depth"],
            min_samples_split=MODELS["random_forest"]["min_samples_split"],
            min_samples_leaf=MODELS["random_forest"]["min_samples_leaf"],
            random_state=SEED,
            class_weight="balanced"
        )
        return Pipeline([("imputer", imputer), ("clf", clf)])

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

def train_and_evaluate_models(feature_df=None):
    """
    Executes 5-fold cross-validation for all three models on participant-level features.
    """
    # FAIL-CLOSED AUTHORIZATION GUARD
    if not TRAINING_AUTHORIZED:
        raise PermissionError(
            "\n[SAFETY BLOCK] TRAINING_AUTHORIZED is False.\n"
            "Execution on real participant data is locked. Run with synthetic test first.\n"
            "Explicit authorization required from senior research supervisor."
        )

    if feature_df is None:
        feat_csv = DATA_DIR / "dev80_handcrafted_features.csv"
        if not feat_csv.is_file():
            print("Feature matrix not found. Extracting from Dev-80 scans...")
            feature_df = build_development_feature_matrix(save_csv=True)
        else:
            feature_df = pd.read_csv(feat_csv)

    print("=" * 80)
    print("EXP-08: TRAINING INTERPRETABLE DERMATOGLYPHIC CLASSIFIERS")
    print(f"Cohort: {len(feature_df)} Dev Participants | Folds: {N_OUTER_FOLDS}")
    print("=" * 80)

    # Prepare features and targets
    meta_cols = ["participant_id", "blood_group", "target_label"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols and not c.startswith("majority_pattern")]
    
    # Exclude non-numeric string columns
    numeric_feature_cols = [c for c in feature_cols if np.issubdtype(feature_df[c].dtype, np.number)]
    print(f"Extracted {len(numeric_feature_cols)} numeric features per participant:")
    for fc in numeric_feature_cols:
        print(f"  - {fc}")

    X = feature_df[numeric_feature_cols].values
    y = feature_df["target_label"].values
    groups = feature_df["participant_id"].values

    sgkf = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(X, y, groups))

    all_results = {}

    for model_key, spec in MODELS.items():
        print(f"\n--- Training {spec['name']} ({model_key}) ---")
        oof_probs = np.zeros((len(y), 4), dtype=np.float32)

        for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
            X_train, y_train = X[t_idx], y[t_idx]
            X_val, y_val = X[v_idx], y[v_idx]

            pipeline = build_model_pipeline(model_key)
            pipeline.fit(X_train, y_train)

            # Predict probabilities
            if hasattr(pipeline.named_steps["clf"], "predict_proba"):
                probs = pipeline.predict_proba(X_val)
                # Check for models that might drop unrepresented classes
                if probs.shape[1] < 4:
                    full_probs = np.zeros((len(v_idx), 4), dtype=np.float32)
                    for col_idx, cls_label in enumerate(pipeline.named_steps["clf"].classes_):
                        full_probs[:, cls_label] = probs[:, col_idx]
                    probs = full_probs
            else:
                # Majority baseline predicts argmax
                preds = pipeline.predict(X_val)
                probs = np.zeros((len(v_idx), 4), dtype=np.float32)
                for i, p in enumerate(preds):
                    probs[i, p] = 1.0

            oof_probs[v_idx] = probs
            f_acc = accuracy_score(y_val, np.argmax(probs, axis=1))
            f_bal = balanced_accuracy_score(y_val, np.argmax(probs, axis=1))
            f_f1 = f1_score(y_val, np.argmax(probs, axis=1), average="macro", zero_division=0)
            print(f"  Fold {f_idx}: OOF Acc = {f_acc*100:.2f}% | Bal Acc = {f_bal*100:.2f}% | Macro F1 = {f_f1:.4f}")

        # Overall OOF metrics
        oof_preds = np.argmax(oof_probs, axis=1)
        tot_acc = accuracy_score(y, oof_preds)
        tot_bal = balanced_accuracy_score(y, oof_preds)
        tot_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)

        print(f"  => OVERALL OOF: Acc = {tot_acc*100:.2f}% | Bal Acc = {tot_bal*100:.2f}% | Macro F1 = {tot_f1:.4f}")

        # Save OOF predictions
        np.save(RESULTS_DIR / f"{model_key}_oof_probs.npy", oof_probs)
        all_results[model_key] = {
            "accuracy": float(tot_acc),
            "balanced_accuracy": float(tot_bal),
            "macro_f1": float(tot_f1)
        }

    # Construct and save full participant-level out-of-fold predictions CSV
    fold_col = np.zeros(len(y), dtype=int)
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        fold_col[v_idx] = f_idx

    oof_df = pd.DataFrame({
        "participant_id": feature_df["participant_id"],
        "blood_group": feature_df["blood_group"],
        "target_label": feature_df["target_label"],
        "fold": fold_col
    })

    for model_key in MODELS.keys():
        probs = np.load(RESULTS_DIR / f"{model_key}_oof_probs.npy")
        preds = np.argmax(probs, axis=1)
        oof_df[f"{model_key}_pred_label"] = preds
        oof_df[f"{model_key}_pred_blood_group"] = [NATIVE_CLASSES[p] for p in preds]
        for c_idx, c_name in enumerate(NATIVE_CLASSES):
            oof_df[f"{model_key}_prob_{c_name}"] = probs[:, c_idx]

    pred_csv = RESULTS_DIR / "exp08_participant_oof_predictions.csv"
    oof_df.to_csv(pred_csv, index=False)
    print(f"\nParticipant-level out-of-fold predictions saved to: {pred_csv}")

    # Save feature names list
    with open(DATA_DIR / "feature_names.json", "w") as f:
        json.dump(numeric_feature_cols, f, indent=2)

    return all_results

if __name__ == "__main__":
    try:
        train_and_evaluate_models()
    except PermissionError as e:
        print(e)
        sys.exit(0)
