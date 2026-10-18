"""
Safe, Reproducible Independent Evaluation Entry Point for Frozen Candidate (EXP-06C Config A).

STRICT SAFETY CONTROLS:
- TRAINING_AUTHORIZED is hardcoded to False.
- EVALUATION_AUTHORIZED is False by default. Real-cohort evaluation refuses to execute
  unless explicit authorization is verified.
- Cryptographic checkpoint verification (SHA-256) is mandatory before any model is instantiated.
- Evaluation is strictly restricted to the 21-participant locked test cohort.
- Hospital cohort (data/captured) remains strictly inaccessible.
- Output files are write-protected against accidental overwrites.
- Provides a synthetic testing mode (--synthetic_test) for functional validation
  without touching any protected files.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from scipy.stats import binomtest
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from src.preprocessing import transform_c5

# Enforce fail-closed safety guards
TRAINING_AUTHORIZED = False
EVALUATION_AUTHORIZED = False

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# Cryptographic SHA-256 ledger for the 5 frozen EXP-06C Config A checkpoints
FROZEN_CHECKPOINT_LEDGER = {
    1: {
        "path": "experiments/exp06c/checkpoints/config_a/fold_1_best.weights.h5",
        "sha256": "ba4917ce70382e33ec2041de0ac510ad23e32a4eb83675a5c0c703c1064e2afc",
        "size_bytes": 52899216
    },
    2: {
        "path": "experiments/exp06c/checkpoints/config_a/fold_2_best.weights.h5",
        "sha256": "615bed411044ccb82612b53083f744dc6ce3f868668c72ea5b2fedc3ca8e6d6c",
        "size_bytes": 52899472
    },
    3: {
        "path": "experiments/exp06c/checkpoints/config_a/fold_3_best.weights.h5",
        "sha256": "e36963fddede5c53ed474d497e6ca15ee1db754d215bebd48e70970aebe90ec1",
        "size_bytes": 52899472
    },
    4: {
        "path": "experiments/exp06c/checkpoints/config_a/fold_4_best.weights.h5",
        "sha256": "58bbf9643c0e200c0bbc7608284e77ba7a9e646918222195cc11fd7dfde93de3",
        "size_bytes": 52899472
    },
    5: {
        "path": "experiments/exp06c/checkpoints/config_a/fold_5_best.weights.h5",
        "sha256": "6d15284af05cba22d95a8b49b011f32a46cee085e641ee40ea3da36b089f4fe6",
        "size_bytes": 52899472
    }
}

# Historical Phase 3 Baseline E2 performance on the 21 test participants
HISTORICAL_PHASE3_BASELINE = {
    "accuracy": 0.380952,
    "balanced_accuracy": 0.306250,
    "macro_f1": 0.250000,
    "correct_count": 8,
    "total_participants": 21
}


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_integrity(base_dir: str = ".") -> bool:
    """
    Verifies that all 5 candidate checkpoints exist and match their expected SHA-256 hashes.
    Raises RuntimeError on any mismatch.
    """
    print("\n[SECURITY] Verifying cryptographic checkpoint integrity...")
    all_passed = True
    for fold, info in FROZEN_CHECKPOINT_LEDGER.items():
        fp = os.path.join(base_dir, info["path"])
        if not os.path.exists(fp):
            print(f"  [ERROR] Checkpoint missing: {fp}")
            all_passed = False
            continue
        actual_hash = compute_sha256(fp)
        expected_hash = info["sha256"]
        if actual_hash != expected_hash:
            print(f"  [ERROR] Checkpoint hash mismatch for Fold {fold}!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            all_passed = False
        else:
            print(f"  [PASS] Fold {fold} verified bit-for-bit: {info['path']}")
    
    if not all_passed:
        raise RuntimeError("Checkpoint integrity verification FAILED. Aborting.")
    print("[SECURITY] All 5 candidate checkpoints verified bit-for-bit unchanged.")
    return True


def build_candidate_inference_model() -> tf.keras.Model:
    """
    Constructs the frozen candidate model (EXP-06C Config A).
    Backbone: DenseNet-121 (strictly non-trainable, ImageNet weights).
    Fusion: 3-Scan horizontal concatenation (3072-d).
    Classification Head: Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax).
    Projection Head: Dense(128, None) (Linear).
    """
    inputs = tf.keras.Input(shape=(3, 112, 112, 3), name="participant_scans")
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]

    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False

    pool = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    f1, f2, f3 = pool(base(s1)), pool(base(s2)), pool(base(s3))
    fused = tf.keras.layers.Concatenate(name="fused_features")([f1, f2, f3])

    dense_512 = tf.keras.layers.Dense(512, activation="relu", name="dense_512")
    dropout_05 = tf.keras.layers.Dropout(0.5, name="dropout_0.5")
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")

    d = dense_512(fused)
    drop = dropout_05(d)
    out_clf = clf(drop)

    emb = tf.keras.layers.Dense(128, activation=None, name="emb")
    out_emb = emb(fused)

    model = tf.keras.Model(inputs=inputs, outputs=[out_clf, out_emb], name="EXP06C_Candidate")
    model.trainable = False
    return model


class FrozenCandidateEnsemble:
    """
    Manages loading and soft-voting inference across the 5 frozen fold checkpoints.
    """
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        verify_checkpoint_integrity(base_dir)
        self.models = []
        self._load_all_folds()

    def _load_all_folds(self):
        print("\n[MODEL] Instantiating and loading 5-fold ensemble...")
        for fold in range(1, 6):
            model = build_candidate_inference_model()
            ckpt_path = os.path.join(self.base_dir, FROZEN_CHECKPOINT_LEDGER[fold]["path"])
            model.load_weights(ckpt_path)
            model.trainable = False
            self.models.append(model)
            print(f"  Loaded Fold {fold} checkpoint from {ckpt_path}")
        print(f"[MODEL] 5-fold ensemble loaded successfully. Trainable variables: {sum(len(m.trainable_variables) for m in self.models)} (Expected 0).")

    def predict_participant(self, participant_scans_batch: np.ndarray) -> dict:
        """
        Executes inference for a single participant batch: shape (1, 3, 112, 112, 3), raw float32 [0, 255].
        Returns mean ensemble probabilities and predicted canonical label.
        """
        assert participant_scans_batch.shape == (1, 3, 112, 112, 3), f"Invalid input shape: {participant_scans_batch.shape}"
        fold_probs = []
        for model in self.models:
            clf_preds, _ = model(participant_scans_batch, training=False)
            probs = clf_preds.numpy()[0]
            fold_probs.append(probs)

        ensemble_probs = np.mean(fold_probs, axis=0)
        # Normalize to strictly sum to 1.0
        ensemble_probs = ensemble_probs / np.sum(ensemble_probs)
        pred_idx = int(np.argmax(ensemble_probs))
        pred_label = CANONICAL_CLASSES[pred_idx]

        return {
            "predicted_label": pred_label,
            "predicted_index": pred_idx,
            "probabilities": {c: float(p) for c, p in zip(CANONICAL_CLASSES, ensemble_probs)},
            "fold_probabilities": [[float(p) for p in fp] for fp in fold_probs]
        }


def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=2000, seed=42):
    """
    Computes 95% non-parametric bootstrap confidence intervals on participant-level predictions.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_list, acc_list, bacc_list = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = [y_true[i] for i in idx]
        yp_sample = [y_pred[i] for i in idx]

        acc_list.append(accuracy_score(yt_sample, yp_sample))
        bacc_list.append(balanced_accuracy_score(yt_sample, yp_sample))
        _, _, f1, _ = precision_recall_fscore_support(yt_sample, yp_sample, average="macro", zero_division=0)
        f1_list.append(f1)

    return {
        "macro_f1": {
            "mean": float(np.mean(f1_list)),
            "ci_95": [float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))]
        },
        "accuracy": {
            "mean": float(np.mean(acc_list)),
            "ci_95": [float(np.percentile(acc_list, 2.5)), float(np.percentile(acc_list, 97.5))]
        },
        "balanced_accuracy": {
            "mean": float(np.mean(bacc_list)),
            "ci_95": [float(np.percentile(bacc_list, 2.5)), float(np.percentile(bacc_list, 97.5))]
        }
    }


def evaluate_primary_cohort(manifest_path="reports/phase3_participant_benchmark/participant_split.csv",
                            output_dir="reports/final_evaluation",
                            base_dir="."):
    """
    Executes pre-registered evaluation on the locked 21-participant test cohort.
    STRICT RESTRICTION: Evaluates ONLY split == 'TEST' from participant_split.csv.
    Prohibits any access to the hospital cohort (data/captured).
    """
    if not EVALUATION_AUTHORIZED:
        raise PermissionError(
            "[SECURITY ALERT] EVALUATION_AUTHORIZED is False. Real evaluation cannot be executed without explicit authorization."
        )

    print("\n" + "="*80)
    print("EXECUTING PRE-REGISTERED EVALUATION: PRIMARY LOCKED 21-PARTICIPANT COHORT")
    print("="*80)

    # 1. Load manifest and filter strictly to TEST split
    df = pd.read_csv(os.path.join(base_dir, manifest_path))
    test_df = df[df["split"] == "TEST"].copy()
    
    participants = sorted(test_df["participant_id"].unique())
    assert len(participants) == 21, f"Expected exactly 21 participants, got {len(participants)}"
    assert len(test_df) == 63, f"Expected exactly 63 scans, got {len(test_df)}"
    print(f"[COHORT] Verified 21 participants (63 scans) in TEST split.")

    # 2. Check for anti-overwrite
    pred_csv_path = os.path.join(output_dir, "primary_evaluation_predictions.csv")
    if os.path.exists(pred_csv_path):
        raise FileExistsError(f"Prediction file already exists: {pred_csv_path}. Overwrite is prohibited.")

    # 3. Instantiate ensemble (with checkpoint hash verification)
    ensemble = FrozenCandidateEnsemble(base_dir=base_dir)

    # 4. Run single-pass inference on each participant
    results = []
    y_true, y_pred = [], []

    for pid in participants:
        p_scans = test_df[test_df["participant_id"] == pid].sort_values("scan_path")
        assert len(p_scans) == 3, f"Participant {pid} does not have exactly 3 scans!"
        
        true_label = p_scans["blood_group"].iloc[0]
        assert true_label in CANONICAL_CLASSES, f"Unknown class {true_label}"

        # Preprocess scans
        scan_arrays = []
        for scan_file in p_scans["scan_path"]:
            # Ensure path does NOT touch hospital cohort
            assert "captured" not in scan_file, "Security violation: hospital cohort path detected!"
            arr, _, _ = transform_c5(scan_file)
            scan_arrays.append(arr)

        batch_tensor = np.expand_dims(np.stack(scan_arrays, axis=0), axis=0).astype(np.float32)
        pred_res = ensemble.predict_participant(batch_tensor)

        y_true.append(true_label)
        y_pred.append(pred_res["predicted_label"])

        row = {
            "participant_id": pid,
            "true_label": true_label,
            "predicted_label": pred_res["predicted_label"],
            "prob_A+": pred_res["probabilities"]["A+"],
            "prob_AB+": pred_res["probabilities"]["AB+"],
            "prob_B+": pred_res["probabilities"]["B+"],
            "prob_O+": pred_res["probabilities"]["O+"]
        }
        results.append(row)

    # 5. Save predictions CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(pred_csv_path, index=False)
    print(f"[OUTPUT] Participant-level predictions saved to {pred_csv_path}")

    # 6. Compute primary & secondary metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    # 7. Compute bootstrap CI
    ci_res = compute_bootstrap_ci(y_true, y_pred, n_bootstraps=2000, seed=42)

    # 8. Save results JSON
    metrics_data = {
        "candidate": "EXP-06C Config A",
        "cohort": "PRIMARY_LOCKED_21",
        "num_participants": 21,
        "num_correct": int(np.sum(np.array(y_true) == np.array(y_pred))),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "bootstrap_ci_95": ci_res,
        "per_class": {
            c: {"support": int(s), "precision": float(p), "recall": float(r), "f1": float(f)}
            for c, s, p, r, f in zip(CANONICAL_CLASSES, s_cls, p_cls, r_cls, f1_cls)
        },
        "confusion_matrix": cm.tolist(),
        "canonical_classes": CANONICAL_CLASSES,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    metrics_json_path = os.path.join(output_dir, "primary_evaluation_results.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OUTPUT] Metrics results saved to {metrics_json_path}")
    print("[EVALUATION COMPLETE] Primary locked test cohort evaluation finished.")


def run_synthetic_validation(base_dir: str = "."):
    """
    Performs end-to-end synthetic functional testing.
    Verifies model loading, checkpoint verification, inference execution,
    probability sum properties, canonical ordering, and determinism.
    Does NOT touch any real or protected cohort images.
    """
    print("\n" + "="*80)
    print("RUNNING SAFE SYNTHETIC VALIDATION OF EVALUATION INTERFACE")
    print("="*80)

    ensemble = FrozenCandidateEnsemble(base_dir=base_dir)

    np.random.seed(42)
    dummy_scans = np.random.uniform(0.0, 255.0, size=(1, 3, 112, 112, 3)).astype(np.float32)

    print("\n[SYNTHETIC TEST] Feeding dummy tensor: shape (1, 3, 112, 112, 3)...")
    res1 = ensemble.predict_participant(dummy_scans)
    res2 = ensemble.predict_participant(dummy_scans)

    # 1. Determinism check
    assert res1["predicted_label"] == res2["predicted_label"], "Determinism failure: predictions differ!"
    assert np.allclose(list(res1["probabilities"].values()), list(res2["probabilities"].values())), "Determinism failure: probabilities differ!"
    print("  [PASS] Inference is 100% deterministic.")

    # 2. Probability sum check
    prob_sum = sum(res1["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-5, f"Probability sum invalid: {prob_sum}"
    print(f"  [PASS] Probability vector sums to 1.0: {prob_sum:.6f}")

    # 3. Canonical class check
    assert list(res1["probabilities"].keys()) == CANONICAL_CLASSES, "Canonical class order mismatch!"
    print(f"  [PASS] Canonical classes strictly preserved: {CANONICAL_CLASSES}")

    # 4. Argmax check
    max_class = max(res1["probabilities"], key=res1["probabilities"].get)
    assert res1["predicted_label"] == max_class, "Argmax mismatch!"
    print(f"  [PASS] Predicted label ({res1['predicted_label']}) matches argmax probability ({res1['probabilities'][max_class]:.4f}).")

    print("\n[SYNTHETIC TEST SUMMARY] Synthetic validation PASSED with 0 errors.")
    print("Ensemble inference pipeline is fully functional and safely locked.")
    print("="*80 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Interface for Frozen EXP-06C Config A")
    parser.add_argument("--synthetic_test", action="store_true", help="Run safe synthetic validation only.")
    parser.add_argument("--authorize_evaluation", action="store_true", help="Authorize execution on real cohort (Requires prior written protocol approval).")
    args = parser.parse_args()

    if args.synthetic_test:
        run_synthetic_validation()
        sys.exit(0)

    if not args.authorize_evaluation or not EVALUATION_AUTHORIZED:
        print("[SAFETY LOCK] Real-cohort evaluation is NOT AUTHORIZED.")
        print("To run synthetic validation: python evaluate_frozen_candidate.py --synthetic_test")
        print("Execution halted. Zero protected files accessed.")
        sys.exit(1)

    # If both flags are explicitly enabled (authorized run)
    evaluate_primary_cohort()
