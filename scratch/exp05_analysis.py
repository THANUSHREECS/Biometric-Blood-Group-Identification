import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_REPORT_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp05"
EXP_CKPT_BASE = PROJECT_ROOT / "experiments" / "exp05" / "checkpoints"

# Protected source checkpoints
ORIG_E2_DIR = PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2"
ORIG_CHECKPOINTS = [ORIG_E2_DIR / f"fold_{i}_best.weights.h5" for i in range(1, 6)]
EXP03_CHECKPOINTS = [PROJECT_ROOT / "experiments" / "exp03" / "checkpoints" / "arm1" / f"fold_{i}_best.weights.h5" for i in range(1, 6)]
PROD_MODEL = PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras"

CLASS_NAMES = ["A+", "AB+", "B+", "O+"]

def sha256_file(filepath):
    p = Path(filepath)
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def mcnemar_test(y_true, y_pred1, y_pred2):
    # b: 1 correct, 2 incorrect
    # c: 1 incorrect, 2 correct
    corr1 = (y_true == y_pred1)
    corr2 = (y_true == y_pred2)
    b = np.sum(corr1 & ~corr2)
    c = np.sum(~corr1 & corr2)
    n_discordant = b + c
    if n_discordant == 0:
        return 0.0, 1.0, b, c
    # Exact binomial test or continuity-corrected McNemar chi-square
    # with continuity correction: chi2 = (|b - c| - 1)^2 / (b + c)
    stat = (abs(b - c) - 1)**2 / (b + c) if abs(b - c) > 0 else 0.0
    from scipy.stats import chi2
    p_val = 1.0 - chi2.cdf(stat, df=1)
    return stat, p_val, int(b), int(c)

def main():
    print("=" * 80)
    print("EXP-05 POST-TRAINING ANALYSIS & INTEGRITY AUDIT")
    print("=" * 80)

    # 1. Load OOF predictions
    oof_a = pd.read_csv(EXP_REPORT_DIR / "config_a_oof_predictions.csv")
    oof_b = pd.read_csv(EXP_REPORT_DIR / "config_b_oof_predictions.csv")
    oof_c = pd.read_csv(EXP_REPORT_DIR / "config_c_oof_predictions.csv")

    print(f"OOF A shape: {oof_a.shape}, OOF B shape: {oof_b.shape}, OOF C shape: {oof_c.shape}")

    # Check participant uniqueness and alignment
    for name, df in [("Config A", oof_a), ("Config B", oof_b), ("Config C", oof_c)]:
        assert len(df) == 80, f"{name} length is {len(df)}, expected 80"
        assert df["participant_id"].nunique() == 80, f"{name} participant_id is not unique!"

    # Ensure identical participant order
    oof_a = oof_a.sort_values("participant_id").reset_index(drop=True)
    oof_b = oof_b.sort_values("participant_id").reset_index(drop=True)
    oof_c = oof_c.sort_values("participant_id").reset_index(drop=True)

    np.testing.assert_array_equal(oof_a["participant_id"].values, oof_b["participant_id"].values)
    np.testing.assert_array_equal(oof_a["participant_id"].values, oof_c["participant_id"].values)
    np.testing.assert_array_equal(oof_a["true_label"].values, oof_b["true_label"].values)
    np.testing.assert_array_equal(oof_a["true_label"].values, oof_c["true_label"].values)
    print("[PASS] Participant alignment verified: All 80 development participants align exactly across A, B, and C.")

    y_true = oof_a["true_label"].values
    y_pred_a = oof_a["predicted_label"].values
    y_pred_b = oof_b["predicted_label"].values
    y_pred_c = oof_c["predicted_label"].values

    # Detailed metrics
    results = {}
    for cid, name, y_pred in [("config_a", "Config A (Concat)", y_pred_a),
                              ("config_b", "Config B (Gated)", y_pred_b),
                              ("config_c", "Config C (Attention)", y_pred_c)]:
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_cls, r_cls, f1_cls, supp = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_NAMES, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
        
        results[cid] = {
            "name": name,
            "acc": acc,
            "bal_acc": bal_acc,
            "macro_p": macro_p,
            "macro_r": macro_r,
            "macro_f1": macro_f1,
            "per_class": {
                c: {"precision": p, "recall": r, "f1": f, "support": s, "correct": cm[i, i]}
                for i, (c, p, r, f, s) in enumerate(zip(CLASS_NAMES, p_cls, r_cls, f1_cls, supp))
            },
            "cm": cm.tolist()
        }

    # Print summary
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"{'Metric':<25} {'Config A (Concat)':<20} {'Config B (Gated)':<20} {'Config C (Attention)':<20}")
    print("-" * 85)
    for k, title in [("acc", "Accuracy"), ("bal_acc", "Balanced Accuracy"),
                     ("macro_p", "Macro Precision"), ("macro_r", "Macro Recall"),
                     ("macro_f1", "Macro F1")]:
        v_a = results["config_a"][k]
        v_b = results["config_b"][k]
        v_c = results["config_c"][k]
        print(f"{title:<25} {v_a*100:6.2f}%{'':<13} {v_b*100:6.2f}%{'':<13} {v_c*100:6.2f}%")

    print("\n--- PER-CLASS RECALL & F1 ---")
    for c in CLASS_NAMES:
        ra = results["config_a"]["per_class"][c]["recall"]
        rb = results["config_b"]["per_class"][c]["recall"]
        rc = results["config_c"]["per_class"][c]["recall"]
        fa = results["config_a"]["per_class"][c]["f1"]
        fb = results["config_b"]["per_class"][c]["f1"]
        fc = results["config_c"]["per_class"][c]["f1"]
        supp = results["config_a"]["per_class"][c]["support"]
        ca = results["config_a"]["per_class"][c]["correct"]
        cb = results["config_b"]["per_class"][c]["correct"]
        cc = results["config_c"]["per_class"][c]["correct"]
        print(f"{c:<5} (N={supp}):")
        print(f"   Recall: A={ra*100:5.2f}% ({ca}/{supp}) | B={rb*100:5.2f}% ({cb}/{supp}) | C={rc*100:5.2f}% ({cc}/{supp})")
        print(f"   F1:     A={fa:0.4f}             | B={fb:0.4f}             | C={fc:0.4f}")

    # McNemar tests
    print("\n--- PAIRED MCNEMAR TESTS ---")
    stat_ab, p_ab, b_ab, c_ab = mcnemar_test(y_true, y_pred_a, y_pred_b)
    stat_ac, p_ac, b_ac, c_ac = mcnemar_test(y_true, y_pred_a, y_pred_c)
    stat_bc, p_bc, b_bc, c_bc = mcnemar_test(y_true, y_pred_b, y_pred_c)

    print(f"Config A vs Config B: b(A+ B-)={b_ab}, c(A- B+)={c_ab}, Chi2={stat_ab:.4f}, p-value={p_ab:.4f}")
    print(f"Config A vs Config C: b(A+ C-)={b_ac}, c(A- C+)={c_ac}, Chi2={stat_ac:.4f}, p-value={p_ac:.4f}")
    print(f"Config B vs Config C: b(B+ C-)={b_bc}, c(B- C+)={c_bc}, Chi2={stat_bc:.4f}, p-value={p_bc:.4f}")

    # Fold level metrics from experiment_config.json
    with open(EXP_REPORT_DIR / "experiment_config.json", "r") as f:
        exp_json = json.load(f)
    exp_dict = {item["config_id"]: item for item in exp_json} if isinstance(exp_json, list) else exp_json

    print("\n--- FOLD-LEVEL VALIDATION MACRO F1 ---")
    fold_f1s = {}
    for cid in ["config_a", "config_b", "config_c"]:
        folds = exp_dict[cid]["folds"]
        f1s = [f["best_val_f1"] for f in folds]
        epochs = [f["best_epoch"] for f in folds]
        fold_f1s[cid] = f1s
        print(f"{cid}: Folds={[round(x, 4) for x in f1s]}, Mean={np.mean(f1s):.4f}, Std={np.std(f1s):.4f}, Best Epochs={epochs}")

    # Checkpoint integrity check
    print("\n--- CHECKPOINT INTEGRITY AUDIT ---")
    protected_hashes = {
        "Production Model": (PROD_MODEL, sha256_file(PROD_MODEL)),
    }
    for i, p in enumerate(ORIG_CHECKPOINTS, 1):
        protected_hashes[f"Original E2 Fold {i}"] = (p, sha256_file(p))
    for i, p in enumerate(EXP03_CHECKPOINTS, 1):
        protected_hashes[f"EXP-03 Arm 1 Fold {i}"] = (p, sha256_file(p))

    # Pre-training ledger hashes for verification
    ledger_ref = {
        "Production Model": "7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9",
        "Original E2 Fold 1": "7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d",
        "Original E2 Fold 2": "9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede",
        "Original E2 Fold 3": "8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3",
        "Original E2 Fold 4": "cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519",
        "Original E2 Fold 5": "aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d",
        "EXP-03 Arm 1 Fold 1": "0d047b7a10153718b067e84a27a6ad2bc0486ccf1d35057a07da6e2e2d627ac5",
        "EXP-03 Arm 1 Fold 2": "dd94e4c44887399d700ac2de195f513a5b1db73ac561f4ff8c2345d5156f7063",
        "EXP-03 Arm 1 Fold 3": "f763f67068a4411dbd4a405e39248f2ead1b1b1fe228c4b1bca5da807005b3e3",
        "EXP-03 Arm 1 Fold 4": "589f1e52275f68ed02329b1d4a7a9c16f406ed3868b8afeef9706ebace61b496",
        "EXP-03 Arm 1 Fold 5": "848aea5498e469871d5fdcd136d849f971c9c3c485a4c66afb2505d3b0a46776"
    }

    integrity_passed = True
    for name, (path, cur_hash) in protected_hashes.items():
        ref_hash = ledger_ref.get(name, "UNKNOWN")
        match = (cur_hash == ref_hash)
        if not match:
            integrity_passed = False
        print(f"  {name:<22}: {'MATCH' if match else 'FAIL'} | {cur_hash}")

    print(f"\nProtected Checkpoint Integrity: {'[PASS] All 11 protected checkpoints unchanged' if integrity_passed else '[FAIL] Hash mismatch detected!'}")

    # New EXP-05 checkpoint hashes
    print("\n--- NEW EXP-05 CHECKPOINTS HASHES ---")
    new_hashes = {}
    for cid in ["config_a", "config_b", "config_c"]:
        ckpt_dir = EXP_CKPT_BASE / cid
        for f in range(1, 6):
            p = ckpt_dir / f"fold_{f}_best.weights.h5"
            h = sha256_file(p)
            new_hashes[f"{cid}_fold_{f}"] = (p, h)
            print(f"  {cid}_fold_{f}: {h} ({p.stat().st_size} bytes)")

    # Save complete analysis JSON
    analysis_payload = {
        "results": results,
        "mcnemar": {
            "ab": {"stat": stat_ab, "p_value": p_ab, "b": b_ab, "c": c_ab},
            "ac": {"stat": stat_ac, "p_value": p_ac, "b": b_ac, "c": c_ac},
            "bc": {"stat": stat_bc, "p_value": p_bc, "b": b_bc, "c": c_bc}
        },
        "fold_f1s": fold_f1s,
        "protected_integrity_passed": integrity_passed,
        "protected_hashes": {k: v[1] for k, v in protected_hashes.items()},
        "new_ckpt_hashes": {k: {"path": str(v[0]), "hash": v[1]} for k, v in new_hashes.items()}
    }

    def convert_types(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list)):
            return [convert_types(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    with open(EXP_REPORT_DIR / "post_analysis.json", "w") as f:
        json.dump(convert_types(analysis_payload), f, indent=2)
    print("\nSaved post_analysis.json successfully.")

if __name__ == "__main__":
    main()
