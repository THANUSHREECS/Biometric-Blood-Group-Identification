import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from scipy.stats import binomtest, fisher_exact

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP04_DIR = PROJECT_ROOT / "reports" / "accuracy_improvement" / "exp04"
CKPT_DIR = PROJECT_ROOT / "experiments" / "exp04" / "checkpoints"
SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"

CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]

def sha256_of_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("EXP-04 PHASE 3: INDEPENDENT VALIDATION & STATISTICAL AUDIT")
    print("=" * 80)

    # 1. Dataset quarantine & split check
    df_split = pd.read_csv(SPLIT_CSV)
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]
    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(df_split[df_split["split"] == "TEST"]["participant_id"].unique())
    assert len(dev_pids) == 80, f"Expected 80 dev participants, got {len(dev_pids)}"
    assert len(test_pids) == 21, f"Expected 21 test participants, got {len(test_pids)}"
    assert len(dev_pids & test_pids) == 0, "Quarantine violation!"

    # 2. OOF file validation
    configs = ["config_a", "config_b", "config_c"]
    oof_dfs = {}
    for cid in configs:
        oof_path = EXP04_DIR / f"{cid}_oof_predictions.csv"
        assert oof_path.is_file(), f"Missing OOF file: {oof_path}"
        df = pd.read_csv(oof_path)
        assert len(df) == 80, f"[{cid}] Expected 80 OOF rows, got {len(df)}"
        assert len(df["participant_id"].unique()) == 80, f"[{cid}] Duplicate participants!"
        pids_in_oof = set(df["participant_id"])
        assert pids_in_oof == dev_pids, f"[{cid}] Mismatch in participant IDs!"
        assert len(pids_in_oof & test_pids) == 0, f"[{cid}] Test participant in OOF!"

        # Check probabilities
        for _, row in df.iterrows():
            probs = json.loads(row["canonical_probabilities"])
            assert len(probs) == 4, f"[{cid}] Expected 4 probabilities, got {len(probs)}"
            assert all(np.isfinite(p) and p >= 0.0 for p in probs), f"[{cid}] Invalid prob!"
            assert abs(sum(probs) - 1.0) < 1e-4, f"[{cid}] Probs do not sum to 1: {sum(probs)}"
            pred_idx = np.argmax(probs)
            assert CANONICAL_CLASSES[pred_idx] == row["predicted_label"], f"[{cid}] Prediction mismatch!"

        oof_dfs[cid] = df.sort_values("participant_id").reset_index(drop=True)
        print(f"[{cid}] PASS: 80 unique development participants, valid probabilities, correct alignment.")

    # 3. Metrics Verification
    results = {}
    for cid in configs:
        df = oof_dfs[cid]
        y_true = df["true_label"].tolist()
        y_pred = df["predicted_label"].tolist()

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
        p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(y_true, y_pred, labels=CANONICAL_CLASSES, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

        # Fold-level breakdown
        fold_metrics = []
        for fid in range(1, 6):
            f_df = df[df["fold_id"] == fid]
            assert len(f_df) == 16, f"Fold {fid} count: {len(f_df)}"
            f_acc = accuracy_score(f_df["true_label"], f_df["predicted_label"])
            f_bacc = balanced_accuracy_score(f_df["true_label"], f_df["predicted_label"])
            _, _, f_f1, _ = precision_recall_fscore_support(f_df["true_label"], f_df["predicted_label"], labels=CANONICAL_CLASSES, average="macro", zero_division=0)
            fold_metrics.append({"fold": fid, "acc": f_acc, "bacc": f_bacc, "macro_f1": f_f1})

        f_accs = [m["acc"] for m in fold_metrics]
        f_baccs = [m["bacc"] for m in fold_metrics]
        f_f1s = [m["macro_f1"] for m in fold_metrics]

        results[cid] = {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": f1_m,
            "macro_precision": p_m,
            "macro_recall": r_m,
            "per_class": {
                c: {"precision": p_c[i], "recall": r_c[i], "f1": f1_c[i], "support": int(supp_c[i])}
                for i, c in enumerate(CANONICAL_CLASSES)
            },
            "confusion_matrix": cm,
            "fold_metrics": fold_metrics,
            "fold_std": {
                "acc_std": float(np.std(f_accs, ddof=1)),
                "bacc_std": float(np.std(f_baccs, ddof=1)),
                "f1_std": float(np.std(f_f1s, ddof=1))
            }
        }

    # 4. Statistical significance tests
    y_true = oof_dfs["config_a"]["true_label"].values
    correct_a = (oof_dfs["config_a"]["predicted_label"].values == y_true)
    correct_b = (oof_dfs["config_b"]["predicted_label"].values == y_true)
    correct_c = (oof_dfs["config_c"]["predicted_label"].values == y_true)

    # McNemar's exact test for overall accuracy
    def mcnemar_test(corr1, corr2):
        b = int(np.sum(corr1 & (~corr2))) # correct in 1, wrong in 2
        c = int(np.sum((~corr1) & corr2)) # wrong in 1, correct in 2
        res = binomtest(min(b, c), b + c, 0.5) if (b + c) > 0 else None
        p_val = res.pvalue if res else 1.0
        return b, c, p_val

    b_ba, c_ba, p_ba = mcnemar_test(correct_a, correct_b)
    b_ca, c_ca, p_ca = mcnemar_test(correct_a, correct_c)

    # Fisher exact test for AB+ recall
    ab_mask = (y_true == "AB+")
    ab_corr_a = int(np.sum(correct_a[ab_mask]))
    ab_corr_b = int(np.sum(correct_b[ab_mask]))
    ab_corr_c = int(np.sum(correct_c[ab_mask]))
    ab_total = int(np.sum(ab_mask))

    _, p_ab_ba = fisher_exact([[ab_corr_b, ab_total - ab_corr_b], [ab_corr_a, ab_total - ab_corr_a]])
    _, p_ab_ca = fisher_exact([[ab_corr_c, ab_total - ab_corr_c], [ab_corr_a, ab_total - ab_corr_a]])

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"{'Metric':<20} {'Config A (Frozen)':<22} {'Config B (DB4 Conv)':<22} {'Config C (DB3+4)':<22}")
    print("-" * 86)
    print(f"{'Accuracy':<20} {results['config_a']['accuracy']*100:.2f}% (std {results['config_a']['fold_std']['acc_std']*100:.2f}%)   {results['config_b']['accuracy']*100:.2f}% (std {results['config_b']['fold_std']['acc_std']*100:.2f}%)   {results['config_c']['accuracy']*100:.2f}% (std {results['config_c']['fold_std']['acc_std']*100:.2f}%)")
    print(f"{'Balanced Acc':<20} {results['config_a']['balanced_accuracy']*100:.2f}% (std {results['config_a']['fold_std']['bacc_std']*100:.2f}%)   {results['config_b']['balanced_accuracy']*100:.2f}% (std {results['config_b']['fold_std']['bacc_std']*100:.2f}%)   {results['config_c']['balanced_accuracy']*100:.2f}% (std {results['config_c']['fold_std']['bacc_std']*100:.2f}%)")
    print(f"{'Macro F1':<20} {results['config_a']['macro_f1']:.4f} (std {results['config_a']['fold_std']['f1_std']:.4f})     {results['config_b']['macro_f1']:.4f} (std {results['config_b']['fold_std']['f1_std']:.4f})     {results['config_c']['macro_f1']:.4f} (std {results['config_c']['fold_std']['f1_std']:.4f})")
    print("-" * 86)
    for c in CANONICAL_CLASSES:
        rec_a = results['config_a']['per_class'][c]['recall'] * 100
        rec_b = results['config_b']['per_class'][c]['recall'] * 100
        rec_c = results['config_c']['per_class'][c]['recall'] * 100
        supp = results['config_a']['per_class'][c]['support']
        print(f"{c + ' Recall (N=' + str(supp) + ')':<20} {rec_a:.2f}% ({int(round(rec_a*supp/100))}/{supp})         {rec_b:.2f}% ({int(round(rec_b*supp/100))}/{supp})         {rec_c:.2f}% ({int(round(rec_c*supp/100))}/{supp})")

    print("\n--- STATISTICAL SIGNIFICANCE (vs Config A Reference) ---")
    print(f"Config B vs Config A (Overall Accuracy McNemar): b={b_ba}, c={c_ba}, p={p_ba:.4f} ({'Not significant' if p_ba > 0.05 else 'Significant'})")
    print(f"Config C vs Config A (Overall Accuracy McNemar): b={b_ca}, c={c_ca}, p={p_ca:.4f} ({'Not significant' if p_ca > 0.05 else 'Significant'})")
    print(f"Config B vs Config A (AB+ Recall Fisher Exact):   {ab_corr_b}/7 vs {ab_corr_a}/7, p={p_ab_ba:.4f}")
    print(f"Config C vs Config A (AB+ Recall Fisher Exact):   {ab_corr_c}/7 vs {ab_corr_a}/7, p={p_ab_ca:.4f}")

    # 5. Checkpoint Hashes
    print("\n--- CHECKPOINT INTEGRITY AUDIT ---")
    ckpt_hashes = {}
    for cid in configs:
        ckpt_hashes[cid] = {}
        for fid in range(1, 6):
            p = CKPT_DIR / cid / f"fold_{fid}_best.weights.h5"
            assert p.is_file(), f"Missing checkpoint: {p}"
            sz = p.stat().st_size
            h = sha256_of_file(p)
            ckpt_hashes[cid][f"fold_{fid}"] = {"size": sz, "sha256": h}
            print(f"  [{cid}] fold_{fid}: size={sz:,} bytes | sha256={h[:16]}...")

    # Save validation results
    val_out = {
        "metrics": {cid: {
            "accuracy": results[cid]["accuracy"],
            "balanced_accuracy": results[cid]["balanced_accuracy"],
            "macro_f1": results[cid]["macro_f1"],
            "fold_std": results[cid]["fold_std"],
            "per_class": results[cid]["per_class"],
            "confusion_matrix": results[cid]["confusion_matrix"].tolist(),
            "fold_metrics": results[cid]["fold_metrics"]
        } for cid in configs},
        "statistical_tests": {
            "mcnemar_b_vs_a": {"b": b_ba, "c": c_ba, "p_value": p_ba},
            "mcnemar_c_vs_a": {"b": b_ca, "c": c_ca, "p_value": p_ca},
            "fisher_ab_b_vs_a": {"ab_b": ab_corr_b, "ab_a": ab_corr_a, "p_value": p_ab_ba},
            "fisher_ab_c_vs_a": {"ab_c": ab_corr_c, "ab_a": ab_corr_a, "p_value": p_ab_ca}
        },
        "checkpoints": ckpt_hashes
    }

    with open(EXP04_DIR / "independent_validation_summary.json", "w") as f:
        json.dump(val_out, f, indent=2)
    print(f"\nSaved independent validation summary to: {EXP04_DIR / 'independent_validation_summary.json'}")

if __name__ == "__main__":
    main()
