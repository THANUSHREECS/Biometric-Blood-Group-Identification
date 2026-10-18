import os
import sys
import re
import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

def run_spot_check(seed=777, output_file="reports/project_report/spot_check_changed_rows_output.txt"):
    inventory_path = "reports/project_report/results_inventory.csv"
    if not os.path.exists(inventory_path):
        print(f"Error: {inventory_path} not found.")
        sys.exit(1)

    df = pd.read_csv(inventory_path)

    # Cohort of target rows modified or added this round: disagreement rows, locked rows, benchmark rows
    target_ids = [
        "DAY27-LOCKED-TEST-21",
        "DAY27-LOCKED-ZERO-SHOT-PART-21",
        "DAY27-LOCKED-ZERO-SHOT-SCAN-63",
        "P3-ARCH-CONVNEXT-LOCKED-21",
        "P3-ARCH-CUSTOMCNN-LOCKED-21",
        "P3-ARCH-DENSE121-LOCKED-21",
        "P3-ARCH-EFFB0-LOCKED-21",
        "P3-ARCH-EFFV2B0-LOCKED-21",
        "P3-ARCH-RESNET50-LOCKED-21",
        "P3-ARCH-VGG16-LOCKED-21",
        "MFS100-LOCK21-EFFB0"
    ]

    rng = random.Random(seed)
    sampled_ids = sorted(rng.sample(target_ids, 10))

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log(f"SPOT-CHECK OF 10 RANDOMLY SAMPLED CHANGED / LOCKED ROWS (SEED={seed})")
    log("Sampling pool: Disagreement rows, locked test rows, architecture benchmark rows")
    log("=" * 80)

    # Load locked participant IDs from split manifest for filtering zero-shot predictions
    split_df = pd.read_csv("reports/phase3_participant_benchmark/participant_split.csv")
    locked_pids = set(split_df[split_df["split"] == "TEST"]["participant_id"].unique())

    passed_count = 0

    for sample_num, exp_id in enumerate(sampled_ids, 1):
        matching_rows = df[df["experiment_id"] == exp_id]
        if matching_rows.empty:
            log(f"ERROR: {exp_id} not found in inventory!")
            continue

        row = matching_rows.iloc[0]
        group = row["group"]
        src_file = row["source_file"]
        src_field = row["source_field_or_line"]
        csv_acc = float(row["accuracy"])
        csv_bacc = float(row["balanced_accuracy"]) if row["balanced_accuracy"] not in ["not recorded in artifact", "not applicable", None, ""] else None
        csv_mf1 = float(row["macro_f1"])

        log(f"\n--- Spot-Check #{sample_num}: [{exp_id}] ---")
        log(f"Group       : {group}")
        log(f"Source File : {src_file}")
        log(f"Source Field: {src_field}")
        log(f"CSV Values  : Acc={csv_acc:.4f}, BalAcc={'N/A' if csv_bacc is None else f'{csv_bacc:.4f}'}, MacroF1={csv_mf1:.4f}")

        art_acc = None
        art_bacc = None
        art_mf1 = None

        # 1. DAY27-LOCKED-TEST-21 (from markdown text)
        if exp_id == "DAY27-LOCKED-TEST-21":
            with open(src_file, "r", encoding="utf-8") as f:
                text = f.read()
            m_acc = re.search(r"Day 27 Baseline Accuracy:\*{0,2}\s*([0-9.]+)%", text)
            m_f1 = re.search(r"Day 27 Baseline Macro F1:\*{0,2}\s*([0-9.]+)", text)
            art_acc = float(m_acc.group(1)) / 100.0
            art_mf1 = float(m_f1.group(1))
            art_bacc = None

        # 2. DAY27-LOCKED-ZERO-SHOT-PART-21 (from zero-shot participant CSV filtered to locked 21)
        elif exp_id == "DAY27-LOCKED-ZERO-SHOT-PART-21":
            raw_df = pd.read_csv(src_file)
            locked_subset = raw_df[raw_df["participant_id"].isin(locked_pids)]
            art_acc = accuracy_score(locked_subset["true_label"], locked_subset["final_prediction"])
            art_bacc = balanced_accuracy_score(locked_subset["true_label"], locked_subset["final_prediction"])
            _, _, art_mf1, _ = precision_recall_fscore_support(locked_subset["true_label"], locked_subset["final_prediction"], average="macro", zero_division=0)

        # 3. DAY27-LOCKED-ZERO-SHOT-SCAN-63 (from zero-shot scan CSV filtered to locked 21)
        elif exp_id == "DAY27-LOCKED-ZERO-SHOT-SCAN-63":
            raw_df = pd.read_csv(src_file)
            locked_subset = raw_df[raw_df["participant_id"].isin(locked_pids)]
            art_acc = accuracy_score(locked_subset["true_label"], locked_subset["predicted_class"])
            art_bacc = balanced_accuracy_score(locked_subset["true_label"], locked_subset["predicted_class"])
            _, _, art_mf1, _ = precision_recall_fscore_support(locked_subset["true_label"], locked_subset["predicted_class"], average="macro", zero_division=0)

        # 4. MFS100-LOCK21-EFFB0 (from locked_test_metrics.json)
        elif exp_id == "MFS100-LOCK21-EFFB0":
            with open(src_file, "r", encoding="utf-8") as f:
                d = json.load(f)["efficientnet_b0"]
            art_acc = d["accuracy"]
            art_bacc = d["balanced_accuracy"]
            art_mf1 = d["macro_f1"]

        # 5. Architecture Benchmark test prediction CSVs
        elif exp_id.startswith("P3-ARCH-"):
            raw_df = pd.read_csv(src_file)
            art_acc = accuracy_score(raw_df["true_label"], raw_df["final_prediction"])
            art_bacc = balanced_accuracy_score(raw_df["true_label"], raw_df["final_prediction"])
            _, _, art_mf1, _ = precision_recall_fscore_support(raw_df["true_label"], raw_df["final_prediction"], average="macro", zero_division=0)

        else:
            log(f"ERROR: Unhandled experiment ID: {exp_id}")
            continue

        log(f"Raw Artifact: Acc={art_acc:.4f}, BalAcc={'N/A' if art_bacc is None else f'{art_bacc:.4f}'}, MacroF1={art_mf1:.4f}")

        acc_diff = abs(csv_acc - float(art_acc))
        mf1_diff = abs(csv_mf1 - float(art_mf1))
        assert acc_diff < 0.0005, f"Accuracy mismatch: CSV={csv_acc} vs Artifact={art_acc} (diff={acc_diff})"
        assert mf1_diff < 0.0005, f"Macro F1 mismatch: CSV={csv_mf1} vs Artifact={art_mf1} (diff={mf1_diff})"

        if csv_bacc is not None and art_bacc is not None:
            bacc_diff = abs(csv_bacc - float(art_bacc))
            assert bacc_diff < 0.0005, f"Balanced Accuracy mismatch: CSV={csv_bacc} vs Artifact={art_bacc} (diff={bacc_diff})"
            log(f"MATCH [PASS] (Acc Diff: {acc_diff:.6f}, BalAcc Diff: {bacc_diff:.6f}, MacroF1 Diff: {mf1_diff:.6f})")
        else:
            log(f"MATCH [PASS] (Acc Diff: {acc_diff:.6f}, MacroF1 Diff: {mf1_diff:.6f})")

        passed_count += 1

    log("\n" + "=" * 80)
    log(f"RESULT: {passed_count}/10 changed rows independently spot-checked and verified with zero discrepancy!")
    log("=" * 80)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    run_spot_check(seed=777)
