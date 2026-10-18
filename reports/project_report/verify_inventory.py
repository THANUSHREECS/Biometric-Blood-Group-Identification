import os
import sys
import json
import csv
import re
import math
import argparse
import pandas as pd

def parse_num(val):
    if val is None or val in ["not recorded in artifact", "not applicable", ""]:
        return None
    try:
        return float(val)
    except ValueError:
        return None

def verify_inventory(csv_path="reports/project_report/results_inventory.csv"):
    if not os.path.exists(csv_path):
        print(f"ERROR: Inventory file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    checked_rows = 0
    mismatches = []

    # Cache for opened source files
    json_cache = {}
    csv_cache = {}
    text_cache = {}

    def get_json(path):
        if path not in json_cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing source file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                json_cache[path] = json.load(f)
        return json_cache[path]

    def get_csv(path):
        if path not in csv_cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing source file: {path}")
            csv_cache[path] = pd.read_csv(path)
        return csv_cache[path]

    def get_text(path):
        if path not in text_cache:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing source file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                text_cache[path] = f.read()
        return text_cache[path]

    print(f"Starting verification of {csv_path} ({total_rows} rows)...")

    for idx, row in df.iterrows():
        exp_id = row["experiment_id"]
        verified = row.get("verified_from_artifact", "no")
        if verified != "yes":
            continue

        checked_rows += 1
        src_file = row["source_file"]
        src_field = row["source_field_or_line"]
        src_type = row["source_type"]

        # Expected metrics from CSV
        row_metrics = {
            "accuracy": parse_num(row.get("accuracy")),
            "balanced_accuracy": parse_num(row.get("balanced_accuracy")),
            "macro_precision": parse_num(row.get("macro_precision")),
            "macro_recall": parse_num(row.get("macro_recall")),
            "macro_f1": parse_num(row.get("macro_f1")),
        }

        # Check source file existence
        if not os.path.exists(src_file):
            mismatches.append(f"Row {exp_id}: Source file not found: {src_file}")
            continue

        artifact_metrics = {}

        try:
            if src_type == "json":
                data = get_json(src_file)

                if src_file == "outputs/original_6000_models_audit_results.json":
                    sub = data[src_field]
                    artifact_metrics["accuracy"] = sub.get("accuracy")
                    artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                    artifact_metrics["macro_precision"] = sub.get("macro_precision")
                    artifact_metrics["macro_recall"] = sub.get("macro_recall")
                    artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "outputs/ensemble_weights.json":
                    artifact_metrics["accuracy"] = data["locked_test_accuracy"]
                    pcr = data.get("per_class_report", {})
                    recalls = [pcr[c]["recall"] for c in pcr]
                    precs = [pcr[c]["precision"] for c in pcr]
                    f1s = [pcr[c]["f1-score"] for c in pcr]
                    artifact_metrics["balanced_accuracy"] = sum(recalls) / len(recalls)
                    artifact_metrics["macro_precision"] = sum(precs) / len(precs)
                    artifact_metrics["macro_recall"] = sum(recalls) / len(recalls)
                    artifact_metrics["macro_f1"] = sum(f1s) / len(f1s)

                elif src_file == "outputs/ensemble_optimization_v2_results.json":
                    artifact_metrics["accuracy"] = data["optimized_v2_val_accuracy"]
                    pcm = data.get("per_class_metrics", {})
                    v_recalls = [pcm[c]["recall"] for c in pcm]
                    v_precs = [pcm[c]["precision"] for c in pcm]
                    artifact_metrics["balanced_accuracy"] = sum(v_recalls) / len(v_recalls)
                    artifact_metrics["macro_precision"] = sum(v_precs) / len(v_precs)
                    artifact_metrics["macro_recall"] = sum(v_recalls) / len(v_recalls)
                    artifact_metrics["macro_f1"] = data["optimized_v2_val_macro_f1"]

                elif src_file == "outputs/ensemble_val_results.json":
                    artifact_metrics["accuracy"] = data["val_accuracy"]
                    artifact_metrics["macro_f1"] = data["val_macro_f1"]

                elif src_file == "reports/model_fusion/all_models_eval_data.json":
                    m = re.search(r"models\['(.*?)'\]\['(.*?)'\]", src_field)
                    if m:
                        model_key, level = m.group(1), m.group(2)
                        sub = data["models"][model_key][level]
                        artifact_metrics["accuracy"] = sub.get("acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("bal_acc")
                        if "prec" in sub and isinstance(sub["prec"], list):
                            artifact_metrics["macro_precision"] = sum(sub["prec"]) / len(sub["prec"])
                        if "rec" in sub and isinstance(sub["rec"], list):
                            artifact_metrics["macro_recall"] = sum(sub["rec"]) / len(sub["rec"])
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file in [
                    "reports/efficientnet_e2_ensemble_experiment/development_metrics.json",
                    "reports/efficientnet_e2_ensemble_experiment/separate_26_cohort_metrics.json",
                    "reports/efficientnet_e2_ensemble_experiment/locked_test_metrics.json"
                ]:
                    sub = data[src_field]
                    artifact_metrics["accuracy"] = sub.get("accuracy")
                    artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                    artifact_metrics["macro_precision"] = sub.get("macro_precision")
                    artifact_metrics["macro_recall"] = sub.get("macro_recall")
                    artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "reports/final/eval_101_raw_results.json":
                    sub = data[src_field]
                    artifact_metrics["accuracy"] = sub.get("accuracy")
                    artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                    artifact_metrics["macro_precision"] = sub.get("macro_precision")
                    artifact_metrics["macro_recall"] = sub.get("macro_recall")
                    artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file in [
                    "reports/accuracy_improvement/exp01/metrics.json",
                    "reports/accuracy_improvement/exp02/metrics.json"
                ]:
                    artifact_metrics["accuracy"] = data.get("accuracy")
                    artifact_metrics["balanced_accuracy"] = data.get("balanced_accuracy")
                    artifact_metrics["macro_precision"] = data.get("macro_precision")
                    artifact_metrics["macro_recall"] = data.get("macro_recall")
                    artifact_metrics["macro_f1"] = data.get("macro_f1")

                elif src_file == "reports/accuracy_improvement/exp04/independent_validation_summary.json":
                    m = re.search(r"metrics\['(.*?)'\]", src_field)
                    if m:
                        cfg = m.group(1)
                        sub = data["metrics"][cfg]
                        artifact_metrics["accuracy"] = sub.get("accuracy")
                        artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                        pcm = sub.get("per_class", {})
                        if pcm:
                            precs = [pcm[c]["precision"] for c in pcm]
                            recs = [pcm[c]["recall"] for c in pcm]
                            artifact_metrics["macro_precision"] = sum(precs) / len(precs)
                            artifact_metrics["macro_recall"] = sum(recs) / len(recs)
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "reports/accuracy_improvement/exp05/post_analysis.json":
                    m = re.search(r"results\['(.*?)'\]", src_field)
                    if m:
                        cfg = m.group(1)
                        sub = data["results"][cfg]
                        artifact_metrics["accuracy"] = sub.get("acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("bal_acc")
                        artifact_metrics["macro_precision"] = sub.get("macro_p")
                        artifact_metrics["macro_recall"] = sub.get("macro_r")
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "reports/accuracy_improvement/exp06a/statistical_analysis_results.json":
                    m = re.search(r"cid='(.*?)'", src_field)
                    if m:
                        cid = m.group(1)
                        for entry in data.get("summary", []):
                            if entry.get("cid") == cid:
                                artifact_metrics["accuracy"] = entry.get("acc")
                                artifact_metrics["balanced_accuracy"] = entry.get("bacc")
                                artifact_metrics["macro_precision"] = entry.get("macro_p")
                                artifact_metrics["macro_recall"] = entry.get("macro_r")
                                artifact_metrics["macro_f1"] = entry.get("macro_f1")
                                break

                elif src_file == "reports/accuracy_improvement/exp06b/statistical_analysis_results.json":
                    m = re.search(r"arm_id='(.*?)'", src_field)
                    if m:
                        arm_id = m.group(1)
                        for entry in data.get("summary", []):
                            if entry.get("arm_id") == arm_id:
                                artifact_metrics["accuracy"] = entry.get("accuracy")
                                artifact_metrics["balanced_accuracy"] = entry.get("balanced_accuracy")
                                artifact_metrics["macro_precision"] = entry.get("macro_precision")
                                artifact_metrics["macro_recall"] = entry.get("macro_recall")
                                artifact_metrics["macro_f1"] = entry.get("macro_f1")
                                break

                elif src_file == "reports/accuracy_improvement/exp06c/statistical_analysis_results.json":
                    m = re.search(r"config_id='(.*?)'", src_field)
                    if m:
                        cid = m.group(1)
                        for entry in data.get("configurations", []):
                            if entry.get("config_id") == cid:
                                artifact_metrics["accuracy"] = entry.get("accuracy")
                                artifact_metrics["balanced_accuracy"] = entry.get("balanced_accuracy")
                                artifact_metrics["macro_precision"] = entry.get("macro_precision")
                                artifact_metrics["macro_recall"] = entry.get("macro_recall")
                                artifact_metrics["macro_f1"] = entry.get("macro_f1")
                                break

                elif src_file == "reports/final_evaluation/primary_evaluation_results.json":
                    artifact_metrics["accuracy"] = data.get("accuracy")
                    artifact_metrics["balanced_accuracy"] = data.get("balanced_accuracy")
                    artifact_metrics["macro_precision"] = data.get("macro_precision")
                    artifact_metrics["macro_recall"] = data.get("macro_recall")
                    artifact_metrics["macro_f1"] = data.get("macro_f1")

                elif src_file in [
                    "experiments/leakage_audit/exp1_duplicates_and_perceptual_hashes.json",
                    "experiments/leakage_audit/exp1_perceptual_hashes_trainval_reference.json",
                    "experiments/leakage_audit/exp1_percolation_finer_scan.json",
                    "experiments/leakage_audit/exp1_clustering_summary.json"
                ]:
                    # audit diagnostic rows (no classification metrics)
                    artifact_metrics = {}

                elif src_file == "experiments/leakage_audit/exp2_signal_tests_results.json":
                    if "development_80_participants.observed" in src_field:
                        sub = data["development_80_participants"]["observed"]
                        artifact_metrics["accuracy"] = sub.get("accuracy")
                        artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")
                    elif "primary_test_21_participants.observed" in src_field:
                        sub = data["primary_test_21_participants"]["observed"]
                        artifact_metrics["accuracy"] = sub.get("accuracy")
                        artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "experiments/leakage_audit/phase1_readonly_results.json":
                    m = re.search(r"metadata_baseline_1b\[(\d+)\]", src_field)
                    if m:
                        idx_m = int(m.group(1))
                        sub = data["metadata_baseline_1b"][idx_m]
                        artifact_metrics["accuracy"] = sub.get("accuracy")
                        artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                        artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "experiments/leakage_audit/phase2_selection_rules_comparison.json":
                    rule_key = src_field.split()[0].split(".")[0]
                    if "3-seed mean" in src_field:
                        # Recompute sample mean across seeds 42, 43, 44 as in build_inventory.py
                        accs = [data[rule_key][s]["overall"]["accuracy"] for s in ["seed_42", "seed_43", "seed_44"]]
                        baccs = [data[rule_key][s]["overall"]["balanced_accuracy"] for s in ["seed_42", "seed_43", "seed_44"]]
                        mf1s = [data[rule_key][s]["overall"]["macro_f1"] for s in ["seed_42", "seed_43", "seed_44"]]
                        artifact_metrics["accuracy"] = sum(accs) / len(accs)
                        artifact_metrics["balanced_accuracy"] = sum(baccs) / len(baccs)
                        artifact_metrics["macro_f1"] = sum(mf1s) / len(mf1s)
                    else:
                        m_seed = re.search(r"(seed_\d+)", src_field)
                        if m_seed:
                            s = m_seed.group(1)
                            sub = data[rule_key][s]["overall"]
                            artifact_metrics["accuracy"] = sub.get("accuracy")
                            artifact_metrics["balanced_accuracy"] = sub.get("balanced_accuracy")
                            artifact_metrics["macro_f1"] = sub.get("macro_f1")

                elif src_file == "experiments/leakage_audit/selection_inclusive_null_summary.json":
                    artifact_metrics["accuracy"] = data["accuracy"]["mean"]
                    artifact_metrics["balanced_accuracy"] = data["balanced_accuracy"]["mean"]
                    artifact_metrics["macro_f1"] = data["macro_f1"]["mean"]

                elif src_file == "experiments/leakage_audit/phase3_purged_cv_summary.json":
                    cv_csv = get_csv("experiments/leakage_audit/phase3_purged_cv_results.csv")
                    if "primary_arm_fixed_epochs" in src_field:
                        arm_key = "primary_arm_fixed_epochs"
                        arm_val = "Arm1_FixedEpochs"
                    else:
                        arm_key = "secondary_arm_inner_val"
                        arm_val = "Arm2_InnerValidation"

                    m_pct = re.search(r"\['(.*?)'\]", src_field)
                    tag = m_pct.group(1) if m_pct else ""

                    if "0%_baseline" in tag:
                        sub_df = cv_csv[(cv_csv["arm"] == arm_val) & (cv_csv["condition"] == "0%_unpurged_baseline")]
                        artifact_metrics["accuracy"] = sub_df["val_accuracy"].mean()
                        artifact_metrics["balanced_accuracy"] = data[arm_key]["0%_baseline"]["mean"]
                        artifact_metrics["macro_f1"] = sub_df["val_macro_f1"].mean()
                    else:
                        frac_map = {"10%_purge": 0.1, "25%_purge": 0.25, "50%_purge": 0.5}
                        frac = frac_map.get(tag, 0.1)
                        sub_df = cv_csv[(cv_csv["arm"] == arm_val) & (cv_csv["condition"] == "similarity_purged") & (cv_csv["purge_fraction"] == frac)]
                        artifact_metrics["accuracy"] = sub_df["val_accuracy"].mean()
                        artifact_metrics["balanced_accuracy"] = data[arm_key][tag]["purged_mean"]
                        artifact_metrics["macro_f1"] = sub_df["val_macro_f1"].mean()

                elif src_file == "experiments/leakage_audit/exp3_trivial_image_properties_results.json":
                    m_mod = re.search(r"models\['(.*?)'\]\['(.*?)'\]", src_field)
                    if m_mod:
                        mod, cond = m_mod.group(1), m_mod.group(2)
                        sub = data["models"][mod][cond]
                        artifact_metrics["accuracy"] = sub.get("mean_acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("mean_bal_acc")
                        artifact_metrics["macro_f1"] = sub.get("mean_macro_f1")

                elif src_file == "experiments/leakage_audit/exp3_trivial_ablation_results.json":
                    m_grp = re.search(r"groups\['(.*?)'\]\['(.*?)'\]\['(.*?)'\]", src_field)
                    if m_grp:
                        grp, mod, cond = m_grp.group(1), m_grp.group(2), m_grp.group(3)
                        sub = data["groups"][grp][mod][cond]["summary"]
                        artifact_metrics["accuracy"] = sub.get("mean_acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("mean_bal")
                        artifact_metrics["macro_f1"] = sub.get("mean_f1")

                elif src_file == "experiments/leakage_audit/exp3_effnet_concat_results.json":
                    if "without_trivial_features" in src_field:
                        sub = data["without_trivial_features"]["summary"]
                        artifact_metrics["accuracy"] = sub.get("mean_acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("mean_bal")
                        artifact_metrics["macro_f1"] = sub.get("mean_f1")
                    elif "with_trivial_features_concatenated" in src_field:
                        sub = data["with_trivial_features_concatenated"]["summary"]
                        artifact_metrics["accuracy"] = sub.get("mean_acc")
                        artifact_metrics["balanced_accuracy"] = sub.get("mean_bal")
                        artifact_metrics["macro_f1"] = sub.get("mean_f1")

            elif src_type == "csv":
                csv_data = get_csv(src_file)

                if src_file == "reports/DAY27_FULL_DATASET_COMPARISON.csv":
                    m_row = re.search(r"(.*?):(.*)", src_field)
                    if m_row:
                        r_name, c_name = m_row.group(1), m_row.group(2)
                        sub = csv_data[csv_data["model"] == r_name]
                        if not sub.empty:
                            artifact_metrics["accuracy"] = sub[c_name].values[0]
                            if "canonical_test_macro_f1" in sub.columns and c_name == "canonical_test_accuracy":
                                artifact_metrics["macro_f1"] = sub["canonical_test_macro_f1"].values[0]
                            elif "mfs100_macro_f1" in sub.columns and c_name == "mfs100_accuracy":
                                artifact_metrics["macro_f1"] = sub["mfs100_macro_f1"].values[0]

                elif src_file == "reports/accuracy_improvement/exp03/metrics_comparison.csv":
                    m_arm = re.search(r"arm_id=(.*)", src_field)
                    if m_arm:
                        arm_id = m_arm.group(1)
                        sub = csv_data[csv_data["arm_id"] == arm_id]
                        if not sub.empty:
                            artifact_metrics["accuracy"] = sub["accuracy"].values[0]
                            artifact_metrics["balanced_accuracy"] = sub["balanced_accuracy"].values[0]
                            artifact_metrics["macro_precision"] = sub["macro_precision"].values[0]
                            artifact_metrics["macro_recall"] = sub["macro_recall"].values[0]
                            artifact_metrics["macro_f1"] = sub["macro_f1"].values[0]

                elif src_file == "reports/accuracy_improvement/exp06d/metrics_comparison.csv":
                    m_cfg = re.search(r"Config ID=(.*)", src_field)
                    if m_cfg:
                        cfg_id = m_cfg.group(1)
                        sub = csv_data[csv_data["Config ID"] == cfg_id]
                        if not sub.empty:
                            artifact_metrics["accuracy"] = float(str(sub["Accuracy"].values[0]).replace("%", "")) / 100.0
                            artifact_metrics["balanced_accuracy"] = float(str(sub["Balanced Accuracy"].values[0]).replace("%", "")) / 100.0
                            artifact_metrics["macro_f1"] = float(sub["Macro F1 (OOF)"].values[0])

                elif "reports/phase3_architecture_benchmark/" in src_file and src_file.endswith("_test_preds.csv"):
                    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
                    acc = accuracy_score(csv_data["true_label"], csv_data["final_prediction"])
                    bacc = balanced_accuracy_score(csv_data["true_label"], csv_data["final_prediction"])
                    p, r, f1, _ = precision_recall_fscore_support(csv_data["true_label"], csv_data["final_prediction"], average="macro", zero_division=0)
                    artifact_metrics["accuracy"] = acc
                    artifact_metrics["balanced_accuracy"] = bacc
                    artifact_metrics["macro_precision"] = p
                    artifact_metrics["macro_recall"] = r
                    artifact_metrics["macro_f1"] = f1

                elif src_file == "reports/final_mfs100/final_zero_shot_participant_predictions.csv" and "TEST split" in src_field:
                    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
                    split_df = get_csv("reports/phase3_participant_benchmark/participant_split.csv")
                    locked_pids = split_df[split_df["split"] == "TEST"]["participant_id"].unique().tolist()
                    sub = csv_data[csv_data["participant_id"].isin(locked_pids)]
                    acc = accuracy_score(sub["true_label"], sub["final_prediction"])
                    bacc = balanced_accuracy_score(sub["true_label"], sub["final_prediction"])
                    p, r, f1, _ = precision_recall_fscore_support(sub["true_label"], sub["final_prediction"], average="macro", zero_division=0)
                    artifact_metrics["accuracy"] = acc
                    artifact_metrics["balanced_accuracy"] = bacc
                    artifact_metrics["macro_precision"] = p
                    artifact_metrics["macro_recall"] = r
                    artifact_metrics["macro_f1"] = f1

                elif src_file == "reports/final_mfs100/final_zero_shot_scan_predictions.csv" and "TEST split" in src_field:
                    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
                    split_df = get_csv("reports/phase3_participant_benchmark/participant_split.csv")
                    locked_pids = split_df[split_df["split"] == "TEST"]["participant_id"].unique().tolist()
                    sub = csv_data[csv_data["participant_id"].isin(locked_pids)]
                    acc = accuracy_score(sub["true_label"], sub["predicted_class"])
                    bacc = balanced_accuracy_score(sub["true_label"], sub["predicted_class"])
                    p, r, f1, _ = precision_recall_fscore_support(sub["true_label"], sub["predicted_class"], average="macro", zero_division=0)
                    artifact_metrics["accuracy"] = acc
                    artifact_metrics["balanced_accuracy"] = bacc
                    artifact_metrics["macro_precision"] = p
                    artifact_metrics["macro_recall"] = r
                    artifact_metrics["macro_f1"] = f1

            elif src_type == "md-only":
                text = get_text(src_file)
                if src_file == "reports/phase3_final_evaluation/FINAL_TEST_REPORT.md":
                    m_acc = re.search(r"Day 27 Baseline Accuracy:\*\*\s*([\d\.]+)%", text)
                    m_f1 = re.search(r"Day 27 Baseline Macro F1:\*\*\s*([\d\.]+)", text)
                    if m_acc:
                        artifact_metrics["accuracy"] = float(m_acc.group(1)) / 100.0
                    if m_f1:
                        artifact_metrics["macro_f1"] = float(m_f1.group(1))

        except Exception as e:
            mismatches.append(f"Row {exp_id}: Error reading {src_file} ({src_field}): {str(e)}")
            continue

        # Compare metrics
        for metric_name, row_val in row_metrics.items():
            if row_val is not None:
                art_val = artifact_metrics.get(metric_name)
                if art_val is None:
                    mismatches.append(f"Row {exp_id} [{metric_name}]: CSV has {row_val:.4f} but not extracted from {src_file} ({src_field})")
                else:
                    try:
                        art_val_num = float(art_val)
                        if abs(row_val - art_val_num) > 0.0005:
                            mismatches.append(f"Row {exp_id} [{metric_name}]: CSV={row_val:.4f} vs Artifact={art_val_num:.4f} (file={src_file}, field={src_field})")
                    except ValueError:
                        mismatches.append(f"Row {exp_id} [{metric_name}]: Cannot parse artifact value '{art_val}' as float")

    print("\n" + "=" * 60)
    print(f"VERIFICATION SUMMARY: {csv_path}")
    print(f"Total Rows: {total_rows} | Checked Rows: {checked_rows} | Mismatches: {len(mismatches)}")
    print("=" * 60)

    if mismatches:
        print("\nMISMATCHES FOUND:")
        for m in mismatches:
            print(f"  - {m}")
        return False
    else:
        print("\nALL CHECKED VALUES MATCH BACKING ARTIFACTS EXACTLY (0 MISMATCHES).")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify inventory metrics against backing artifacts.")
    parser.add_argument("--inventory", default="reports/project_report/results_inventory.csv", help="Path to inventory CSV")
    args = parser.parse_args()

    success = verify_inventory(args.inventory)
    sys.exit(0 if success else 1)
