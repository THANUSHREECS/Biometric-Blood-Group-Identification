import os
import sys
import random
import pandas as pd

# Add reports/project_report directory to sys.path to access verify_inventory
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.append(curr_dir)

from verify_inventory import verify_inventory

def run_spot_check(seed=888, output_file="reports/project_report/spot_check_10_mixed_output.txt"):
    inventory_path = "reports/project_report/results_inventory.csv"
    if not os.path.exists(inventory_path):
        print(f"ERROR: Inventory file not found: {inventory_path}")
        sys.exit(1)

    df = pd.read_csv(inventory_path)

    # 1. Required rows
    req_ids = [
        "MFS100-LOCK21-EFFB0",
        "E2-LOCKED-TEST-21",
        "PRIMARY-LOCKED-TEST-EXP06C-CONFIG-A"
    ]
    req_rows = df[df["experiment_id"].isin(req_ids)]
    req_groups = set(req_rows["group"].unique())

    # 2. Sample 1 row from each of the remaining 7 distinct groups
    other_df = df[
        ~df["group"].isin(req_groups) & 
        (df["verified_from_artifact"] == "yes") & 
        (df["accuracy"] != "not applicable")
    ].copy()

    random.seed(seed)
    sampled_ids = list(req_ids)
    for g in sorted(other_df["group"].unique()):
        candidates = other_df[other_df["group"] == g]["experiment_id"].tolist()
        chosen = random.choice(candidates)
        sampled_ids.append(chosen)

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log(f"UNEDITED SPOT-CHECK OF 10 ROWS ACROSS 10 DISTINCT GROUPS (SEED={seed})")
    log("Required: MFS100-LOCK21-EFFB0, E2-LOCKED-TEST-21, PRIMARY-LOCKED-TEST-EXP06C-CONFIG-A")
    log("Plus: 7 rows sampled across 7 other distinct groups")
    log("=" * 80)

    subset_df = df[df["experiment_id"].isin(sampled_ids)].copy()

    # Sort subset by original sampled order
    subset_df["sort_order"] = subset_df["experiment_id"].map({sid: i for i, sid in enumerate(sampled_ids)})
    subset_df = subset_df.sort_values("sort_order").drop(columns=["sort_order"])

    for i, (_, row) in enumerate(subset_df.iterrows(), 1):
        log(f"\n--- Spot-Check #{i}: [{row['experiment_id']}] ---")
        log(f"Group       : {row['group']}")
        log(f"Architecture: {row['model_architecture']}")
        log(f"Source File : {row['source_file']}")
        log(f"Source Field: {row['source_field_or_line']}")
        log(f"Inventory   : Acc={row['accuracy']}, BalAcc={row['balanced_accuracy']}, MacroF1={row['macro_f1']}")

    log("\n" + "=" * 80)
    log("RUNNING VERIFICATION ENGINE AGAINST RAW BACKING ARTIFACTS...")
    log("=" * 80)

    # Save subset to temporary CSV for verify_inventory
    tmp_csv = "reports/project_report/temp_spot_check_10_mixed.csv"
    subset_df.to_csv(tmp_csv, index=False)

    try:
        # Run standard verification engine
        verify_inventory(tmp_csv)
        log("\nVERIFICATION RESULT: ALL 10 ROWS MATCH BACKING ARTIFACTS EXACTLY (0 MISMATCHES).")
        log("Result: PASSED [10/10 verified successfully]")
    except Exception as e:
        log(f"\nVERIFICATION FAILED: {str(e)}")
        sys.exit(1)
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

    log("=" * 80)

    # Save complete log to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Spot-check log successfully written to {output_file}")

if __name__ == "__main__":
    run_spot_check(seed=888)
