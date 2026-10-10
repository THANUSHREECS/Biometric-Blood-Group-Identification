import os

def replace_in_file(path, old, new):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Replaced in {path}")

files = [
    "README.md",
    "reports/external_evaluation/VIVA_PREPARATION.md",
    "live_app/frontend/index.html",
    "src/report_generator.py",
    "results/AUDIT_SUMMARY.md",
    "reports/external_evaluation/FINAL_PROJECT_READINESS_REPORT.md",
    "reports/external_evaluation/EXTERNAL_DEMO_SCRIPT.md"
]

to_replace = [
    ("~0.00% (group-aware generalization)", "Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible."),
    ("~0.00%", "Subject independence cannot be verified."),
    ("model learned 4 identities", "Dataset artifacts are possible."),
    ("54.00% of the test images share their closest nearest neighbor with the exact same cluster in the training set, strongly confirming identity leakage.", "Subject independence cannot be verified (no subject IDs)."),
    ("The live app's 5-fold ensemble inherently flattens out extreme probability spikes", "Dataset artifacts are possible."),
    ("86.02% (random split, includes identity leakage)", "86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible."),
    ("86.02%", "86.0% on a random split"),
]

for f in files:
    for old, new in to_replace:
        replace_in_file(f, old, new)
