import os
import re

files_to_edit = [
    "README.md",
    "reports/external_evaluation/VIVA_PREPARATION.md",
    "live_app/frontend/index.html",
    "src/report_generator.py",
    "results/AUDIT_SUMMARY.md",
    "reports/external_evaluation/EXTERNAL_DEMO_SCRIPT.md",
    "reports/external_evaluation/FINAL_DEMO_CHECKLIST.md"
]

remove_phrases = [
    r"~0\.00% group-aware",
    r"model learned 4 identities",
    r"cluster = one person's burst",
    r"54% proves identity leakage",
    r"ensemble flattens confidence"
]

replacement_text = "86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible."

for fpath in files_to_edit:
    if not os.path.exists(fpath):
        continue

    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    modified = False
    
    # Try replacing some specific matches
    for phrase in remove_phrases:
        if re.search(phrase, content, flags=re.IGNORECASE):
            content = re.sub(phrase, replacement_text, content, flags=re.IGNORECASE)
            modified = True
            
    if modified:
        print(f"Modified {fpath}")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
