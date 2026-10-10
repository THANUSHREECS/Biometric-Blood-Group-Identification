import os

files_to_edit = {
    "live_app/frontend/index.html": [
        ("38.10%", "origin unknown"),
        (">38.10%<", ">origin unknown<")
    ],
    "live_app/frontend/src/App.tsx": [
        ("38.10%", "origin unknown")
    ],
    "app.py": [
        ("38.10%", "origin unknown")
    ],
    "src/report_generator.py": [
        ("38.10%", "origin unknown")
    ],
    "reports/external_evaluation/FOUR_CLASS_TEST_RESULTS.md": [
        ("- **Accuracy:** 38.10%", "- **Accuracy:** origin unknown")
    ],
    "results/AUDIT_SUMMARY.md": [
        ("38.10%", "origin unknown")
    ],
    "README.md": [
        ("38.10%", "origin unknown")
    ]
}

for filepath, replacements in files_to_edit.items():
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        modified = False
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                modified = True
        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Removed 38.10% from {filepath}")
