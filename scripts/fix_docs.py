import os

def replace_in_file(path, old, new):
    if not os.path.exists(path): return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

# Fix README and VIVA
replacements = [
    ("38.10%", "86.02% (random split, includes identity leakage) | ~0.00% (group-aware)"),
    ("38%", "86% (random split)"),
    ("zero cross-split leakage", ""),
    ("ridge patterns correlate strongly with ABO", "we investigate if ridge patterns correlate with ABO"),
    ("The observed 38.10% accuracy", "The observed 86.02% random-split accuracy"),
]

files = [
    "README.md",
    "reports/external_evaluation/VIVA_PREPARATION.md",
    "templates/index.html",
    "static/js/app.js",
    "src/report_generator.py"
]

for f in files:
    for old, new in replacements:
        replace_in_file(f, old, new)

# Special README fixes for limitation 4
readme_path = "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    readme = f.read()

# Fix limitation 4
if "While the true participant-isolated 4-class accuracy of 86.02%" in readme:
    readme = readme.replace("While the true participant-isolated 4-class accuracy of 86.02%", "While the random-split 4-class accuracy of 86.02%")
if "statistically higher than random guessing (25%, $p < 0.001$)" in readme:
    readme = readme.replace("statistically higher than random guessing (25%, $p < 0.001$)", "statistically higher than random guessing (25%, $p < 0.001$) on leaked data")

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

print("Documentation and headers fixed.")
