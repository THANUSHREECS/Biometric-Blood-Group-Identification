import os

files_to_edit = {
    "src/predictor.py": [
        ("38.10%", "origin unknown"),
        ("~38%", "origin unknown")
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
            print(f"Removed 38% claims from {filepath}")
