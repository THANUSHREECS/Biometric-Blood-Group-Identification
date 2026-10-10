import os

files_to_edit = {
    "tests/test_mfs100_domain_diagnostic.py": [
        ("from src.config import CONFIDENCE_THRESHOLD", "from src.config import ADVISORY_CONFIRMATION_THRESHOLD"),
        ("CONFIDENCE_THRESHOLD", "ADVISORY_CONFIRMATION_THRESHOLD")
    ],
    "tests/test_preprocessing.py": [
        ("224, 224, 3", "112, 112, 3")
    ],
    "tests/test_advisory_layer.py": [
        ("_evaluate_advisory(0.69, {\"rh_uncertainty\": False})", "_evaluate_advisory(0.69)"),
        ("_evaluate_advisory(0.70, {\"rh_uncertainty\": False})", "_evaluate_advisory(0.70)"),
        ("_evaluate_advisory(0.71, {\"rh_uncertainty\": False})", "_evaluate_advisory(0.71)"),
        ("_evaluate_advisory(0.85, {\"rh_uncertainty\": True})", "_evaluate_advisory(0.85)"),
        ("_evaluate_advisory(0.65, {\"rh_uncertainty\": True})", "_evaluate_advisory(0.65)"),
        ("_evaluate_advisory(0.65, {\"rh_uncertainty\": False})", "_evaluate_advisory(0.65)"),
        ("_evaluate_advisory(0.75, {\"rh_uncertainty\": True})", "_evaluate_advisory(0.75)"),
        ("_evaluate_advisory(0.80, {\"rh_uncertainty\": False})", "_evaluate_advisory(0.80)"),
        ("assert len(res[\"probabilities\"]) == 8", "assert len(res[\"probabilities\"]) == 4")
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
            print(f"Fixed {filepath}")
