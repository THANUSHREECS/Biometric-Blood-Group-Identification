"""
scratch/run_negative_test.py
Demonstrates that verify_leakage_audit_report_concordance.py catches mismatches
and exits with a non-zero exit code when a reported number is changed.
"""

import sys
import subprocess
from pathlib import Path

orig_path = Path("reports/leakage_audit/leakage_audit_report.md")
orig_text = orig_path.read_text(encoding="utf-8")

# Tamper with one specific reported number in Section 3.1
target_str = "Hamming $\\le 5$: 46 images ($11.08\\%$)"
tampered_str = "Hamming $\\le 5$: 99 images ($23.85\\%$)"

assert target_str in orig_text, f"Target string not found in {orig_path}"
tampered_text = orig_text.replace(target_str, tampered_str)

temp_file = Path("scratch/temp_tampered_report.md")
temp_file.write_text(tampered_text, encoding="utf-8")

try:
    res = subprocess.run(
        [sys.executable, "scripts/verify_leakage_audit_report_concordance.py", str(temp_file)],
        capture_output=True,
        text=True
    )
    print("=== NEGATIVE CONCORDANCE TEST OUTPUT ===")
    print("STDOUT:\n" + res.stdout)
    if res.stderr:
        print("STDERR:\n" + res.stderr)
    print(f"EXIT CODE: {res.returncode}")
    assert res.returncode == 1, f"Expected exit code 1, got {res.returncode}"
    print("\nNEGATIVE TEST PASSED: Non-zero exit code confirmed on tampered report.")
finally:
    temp_file.unlink(missing_ok=True)
