"""
scripts/verify_audit_manifest.py
Independent SHA-256 Manifest Verification Script.

Purpose:
Reads reports/leakage_audit/audit_freeze_manifest.sha256, computes the SHA-256
hash of the working-tree bytes of each listed file, and verifies concordance.
Handles both CRLF and LF manifest line endings.
Exits with code 0 on perfect match, or code 1 on any discrepancy.
"""

import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "reports" / "leakage_audit" / "audit_freeze_manifest.sha256"

if not MANIFEST_PATH.exists():
    print(f"ERROR: Manifest file not found at {MANIFEST_PATH}", file=sys.stderr)
    sys.exit(1)

# Read raw bytes and split by any newline style (\r\n or \n)
manifest_content = MANIFEST_PATH.read_bytes().decode("utf-8")
lines = [line.strip() for line in manifest_content.splitlines() if line.strip()]

print("=" * 70)
print(f"VERIFYING SHA-256 MANIFEST ({len(lines)} files listed)")
print(f"Manifest Path: {MANIFEST_PATH}")
print("=" * 70)

mismatches = []
missing_files = []
verified = 0

for line_idx, line in enumerate(lines, 1):
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        mismatches.append(f"Line {line_idx}: malformed manifest line: '{line}'")
        continue
    
    expected_hash, file_rel_path = parts[0].strip(), parts[1].strip()
    target_file = PROJECT_ROOT / file_rel_path
    
    if not target_file.exists():
        missing_files.append(f"MISSING: {file_rel_path}")
        continue
    
    # Hash working-tree bytes directly
    file_bytes = target_file.read_bytes()
    actual_hash = hashlib.sha256(file_bytes).hexdigest()
    
    if actual_hash.lower() != expected_hash.lower():
        mismatches.append(
            f"HASH MISMATCH on {file_rel_path}:\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
    else:
        verified += 1

print(f"\nVerification Summary:")
print(f"  - Verified Files: {verified} / {len(lines)}")
print(f"  - Missing Files:  {len(missing_files)}")
print(f"  - Hash Mismatches:{len(mismatches)}")

if missing_files or mismatches:
    print("\nFAILURES DETECTED:")
    for f in missing_files:
        print("  -", f)
    for m in mismatches:
        print("  -", m)
    sys.exit(1)

print("\nSUCCESS: All files listed in manifest match working-tree bytes bit-for-bit.")
sys.exit(0)
