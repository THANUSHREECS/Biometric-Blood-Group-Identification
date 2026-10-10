import csv
import os
from pathlib import Path
import re

# Paths relative to project root
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "mfs100_ground_truth.csv"
CAPTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "captured"

REQUIRED_COLUMNS = [
    "subject_id",
    "capture_filename",
    "confirmed_blood_group",
    "finger_used",
    "nfiq",
    "quality",
    "confirmation_source",
    "notes",
]

VALID_BLOOD_GROUPS = {"A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"}
SUBJECT_ID_REGEX = re.compile(r"^S\d{3}$")


def error(msg):
    print(f"ERROR: {msg}")
    return False

def warning(msg):
    print(f"WARNING: {msg}")
    return True


def validate():
    if not CSV_PATH.is_file():
        error(f"CSV file not found at {CSV_PATH}")
        return
    if not CAPTURE_DIR.is_dir():
        error(f"Capture directory not found at {CAPTURE_DIR}")
        return

    with open(CSV_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        # Verify required columns
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            error(f"Missing required columns: {', '.join(missing)}")
            return
        rows = list(reader)
        if not rows:
            warning("CSV file contains no data rows.")
            return

        seen = set()
        subjects = set()
        all_ok = True
        for idx, row in enumerate(rows, start=2):  # header is line 1
            # Empty field check
            for col in REQUIRED_COLUMNS:
                if not row.get(col):
                    all_ok = error(f"Row {idx}: column '{col}' is empty.") or all_ok

            subject_id = row["subject_id"].strip()
            if not SUBJECT_ID_REGEX.match(subject_id):
                all_ok = error(f"Row {idx}: subject_id '{subject_id}' does not match S### pattern.") or all_ok
            else:
                subjects.add(subject_id)

            capture_fn = row["capture_filename"].strip()
            if not (CAPTURE_DIR / capture_fn).is_file():
                all_ok = error(f"Row {idx}: capture file '{capture_fn}' not found in {CAPTURE_DIR}.") or all_ok

            bg = row["confirmed_blood_group"].strip()
            if bg not in VALID_BLOOD_GROUPS:
                all_ok = error(f"Row {idx}: blood group '{bg}' is invalid.") or all_ok

            finger = row["finger_used"].strip()
            if finger != "LEFT_THUMB":
                all_ok = error(f"Row {idx}: finger_used must be LEFT_THUMB, got '{finger}'.") or all_ok

            # NFIQ numeric check
            nfiq = row["nfiq"].strip()
            if not nfiq.isdigit():
                all_ok = error(f"Row {idx}: nfiq '{nfiq}' is not numeric.") or all_ok

            # Quality numeric 0‑100 check
            quality = row["quality"].strip()
            try:
                q = float(quality)
                if not (0 <= q <= 100):
                    all_ok = error(f"Row {idx}: quality '{quality}' not in 0‑100 range.") or all_ok
            except ValueError:
                all_ok = error(f"Row {idx}: quality '{quality}' is not numeric.") or all_ok

            # Duplicate (subject_id, capture_filename)
            key = (subject_id, capture_fn)
            if key in seen:
                all_ok = error(f"Row {idx}: duplicate entry for subject '{subject_id}' and capture '{capture_fn}'.") or all_ok
            else:
                seen.add(key)

        # Summary
        print("--- Validation Summary ---")
        print(f"Total rows validated: {len(rows)}")
        print(f"Unique subjects: {len(subjects)}")
        if all_ok:
            print("All checks passed.")
        else:
            print("One or more errors detected.")

if __name__ == "__main__":
    validate()
