# Append a single row to the ground‑truth CSV safely
import csv
import os
from pathlib import Path
from typing import Dict

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "mfs100_ground_truth.csv"

_REQUIRED_FIELDS = [
    "subject_id",
    "capture_filename",
    "confirmed_blood_group",
    "finger_used",
    "nfiq",
    "quality",
    "confirmation_source",
    "notes",
]

def _row_exists(subject_id: str, capture_filename: str) -> bool:
    if not CSV_PATH.is_file():
        return False
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("subject_id") == subject_id and r.get("capture_filename") == capture_filename:
                return True
    return False

def append_row(row: Dict[str, str]) -> None:
    """Append a validated row to ``data/mfs100_ground_truth.csv``.

    The caller must ensure that ``row`` contains all required keys and that the
    values are already validated (e.g., blood‑group, numeric fields). This helper
    guarantees that duplicate ``(subject_id, capture_filename)`` entries are not
    written.
    """
    # Ensure CSV exists with header
    if not CSV_PATH.is_file():
        # Create file with header
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_REQUIRED_FIELDS)
            writer.writeheader()

    # Prevent duplicate entry
    if _row_exists(row["subject_id"], row["capture_filename"]):
        raise ValueError(f"Duplicate entry for subject {row['subject_id']} and file {row['capture_filename']}")

    # Append
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_REQUIRED_FIELDS)
        writer.writerow(row)
