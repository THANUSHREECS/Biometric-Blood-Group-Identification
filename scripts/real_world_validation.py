import os
import csv
from typing import List, Dict

from src.config import CONFIDENCE_THRESHOLD
from src.predictor import BloodGroupPredictor
from src.mfs100.scanner import MFS100Scanner

# ---------------------------------------------------------------------------
# Real‑World Validation Script
# ---------------------------------------------------------------------------
# This script performs a systematic validation of the complete system using
# the physical Mantra MFS100 scanner with the standardized LEFT THUMB protocol.
# It records detailed metadata for each capture and writes results to a CSV
# file plus a markdown summary report.
#
# IMPORTANT RESTRICTIONS (enforced by design):
#   • No model retraining or weight changes.
#   • Champion Ensemble V2 remains frozen.
#   • Locked test set is never accessed.
#   • LEFT_THUMB is acquisition metadata only.
#   • No fabricated results – everything is captured live.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Configuration – define the subjects you will test.
# ---------------------------------------------------------------------------
# Each entry should contain a unique ``sample_id`` and the laboratory‑confirmed
# blood group (e.g. "AB+", "O-", etc.).  Extend this list with the actual
# participants before running the script.
# ---------------------------------------------------------------------------
SUBJECTS: List[Dict[str, str]] = [
    {"sample_id": "S001", "confirmed_group": "O+"},
    {"sample_id": "S002", "confirmed_group": "AB+"},
    # Add more subjects as needed.
]

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _format_row(row: Dict) -> List:
    """Return a CSV‑compatible list preserving column order used in the report."""
    return [
        row["sample_id"],
        row["confirmed_group"],
        row["finger"],
        row["scan_number"],
        row.get("nfiq", ""),
        row.get("quality", ""),
        row["predicted_group"],
        f"{row['confidence']:.4f}",
        row.get("abo_prediction", ""),
        row.get("rh_prediction", ""),
        row["is_correct"],
    ]

def run_validation() -> None:
    # Force hardware mode – do NOT use mock.
    scanner = MFS100Scanner(mock=False)
    predictor = BloodGroupPredictor(confidence_threshold=CONFIDENCE_THRESHOLD)

    results_csv = os.path.abspath(os.path.join("outputs", "REAL_WORLD_VALIDATION_RESULTS.csv"))
    _ensure_dir(results_csv)

    header = [
        "sample_id",
        "confirmed_group",
        "finger",
        "scan_number",
        "nfiq",
        "quality",
        "predicted_group",
        "confidence",
        "abo_prediction",
        "rh_prediction",
        "is_correct",
    ]

    all_rows: List[Dict] = []

    for subj in SUBJECTS:
        sid = subj["sample_id"]
        true_bg = subj["confirmed_group"]
        scan_meta: List[Dict] = []
        scan_paths: List[str] = []
        # -------------------------------------------------------------------
        # Three individual impressions
        # -------------------------------------------------------------------
        for i in range(1, 4):
            img_path, _, meta = scanner.capture_with_meta(sample_class=None)
            scan_paths.append(img_path)
            scan_meta.append(meta)

            pred = predictor.predict(img_path, source="mfs100")
            abo_pred = pred.get("abo_analysis", {}).get("abo") if isinstance(pred.get("abo_analysis"), dict) else "?"
            rh_pred = pred.get("abo_analysis", {}).get("rh") if isinstance(pred.get("abo_analysis"), dict) else "?"

            row = {
                "sample_id": sid,
                "confirmed_group": true_bg,
                "finger": "LEFT_THUMB",
                "scan_number": i,
                "nfiq": meta.get("nfiq", ""),
                "quality": meta.get("quality", ""),
                "predicted_group": pred["blood_group"],
                "confidence": pred["confidence"],
                "abo_prediction": abo_pred,
                "rh_prediction": rh_pred,
                "is_correct": pred["blood_group"] == true_bg,
            }
            all_rows.append(row)

        # -------------------------------------------------------------------
        # Consensus (3‑scan majority) prediction
        # -------------------------------------------------------------------
        consensus_pred = predictor.predict_consensus(scan_paths, source="mfs100")
        consensus_abo = consensus_pred.get("abo_analysis", {}).get("abo") if isinstance(consensus_pred.get("abo_analysis"), dict) else "?"
        consensus_rh = consensus_pred.get("abo_analysis", {}).get("rh") if isinstance(consensus_pred.get("abo_analysis"), dict) else "?"
        consensus_row = {
            "sample_id": sid,
            "confirmed_group": true_bg,
            "finger": "LEFT_THUMB",
            "scan_number": "CONSENSUS",
            "nfiq": min(m.get("nfiq", 0) for m in scan_meta),
            "quality": round(sum(m.get("quality", 0) for m in scan_meta) / len(scan_meta)),
            "predicted_group": consensus_pred["blood_group"],
            "confidence": consensus_pred["confidence"],
            "abo_prediction": consensus_abo,
            "rh_prediction": consensus_rh,
            "is_correct": consensus_pred["blood_group"] == true_bg,
        }
        all_rows.append(consensus_row)

    # Write CSV
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in all_rows:
            writer.writerow(_format_row(r))

    print(f"Results saved to {results_csv}")

    # -------------------------------------------------------------------
    # Summary statistics & markdown report
    # -------------------------------------------------------------------
    total_subjects = len(SUBJECTS)
    total_scans = total_subjects * 3
    correct = sum(1 for r in all_rows if r["is_correct"] and r["scan_number"] != "CONSENSUS")
    accuracy = correct / total_scans if total_scans else 0.0

    # Per‑blood‑group accuracy
    grp_stats: Dict[str, Dict[str, int]] = {}
    for r in all_rows:
        if r["scan_number"] == "CONSENSUS":
            continue
        grp = r["confirmed_group"]
        grp_stats.setdefault(grp, {"total": 0, "correct": 0})
        grp_stats[grp]["total"] += 1
        if r["is_correct"]:
            grp_stats[grp]["correct"] += 1

    per_group_lines = []
    for grp, stats in grp_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        per_group_lines.append(f"- **{grp}**: {stats['correct']}/{stats['total']} ({acc:.1%})")

    # ABO / Rh accuracy (where derivable)
    abo_correct = sum(
        1 for r in all_rows
        if r.get("abo_prediction") not in (None, "?") and r["abo_prediction"] == r["confirmed_group"].replace("+", "").replace("-", "")
    )
    rh_correct = sum(
        1 for r in all_rows
        if r.get("rh_prediction") not in (None, "?") and (
            (r["rh_prediction"] == "+" and r["confirmed_group"].endswith("+")) or
            (r["rh_prediction"] == "-" and r["confirmed_group"].endswith("-"))
        )
    )
    total_abo = sum(1 for r in all_rows if r.get("abo_prediction") not in (None, "?"))
    total_rh = sum(1 for r in all_rows if r.get("rh_prediction") not in (None, "?"))
    abo_acc = abo_correct / total_abo if total_abo else 0.0
    rh_acc = rh_correct / total_rh if total_rh else 0.0

    avg_conf = sum(r["confidence"] for r in all_rows if isinstance(r["confidence"], (int, float))) / len(all_rows)
    low_conf = sum(1 for r in all_rows if isinstance(r["confidence"], (int, float)) and r["confidence"] < CONFIDENCE_THRESHOLD)

    # Consensus accuracy
    consensus_correct = sum(1 for r in all_rows if r["scan_number"] == "CONSENSUS" and r["is_correct"])
    consensus_acc = consensus_correct / total_subjects if total_subjects else 0.0

    # Repeatability (all three scans agree)
    repeatable = 0
    for subj in SUBJECTS:
        preds = [r["predicted_group"] for r in all_rows if r["sample_id"] == subj["sample_id"] and isinstance(r["scan_number"], int)]
        if len(set(preds)) == 1:
            repeatable += 1
    repeatability = repeatable / total_subjects if total_subjects else 0.0

    report_path = os.path.abspath(os.path.join("docs", "REAL_WORLD_VALIDATION_REPORT.md"))
    _ensure_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as md:
        md.write("# Real‑World Validation Report (Physical MFS100)\n\n")
        md.write("**Scope** – Validation uses the physical Mantra MFS100 scanner with the LEFT THUMB protocol. All participants have a laboratory‑confirmed blood group. This is experimental research; no clinical claims are made.\n\n")
        md.write(f"- Total subjects: **{total_subjects}**\n")
        md.write(f"- Total individual scans: **{total_scans}**\n")
        md.write(f"- Correct individual predictions: **{correct}** ({accuracy:.1%})\n")
        md.write("- Per‑blood‑group accuracy:\n")
        md.write("\n".join(per_group_lines) + "\n")
        md.write(f"- ABO accuracy (where derivable): **{abo_acc:.1%}**\n")
        md.write(f"- Rh accuracy (where derivable): **{rh_acc:.1%}**\n")
        md.write(f"- Average confidence: **{avg_conf:.2%}**\n")
        md.write(f"- Low‑confidence (< {CONFIDENCE_THRESHOLD:.2f}) count: **{low_conf}**\n")
        md.write(f"- Consensus accuracy (3‑scan majority): **{consensus_acc:.1%}**\n")
        md.write(f"- Repeatability (all 3 scans agree): **{repeatability:.1%}**\n\n")
        md.write("---\n\n")
        md.write("**Disclaimer** – The AI predictions are preliminary estimates and **must** be confirmed with standard laboratory serological blood typing before any clinical use. This report is for research and demonstration purposes only.\n")

    print(f"Report written to {report_path}")

if __name__ == "__main__":
    run_validation()
