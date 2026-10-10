"""
scripts/live_stress_test.py
============================
Comprehensive Mantra MFS100 Live Scanner Stress Test
-----------------------------------------------------
Tests real-world prediction accuracy and consistency across multiple human subjects
using the physical scanner. Records per-scan quality, confidence, crop metadata,
and computes consistency scores. Generates a detailed HTML report.

STRICT CONSTRAINTS:
  - Does NOT access the locked 898-image test set
  - Does NOT modify or retrain any model
  - Does NOT change ensemble weights (ResNet50 58%, VGG16 38%, EfficientNetB0 4%)
  - Source is always "mfs100" for hardware captures (applies foreground crop + ridge enhancement)

Usage:
  python scripts/live_stress_test.py
  python scripts/live_stress_test.py --scans 5 --subjects 4
  python scripts/live_stress_test.py --mock  (runs in mock mode for offline testing)
"""

import os
import sys
import json
import time
import argparse
import datetime
import statistics
import webbrowser
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL   = "http://127.0.0.1:5000"
BRIDGE_URL = "http://127.0.0.1:5100"
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

NFIQ_LABELS = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
CONFIDENCE_THRESHOLD = 0.60
VALID_BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# ── Colour helpers (ANSI) ─────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def cprint(msg, color=RESET):
    print(f"{color}{msg}{RESET}")

def banner(title):
    width = 64
    print()
    cprint("=" * width, CYAN)
    cprint(f"  {title}", BOLD + CYAN)
    cprint("=" * width, CYAN)

def section(title):
    print()
    cprint(f"-- {title} " + "-" * max(0, 56 - len(title)), YELLOW)

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def safe_get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        return r, None
    except Exception as e:
        return None, str(e)

def safe_post(url, timeout=90, **kwargs):
    try:
        r = requests.post(url, timeout=timeout, **kwargs)
        return r, None
    except Exception as e:
        return None, str(e)

# ── Pre-flight checks ─────────────────────────────────────────────────────────
def check_services(mock_mode):
    section("Pre-flight Service Check")

    r, err = safe_get(f"{BASE_URL}/api/status")
    if err or r is None or r.status_code != 200:
        cprint(f"  [!!] Web app not reachable at {BASE_URL}: {err or r.status_code}", RED)
        cprint("       Start the app first:  python app.py", RED)
        return False

    app_data = r.json()
    model_info = app_data.get("model", {})
    cprint(f"  [OK] Web app reachable -- Model: {model_info.get('name', '?')}", GREEN)
    cprint(f"       Accuracy: {model_info.get('accuracy', '?')}  |  "
           f"Ensemble: {model_info.get('type', '?')}", GREEN)

    if not mock_mode:
        r2, err2 = safe_get(f"{BRIDGE_URL}/status")
        if err2 or r2 is None or r2.status_code != 200:
            cprint(f"  [!!] Bridge not reachable at {BRIDGE_URL}: {err2 or (r2.status_code if r2 else 'no response')}", RED)
            cprint("       Start it:  C:\\Python310-32\\python.exe -m src.mfs100_bridge.bridge", RED)
            return False
        bd = r2.json()
        cprint(f"  [OK] MFS100 Bridge running -- arch={bd.get('python_arch','?')}", GREEN)

        r3, err3 = safe_post(f"{BASE_URL}/api/scanner/mode", json={"mock": False})
        if err3 or r3 is None or r3.status_code != 200:
            cprint(f"  [!!] Could not switch to hardware mode: {err3 or r3.text}", RED)
            return False
        cprint("  [OK] Switched to MFS100 Hardware mode", GREEN)
    else:
        safe_post(f"{BASE_URL}/api/scanner/mode", json={"mock": True})
        cprint("  [OK] Running in MOCK MODE (no physical scanner required)", YELLOW)

    return True

# ── Single capture via app API ────────────────────────────────────────────────
def do_single_capture():
    r, err = safe_post(f"{BASE_URL}/api/capture", timeout=90)
    if err or r is None:
        return None, f"Request error: {err}"
    if r.status_code != 200:
        try:
            msg = r.json().get("error", r.text)
        except Exception:
            msg = r.text
        return None, f"HTTP {r.status_code}: {msg}"
    return r.json(), None

# ── Consensus capture (3-impression soft voting) ──────────────────────────────
def do_consensus_capture():
    r, err = safe_post(f"{BASE_URL}/api/capture/consensus", timeout=180)
    if err or r is None:
        return None, f"Request error: {err}"
    if r.status_code != 200:
        try:
            msg = r.json().get("error", r.text)
        except Exception:
            msg = r.text
        return None, f"HTTP {r.status_code}: {msg}"
    return r.json(), None

# ── Statistics helpers ────────────────────────────────────────────────────────
def consistency_score(predictions):
    if not predictions:
        return None, 0, 0.0, False
    freq = defaultdict(int)
    for p in predictions:
        freq[p] += 1
    dominant = max(freq, key=freq.get)
    count = freq[dominant]
    pct = (count / len(predictions)) * 100
    return dominant, count, pct, (pct >= 66.7)

def confidence_trend(confidences):
    if not confidences:
        return {}
    return {
        "mean": round(statistics.mean(confidences), 4),
        "min":  round(min(confidences), 4),
        "max":  round(max(confidences), 4),
        "std":  round(statistics.stdev(confidences) if len(confidences) > 1 else 0.0, 4)
    }

# ── Per-subject scan session ──────────────────────────────────────────────────
def run_subject_session(subject_id, subject_label, num_scans, use_consensus, known_blood_group=None):
    banner(f"Subject {subject_id}: {subject_label}")
    if known_blood_group:
        cprint(f"  Known blood group: {BOLD}{known_blood_group}{RESET}", CYAN)

    subject_result = {
        "subject_id": subject_id,
        "subject_label": subject_label,
        "known_blood_group": known_blood_group,
        "num_scans_requested": num_scans,
        "scans": [],
        "consensus_result": None,
        "aggregate": {}
    }

    predictions   = []
    confidences   = []
    quality_scores = []
    nfiq_scores   = []
    crop_applied_count = 0

    for scan_idx in range(1, num_scans + 1):
        section(f"Scan {scan_idx} of {num_scans}")
        input(f"  Place finger on sensor and press ENTER to capture scan {scan_idx}...")

        t_start = time.time()
        result, err = do_single_capture()
        elapsed = round(time.time() - t_start, 2)

        if err or result is None:
            cprint(f"  [!!] Capture failed: {err}", RED)
            subject_result["scans"].append({
                "scan_index": scan_idx,
                "status": "CAPTURE_ERROR",
                "error": str(err),
                "elapsed_s": elapsed
            })
            continue

        blood_group  = result.get("blood_group", "?")
        confidence   = float(result.get("confidence", 0.0))
        pred_status  = result.get("status", "?")
        # API returns quality_metadata (dict with quality/nfiq/pressure_status/placement_advice)
        quality_meta = result.get("quality_metadata") or result.get("quality_meta") or {}
        crop_meta    = result.get("crop_metadata") or {}
        probs        = result.get("probabilities", {})
        scanner_mode = result.get("scanner_mode", "mock")

        quality  = quality_meta.get("quality", "N/A")
        nfiq     = quality_meta.get("nfiq", "N/A")
        pressure = quality_meta.get("pressure_status", "N/A")
        advice   = quality_meta.get("placement_advice", "")
        crop_ok  = crop_meta.get("crop_applied", False)

        predictions.append(blood_group)
        confidences.append(confidence)
        if isinstance(quality, int):
            quality_scores.append(quality)
        if isinstance(nfiq, int):
            nfiq_scores.append(nfiq)
        if crop_ok:
            crop_applied_count += 1

        is_low = confidence < CONFIDENCE_THRESHOLD
        conf_color = GREEN if not is_low else YELLOW

        cprint(f"  -> Blood Group : {BOLD}{blood_group}{RESET}", GREEN)
        cprint(f"  -> Confidence  : {confidence*100:.1f}%{'  [WARNING: LOW]' if is_low else ''}", conf_color)

        if known_blood_group:
            matched = blood_group == known_blood_group
            cprint(f"  -> Accuracy    : {'[MATCH]' if matched else '[MISMATCH]'}",
                   GREEN if matched else RED)

        cprint(f"  -> Quality     : {quality}/100  NFIQ={nfiq} ({NFIQ_LABELS.get(nfiq, '?')})", CYAN)
        cprint(f"  -> Pressure    : {pressure}", CYAN)
        cprint(f"  -> Crop Applied: {'Yes' if crop_ok else 'No'}", CYAN)
        if advice:
            cprint(f"  -> Advice      : {advice}", YELLOW)

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        cprint(f"  -> Top-3 Probs : " + ", ".join(f"{k}={v*100:.1f}%" for k, v in sorted_probs), RESET)

        subject_result["scans"].append({
            "scan_index": scan_idx,
            "blood_group": blood_group,
            "confidence": round(confidence, 4),
            "is_low_confidence": is_low,
            "status": pred_status,
            "quality": quality,
            "nfiq": nfiq,
            "pressure_status": pressure,
            "placement_advice": advice,
            "crop_applied": crop_ok,
            "crop_metadata": crop_meta,
            "probabilities": probs,
            "elapsed_s": elapsed,
            "correct": (blood_group == known_blood_group) if known_blood_group else None
        })

        if scan_idx < num_scans:
            time.sleep(0.5)

    # 3-scan consensus
    if use_consensus:
        section("3-Scan Consensus Mode")
        cprint("  The server will prompt for 3 consecutive impressions.", CYAN)
        input("  Press ENTER when ready to begin consensus capture...")

        t_start = time.time()
        cons_result, cons_err = do_consensus_capture()
        elapsed_cons = round(time.time() - t_start, 2)

        if cons_err or cons_result is None:
            cprint(f"  [!!] Consensus capture failed: {cons_err}", RED)
        else:
            cons_bg   = cons_result.get("blood_group", "?")
            cons_conf = float(cons_result.get("confidence", 0.0))
            cons_meta = cons_result.get("consensus_metadata", {})
            agreement = cons_meta.get("agreement_pct", 0)

            cprint(f"  -> Consensus Blood Group : {BOLD}{cons_bg}{RESET}", GREEN)
            cprint(f"  -> Consensus Confidence  : {cons_conf*100:.1f}%",
                   GREEN if cons_conf >= CONFIDENCE_THRESHOLD else YELLOW)
            cprint(f"  -> Agreement Rate        : {agreement}%",
                   GREEN if agreement >= 66 else YELLOW)
            if known_blood_group:
                ok = cons_bg == known_blood_group
                cprint(f"  -> Consensus Accuracy    : {'[MATCH]' if ok else '[MISMATCH]'}",
                       GREEN if ok else RED)

            subject_result["consensus_result"] = {
                "blood_group": cons_bg,
                "confidence": round(cons_conf, 4),
                "agreement_pct": agreement,
                "individual_scans": cons_meta.get("individual_scans", []),
                "correct": (cons_bg == known_blood_group) if known_blood_group else None,
                "elapsed_s": elapsed_cons
            }

    # Aggregate stats for this subject
    successful_scans = [s for s in subject_result["scans"] if "blood_group" in s]
    dom_class, dom_count, dom_pct, is_consistent = consistency_score(predictions)
    conf_stats = confidence_trend(confidences)
    correct_count  = sum(1 for s in successful_scans if s.get("correct") is True)
    total_with_gt  = sum(1 for s in successful_scans if s.get("correct") is not None)

    subject_result["aggregate"] = {
        "total_scans":         num_scans,
        "successful_scans":    len(successful_scans),
        "dominant_prediction": dom_class,
        "dominant_count":      dom_count,
        "consistency_pct":     round(dom_pct, 1),
        "is_consistent":       is_consistent,
        "confidence_stats":    conf_stats,
        "avg_quality":         round(statistics.mean(quality_scores), 1) if quality_scores else None,
        "avg_nfiq":            round(statistics.mean(nfiq_scores), 2) if nfiq_scores else None,
        "crop_applied_count":  crop_applied_count,
        "correct_count":       correct_count if known_blood_group else None,
        "accuracy_pct":        round(correct_count / total_with_gt * 100, 1) if total_with_gt > 0 else None,
        "all_predictions":     predictions,
        "all_confidences":     confidences
    }

    section(f"Subject {subject_id} Summary")
    agg = subject_result["aggregate"]
    cprint(f"  Successful scans    : {agg['successful_scans']}/{num_scans}", GREEN)
    cprint(f"  Dominant prediction : {dom_class} ({dom_count}/{len(predictions)} = {dom_pct:.0f}%)",
           GREEN if is_consistent else YELLOW)
    cprint(f"  Consistency         : {'CONSISTENT' if is_consistent else 'INCONSISTENT'} (>=67% threshold)",
           GREEN if is_consistent else YELLOW)
    if confidences:
        cprint(f"  Confidence mean     : {conf_stats.get('mean',0)*100:.1f}%  "
               f"[min={conf_stats.get('min',0)*100:.1f}%, max={conf_stats.get('max',0)*100:.1f}%, "
               f"std={conf_stats.get('std',0)*100:.1f}%]", CYAN)
    if agg["avg_quality"] is not None:
        cprint(f"  Avg Quality Score   : {agg['avg_quality']}/100", CYAN)
    if known_blood_group and agg["accuracy_pct"] is not None:
        cprint(f"  Scan Accuracy       : {correct_count}/{total_with_gt} = {agg['accuracy_pct']}%",
               GREEN if agg["accuracy_pct"] >= 66.7 else RED)

    return subject_result


# ── HTML report generator ─────────────────────────────────────────────────────
def generate_html_report(all_subjects, timestamp_str, mock_mode):
    total_scans      = sum(s["aggregate"]["total_scans"] for s in all_subjects)
    successful_total = sum(s["aggregate"]["successful_scans"] for s in all_subjects)
    consistent_count = sum(1 for s in all_subjects if s["aggregate"]["is_consistent"])
    all_confidences  = [c for s in all_subjects for c in s["aggregate"].get("all_confidences", [])]
    subjects_gt      = [s for s in all_subjects if s.get("known_blood_group")]
    total_correct    = sum(s["aggregate"].get("correct_count") or 0 for s in subjects_gt)
    total_gt_scans   = sum(s["aggregate"]["successful_scans"] for s in subjects_gt)
    global_accuracy  = round(total_correct / total_gt_scans * 100, 1) if total_gt_scans > 0 else None
    avg_conf         = round(statistics.mean(all_confidences) * 100, 1) if all_confidences else 0
    consistency_rate = round(consistent_count / len(all_subjects) * 100, 1) if all_subjects else 0

    def badge(text, bg):
        return (f'<span style="background:{bg};color:#fff;padding:3px 10px;border-radius:12px;'
                f'font-size:0.82em;font-weight:600;margin-left:6px">{text}</span>')

    def conf_bar(val):
        pct = round(val * 100, 1)
        color = "#22c55e" if val >= 0.6 else "#f59e0b" if val >= 0.4 else "#ef4444"
        return (f'<div style="display:inline-flex;align-items:center;gap:8px">'
                f'<div style="background:#0f172a;border-radius:6px;overflow:hidden;height:14px;width:120px">'
                f'<div style="width:{pct}%;background:{color};height:100%"></div></div>'
                f'<span style="font-size:0.85em">{pct}%</span></div>')

    # Build subject cards
    subject_cards_html = ""
    for s in all_subjects:
        agg   = s["aggregate"]
        known = s.get("known_blood_group") or ""
        col   = "#22c55e" if agg["is_consistent"] else "#f59e0b"

        rows = ""
        for sc in s["scans"]:
            if "blood_group" not in sc:
                rows += (f'<tr><td style="padding:6px 8px">{sc["scan_index"]}</td>'
                         f'<td colspan="7" style="padding:6px 8px;color:#ef4444">'
                         f'ERROR: {sc.get("error","?")}</td></tr>')
                continue
            bg   = sc["blood_group"]
            niq  = NFIQ_LABELS.get(sc.get("nfiq"), "?")
            crop_html = ('<span style="color:#22c55e">Cropped</span>' if sc.get("crop_applied")
                         else '<span style="color:#64748b">—</span>')
            if known:
                ok = bg == known
                ok_html = (f'<span style="color:{"#22c55e" if ok else "#ef4444"}">'
                           f'{"&#10003;" if ok else "&#10007;"}</span>')
            else:
                ok_html = "—"
            rows += (f'<tr style="border-bottom:1px solid #1e293b">'
                     f'<td style="padding:6px 8px">{sc["scan_index"]}</td>'
                     f'<td style="padding:6px 8px"><strong>{bg}</strong></td>'
                     f'<td style="padding:6px 8px">{conf_bar(sc["confidence"])}</td>'
                     f'<td style="padding:6px 8px;color:{"#f59e0b" if sc["is_low_confidence"] else "#22c55e"}">'
                     f'{"LOW" if sc["is_low_confidence"] else "OK"}</td>'
                     f'<td style="padding:6px 8px">{sc.get("quality","?")}/100</td>'
                     f'<td style="padding:6px 8px">NFIQ {sc.get("nfiq","?")} ({niq})</td>'
                     f'<td style="padding:6px 8px">{crop_html}</td>'
                     f'<td style="padding:6px 8px;text-align:center">{ok_html}</td>'
                     f'</tr>')

        cons_html = ""
        if s.get("consensus_result"):
            cr = s["consensus_result"]
            cons_badge = ""
            if known:
                ok = cr["blood_group"] == known
                cons_badge = badge("MATCH" if ok else "MISMATCH",
                                   "#22c55e" if ok else "#ef4444")
            cons_html = (f'<div style="margin-top:12px;padding:10px 16px;background:#0f172a;'
                         f'border-radius:8px;border-left:4px solid #6366f1">'
                         f'<strong>3-Scan Consensus:</strong>&nbsp;'
                         f'<span style="font-size:1.1em;color:#a5f3fc"><strong>{cr["blood_group"]}</strong></span>'
                         f'&nbsp;{conf_bar(cr["confidence"])}'
                         f'&nbsp; Agreement: <strong>{cr["agreement_pct"]}%</strong>{cons_badge}</div>')

        acc_badge = ""
        if known and agg.get("accuracy_pct") is not None:
            acc_badge = badge(f'{agg["accuracy_pct"]}% scans correct',
                              "#22c55e" if agg["accuracy_pct"] >= 66.7 else "#ef4444")

        subject_cards_html += (
            f'<div style="background:#1e293b;border-radius:12px;padding:20px 24px;margin-bottom:24px">'
            f'<h3 style="margin:0 0 6px;font-size:1.1em">'
            f'Subject {s["subject_id"]}: {s["subject_label"]}'
            f'{badge("Known: " + known, "#6366f1") if known else ""}'
            f'{badge("Dominant: " + (agg["dominant_prediction"] or "?"), col)}'
            f'{badge(str(agg["consistency_pct"]) + "% consistent", col)}'
            f'{acc_badge}'
            f'</h3>'
            f'<p style="color:#94a3b8;margin:2px 0 12px;font-size:0.85em">'
            f'{agg["successful_scans"]}/{agg["total_scans"]} scans successful &middot; '
            f'Mean confidence: {round(agg["confidence_stats"].get("mean",0)*100,1)}% &middot; '
            f'Avg Quality: {agg.get("avg_quality","N/A")}/100 &middot; '
            f'Crops: {agg["crop_applied_count"]}/{agg["successful_scans"]}'
            f'</p>'
            f'<div style="overflow-x:auto">'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.85em">'
            f'<thead><tr style="color:#64748b;border-bottom:1px solid #334155">'
            f'<th style="padding:6px 8px;text-align:left">#</th>'
            f'<th style="padding:6px 8px;text-align:left">Prediction</th>'
            f'<th style="padding:6px 8px;text-align:left">Confidence</th>'
            f'<th style="padding:6px 8px;text-align:left">Status</th>'
            f'<th style="padding:6px 8px;text-align:left">Quality</th>'
            f'<th style="padding:6px 8px;text-align:left">NFIQ</th>'
            f'<th style="padding:6px 8px;text-align:left">Crop</th>'
            f'<th style="padding:6px 8px;text-align:center">Correct</th>'
            f'</tr></thead>'
            f'<tbody>{rows}</tbody>'
            f'</table></div>'
            f'{cons_html}'
            f'</div>'
        )

    acc_summary_html = ""
    if global_accuracy is not None:
        acc_col = "#22c55e" if global_accuracy >= 66.7 else "#ef4444"
        acc_summary_html = (
            f'<div style="padding:12px 20px;background:#0f172a;border-radius:8px;margin-bottom:20px">'
            f'<strong>Overall Scan Accuracy (subjects with known blood group):</strong>'
            f'<span style="color:{acc_col};font-size:1.3em;margin-left:16px">'
            f'{total_correct}/{total_gt_scans} = <strong>{global_accuracy}%</strong></span>'
            f'</div>'
        )

    acc_stat_card = ""
    if global_accuracy is not None:
        acc_col = "#22c55e" if global_accuracy >= 66.7 else "#ef4444"
        acc_stat_card = (
            f'<div class="stat-card">'
            f'<div class="value" style="color:{acc_col}">{global_accuracy}%</div>'
            f'<div class="label">Scan Accuracy<br>(known subjects)</div>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MFS100 Live Stress Test Report</title>
<style>
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;padding:0;font-family:'Segoe UI',system-ui,sans-serif;background:#0a0f1e;color:#e2e8f0}}
.header{{background:linear-gradient(135deg,#1e3a8a 0%,#6d28d9 100%);padding:36px 48px 28px}}
.header h1{{margin:0 0 6px;font-size:1.75em}}
.header p{{margin:3px 0;color:#bfdbfe;font-size:0.88em}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;padding:22px 48px}}
.stat-card{{background:#1e293b;border-radius:10px;padding:14px 18px;text-align:center}}
.stat-card .value{{font-size:1.9em;font-weight:700;color:#38bdf8}}
.stat-card .label{{font-size:0.75em;color:#94a3b8;margin-top:4px;line-height:1.4}}
.content{{padding:0 48px 48px}}
.constraint-box{{background:#1e293b;border-left:4px solid #22c55e;border-radius:8px;
  padding:10px 18px;margin-bottom:20px;font-size:0.82em;color:#86efac}}
.footer{{text-align:center;padding:20px;color:#475569;font-size:0.8em;border-top:1px solid #1e293b}}
</style>
</head>
<body>
<div class="header">
  <h1>&#128300; MFS100 Live Scanner Stress Test Report</h1>
  <p>Biometric Blood Group Identification System &mdash; Champion Ensemble V2 (85.41% locked accuracy)</p>
  <p>Generated: {timestamp_str} &nbsp;|&nbsp;
     Mode: {'MOCK' if mock_mode else 'REAL MFS100 HARDWARE'} &nbsp;|&nbsp;
     Subjects: {len(all_subjects)} &nbsp;|&nbsp;
     Total Scans: {total_scans}</p>
</div>

<div class="summary">
  <div class="stat-card">
    <div class="value">{len(all_subjects)}</div>
    <div class="label">Subjects Tested</div>
  </div>
  <div class="stat-card">
    <div class="value">{successful_total}/{total_scans}</div>
    <div class="label">Successful Scans</div>
  </div>
  <div class="stat-card">
    <div class="value" style="color:{'#22c55e' if consistency_rate>=66 else '#f59e0b'}">{consistency_rate}%</div>
    <div class="label">Subjects Consistent<br>(&#8805;67% same prediction)</div>
  </div>
  <div class="stat-card">
    <div class="value">{avg_conf}%</div>
    <div class="label">Avg Confidence</div>
  </div>
  <div class="stat-card">
    <div class="value">{consistent_count}/{len(all_subjects)}</div>
    <div class="label">Consistent Subjects</div>
  </div>
  {acc_stat_card}
</div>

<div class="content">
  <div class="constraint-box">
    &#10003; Model weights unchanged (ResNet50 58% + VGG16 38% + EfficientNetB0 4%) &nbsp;|&nbsp;
    &#10003; Locked test set not accessed &nbsp;|&nbsp;
    &#10003; Source-aware preprocessing applied (MFS100 crop + ridge enhancement) &nbsp;|&nbsp;
    &#10003; No retraining performed
  </div>

  {acc_summary_html}

  <h2 style="margin:0 0 16px">Per-Subject Results</h2>
  {subject_cards_html}

  <h2 style="margin:28px 0 12px">Raw JSON Data</h2>
  <details>
    <summary style="cursor:pointer;color:#94a3b8;padding:8px 0">Click to expand full JSON</summary>
    <pre style="background:#0f172a;border-radius:8px;padding:18px;overflow-x:auto;
                font-size:0.76em;color:#a5f3fc;max-height:600px;overflow-y:auto">{json.dumps(all_subjects, indent=2, default=str)}</pre>
  </details>
</div>

<div class="footer">
  Biometric Blood Group Identification System &mdash; Research Use Only. Not for clinical diagnosis.
  &nbsp;|&nbsp; Champion Ensemble V2 &nbsp;|&nbsp; 85.41% Test Accuracy on Locked 898-image Split
</div>
</body>
</html>"""

    ts_safe = timestamp_str.replace(":", "-").replace(" ", "_")
    report_path = os.path.join(OUTPUTS_DIR, f"stress_test_report_{ts_safe}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MFS100 Live Scanner Stress Test")
    parser.add_argument("--scans",        type=int, default=5,  help="Scans per subject (default: 5)")
    parser.add_argument("--subjects",     type=int, default=None, help="Number of subjects (or ask interactively)")
    parser.add_argument("--mock",         action="store_true",   help="Force mock mode")
    parser.add_argument("--no-consensus", action="store_true",   help="Skip 3-scan consensus per subject")
    args = parser.parse_args()

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner("MFS100 LIVE SCANNER STRESS TEST")
    cprint("  Champion Ensemble V2 -- ResNet50 58% + VGG16 38% + EfficientNetB0 4%", CYAN)
    cprint("  Locked accuracy: 85.41% on 898-image test set", CYAN)
    cprint(f"  Mode: {'MOCK' if args.mock else 'REAL MFS100 HARDWARE'}", YELLOW if args.mock else GREEN)
    cprint(f"  Scans per subject: {args.scans}", CYAN)
    cprint(f"  3-Scan Consensus: {'OFF' if args.no_consensus else 'ON'}", CYAN)

    if not check_services(mock_mode=args.mock):
        sys.exit(1)

    # Number of subjects
    if args.subjects is not None:
        num_subjects = args.subjects
    else:
        print()
        while True:
            try:
                num_subjects = int(input(f"{CYAN}  How many subjects to test? (e.g. 4): {RESET}").strip())
                if num_subjects > 0:
                    break
                cprint("  Enter a positive integer.", RED)
            except ValueError:
                cprint("  Invalid input.", RED)

    use_consensus = not args.no_consensus
    all_subject_results = []

    for i in range(1, num_subjects + 1):
        print()
        cprint(f"== SUBJECT {i} of {num_subjects} ==", BOLD + CYAN)
        label = input(f"  Subject name / label (e.g. Alice): ").strip()
        if not label:
            label = f"Subject_{i}"

        known_bg_raw = input(
            "  Known blood group (A+/A-/B+/B-/AB+/AB-/O+/O-) or ENTER to skip: "
        ).strip().upper()
        known_bg = known_bg_raw if known_bg_raw in VALID_BLOOD_GROUPS else None
        if not known_bg:
            cprint("  (No ground truth -- accuracy tracking disabled for this subject)", YELLOW)

        result = run_subject_session(
            subject_id=i,
            subject_label=label,
            num_scans=args.scans,
            use_consensus=use_consensus,
            known_blood_group=known_bg
        )
        all_subject_results.append(result)

        if i < num_subjects:
            input(f"\n{CYAN}  Press ENTER to continue to next subject...{RESET}")

    # Global summary
    banner("STRESS TEST COMPLETE -- FINAL SUMMARY")

    total_s  = sum(s["aggregate"]["total_scans"] for s in all_subject_results)
    succ_s   = sum(s["aggregate"]["successful_scans"] for s in all_subject_results)
    cons_s   = [s for s in all_subject_results if s["aggregate"]["is_consistent"]]
    all_conf = [c for s in all_subject_results for c in s["aggregate"].get("all_confidences", [])]
    gt_subs  = [s for s in all_subject_results if s.get("known_blood_group")]

    cprint(f"\n  Subjects tested         : {len(all_subject_results)}", CYAN)
    cprint(f"  Total scans             : {succ_s}/{total_s} successful", CYAN)
    pct_cons = round(len(cons_s) / len(all_subject_results) * 100) if all_subject_results else 0
    cprint(f"  Consistent subjects     : {len(cons_s)}/{len(all_subject_results)} ({pct_cons}%)",
           GREEN if pct_cons >= 50 else YELLOW)
    if all_conf:
        cprint(f"  Mean confidence (all)   : {statistics.mean(all_conf)*100:.1f}%", CYAN)

    if gt_subs:
        tot_corr = sum(s["aggregate"].get("correct_count") or 0 for s in gt_subs)
        tot_gt   = sum(s["aggregate"]["successful_scans"] for s in gt_subs)
        g_acc    = round(tot_corr / tot_gt * 100, 1) if tot_gt > 0 else 0
        cprint(f"  Overall scan accuracy   : {tot_corr}/{tot_gt} = {g_acc}%",
               GREEN if g_acc >= 66.7 else YELLOW)

    print()
    cprint("  Per-subject breakdown:", BOLD)
    for s in all_subject_results:
        agg  = s["aggregate"]
        dom  = agg.get("dominant_prediction", "?")
        cons = f"{agg['consistency_pct']:.0f}% consistent"
        acc  = f"  accuracy={agg['accuracy_pct']}%" if agg.get("accuracy_pct") is not None else ""
        sym  = "[+]" if agg["is_consistent"] else "[~]"
        col  = GREEN if agg["is_consistent"] else YELLOW
        cprint(f"    {sym} {s['subject_label']:<20s}  dominant={dom}  {cons}{acc}", col)

    # Save JSON
    ts_safe   = timestamp_str.replace(":", "-").replace(" ", "_")
    json_path = os.path.join(OUTPUTS_DIR, f"stress_test_{ts_safe}.json")
    payload   = {
        "timestamp":        timestamp_str,
        "mock_mode":        args.mock,
        "scans_per_subject": args.scans,
        "num_subjects":     len(all_subject_results),
        "constraints": {
            "model_weights_unchanged": True,
            "test_set_not_accessed":   True,
            "source_aware_preprocessing": True,
            "no_retraining":           True
        },
        "subjects": all_subject_results
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    cprint(f"\n  [SAVED] JSON  -> {json_path}", GREEN)

    # HTML report
    report_path = generate_html_report(all_subject_results, timestamp_str, args.mock)
    cprint(f"  [SAVED] HTML  -> {report_path}", GREEN)

    try:
        webbrowser.open(f"file:///{report_path.replace(os.sep, '/')}")
        cprint("  [INFO]  Report opened in browser.", CYAN)
    except Exception:
        pass

    cprint("\n[DONE] Stress test complete!\n", GREEN + BOLD)


if __name__ == "__main__":
    main()
