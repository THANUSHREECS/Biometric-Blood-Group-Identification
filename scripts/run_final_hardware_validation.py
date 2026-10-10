"""
scripts/run_final_hardware_validation.py
Interactive CLI runner for the Final Real-Hardware Validation.
Walks the user through Sections A, B, C, D, E, F, G, and H.
"""

import os
import sys
import time
import json
import requests

BASE_URL = "http://127.0.0.1:5000"

import os
import sys
import time
import json
import sqlite3
import requests

BASE_URL = "http://127.0.0.1:5000"

def get_status():
    res = requests.get(f"{BASE_URL}/api/scanner/status")
    return res.json()

def capture_single():
    res = requests.post(f"{BASE_URL}/api/capture", timeout=25)
    return res.json(), res.status_code

def capture_consensus():
    res = requests.post(f"{BASE_URL}/api/capture/consensus", timeout=35)
    return res.json(), res.status_code

def upload_file(filepath):
    with open(filepath, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/upload", files={"file": f}, timeout=25)
    return res.json(), res.status_code

def main():
    print("================================================================")
    print("  MANTRA MFS100 FINAL REAL-HARDWARE VALIDATION RUNNER")
    print("================================================================")
    
    st = get_status()
    print("Scanner Mode:", "Physical MFS100" if not st.get("mock_mode") else "Mock Mode")
    print("Device Serial:", st.get("device", {}).get("serial"))
    print("Status:", st.get("device", {}).get("status"))
    print("----------------------------------------------------------------\n")

    results = {
        "hardware": st.get("device"),
        "single_scan_series": [],
        "consensus_series": None,
        "different_finger_series": [],
        "quality_tests": [],
        "upload_regression": [],
        "pdf_tests": [],
        "database_records": []
    }

    # SECTION A: 5 Consecutive Scans of Same Finger
    print("=== SECTION A: Single-Scan Physical MFS100 Test (5 Consecutive Scans) ===")
    print("Finger: Right Thumb (or designated subject finger)")
    for i in range(1, 6):
        input(f"\n[Press ENTER when finger is firmly on the sensor for Scan {i}/5]... ")
        print(f"Capturing Scan {i}...")
        data, code = capture_single()
        if code != 200:
            print(f"❌ Scan {i} failed: {data.get('error')}. Retrying scan...")
            input(f"[Press ENTER to retry Scan {i}]... ")
            data, code = capture_single()
        
        qm = data.get("quality_metadata", {})
        cm = data.get("crop_metadata", {})
        abo = data.get("abo_analysis", {})
        rec = {
            "capture_num": i,
            "nfiq": qm.get("nfiq"),
            "quality": qm.get("quality"),
            "crop_bbox": cm.get("bbox"),
            "crop_applied": cm.get("crop_applied"),
            "prediction": data.get("blood_group"),
            "confidence": data.get("confidence"),
            "abo_group": abo.get("abo_group"),
            "rh_factor": abo.get("rh_factor"),
            "rh_margin": abo.get("rh_margin"),
            "rh_uncertainty": abo.get("rh_uncertainty"),
            "rh_warning": abo.get("rh_warning"),
            "consensus_recommended": bool(qm.get("nfiq", 1) >= 2 or qm.get("quality", 100) < 80)
        }
        print(f" -> Scan {i}: {rec['prediction']} ({rec['confidence']*100:.1f}%) | Quality: {rec['quality']}/100, NFIQ: {rec['nfiq']}")
        print(f"    ABO: {rec['abo_group']} | Rh: {rec['rh_factor']} (Margin: {rec['rh_margin']}) | Rh Uncertainty: {rec['rh_uncertainty']}")
        results["single_scan_series"].append(rec)

    # SECTION B: 3-Scan Consensus Test
    print("\n=== SECTION B: Three-Scan Consensus Test ===")
    print("Keep finger on sensor while 3 consecutive impressions are taken...")
    input("[Press ENTER to start 3-Scan Consensus capture]... ")
    con_data, con_code = capture_consensus()
    if con_code == 200:
        c_meta = con_data.get("consensus_metadata", {})
        c_abo = con_data.get("abo_analysis", {})
        results["consensus_series"] = {
            "prediction": con_data.get("blood_group"),
            "confidence": con_data.get("confidence"),
            "num_scans": c_meta.get("num_scans"),
            "agreement_pct": c_meta.get("agreement_pct"),
            "individual_scans": c_meta.get("individual_scans"),
            "abo_group": c_abo.get("abo_group"),
            "rh_factor": c_abo.get("rh_factor"),
            "rh_uncertainty": c_abo.get("rh_uncertainty"),
            "rh_warning": c_abo.get("rh_warning")
        }
        print(f" -> Consensus Result: {results['consensus_series']['prediction']} ({results['consensus_series']['confidence']*100:.1f}%)")
        print(f"    Agreement: {results['consensus_series']['agreement_pct']}% across {results['consensus_series']['num_scans']} impressions")
        for s in c_meta.get("individual_scans", []):
            print(f"    Impression #{s['scan_index']}: {s['blood_group']} ({s['confidence']*100:.1f}%)")
    else:
        print("❌ Consensus capture failed:", con_data.get("error"))

    # SECTION C: Different-Finger Test
    print("\n=== SECTION C: Different-Finger Test (Same Person) ===")
    for finger_name in ["Right Index Finger", "Right Middle Finger"]:
        input(f"\n[Place {finger_name} on sensor and press ENTER]... ")
        f_data, f_code = capture_single()
        if f_code == 200:
            f_abo = f_data.get("abo_analysis", {})
            f_qm = f_data.get("quality_metadata", {})
            f_rec = {
                "finger": finger_name,
                "prediction": f_data.get("blood_group"),
                "confidence": f_data.get("confidence"),
                "abo_group": f_abo.get("abo_group"),
                "rh_factor": f_abo.get("rh_factor"),
                "rh_uncertainty": f_abo.get("rh_uncertainty"),
                "quality": f_qm.get("quality"),
                "nfiq": f_qm.get("nfiq")
            }
            print(f" -> {finger_name}: {f_rec['prediction']} ({f_rec['confidence']*100:.1f}%) | ABO: {f_rec['abo_group']}")
            results["different_finger_series"].append(f_rec)

    # SECTION D: Quality Guidance Test
    print("\n=== SECTION D: Placement & Quality Guidance Test ===")
    input("[Place finger OPTIMALLY flat and centered, press ENTER]... ")
    q_opt_data, _ = capture_single()
    results["quality_tests"].append({
        "condition": "Optimal Placement",
        "quality": q_opt_data.get("quality_metadata"),
        "consensus_prompted": bool(q_opt_data.get("quality_metadata", {}).get("nfiq", 1) >= 2)
    })
    
    input("[Place finger PARTIAL / LIGHT pressure, press ENTER]... ")
    q_part_data, _ = capture_single()
    results["quality_tests"].append({
        "condition": "Partial/Light Placement",
        "quality": q_part_data.get("quality_metadata"),
        "consensus_prompted": bool(q_part_data.get("quality_metadata", {}).get("nfiq", 1) >= 2 or q_part_data.get("quality_metadata", {}).get("quality", 100) < 80)
    })

    # SECTION E: PDF Test
    print("\n=== SECTION E: PDF Clinical Report Test ===")
    report_id_1 = results["single_scan_series"][0].get("capture_num") # first scan
    # Check reports directory
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    pdf_files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".pdf")], key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)))
    print(f"Found {len(pdf_files)} PDF reports generated.")
    if pdf_files:
        latest_pdf = os.path.join(reports_dir, pdf_files[-1])
        print("Latest PDF:", latest_pdf, f"({os.path.getsize(latest_pdf)} bytes)")
        results["pdf_tests"].append({
            "latest_report": pdf_files[-1],
            "size_bytes": os.path.getsize(latest_pdf),
            "exists": True
        })

    # SECTION F: Database Audit Test
    print("\n=== SECTION F: Database Audit Log Test ===")
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "blood_group.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, input_type, predicted_group, confidence, status FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    print(f"Last {len(rows)} database records in blood_group.db:")
    for r in rows:
        print(f"  ID {r[0]}: [{r[1]}] {r[2]} -> {r[3]} ({r[4]*100:.1f}%) [{r[5]}]")
        results["database_records"].append({
            "id": r[0], "timestamp": r[1], "input_type": r[2], "predicted_group": r[3], "confidence": r[4], "status": r[5]
        })
    conn.close()

    # SECTION G: Upload Regression Test
    print("\n=== SECTION G: Upload Regression Test (Validation / Non-Test Data) ===")
    val_sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "B+.bmp")
    if os.path.exists(val_sample):
        up_data, up_code = upload_file(val_sample)
        if up_code == 200:
            up_cm = up_data.get("crop_metadata", {})
            up_abo = up_data.get("abo_analysis", {})
            up_rec = {
                "source": "B+.bmp (Sample)",
                "prediction": up_data.get("blood_group"),
                "confidence": up_data.get("confidence"),
                "crop_applied": up_cm.get("crop_applied", False),
                "abo_group": up_abo.get("abo_group"),
                "rh_factor": up_abo.get("rh_factor"),
                "rh_uncertainty": up_abo.get("rh_uncertainty")
            }
            print(f" -> Upload B+.bmp: {up_rec['prediction']} ({up_rec['confidence']*100:.1f}%) | Crop Applied: {up_rec['crop_applied']} (Should be False)")
            results["upload_regression"].append(up_rec)

    # Save JSON results
    out_json = os.path.join(os.path.dirname(__file__), "..", "outputs", "final_hardware_validation_raw.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw validation results to: {out_json}")
    print("================================================================")
    print("  HARDWARE VALIDATION RUN COMPLETE")
    print("================================================================")

if __name__ == "__main__":
    main()

