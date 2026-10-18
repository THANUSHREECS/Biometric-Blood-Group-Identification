"""
app.py
Production Flask Web Server for Biometric Blood Group Identification System.
Features:
- Centralized model loading once on application startup (thread-safe, reusable)
- Standard REST API endpoints (/api/status, /api/upload, /api/capture, /api/history, /api/report/<id>)
- Comprehensive input validation and secure file storage
- SQLite audit history integration
- PDF reports generated lazily on first download (not at prediction time)
- Mantra MFS100 hardware & mock capture integration
- Per-request timing printed via time.perf_counter (speed-up-inference branch)
"""

import os
import sys
import time
import uuid
import datetime
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    ADVISORY_CONFIRMATION_THRESHOLD,
    MFS100_MOCK,
    CAPTURED_IMAGE_DIR,
    DATABASE_PATH,
    MODEL_PATH
)
from src.predictor import BloodGroupPredictor
from src.database import (
    init_db, insert_prediction, get_all_predictions, get_prediction_by_id,
    create_research_session, insert_research_capture, update_session_inference,
    get_research_session, get_all_research_sessions, get_research_counts
)
from src.image_quality import analyze_image_quality
from src.mfs100.scanner import MFS100Scanner
from src.report_generator import generate_pdf_report, REPORTS_DIR

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
CORS(app)

# Security & Upload Configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 # 10 MB Max
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

os.makedirs(CAPTURED_IMAGE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Initialize SQLite Database
init_db(DATABASE_PATH)

# 2. Initialize Model Predictor Once at Startup
predictor = BloodGroupPredictor(model_path=MODEL_PATH, ADVISORY_CONFIRMATION_THRESHOLD=ADVISORY_CONFIRMATION_THRESHOLD)

# 3. Initialize Mantra MFS100 Scanner Adapter
scanner = MFS100Scanner(mock=MFS100_MOCK)

# Register ground truth collection blueprint
from src.ground_truth import ground_truth_bp
app.register_blueprint(ground_truth_bp, url_prefix='/ground-truth')

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================================================================
# Web UI Routes
# =========================================================================

@app.route("/")
def index():
    """Renders the main clinical dashboard."""
    return render_template("index.html")

@app.route("/history")
def history_page():
    """Renders the prediction audit history view."""
    return render_template("history.html")

# =========================================================================
# API Endpoints
# =========================================================================

@app.route("/api/config", methods=["GET"])
def get_config():
    from src.config import ADVISORY_CONFIRMATION_THRESHOLD, IMAGE_SIZE, CLASS_NAMES
    return jsonify({
        "threshold": ADVISORY_CONFIRMATION_THRESHOLD * 100,
        "image_size": IMAGE_SIZE[0],
        "class_order": CLASS_NAMES,
        "research_benchmarks": {
            "combined_methodological": {
                "accuracy": "50.50%",
                "count": "51/101",
                "explanation": "The combined 50.50% result is calculated by combining out-of-fold predictions for 80 development participants with predictions for 21 participants in the locked test set. It is a methodological combination of these two evaluation groups, not a new independent 101-participant test. The development and locked-test results are reported separately to preserve evaluation transparency."
            },
            "development_oof": {
                "accuracy": "53.75%",
                "count": "43/80",
                "explanation": "80 development participants evaluated strictly via 5-fold StratifiedGroupKFold out-of-fold cross-validation."
            },
            "locked_test": {
                "accuracy": "38.10%",
                "count": "8/21",
                "explanation": "21 strictly held-out test participants evaluated on frozen checkpoints. Matches always-O+ majority baseline (8/21, p=0.58)."
            }
        }
    })

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns system health, model readiness, scanner status, and real database statistics."""
    device_info = scanner.get_device_info()
    model_info = predictor.get_model_info()
    counts = get_research_counts()
    return jsonify({
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_info,
        "scanner": device_info,
        "research_stats": counts,
        "advisory_layer": {
            "enabled": True,
            "threshold": getattr(predictor, "advisory_threshold", 0.70),
            "status": "healthy"
        },
        "disclaimer": "RESEARCH PROTOTYPE ONLY. Fingerprint-based blood-group classification has not been clinically validated. Output must not be used for medical decisions, blood transfusions, or emergency care. Laboratory serological testing is mandatory."
    })


@app.route("/api/upload", methods=["POST"])
def upload_and_predict():
    """Handles fingerprint image upload (supports 3 scans for genuine E2 consensus or 1 scan fallback).
    PDF report is generated lazily on first /api/report/<id> download.
    """
    # 1. Collect files from either multi-slot (scan1, scan2, scan3), list ('files' / 'scans'), or single ('file')
    files_list = []
    if "scan1" in request.files or "scan2" in request.files or "scan3" in request.files:
        for key in ["scan1", "scan2", "scan3"]:
            f = request.files.get(key)
            if not f or f.filename == "":
                return jsonify({"error": f"Missing scan input for '{key}'. E2 model requires exactly 3 valid fingerprint scans."}), 400
            files_list.append(f)
    elif "files" in request.files:
        files_list = request.files.getlist("files")
    elif "scans" in request.files:
        files_list = request.files.getlist("scans")
    elif "file" in request.files:
        files_list = [request.files["file"]]
    else:
        return jsonify({"error": "No fingerprint file(s) uploaded in request. Provide 3 scans ('scan1', 'scan2', 'scan3' or 'files') or single 'file'."}), 400

    if len(files_list) not in [1, 3]:
        return jsonify({"error": f"Invalid number of scans ({len(files_list)}). E2 model requires exactly 3 fingerprint scans for full consensus inference."}), 400

    # 2. Validate all files
    from PIL import Image
    saved_paths = []
    saved_names = []

    for idx, f in enumerate(files_list):
        if not f or f.filename == "":
            return jsonify({"error": f"Scan {idx+1} is empty or missing filename."}), 400
        if not allowed_file(f.filename):
            return jsonify({"error": f"Scan {idx+1} ('{f.filename}') has invalid format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        ext = f.filename.rsplit(".", 1)[1].lower()
        unique_name = f"upload_{uuid.uuid4().hex[:12]}_s{idx+1}.{ext}"
        saved_path = os.path.join(CAPTURED_IMAGE_DIR, unique_name)
        f.save(saved_path)

        # Validate readability
        try:
            with Image.open(saved_path) as img:
                img.verify()
        except Exception as e:
            return jsonify({"error": f"Scan {idx+1} is corrupted or cannot be decoded as an image: {str(e)}"}), 400

        saved_paths.append(saved_path)
        saved_names.append(unique_name)

    try:
        t_req = time.perf_counter()

        # 3. Perform genuine inference
        if len(saved_paths) == 3:
            t_pred = time.perf_counter()
            pred_res = predictor.predict_consensus(saved_paths, source="upload")
            print(f"[TIMING][upload] predict_consensus(3 scans): {(time.perf_counter()-t_pred)*1000:.1f} ms")
            input_type_label = "Upload (3 Fingerprint Scans)"
            image_urls = [f"/api/image/{name}" for name in saved_names]
            input_source_desc = "Uploaded Images (3 Fingerprint Scans)"
        else:
            t_pred = time.perf_counter()
            pred_res = predictor.predict(saved_paths[0], source="upload")
            print(f"[TIMING][upload] predict(replicated): {(time.perf_counter()-t_pred)*1000:.1f} ms")
            input_type_label = "Upload (Single Replicated)"
            image_urls = [f"/api/image/{saved_names[0]}", f"/api/image/{saved_names[0]}", f"/api/image/{saved_names[0]}"]
            input_source_desc = "Uploaded Image (Single Replicated)"

        # 4. DB insert
        t_db = time.perf_counter()
        rec_id = insert_prediction(
            input_type=input_type_label,
            image_path=saved_names[-1],
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"]
        )
        print(f"[TIMING][upload] DB insert: {(time.perf_counter()-t_db)*1000:.1f} ms")
        print(f"[TIMING][upload] TOTAL:     {(time.perf_counter()-t_req)*1000:.1f} ms")

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{saved_names[-1]}",
            "image_urls": image_urls,
            "input_source": input_source_desc,
            "report_url": f"/api/report/{rec_id}",
            "is_low_confidence": pred_res["is_low_confidence"],
            "crop_metadata": pred_res.get("crop_metadata"),
            "needs_confirmation": pred_res.get("needs_confirmation", False),
            "advisory_threshold": pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            "advisory_level": pred_res.get("advisory_level", "CONFIDENCE_OK"),
            "advisory_reason": pred_res.get("advisory_reason", "NONE"),
            "advisory_message": pred_res.get("advisory_message", "")
        })

    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

@app.route("/api/capture", methods=["POST"])
def capture_and_predict():
    """Captures fingerprint from MFS100 (or Mock Mode) and classifies blood group.
    PDF report is generated lazily on first /api/report/<id> download.
    """
    try:
        t_req = time.perf_counter()

        # Only accept sample_class when genuinely in mock mode
        sample_class = None
        if scanner.mock and request.is_json:
            sample_class = request.json.get("sample_class")
        
        # --- Stage: Scanner capture ---
        t_cap = time.perf_counter()
        captured_path, _, quality_meta = scanner.capture_with_meta(sample_class=sample_class)
        image_name = os.path.basename(captured_path)
        print(f"[TIMING][capture] Scanner capture: {(time.perf_counter()-t_cap)*1000:.1f} ms")

        # --- Stage: Prediction ---
        source = "mfs100" if not scanner.mock else "upload"
        t_pred = time.perf_counter()
        pred_res = predictor.predict(captured_path, source=source)
        print(f"[TIMING][capture] predict():       {(time.perf_counter()-t_pred)*1000:.1f} ms")

        finger_used = "LEFT_THUMB" if not scanner.mock else None

        # --- Stage: DB insert ---
        t_db = time.perf_counter()
        rec_id = insert_prediction(
            input_type="MFS100" if not scanner.mock else "MFS100 (Mock)",
            image_path=image_name,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"],
            finger_used=finger_used
        )
        print(f"[TIMING][capture] DB insert:       {(time.perf_counter()-t_db)*1000:.1f} ms")
        print(f"[TIMING][capture] TOTAL:           {(time.perf_counter()-t_req)*1000:.1f} ms")

        input_source_desc = "Physical MFS100" if not scanner.mock else "MFS100 Mock Mode"

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{image_name}",
            "image_urls": [f"/api/image/{image_name}", f"/api/image/{image_name}", f"/api/image/{image_name}"],
            "input_source": input_source_desc,
            "report_url": f"/api/report/{rec_id}",
            "is_low_confidence": pred_res["is_low_confidence"],
            "scanner_mode": "mock" if scanner.mock else "hardware",
            "model_name": pred_res.get("model_name", "E2 Three-Scan Fusion"),
            "crop_metadata": pred_res.get("crop_metadata"),
            "quality_metadata": quality_meta,
            "finger_used": finger_used,
            "needs_confirmation": pred_res.get("needs_confirmation", False),
            "advisory_threshold": pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            "advisory_level": pred_res.get("advisory_level", "CONFIDENCE_OK"),
            "advisory_reason": pred_res.get("advisory_reason", "NONE"),
            "advisory_message": pred_res.get("advisory_message", "")
        })

    except Exception as e:
        return jsonify({"error": f"Capture error: {str(e)}"}), 500

@app.route("/api/capture/consensus", methods=["POST"])
def capture_consensus_and_predict():
    """Performs 3 consecutive impressions and computes consensus blood group.
    PDF report is generated lazily on first /api/report/<id> download.
    """
    try:
        t_req = time.perf_counter()

        sample_class = None
        if scanner.mock and request.is_json:
            sample_class = request.json.get("sample_class")

        source = "mfs100" if not scanner.mock else "upload"
        captured_paths = []
        captured_metas = []

        num_scans = 3
        for i in range(num_scans):
            # --- Stage: Scanner capture per scan ---
            t_cap = time.perf_counter()
            p, _, qm = scanner.capture_with_meta(sample_class=sample_class)
            print(f"[TIMING][consensus] Scan {i+1} capture: {(time.perf_counter()-t_cap)*1000:.1f} ms")
            captured_paths.append(p)
            captured_metas.append(qm)
            if i < num_scans - 1 and not scanner.mock:
                time.sleep(0.4)

        # --- Stage: predict_consensus (preprocessing + inference + ensembling timed internally) ---
        t_pred = time.perf_counter()
        pred_res = predictor.predict_consensus(captured_paths, source=source)
        print(f"[TIMING][consensus] predict_consensus(): {(time.perf_counter()-t_pred)*1000:.1f} ms")
        last_image_name = os.path.basename(captured_paths[-1])

        # Determine acquisition finger metadata for real hardware only
        finger_used = "LEFT_THUMB" if not scanner.mock else None

        # Input type label includes Left Thumb protocol note for real hardware
        input_type_label = (
            "MFS100 Consensus - Left Thumb (3 Impressions)" if not scanner.mock
            else "MFS100 Mock (Consensus)"
        )

        # --- Stage: DB insert ---
        t_db = time.perf_counter()
        rec_id = insert_prediction(
            input_type=input_type_label,
            image_path=last_image_name,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"],
            finger_used=finger_used
        )
        print(f"[TIMING][consensus] DB insert:           {(time.perf_counter()-t_db)*1000:.1f} ms")

        # PDF generation deferred to first download of /api/report/<id>
        avg_qual = round(sum(m.get("quality", 75) for m in captured_metas) / len(captured_metas))
        best_nfiq = min(m.get("nfiq", 2) for m in captured_metas)

        print(f"[TIMING][consensus] TOTAL:               {(time.perf_counter()-t_req)*1000:.1f} ms")

        image_urls = [f"/api/image/{os.path.basename(p)}" for p in captured_paths]
        input_source_desc = "Physical MFS100 (3 Impressions)" if not scanner.mock else "MFS100 Mock Mode (3 Impressions)"

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{last_image_name}",
            "image_urls": image_urls,
            "input_source": input_source_desc,
            "report_url": f"/api/report/{rec_id}",
            "is_low_confidence": pred_res["is_low_confidence"],
            "scanner_mode": "mock" if scanner.mock else "hardware",
            "model_name": pred_res.get("model_name", "E2 Three-Scan Fusion"),
            "crop_metadata": pred_res.get("crop_metadata"),
            "consensus_metadata": pred_res.get("consensus_metadata"),
            "finger_used": finger_used,
            "needs_confirmation": pred_res.get("needs_confirmation", False),
            "advisory_threshold": pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            "advisory_level": pred_res.get("advisory_level", "CONFIDENCE_OK"),
            "advisory_reason": pred_res.get("advisory_reason", "NONE"),
            "advisory_message": pred_res.get("advisory_message", ""),
            "quality_metadata": {
                "quality": avg_qual,
                "nfiq": best_nfiq,
                "pressure_status": captured_metas[-1].get("pressure_status", "OPTIMAL"),
                "placement_advice": f"3-Scan Left Thumb Consensus (Avg Quality: {avg_qual}/100)."
                    if finger_used else f"Consensus across 3 impressions (Avg Quality: {avg_qual}/100)."
            }
        })


    except Exception as e:
        return jsonify({"error": f"Consensus capture error: {str(e)}"}), 500

@app.route("/api/predict", methods=["POST"])
def predict_generic():
    """Generic prediction endpoint accepting either upload file or capture trigger."""
    if "file" in request.files:
        return upload_and_predict()
    else:
        return capture_and_predict()

@app.route("/api/sample-scans/<group>", methods=["GET"])
def get_sample_scans(group):
    """Returns 3 authentic sample scans as base64 for quick demonstration."""
    sample_map = {
        "A+": ("A+", "P0010"),
        "B+": ("B+", "P0011"),
        "O+": ("O+", "P001"),
        "AB+": ("AB+", "P00100")
    }
    if group not in sample_map:
        return jsonify({"error": f"Unknown blood group: {group}"}), 400

    bg_folder, pid = sample_map[group]
    folder = os.path.join(PROJECT_ROOT, "data", "mfs100_final", "raw", bg_folder, pid)
    scans = []
    import base64
    for i in range(1, 4):
        p = os.path.join(folder, f"scan_0{i}.bmp")
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                scans.append({
                    "name": f"{pid}_scan_0{i}.bmp",
                    "data": f"data:image/bmp;base64,{b64}"
                })
    return jsonify({"group": group, "pid": pid, "scans": scans})

@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns recent prediction logs from SQLite database."""
    limit = request.args.get("limit", 50, type=int)
    history = get_all_predictions(limit=limit)
    return jsonify({
        "total": len(history),
        "records": history
    })

@app.route("/api/report/<int:prediction_id>", methods=["GET"])
def download_report(prediction_id):
    """Downloads the PDF clinical report for a given prediction ID.
    PDF is generated lazily here on first download (speed-up-inference: not generated at prediction time).
    """
    rec = get_prediction_by_id(prediction_id)
    if not rec:
        return jsonify({"error": "Prediction record not found"}), 404

    # Locate pre-existing report matching record ID
    reports = [f for f in os.listdir(REPORTS_DIR) if f.startswith(f"blood_group_report_{prediction_id}_")]
    if reports:
        report_file = os.path.join(REPORTS_DIR, reports[0])
        return send_file(report_file, as_attachment=True, download_name=f"BloodGroup_Report_{prediction_id}.pdf")

    # Generate on first download request
    t_pdf = time.perf_counter()
    img_full = os.path.join(CAPTURED_IMAGE_DIR, rec["image_path"])
    pdf_path = generate_pdf_report(
        prediction_id=prediction_id,
        timestamp=rec["timestamp"],
        input_type=rec["input_type"],
        image_path=img_full,
        predicted_group=rec["predicted_group"],
        confidence=rec["confidence"],
        status=rec["status"]
    )
    print(f"[TIMING][report] PDF generation (lazy): {(time.perf_counter()-t_pdf)*1000:.1f} ms")
    return send_file(pdf_path, as_attachment=True, download_name=f"BloodGroup_Report_{prediction_id}.pdf")

@app.route("/api/image/<path:filename>", methods=["GET"])
def get_image(filename):
    """Safely serves captured fingerprint images without exposing filesystem paths."""
    safe_name = secure_filename(filename)
    return send_from_directory(CAPTURED_IMAGE_DIR, safe_name)

@app.route("/api/scanner/status", methods=["GET"])
def scanner_status():
    """Returns current scanner and bridge connectivity status."""
    info = scanner.get_device_info()
    bridge_health = scanner.check_bridge_health() if not scanner.mock else {"reachable": True, "mock": True}
    return jsonify({
        "mock_mode": scanner.mock,
        "is_connected": scanner.is_connected(),
        "bridge_health": bridge_health,
        "device": info
    })

@app.route("/api/scanner/mode", methods=["POST"])
def scanner_set_mode():
    """Toggles or sets scanner mock mode at runtime."""
    data = request.get_json(silent=True) or {}
    use_mock = data.get("mock")
    if use_mock is None:
        new_mock = not scanner.mock
    else:
        new_mock = bool(use_mock)
    
    success = scanner.set_mode(new_mock)
    return jsonify({
        "success": success,
        "mock_mode": scanner.mock,
        "is_connected": scanner.is_connected(),
        "info": scanner.get_device_info()
    })

@app.route("/api/scanner/connect", methods=["POST"])
def scanner_connect():
    """Connects to scanner."""
    res = scanner.connect()
    return jsonify({"connected": res, "info": scanner.get_device_info()})

@app.route("/api/scanner/disconnect", methods=["POST"])
def scanner_disconnect():
    """Disconnects scanner."""
    res = scanner.disconnect()
    return jsonify({"disconnected": res, "info": scanner.get_device_info()})

# =========================================================================
# Research Session & Multi-Capture Protocol API Endpoints
# =========================================================================

@app.route("/api/research/stats", methods=["GET"])
def research_stats():
    """Returns real database statistics: participant counts, sessions, and valid captures."""
    try:
        counts = get_research_counts()
        return jsonify({"success": True, "stats": counts})
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve stats: {str(e)}"}), 500

@app.route("/api/research/session", methods=["POST"])
def research_create_session():
    """Registers a participant research session with pseudonymity and explicit consent."""
    try:
        data = request.get_json(silent=True) or {}
        participant_id = (data.get("participant_id") or "").strip()
        consent_given = bool(data.get("consent_given", False))
        self_reported_group = (data.get("self_reported_group") or "").strip() or None
        label_source = data.get("label_source", "self_reported")
        verification_status = data.get("verification_status", "unverified")
        notes = (data.get("notes") or "").strip()

        if not consent_given:
            return jsonify({"error": "Research consent is mandatory under the academic research protocol."}), 400

        if not participant_id:
            # Generate pseudonymous ID: P-xxxx
            participant_id = f"P-{uuid.uuid4().hex[:6].upper()}"

        allowed_groups = {"A+", "AB+", "B+", "O+"}
        if self_reported_group and self_reported_group not in allowed_groups:
            return jsonify({"error": f"Invalid blood group '{self_reported_group}'. Must be one of: {', '.join(sorted(allowed_groups))}"}), 400

        # Safety: self-reported labels must never be automatically marked verified
        if label_source == "self_reported":
            verification_status = "unverified"

        session_id = str(uuid.uuid4())
        create_research_session(
            session_id=session_id,
            participant_id=participant_id,
            consent_given=consent_given,
            self_reported_group=self_reported_group,
            label_source=label_source,
            verification_status=verification_status,
            notes=notes
        )

        return jsonify({
            "success": True,
            "session_id": session_id,
            "participant_id": participant_id,
            "consent_given": consent_given,
            "status": "CREATED",
            "session": {
                "session_id": session_id,
                "participant_id": participant_id,
                "consent_given": consent_given,
                "self_reported_group": self_reported_group,
                "label_source": label_source,
                "verification_status": verification_status,
                "status": "CREATED",
                "notes": notes
            },
            "message": "Participant research session created successfully."
        }), 201

    except Exception as e:
        return jsonify({"error": f"Session creation failed: {str(e)}"}), 500

@app.route("/api/research/session/<session_id>", methods=["GET"])
def research_get_session(session_id):
    """Retrieves session details and all associated Left Thumb impressions."""
    try:
        sess = get_research_session(session_id)
        if not sess:
            return jsonify({"error": "Research session not found"}), 404

        # Add safe relative image URLs for UI rendering
        for cap in sess.get("captures", []):
            cap["image_url"] = f"/api/image/{cap['image_filename']}"

        return jsonify({"success": True, "session": sess})
    except Exception as e:
        return jsonify({"error": f"Failed to get session: {str(e)}"}), 500

@app.route("/api/research/sessions", methods=["GET"])
def research_list_sessions():
    """Lists recent research sessions with participant counts and status."""
    try:
        limit = request.args.get("limit", 50, type=int)
        sessions = get_all_research_sessions(limit=limit)
        return jsonify({"success": True, "total": len(sessions), "sessions": sessions})
    except Exception as e:
        return jsonify({"error": f"Failed to list sessions: {str(e)}"}), 500

@app.route("/api/research/capture", methods=["POST"])
def research_capture_impression():
    """
    Captures a single Left Thumb impression (1, 2, or 3) for the specified session.
    Performs real quality validation, records metadata, and returns quality feedback.
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")
        impression_number = data.get("impression_number")

        if not session_id:
            return jsonify({"error": "Missing required field: 'session_id'"}), 400

        if impression_number not in [1, 2, 3]:
            return jsonify({"error": "Invalid 'impression_number'. Must be 1, 2, or 3."}), 400

        sess = get_research_session(session_id)
        if not sess:
            return jsonify({"error": f"Session '{session_id}' not found."}), 404

        # Verify device connectivity
        if not scanner.mock and not scanner.is_connected():
            connected = scanner.connect()
            if not connected:
                return jsonify({
                    "error": "Mantra MFS100 scanner is not connected or bridge is unreachable.",
                    "recovery_instruction": "Ensure 32-bit bridge is running on port 5100 and USB scanner is plugged in."
                }), 503

        # Capture from scanner adapter
        captured_path, _, meta = scanner.capture_with_meta()
        image_name = os.path.basename(captured_path)

        # Real signal-processing quality validation
        quality_eval = analyze_image_quality(captured_path, meta)

        capture_id = str(uuid.uuid4())
        insert_research_capture(
            capture_id=capture_id,
            session_id=session_id,
            impression_number=impression_number,
            finger_designation="LEFT_THUMB",
            image_filename=image_name,
            width=quality_eval.get("width", 316),
            height=quality_eval.get("height", 354),
            quality_score=quality_eval.get("quality_score_100", 75),
            nfiq_score=quality_eval.get("nfiq_score", 2),
            fg_coverage=quality_eval.get("fg_coverage"),
            ridge_coherence=quality_eval.get("ridge_coherence"),
            ridge_contrast=quality_eval.get("ridge_contrast"),
            unusable_proportion=quality_eval.get("unusable_proportion"),
            is_acceptable=1 if quality_eval.get("is_acceptable", True) else 0,
            rejection_reason=quality_eval.get("rejection_reason")
        )

        return jsonify({
            "success": True,
            "capture_id": capture_id,
            "session_id": session_id,
            "impression_number": impression_number,
            "finger_designation": "LEFT_THUMB",
            "image_filename": image_name,
            "image_url": f"/api/image/{image_name}",
            "quality": quality_eval,
            "is_acceptable": quality_eval.get("is_acceptable", True),
            "rejection_reason": quality_eval.get("rejection_reason"),
            "scanner_mode": "mock" if scanner.mock else "hardware",
            "capture": {
                "capture_id": capture_id,
                "session_id": session_id,
                "impression_number": impression_number,
                "finger": "L_THUMB",
                "image_filename": image_name,
                "image_url": f"/api/image/{image_name}",
                "quality_score": quality_eval.get("quality_score_100", 75),
                "nfiq_score": quality_eval.get("nfiq_score", 2),
                "quality_analysis": quality_eval,
                "is_acceptable": quality_eval.get("is_acceptable", True),
                "rejection_reason": quality_eval.get("rejection_reason")
            }
        })

    except Exception as e:
        return jsonify({"error": f"Capture error: {str(e)}"}), 500

@app.route("/api/research/infer", methods=["POST"])
def research_infer_session():
    """
    Performs 3-scan participant-level inference using the verified EXP-06C Config A linear model.
    Enforces that exactly 3 acceptable Left Thumb impressions exist for the session.
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing required field: 'session_id'"}), 400

        sess = get_research_session(session_id)
        if not sess:
            return jsonify({"error": "Research session not found"}), 404

        captures = sess.get("captures", [])
        # Find acceptable captures for impressions 1, 2, 3 (taking the latest acceptable capture for each)
        imp_map = {}
        for c in captures:
            if c.get("is_acceptable", 0) == 1:
                imp_map[c["impression_number"]] = c

        missing = [i for i in [1, 2, 3] if i not in imp_map]
        if missing:
            return jsonify({
                "error": f"Inference requires exactly 3 valid Left Thumb impressions. Missing impressions: {missing}.",
                "recovery_instruction": "Please complete all three captures before running inference."
            }), 400

        c1, c2, c3 = imp_map[1], imp_map[2], imp_map[3]
        p1 = os.path.join(CAPTURED_IMAGE_DIR, c1["image_filename"])
        p2 = os.path.join(CAPTURED_IMAGE_DIR, c2["image_filename"])
        p3 = os.path.join(CAPTURED_IMAGE_DIR, c3["image_filename"])

        t_req_start = time.perf_counter()
        print(f"[RESEARCH][infer] Request received for session: {session_id}")

        for p, idx in [(p1, 1), (p2, 2), (p3, 3)]:
            if not os.path.exists(p):
                return jsonify({"error": f"Image file for impression {idx} is missing on disk."}), 404
        t_val_done = time.perf_counter()
        print(f"[RESEARCH][infer] Image validation passed in {(t_val_done - t_req_start)*1000:.1f} ms")

        # Run 3-scan consensus prediction using EXP-06C Config A ensemble
        t_model_start = time.perf_counter()
        pred_res = predictor.predict_consensus([p1, p2, p3], source="mfs100" if not scanner.mock else "upload")
        t_model_done = time.perf_counter()
        print(f"[RESEARCH][infer] Consensus inference completed in {(t_model_done - t_model_start)*1000:.1f} ms")

        # Save results in research_sessions table
        t_db_start = time.perf_counter()
        update_session_inference(
            session_id=session_id,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            canonical_probabilities=pred_res["probabilities"],
            model_name=pred_res["model_name"],
            model_version="1.0-exp06c"
        )
        t_db_done = time.perf_counter()
        print(f"[RESEARCH][infer] Database persistence in {(t_db_done - t_db_start)*1000:.1f} ms")
        print(f"[RESEARCH][infer] TOTAL execution time: {(time.perf_counter() - t_req_start)*1000:.1f} ms")

        return jsonify({
            "success": True,
            "session_id": session_id,
            "participant_id": sess["participant_id"],
            "created_at": sess["created_at"],
            "self_reported_group": sess.get("self_reported_group"),
            "label_source": sess.get("label_source"),
            "verification_status": sess.get("verification_status"),
            "predicted_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "probabilities": pred_res["probabilities"],
            "classes": ["A+", "AB+", "B+", "O+"],
            "model_name": pred_res["model_name"],
            "feature_dims": "3072-d Concatenation (3 x 1024-d)",
            "status": pred_res["status"],
            "message": pred_res["message"],
            "is_low_confidence": pred_res["is_low_confidence"],
            "advisory": {
                "needs_confirmation": pred_res.get("needs_confirmation", False),
                "level": pred_res.get("advisory_level", "CONFIDENCE_OK"),
                "threshold": pred_res.get("advisory_threshold", 0.70),
                "message": pred_res.get("advisory_message", "")
            },
            "captures": [
                {
                    "impression_number": 1,
                    "image_filename": c1["image_filename"],
                    "image_url": f"/api/image/{c1['image_filename']}",
                    "quality_score": c1["quality_score"],
                    "nfiq_score": c1["nfiq_score"],
                    "ridge_coherence": c1.get("ridge_coherence"),
                    "fg_coverage": c1.get("fg_coverage")
                },
                {
                    "impression_number": 2,
                    "image_filename": c2["image_filename"],
                    "image_url": f"/api/image/{c2['image_filename']}",
                    "quality_score": c2["quality_score"],
                    "nfiq_score": c2["nfiq_score"],
                    "ridge_coherence": c2.get("ridge_coherence"),
                    "fg_coverage": c2.get("fg_coverage")
                },
                {
                    "impression_number": 3,
                    "image_filename": c3["image_filename"],
                    "image_url": f"/api/image/{c3['image_filename']}",
                    "quality_score": c3["quality_score"],
                    "nfiq_score": c3["nfiq_score"],
                    "ridge_coherence": c3.get("ridge_coherence"),
                    "fg_coverage": c3.get("fg_coverage")
                }
            ],
            "research_disclaimer": "RESEARCH PROTOTYPE — NOT FOR MEDICAL USE\n\nThis experimental system has not demonstrated reliable, generalizable blood-group classification performance.\n\nIts output must not be used for medical diagnosis, emergency treatment, or transfusion decisions.\n\nBlood group must be established through validated clinical testing.",
            "disclaimer": "RESEARCH PROTOTYPE — NOT FOR MEDICAL USE\n\nThis experimental system has not demonstrated reliable, generalizable blood-group classification performance.\n\nIts output must not be used for medical diagnosis, emergency treatment, or transfusion decisions.\n\nBlood group must be established through validated clinical testing."
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

# Global error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
