"""
app.py
Production Flask Web Server for Biometric Blood Group Identification System.
Features:
- Centralized model loading once on application startup (thread-safe, reusable)
- Standard REST API endpoints (/api/status, /api/upload, /api/capture, /api/history, /api/report/<id>)
- Comprehensive input validation and secure file storage
- SQLite audit history integration
- Automated PDF report generation (ReportLab)
- Mantra MFS100 hardware & mock capture integration
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
    IMAGE_SIZE,
    ADVISORY_CONFIRMATION_THRESHOLD,
    MFS100_MOCK,
    CAPTURED_IMAGE_DIR,
    DATABASE_PATH,
    MODEL_PATH
)
from src.predictor import BloodGroupPredictor
from src.database import init_db, insert_prediction, get_all_predictions, get_prediction_by_id
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
        "class_order": CLASS_NAMES
    })

@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns system health, model readiness, and scanner status."""
    device_info = scanner.get_device_info()
    model_info = predictor.get_model_info()
    return jsonify({
        "status": "online",
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_info,
        "scanner": device_info,
        "advisory_layer": {
            "enabled": True,
            "threshold": getattr(predictor, "advisory_threshold", 0.70),
            "status": "healthy"
        },
        "disclaimer": "Research/prototype fingerprint-based system only. Confirm using standard laboratory blood typing."
    })


@app.route("/api/upload", methods=["POST"])
def upload_and_predict():
    """Handles fingerprint image upload, prediction, logging, and report generation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Generate unique secure filename
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"upload_{uuid.uuid4().hex[:12]}.{ext}"
        saved_path = os.path.join(CAPTURED_IMAGE_DIR, unique_name)
        file.save(saved_path)

        # Run Prediction (source="upload" keeps standard dataset preprocessing without crop)
        pred_res = predictor.predict(saved_path, source="upload")

        # Log into SQLite
        rec_id = insert_prediction(
            input_type="upload",
            image_path=unique_name,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"]
        )

        # Generate Clinical PDF Report
        pdf_path = generate_pdf_report(
            prediction_id=rec_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_type="Upload",
            image_path=saved_path,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            probabilities=pred_res["probabilities"],
            status=pred_res["status"],
            model_name=pred_res.get("model_name"),
            advisory_level=pred_res.get("advisory_level"),
            advisory_reason=pred_res.get("advisory_reason"),
            advisory_message=pred_res.get("advisory_message"),
            advisory_threshold=pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            needs_confirmation=pred_res.get("needs_confirmation", False),
            crop_metadata=pred_res.get("crop_metadata")
        )

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{unique_name}",
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
    """Captures fingerprint from MFS100 (or Mock Mode) and classifies blood group."""
    try:
        # Only accept sample_class when genuinely in mock mode
        sample_class = None
        if scanner.mock and request.is_json:
            sample_class = request.json.get("sample_class")
        
        # Capture from scanner or mock engine with quality metadata
        captured_path, _, quality_meta = scanner.capture_with_meta(sample_class=sample_class)
        image_name = os.path.basename(captured_path)

        # Run Prediction via E2 Champion Ensemble with source-aware cropping and ridge enhancement for real hardware
        source = "mfs100" if not scanner.mock else "upload"
        pred_res = predictor.predict(captured_path, source=source)

        # Determine acquisition finger metadata for real hardware only
        # (Left Thumb is the standardized acquisition protocol for physical demonstrations.)
        # The scanner does NOT verify finger identity; this is a convention-based metadata field.
        finger_used = "LEFT_THUMB" if not scanner.mock else None

        # Log to Database
        rec_id = insert_prediction(
            input_type="MFS100" if not scanner.mock else "MFS100 (Mock)",
            image_path=image_name,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"],
            finger_used=finger_used
        )

        # Generate Report
        # For real hardware, the input_type label includes the Left Thumb protocol note.
        pdf_input_type_single = (
            "Mantra MFS100 Scanner | Standard Finger: LEFT THUMB" if not scanner.mock
            else "MFS100 (Mock Mode)"
        )
        generate_pdf_report(
            prediction_id=rec_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_type=pdf_input_type_single,
            image_path=captured_path,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            probabilities=pred_res["probabilities"],
            status=pred_res["status"],
            finger_used=finger_used,
            model_name=pred_res.get("model_name"),
            advisory_level=pred_res.get("advisory_level"),
            advisory_reason=pred_res.get("advisory_reason"),
            advisory_message=pred_res.get("advisory_message"),
            advisory_threshold=pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            needs_confirmation=pred_res.get("needs_confirmation", False),
            crop_metadata=pred_res.get("crop_metadata")
        )

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{image_name}",
            "report_url": f"/api/report/{rec_id}",
            "is_low_confidence": pred_res["is_low_confidence"],
            "scanner_mode": "mock" if scanner.mock else "hardware",
            "model_name": pred_res.get("model_name", "E2 Champion Ensemble"),
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
    """Performs 3 consecutive impressions and computes consensus blood group."""
    try:
        sample_class = None
        if scanner.mock and request.is_json:
            sample_class = request.json.get("sample_class")

        source = "mfs100" if not scanner.mock else "upload"
        captured_paths = []
        captured_metas = []

        num_scans = 3
        for i in range(num_scans):
            p, _, qm = scanner.capture_with_meta(sample_class=sample_class)
            captured_paths.append(p)
            captured_metas.append(qm)
            if i < num_scans - 1 and not scanner.mock:
                time.sleep(0.4)

        pred_res = predictor.predict_consensus(captured_paths, source=source)
        last_image_name = os.path.basename(captured_paths[-1])

        # Determine acquisition finger metadata for real hardware only
        finger_used = "LEFT_THUMB" if not scanner.mock else None

        # Input type label includes Left Thumb protocol note for real hardware
        input_type_label = (
            "MFS100 Consensus - Left Thumb (3 Impressions)" if not scanner.mock
            else "MFS100 Mock (Consensus)"
        )
        pdf_input_type = (
            "MFS100 Consensus (3 Impressions) | Standard Finger: LEFT THUMB" if not scanner.mock
            else "MFS100 Mock Consensus"
        )

        rec_id = insert_prediction(
            input_type=input_type_label,
            image_path=last_image_name,
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            status=pred_res["status"],
            notes=pred_res["message"],
            finger_used=finger_used
        )

        generate_pdf_report(
            prediction_id=rec_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_type=pdf_input_type,
            image_path=captured_paths[-1],
            predicted_group=pred_res["blood_group"],
            confidence=pred_res["confidence"],
            probabilities=pred_res["probabilities"],
            status=pred_res["status"],
            finger_used=finger_used,
            model_name=pred_res.get("model_name"),
            advisory_level=pred_res.get("advisory_level"),
            advisory_reason=pred_res.get("advisory_reason"),
            advisory_message=pred_res.get("advisory_message"),
            advisory_threshold=pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD),
            needs_confirmation=pred_res.get("needs_confirmation", False),
            crop_metadata=pred_res.get("crop_metadata")
        )

        avg_qual = round(sum(m.get("quality", 75) for m in captured_metas) / len(captured_metas))
        best_nfiq = min(m.get("nfiq", 2) for m in captured_metas)

        return jsonify({
            "success": True,
            "prediction_id": rec_id,
            "blood_group": pred_res["blood_group"],
            "confidence": pred_res["confidence"],
            "status": pred_res["status"],
            "message": pred_res["message"],
            "probabilities": pred_res["probabilities"],
            "image_url": f"/api/image/{last_image_name}",
            "report_url": f"/api/report/{rec_id}",
            "is_low_confidence": pred_res["is_low_confidence"],
            "scanner_mode": "mock" if scanner.mock else "hardware",
            "model_name": pred_res.get("model_name", "E2 Champion Ensemble Consensus"),
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
    """Downloads the generated PDF clinical report for a given prediction ID."""
    rec = get_prediction_by_id(prediction_id)
    if not rec:
        return jsonify({"error": "Prediction record not found"}), 404

    # Locate report matching record ID
    reports = [f for f in os.listdir(REPORTS_DIR) if f.startswith(f"blood_group_report_{prediction_id}_")]
    if reports:
        report_file = os.path.join(REPORTS_DIR, reports[0])
        return send_file(report_file, as_attachment=True, download_name=f"BloodGroup_Report_{prediction_id}.pdf")

    # If not already on disk, regenerate on the fly
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

# Global error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
