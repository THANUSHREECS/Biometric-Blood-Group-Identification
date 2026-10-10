import os
import sys
import uuid
import time
import json
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import tensorflow as tf

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.mfs100.scanner import MFS100Scanner
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

app = FastAPI(title="Biometric Blood Group Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "day27_full_dataset", "four_class", "best_model.keras")
FOUR_CLASSES = ["A+", "AB+", "B+", "O+"]
CLASS_MAPPING = {i: c for i, c in enumerate(FOUR_CLASSES)}
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "live_mfs100_tests")

model = None
scanner = MFS100Scanner(mock=False)
sessions = {}

def transform_c5(image_input, target_size=(112, 112)):
    pil_img = load_image_to_pil(image_input)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=target_size)
    return np.array(resized, dtype=np.float32)

@app.on_event("startup")
def load_resources():
    global model
    print("Loading Day 27 model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    os.makedirs(REPORTS_DIR, exist_ok=True)

@app.get("/api/mfs100/status")
def get_scanner_status():
    try:
        health = scanner.check_bridge_health()
        if not health.get("reachable"):
            return {
                "scanner_detected": False,
                "error": "Cannot connect to MFS100: Bridge offline at http://127.0.0.1:5100"
            }
        
        # Bridge is reachable, check device
        connected = scanner.is_connected()
        if not connected:
            connected = scanner.connect()
            
        if connected:
            return {
                "scanner_detected": True,
                "sdk_available": True,
                "bridge_available": True,
                "capture_available": True
            }
        else:
            return {
                "scanner_detected": False,
                "error": "Bridge is reachable but physical scanner could not be connected."
            }
    except Exception as e:
        return {
            "scanner_detected": False,
            "error": str(e)
        }

class StartSessionResponse(BaseModel):
    session_id: str
    status: str

@app.post("/api/mfs100/start-session", response_model=StartSessionResponse)
def start_session():
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(REPORTS_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    sessions[session_id] = {
        "id": session_id,
        "scans": [],
        "dir": session_dir,
        "start_time": datetime.now().isoformat()
    }
    return StartSessionResponse(session_id=session_id, status="ready")

class CaptureRequest(BaseModel):
    session_id: str
    scan_number: int

@app.post("/api/mfs100/capture")
def capture_fingerprint(req: CaptureRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    sess = sessions[req.session_id]
    if len(sess["scans"]) >= 3:
        raise HTTPException(status_code=400, detail="Maximum 3 scans allowed")
        
    scan_path = os.path.join(sess["dir"], f"scan_{req.scan_number}.bmp")
    
    try:
        # We enforce physical connection or attempt to capture
        if not scanner.is_connected():
            connected = scanner.connect()
            if not connected:
                raise Exception("Mantra MFS100 scanner not detected.")
                
        # capture hardware image directly
        saved_path, img_bytes = scanner._capture_hardware(scan_path)
        return {"success": True, "scan_number": req.scan_number, "image_path": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProcessRequest(BaseModel):
    session_id: str
    scan_number: int
    image_path: str

@app.post("/api/mfs100/process")
def process_fingerprint(req: ProcessRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    sess = sessions[req.session_id]
    
    try:
        if req.image_path.startswith("--image "):
            # For development mode
            img_path = req.image_path.split("--image ")[1].strip()
        else:
            img_path = req.image_path
            
        input_array = transform_c5(img_path, target_size=(112, 112))
        input_tensor = np.expand_dims(input_array, axis=0)
        
        preds = model.predict(input_tensor, verbose=0)[0]
        
        pred_idx = np.argmax(preds)
        pred_class = CLASS_MAPPING[pred_idx]
        confidence = float(preds[pred_idx])
        
        probs = {CLASS_MAPPING[i]: float(preds[i]) for i in range(len(FOUR_CLASSES))}
        
        scan_data = {
            "timestamp": datetime.now().isoformat(),
            "capture_number": req.scan_number,
            "raw_image_path": img_path,
            "prediction": pred_class,
            "confidence": confidence,
            "prob_A_plus": probs["A+"],
            "prob_B_plus": probs["B+"],
            "prob_AB_plus": probs["AB+"],
            "prob_O_plus": probs["O+"]
        }
        
        sess["scans"].append(scan_data)
        
        return {
            "success": True,
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FinalizeRequest(BaseModel):
    session_id: str

@app.post("/api/mfs100/finalize")
def finalize_session(req: FinalizeRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    sess = sessions[req.session_id]
    if len(sess["scans"]) != 3:
        raise HTTPException(status_code=400, detail="Need exactly 3 scans to finalize")
        
    # Calculate average probabilities
    avg_probs = {c: 0.0 for c in FOUR_CLASSES}
    for scan in sess["scans"]:
        avg_probs["A+"] += scan["prob_A_plus"]
        avg_probs["B+"] += scan["prob_B_plus"]
        avg_probs["AB+"] += scan["prob_AB_plus"]
        avg_probs["O+"] += scan["prob_O_plus"]
        
    for c in FOUR_CLASSES:
        avg_probs[c] /= 3.0
        
    final_pred = max(avg_probs, key=avg_probs.get)
    final_conf = avg_probs[final_pred]
    
    # Calculate consistency
    preds = [scan["prediction"] for scan in sess["scans"]]
    agree_count = preds.count(final_pred)
    
    if agree_count == 3:
        consistency = "HIGH"
    elif agree_count == 2:
        consistency = "MODERATE"
    else:
        consistency = "LOW"
        
    sess["combined_prob_A_plus"] = avg_probs["A+"]
    sess["combined_prob_B_plus"] = avg_probs["B+"]
    sess["combined_prob_AB_plus"] = avg_probs["AB+"]
    sess["combined_prob_O_plus"] = avg_probs["O+"]
    sess["final_prediction"] = final_pred
    sess["final_confidence"] = final_conf
    sess["agreement_count"] = agree_count
    sess["consistency_level"] = consistency
    
    # Save to json
    json_path = os.path.join(sess["dir"], "prediction.json")
    with open(json_path, 'w') as f:
        json.dump(sess, f, indent=4)
        
    # Update global CSV
    csv_path = os.path.join(REPORTS_DIR, "live_sessions.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["session_id", "timestamp", "final_prediction", "final_confidence", "consistency", "agreement_count"])
        writer.writerow([req.session_id, sess["start_time"], final_pred, final_conf, consistency, agree_count])
        
    return {
        "success": True,
        "final_prediction": final_pred,
        "final_confidence": final_conf,
        "combined_probabilities": avg_probs,
        "consistency": {
            "level": consistency,
            "agreement_count": agree_count
        },
        "scans": sess["scans"]
    }

@app.post("/api/mfs100/reset")
def reset_session(req: FinalizeRequest):
    if req.session_id in sessions:
        del sessions[req.session_id]
    return {"success": True}

# ==================================================
# FINAL MFS100 RESEARCH DATA COLLECTION MODULE
# ==================================================
import shutil

class ResearchStartRequest(BaseModel):
    participant_id: str
    consent: bool

@app.post("/api/mfs100/research/start-session")
def research_start_session(req: ResearchStartRequest):
    if not req.consent:
        raise HTTPException(status_code=400, detail="Consent is required for research collection.")
    
    # Check if participant already exists in metadata
    csv_path = os.path.join(PROJECT_ROOT, "data", "mfs100_final", "metadata", "participants.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['participant_id'] == req.participant_id:
                    raise HTTPException(status_code=400, detail=f"Participant {req.participant_id} already exists.")
                    
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(PROJECT_ROOT, "scratch", f"research_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    
    sessions[session_id] = {
        "id": session_id,
        "participant_id": req.participant_id,
        "scans": [],
        "dir": session_dir,
        "start_time": datetime.now().isoformat(),
        "is_research": True
    }
    return {"session_id": session_id, "status": "ready"}

class ResearchSaveMetadataRequest(BaseModel):
    session_id: str
    blood_group: str
    verification_source: str

@app.post("/api/mfs100/research/save-metadata")
def research_save_metadata(req: ResearchSaveMetadataRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    sess = sessions[req.session_id]
    if len(sess["scans"]) != 3:
        raise HTTPException(status_code=400, detail="Must have 3 scans to save metadata")
        
    p_id = sess["participant_id"]
    bg = req.blood_group
    
    # Copy files to final structure
    final_raw_dir = os.path.join(PROJECT_ROOT, "data", "mfs100_final", "raw", bg, p_id)
    os.makedirs(final_raw_dir, exist_ok=True)
    
    for i, scan in enumerate(sess["scans"]):
        scan_num = i + 1
        src_path = scan["raw_image_path"]
        dst_path = os.path.join(final_raw_dir, f"scan_{scan_num:02d}.bmp")
        shutil.copy(src_path, dst_path)
        
        # Log to collection_log.csv
        log_csv = os.path.join(PROJECT_ROOT, "data", "mfs100_final", "metadata", "collection_log.csv")
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                scan["timestamp"],
                p_id,
                scan_num,
                dst_path,
                scan["prediction"],
                scan["confidence"],
                scan["prob_A_plus"],
                scan["prob_B_plus"],
                scan["prob_AB_plus"],
                scan["prob_O_plus"]
            ])
            
    # Log to participants.csv
    part_csv = os.path.join(PROJECT_ROOT, "data", "mfs100_final", "metadata", "participants.csv")
    with open(part_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            p_id,
            bg,
            req.verification_source,
            sess["start_time"],
            "LEFT THUMB",
            "Yes",
            3
        ])
        
    return {"success": True}
