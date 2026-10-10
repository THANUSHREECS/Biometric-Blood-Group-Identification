import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD
from src.predictor import BloodGroupPredictor

predictor = BloodGroupPredictor(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)

files = [
    'data/captured/capture_1789115056057.bmp',
    'data/captured/capture_1789115057360.bmp',
    'data/captured/capture_1789115058674.bmp'
]

print("=== INDIVIDUAL PREDICTIONS ===")
ind_results = []
for i, f in enumerate(files):
    res = predictor.predict(f, source="mfs100")
    ind_results.append(res)
    print(f"Capture {i+1}: {os.path.basename(f)}")
    print(f"  Predicted 8-Class: {res['blood_group']}")
    print(f"  Confidence: {res['confidence']*100:.2f}%")
    print("  Probabilities:")
    for k in sorted(res['probabilities'].keys()):
        print(f"    {k}: {res['probabilities'][k]*100:.2f}%")
    abo = res.get("abo_analysis", {})
    print(f"  Derived ABO: {abo.get('abo_group')} ({abo.get('abo_confidence', 0)*100:.2f}%)")
    print(f"  Derived Rh: {abo.get('rh_factor')} ({abo.get('rh_confidence', 0)*100:.2f}%)")
    print(f"  Advisory Status: {res.get('advisory_level')}")
    print(f"  Advisory Reason: {res.get('advisory_reason')}")
    print()

print("=== CONSENSUS PREDICTION ===")
c_res = predictor.predict_consensus(files, source="mfs100")
print(f"Consensus Blood Group: {c_res['blood_group']}")
print(f"Consensus Confidence: {c_res['confidence']*100:.2f}%")
print("Consensus Probabilities:")
for k in sorted(c_res['probabilities'].keys()):
    print(f"  {k}: {c_res['probabilities'][k]*100:.2f}%")
c_abo = c_res.get("abo_analysis", {})
print(f"Consensus Derived ABO: {c_abo.get('abo_group')} ({c_abo.get('abo_confidence', 0)*100:.2f}%)")
print(f"Consensus Derived Rh: {c_abo.get('rh_factor')} ({c_abo.get('rh_confidence', 0)*100:.2f}%)")
print(f"Consensus Advisory Status: {c_res.get('advisory_level')}")
print(f"Consensus Advisory Reason: {c_res.get('advisory_reason')}")
print(f"Agreement: {c_res['consensus_metadata']['agreement_count']}/{c_res['consensus_metadata']['num_scans']}")
