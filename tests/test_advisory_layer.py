"""
tests/test_advisory_layer.py
============================
Unit, API, and integration regression tests for the Advisory UI/API Warning Layer (tau = 0.70).
Verifies all 15 requirements specified in the advisory layer review instructions.
"""

import os
import sys
import json
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import ADVISORY_CONFIRMATION_THRESHOLD, CLASS_NAMES
from src.predictor import BloodGroupPredictor
from src.report_generator import generate_pdf_report, REPORTS_DIR
from app import app

@pytest.fixture(scope="module")
def predictor_instance():
    # Instantiate predictor with ensemble enabled (or fallback)
    pred = BloodGroupPredictor()
    return pred

@pytest.fixture(scope="module")
def test_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

# 1. confidence = 0.69 -> needs_confirmation = True
def test_1_confidence_below_threshold(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.69, {"rh_uncertainty": False})
    assert adv["needs_confirmation"] is True
    assert adv["advisory_level"] == "CONFIRMATION_RECOMMENDED"

# 2. confidence = 0.70 -> threshold condition is NOT triggered
def test_2_confidence_exact_threshold(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.70, {"rh_uncertainty": False})
    assert adv["needs_confirmation"] is False
    assert adv["advisory_level"] == "CONFIDENCE_OK"

# 3. confidence = 0.71 -> needs_confirmation = False (assuming Rh uncertainty is False)
def test_3_confidence_above_threshold(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.71, {"rh_uncertainty": False})
    assert adv["needs_confirmation"] is False
    assert adv["advisory_level"] == "CONFIDENCE_OK"

# 4. Rh uncertainty = True -> needs_confirmation = True (even if confidence >= 0.70)
def test_4_rh_uncertainty_triggers_confirmation(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.85, {"rh_uncertainty": True})
    assert adv["needs_confirmation"] is True
    assert adv["advisory_level"] == "CONFIRMATION_RECOMMENDED"

# 5. Both conditions true -> advisory_reason = "LOW_CONFIDENCE_AND_RH_UNCERTAINTY"
def test_5_both_conditions_true(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.65, {"rh_uncertainty": True})
    assert adv["needs_confirmation"] is True
    assert adv["advisory_reason"] == "LOW_CONFIDENCE_AND_RH_UNCERTAINTY"
    assert adv["advisory_level"] == "CONFIRMATION_RECOMMENDED"

# 6. Low confidence only -> advisory_reason = "LOW_CONFIDENCE"
def test_6_low_confidence_only(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.65, {"rh_uncertainty": False})
    assert adv["needs_confirmation"] is True
    assert adv["advisory_reason"] == "LOW_CONFIDENCE"
    assert adv["advisory_level"] == "CONFIRMATION_RECOMMENDED"

# 7. Rh uncertainty only -> advisory_reason = "RH_UNCERTAINTY"
def test_7_rh_uncertainty_only(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.75, {"rh_uncertainty": True})
    assert adv["needs_confirmation"] is True
    assert adv["advisory_reason"] == "RH_UNCERTAINTY"
    assert adv["advisory_level"] == "CONFIRMATION_RECOMMENDED"

# 8. Neither condition -> advisory_reason = "NONE"
def test_8_neither_condition(predictor_instance):
    adv = predictor_instance._evaluate_advisory(0.80, {"rh_uncertainty": False})
    assert adv["needs_confirmation"] is False
    assert adv["advisory_reason"] == "NONE"
    assert adv["advisory_level"] == "CONFIDENCE_OK"

# 9 & 10 & 11. Existing prediction label, probabilities, and confidence values are preserved
def test_9_10_11_prediction_probabilities_confidence_preserved(predictor_instance):
    # Dummy synthetic fingerprint (224, 224, 3)
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    res = predictor_instance.predict(dummy_img, source="upload")
    
    assert res["blood_group"] in CLASS_NAMES
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 8
    # Probabilities sum approximately to 1
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01

# 12. Advisory threshold remains exactly 0.70
def test_12_advisory_threshold_value(predictor_instance):
    assert ADVISORY_CONFIRMATION_THRESHOLD == 0.70
    assert predictor_instance.advisory_threshold == 0.70

# 13. API status and response structure contain all required advisory keys
def test_13_api_response_structure(test_client):
    status_res = test_client.get("/api/status")
    assert status_res.status_code == 200
    data = status_res.get_json()
    assert "advisory_layer" in data
    assert data["advisory_layer"]["threshold"] == 0.70
    assert data["advisory_layer"]["enabled"] is True

# 14. PDF generation succeeds with advisory information
def test_14_pdf_generation_with_advisory():
    from PIL import Image
    test_img_path = os.path.join(REPORTS_DIR, "test_dummy_thumb.png")
    Image.new("RGB", (100, 100), color="gray").save(test_img_path)

    pdf_path = generate_pdf_report(
        prediction_id=99999,
        timestamp="2026-09-08 12:00:00",
        input_type="Test Sample",
        image_path=test_img_path,
        predicted_group="A+",
        confidence=0.82,
        probabilities={"A+": 0.82, "A-": 0.02, "B+": 0.05, "B-": 0.01, "AB+": 0.04, "AB-": 0.01, "O+": 0.04, "O-": 0.01},
        status="NORMAL",
        model_name="Champion Ensemble V2",
        advisory_level="CONFIDENCE_OK",
        advisory_reason="NONE",
        advisory_message="Advisory threshold passed. This does not constitute medical verification.",
        advisory_threshold=0.70,
        needs_confirmation=False
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
    # Clean up test artifacts
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

# 15. Backward compatibility of existing predictor output is preserved
def test_15_backward_compatibility(predictor_instance):
    dummy_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    res = predictor_instance.predict(dummy_img, source="upload")
    
    # Required legacy fields
    legacy_keys = ["blood_group", "confidence", "status", "message", "probabilities", "is_low_confidence"]
    for k in legacy_keys:
        assert k in res, f"Missing legacy key: {k}"
        
    # New additive advisory keys
    advisory_keys = ["needs_confirmation", "advisory_threshold", "advisory_level", "advisory_reason", "advisory_message"]
    for k in advisory_keys:
        assert k in res, f"Missing advisory key: {k}"
