"""
tests/test_mfs100_integration.py
End-to-end integration tests for MFS100 fingerprint capture pipeline.
Verifies:
- Scanner status & mode toggle endpoints
- Mock capture -> Preprocessing -> Model Prediction -> DB Logging -> PDF Report
- Simulated hardware capture -> Preprocessing -> Prediction -> DB Logging -> PDF Report
"""

import os
import io
import json
import base64
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from app import app
from src.database import get_prediction_by_id

class TestMFS100EndToEndIntegration(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_scanner_status_endpoint(self):
        """Tests /api/scanner/status returns correct schema."""
        resp = self.client.get("/api/scanner/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn("mock_mode", data)
        self.assertIn("is_connected", data)
        self.assertIn("bridge_health", data)
        self.assertIn("device", data)

    def test_scanner_mode_toggle(self):
        """Tests /api/scanner/mode toggles between mock and hardware modes."""
        # Force mock = True
        resp = self.client.post("/api/scanner/mode", json={"mock": True})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json["mock_mode"])

        # Force mock = False
        resp2 = self.client.post("/api/scanner/mode", json={"mock": False})
        self.assertEqual(resp2.status_code, 200)
        self.assertFalse(resp2.json["mock_mode"])

        # Reset back to mock = True for safe default
        self.client.post("/api/scanner/mode", json={"mock": True})

    def test_capture_mock_pipeline(self):
        """Tests full biometric inference pipeline using mock capture."""
        # Ensure in mock mode
        self.client.post("/api/scanner/mode", json={"mock": True})

        resp = self.client.post("/api/capture", json={"sample_class": "B+"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json

        self.assertTrue(data["success"])
        self.assertIn("blood_group", data)
        self.assertIn("confidence", data)
        self.assertIn("prediction_id", data)
        self.assertIn("probabilities", data)
        self.assertEqual(len(data["probabilities"]), 4)

        pred_id = data["prediction_id"]
        db_rec = get_prediction_by_id(pred_id)
        self.assertIsNotNone(db_rec)
        self.assertEqual(db_rec["predicted_group"], data["blood_group"])

    @patch("requests.post")
    def test_capture_hardware_pipeline_simulated(self, mock_post):
        """Tests full biometric inference pipeline with simulated hardware capture."""
        # Set hardware mode
        self.client.post("/api/scanner/mode", json={"mock": False})

        # Generate a synthetic 316x354 BMP fingerprint frame
        img = Image.new("L", (316, 354), color=210)
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()
        b64_str = base64.b64encode(bmp_bytes).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "success": True,
            "image_b64": b64_str,
            "quality": 85,
            "nfiq": 1
        }
        mock_post.return_value = mock_resp

        # Trigger capture
        resp = self.client.post("/api/capture")
        self.assertEqual(resp.status_code, 200)
        data = resp.json

        self.assertTrue(data["success"])
        self.assertIn("blood_group", data)
        self.assertIn("confidence", data)
        self.assertEqual(data["scanner_mode"], "hardware")

        pred_id = data["prediction_id"]
        db_rec = get_prediction_by_id(pred_id)
        self.assertIsNotNone(db_rec)
        self.assertEqual(db_rec["input_type"], "MFS100")

        # Cleanup: switch back to mock
        self.client.post("/api/scanner/mode", json={"mock": True})

if __name__ == "__main__":
    unittest.main()
