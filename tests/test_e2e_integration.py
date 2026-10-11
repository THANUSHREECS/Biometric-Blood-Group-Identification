import os
import unittest
import io
import json
from app import app
from src.config import SAMPLE_IMAGE_DIR

class TestEndToEndWorkflow(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_upload_and_download_report_workflow(self):
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        self.assertTrue(os.path.exists(sample_path))

        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        # 1. Test POST /api/upload
        data = {
            "file": (io.BytesIO(file_bytes), "sample_test_A+.bmp")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()

        self.assertTrue(res_json["success"])
        self.assertIn("prediction_id", res_json)
        self.assertIn(res_json["blood_group"], ["A+", "B+", "AB+", "O+"])
        self.assertIn("probabilities", res_json)
        self.assertEqual(len(res_json["probabilities"]), 4)

        pred_id = res_json["prediction_id"]

        # 2. Test GET /api/report/<id>
        rep_res = self.client.get(f"/api/report/{pred_id}")
        self.assertEqual(rep_res.status_code, 200)
        self.assertEqual(rep_res.mimetype, "application/pdf")
        self.assertGreater(len(rep_res.data), 5000)

    def test_mfs100_capture_and_history_workflow(self):
        # 1. Test POST /api/capture
        cap_res = self.client.post("/api/capture", json={"sample_class": "O+"})
        self.assertEqual(cap_res.status_code, 200)
        cap_json = cap_res.get_json()

        self.assertTrue(cap_json["success"])
        self.assertIn("prediction_id", cap_json)
        self.assertEqual(cap_json["scanner_mode"], "mock")

        # 2. Test GET /api/history
        hist_res = self.client.get("/api/history")
        self.assertEqual(hist_res.status_code, 200)
        hist_json = hist_res.get_json()
        self.assertGreater(hist_json["total"], 0)

if __name__ == "__main__":
    unittest.main()
