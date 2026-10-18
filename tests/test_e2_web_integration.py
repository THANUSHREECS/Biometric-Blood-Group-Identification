import os
import io
import json
import unittest
import numpy as np
from app import app
from src.config import PROJECT_ROOT, SAMPLE_IMAGE_DIR, CLASS_NAMES
from src.predictor import BloodGroupPredictor

class TestE2WebIntegration(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_e2_checkpoints_and_class_mapping(self):
        """Verify that all 5 E2 checkpoints exist, load cleanly, and map classes correctly."""
        predictor = BloodGroupPredictor()
        self.assertTrue(predictor.is_model_ready())
        self.assertEqual(len(predictor.fold_models), 5)
        self.assertEqual(predictor.class_names, ["A+", "B+", "AB+", "O+"])
        self.assertEqual(CLASS_NAMES, ["A+", "B+", "AB+", "O+"])

    def test_three_scan_upload_success(self):
        """Test POST /api/upload with 3 valid fingerprint scans."""
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        self.assertTrue(os.path.exists(sample_path))
        with open(sample_path, "rb") as f:
            b1 = f.read()

        sample_path_b = os.path.join(SAMPLE_IMAGE_DIR, "B+.bmp")
        with open(sample_path_b, "rb") as f:
            b2 = f.read()

        sample_path_o = os.path.join(SAMPLE_IMAGE_DIR, "O+.bmp")
        with open(sample_path_o, "rb") as f:
            b3 = f.read()

        data = {
            "scan1": (io.BytesIO(b1), "test_scan1.bmp"),
            "scan2": (io.BytesIO(b2), "test_scan2.bmp"),
            "scan3": (io.BytesIO(b3), "test_scan3.bmp")
        }

        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()

        self.assertTrue(res_json["success"])
        self.assertIn("prediction_id", res_json)
        self.assertIn(res_json["blood_group"], ["A+", "B+", "AB+", "O+"])
        self.assertIn("probabilities", res_json)
        self.assertEqual(len(res_json["probabilities"]), 4)
        self.assertIn("image_urls", res_json)
        self.assertEqual(len(res_json["image_urls"]), 3)
        self.assertEqual(res_json["input_source"], "Uploaded Images (3 Fingerprint Scans)")
        self.assertGreaterEqual(res_json["confidence"], 0.0)
        self.assertLessEqual(res_json["confidence"], 1.0)

    def test_missing_scan_input(self):
        """Test that missing one of the 3 scans returns HTTP 400 with an understandable error."""
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        with open(sample_path, "rb") as f:
            b1 = f.read()

        # Only 2 scans provided
        data = {
            "scan1": (io.BytesIO(b1), "test_scan1.bmp"),
            "scan2": (io.BytesIO(b1), "test_scan2.bmp")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_json = res.get_json()
        self.assertIn("error", res_json)

    def test_corrupted_image_input(self):
        """Test that a corrupted file returns HTTP 400 with an error."""
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        with open(sample_path, "rb") as f:
            b1 = f.read()

        corrupt_bytes = b"This is not a valid BMP or image data at all!"
        data = {
            "scan1": (io.BytesIO(b1), "test_scan1.bmp"),
            "scan2": (io.BytesIO(corrupt_bytes), "corrupt_scan2.bmp"),
            "scan3": (io.BytesIO(b1), "test_scan3.bmp")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_json = res.get_json()
        self.assertIn("error", res_json)
        self.assertIn("corrupted", res_json["error"].lower())

    def test_invalid_extension(self):
        """Test that an invalid file extension is rejected."""
        data = {
            "scan1": (io.BytesIO(b"some data"), "scan1.exe"),
            "scan2": (io.BytesIO(b"some data"), "scan2.exe"),
            "scan3": (io.BytesIO(b"some data"), "scan3.exe")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        res_json = res.get_json()
        self.assertIn("error", res_json)

    def test_research_dashboard_benchmarks(self):
        """Verify that /api/config returns the audited benchmark metrics."""
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("research_benchmarks", data)
        benchmarks = data["research_benchmarks"]

        # 1. Combined methodological evaluation
        self.assertEqual(benchmarks["combined_methodological"]["accuracy"], "50.50%")
        self.assertEqual(benchmarks["combined_methodological"]["count"], "51/101")
        self.assertIn("out-of-fold predictions for 80 development participants", benchmarks["combined_methodological"]["explanation"])

        # 2. Development out-of-fold evaluation
        self.assertEqual(benchmarks["development_oof"]["accuracy"], "53.75%")
        self.assertEqual(benchmarks["development_oof"]["count"], "43/80")

        # 3. Separate locked-test evaluation
        self.assertEqual(benchmarks["locked_test"]["accuracy"], "38.10%")
        self.assertEqual(benchmarks["locked_test"]["count"], "8/21")

    def test_sample_scans_endpoint(self):
        """Verify that /api/sample-scans/<group> serves 3 authentic scans."""
        res = self.client.get("/api/sample-scans/A+")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["group"], "A+")
        self.assertEqual(len(data["scans"]), 3)
        self.assertTrue(data["scans"][0]["data"].startswith("data:image/bmp;base64,"))

if __name__ == "__main__":
    unittest.main()
