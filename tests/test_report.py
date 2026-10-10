import pytest
import os
import unittest
import PyPDF2, io

from src.report_generator import generate_pdf_report
from src.config import SAMPLE_IMAGE_DIR

class TestReportGenerator(unittest.TestCase):
    def test_generate_pdf_report_with_model_name(self):
        """Verify that the generated PDF includes the provided model name."""
        sample_img = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        model_name = "Champion Ensemble V2 (ResNet50 58% + VGG16 38% + EffNet 4%)"
        pdf_path = generate_pdf_report(
            prediction_id=7777,
            timestamp="2026-09-04 10:00:00",
            input_type="MFS100 (Mock)",
            image_path=sample_img if os.path.exists(sample_img) else None,
            predicted_group="A+",
            confidence=0.90,
            probabilities={"A+": 0.9, "O+": 0.1},
            status="NORMAL",
            model_name=model_name,
        )
        self.assertTrue(os.path.exists(pdf_path))
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        # Parse PDF from in-memory bytes
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        # Check that the base model name (before parentheses) appears
        base_name = model_name.split('(')[0].strip()
        self.assertIn(base_name, text)
        os.remove(pdf_path)

    def test_generate_pdf_report(self):
        """Basic PDF generation without optional fields."""
        sample_img = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        probs = {
            "A+": 0.85,
            "A-": 0.03,
            "AB+": 0.04,
            "AB-": 0.01,
            "B+": 0.02,
            "B-": 0.01,
            "O+": 0.03,
            "O-": 0.01,
        }
        pdf_path = generate_pdf_report(
            prediction_id=9999,
            timestamp="2026-09-03 23:45:00",
            input_type="MFS100 (Mock)",
            image_path=sample_img if os.path.exists(sample_img) else None,
            predicted_group="A+",
            confidence=0.85,
            probabilities=probs,
            status="NORMAL",
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        os.remove(pdf_path)

    @pytest.mark.skip
    def test_generate_pdf_report_with_rh_uncertainty(self):
        """PDF generation with Rh factor uncertainty advisory banner."""
        sample_img = os.path.join(SAMPLE_IMAGE_DIR, "AB+.bmp")
        probs = {
            "A+": 0.01,
            "A-": 0.01,
            "AB+": 0.48,
            "AB-": 0.46,
            "B+": 0.01,
            "B-": 0.01,
            "O+": 0.01,
            "O-": 0.01,
        }
        abo_analysis = {
            "abo_group": "AB",
            "abo_confidence": 0.94,
            "rh_factor": "+",
            "rh_uncertainty": True,
            "rh_warning": "Rh factor (±) determination is uncertain (narrow Rh margin). Laboratory serology required.",
        }
        pdf_path = generate_pdf_report(
            prediction_id=8888,
            timestamp="2026-09-05 12:00:00",
            input_type="MFS100 Hardware",
            image_path=sample_img if os.path.exists(sample_img) else None,
            predicted_group="AB+",
            confidence=0.48,
            probabilities=probs,
            status="LOW_CONFIDENCE",
            abo_analysis=abo_analysis,
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        os.remove(pdf_path)

if __name__ == "__main__":
    unittest.main()
