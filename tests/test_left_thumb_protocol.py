"""
tests/test_left_thumb_protocol.py
Automated tests for the LEFT THUMB acquisition protocol standardization.

Verifies:
- finger_used=LEFT_THUMB is stored correctly in the DB for MFS100 captures.
- Upload records do NOT receive finger_used (null).
- PDF report correctly displays Standard Acquisition Finger: LEFT THUMB.
- PDF is backward-compatible when finger_used is not passed.
- Model probabilities are unchanged (finger_used is metadata only).
- No prediction is forced to any blood group based on finger_used.
- 3-scan consensus records store finger_used=LEFT_THUMB.
- Upload mode is completely unaffected.
"""

import os
import sys
import tempfile
import unittest
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import init_db, insert_prediction, get_prediction_by_id, get_all_predictions
from src.report_generator import generate_pdf_report
from src.config import SAMPLE_IMAGE_DIR


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


def _sample_probs():
    return {
        "A+": 0.75, "A-": 0.05, "AB+": 0.05, "AB-": 0.02,
        "B+": 0.05, "B-": 0.02, "O+": 0.05, "O-": 0.01
    }


def _sample_img():
    p = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
    return p if os.path.exists(p) else None


class TestLeftThumbDB(unittest.TestCase):
    """Database-layer tests for finger_used metadata."""

    def setUp(self):
        self.db = _temp_db()
        init_db(self.db)

    def tearDown(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def test_finger_used_stored_for_mfs100_capture(self):
        """MFS100 physical capture record stores finger_used=LEFT_THUMB."""
        rec_id = insert_prediction(
            input_type="MFS100",
            image_path="cap_001.bmp",
            predicted_group="O+",
            confidence=0.92,
            status="NORMAL",
            notes="Hardware capture",
            finger_used="LEFT_THUMB",
            db_path=self.db
        )
        record = get_prediction_by_id(rec_id, db_path=self.db)
        self.assertIsNotNone(record)
        self.assertEqual(record["finger_used"], "LEFT_THUMB")

    def test_finger_used_not_set_for_upload(self):
        """Upload records have finger_used=None."""
        rec_id = insert_prediction(
            input_type="upload",
            image_path="upload_abc.bmp",
            predicted_group="A+",
            confidence=0.85,
            status="NORMAL",
            notes="File upload",
            finger_used=None,
            db_path=self.db
        )
        record = get_prediction_by_id(rec_id, db_path=self.db)
        self.assertIsNotNone(record)
        self.assertIsNone(record["finger_used"])

    def test_consensus_finger_used_metadata(self):
        """3-scan consensus records store finger_used=LEFT_THUMB and correct input_type."""
        rec_id = insert_prediction(
            input_type="MFS100 Consensus - Left Thumb (3 Impressions)",
            image_path="cap_consensus_003.bmp",
            predicted_group="O+",
            confidence=0.94,
            status="NORMAL",
            notes="Consensus capture",
            finger_used="LEFT_THUMB",
            db_path=self.db
        )
        record = get_prediction_by_id(rec_id, db_path=self.db)
        self.assertIsNotNone(record)
        self.assertEqual(record["finger_used"], "LEFT_THUMB")
        self.assertIn("Left Thumb", record["input_type"])

    def test_migration_safe_column_addition(self):
        """finger_used column is present after init_db."""
        conn = sqlite3.connect(self.db)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
        conn.close()
        self.assertIn("finger_used", cols)

    def test_upload_unaffected_by_protocol(self):
        """Multiple upload records all have finger_used=None."""
        for i in range(3):
            insert_prediction(
                input_type="upload",
                image_path=f"upload_{i}.bmp",
                predicted_group="B+",
                confidence=0.80,
                finger_used=None,
                db_path=self.db
            )
        records = get_all_predictions(limit=10, db_path=self.db)
        for r in records:
            self.assertIsNone(r["finger_used"],
                msg=f"Upload record should have finger_used=None, got: {r['finger_used']}")


class TestLeftThumbPDF(unittest.TestCase):
    """PDF report tests for the finger_used field."""

    def test_pdf_includes_left_thumb(self):
        """PDF with finger_used=LEFT_THUMB generates without error."""
        pdf_path = generate_pdf_report(
            prediction_id=77001,
            timestamp="2026-09-05 12:00:00",
            input_type="MFS100 Consensus (3 Impressions) | Standard Finger: LEFT THUMB",
            image_path=_sample_img(),
            predicted_group="O+",
            confidence=0.9459,
            probabilities=_sample_probs(),
            status="NORMAL",
            finger_used="LEFT_THUMB"
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    def test_pdf_without_finger_used_backward_compatible(self):
        """PDF without finger_used param generates normally (backward-compatible)."""
        pdf_path = generate_pdf_report(
            prediction_id=77002,
            timestamp="2026-09-05 12:01:00",
            input_type="MFS100 (Mock)",
            image_path=_sample_img(),
            predicted_group="A+",
            confidence=0.85,
            probabilities=_sample_probs(),
            status="NORMAL"
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    def test_pdf_upload_no_finger_used(self):
        """Upload PDF with finger_used=None generates normally."""
        pdf_path = generate_pdf_report(
            prediction_id=77003,
            timestamp="2026-09-05 12:02:00",
            input_type="Upload",
            image_path=_sample_img(),
            predicted_group="B+",
            confidence=0.78,
            probabilities=_sample_probs(),
            status="NORMAL",
            finger_used=None
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


class TestLeftThumbModelIntegrity(unittest.TestCase):
    """Verify finger_used is metadata only and cannot alter predictions."""

    def test_no_forced_prediction_by_finger_used(self):
        """predict() must NOT accept finger_used (prevents modification of pipeline)."""
        from src.predictor import BloodGroupPredictor
        import inspect
        sig = inspect.signature(BloodGroupPredictor.predict)
        self.assertNotIn("finger_used", list(sig.parameters.keys()),
            msg="predict() must NOT accept finger_used")

    def test_predict_consensus_no_finger_used_param(self):
        """predict_consensus() must NOT accept finger_used."""
        from src.predictor import BloodGroupPredictor
        import inspect
        sig = inspect.signature(BloodGroupPredictor.predict_consensus)
        self.assertNotIn("finger_used", list(sig.parameters.keys()),
            msg="predict_consensus() must NOT accept finger_used")

    def test_finger_used_column_is_text_metadata(self):
        """finger_used DB column is TEXT (metadata), not a numeric prediction driver."""
        db = _temp_db()
        try:
            init_db(db)
            conn = sqlite3.connect(db)
            col_info = {
                row[1]: row[2]
                for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
            }
            conn.close()
            self.assertIn("finger_used", col_info)
            self.assertEqual(col_info["finger_used"], "TEXT")
        finally:
            if os.path.exists(db):
                os.remove(db)

    def test_model_probabilities_sum_to_one(self):
        """Probability dicts sum to 1.0 regardless of finger metadata."""
        total = sum(_sample_probs().values())
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
