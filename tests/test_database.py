import os
import unittest
import tempfile
from src.database import init_db, insert_prediction, get_prediction_by_id, get_all_predictions

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_blood_group.db")

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_init_and_insert(self):
        init_db(self.test_db)
        rec_id = insert_prediction(
            input_type="upload",
            image_path="test_image.bmp",
            predicted_group="O+",
            confidence=0.88,
            status="NORMAL",
            notes="Test prediction",
            db_path=self.test_db
        )
        self.assertIsInstance(rec_id, int)
        self.assertGreater(rec_id, 0)

        # Fetch by ID
        rec = get_prediction_by_id(rec_id, db_path=self.test_db)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["predicted_group"], "O+")
        self.assertEqual(rec["confidence"], 0.88)
        self.assertEqual(rec["input_type"], "upload")

        # Fetch all
        all_recs = get_all_predictions(limit=10, db_path=self.test_db)
        self.assertEqual(len(all_recs), 1)

if __name__ == "__main__":
    unittest.main()
