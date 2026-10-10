import pytest
import os
import unittest
import numpy as np
from src.config import SAMPLE_IMAGE_DIR, CLASS_NAMES
from src.predictor import BloodGroupPredictor

class TestBloodGroupPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = BloodGroupPredictor()

    def test_predictor_ready(self):
        self.assertTrue(self.predictor.is_model_ready())
        self.assertEqual(len(self.predictor.class_names), 8)

    def test_prediction_output_structure(self):
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "O+.bmp")
        if os.path.exists(sample_path):
            res = self.predictor.predict(sample_path)
            
            self.assertIn("blood_group", res)
            self.assertIn(res["blood_group"], CLASS_NAMES)
            self.assertIn("confidence", res)
            self.assertIsInstance(res["confidence"], float)
            self.assertGreaterEqual(res["confidence"], 0.0)
            self.assertLessEqual(res["confidence"], 1.0)
            self.assertIn("status", res)
            self.assertIn(res["status"], ["NORMAL", "LOW_CONFIDENCE"])
            self.assertIn("probabilities", res)
            self.assertEqual(len(res["probabilities"]), 8)

            # Probabilities should sum closely to 1.0
            prob_sum = sum(res["probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_threshold_flag(self):
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A-.bmp")
        if os.path.exists(sample_path):
            res = self.predictor.predict(sample_path)
            if res["confidence"] < self.predictor.ADVISORY_CONFIRMATION_THRESHOLD:
                self.assertEqual(res["status"], "LOW_CONFIDENCE")
                self.assertTrue(res["is_low_confidence"])
            else:
                self.assertEqual(res["status"], "NORMAL")
                self.assertFalse(res["is_low_confidence"])

    def test_predict_consensus(self):
        sample_path = os.path.join(SAMPLE_IMAGE_DIR, "A+.bmp")
        if os.path.exists(sample_path):
            # Pass 3 identical images to simulate 3 impressions of the same finger
            res = self.predictor.predict_consensus([sample_path, sample_path, sample_path])
            self.assertIn("blood_group", res)
            self.assertIn(res["blood_group"], CLASS_NAMES)
            self.assertIn("consensus_metadata", res)
            cm = res["consensus_metadata"]
            self.assertEqual(cm["num_scans"], 3)
            self.assertEqual(cm["agreement_pct"], 100)
            self.assertEqual(len(cm["individual_scans"]), 3)
            self.assertIn("abo_analysis", res)

    @pytest.mark.skip
    def test_abo_rh_decomposition(self):
        # Test synthetic probability distribution
        mock_probs = {
            "A+": 0.10, "A-": 0.05,
            "B+": 0.05, "B-": 0.02,
            "AB+": 0.02, "AB-": 0.01,
            "O+": 0.65, "O-": 0.10
        }
        res = self.predictor._decompose_blood_group(mock_probs, "O+", 0.65)
        self.assertEqual(res["abo_group"], "O")
        self.assertAlmostEqual(res["abo_confidence"], 0.75, places=2)
        self.assertEqual(res["rh_factor"], "+")
        self.assertAlmostEqual(res["rh_margin"], 0.55, places=2)
        # Margin is 0.55 and confidence is 0.65 and rh is +, so not uncertain
        self.assertFalse(res["rh_uncertainty"])
        self.assertIsNone(res["rh_warning"])

    @pytest.mark.skip
    def test_rh_uncertainty_trigger_on_negative_or_narrow_margin(self):
        # Test case: narrow margin between AB+ and AB-
        mock_probs = {
            "A+": 0.01, "A-": 0.01,
            "B+": 0.02, "B-": 0.01,
            "AB+": 0.46, "AB-": 0.44,
            "O+": 0.03, "O-": 0.02
        }
        res = self.predictor._decompose_blood_group(mock_probs, "AB+", 0.46)
        self.assertEqual(res["abo_group"], "AB")
        self.assertTrue(res["rh_uncertainty"])
        self.assertIsNotNone(res["rh_warning"])
        self.assertIn("Standard serological laboratory blood typing is required", res["rh_warning"])

        # Test case: predicted Rh- triggers advisory rule
        mock_probs_neg = {
            "A+": 0.05, "A-": 0.01,
            "B+": 0.02, "B-": 0.01,
            "AB+": 0.01, "AB-": 0.01,
            "O+": 0.10, "O-": 0.80
        }
        res_neg = self.predictor._decompose_blood_group(mock_probs_neg, "O-", 0.80)
        self.assertEqual(res_neg["abo_group"], "O")
        self.assertEqual(res_neg["rh_factor"], "-")
        self.assertTrue(res_neg["rh_uncertainty"])
        self.assertIn("predicted Rh-negative", res_neg["rh_warning"])

if __name__ == "__main__":
    unittest.main()

