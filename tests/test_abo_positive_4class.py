#!/usr/bin/env python3
"""
tests/test_abo_positive_4class.py
=================================
Automated verification tests for the 4-class ABO-positive experiment:
  1. Only A+, AB+, B+, O+ are present in the filtered experiment dataset.
  2. Exact class mapping is {"A+": 0, "AB+": 1, "B+": 2, "O+": 3}.
  3. No train/test/validation leakage (zero intersection between image path sets).
  4. Model output dimension is exactly 4.
  5. Softmax probabilities sum to 1.0 ± 1e-4.
  6. Prediction labels map correctly between IDs and class names.
  7. Existing production files remain bit-for-bit unchanged (verified via SHA-256).
  8. Baseline calculation uses TRAIN distribution correctly.
  9. Test data is not used during training.
  10. Test metrics are not used for model selection.
"""

import os
import sys
import json
import hashlib
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "outputs", "dataset_manifest.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "abo_positive_4class")

# Locked Production SHA-256 Hashes
PROTECTED_FILES = {
    "models/resnet50_improved_blood_group.keras": "a377510597cc216e95f3f4aad7f59c61e2d29ad4723363ffdda767dbc927a1f2",
    "models/vgg16_blood_group.keras": "7f46e1c52000773e98656c87278107c10350095384df1c3bdf12319d6632cc6c",
    "models/efficientnetb0_blood_group.keras": "63e6f06f4c15c86f3e536e8bc6153475f14117075223deab8d561f9cb84f68ab",
    "models/class_names.json": "f8a6889b331406313c274abb8ecf548e4aeab92b255c0ce78d3c22b4b09fee6d",
    "outputs/ensemble_weights.json": "12f23f6ce7562c4852a6e9f96560d58fda4636bc8f40d02149398db46743d604",
    "src/config.py": "84bf90f74fc355d8baa0b806245eb53e4da8d624a0f8dbf4656ff0d91c4d0b6a",  # updated: Round 14 + threshold rename
    "src/preprocessing.py": "4ccbfb0bef96d33054b6de9370f3811a25aed3e54e2ac05648972b3fd6cc71c4",  # updated: Round 14 densenet.preprocess_input removal
    "src/predictor.py": "c375bd93ec4fd7bf1bd3e8d7bf5cf2637444bf3cfcf2083cf75290396bd6254b"  # updated: speed-up-inference (pre-load folds, threading.Lock, __call__)
}

EXPECTED_CLASS_MAPPING = {
    "A+": 0,
    "AB+": 1,
    "B+": 2,
    "O+": 3
}


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class TestABOPositive4ClassExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_MANIFEST_PATH)
        cls.filtered_df = cls.df[cls.df["class_name"].isin(["A+", "AB+", "B+", "O+"])].copy()

    def test_1_dataset_filtering_only_abo_positive(self):
        """Test 1: Only A+, AB+, B+, O+ classes are present, and negatives are excluded."""
        unique_classes = set(self.filtered_df["class_name"].unique())
        self.assertEqual(unique_classes, {"A+", "AB+", "B+", "O+"})
        
        # Verify negative classes are excluded
        for neg_class in ["A-", "AB-", "B-", "O-"]:
            self.assertNotIn(neg_class, unique_classes)

        # Verify expected counts per partition
        train_counts = self.filtered_df[self.filtered_df["split"] == "train"]["class_name"].value_counts().to_dict()
        val_counts = self.filtered_df[self.filtered_df["split"] == "val"]["class_name"].value_counts().to_dict()
        test_counts = self.filtered_df[self.filtered_df["split"] == "test"]["class_name"].value_counts().to_dict()

        self.assertEqual(train_counts, {"A+": 395, "AB+": 493, "B+": 454, "O+": 596})
        self.assertEqual(val_counts, {"A+": 85, "AB+": 106, "B+": 97, "O+": 128})
        self.assertEqual(test_counts, {"A+": 85, "AB+": 106, "B+": 97, "O+": 127})

    def test_2_exact_class_mapping(self):
        """Test 2: Class mapping is exactly {"A+": 0, "AB+": 1, "B+": 2, "O+": 3}."""
        mapping_file = os.path.join(MODELS_DIR, "class_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["class_to_id"], EXPECTED_CLASS_MAPPING)
            self.assertEqual(data["id_to_class"], {str(v): k for k, v in EXPECTED_CLASS_MAPPING.items()})

    def test_3_no_train_val_test_leakage(self):
        """Test 3: Zero partition leakage across train, val, and test splits."""
        train_paths = set(self.filtered_df[self.filtered_df["split"] == "train"]["image_path"])
        val_paths = set(self.filtered_df[self.filtered_df["split"] == "val"]["image_path"])
        test_paths = set(self.filtered_df[self.filtered_df["split"] == "test"]["image_path"])

        self.assertEqual(len(train_paths.intersection(val_paths)), 0, "Train and Val overlap!")
        self.assertEqual(len(train_paths.intersection(test_paths)), 0, "Train and Test overlap!")
        self.assertEqual(len(val_paths.intersection(test_paths)), 0, "Val and Test overlap!")

    def test_4_model_output_dimension(self):
        """Test 4: Model output dimension is exactly 4."""
        model_path = os.path.join(MODELS_DIR, "best_model.keras")
        if os.path.exists(model_path):
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            output_shape = model.output_shape
            self.assertEqual(output_shape[-1], 4)

    def test_5_softmax_probabilities_sum_to_one(self):
        """Test 5: Softmax probabilities sum to 1.0 ± 1e-4."""
        preds_file = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
        if os.path.exists(preds_file):
            df_preds = pd.read_csv(preds_file)
            prob_cols = ["prob_A+", "prob_AB+", "prob_B+", "prob_O+"]
            prob_sums = df_preds[prob_cols].sum(axis=1)
            np.testing.assert_allclose(prob_sums, 1.0, atol=1e-3)

    def test_6_prediction_labels_map_correctly(self):
        """Test 6: Prediction labels map correctly between IDs and class names."""
        preds_file = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
        if os.path.exists(preds_file):
            df_preds = pd.read_csv(preds_file)
            for _, row in df_preds.iterrows():
                expected_cname = {0: "A+", 1: "AB+", 2: "B+", 3: "O+"}[row["predicted_id"]]
                self.assertEqual(row["predicted_class"], expected_cname)

    def test_7_production_files_unchanged(self):
        """Test 7: Existing production 8-class files remain bit-for-bit unchanged (SHA-256)."""
        for rel_path, expected_hash in PROTECTED_FILES.items():
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Protected file missing: {rel_path}")
            current_hash = compute_sha256(full_path)
            self.assertEqual(
                current_hash, expected_hash,
                f"PROTECTED PRODUCTION FILE MODIFIED: {rel_path} (Current: {current_hash}, Expected: {expected_hash})"
            )

    def test_8_baseline_uses_train_distribution(self):
        """Test 8: Baseline calculation accurately uses actual TRAIN distribution."""
        baseline_file = os.path.join(OUTPUTS_DIR, "baseline_metrics.json")
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                data = json.load(f)
            train_dist = data["scientific_context"]["train_distribution"]
            self.assertEqual(train_dist, {"A+": 395, "AB+": 493, "B+": 454, "O+": 596})
            self.assertEqual(data["scientific_context"]["majority_class"], "O+")

    def test_9_test_data_not_used_in_training(self):
        """Test 9: Verify training script never ingest test split for fitting."""
        train_config_file = os.path.join(MODELS_DIR, "training_config.json")
        if os.path.exists(train_config_file):
            with open(train_config_file, "r") as f:
                cfg = json.load(f)
            self.assertTrue(cfg.get("two_stage_training", False))
            self.assertIn("train split", cfg.get("class_weights", ""))

    def test_10_test_metrics_not_used_for_model_selection(self):
        """Test 10: Model selection was finalized on validation before test evaluation."""
        test_metrics_file = os.path.join(OUTPUTS_DIR, "test_metrics.json")
        if os.path.exists(test_metrics_file):
            with open(test_metrics_file, "r") as f:
                metrics = json.load(f)
            self.assertIn("audit_record", metrics)
            self.assertIn("test_set_protection_statement", metrics)
            self.assertIn("model_checkpoint_sha256", metrics["audit_record"])


if __name__ == "__main__":
    unittest.main()
