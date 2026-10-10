import os
import unittest
import pandas as pd
import json

class TestDatasetPreparation(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.manifest_path = os.path.join(self.project_root, "outputs", "dataset_manifest.csv")
        self.report_path = os.path.join(self.project_root, "outputs", "dataset_report.json")
        self.class_names_path = os.path.join(self.project_root, "models", "class_names.json")
        self.quarantine_manifest_path = os.path.join(self.project_root, "data", "quarantine", "quarantine_manifest.json")
        self.raw_dir = os.path.join(self.project_root, "data", "raw")
        self.processed_dir = os.path.join(self.project_root, "data", "processed")
        self.sample_dir = os.path.join(self.project_root, "data", "sample")

    def test_class_names_json(self):
        self.assertTrue(os.path.exists(self.class_names_path))
        with open(self.class_names_path) as f:
            data = json.load(f)
        self.assertIn("class_names", data)
        self.assertEqual(len(data["class_names"]), 8)
        expected = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
        self.assertEqual(data["class_names"], expected)
        for idx, name in enumerate(expected):
            self.assertEqual(data["class_to_id"][name], idx)

    def test_manifest_csv_integrity(self):
        self.assertTrue(os.path.exists(self.manifest_path))
        df = pd.read_csv(self.manifest_path)
        # Check required columns
        for col in ["image_path", "class_name", "class_id", "split"]:
            self.assertIn(col, df.columns)
        
        # Check total clean count
        self.assertEqual(len(df), 5985)
        
        # Check all splits exist
        splits = set(df["split"].unique())
        self.assertEqual(splits, {"train", "val", "test"})
        
        # Check split counts
        train_count = (df["split"] == "train").sum()
        val_count = (df["split"] == "val").sum()
        test_count = (df["split"] == "test").sum()
        self.assertEqual(train_count, 4189)
        self.assertEqual(val_count, 898)
        self.assertEqual(test_count, 898)
        
        # Check that each class is in all 3 splits
        for c in ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]:
            self.assertGreater((df[df["split"] == "train"]["class_name"] == c).sum(), 0)
            self.assertGreater((df[df["split"] == "val"]["class_name"] == c).sum(), 0)
            self.assertGreater((df[df["split"] == "test"]["class_name"] == c).sum(), 0)

    def test_quarantine_manifest(self):
        self.assertTrue(os.path.exists(self.quarantine_manifest_path))
        with open(self.quarantine_manifest_path) as f:
            data = json.load(f)
        self.assertEqual(data["duplicate_groups_count"], 11)
        self.assertEqual(data["total_quarantined"], 15)

    def test_sample_fingerprints(self):
        self.assertTrue(os.path.exists(self.sample_dir))
        samples = os.listdir(self.sample_dir)
        self.assertEqual(len(samples), 8)

if __name__ == "__main__":
    unittest.main()
