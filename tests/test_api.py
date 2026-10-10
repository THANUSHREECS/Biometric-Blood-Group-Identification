import os
import unittest
import json
from app import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_get_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"HemoPrint AI", response.data)
        self.assertIn(b"MANDATORY MEDICAL DISCLAIMER", response.data)

    def test_get_history_page(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Audit Trail", response.data)

    def test_api_status(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("model", data)
        self.assertIn("scanner", data)

    def test_api_history(self):
        response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("records", data)
        self.assertIn("total", data)

    def test_scanner_connect_disconnect(self):
        conn_res = self.client.post("/api/scanner/connect")
        self.assertEqual(conn_res.status_code, 200)
        data = conn_res.get_json()
        self.assertTrue(data["connected"])

        disconn_res = self.client.post("/api/scanner/disconnect")
        self.assertEqual(disconn_res.status_code, 200)
        data_d = disconn_res.get_json()
        self.assertTrue(data_d["disconnected"])

if __name__ == "__main__":
    unittest.main()
