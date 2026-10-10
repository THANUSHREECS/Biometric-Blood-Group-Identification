"""
tests/test_mfs100_scanner.py
Unit tests for the refactored MFS100Scanner adapter.
Verifies both mock mode and bridge HTTP mode.
"""

import os
import io
import base64
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.mfs100.scanner import MFS100Scanner

class TestMFS100Scanner(unittest.TestCase):
    def setUp(self):
        # Default scanner in mock mode
        self.scanner = MFS100Scanner(mock=True)

    def test_mock_connect_and_disconnect(self):
        """Tests standard mock connection lifecycle."""
        self.assertTrue(self.scanner.connect())
        self.assertTrue(self.scanner.is_connected())

        info = self.scanner.get_device_info()
        self.assertEqual(info["model"], "MFS100")
        self.assertTrue(info["mock_mode"])
        self.assertTrue(info["is_connected"])

        self.assertTrue(self.scanner.disconnect())
        self.assertFalse(self.scanner.is_connected())

    def test_mock_capture(self):
        """Tests capture in mock mode."""
        self.scanner.connect()
        img_path, img_bytes = self.scanner.capture()
        self.assertTrue(os.path.exists(img_path))
        self.assertGreater(len(img_bytes), 1000)
        self.assertTrue(img_bytes.startswith(b"BM"))

        # Capture specific sample
        img_path_a, img_bytes_a = self.scanner.capture(sample_class="A+")
        self.assertTrue(os.path.exists(img_path_a))
        self.assertGreater(len(img_bytes_a), 1000)

    @patch("requests.post")
    def test_hardware_mode_connect_success(self, mock_post):
        """Tests hardware mode connection when bridge responds successfully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "success": True,
            "device": {
                "serial": "11215389",
                "model": "MFS100",
                "make": "MANTRA",
                "width": 316,
                "height": 354
            }
        }
        mock_post.return_value = mock_resp

        hw_scanner = MFS100Scanner(mock=False, bridge_url="http://127.0.0.1:5100")
        connected = hw_scanner.connect()
        self.assertTrue(connected)
        self.assertTrue(hw_scanner.is_connected())
        self.assertFalse(hw_scanner.mock)

        info = hw_scanner.get_device_info()
        self.assertEqual(info["serial"], "11215389")
        self.assertEqual(info["status"], "CONNECTED (PHYSICAL)")

    @patch("requests.post")
    def test_hardware_mode_capture(self, mock_post):
        """Tests hardware mode capture and BMP validation."""
        # Create a valid BMP
        img = Image.new("L", (316, 354), color=200)
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        valid_bmp = buf.getvalue()
        b64_str = base64.b64encode(valid_bmp).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "success": True,
            "image_b64": b64_str,
            "quality": 82,
            "nfiq": 1
        }
        mock_post.return_value = mock_resp

        hw_scanner = MFS100Scanner(mock=False, bridge_url="http://127.0.0.1:5100")
        hw_scanner._connected = True

        out_path, img_bytes = hw_scanner.capture()
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(img_bytes, valid_bmp)
        self.assertTrue(img_bytes.startswith(b"BM"))

    @patch("requests.get")
    def test_bridge_healthcheck(self, mock_get):
        """Tests bridge health check probe."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"running": True, "python_arch": "32bit"}
        mock_get.return_value = mock_resp

        hw_scanner = MFS100Scanner(mock=False, bridge_url="http://127.0.0.1:5100")
        health = hw_scanner.check_bridge_health()
        self.assertTrue(health["reachable"])
        self.assertEqual(health["data"]["python_arch"], "32bit")

if __name__ == "__main__":
    unittest.main()
