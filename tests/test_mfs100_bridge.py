"""
tests/test_mfs100_bridge.py
Unit and endpoint tests for the MFS100 32-bit HTTP Bridge.
"""

import os
import io
import json
import base64
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

class TestMFS100BridgeEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We can test the Flask app in bridge.py directly
        import src.mfs100_bridge.bridge as bridge_mod
        cls.bridge_mod = bridge_mod
        cls.app = bridge_mod.app.test_client()

    def test_status_endpoint(self):
        """Tests GET /status returns expected keys and structure."""
        resp = self.app.get("/status")
        self.assertIn(resp.status_code, (200, 500))
        data = json.loads(resp.data)
        self.assertIn("running", data)
        self.assertTrue(data["running"])
        self.assertIn("version", data)
        self.assertIn("python_arch", data)

    def test_connect_endpoint_mocked(self):
        """Tests POST /connect endpoint with mocked SDK instance."""
        mock_mfs = MagicMock()
        mock_mfs.Init.return_value = 0
        mock_dev = MagicMock()
        mock_dev.SerialNo = "TEST12345"
        mock_dev.Make = "MANTRA"
        mock_dev.Model = "MFS100"
        mock_dev.Width = 316
        mock_dev.Height = 354
        mock_mfs.GetDeviceInfo.return_value = mock_dev

        with patch.object(self.bridge_mod, "mfs_instance", mock_mfs), \
             patch.object(self.bridge_mod, "sdk_loaded", True):
            resp = self.app.post("/connect")
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data["success"])
            self.assertEqual(data["device"]["serial"], "TEST12345")

    def test_connect_failure(self):
        """Tests POST /connect when Init fails."""
        mock_mfs = MagicMock()
        mock_mfs.Init.return_value = -1307 # Device not connected
        mock_mfs.GetErrorMsg.return_value = "Device not connected"

        with patch.object(self.bridge_mod, "mfs_instance", mock_mfs), \
             patch.object(self.bridge_mod, "sdk_loaded", True):
            resp = self.app.post("/connect")
            self.assertEqual(resp.status_code, 400)
            data = json.loads(resp.data)
            self.assertFalse(data["success"])
            self.assertEqual(data["error_code"], -1307)

    def test_capture_endpoint_mocked(self):
        """Tests POST /capture successfully encodes and returns image data."""
        # Create a tiny 10x10 BMP image
        img = Image.new("L", (316, 354), color=128)
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        bmp_bytes = buf.getvalue()

        mock_mfs = MagicMock()
        mock_mfs.Init.return_value = 0
        mock_fd = MagicMock()
        mock_fd.Quality = 75
        mock_fd.Nfiq = 1
        mock_fd.FingerImage = MagicMock()

        # Mock AutoCapture to return (0, mock_fd)
        mock_mfs.AutoCapture.return_value = (0, mock_fd)
        mock_mfs.BitmapToBytes.return_value = bmp_bytes

        with patch.object(self.bridge_mod, "mfs_instance", mock_mfs), \
             patch.object(self.bridge_mod, "FingerDataClass", MagicMock(return_value=mock_fd)), \
             patch.object(self.bridge_mod, "sdk_loaded", True), \
             patch.object(self.bridge_mod, "is_device_initialized", True):
            resp = self.app.post("/capture", json={"timeout_ms": 2000, "quality": 60})
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data["success"])
            self.assertEqual(data["quality"], 75)
            self.assertEqual(data["nfiq"], 1)
            self.assertIn("image_b64", data)

            decoded = base64.b64decode(data["image_b64"])
            self.assertTrue(decoded.startswith(b"BM"))

    def test_disconnect_endpoint(self):
        """Tests POST /disconnect."""
        mock_mfs = MagicMock()
        with patch.object(self.bridge_mod, "mfs_instance", mock_mfs), \
             patch.object(self.bridge_mod, "sdk_loaded", True):
            resp = self.app.post("/disconnect")
            self.assertEqual(resp.status_code, 200)
            data = json.loads(resp.data)
            self.assertTrue(data["success"])

if __name__ == "__main__":
    unittest.main()
import os
import sys
import json
import hashlib
import numpy as np
import pytest
from PIL import Image

