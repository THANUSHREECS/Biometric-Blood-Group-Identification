"""
src/mfs100/scanner.py
Unified adapter for the Mantra MFS100 Optical Fingerprint Scanner.

Provides:
- Production-grade Mock Mode using authentic sample fingerprints (data/sample/)
- Real hardware integration via out-of-process 32-bit HTTP Bridge (127.0.0.1:5100)
- Seamless compatibility with 64-bit Python web application
"""

import os
import sys
import io
import time
import random
import base64
import logging
import requests
from PIL import Image

from src.mfs100.config import (
    DEFAULT_MFS100_DLL_PATH,
    MFS100_MOCK,
    MFS100_TIMEOUT_MS,
    MFS100_QUALITY_THRESHOLD,
    PYTHON_IS_64BIT,
    MFS100_BRIDGE_URL,
    MFS100_BRIDGE_TIMEOUT
)
from src.config import SAMPLE_IMAGE_DIR, CAPTURED_IMAGE_DIR

logger = logging.getLogger("mfs100_scanner")

class MFS100Scanner:
    """
    Standardized adapter for Mantra MFS100 Fingerprint Scanner.
    Communicates with physical hardware via the local 32-bit bridge service,
    or falls back to mock emulation for development and testing.
    """

    def __init__(self, mock=None, bridge_url=None, dll_path=None):
        self.mock = MFS100_MOCK if mock is None else bool(mock)
        self.bridge_url = bridge_url or MFS100_BRIDGE_URL
        self.dll_path = dll_path or DEFAULT_MFS100_DLL_PATH
        self._connected = False
        self._device_info = {
            "make": "Mantra Softech India Pvt. Ltd.",
            "model": "MFS100",
            "serial": "MFS100_DEMO",
            "firmware": "v9.0.2.5",
            "mock_mode": self.mock,
            "status": "DISCONNECTED"
        }
        self.last_capture_meta = None

        os.makedirs(CAPTURED_IMAGE_DIR, exist_ok=True)
        os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)

        if not self.mock:
            self._probe_bridge()

    def _probe_bridge(self):
        """Probes the 32-bit HTTP bridge to check availability."""
        try:
            resp = requests.get(f"{self.bridge_url}/status", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                self._device_info["bridge_status"] = "REACHABLE"
                self._device_info["bridge_arch"] = data.get("python_arch", "unknown")
                if data.get("device"):
                    self._device_info.update(data["device"])
                logger.info("32-bit MFS100 Bridge reachable.")
            else:
                self._device_info["bridge_status"] = f"HTTP_{resp.status_code}"
        except requests.RequestException as e:
            self._device_info["bridge_status"] = "UNREACHABLE"
            logger.warning(f"32-bit MFS100 Bridge is unreachable at {self.bridge_url}: {e}")

    def check_bridge_health(self):
        """Returns bridge connectivity and diagnostic details."""
        try:
            resp = requests.get(f"{self.bridge_url}/status", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                return {"reachable": True, "data": data}
            return {"reachable": False, "error": f"HTTP {resp.status_code}"}
        except requests.RequestException as e:
            return {
                "reachable": False,
                "error": f"Cannot reach bridge at {self.bridge_url}. Is the 32-bit bridge running? ({e})"
            }

    def set_mode(self, mock_mode: bool):
        """
        Dynamically and cleanly switches between mock emulation and physical hardware.
        Ensures device handles and status flags are properly synchronized.
        """
        self.disconnect()
        self.mock = bool(mock_mode)
        self._connected = False
        self._device_info["mock_mode"] = self.mock

        if self.mock:
            self._connected = True
            self._device_info["status"] = "CONNECTED (MOCK)"
            logger.info("MFS100Scanner mode switched to: MOCK MODE")
            return True
        else:
            self._device_info["status"] = "CONNECTING (PHYSICAL)"
            logger.info("MFS100Scanner mode switched to: PHYSICAL HARDWARE")
            self._probe_bridge()
            connected = self.connect()
            return connected

    def connect(self):
        """Connects to the MFS100 scanner or initializes mock mode."""
        if self.mock:
            self._connected = True
            self._device_info["mock_mode"] = True
            self._device_info["status"] = "CONNECTED (MOCK)"
            return True

        try:
            resp = requests.post(f"{self.bridge_url}/connect", timeout=MFS100_BRIDGE_TIMEOUT)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200 and data.get("success"):
                self._connected = True
                self._device_info["mock_mode"] = False
                self._device_info["status"] = "CONNECTED (PHYSICAL)"
                if data.get("device"):
                    self._device_info.update(data["device"])
                logger.info(f"Connected to physical MFS100: {self._device_info.get('serial')}")
                return True
            else:
                self._connected = False
                err = data.get("error", resp.text or f"HTTP {resp.status_code}")
                self._device_info["status"] = f"ERROR ({err})"
                logger.error(f"Failed to connect physical MFS100: {err}")
                return False

        except requests.ConnectionError:
            self._connected = False
            self._device_info["status"] = "ERROR (Bridge offline. Run 32-bit bridge first)"
            logger.error(f"Cannot connect to MFS100: Bridge offline at {self.bridge_url}")
            return False
        except Exception as e:
            self._connected = False
            self._device_info["status"] = f"ERROR ({str(e)})"
            return False

    def disconnect(self):
        """Disconnects from the scanner."""
        if self.mock:
            self._connected = False
            self._device_info["status"] = "DISCONNECTED"
            return True

        try:
            requests.post(f"{self.bridge_url}/disconnect", timeout=5.0)
        except Exception as e:
            logger.warning(f"Error disconnecting via bridge: {e}")

        self._connected = False
        self._device_info["status"] = "DISCONNECTED"
        return True

    def is_connected(self):
        """Returns connection state."""
        return self._connected

    def get_device_info(self):
        """Returns dictionary of device status and specifications."""
        info = dict(self._device_info)
        info["is_connected"] = self._connected
        info["mock_mode"] = self.mock
        return info

    def capture(self, timeout_ms=MFS100_TIMEOUT_MS, sample_class=None):
        """
        Captures a fingerprint image.
        Returns:
            saved_image_path (str): Path to the saved captured BMP image.
            image_bytes (bytes): Raw bytes of the captured fingerprint image.
        """
        if not self._connected:
            connected = self.connect()
            if not connected and not self.mock:
                raise RuntimeError(
                    f"MFS100 scanner is not connected. Status: {self._device_info.get('status')}"
                )

        timestamp = int(time.time() * 1000)
        capture_filename = f"capture_{timestamp}.bmp"
        captured_path = os.path.join(CAPTURED_IMAGE_DIR, capture_filename)

        if self.mock:
            return self._capture_mock(captured_path, sample_class=sample_class)
        else:
            return self._capture_hardware(captured_path, timeout_ms=timeout_ms)

    def get_last_capture_meta(self):
        """Returns metadata from the most recent capture (Quality, NFIQ, Pressure, Placement advice)."""
        return self.last_capture_meta or {
            "quality": 80,
            "nfiq": 2,
            "pressure_status": "OPTIMAL",
            "placement_advice": "Finger placement optimal.",
            "source": "mock" if self.mock else "hardware"
        }

    def capture_with_meta(self, timeout_ms=MFS100_TIMEOUT_MS, sample_class=None):
        """
        Captures a fingerprint image and returns (saved_image_path, image_bytes, metadata).
        """
        path, b = self.capture(timeout_ms=timeout_ms, sample_class=sample_class)
        return path, b, self.get_last_capture_meta()

    def _capture_mock(self, output_path, sample_class=None):
        """Simulates capture by reading genuine fingerprint samples from data/sample/."""
        samples = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(".bmp")]
        if not samples:
            raise FileNotFoundError(
                f"No sample fingerprints found in {SAMPLE_IMAGE_DIR}. Run scripts/prepare_dataset.py first!"
            )

        if sample_class and f"{sample_class}.bmp" in samples:
            chosen_sample = f"{sample_class}.bmp"
        else:
            chosen_sample = random.choice(samples)

        sample_full_path = os.path.join(SAMPLE_IMAGE_DIR, chosen_sample)
        with open(sample_full_path, "rb") as f:
            img_bytes = f.read()

        # Save to captured folder
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        self.last_capture_meta = {
            "quality": 92,
            "nfiq": 1,
            "pressure_status": "OPTIMAL",
            "placement_advice": "Mock biometric sample (simulated optimal quality).",
            "source": "mock"
        }

        return output_path, img_bytes

    def _capture_hardware(self, output_path, timeout_ms=MFS100_TIMEOUT_MS):
        """Captures raw image from physical MFS100 scanner via the 32-bit bridge service."""
        try:
            payload = {
                "timeout_ms": timeout_ms,
                "quality": MFS100_QUALITY_THRESHOLD
            }
            resp = requests.post(
                f"{self.bridge_url}/capture",
                json=payload,
                timeout=max(int(timeout_ms / 1000) + 5, MFS100_BRIDGE_TIMEOUT)
            )

            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code != 200 or not data.get("success"):
                err_msg = data.get("error", resp.text or f"Capture failed with HTTP {resp.status_code}")
                raise RuntimeError(f"MFS100 Capture Failed: {err_msg}")

            image_b64 = data.get("image_b64")
            if not image_b64:
                raise ValueError("Bridge response did not contain image data")

            img_bytes = base64.b64decode(image_b64)

            # Quality and format validation
            if len(img_bytes) < 1000:
                raise ValueError(f"Captured image payload too small ({len(img_bytes)} bytes)")

            if not img_bytes.startswith(b"BM"):
                raise ValueError("Captured image does not have a valid BMP file header")

            # Validate image can be opened by PIL
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                width, height = pil_img.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid image dimensions: {width}x{height}")

            # Save to disk
            with open(output_path, "wb") as f:
                f.write(img_bytes)

            quality = int(data.get("quality", 70))
            nfiq = int(data.get("nfiq", 2))

            if quality >= 70 and nfiq in (1, 2):
                pressure_status = "OPTIMAL"
                placement_advice = "Good placement and pressure. High-quality capture."
            elif quality < 50 or nfiq in (4, 5):
                pressure_status = "TOO_LIGHT"
                placement_advice = "Impression is faint or off-center. Press slightly firmer and center finger on sensor."
            else:
                pressure_status = "MODERATE"
                placement_advice = "Acceptable capture. Holding finger steady and centered improves confidence."

            self.last_capture_meta = {
                "quality": quality,
                "nfiq": nfiq,
                "pressure_status": pressure_status,
                "placement_advice": placement_advice,
                "source": "hardware"
            }

            logger.info(
                f"Physical fingerprint captured successfully! "
                f"Quality={quality}, NFIQ={nfiq}, Pressure={pressure_status}, Saved to: {output_path}"
            )
            return output_path, img_bytes

        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to 32-bit MFS100 bridge at {self.bridge_url}. "
                "Ensure the bridge service is running via C:\\Python310-32\\python.exe -m src.mfs100_bridge.bridge"
            )
        except Exception as e:
            logger.exception("Hardware capture error")
            raise e
