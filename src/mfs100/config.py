"""
src/mfs100/config.py
Configuration for Mantra MFS100 fingerprint scanner integration.
"""

import os
import platform

# Default SDK DLL Path on Windows
DEFAULT_MFS100_DLL_PATH = os.environ.get(
    "MFS100_SDK_PATH",
    r"C:\Program Files\Mantra\MFS100\Driver\MFS100Test\MFS100Dll.dll"
)

# Mock Mode Configuration (defaults to True per user instruction regarding 32-bit/64-bit safety)
MFS100_MOCK = os.environ.get("MFS100_MOCK", "true").lower() in ("true", "1", "yes")

# Capture timeout in milliseconds
MFS100_TIMEOUT_MS = int(os.environ.get("MFS100_TIMEOUT_MS", "10000"))

# Min quality threshold (0-100)
MFS100_QUALITY_THRESHOLD = int(os.environ.get("MFS100_QUALITY_THRESHOLD", "55"))

# Python bitness detection
PYTHON_IS_64BIT = platform.architecture()[0] == "64bit"

# 32-bit HTTP Bridge settings (for connecting 64-bit Python to 32-bit Mantra SDK)
MFS100_BRIDGE_HOST = os.environ.get("MFS100_BRIDGE_HOST", "127.0.0.1")
MFS100_BRIDGE_PORT = int(os.environ.get("MFS100_BRIDGE_PORT", "5100"))
MFS100_BRIDGE_URL = os.environ.get("MFS100_BRIDGE_URL", f"http://{MFS100_BRIDGE_HOST}:{MFS100_BRIDGE_PORT}")
MFS100_BRIDGE_TIMEOUT = int(os.environ.get("MFS100_BRIDGE_TIMEOUT", "20"))
