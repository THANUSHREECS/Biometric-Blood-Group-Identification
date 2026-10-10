"""
src/mfs100_bridge/config.py
Configuration for the 32-bit MFS100 HTTP Bridge process.
"""

import os

# Server binding (localhost only for security)
BRIDGE_HOST = os.environ.get("MFS100_BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("MFS100_BRIDGE_PORT", "5100"))

# Mantra SDK installation directory containing MANTRA.MFS100.dll and MFS100Dll.dll
MFS100_SDK_DIR = os.environ.get(
    "MFS100_SDK_DIR",
    r"C:\Program Files\Mantra\MFS100\Driver\MFS100Test"
)

# Default capture quality threshold (0-100, 55 recommended by Mantra)
DEFAULT_QUALITY = int(os.environ.get("MFS100_QUALITY_THRESHOLD", "55"))

# Default capture timeout in milliseconds (10 seconds)
DEFAULT_TIMEOUT_MS = int(os.environ.get("MFS100_TIMEOUT_MS", "10000"))
