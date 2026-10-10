"""
src/mfs100_bridge/bridge.py
32-bit HTTP REST Bridge for Mantra MFS100 Optical Fingerprint Scanner.

This process runs strictly in a 32-bit Python environment (C:\\Python310-32\\python.exe)
and provides a JSON HTTP API over localhost (127.0.0.1:5100) allowing 64-bit applications
(such as the main Flask web server) to communicate seamlessly with the 32-bit Mantra SDK.
"""

import os
import sys
import time
import base64
import platform
import threading
import logging
from flask import Flask, jsonify, request

from src.mfs100_bridge.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    MFS100_SDK_DIR,
    DEFAULT_QUALITY,
    DEFAULT_TIMEOUT_MS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [MFS100-Bridge] %(message)s"
)
logger = logging.getLogger("mfs100_bridge")

# Verify 32-bit Python environment
arch = platform.architecture()[0]
if arch != "32bit":
    logger.warning(f"Bridge is running in {arch} Python! MFS100 32-bit DLL requires 32-bit Python.")

# Initialize SDK wrapper
sdk_loaded = False
mfs_instance = None
FingerDataClass = None
load_error = None

try:
    if os.path.exists(MFS100_SDK_DIR):
        sys.path.append(MFS100_SDK_DIR)
        os.add_dll_directory(MFS100_SDK_DIR) if hasattr(os, "add_dll_directory") else None

    import clr
    clr.AddReference("System.Drawing")
    clr.AddReference("MANTRA.MFS100")
    from MANTRA import MFS100, FingerData
    FingerDataClass = FingerData
    mfs_instance = MFS100()
    sdk_loaded = True
    logger.info("MANTRA.MFS100 .NET assembly loaded successfully.")
except Exception as e:
    load_error = str(e)
    logger.error(f"Failed to load MANTRA.MFS100 SDK: {e}")

# Bridge state
scanner_lock = threading.Lock()
is_device_initialized = False

app = Flask("mfs100_bridge")

def _get_device_dict():
    """Helper to extract device info dictionary from MFS100 instance."""
    if not sdk_loaded or not mfs_instance:
        return None
    try:
        dev = mfs_instance.GetDeviceInfo()
        if dev:
            return {
                "serial": str(dev.SerialNo),
                "make": str(dev.Make),
                "model": str(dev.Model),
                "width": int(dev.Width),
                "height": int(dev.Height),
                "sdk_version": str(mfs_instance.GetSDKVersion()) if hasattr(mfs_instance, "GetSDKVersion") else "v9.0.2.5"
            }
    except Exception as e:
        logger.warning(f"Error reading device info: {e}")
    return None

@app.route("/status", methods=["GET"])
def get_status():
    """Healthcheck and device status endpoint."""
    with scanner_lock:
        device_connected = False
        if sdk_loaded and mfs_instance:
            try:
                device_connected = bool(mfs_instance.IsConnected())
            except Exception:
                device_connected = False

        device_info = _get_device_dict() if is_device_initialized else None

        return jsonify({
            "status": "ok" if sdk_loaded else "error",
            "running": True,
            "version": "1.0.0",
            "python_arch": platform.architecture()[0],
            "python_version": sys.version.split()[0],
            "sdk_loaded": sdk_loaded,
            "sdk_error": load_error,
            "is_initialized": is_device_initialized,
            "is_device_connected": device_connected,
            "device": device_info
        }), (200 if sdk_loaded else 500)

@app.route("/connect", methods=["POST"])
def connect_device():
    """Initializes and connects to the physical MFS100 scanner."""
    global is_device_initialized
    if not sdk_loaded or not mfs_instance:
        return jsonify({
            "success": False,
            "error": f"MFS100 SDK is not loaded: {load_error}"
        }), 500

    with scanner_lock:
        try:
            ret = mfs_instance.Init()
            err_msg = str(mfs_instance.GetErrorMsg(ret)) if hasattr(mfs_instance, "GetErrorMsg") else ""
            if ret == 0:
                is_device_initialized = True
                device_info = _get_device_dict()
                logger.info(f"Scanner initialized successfully: {device_info}")
                return jsonify({
                    "success": True,
                    "message": "Scanner initialized successfully",
                    "device": device_info
                })
            else:
                is_device_initialized = False
                logger.warning(f"Scanner initialization failed: code {ret} - {err_msg}")
                return jsonify({
                    "success": False,
                    "error": err_msg or f"Init failed with code {ret}",
                    "error_code": int(ret)
                }), 400
        except Exception as e:
            is_device_initialized = False
            logger.exception("Exception during scanner connect")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

@app.route("/disconnect", methods=["POST"])
def disconnect_device():
    """Uninitializes the physical MFS100 scanner."""
    global is_device_initialized
    if not sdk_loaded or not mfs_instance:
        return jsonify({"success": True, "message": "SDK was not loaded"})

    with scanner_lock:
        try:
            mfs_instance.Uninit()
        except Exception as e:
            logger.warning(f"Error during Uninit: {e}")
        finally:
            is_device_initialized = False

    logger.info("Scanner disconnected / uninitialized.")
    return jsonify({
        "success": True,
        "message": "Scanner disconnected successfully"
    })

@app.route("/device-info", methods=["GET"])
def device_info():
    """Retrieves hardware specs and serial number."""
    with scanner_lock:
        if not sdk_loaded or not mfs_instance:
            return jsonify({"success": False, "error": "SDK not loaded"}), 500
        if not is_device_initialized:
            return jsonify({"success": False, "error": "Scanner is not initialized"}), 400

        info = _get_device_dict()
        if info:
            return jsonify({"success": True, "device": info})
        return jsonify({"success": False, "error": "Unable to read device info"}), 500

@app.route("/capture", methods=["POST"])
def capture():
    """
    Triggers biometric auto-capture on the physical scanner.
    Waits for finger placement up to timeout_ms.
    Returns:
        JSON with base64 encoded BMP image, quality, nfiq, and dimensions.
    """
    global is_device_initialized
    if not sdk_loaded or not mfs_instance:
        return jsonify({
            "success": False,
            "error": f"MFS100 SDK not loaded: {load_error}"
        }), 500

    req_data = request.get_json(silent=True) or {}
    timeout_ms = int(req_data.get("timeout_ms", DEFAULT_TIMEOUT_MS))
    quality_threshold = int(req_data.get("quality", DEFAULT_QUALITY))

    with scanner_lock:
        # Auto-initialize if not yet done
        if not is_device_initialized:
            ret = mfs_instance.Init()
            if ret == 0:
                is_device_initialized = True
            else:
                err_msg = str(mfs_instance.GetErrorMsg(ret))
                return jsonify({
                    "success": False,
                    "error": f"Failed to initialize scanner: {err_msg}",
                    "error_code": int(ret)
                }), 400

        try:
            fd = FingerDataClass()
            logger.info(f"Starting AutoCapture (timeout={timeout_ms}ms, quality_thresh={quality_threshold})...")
            res = mfs_instance.AutoCapture(fd, timeout_ms, False, True)

            # pythonnet ByRef return handling
            if isinstance(res, tuple):
                ret_code = res[0]
                fd_out = res[1] if (len(res) > 1 and res[1] is not None) else fd
            else:
                ret_code = res
                fd_out = fd

            if ret_code == 0 and fd_out is not None and fd_out.FingerImage is not None:
                # Convert System.Drawing.Bitmap to BMP bytes using SDK helper
                bmp_system_bytes = mfs_instance.BitmapToBytes(fd_out.FingerImage)
                bmp_bytes = bytes(bmp_system_bytes)

                quality = int(fd_out.Quality)
                nfiq = int(fd_out.Nfiq)
                image_b64 = base64.b64encode(bmp_bytes).decode("utf-8")

                dev_info = _get_device_dict() or {}
                width = dev_info.get("width", 316)
                height = dev_info.get("height", 354)

                logger.info(f"Capture successful! Quality={quality}, NFIQ={nfiq}, Size={len(bmp_bytes)} bytes")
                return jsonify({
                    "success": True,
                    "image_b64": image_b64,
                    "quality": quality,
                    "nfiq": nfiq,
                    "width": width,
                    "height": height,
                    "format": "BMP"
                })
            else:
                err_msg = str(mfs_instance.GetErrorMsg(ret_code)) if hasattr(mfs_instance, "GetErrorMsg") else f"Code {ret_code}"
                logger.warning(f"Capture did not succeed: code {ret_code} - {err_msg}")
                return jsonify({
                    "success": False,
                    "error": err_msg,
                    "error_code": int(ret_code)
                }), 400

        except Exception as e:
            logger.exception("Exception during AutoCapture")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

def main():
    logger.info(f"Starting MFS100 32-bit Bridge on http://{BRIDGE_HOST}:{BRIDGE_PORT}")
    app.run(host=BRIDGE_HOST, port=BRIDGE_PORT, threaded=True)

if __name__ == "__main__":
    main()
