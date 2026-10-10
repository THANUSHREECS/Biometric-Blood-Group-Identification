import requests
import time

base_url = "http://localhost:8000/api/mfs100"

print("Starting session...")
res = requests.post(f"{base_url}/start-session").json()
sid = res['session_id']
print(f"Session ID: {sid}")

print("Please place LEFT THUMB on scanner. Capturing...")

max_retries = 10
cap = None
for i in range(max_retries):
    cap = requests.post(f"{base_url}/capture", json={"session_id": sid, "scan_number": 1}).json()
    if 'image_path' in cap:
        print("Capture successful:", cap)
        break
    else:
        print(f"Attempt {i+1} failed: {cap}. Retrying in 2 seconds...")
        time.sleep(2)

if cap and 'image_path' in cap:
    print("Processing...")
    proc = requests.post(f"{base_url}/process", json={"session_id": sid, "scan_number": 1, "image_path": cap['image_path']}).json()
    print("Process response:", proc)
else:
    print("Could not capture image after multiple attempts.")
