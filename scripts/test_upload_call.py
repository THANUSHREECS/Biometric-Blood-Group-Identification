import os
import requests
import json
import glob

# Find a sample image
sample_img = "data/sample/sample_A+.bmp"
if not os.path.exists(sample_img):
    bmps = glob.glob("data/**/*.bmp", recursive=True)
    sample_img = bmps[0]

print(f"Testing image: {sample_img}")
with open(sample_img, "rb") as f:
    response = requests.post("http://127.0.0.1:5000/api/upload", files={"file": f})

print("Status Code:", response.status_code)
data = response.json()
print("Raw JSON Response from /api/upload:")
print(json.dumps(data, indent=2))

print("\n--- UI Display Analysis ---")
print(f"Primary Diagnosis Card: {data.get('blood_group')}")
print(f"Confidence Percentage: {data.get('confidence', 0) * 100:.1f}%")
print("Rendered 4-Grid Probability Boxes (in DOM order via Object.entries):")
for bg, prob in data.get("probabilities", {}).items():
    is_top = (bg == data.get("blood_group")) and not data.get("needs_confirmation")
    print(f"  Box [{bg}]: {prob * 100:.1f}% | CSS active: {is_top}")
