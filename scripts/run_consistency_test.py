import time
import csv
import os
import requests
import numpy as np

print("=========================================")
print("  CONSISTENCY & LIVE VALIDATION TOOL")
print("=========================================")
print("NOTE: This tool communicates with the live Flask backend and real MFS100 scanner.")
print("Make sure 'python -m src.mfs100_bridge.bridge' and 'python app.py' are running.")

mode = input("Select Mode (1: Consistency Test, 2: Live Validation): ")

os.makedirs("results", exist_ok=True)
API_URL = "http://127.0.0.1:5000/api/capture"

if mode == "1":
    print("Running 10-scan consistency test on Left Thumb...")
    log_path = "results/consistency_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scan_ID", "Prob_A+", "Prob_AB+", "Prob_B+", "Prob_O+", "Top_Class"])
        
        preds = []
        for i in range(10):
            input(f"\nPlace thumb on scanner and press Enter for Scan {i+1}/10...")
            try:
                # We request a hardware scan by not uploading a file
                res = requests.post(API_URL, data={'source': 'hardware'})
                if res.status_code == 200:
                    data = res.json()
                    probs = data.get("probabilities", {})
                    top_class = data.get("blood_group", "Unknown")
                    writer.writerow([i+1, probs.get("A+", 0), probs.get("AB+", 0), probs.get("B+", 0), probs.get("O+", 0), top_class])
                    preds.append(top_class)
                    print(f"Scan {i+1} success: {top_class}")
                else:
                    print(f"Scan failed: {res.text}")
            except Exception as e:
                print(f"Error: {e}")
                
    print(f"\nLog saved to {log_path}")
    if preds:
        counts = {c: preds.count(c) for c in set(preds)}
        top_overall = max(counts, key=counts.get)
        print(f"Summary: % of scans agreeing with top class ({top_overall}): {counts[top_overall]/len(preds) * 100:.1f}%")

elif mode == "2":
    print("Live Validation Logging Mode")
    part_id = input("Enter anonymous participant ID: ")
    true_bg = input("Enter true blood group (A+, B+, AB+, O+): ")
    print("Please capture 3 scans of the left thumb...")
    
    log_path = "results/live_validation.csv"
    file_exists = os.path.exists(log_path)
    
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Participant_ID", "True_BG", "Predicted_BG", "Prob_A+", "Prob_B+", "Prob_AB+", "Prob_O+"])
        
        input("Place thumb on scanner and press Enter to trigger 3-scan ensemble...")
        res = requests.post(API_URL, data={'source': 'hardware'})
        if res.status_code == 200:
            data = res.json()
            probs = data.get("probabilities", {})
            pred = data.get("blood_group", "Unknown")
            writer.writerow([part_id, true_bg, pred, probs.get("A+",0), probs.get("B+",0), probs.get("AB+",0), probs.get("O+",0)])
            print(f"Saved to {log_path}. Predicted: {pred}")
        else:
            print("API Error.")
else:
    print("Invalid mode.")
