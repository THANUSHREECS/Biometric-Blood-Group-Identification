import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import norm

def wilson_score_interval(successes, n, confidence=0.95):
    if n == 0: return 0, 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    center_adjusted_prob = p + z**2 / (2 * n)
    adjusted_std_dev = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (center_adjusted_prob - z * adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + z * adjusted_std_dev) / denominator
    return lower_bound, upper_bound

part_df = pd.read_csv("data/mfs100_final/metadata/participants.csv")
log_df = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
df = pd.merge(log_df, part_df, on="participant_id", how="left")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predictor import BloodGroupPredictor
predictor = BloodGroupPredictor(use_ensemble=True)

participants = df['participant_id'].unique()
y_true = []
y_pred = []
o_plus_pred = []
a_plus_pred = []
b_plus_pred = []
ab_plus_pred = []

print("Running participant-level evaluation...")
for pid in participants:
    sub = df[df['participant_id'] == pid]
    paths = sub['image_path'].tolist()
    if len(paths) != 3:
        print(f"Participant {pid} does not have exactly 3 scans.")
        continue
    y_true.append(sub['blood_group'].iloc[0])
    # Call consensus predictor
    res = predictor.predict_consensus(paths)
    y_pred.append(res['blood_group'])
    o_plus_pred.append("O+")
    a_plus_pred.append("A+")
    b_plus_pred.append("B+")
    ab_plus_pred.append("AB+")

# Metrics
acc = accuracy_score(y_true, y_pred)
ci_lower, ci_upper = wilson_score_interval(int(acc * len(y_true)), len(y_true))
cm = confusion_matrix(y_true, y_pred, labels=["A+", "B+", "AB+", "O+"])

print("--- PARTICIPANT LEVEL EVALUATION ---")
print(f"Total Participants: {len(y_true)}")
print(f"Accuracy: {acc*100:.2f}% (Wilson 95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")
print("Confusion Matrix:")
print(cm)

y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
classes = ["A+", "B+", "AB+", "O+"]
for cls in classes:
    idx = y_true_np == cls
    if np.sum(idx) > 0:
        recall = np.sum(y_pred_np[idx] == cls) / np.sum(idx)
        print(f"Recall for {cls}: {recall*100:.2f}%")

print(f"\nConstant Baseline (Always O+): {accuracy_score(y_true, o_plus_pred)*100:.2f}%")
print(f"Constant Baseline (Always A+): {accuracy_score(y_true, a_plus_pred)*100:.2f}%")
print(f"Constant Baseline (Always B+): {accuracy_score(y_true, b_plus_pred)*100:.2f}%")
print(f"Constant Baseline (Always AB+): {accuracy_score(y_true, ab_plus_pred)*100:.2f}%")

# Permutation test
print("\n--- PERMUTATION TEST ---")
np.random.seed(42)
perm_accs = []
for _ in range(1000):
    shuffled = np.random.permutation(y_true_np)
    perm_accs.append(accuracy_score(shuffled, y_pred_np))
p_value = np.sum(np.array(perm_accs) >= acc) / 1000
print(f"Permutation p-value: {p_value}")

# Check with API
print("\n--- API CONSISTENCY CHECK ---")
first_pid = participants[0]
first_paths = df[df['participant_id'] == first_pid]['image_path'].tolist()
print(f"Checking Participant {first_pid} via API...")

# Send 3 files via API? Wait, the API for hardware might not take 3 files from HTTP unless we call /api/capture.
# We will use the script's output to verify the same predictor is used.
import io
url = "http://127.0.0.1:5000/api/upload"
for path in first_paths[:1]:
    with open(path, "rb") as f:
        file_bytes = f.read()
    data = {"file": (path, file_bytes, "image/bmp")}
    try:
        r = requests.post(url, files={"file": (io.BytesIO(file_bytes), "test.bmp")})
        print("API Response:", r.json())
    except Exception as e:
        print("API not running or error:", e)
