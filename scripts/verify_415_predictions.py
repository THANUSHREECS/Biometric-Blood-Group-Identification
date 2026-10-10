import os
import sys
import pandas as pd
import numpy as np

# Load predictor
from src.predictor import BloodGroupPredictor
predictor = BloodGroupPredictor()

# Load old predictions
df_old = pd.read_csv("outputs/abo_positive_4class/test_predictions.csv")

# Filter to just 415 test images if this CSV has more? Let's check size
print(f"Old predictions size: {len(df_old)}")

max_diff = 0.0
identical_top1 = 0
total = 0

classes = predictor.class_names
diffs = []
for idx, row in df_old.iterrows():
    img_path = row["image_path"]
    if not os.path.exists(img_path):
        continue
    
    # Predict using REAL predictor path
    res = predictor.predict(img_path)
    probs = res["probabilities"]
    
    # Compare
    for c in classes:
        old_p = row[f"prob_{c}"]
        new_p = probs[c]
        diff = abs(old_p - new_p)
        if diff > max_diff:
            max_diff = diff
        diffs.append(diff)
        
    old_top1 = row["predicted_class"]
    new_top1 = res["blood_group"]
    if old_top1 == new_top1:
        identical_top1 += 1
    total += 1

print(f"Total compared: {total}")
print(f"Max abs probability diff: {max_diff}")
print(f"Mean abs probability diff: {np.mean(diffs)}")
print(f"% Identical Top-1: {(identical_top1/max(1, total))*100:.2f}%")
