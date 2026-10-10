import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import json

part_df = pd.read_csv("data/mfs100_final/metadata/participants.csv")
log_df = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
merged = pd.merge(log_df, part_df, on="participant_id", how="left")

# Only 4 classes
classes = ["A+", "B+", "AB+", "O+"]
merged = merged[merged["blood_group"].isin(classes)]

from src.predictor import BloodGroupPredictor
predictor = BloodGroupPredictor()

groups = merged.groupby("participant_id")
y_true = []
y_pred = []

for pid, group in groups:
    # We should have 3 images
    paths = group["image_path"].tolist()
    if len(paths) == 3:
        y_true.append(group["blood_group"].iloc[0])
        res = predictor.predict_consensus(paths)
        y_pred.append(res["blood_group"])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("--- Live Validation (Participant Level) ---")
print(f"Total Participants: {len(y_true)}")
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_true, y_pred, labels=classes)
print("Confusion Matrix:")
print(cm)

print("Predicted Class Counts:")
preds, counts = np.unique(y_pred, return_counts=True)
print(dict(zip(preds, counts)))

recall = recall_score(y_true, y_pred, labels=classes, average=None)
print("Per-class Recall:")
print(dict(zip(classes, recall)))

# Constant-class baselines
majority = pd.Series(y_true).value_counts().index[0]
const_acc = accuracy_score(y_true, [majority]*len(y_true))
print(f"Majority Baseline: {const_acc*100:.2f}%")

# Prior-matched chance
counts = pd.Series(y_true).value_counts(normalize=True)
chance = np.sum(counts**2)
print(f"Prior-matched chance accuracy: {chance*100:.2f}%")

# Bootstrap
np.random.seed(42)
boot_accs = []
n = len(y_true)
for _ in range(1000):
    idx = np.random.choice(n, size=n, replace=True)
    boot_accs.append(accuracy_score(y_true[idx], y_pred[idx]))
print(f"Participant-level Bootstrap 95% CI: [{np.percentile(boot_accs, 2.5)*100:.2f}%, {np.percentile(boot_accs, 97.5)*100:.2f}%]")
