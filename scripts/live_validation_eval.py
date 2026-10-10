import os
import sys
import numpy as np
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, log_loss
)
from scipy.stats import binomtest
from scipy.stats import norm

def wilson_score_interval(successes, n, confidence=0.95):
    if n == 0:
        return 0, 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    center_adjusted_prob = p + z**2 / (2 * n)
    adjusted_std_dev = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (center_adjusted_prob - z * adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + z * adjusted_std_dev) / denominator
    return lower_bound, upper_bound
from src.predictor import BloodGroupPredictor
from src.preprocessing import transform_c5

print("--- LIVE VALIDATION EVALUATION ---")

part_path = "data/mfs100_final/metadata/participants.csv"
log_path = "data/mfs100_final/metadata/collection_log.csv"

if not os.path.exists(part_path) or not os.path.exists(log_path):
    print("Manifest files not found.")
    sys.exit(1)

part_df = pd.read_csv(part_path)
log_df = pd.read_csv(log_path)

# Merge
df = pd.merge(log_df, part_df, on="participant_id", how="left")

# Missing info check
missing_cols = []
for col in ["participant_id", "scan_number", "image_path", "blood_group", "finger", "verification_source"]:
    if col not in df.columns:
        missing_cols.append(col)

print("A1. Dataset Profile")
num_participants = df["participant_id"].nunique()
num_scans = len(df)
print(f"Participants: {num_participants}")
print(f"Scans: {num_scans}")
scans_per_part = df.groupby("participant_id")["scan_number"].count()
print(f"Scans per participant: min={scans_per_part.min()}, max={scans_per_part.max()}, mean={scans_per_part.mean():.2f}")

print("\nClass Counts (Scans):")
print(df["blood_group"].value_counts())

has_rh_neg = any("-" in str(bg) for bg in df["blood_group"].unique())
print(f"Any Rh-negative?: {has_rh_neg}")

has_non_left_thumb = any(str(f).upper() != "LEFT THUMB" for f in df["finger"].unique())
print(f"Any non-left-thumb?: {has_non_left_thumb}")

if missing_cols:
    print(f"Missing Information: {', '.join(missing_cols)}")
else:
    print("No required columns missing.")

# Filter to 4-class for evaluation subset
df_eval = df[df["blood_group"].isin(["A+", "B+", "AB+", "O+"])].copy()
df_eval.reset_index(drop=True, inplace=True)

# 2. Extract Contrast/Intensity and run predictor
print("\nExtracting metrics and predicting...")
predictor = BloodGroupPredictor(use_ensemble=True)
intensities = []
contrasts = []
confs = []
preds = []
prob_A = []
prob_B = []
prob_AB = []
prob_O = []
nfiq_list = [] # Assuming NFIQ not available unless in CSV
batch_size = 32
import math
for i in range(math.ceil(len(df_eval) / batch_size)):
    batch = df_eval.iloc[i*batch_size : (i+1)*batch_size]
    batch_scans = []
    
    for idx, row in batch.iterrows():
        p = row["image_path"]
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                intensities.append(np.mean(img))
                contrasts.append(np.std(img))
            else:
                intensities.append(0)
                contrasts.append(0)
                
            arr, _, _ = transform_c5(p)
            batch_scans.append([arr, arr, arr])
        else:
            intensities.append(0)
            contrasts.append(0)
            batch_scans.append([np.zeros((112,112,3)), np.zeros((112,112,3)), np.zeros((112,112,3))])
            
    X_input = np.array(batch_scans)
    all_probs = []
    for cp in predictor.checkpoint_paths:
        predictor.model.load_weights(cp)
        out = predictor.model.predict(X_input, verbose=0)
        all_probs.append(out[0])
        
    mean_probs = np.mean(all_probs, axis=0) # shape (batch, 4)
    
    for j in range(len(batch)):
        best_idx = np.argmax(mean_probs[j])
        preds.append(predictor.class_names[best_idx])
        confs.append(mean_probs[j][best_idx])
        prob_A.append(mean_probs[j][0])
        prob_AB.append(mean_probs[j][1])
        prob_B.append(mean_probs[j][2])
        prob_O.append(mean_probs[j][3])


df_eval["intensity"] = intensities
df_eval["contrast"] = contrasts
df_eval["pred"] = preds
df_eval["conf"] = confs
df_eval["prob_A+"] = prob_A
df_eval["prob_AB+"] = prob_AB
df_eval["prob_B+"] = prob_B
df_eval["prob_O+"] = prob_O

# 3. Per participant aggregation
part_df_eval = df_eval.groupby("participant_id").agg({
    "blood_group": "first",
    "prob_A+": "mean",
    "prob_AB+": "mean",
    "prob_B+": "mean",
    "prob_O+": "mean",
    "intensity": "mean",
    "contrast": "mean"
}).reset_index()

part_preds, part_confs = [], []
for i, row in part_df_eval.iterrows():
    probs = [row["prob_A+"], row["prob_AB+"], row["prob_B+"], row["prob_O+"]]
    best_idx = np.argmax(probs)
    part_preds.append(["A+", "AB+", "B+", "O+"][best_idx])
    part_confs.append(probs[best_idx])
part_df_eval["pred"] = part_preds
part_df_eval["conf"] = part_confs

# 4. Metrics
print("\n--- METRICS (Scans) ---")
y_true = df_eval["blood_group"]
y_pred = df_eval["pred"]

n_per_class = y_true.value_counts()
majority_class = n_per_class.idxmax()
majority_acc = n_per_class.max() / len(y_true)
print(f"Majority Baseline: {majority_acc*100:.2f}% ({majority_class})")

acc = accuracy_score(y_true, y_pred)
ci_low, ci_high = wilson_score_interval(sum(y_true == y_pred), len(y_true))
print(f"Accuracy: {acc*100:.2f}% (Wilson 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%])")

bal_acc = balanced_accuracy_score(y_true, y_pred)
mac_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Balanced Accuracy: {bal_acc*100:.2f}%")
print(f"Macro F1: {mac_f1:.4f}")

# Top-2
top2_correct = 0
probs_matrix = df_eval[["prob_A+", "prob_AB+", "prob_B+", "prob_O+"]].values
class_list = ["A+", "AB+", "B+", "O+"]
for i, row in df_eval.iterrows():
    top2_idx = np.argsort(probs_matrix[i])[-2:]
    top2_classes = [class_list[idx] for idx in top2_idx]
    if row["blood_group"] in top2_classes:
        top2_correct += 1
print(f"Top-2 Accuracy: {top2_correct/len(y_true)*100:.2f}%")

# Log loss
# Map true to indices
y_true_idx = [class_list.index(y) for y in y_true]
ll = log_loss(y_true_idx, probs_matrix, labels=[0,1,2,3])
print(f"Log Loss: {ll:.4f}")

# 70% threshold
mask_70 = df_eval["conf"] >= 0.70
cov = mask_70.mean()
acc_70 = accuracy_score(y_true[mask_70], y_pred[mask_70]) if cov > 0 else 0
print(f"Coverage at 70%: {cov*100:.2f}%")
print(f"Accuracy on Covered: {acc_70*100:.2f}%")

# Binomial test
p_maj = binomtest(sum(y_true == y_pred), len(y_true), p=majority_acc, alternative='greater').pvalue
p_25 = binomtest(sum(y_true == y_pred), len(y_true), p=0.25, alternative='greater').pvalue
print(f"Binomial p-value vs Majority: {p_maj:.4f}")
print(f"Binomial p-value vs Random (25%): {p_25:.4f}")

# Plot
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.scatter(df_eval["contrast"], df_eval["conf"], alpha=0.5, c=(y_true==y_pred))
plt.title("Confidence vs Contrast (Yellow=Correct)")
plt.xlabel("Contrast (Std Dev)")
plt.ylabel("Confidence")

plt.subplot(122)
df_eval["pred"].value_counts().plot(kind='bar')
plt.title("Predicted Class Distribution")
plt.tight_layout()
plt.savefig("results/live_validation_figs.png")

# Save results
out_json = {
    "num_participants": int(num_participants),
    "num_scans": int(num_scans),
    "accuracy": acc,
    "ci_lower": ci_low,
    "ci_upper": ci_high,
    "balanced_accuracy": bal_acc,
    "macro_f1": mac_f1,
    "top2_accuracy": top2_correct/len(y_true),
    "log_loss": ll,
    "majority_baseline": majority_acc,
    "p_value_maj": p_maj,
    "p_value_25": p_25
}
with open("results/live_validation_results.json", "w") as f:
    json.dump(out_json, f, indent=2)
print("\nSaved live_validation_results.json and live_validation_figs.png")
