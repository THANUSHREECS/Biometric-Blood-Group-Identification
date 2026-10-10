import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

print("Reconciling Numbers...")

def get_ece(probs, labels, num_bins=10):
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == labels)
    ece = 0.0
    bins = np.linspace(0, 1, num_bins+1)
    for i in range(num_bins):
        mask = (confs > bins[i]) & (confs <= bins[i+1])
        if np.sum(mask) > 0:
            ece += np.abs(np.mean(accs[mask]) - np.mean(confs[mask])) * np.sum(mask) / len(probs)
    return ece

val_preds = "outputs/abo_positive_4class/val_predictions.csv"
test_preds = "outputs/abo_positive_4class/test_predictions.csv"

if os.path.exists(val_preds) and os.path.exists(test_preds):
    val_df = pd.read_csv(val_preds)
    test_df = pd.read_csv(test_preds)
    
    class_names = ["A+", "AB+", "B+", "O+"]
    val_probs = val_df[[f"prob_{c}" for c in class_names]].values
    val_labels = val_df["ground_truth_id"].values
    
    test_probs = test_df[[f"prob_{c}" for c in class_names]].values
    test_labels = test_df["ground_truth_id"].values
    
    def temp_loss(t):
        scaled = np.exp(np.log(val_probs + 1e-9) / t[0])
        scaled = scaled / np.sum(scaled, axis=1, keepdims=True)
        return get_ece(scaled, val_labels)
        
    res = minimize(temp_loss, [1.5], bounds=[(0.5, 5.0)])
    opt_t = res.x[0]
    
    test_scaled = np.exp(np.log(test_probs + 1e-9) / opt_t)
    test_scaled = test_scaled / np.sum(test_scaled, axis=1, keepdims=True)
    
    uncal_ece = get_ece(test_probs, test_labels)
    cal_ece = get_ece(test_scaled, test_labels)
    
    print(f"Number of test samples: {len(test_probs)}")
    print(f"Optimal Temperature (Val): {opt_t:.4f}")
    print(f"Uncalibrated Test ECE: {uncal_ece:.4f}")
    print(f"Calibrated Test ECE: {cal_ece:.4f}")
    
    confs = np.max(test_scaled, axis=1)
    mask_70 = confs >= 0.70
    cov = np.mean(mask_70)
    acc = np.mean((np.argmax(test_scaled, axis=1) == test_labels)[mask_70]) if cov > 0 else 0
    print(f"70% Threshold Coverage: {cov*100:.2f}%")
    print(f"Accuracy on Covered Samples: {acc*100:.2f}%")

    # Bootstrap for Majority Baseline (30.6%) vs Metadata Artifact Classifier (34.00%)
    # Let's say random baseline is 25%
    # Permutation testing is done in artifact_baseline.py
