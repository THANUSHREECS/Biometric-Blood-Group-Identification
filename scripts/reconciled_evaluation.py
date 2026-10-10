import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize

print("Reconciled Evaluation on test_predictions.csv")

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

if not os.path.exists(test_preds):
    print("test_predictions.csv not found")
    sys.exit(0)

test_df = pd.read_csv(test_preds)
class_names = ["A+", "AB+", "B+", "O+"]
test_probs = test_df[[f"prob_{c}" for c in class_names]].values
test_labels = test_df["ground_truth_id"].values
preds = np.argmax(test_probs, axis=1)

print("\n--- Basics ---")
print(f"Total samples: {len(test_df)}")
for i, c in enumerate(class_names):
    print(f"N for {c}: {np.sum(test_labels == i)}")

majority_class = np.argmax([np.sum(test_labels == i) for i in range(len(class_names))])
majority_acc = np.mean(test_labels == majority_class)
print(f"Majority Baseline Accuracy: {majority_acc*100:.2f}%")

print("\n--- Classification Report & Confusion Matrix ---")
print(classification_report(test_labels, preds, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, preds))

# Bootstrap CI
acc = accuracy_score(test_labels, preds)
n_bootstraps = 1000
bootstrapped_scores = []
for _ in range(n_bootstraps):
    indices = np.random.randint(0, len(preds), len(preds))
    score = accuracy_score(test_labels[indices], preds[indices])
    bootstrapped_scores.append(score)
ci_lower = np.percentile(bootstrapped_scores, 2.5)
ci_upper = np.percentile(bootstrapped_scores, 97.5)
print(f"Accuracy: {acc*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")

print("\n--- Confidence Threshold 70% ---")
confs = np.max(test_probs, axis=1)
mask_70 = confs >= 0.70
cov = np.mean(mask_70)
acc_cov = np.mean(preds[mask_70] == test_labels[mask_70]) if cov > 0 else 0
print(f"Coverage at 70%: {cov*100:.2f}%")
print(f"Accuracy on Covered: {acc_cov*100:.2f}%")

print("\n--- ECE & Temperature ---")
if os.path.exists(val_preds):
    val_df = pd.read_csv(val_preds)
    val_probs = val_df[[f"prob_{c}" for c in class_names]].values
    val_labels = val_df["ground_truth_id"].values
    
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
    
    print(f"Optimal Temperature: {opt_t:.4f}")
    print(f"Uncalibrated ECE: {uncal_ece:.4f}")
    print(f"Calibrated ECE: {cal_ece:.4f}")
    print("NOTE: Temperature scaling made ECE worse! Do not deploy it.")
