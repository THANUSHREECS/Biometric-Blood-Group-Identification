import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

# Data paths
final_data_root = os.path.join(project_root, "data", "mfs100_final", "raw")
model_path = os.path.join(project_root, "models", "day27_full_dataset", "four_class", "best_model.keras")
reports_dir = os.path.join(project_root, "reports", "final_mfs100")
os.makedirs(reports_dir, exist_ok=True)

# Load model
model = tf.keras.models.load_model(model_path)
input_shape = model.input_shape

print(f"Model loaded: {model_path}")
print(f"Model input shape: {input_shape}")
print(f"Preprocessing function: C5 (composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target)")
print(f"Preprocessing source file: src/preprocessing.py")

# Classes
classes = ["A+", "AB+", "B+", "O+"]
class_mapping = {i: c for i, c in enumerate(classes)}
rev_class_mapping = {c: i for i, c in enumerate(classes)}

# Load and process data
participants = []
scan_data = []

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

for bg in classes:
    bg_dir = os.path.join(final_data_root, bg)
    if not os.path.exists(bg_dir): continue
    for pid in os.listdir(bg_dir):
        p_dir = os.path.join(bg_dir, pid)
        if os.path.isdir(p_dir):
            scans = sorted(glob.glob(os.path.join(p_dir, "*.bmp")))
            if len(scans) != 3:
                continue
                
            participant_probs = []
            scan_preds = []
            
            for i, scan_path in enumerate(scans):
                img_array = transform_c5(scan_path)
                img_tensor = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_tensor, verbose=0)[0]
                
                scan_pred_class = class_mapping[np.argmax(preds)]
                scan_conf = float(np.max(preds))
                
                scan_row = {
                    "participant_id": pid,
                    "scan_number": i + 1,
                    "true_label": bg,
                    "predicted_class": scan_pred_class,
                    "confidence": scan_conf,
                    "prob_A_plus": float(preds[0]),
                    "prob_AB_plus": float(preds[1]),
                    "prob_B_plus": float(preds[2]),
                    "prob_O_plus": float(preds[3])
                }
                scan_data.append(scan_row)
                scan_preds.append(scan_row)
                participant_probs.append(preds)
            
            avg_probs = np.mean(participant_probs, axis=0)
            final_pred_idx = np.argmax(avg_probs)
            final_pred_class = class_mapping[final_pred_idx]
            final_conf = float(avg_probs[final_pred_idx])
            
            # Agreement
            p_preds = [s["predicted_class"] for s in scan_preds]
            agree_count = p_preds.count(final_pred_class)
            
            p_row = {
                "participant_id": pid,
                "true_label": bg,
                "scan1_prediction": p_preds[0],
                "scan1_confidence": scan_preds[0]["confidence"],
                "scan2_prediction": p_preds[1],
                "scan2_confidence": scan_preds[1]["confidence"],
                "scan3_prediction": p_preds[2],
                "scan3_confidence": scan_preds[2]["confidence"],
                "final_prediction": final_pred_class,
                "final_confidence": final_conf,
                "agreement_count": agree_count,
                "agreement_level": f"{agree_count}/3",
                "prob_A_plus": float(avg_probs[0]),
                "prob_AB_plus": float(avg_probs[1]),
                "prob_B_plus": float(avg_probs[2]),
                "prob_O_plus": float(avg_probs[3])
            }
            participants.append(p_row)

# Save scan-level CSV
scan_df = pd.DataFrame(scan_data)
scan_df.to_csv(os.path.join(reports_dir, "final_zero_shot_scan_predictions.csv"), index=False)

# Save participant-level CSV
p_df = pd.DataFrame(participants)
p_df.to_csv(os.path.join(reports_dir, "final_zero_shot_participant_predictions.csv"), index=False)

# Calculate primary metrics
y_true_p = p_df["true_label"].values
y_pred_p = p_df["final_prediction"].values

p_acc = accuracy_score(y_true_p, y_pred_p)
p_macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")

# O+ prediction share
o_plus_preds = (y_pred_p == "O+").sum()
o_plus_share = o_plus_preds / len(y_pred_p)

# Per-class metrics
class_metrics = []
report = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
for cls in classes:
    if cls in report:
        class_metrics.append({
            "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"]
        })
    else:
        class_metrics.append({
            "class": cls,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": sum(y_true_p == cls)
        })

cm_df = pd.DataFrame(class_metrics)
cm_df.to_csv(os.path.join(reports_dir, "final_zero_shot_class_metrics.csv"), index=False)

# Confusion matrices
cm = confusion_matrix(y_true_p, y_pred_p, labels=classes)
cm_norm = confusion_matrix(y_true_p, y_pred_p, labels=classes, normalize='true')

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix (Participant-level)")
plt.savefig(os.path.join(reports_dir, "final_zero_shot_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Normalized Confusion Matrix (Participant-level)")
plt.savefig(os.path.join(reports_dir, "final_zero_shot_confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
plt.close()

# Agreement counts
agree_3 = sum(p_df["agreement_count"] == 3)
agree_2 = sum(p_df["agreement_count"] == 2)
agree_1 = sum(p_df["agreement_count"] == 1)

# Bootstrap
np.random.seed(42)
n_iterations = 10000
boot_accs = []
boot_f1s = []

n_size = len(p_df)
for i in range(n_iterations):
    indices = np.random.randint(0, n_size, n_size)
    y_true_boot = y_true_p[indices]
    y_pred_boot = y_pred_p[indices]
    
    boot_accs.append(accuracy_score(y_true_boot, y_pred_boot))
    boot_f1s.append(f1_score(y_true_boot, y_pred_boot, average='macro'))

acc_mean = np.mean(boot_accs)
acc_lower = np.percentile(boot_accs, 2.5)
acc_upper = np.percentile(boot_accs, 97.5)

f1_mean = np.mean(boot_f1s)
f1_lower = np.percentile(boot_f1s, 2.5)
f1_upper = np.percentile(boot_f1s, 97.5)

boot_results = {
    "accuracy_mean": acc_mean,
    "accuracy_ci_lower": acc_lower,
    "accuracy_ci_upper": acc_upper,
    "macro_f1_mean": f1_mean,
    "macro_f1_ci_lower": f1_lower,
    "macro_f1_ci_upper": f1_upper,
    "n_participants": n_size,
    "bootstrap_iterations": n_iterations,
    "seed": 42
}
with open(os.path.join(reports_dir, "final_zero_shot_bootstrap.json"), "w") as f:
    json.dump(boot_results, f, indent=4)

# Output summary to file to read later
with open(os.path.join(reports_dir, "summary.txt"), "w") as f:
    f.write(f"{n_size}\n")
    f.write(f"{len(scan_df)}\n")
    f.write(f"{p_acc*100:.2f}\n")
    f.write(f"{p_macro_f1:.4f}\n")
    f.write(f"{cm_df[cm_df['class']=='A+']['recall'].values[0]*100:.2f}\n")
    f.write(f"{cm_df[cm_df['class']=='B+']['recall'].values[0]*100:.2f}\n")
    f.write(f"{cm_df[cm_df['class']=='AB+']['recall'].values[0]*100:.2f}\n")
    f.write(f"{cm_df[cm_df['class']=='O+']['recall'].values[0]*100:.2f}\n")
    f.write(f"{o_plus_share*100:.2f}\n")
    f.write(f"{agree_3}\n")
    f.write(f"{agree_2}\n")
    f.write(f"{agree_1}\n")
    f.write(f"{acc_lower*100:.2f}\n")
    f.write(f"{acc_upper*100:.2f}\n")
    f.write(f"{f1_lower:.4f}\n")
    f.write(f"{f1_upper:.4f}\n")
