import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

# Paths
data_dir = os.path.join(project_root, "data", "mfs100_final", "raw")
output_dir = os.path.join(project_root, "reports", "phase3_participant_benchmark")
model_path = os.path.join(project_root, "models", "day27_full_dataset", "four_class", "best_model.keras")

os.makedirs(output_dir, exist_ok=True)

classes = ["A+", "B+", "AB+", "O+"]
class_mapping = {i: c for i, c in enumerate(classes)}

# 1. IDENTIFY PARTICIPANTS
participants = []
for bg in classes:
    bg_dir = os.path.join(data_dir, bg)
    if not os.path.exists(bg_dir): continue
    for pid in os.listdir(bg_dir):
        p_dir = os.path.join(bg_dir, pid)
        if os.path.isdir(p_dir):
            scans = sorted(glob.glob(os.path.join(p_dir, "*.bmp")))
            if len(scans) == 3:
                participants.append({
                    "participant_id": pid,
                    "blood_group": bg,
                    "scans": scans
                })

df_p = pd.DataFrame(participants)

# Verify
total_participants = len(df_p)
total_images = sum(len(p["scans"]) for p in participants)
class_counts = df_p["blood_group"].value_counts().to_dict()

# 2 & 3. CREATE PARTICIPANT-LEVEL SPLIT
dev_df, test_df = train_test_split(df_p, test_size=0.20, random_state=42, stratify=df_p["blood_group"])
train_df, val_df = train_test_split(dev_df, test_size=0.20, random_state=42, stratify=dev_df["blood_group"])

split_assignments = {}
for pid in train_df["participant_id"]: split_assignments[pid] = "TRAIN"
for pid in val_df["participant_id"]: split_assignments[pid] = "VALIDATION"
for pid in test_df["participant_id"]: split_assignments[pid] = "TEST"

# 4. SAVE SPLITS
split_rows = []
for p in participants:
    pid = p["participant_id"]
    bg = p["blood_group"]
    split = split_assignments[pid]
    for scan in p["scans"]:
        split_rows.append({
            "participant_id": pid,
            "blood_group": bg,
            "scan_path": scan,
            "split": split
        })
split_csv = pd.DataFrame(split_rows)
split_csv.to_csv(os.path.join(output_dir, "participant_split.csv"), index=False)

split_stats = {
    "TRAIN": {"participants": len(train_df), "images": len(train_df)*3, "classes": train_df["blood_group"].value_counts().to_dict()},
    "VALIDATION": {"participants": len(val_df), "images": len(val_df)*3, "classes": val_df["blood_group"].value_counts().to_dict()},
    "TEST": {"participants": len(test_df), "images": len(test_df)*3, "classes": test_df["blood_group"].value_counts().to_dict()}
}
with open(os.path.join(output_dir, "split_statistics.json"), "w") as f:
    json.dump(split_stats, f, indent=4)

# 5. LEAKAGE AUDIT
leakage_issues = []
train_set = set(train_df["participant_id"])
val_set = set(val_df["participant_id"])
test_set = set(test_df["participant_id"])

if len(train_set.intersection(val_set)) > 0: leakage_issues.append("Train and Val overlap")
if len(train_set.intersection(test_set)) > 0: leakage_issues.append("Train and Test overlap")
if len(val_set.intersection(test_set)) > 0: leakage_issues.append("Val and Test overlap")

scan_splits = {}
for row in split_rows:
    scan = row["scan_path"]
    if scan in scan_splits and scan_splits[scan] != row["split"]:
        leakage_issues.append(f"Scan {scan} in multiple splits")
    scan_splits[scan] = row["split"]

with open(os.path.join(output_dir, "LEAKAGE_AUDIT.md"), "w") as f:
    f.write("# LEAKAGE AUDIT\n\n")
    if len(leakage_issues) == 0:
        f.write("Status: PASS\n\n")
        f.write("- No participant appears in more than one split.\n")
        f.write("- No exact image appears in more than one split.\n")
        f.write("- All 3 scans from each participant are in one split.\n")
        f.write("- No test participant appears in train or validation.\n\n")
    else:
        f.write("Status: FAIL\n\n")
        for issue in leakage_issues:
            f.write(f"- {issue}\n")
    f.write("## Participant Counts\n")
    f.write(f"TRAIN participants: {len(train_df)}\n")
    f.write(f"VALIDATION participants: {len(val_df)}\n")
    f.write(f"TEST participants: {len(test_df)}\n\n")
    f.write("## Per-Class Participant Counts\n")
    f.write(f"TRAIN: {split_stats['TRAIN']['classes']}\n")
    f.write(f"VALIDATION: {split_stats['VALIDATION']['classes']}\n")
    f.write(f"TEST: {split_stats['TEST']['classes']}\n")

# 6. BASELINE EVALUATION ON TEST
def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

model = tf.keras.models.load_model(model_path)

test_preds = []
for idx, row in test_df.iterrows():
    pid = row["participant_id"]
    bg = row["blood_group"]
    probs = []
    p_classes = []
    for scan in row["scans"]:
        img = transform_c5(scan)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)[0]
        probs.append(pred)
        p_classes.append(class_mapping[np.argmax(pred)])
        
    avg_probs = np.mean(probs, axis=0)
    final_pred_idx = np.argmax(avg_probs)
    final_pred = class_mapping[final_pred_idx]
    final_conf = float(avg_probs[final_pred_idx])
    
    agree = p_classes.count(final_pred)
    
    test_preds.append({
        "participant_id": pid,
        "true_label": bg,
        "final_prediction": final_pred,
        "final_confidence": final_conf,
        "agreement": agree
    })

df_res = pd.DataFrame(test_preds)
y_true = df_res["true_label"].values
y_pred = df_res["final_prediction"].values

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
o_plus_share = sum(y_pred == "O+") / len(y_pred)
mean_conf = df_res["final_confidence"].mean()
a3 = sum(df_res["agreement"] == 3)
a2 = sum(df_res["agreement"] == 2)
a1 = sum(df_res["agreement"] == 1)

# CM
cm = confusion_matrix(y_true, y_pred, labels=classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix (TEST set Participant-level)")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bootstrap
np.random.seed(42)
boot_accs = []
boot_f1s = []
n_size = len(df_res)
for i in range(10000):
    indices = np.random.randint(0, n_size, n_size)
    yt = y_true[indices]
    yp = y_pred[indices]
    boot_accs.append(accuracy_score(yt, yp))
    boot_f1s.append(f1_score(yt, yp, average='macro'))

boot_res = {
    "accuracy_mean": float(np.mean(boot_accs)),
    "accuracy_ci_lower": float(np.percentile(boot_accs, 2.5)),
    "accuracy_ci_upper": float(np.percentile(boot_accs, 97.5)),
    "macro_f1_mean": float(np.mean(boot_f1s)),
    "macro_f1_ci_lower": float(np.percentile(boot_f1s, 2.5)),
    "macro_f1_ci_upper": float(np.percentile(boot_f1s, 97.5))
}
with open(os.path.join(output_dir, "bootstrap_results.json"), "w") as f:
    json.dump(boot_res, f, indent=4)

# 8. OUTPUT REPORTS
with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write("# Phase 3 Participant Benchmark\nContains the official train/val/test split for the 101 participants.\n")

with open(os.path.join(output_dir, "EXPERIMENT0_REPORT.md"), "w") as f:
    f.write("# EXPERIMENT 0: Benchmark Setup\n")
    f.write(f"Total Participants: {total_participants}\n")
    f.write(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n\n")
    f.write("## Test Set Baseline (Day 27 Model)\n")
    f.write(f"- Accuracy: {acc:.4f}\n")
    f.write(f"- Macro F1: {macro_f1:.4f}\n")
    f.write(f"- O+ Share: {o_plus_share:.4f}\n")
    f.write(f"- Mean Conf: {mean_conf:.4f}\n")
    f.write(f"- 3/3 Agreement: {a3}\n")
    f.write(f"- 2/3 Agreement: {a2}\n")
    f.write(f"- 1/3 Agreement: {a1}\n")
    f.write("\n## Per-class Performance:\n")
    for cls in classes:
        if cls in report:
            f.write(f"### {cls}\n")
            f.write(f"- Precision: {report[cls]['precision']:.4f}\n")
            f.write(f"- Recall: {report[cls]['recall']:.4f}\n")
            f.write(f"- F1: {report[cls]['f1-score']:.4f}\n")

print(f"RESULTS_MARKER|{total_participants}|{total_images}|{class_counts}|{len(train_df)}|{len(val_df)}|{len(test_df)}|{acc}|{macro_f1}")
