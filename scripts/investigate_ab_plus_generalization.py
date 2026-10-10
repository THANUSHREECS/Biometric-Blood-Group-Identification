"""
scripts/investigate_ab_plus_generalization.py
Validation-Only Generalization Investigation for Blood Group AB+.

Strict constraints:
- Uses ONLY train and validation data.
- NEVER accesses or evaluates the locked 898-image test set.
- Does not modify Champion Ensemble V2 or its weights.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    DATASET_MANIFEST_PATH, CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS,
    PROJECT_ROOT, CONFIDENCE_THRESHOLD
)
from src.predictor import BloodGroupPredictor
from src.preprocessing import preprocess_for_ensemble

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ab_plus_diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75, flush=True)
print("=== 1. DATASET MANIFEST AUDIT (TRAIN & VALIDATION ONLY) ===", flush=True)
print("=" * 75, flush=True)

df = pd.read_csv(DATASET_MANIFEST_PATH)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "val"].copy()

print(f"Total Training Samples:   {len(train_df)}", flush=True)
print(f"Total Validation Samples: {len(val_df)}", flush=True)
print("LOCKED TEST SET STATUS: STRICTLY EXCLUDED (0 samples loaded/accessed)", flush=True)

print("\nTraining Split Class Counts & Percentages:", flush=True)
train_counts = train_df["class_name"].value_counts()[CLASS_NAMES]
for cls in CLASS_NAMES:
    cnt = int(train_counts[cls])
    pct = (cnt / len(train_df)) * 100
    print(f"  {cls:4s}: {cnt:4d} samples ({pct:5.2f}%)", flush=True)

print("\nValidation Split Class Counts & Percentages:", flush=True)
val_counts = val_df["class_name"].value_counts()[CLASS_NAMES]
for cls in CLASS_NAMES:
    cnt = int(val_counts[cls])
    pct = (cnt / len(val_df)) * 100
    print(f"  {cls:4s}: {cnt:4d} samples ({pct:5.2f}%)", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 2. LOADING CHAMPION ENSEMBLE ENGINE ===", flush=True)
print("=" * 75, flush=True)

t0 = time.time()
predictor = BloodGroupPredictor()
print(f"[+] Loaded Champion Ensemble V2 engine in {time.time() - t0:.2f}s", flush=True)
print(f"[+] Ensemble weights: {predictor.weights}", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 3. RUNNING VALIDATION INFERENCE (ALL 898 VAL SAMPLES) ===", flush=True)
print("=" * 75, flush=True)

results = []
t_start = time.time()
num_val = len(val_df)

for idx, (_, row) in enumerate(val_df.iterrows(), 1):
    img_rel_or_abs = row["image_path"]
    full_path = img_rel_or_abs if os.path.isabs(img_rel_or_abs) else os.path.join(PROJECT_ROOT, img_rel_or_abs)
    true_label = row["class_name"]

    # Preprocess once for submodel breakdown & ensemble
    tensors = preprocess_for_ensemble(full_path, source="upload")

    # Submodel predictions
    p_rn = predictor.resnet_model.predict(tensors["resnet50"], verbose=0)[0]
    p_vg = predictor.vgg_model.predict(tensors["vgg16"], verbose=0)[0]
    p_ef = predictor.eff_model.predict(tensors["efficientnetb0"], verbose=0)[0]

    # Weighted Ensemble
    p_ens = (
        predictor.weights["resnet50"] * p_rn +
        predictor.weights["vgg16"] * p_vg +
        predictor.weights["efficientnetb0"] * p_ef
    )

    pred_idx = int(np.argmax(p_ens))
    pred_label = CLASS_NAMES[pred_idx]
    conf = float(np.max(p_ens))

    rn_label = CLASS_NAMES[int(np.argmax(p_rn))]
    vg_label = CLASS_NAMES[int(np.argmax(p_vg))]
    ef_label = CLASS_NAMES[int(np.argmax(p_ef))]

    # Extract image pixel statistics (for feature analysis)
    img_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    mean_val = float(img_gray.mean())
    std_val = float(img_gray.std())
    lap_sharp = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    rec = {
        "index": idx,
        "filename": os.path.basename(full_path),
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": conf,
        "is_correct": (true_label == pred_label),
        "is_low_conf": (conf < CONFIDENCE_THRESHOLD),
        "rn_pred": rn_label,
        "vg_pred": vg_label,
        "ef_pred": ef_label,
        "p_O_neg": float(p_ens[CLASS_TO_ID["O-"]]),
        "p_O_pos": float(p_ens[CLASS_TO_ID["O+"]]),
        "p_AB_pos": float(p_ens[CLASS_TO_ID["AB+"]]),
        "p_AB_neg": float(p_ens[CLASS_TO_ID["AB-"]]),
        "mean_intensity": mean_val,
        "std_intensity": std_val,
        "laplacian_sharpness": lap_sharp
    }
    results.append(rec)

    if idx % 100 == 0 or idx == num_val:
        elapsed = time.time() - t_start
        print(f"  Processed {idx:3d}/{num_val} ({idx/num_val*100:5.1f}%) in {elapsed:.1f}s...", flush=True)

res_df = pd.DataFrame(results)

print("\n" + "=" * 75, flush=True)
print("=== 4. OVERALL VALIDATION ACCURACY ===", flush=True)
print("=" * 75, flush=True)

overall_ens_acc = (res_df["true_label"] == res_df["pred_label"]).mean() * 100
overall_rn_acc  = (res_df["true_label"] == res_df["rn_pred"]).mean() * 100
overall_vg_acc  = (res_df["true_label"] == res_df["vg_pred"]).mean() * 100
overall_ef_acc  = (res_df["true_label"] == res_df["ef_pred"]).mean() * 100

print(f"Validation Accuracy (898 samples):", flush=True)
print(f"  ResNet50 alone:         {overall_rn_acc:5.2f}%", flush=True)
print(f"  VGG16 alone:            {overall_vg_acc:5.2f}%", flush=True)
print(f"  EfficientNetB0 alone:   {overall_ef_acc:5.2f}%", flush=True)
print(f"  Champion Ensemble V2:   {overall_ens_acc:5.2f}%", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 5. VALIDATION CONFUSION MATRIX (8x8) ===", flush=True)
print("=" * 75, flush=True)

from sklearn.metrics import confusion_matrix

y_true = [CLASS_TO_ID[c] for c in res_df["true_label"]]
y_pred = [CLASS_TO_ID[c] for c in res_df["pred_label"]]

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
print(cm_df.to_string(), flush=True)

cm_csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix_val.csv")
cm_df.to_csv(cm_csv_path)
print(f"\nSaved confusion matrix to: {cm_csv_path}", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 6. FOCUSED AB+ CONFUSION & ERROR ANALYSIS ===", flush=True)
print("=" * 75, flush=True)

ab_df = res_df[res_df["true_label"] == "AB+"].copy()
total_ab = len(ab_df)
ab_correct = (ab_df["pred_label"] == "AB+").sum()
ab_to_o_neg = (ab_df["pred_label"] == "O-").sum()
ab_to_o_pos = (ab_df["pred_label"] == "O+").sum()
ab_to_other = total_ab - ab_correct - ab_to_o_neg - ab_to_o_pos

print(f"Total True AB+ Validation Samples:   {total_ab}", flush=True)
print(f"AB+ Correctly Classified (AB+):       {ab_correct:3d} ({ab_correct/total_ab*100:5.2f}%)", flush=True)
print(f"AB+ Misclassified as O-:              {ab_to_o_neg:3d} ({ab_to_o_neg/total_ab*100:5.2f}%)", flush=True)
print(f"AB+ Misclassified as O+:              {ab_to_o_pos:3d} ({ab_to_o_pos/total_ab*100:5.2f}%)", flush=True)
print(f"Total AB+ Confused with Group O:      {ab_to_o_neg + ab_to_o_pos:3d} ({(ab_to_o_neg + ab_to_o_pos)/total_ab*100:5.2f}%)", flush=True)
print(f"AB+ Misclassified as Other Classes:   {ab_to_other:3d} ({ab_to_other/total_ab*100:5.2f}%)", flush=True)

print("\nComplete Prediction Breakdown for AB+ Validation Samples:", flush=True)
for cls in CLASS_NAMES:
    c_count = int((ab_df["pred_label"] == cls).sum())
    pct = (c_count / total_ab) * 100
    tag = "CORRECT" if cls == "AB+" else ("MAJOR ERROR" if c_count >= 5 else "")
    print(f"  AB+ -> {cls:4s}: {c_count:3d} ({pct:5.2f}%) {tag}", flush=True)

print("\nReverse Confusion: True O- and True O+ misclassified as AB+:", flush=True)
o_neg_df = res_df[res_df["true_label"] == "O-"]
o_pos_df = res_df[res_df["true_label"] == "O+"]
o_neg_to_ab = int((o_neg_df["pred_label"] == "AB+").sum())
o_pos_to_ab = int((o_pos_df["pred_label"] == "AB+").sum())
print(f"  True O- (n={len(o_neg_df)}) misclassified as AB+: {o_neg_to_ab:2d} ({o_neg_to_ab/len(o_neg_df)*100:.2f}%)", flush=True)
print(f"  True O+ (n={len(o_pos_df)}) misclassified as AB+: {o_pos_to_ab:2d} ({o_pos_to_ab/len(o_pos_df)*100:.2f}%)", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 7. PER-MODEL PERFORMANCE SPECIFICALLY ON AB+ ===", flush=True)
print("=" * 75, flush=True)

rn_ab_acc = (ab_df["rn_pred"] == "AB+").mean() * 100
vg_ab_acc = (ab_df["vg_pred"] == "AB+").mean() * 100
ef_ab_acc = (ab_df["ef_pred"] == "AB+").mean() * 100

print(f"AB+ Accuracy / Recall by Individual Sub-Model:", flush=True)
print(f"  ResNet50 (58% weight):       {rn_ab_acc:5.2f}% ({(ab_df['rn_pred'] == 'AB+').sum()}/{total_ab})", flush=True)
print(f"  VGG16 (38% weight):          {vg_ab_acc:5.2f}% ({(ab_df['vg_pred'] == 'AB+').sum()}/{total_ab})", flush=True)
print(f"  EfficientNetB0 (4% weight):  {ef_ab_acc:5.2f}% ({(ab_df['ef_pred'] == 'AB+').sum()}/{total_ab})", flush=True)
print(f"  Champion Ensemble V2:        {ab_correct/total_ab*100:5.2f}% ({ab_correct}/{total_ab})", flush=True)

print("\nAB+ Errors to O- and O+ by Individual Model:", flush=True)
print(f"  ResNet50 -> O-: {(ab_df['rn_pred'] == 'O-').sum():2d}, ResNet50 -> O+: {(ab_df['rn_pred'] == 'O+').sum():2d}", flush=True)
print(f"  VGG16    -> O-: {(ab_df['vg_pred'] == 'O-').sum():2d}, VGG16    -> O+: {(ab_df['vg_pred'] == 'O+').sum():2d}", flush=True)
print(f"  EffNetB0 -> O-: {(ab_df['ef_pred'] == 'O-').sum():2d}, EffNetB0 -> O+: {(ab_df['ef_pred'] == 'O+').sum():2d}", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 8. AB+ CONFIDENCE DISTRIBUTION & THRESHOLD AUDIT ===", flush=True)
print("=" * 75, flush=True)

corr_confs = ab_df[ab_df["is_correct"] == True]["confidence"].values
incorr_confs = ab_df[ab_df["is_correct"] == False]["confidence"].values

print(f"Correct AB+ Predictions (n={len(corr_confs)}):", flush=True)
print(f"  Mean:   {np.mean(corr_confs)*100:5.2f}%", flush=True)
print(f"  Median: {np.median(corr_confs)*100:5.2f}%", flush=True)
print(f"  Min:    {np.min(corr_confs)*100:5.2f}%", flush=True)
print(f"  Max:    {np.max(corr_confs)*100:5.2f}%", flush=True)

print(f"\nIncorrect AB+ Predictions (n={len(incorr_confs)}):", flush=True)
print(f"  Mean:   {np.mean(incorr_confs)*100:5.2f}%", flush=True)
print(f"  Median: {np.median(incorr_confs)*100:5.2f}%", flush=True)
print(f"  Min:    {np.min(incorr_confs)*100:5.2f}%", flush=True)
print(f"  Max:    {np.max(incorr_confs)*100:5.2f}%", flush=True)

low_conf_incorrect = int((ab_df[ab_df["is_correct"] == False]["confidence"] < CONFIDENCE_THRESHOLD).sum())
print(f"\nFraction of incorrect AB+ flagged as LOW_CONFIDENCE (< {CONFIDENCE_THRESHOLD*100:.0f}%):", flush=True)
print(f"  {low_conf_incorrect} / {len(incorr_confs)} ({low_conf_incorrect/len(incorr_confs)*100:.1f}%) successfully caught by threshold!", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== 9. FEATURE & IMAGE CHARACTERISTICS (CORRECT vs INCORRECT AB+) ===", flush=True)
print("=" * 75, flush=True)

corr_df = ab_df[ab_df["is_correct"] == True]
incorr_df = ab_df[ab_df["is_correct"] == False]

print("Image Feature Statistics (Grayscale):", flush=True)
print(f"  Mean Pixel Intensity:     Correct = {corr_df['mean_intensity'].mean():.2f}  |  Incorrect = {incorr_df['mean_intensity'].mean():.2f}", flush=True)
print(f"  Std Pixel Contrast:       Correct = {corr_df['std_intensity'].mean():.2f}   |  Incorrect = {incorr_df['std_intensity'].mean():.2f}", flush=True)
print(f"  Laplacian Ridge Sharpness: Correct = {corr_df['laplacian_sharpness'].mean():.2f} |  Incorrect = {incorr_df['laplacian_sharpness'].mean():.2f}", flush=True)

# Save sample predictions for AB+
ab_csv_path = os.path.join(OUTPUT_DIR, "ab_plus_sample_predictions.csv")
ab_df.to_csv(ab_csv_path, index=False)
print(f"\nSaved all {len(ab_df)} AB+ sample predictions to: {ab_csv_path}", flush=True)

# Save summary JSON
summary = {
    "train_split": {
        "total": len(train_df),
        "ab_plus_count": int(train_counts["AB+"]),
        "ab_plus_pct": float(train_counts["AB+"] / len(train_df) * 100),
        "counts_per_class": {c: int(train_counts[c]) for c in CLASS_NAMES}
    },
    "val_split": {
        "total": len(val_df),
        "ab_plus_count": total_ab,
        "ab_plus_pct": float(total_ab / len(val_df) * 100),
        "counts_per_class": {c: int(val_counts[c]) for c in CLASS_NAMES}
    },
    "overall_accuracy": {
        "ensemble": float(overall_ens_acc),
        "resnet50": float(overall_rn_acc),
        "vgg16": float(overall_vg_acc),
        "efficientnetb0": float(overall_ef_acc)
    },
    "ab_plus_metrics": {
        "total": total_ab,
        "correct": int(ab_correct),
        "accuracy_pct": float(ab_correct / total_ab * 100),
        "to_o_neg": int(ab_to_o_neg),
        "to_o_pos": int(ab_to_o_pos),
        "to_other": int(ab_to_other),
        "to_class_breakdown": {cls: int((ab_df["pred_label"] == cls).sum()) for cls in CLASS_NAMES}
    },
    "per_model_ab_plus_recall": {
        "resnet50": float(rn_ab_acc),
        "vgg16": float(vg_ab_acc),
        "efficientnetb0": float(ef_ab_acc),
        "ensemble": float(ab_correct / total_ab * 100)
    },
    "confidence_distributions": {
        "correct_mean": float(np.mean(corr_confs)),
        "correct_median": float(np.median(corr_confs)),
        "incorrect_mean": float(np.mean(incorr_confs)),
        "incorrect_median": float(np.median(incorr_confs)),
        "fraction_flagged_low_conf": float(low_conf_incorrect / len(incorr_confs))
    }
}

summary_json_path = os.path.join(OUTPUT_DIR, "ab_plus_confusion_summary.json")
with open(summary_json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved complete summary JSON to: {summary_json_path}", flush=True)

print("\n" + "=" * 75, flush=True)
print("=== INVESTIGATION EXECUTION FINISHED ===", flush=True)
print("=" * 75, flush=True)
