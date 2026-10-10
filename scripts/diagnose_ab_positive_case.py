"""
scripts/diagnose_ab_positive_case.py
Diagnostic investigation of the AB+ MFS100 Left Thumb misclassification (Record #202).
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    CAPTURED_IMAGE_DIR, DATASET_MANIFEST_PATH, CLASS_NAMES,
    PROJECT_ROOT
)
from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white, resize_to_target,
    crop_mfs100_foreground, enhance_mfs100_ridge_contrast,
    preprocess_for_ensemble
)
from src.predictor import BloodGroupPredictor
from src.mfs100.scanner import MFS100Scanner

DIAG_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "diagnostic_ab_plus")
os.makedirs(DIAG_OUT_DIR, exist_ok=True)

TARGET_FILE = os.path.join(CAPTURED_IMAGE_DIR, "capture_1788597822275.bmp")
COMPANION_FILES = [
    os.path.join(CAPTURED_IMAGE_DIR, "capture_1788597819674.bmp"),
    os.path.join(CAPTURED_IMAGE_DIR, "capture_1788597820976.bmp"),
    os.path.join(CAPTURED_IMAGE_DIR, "capture_1788597822275.bmp"),
]

print("=" * 70)
print("=== 1. FILE IDENTIFICATION & CONFIRMATION ===")
print("=" * 70)
print(f"Target file: {TARGET_FILE}")
print(f"Exists: {os.path.exists(TARGET_FILE)}")
for i, cf in enumerate(COMPANION_FILES, 1):
    print(f"Companion Impression {i}: {os.path.basename(cf)} (Exists: {os.path.exists(cf)})")

print("\n" + "=" * 70)
print("=== 2. RAW IMAGE & CROPPING METRICS ===")
print("=" * 70)
raw_bgr = cv2.imread(TARGET_FILE)
raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
rh, rw = raw_gray.shape
print(f"Raw Dimensions: {rw}x{rh} px (Width x Height)")
print(f"Raw Pixel Stats: min={raw_gray.min()}, max={raw_gray.max()}, mean={raw_gray.mean():.2f}, std={raw_gray.std():.2f}, median={np.median(raw_gray):.1f}")

# Quality Analysis from MFS100 Hardware Bridge (Log at 14:13:43)
quality_score = 76
nfiq_score = 2
pressure_status = "OPTIMAL" if quality_score >= 70 else "MARGINAL"
placement_advice = "Impression quality 76/100, NFIQ 2 (Good optical capture)."
print(f"Quality Score: {quality_score} / 100 (from MFS100 SDK bridge log)")
print(f"Estimated NFIQ: {nfiq_score} (Good quality)")
print(f"Pressure Status: {pressure_status}")
print(f"Placement Advice: {placement_advice}")

# Crop execution
raw_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
cropped_pil, crop_meta = crop_mfs100_foreground(raw_pil)
cw, ch = cropped_pil.size
cropped_gray = np.array(cropped_pil.convert("L"))

print(f"Crop Applied: {crop_meta.get('crop_applied')}")
print(f"Crop Reason: {crop_meta.get('reason')}")
print(f"Crop Bounding Box (x1, y1, x2, y2): {crop_meta.get('bbox')}")
print(f"Cropped Dimensions: {cw}x{ch} px")
print(f"Cropped Pixel Stats: min={cropped_gray.min()}, max={cropped_gray.max()}, mean={cropped_gray.mean():.2f}, std={cropped_gray.std():.2f}, median={np.median(cropped_gray):.1f}")

# Save cropped image artifact
cropped_save_path = os.path.join(DIAG_OUT_DIR, "ab_plus_cropped.png")
cropped_pil.save(cropped_save_path)
print(f"Saved cropped image to: {cropped_save_path}")

# Enhancement execution
enhanced_pil = enhance_mfs100_ridge_contrast(cropped_pil)
enhanced_gray = np.array(enhanced_pil.convert("L"))
print(f"Enhanced Pixel Stats: min={enhanced_gray.min()}, max={enhanced_gray.max()}, mean={enhanced_gray.mean():.2f}, std={enhanced_gray.std():.2f}, median={np.median(enhanced_gray):.1f}")

# Save enhanced image artifact
enhanced_save_path = os.path.join(DIAG_OUT_DIR, "ab_plus_enhanced.png")
enhanced_pil.save(enhanced_save_path)
print(f"Saved enhanced image to: {enhanced_save_path}")

# Final 224x224 display image
display_224 = resize_to_target(enhanced_pil, (224, 224))
display_save_path = os.path.join(DIAG_OUT_DIR, "ab_plus_224x224_input.png")
display_224.save(display_save_path)
d_gray = np.array(display_224.convert("L"))
print(f"224x224 Input Stats: min={d_gray.min()}, max={d_gray.max()}, mean={d_gray.mean():.2f}, std={d_gray.std():.2f}")

print("\n" + "=" * 70)
print("=== 3. MODEL INFERENCE & COMPLETE PROBABILITY BREAKDOWN ===")
print("=" * 70)
predictor = BloodGroupPredictor()

# Standard MFS100 inference (source="mfs100")
res_mfs = predictor.predict(TARGET_FILE, source="mfs100")
print("--- Standard MFS100 Pipeline Prediction ---")
print(f"Predicted Class: {res_mfs['blood_group']}")
print(f"Confidence: {res_mfs['confidence'] * 100:.2f}%")
print(f"Status: {res_mfs['status']}")
print("Derived ABO Analysis:", json.dumps(res_mfs.get("abo_analysis", {}), indent=2))
print("\nComplete 8-Class Ensemble Probabilities:")
for cls, p in sorted(res_mfs["probabilities"].items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls:4s}: {p * 100:6.2f}%")

# Submodel breakdown
tensors = preprocess_for_ensemble(TARGET_FILE, source="mfs100")
p_resnet = predictor.resnet_model.predict(tensors["resnet50"], verbose=0)[0]
p_vgg = predictor.vgg_model.predict(tensors["vgg16"], verbose=0)[0]
p_eff = predictor.eff_model.predict(tensors["efficientnetb0"], verbose=0)[0]

print("\nSubmodel Breakdown:")
print(f"{'Class':5s} | {'ResNet50 (58%)':15s} | {'VGG16 (38%)':13s} | {'EffNet (4%)':13s} | {'Ensemble':10s}")
print("-" * 65)
for i, cls in enumerate(CLASS_NAMES):
    ens_p = res_mfs["probabilities"][cls]
    print(f"{cls:5s} | {p_resnet[i] * 100:14.2f}% | {p_vgg[i] * 100:12.2f}% | {p_eff[i] * 100:12.2f}% | {ens_p * 100:9.2f}%")

print(f"\nResNet50 Top: {CLASS_NAMES[np.argmax(p_resnet)]} ({np.max(p_resnet) * 100:.2f}%)")
print(f"VGG16 Top:    {CLASS_NAMES[np.argmax(p_vgg)]} ({np.max(p_vgg) * 100:.2f}%)")
print(f"EffNetB0 Top: {CLASS_NAMES[np.argmax(p_eff)]} ({np.max(p_eff) * 100:.2f}%)")

print("\n" + "=" * 70)
print("=== 4. ABLATION: PREPROCESSING & CROPPING IMPACT ===")
print("=" * 70)
# Ablation A: source="upload" (NO crop, NO CLAHE enhancement)
res_upload = predictor.predict(TARGET_FILE, source="upload")
print(f"A. Without Crop/Enhancement (Raw Upload Mode): Predicted {res_upload['blood_group']} ({res_upload['confidence'] * 100:.2f}%)")
print("   Upload Mode Top 3:", sorted(res_upload["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3])

# Ablation B: Crop ONLY (no CLAHE)
crop_only_224 = resize_to_target(cropped_pil, (224, 224))
arr_co = np.expand_dims(np.array(crop_only_224, dtype=np.float32), axis=0)
t_rn_co = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(np.array(ImageEnhance.Contrast(crop_only_224).enhance(1.2), dtype=np.float32), axis=0))
t_vg_co = tf.keras.applications.vgg16.preprocess_input(arr_co.copy())
t_ef_co = tf.keras.applications.efficientnet.preprocess_input(arr_co.copy())
p_rn_co = predictor.resnet_model.predict(t_rn_co, verbose=0)[0]
p_vg_co = predictor.vgg_model.predict(t_vg_co, verbose=0)[0]
p_ef_co = predictor.eff_model.predict(t_ef_co, verbose=0)[0]
p_ens_co = 0.58 * p_rn_co + 0.38 * p_vg_co + 0.04 * p_ef_co
top_co = CLASS_NAMES[np.argmax(p_ens_co)]
print(f"B. Crop ONLY (No Ridge Enhancement):          Predicted {top_co} ({np.max(p_ens_co) * 100:.2f}%)")

# Ablation C: CLAHE enhancement ONLY on raw (no crop)
enh_nocrop = enhance_mfs100_ridge_contrast(raw_pil)
enh_nocrop_224 = resize_to_target(enh_nocrop, (224, 224))
arr_en = np.expand_dims(np.array(enh_nocrop_224, dtype=np.float32), axis=0)
t_rn_en = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(np.array(ImageEnhance.Contrast(enh_nocrop_224).enhance(1.2), dtype=np.float32), axis=0))
t_vg_en = tf.keras.applications.vgg16.preprocess_input(arr_en.copy())
t_ef_en = tf.keras.applications.efficientnet.preprocess_input(arr_en.copy())
p_rn_en = predictor.resnet_model.predict(t_rn_en, verbose=0)[0]
p_vg_en = predictor.vgg_model.predict(t_vg_en, verbose=0)[0]
p_ef_en = predictor.eff_model.predict(t_ef_en, verbose=0)[0]
p_ens_en = 0.58 * p_rn_en + 0.38 * p_vg_en + 0.04 * p_ef_en
top_en = CLASS_NAMES[np.argmax(p_ens_en)]
print(f"C. Ridge Enhancement ONLY (No Crop):          Predicted {top_en} ({np.max(p_ens_en) * 100:.2f}%)")

print("\n" + "=" * 70)
print("=== 5. DETERMINISM VERIFICATION (10 REPEATED RUNS) ===")
print("=" * 70)
probs_runs = []
for r in range(10):
    p_run = predictor.predict(TARGET_FILE, source="mfs100")["probabilities"]
    probs_runs.append([p_run[c] for c in CLASS_NAMES])
probs_runs = np.array(probs_runs)
max_dev = np.max(np.abs(probs_runs - probs_runs[0]))
print(f"10-Run Maximum Absolute Deviation: {max_dev:.2e}")
print(f"Deterministic Inference Confirmed: {max_dev < 1e-6}")

print("\n" + "=" * 70)
print("=== 6. 3-SCAN CONSENSUS EVALUATION ON THIS SESSION ===")
print("=" * 70)
con_res = predictor.predict_consensus(COMPANION_FILES, source="mfs100")
print(f"Consensus Prediction: {con_res['blood_group']}")
print(f"Consensus Confidence: {con_res['confidence'] * 100:.2f}%")
print(f"Consensus Status: {con_res['status']}")
print(f"Agreement Count: {con_res['consensus_metadata']['agreement_count']} / 3")
for s in con_res["consensus_metadata"]["individual_scans"]:
    print(f"  Impression #{s['scan_index']}: {s['blood_group']} ({s['confidence'] * 100:.2f}%)")

print("\n" + "=" * 70)
print("=== 7. PREVIOUS MFS100 CAPTURES OF THIS SESSION ===")
print("=" * 70)
# Look at nearby captures in this time window
import sqlite3
from src.config import DATABASE_PATH
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, timestamp, input_type, image_path, predicted_group, confidence, status FROM predictions WHERE id BETWEEN 195 AND 206 ORDER BY id ASC")
recent_records = cursor.fetchall()
conn.close()
for r in recent_records:
    print(f"Record #{r[0]}: TS={r[1]} | Type={r[2][:30]} | Img={r[3]} | Pred={r[4]:4s} | Conf={r[5]*100:5.1f}% | Status={r[6]}")

print("\n" + "=" * 70)
print("=== 8. COMPARISON WITH VALIDATION DATASET IMAGES (NON-TEST ONLY) ===")
print("=" * 70)
df = pd.read_csv(DATASET_MANIFEST_PATH)
val_df = df[df["split"] == "val"]

def analyze_class_val(class_name, n=5):
    sub = val_df[val_df["class_name"] == class_name]
    print(f"\n--- Validation Samples for {class_name} (Total in Val: {len(sub)}) ---")
    records = []
    for idx, row in sub.head(n).iterrows():
        p = os.path.join(PROJECT_ROOT, row["image_path"])
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("L")
        arr = np.array(im)
        p_res = predictor.predict(p, source="upload")
        correct = (p_res["blood_group"] == class_name)
        records.append({
            "file": os.path.basename(p),
            "size": f"{im.size[0]}x{im.size[1]}",
            "mean": arr.mean(),
            "std": arr.std(),
            "pred": p_res["blood_group"],
            "conf": p_res["confidence"],
            "correct": correct
        })
        print(f"  {os.path.basename(p):25s} | size={im.size[0]}x{im.size[1]} | mean={arr.mean():.1f}, std={arr.std():.1f} | Pred: {p_res['blood_group']:4s} ({p_res['confidence'] * 100:5.1f}%) | Correct: {correct}")
    return records

val_ab_pos = analyze_class_val("AB+", n=5)
val_o_neg  = analyze_class_val("O-", n=5)
val_o_pos  = analyze_class_val("O+", n=5)

print("\n" + "=" * 70)
print("Diagnostic investigation completed.")
print("=" * 70)
