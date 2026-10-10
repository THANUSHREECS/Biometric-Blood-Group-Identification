import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score

from src.config import PROJECT_ROOT
from src.predictor import BloodGroupPredictor
from src.preprocessing import composite_alpha_on_white, resize_to_target, crop_mfs100_foreground, apply_fingerprint_enhancement
import tensorflow as tf

predictor = BloodGroupPredictor()

part_df = pd.read_csv("data/mfs100_final/metadata/participants.csv")
log_df = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
merged = pd.merge(log_df, part_df, on="participant_id", how="left")
classes = ['A+', 'B+', 'AB+', 'O+']
merged = merged[merged['blood_group'].isin(classes)]
groups = merged.groupby("participant_id")

def custom_preprocess(img_path, use_crop, use_clahe):
    img = Image.open(img_path)
    img = composite_alpha_on_white(img)
    if use_crop:
        try:
            img, _ = crop_mfs100_foreground(img)
        except Exception:
            pass
    arr = np.array(img)
    if use_clahe:
        arr = apply_fingerprint_enhancement(arr, enabled=True)
    img = Image.fromarray(arr)
    img = resize_to_target(img, target_size=(112, 112))
    arr = tf.keras.applications.densenet.preprocess_input(np.array(img, dtype=np.float32))
    return arr

variants = {
    "Training-style (No Crop, No CLAHE)": (False, False),
    "Crop Only": (True, False),
    "CLAHE Only": (False, True)
}

print("--- Sensitivity Analysis, Not Tuning ---")
for v_name, (crop, clahe) in variants.items():
    preds = []
    trues = []
    for pid, group in groups:
        paths = group["image_path"].tolist()
        if len(paths) == 3:
            t_bg = group["blood_group"].iloc[0]
            scans = [custom_preprocess(p, crop, clahe) for p in paths]
            X = np.expand_dims(np.array(scans), axis=0)
            
            all_probs = []
            for cp in predictor.checkpoint_paths:
                predictor.model.load_weights(cp)
                probs = predictor.model.predict(X, verbose=0)[0]
                all_probs.append(probs[0])
                
            ensemble_probs = np.mean(all_probs, axis=0)
            best_idx = int(np.argmax(ensemble_probs))
            preds.append(predictor.class_names[best_idx])
            trues.append(t_bg)
    
    acc = accuracy_score(trues, preds)
    print(f"{v_name}: {acc*100:.2f}%", flush=True)
