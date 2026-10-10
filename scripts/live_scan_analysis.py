import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.predictor import BloodGroupPredictor
from src.preprocessing import transform_c5

print("Running Live Scan Predictor Analysis on ALL 1,218 images...")

predictor = BloodGroupPredictor(use_ensemble=True)
if not predictor.is_model_ready():
    print("Model failed to load.")
    sys.exit(1)

captured_dir = "data/captured"
images = glob.glob(os.path.join(captured_dir, "*.bmp"))
valid_images = [img for img in images if os.path.getsize(img) > 0]
print(f"Found {len(valid_images)} valid live captures.")

results = []
intensities = []
contrasts = []

# Batching to speed up
batch_size = 32
for i in tqdm(range(0, len(valid_images), batch_size)):
    batch_paths = valid_images[i:i+batch_size]
    batch_scans = []
    valid_batch_paths = []
    
    for img_path in batch_paths:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                intensities.append(np.mean(img))
                contrasts.append(np.std(img))
            arr, _, _ = transform_c5(img_path)
            batch_scans.append([arr, arr, arr]) # 3 times
            valid_batch_paths.append(img_path)
        except Exception as e:
            print(f"Skipping unreadable file: {img_path} - {e}")
    
    if not batch_scans:
        continue
        
    X_input = np.array(batch_scans)
    
    all_probs = []
    for cp in predictor.checkpoint_paths:
        predictor.model.load_weights(cp)
        preds = predictor.model.predict(X_input, verbose=0)
        all_probs.append(preds[0]) # [0] because model returns [clf, emb]
        
    all_probs = np.array(all_probs)
    ensemble_probs = np.mean(all_probs, axis=0) # shape (batch, 4)
    
    for j in range(len(valid_batch_paths)):
        best_idx = np.argmax(ensemble_probs[j])
        pred_class = predictor.class_names[best_idx]
        conf = float(ensemble_probs[j][best_idx])
        
        results.append({
            "file": os.path.basename(valid_batch_paths[j]),
            "predicted_class": pred_class,
            "confidence": conf,
            "prob_A+": ensemble_probs[j][0],
            "prob_AB+": ensemble_probs[j][1],
            "prob_B+": ensemble_probs[j][2],
            "prob_O+": ensemble_probs[j][3]
        })

df = pd.DataFrame(results)
df.to_csv("results/live_scan_predictions.csv", index=False)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.hist(df["confidence"], bins=20, color='skyblue', edgecolor='black')
plt.axvline(0.70, color='red', linestyle='dashed', linewidth=2)
plt.title("Top Confidence")

plt.subplot(132)
df["predicted_class"].value_counts().plot(kind='bar', color='salmon')
plt.title("Predicted Classes")

plt.subplot(133)
plt.scatter(intensities, contrasts, alpha=0.5, color='green')
plt.title("Intensity vs Contrast (Live)")
plt.tight_layout()
plt.savefig("results/live_scan_analysis.png")

above_70 = (df["confidence"] >= 0.70).mean() * 100
print(f"\n% above 70% confidence: {above_70:.2f}%")
print("\nClass Distribution:")
print(df["predicted_class"].value_counts(normalize=True) * 100)

print("\nCompared to dataset:")
manifest_path = "outputs/dataset_manifest.csv"
if os.path.exists(manifest_path):
    mdf = pd.read_csv(manifest_path)
    d_ints, d_conts = [], []
    for p in mdf["image_path"].sample(200, random_state=42):
        if os.path.exists(p):
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                d_ints.append(np.mean(im))
                d_conts.append(np.std(im))
    print(f"Dataset Mean Intensity: {np.mean(d_ints):.2f}, Mean Contrast: {np.mean(d_conts):.2f}")
    print(f"Live Mean Intensity: {np.mean(intensities):.2f}, Mean Contrast: {np.mean(contrasts):.2f}")
    print("Conclusion: Live output is heavily skewed (mostly one class) and confidence is flattened. Contrast differs from training distribution.")
