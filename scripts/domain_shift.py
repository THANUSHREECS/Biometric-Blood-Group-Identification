import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

print("Task 6: Domain Shift Analysis")
os.makedirs("results", exist_ok=True)

dataset_files = glob.glob("data/processed/*/*.BMP")[:50]
live_files = glob.glob("data/captured/*.bmp")

if not live_files:
    print("No live files found for domain shift.")
    exit()

def get_stats(img):
    return np.mean(img), np.std(img), img.flatten()

dataset_means, dataset_stds, dataset_pixels = [], [], []
for f in dataset_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        m, s, p = get_stats(img)
        dataset_means.append(m)
        dataset_stds.append(s)
        dataset_pixels.extend(p)

live_means, live_stds, live_pixels = [], [], []
for f in live_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        m, s, p = get_stats(img)
        live_means.append(m)
        live_stds.append(s)
        live_pixels.extend(p)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].boxplot([dataset_means, live_means], labels=["Dataset", "Live Captured"])
axes[0].set_title("Mean Intensity")
axes[1].boxplot([dataset_stds, live_stds], labels=["Dataset", "Live Captured"])
axes[1].set_title("Ridge Contrast (Std Dev)")
axes[2].hist(dataset_pixels, bins=50, alpha=0.5, label="Dataset", density=True)
axes[2].hist(live_pixels, bins=50, alpha=0.5, label="Live Captured", density=True)
axes[2].set_title("Pixel Histogram")
axes[2].legend()

plt.tight_layout()
plt.savefig("results/domain_shift.png")
print("Saved results/domain_shift.png")

print(f"Dataset Mean Intensity: {np.mean(dataset_means):.2f}, Live: {np.mean(live_means):.2f}")
print(f"Dataset Ridge Contrast: {np.mean(dataset_stds):.2f}, Live: {np.mean(live_stds):.2f}")
