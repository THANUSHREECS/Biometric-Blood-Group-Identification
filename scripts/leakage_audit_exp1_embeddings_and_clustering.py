"""
scripts/leakage_audit_exp1_embeddings_and_clustering.py
Experiment 1: Part 2 - Deep Embedding Extraction, Cosine Similarity Analysis & Connected-Component Clustering.

Implements User Amendment 1:
- Extracts 1280-d L2-normalized embeddings via frozen ImageNet EfficientNet-B0
- Computes pairwise cosine similarity matrix
- Computes distribution of max cosine similarity to TRAIN for legacy TEST images
- Evaluates connected components across pre-declared thresholds (0.95, 0.90, 0.85) plus fine-grained thresholds (0.98, 0.92)
- Reports: number of components, largest component size, % of total data, singleton count
- Enforces the 5% limit (~138 images) guard: any component > 138 images is flagged and blocked from StratifiedGroupKFold
- Saves cluster assignments to JSON/CSV and plots similarity distributions.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

OUT_DIR = Path("experiments/leakage_audit")
FIG_DIR = Path("reports/leakage_audit/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Positive Manifest
manifest_file = Path("outputs/dataset_manifest.csv")
df_man = pd.read_csv(manifest_file)
pos_classes = ["A+", "AB+", "B+", "O+"]
df_pos = df_man[df_man["class_name"].isin(pos_classes)].copy().reset_index(drop=True)
n_total = len(df_pos)
print(f"Total positive-class images: {n_total}")

# Resolve local paths
def resolve_local_path(p_str):
    idx = p_str.find("data/processed")
    if idx != -1:
        local_p = Path(p_str[idx:])
    else:
        local_p = Path(p_str)
    if not local_p.exists():
        fn = Path(p_str).name
        cname = p_str.split("/")[-2] if "/" in p_str else p_str.split("\\")[-2]
        local_p = Path("data/processed") / cname / fn
    assert local_p.exists(), f"File not found: {local_p}"
    return local_p

df_pos["local_path"] = df_pos["image_path"].apply(resolve_local_path)

# 2. Extract Embeddings with TensorFlow EfficientNetB0
print("\n--- Loading Pretrained EfficientNetB0 Feature Extractor ---")
import tensorflow as tf
from src.preprocessing import composite_alpha_on_white, resize_to_target

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg"
)
base_model.trainable = False

batch_size = 64
embeddings = []

print(f"Extracting embeddings for {n_total} images in batches of {batch_size}...")
for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    batch_paths = df_pos["local_path"].iloc[start:end]
    
    batch_tensors = []
    for p in batch_paths:
        with Image.open(p) as img:
            rgb_img = composite_alpha_on_white(img)
            resized_img = resize_to_target(rgb_img, target_size=(224, 224))
            arr = np.array(resized_img, dtype=np.float32) # float32 [0, 255]
            batch_tensors.append(arr)
            
    batch_arr = np.array(batch_tensors)
    batch_features = base_model(batch_arr, training=False).numpy()
    embeddings.append(batch_features)
    if (start // batch_size) % 10 == 0:
        print(f"  Processed {end}/{n_total} images...")

embeddings = np.vstack(embeddings)
print(f"Extracted feature matrix shape: {embeddings.shape}")

# L2-normalize embeddings for exact cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1.0
norm_embeddings = embeddings / norms

# Save embeddings
emb_file = OUT_DIR / "exp1_efficientnetb0_embeddings.npy"
np.save(emb_file, norm_embeddings)
print(f"[SAVED] Normalized embeddings saved to {emb_file}")

# 3. Compute Pairwise Cosine Similarity
print("\n--- Computing Pairwise Cosine Similarity Matrix ---")
sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
# Ensure numerical stability
sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
np.fill_diagonal(sim_matrix, 1.0)

# Extract off-diagonal similarities for distribution analysis
triu_indices = np.triu_indices(n_total, k=1)
off_diag_sims = sim_matrix[triu_indices]

print("\n--- Off-Diagonal Pairwise Cosine Similarity Distribution ---")
percentiles = [0, 1, 5, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100]
pct_vals = np.percentile(off_diag_sims, percentiles)
for p, v in zip(percentiles, pct_vals):
    print(f"  {p:5.1f}th percentile: {v:.4f}")

# 4. Proximity of Legacy TEST images to TRAIN images
train_indices = df_pos[df_pos["split"] == "train"].index.values
test_indices = df_pos[df_pos["split"] == "test"].index.values
val_indices = df_pos[df_pos["split"] == "val"].index.values

print(f"\n--- Cross-Split Cosine Proximity ---")
print(f"Train images: {len(train_indices)}, Test images: {len(test_indices)}, Val images: {len(val_indices)}")

# For each test image, find maximum cosine similarity to any train image
test_sim_to_train = sim_matrix[np.ix_(test_indices, train_indices)]
max_sim_to_train = np.max(test_sim_to_train, axis=1)

print("\nMax Cosine Similarity from Legacy TEST to TRAIN Distribution:")
test_pct_vals = np.percentile(max_sim_to_train, [0, 5, 25, 50, 75, 90, 95, 99, 100])
for p, v in zip([0, 5, 25, 50, 75, 90, 95, 99, 100], test_pct_vals):
    print(f"  {p:5.1f}th percentile: {v:.4f}")

for th in [0.98, 0.95, 0.92, 0.90, 0.85]:
    cnt = int(np.sum(max_sim_to_train >= th))
    pct = cnt / len(test_indices) * 100
    print(f"  Test images with Cosine Sim >= {th:.2f} to TRAIN: {cnt:3}/{len(test_indices)} ({pct:.2f}%)")

# 5. Connected Component Clustering across Thresholds
THRESHOLDS = [0.98, 0.95, 0.92, 0.90, 0.85]
MAX_PERMITTED_CLUSTER_PCT = 0.05
MAX_PERMITTED_CLUSTER_SIZE = int(n_total * MAX_PERMITTED_CLUSTER_PCT)  # 138 images

print(f"\n==================================================")
print(f"CONNECTED COMPONENT CLUSTERING (5% Guard = {MAX_PERMITTED_CLUSTER_SIZE} images)")
print(f"==================================================")

clustering_results = {}
cluster_labels_by_thresh = {}

for th in THRESHOLDS:
    G = nx.Graph()
    G.add_nodes_from(range(n_total))
    
    # Add edges where cosine similarity >= threshold
    rows, cols = np.where(sim_matrix >= th)
    # Only keep upper triangle (avoid self loops and redundant edges)
    mask = rows < cols
    edges = list(zip(rows[mask], cols[mask]))
    G.add_edges_from(edges)
    
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]
    largest_size = max(comp_sizes)
    largest_pct = largest_size / n_total * 100
    singletons = sum(1 for s in comp_sizes if s == 1)
    multi_comps = sum(1 for s in comp_sizes if s > 1)
    
    # Assign cluster labels
    cluster_map = np.zeros(n_total, dtype=int)
    for c_id, comp in enumerate(components):
        for node in comp:
            cluster_map[node] = c_id
            
    cluster_labels_by_thresh[f"cluster_th_{int(th*100)}"] = cluster_map
    
    violates_guard = largest_size > MAX_PERMITTED_CLUSTER_SIZE
    
    print(f"\nThreshold: {th:.2f}")
    print(f"  Edges in graph: {len(edges)}")
    print(f"  Total components: {len(components)}")
    print(f"  Multi-image components (size >= 2): {multi_comps}")
    print(f"  Singletons (size == 1): {singletons}")
    print(f"  Largest component size: {largest_size} ({largest_pct:.2f}% of data)")
    print(f"  Component size percentiles [50%, 90%, 99%, Max]: {[int(np.percentile(comp_sizes, p)) for p in [50, 90, 99, 100]]}")
    print(f"  5% Guard (<={MAX_PERMITTED_CLUSTER_SIZE} imgs): {'VIOLATED (DO NOT USE FOR CV)' if violates_guard else 'PASSED (Eligible for Grouped CV)'}")
    
    clustering_results[f"threshold_{th:.2f}"] = {
        "threshold": th,
        "n_edges": len(edges),
        "n_components": len(components),
        "multi_image_components": multi_comps,
        "singleton_count": singletons,
        "largest_component_size": largest_size,
        "largest_component_pct": float(largest_pct),
        "violates_5pct_guard": bool(violates_guard),
        "eligible_for_grouped_cv": bool(not violates_guard)
    }

# Save Clustering Report to JSON
clustering_summary = {
    "total_images": n_total,
    "max_permitted_cluster_size_5pct": MAX_PERMITTED_CLUSTER_SIZE,
    "thresholds_evaluated": clustering_results,
    "max_sim_to_train_summary": {
        "mean": float(np.mean(max_sim_to_train)),
        "std": float(np.std(max_sim_to_train)),
        "min": float(np.min(max_sim_to_train)),
        "max": float(np.max(max_sim_to_train)),
        "percentiles": {f"p_{p}": float(v) for p, v in zip([0, 5, 25, 50, 75, 90, 95, 99, 100], test_pct_vals)}
    }
}

json_path = OUT_DIR / "exp1_clustering_summary.json"
with open(json_path, "w") as f:
    json.dump(clustering_summary, f, indent=2)

# Save cluster assignments to CSV
for col_name, arr in cluster_labels_by_thresh.items():
    df_pos[col_name] = arr

csv_out = OUT_DIR / "exp1_positive_images_clusters.csv"
df_pos.to_csv(csv_out, index=False)
print(f"\n[SAVED] Clustering summary saved to {json_path}")
print(f"[SAVED] Cluster assignments saved to {csv_out}")

# 6. Generate Figures
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Overall Cosine Similarity Distribution
ax = axes[0]
ax.hist(off_diag_sims, bins=50, color="#4C72B0", alpha=0.75, edgecolor="black", density=True)
ax.set_title("Pairwise Off-Diagonal Cosine Similarity Distribution", fontsize=11, fontweight="bold")
ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("Density")
ax.axvline(0.95, color="red", linestyle="--", label="Threshold 0.95")
ax.axvline(0.90, color="orange", linestyle="--", label="Threshold 0.90")
ax.axvline(0.85, color="purple", linestyle="--", label="Threshold 0.85")
ax.legend(loc="upper left")

# Plot 2: Max Similarity of Test Images to Train
ax = axes[1]
ax.hist(max_sim_to_train, bins=35, color="#55A868", alpha=0.75, edgecolor="black", density=True)
ax.set_title("Legacy TEST Images: Max Cosine Similarity to TRAIN Set", fontsize=11, fontweight="bold")
ax.set_xlabel("Max Cosine Similarity to Nearest Training Image")
ax.set_ylabel("Density")
ax.axvline(0.95, color="red", linestyle="--", label=f">=0.95 ({np.sum(max_sim_to_train >= 0.95)/len(test_indices)*100:.1f}%)")
ax.axvline(0.90, color="orange", linestyle="--", label=f">=0.90 ({np.sum(max_sim_to_train >= 0.90)/len(test_indices)*100:.1f}%)")
ax.axvline(0.85, color="purple", linestyle="--", label=f">=0.85 ({np.sum(max_sim_to_train >= 0.85)/len(test_indices)*100:.1f}%)")
ax.legend(loc="upper left")

plt.tight_layout()
fig_path = FIG_DIR / "exp1_cosine_similarity_distributions.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[SAVED] Figure written to {fig_path}")
print("=== PART 2 COMPLETED SUCCESSFULLY ===")
