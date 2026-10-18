import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"

emb = np.load(EXP_DIR / "exp1_efficientnetb0_embeddings.npy")
N = len(emb)
print(f"Loaded EfficientNet-B0 embeddings: shape {emb.shape}")

# Cosine similarity matrix (embeddings are already L2-normalized in exp1)
norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb_norm = emb / norms
sim = np.matmul(emb_norm, emb_norm.T)

thresholds = [0.985, 0.980, 0.975, 0.970, 0.965, 0.960, 0.955, 0.950, 0.920, 0.900, 0.850]
results = {
    "total_images": N,
    "embeddings_source": "exp1_efficientnetb0_embeddings.npy (EfficientNet-B0)",
    "thresholds_evaluated": {}
}

for th in thresholds:
    # Adjacency matrix: sim >= th and i != j
    adj = (sim >= th)
    np.fill_diagonal(adj, False)
    
    n_edges = int(np.sum(adj) // 2)
    n_comp, labels = connected_components(sp.csr_matrix(adj), directed=False)
    
    unique, counts = np.unique(labels, return_counts=True)
    largest_comp = int(np.max(counts))
    largest_pct = float(largest_comp / N * 100)
    singletons = int(np.sum(counts == 1))
    multi_img = int(np.sum(counts > 1))
    violates_guard = bool(largest_comp > 138)
    
    results["thresholds_evaluated"][f"threshold_{th:.3f}"] = {
        "threshold": th,
        "n_edges": n_edges,
        "n_components": int(n_comp),
        "multi_image_components": multi_img,
        "singleton_count": singletons,
        "largest_component_size": largest_comp,
        "largest_component_pct": largest_pct,
        "violates_5pct_guard": violates_guard,
        "eligible_for_grouped_cv": not violates_guard
    }
    print(f"th={th:.3f}: edges={n_edges:<7} comp={n_comp:<5} multi={multi_img:<4} single={singletons:<5} largest={largest_comp:<5} ({largest_pct:.2f}%) guard_viol={violates_guard}")

with open(EXP_DIR / "exp1_percolation_finer_scan.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved exp1_percolation_finer_scan.json")
