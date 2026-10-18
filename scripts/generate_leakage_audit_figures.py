import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
FIG_DIR = PROJECT_ROOT / "reports" / "leakage_audit" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

# -------------------------------------------------------------
# Figure 1: Phase 2 Selection-Inclusive Null Distribution
# -------------------------------------------------------------
null_csv = EXP_DIR / "selection_inclusive_null_results.csv"
if null_csv.exists():
    df_null = pd.read_csv(null_csv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1A: Accuracy
    ax1.hist(df_null["accuracy"] * 100, bins=15, color="#93c5fd", edgecolor="#2563eb", alpha=0.75, label="Selection Null (100 perm.)")
    ax1.axvline(x=46.77, color="#1e40af", linestyle="--", linewidth=2.0, label="Null Mean (46.77%)")
    ax1.axvline(x=53.81, color="#b45309", linestyle="-", linewidth=2.5, label="Null 95th Pct (53.81%)")
    ax1.axvline(x=53.75, color="#dc2626", linestyle=":", linewidth=2.2, label="Historical / Seed 42 (53.75%)")
    ax1.axvline(x=50.83, color="#7c3aed", linestyle="-.", linewidth=1.8, label="Rule (i) 3-Seed Mean (50.83%)")
    ax1.axvline(x=31.67, color="#10b981", linestyle="--", linewidth=1.8, label="Clean Rule (a) Inner Val (31.67%)")
    ax1.axvline(x=36.25, color="#059669", linestyle="-.", linewidth=1.8, label="Clean Rule (b) Fixed (36.25%)")
    ax1.set_xlabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Permutation Count", fontsize=11, fontweight="bold")
    ax1.set_title("Accuracy vs. Selection Null Distribution", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.2, framealpha=0.92)
    
    # 1B: Macro F1
    ax2.hist(df_null["macro_f1"], bins=15, color="#cbd5e1", edgecolor="#475569", alpha=0.75, label="Selection Null (100 perm.)")
    ax2.axvline(x=0.4277, color="#334155", linestyle="--", linewidth=2.0, label="Null Mean (0.4277)")
    ax2.axvline(x=0.5218, color="#b45309", linestyle="-", linewidth=2.5, label="Null 95th Pct (0.5218)")
    ax2.axvline(x=0.5487, color="#dc2626", linestyle=":", linewidth=2.2, label="Documented Pooled F1 (0.5487)")
    ax2.axvline(x=0.5218, color="#ea580c", linestyle="--", linewidth=1.8, label="Seed 42 Rerun F1 (0.5218)")
    ax2.axvline(x=0.5058, color="#7c3aed", linestyle="-.", linewidth=1.8, label="Rule (i) 3-Seed Mean (0.5058)")
    ax2.axvline(x=0.2841, color="#10b981", linestyle="--", linewidth=1.8, label="Clean Rule (a) Inner Val (0.2841)")
    ax2.axvline(x=0.3278, color="#059669", linestyle="-.", linewidth=1.8, label="Clean Rule (b) Fixed (0.3278)")
    ax2.set_xlabel("Macro F1-Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Permutation Count", fontsize=11, fontweight="bold")
    ax2.set_title("Macro F1 vs. Selection Null Distribution", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.2, framealpha=0.92)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure1_null_distribution_phase2.png", dpi=300)
    plt.close(fig)
    print("Generated figure1_null_distribution_phase2.png")

# -------------------------------------------------------------
# Figure 2: Phase 3 Purge Curves & Paired Differences
# -------------------------------------------------------------
summary_json = EXP_DIR / "phase3_purged_cv_summary.json"
if summary_json.exists():
    with open(summary_json, "r") as f:
        res = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    fractions = [0, 10, 25, 50]
    
    # Arm 1 data
    arm1 = res["primary_arm_fixed_epochs"]
    arm1_base = arm1["0%_baseline"]["mean"] * 100
    arm1_purged = [arm1_base, arm1["10%_purge"]["purged_mean"] * 100, arm1["25%_purge"]["purged_mean"] * 100, arm1["50%_purge"]["purged_mean"] * 100]
    arm1_purged_err = [arm1["0%_baseline"]["std"] * 100, arm1["10%_purge"]["purged_std"] * 100, arm1["25%_purge"]["purged_std"] * 100, arm1["50%_purge"]["purged_std"] * 100]
    arm1_random = [arm1_base, arm1["10%_purge"]["random_mean"] * 100, arm1["25%_purge"]["random_mean"] * 100, arm1["50%_purge"]["random_mean"] * 100]
    arm1_random_err = [arm1["0%_baseline"]["std"] * 100, arm1["10%_purge"]["random_std"] * 100, arm1["25%_purge"]["random_std"] * 100, arm1["50%_purge"]["random_std"] * 100]
    
    # Arm 2 data
    arm2 = res["secondary_arm_inner_val"]
    arm2_base = arm2["0%_baseline"]["mean"] * 100
    arm2_purged = [arm2_base, arm2["10%_purge"]["purged_mean"] * 100, arm2["25%_purge"]["purged_mean"] * 100, arm2["50%_purge"]["purged_mean"] * 100]
    arm2_purged_err = [arm2["0%_baseline"]["std"] * 100, arm2["10%_purge"]["purged_std"] * 100, arm2["25%_purge"]["purged_std"] * 100, arm2["50%_purge"]["purged_std"] * 100]
    arm2_random = [arm2_base, arm2["10%_purge"]["random_mean"] * 100, arm2["25%_purge"]["random_mean"] * 100, arm2["50%_purge"]["random_mean"] * 100]
    arm2_random_err = [arm2["0%_baseline"]["std"] * 100, arm2["10%_purge"]["random_std"] * 100, arm2["25%_purge"]["random_std"] * 100, arm2["50%_purge"]["random_std"] * 100]
    
    # Plot 2A: Performance curves (Zoomed Y-axis)
    ax1.errorbar(fractions, arm1_purged, yerr=arm1_purged_err, fmt="-o", color="#2563eb", capsize=4, linewidth=2, label="Arm 1: Sim-Purged (Fixed 50ep)")
    ax1.errorbar(fractions, arm1_random, yerr=arm1_random_err, fmt="--s", color="#3b82f6", capsize=4, linewidth=2, label="Arm 1: Random-Drop Control")
    ax1.errorbar(fractions, arm2_purged, yerr=arm2_purged_err, fmt="-^", color="#d97706", capsize=4, linewidth=2, label="Arm 2: Sim-Purged (Inner Val)")
    ax1.errorbar(fractions, arm2_random, yerr=arm2_random_err, fmt="--v", color="#f59e0b", capsize=4, linewidth=2, label="Arm 2: Random-Drop Control")
    
    ax1.set_xlabel("Purge Fraction (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Balanced Accuracy vs. Purge Fraction (Zoomed)", fontsize=12, fontweight="bold")
    ax1.set_xticks(fractions)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax1.set_ylim(78.0, 87.0) # Zoomed y-axis
    
    # Plot 2B: Paired Differences (Purged - Random)
    diff_fractions = [10, 25, 50]
    arm1_deltas = [arm1["10%_purge"]["paired_delta_mean"] * 100, arm1["25%_purge"]["paired_delta_mean"] * 100, arm1["50%_purge"]["paired_delta_mean"] * 100]
    arm1_delta_errs = [arm1["10%_purge"]["paired_delta_std"] * 100, arm1["25%_purge"]["paired_delta_std"] * 100, arm1["50%_purge"]["paired_delta_std"] * 100]
    
    arm2_deltas = [arm2["10%_purge"]["paired_delta_mean"] * 100, arm2["25%_purge"]["paired_delta_mean"] * 100, arm2["50%_purge"]["paired_delta_mean"] * 100]
    arm2_delta_errs = [arm2["10%_purge"]["paired_delta_std"] * 100, arm2["25%_purge"]["paired_delta_std"] * 100, arm2["50%_purge"]["paired_delta_std"] * 100]
    
    x_indices = np.arange(len(diff_fractions))
    width = 0.35
    
    rects1 = ax2.bar(x_indices - width/2, arm1_deltas, width, yerr=arm1_delta_errs, capsize=5, label="Arm 1: Fixed 50 Epochs (Primary)", color="#2563eb", alpha=0.85, edgecolor="#1e40af")
    rects2 = ax2.bar(x_indices + width/2, arm2_deltas, width, yerr=arm2_delta_errs, capsize=5, label="Arm 2: Inner Val (Secondary)", color="#d97706", alpha=0.85, edgecolor="#b45309")
    
    ax2.axhline(0, color="#111827", linewidth=1.2)
    ax2.set_xlabel("Purge Fraction (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Paired Delta: Purged - Random (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Paired Purge vs. Random Difference (Mean ± SD)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(["10%", "25%", "50%"])
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax2.set_ylim(-2.0, 2.0)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure2_purge_vs_random_phase3.png", dpi=300)
    plt.close(fig)
    print("Generated figure2_purge_vs_random_phase3.png")

# -------------------------------------------------------------
# Figure 3: DenseNet-121 Cosine Similarity Distribution (Zoomed to 0.70–1.0)
# -------------------------------------------------------------
feat_npz = EXP_DIR / "original_dataset_densenet121_features.npz"
if feat_npz.exists():
    data = np.load(feat_npz, allow_pickle=True)
    features = data['features']
    splits = data['splits']
    
    # Split: Legacy Test (N=415) vs. Legacy Train+Val (N=2,354)
    test_mask = (splits == 'test')
    trainval_mask = (splits != 'test')
    
    feats_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    test_feats = feats_norm[test_mask]        # N=415
    trainval_feats = feats_norm[trainval_mask] # N=2354
    
    sim_matrix = np.matmul(test_feats, trainval_feats.T)
    max_sims = np.max(sim_matrix, axis=1) # (415,)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    
    # 3A: Zoomed Histogram [0.70, 1.0]
    bins = np.linspace(0.70, 1.0, 31)
    ax1.hist(max_sims, bins=bins, color="#0284c7", edgecolor="#0369a1", alpha=0.8)
    ax1.axvline(x=np.mean(max_sims), color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean = {np.mean(max_sims):.3f}")
    ax1.axvline(x=np.median(max_sims), color="#1e40af", linestyle="-.", linewidth=1.8, label=f"Median = {np.median(max_sims):.3f}")
    ax1.set_xlim(0.70, 1.0)
    ax1.set_xlabel("Max DenseNet-121 Cosine Similarity to Training Set (Zoomed: 0.70–1.00)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Count of Test Images (N=415)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Max Cosine Similarity: Test (N=415) vs. Train+Val (N=2,354)\n[Axis zoomed to 0.70–1.00 to show distribution detail]", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    
    # 3B: Zoomed ECDF [0.70, 1.0]
    sorted_sims = np.sort(max_sims)
    ecdf = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    
    ax2.plot(sorted_sims, ecdf, color="#0284c7", linewidth=2.5, label="Test Images ECDF")
    ax2.axvline(x=0.90, color="#f59e0b", linestyle=":", linewidth=1.5, label="0.90 threshold")
    ax2.axvline(x=0.95, color="#ef4444", linestyle=":", linewidth=1.5, label="0.95 threshold")
    ax2.set_xlim(0.70, 1.0)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Max DenseNet-121 Cosine Similarity to Training Set (Zoomed: 0.70–1.00)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Cumulative Proportion", fontsize=10.5, fontweight="bold")
    ax2.set_title("Empirical Cumulative Distribution Function (ECDF)\n[Axis zoomed to 0.70–1.00 to show upper-tail concentration]", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    
    # Caption noting max-similarity over ~2,350 candidates is high by construction and no grouped-split reference exists
    fig.text(0.5, 0.02, "Note: Maximum similarity over ~2,350 candidate images is high by construction; no grouped-split reference exists for comparison.",
             ha="center", fontsize=9.5, fontstyle="italic", color="#334155")
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "figure3_similarity_distribution_phase3.png", dpi=300)
    plt.close(fig)
    print("Generated figure3_similarity_distribution_phase3.png (zoomed to 0.70-1.0 with caption)")
