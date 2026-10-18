"""
scripts/leakage_audit_exp2_signal_tests.py
Experiment 2: Read-Only Signal-Existence Tests on Saved Predictions.

Analyzes:
1. reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv (80 dev participants)
2. reports/final_evaluation/primary_evaluation_predictions.csv (21 locked test participants)

Evaluates:
- 10,000-shuffle label permutation tests for Accuracy and Macro F1
- Naive baselines (Majority Always-O+, Stratified Random, Uniform Random 25%) with exact binomial p-values
- 10,000-resample non-parametric bootstrap 95% confidence intervals
- Saves all results to JSON, CSV, and generates publication-grade figures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

SEED = 42
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
CANONICAL_CLASSES = ["A+", "AB+", "B+", "O+"]

# Output paths
OUT_DIR = Path("experiments/leakage_audit")
FIG_DIR = Path("reports/leakage_audit/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
dev_file = Path("reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv")
test_file = Path("reports/final_evaluation/primary_evaluation_predictions.csv")

assert dev_file.exists(), f"Missing dev file: {dev_file}"
assert test_file.exists(), f"Missing test file: {test_file}"

df_dev = pd.read_csv(dev_file)
df_test = pd.read_csv(test_file)

print(f"Loaded Dev Predictions: {len(df_dev)} rows")
print(f"Loaded Test Predictions: {len(df_test)} rows")

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CANONICAL_CLASSES, average="macro", zero_division=0)
    return acc, bal_acc, macro_f1

def run_cohort_tests(df, cohort_name, y_true_col, y_pred_col):
    y_true = np.array(df[y_true_col])
    y_pred = np.array(df[y_pred_col])
    n = len(y_true)
    
    obs_acc, obs_bal_acc, obs_f1 = compute_metrics(y_true, y_pred)
    obs_correct = int(np.sum(y_true == y_pred))
    
    print(f"\n==================================================")
    print(f"Cohort: {cohort_name} (N = {n})")
    print(f"Observed Accuracy: {obs_acc:.4f} ({obs_correct}/{n})")
    print(f"Observed Balanced Accuracy: {obs_bal_acc:.4f}")
    print(f"Observed Macro F1: {obs_f1:.4f}")
    
    # Class distribution
    class_counts = {c: int(np.sum(y_true == c)) for c in CANONICAL_CLASSES}
    class_props = {c: count / n for c, count in class_counts.items()}
    maj_class = max(class_counts, key=class_counts.get)
    maj_count = class_counts[maj_class]
    maj_rate = maj_count / n
    strat_rand_rate = sum(p**2 for p in class_props.values())
    unif_rate = 0.25
    
    # Exact binomial tests
    binom_maj = binomtest(obs_correct, n, p=maj_rate, alternative="two-sided")
    binom_strat = binomtest(obs_correct, n, p=strat_rand_rate, alternative="two-sided")
    binom_unif = binomtest(obs_correct, n, p=unif_rate, alternative="two-sided")
    
    print(f"\n--- Baseline Comparisons & Exact Binomial Tests ---")
    print(f"Majority ({maj_class}) Baseline: {maj_rate:.4f} ({maj_count}/{n}) | Binomial p-value: {binom_maj.pvalue:.4e}")
    print(f"Stratified Random Baseline: {strat_rand_rate:.4f} | Binomial p-value: {binom_strat.pvalue:.4e}")
    print(f"Uniform Chance (25%) Baseline: {unif_rate:.4f} | Binomial p-value: {binom_unif.pvalue:.4e}")
    
    # Permutation tests
    print(f"\n--- Running Label Permutation Test ({N_PERMUTATIONS} shuffles, seed={SEED}) ---")
    rng = np.random.default_rng(SEED)
    perm_accs = np.zeros(N_PERMUTATIONS)
    perm_f1s = np.zeros(N_PERMUTATIONS)
    
    for i in range(N_PERMUTATIONS):
        shuffled_y = rng.permutation(y_true)
        p_acc, _, p_f1 = compute_metrics(shuffled_y, y_pred)
        perm_accs[i] = p_acc
        perm_f1s[i] = p_f1
        
    p_val_acc = (np.sum(perm_accs >= obs_acc) + 1) / (N_PERMUTATIONS + 1)
    p_val_f1 = (np.sum(perm_f1s >= obs_f1) + 1) / (N_PERMUTATIONS + 1)
    
    print(f"Permutation Mean Acc: {np.mean(perm_accs):.4f} +/- {np.std(perm_accs):.4f} | Empirical p-value: {p_val_acc:.4e}")
    print(f"Permutation Mean F1:  {np.mean(perm_f1s):.4f} +/- {np.std(perm_f1s):.4f} | Empirical p-value: {p_val_f1:.4e}")
    
    # Bootstrap CIs (10,000 resamples)
    print(f"\n--- Running Non-Parametric Bootstrap ({N_BOOTSTRAP} resamples, seed={SEED}) ---")
    rng_b = np.random.default_rng(SEED + 100)
    boot_accs = np.zeros(N_BOOTSTRAP)
    boot_bal_accs = np.zeros(N_BOOTSTRAP)
    boot_f1s = np.zeros(N_BOOTSTRAP)
    
    indices = np.arange(n)
    for b in range(N_BOOTSTRAP):
        boot_idx = rng_b.choice(indices, size=n, replace=True)
        b_acc, b_bal_acc, b_f1 = compute_metrics(y_true[boot_idx], y_pred[boot_idx])
        boot_accs[b] = b_acc
        boot_bal_accs[b] = b_bal_acc
        boot_f1s[b] = b_f1
        
    ci_acc = [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))]
    ci_bal_acc = [float(np.percentile(boot_bal_accs, 2.5)), float(np.percentile(boot_bal_accs, 97.5))]
    ci_f1 = [float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))]
    
    print(f"Accuracy 95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}] (mean: {np.mean(boot_accs):.4f})")
    print(f"Balanced Accuracy 95% CI: [{ci_bal_acc[0]:.4f}, {ci_bal_acc[1]:.4f}] (mean: {np.mean(boot_bal_accs):.4f})")
    print(f"Macro F1 95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}] (mean: {np.mean(boot_f1s):.4f})")
    
    results = {
        "cohort": cohort_name,
        "n_participants": n,
        "class_distribution": class_counts,
        "observed": {
            "accuracy": float(obs_acc),
            "correct_count": obs_correct,
            "balanced_accuracy": float(obs_bal_acc),
            "macro_f1": float(obs_f1)
        },
        "baselines": {
            "majority_class": maj_class,
            "majority_accuracy": float(maj_rate),
            "majority_binom_pvalue": float(binom_maj.pvalue),
            "stratified_random_accuracy": float(strat_rand_rate),
            "stratified_random_binom_pvalue": float(binom_strat.pvalue),
            "uniform_random_accuracy": float(unif_rate),
            "uniform_random_binom_pvalue": float(binom_unif.pvalue)
        },
        "permutation_test": {
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "mean_acc": float(np.mean(perm_accs)),
            "std_acc": float(np.std(perm_accs)),
            "empirical_p_accuracy": float(p_val_acc),
            "mean_f1": float(np.mean(perm_f1s)),
            "std_f1": float(np.std(perm_f1s)),
            "empirical_p_macro_f1": float(p_val_f1)
        },
        "bootstrap_ci_95": {
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED + 100,
            "accuracy": {"mean": float(np.mean(boot_accs)), "ci_95": ci_acc},
            "balanced_accuracy": {"mean": float(np.mean(boot_bal_accs)), "ci_95": ci_bal_acc},
            "macro_f1": {"mean": float(np.mean(boot_f1s)), "ci_95": ci_f1}
        }
    }
    
    return results, perm_accs, perm_f1s

res_dev, perm_dev_acc, perm_dev_f1 = run_cohort_tests(
    df_dev, "Development_OOF_80", y_true_col="true_label", y_pred_col="predicted_label"
)

res_test, perm_test_acc, perm_test_f1 = run_cohort_tests(
    df_test, "Primary_Locked_Test_21", y_true_col="true_label", y_pred_col="predicted_label"
)

# 2. Save full metrics to JSON
all_results = {
    "experiment": "EXP-2: Signal-Existence Tests on Saved Predictions",
    "timestamp": pd.Timestamp.now().isoformat(),
    "development_80_participants": res_dev,
    "primary_test_21_participants": res_test
}

json_path = OUT_DIR / "exp2_signal_tests_results.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n[SAVED] JSON results written to {json_path}")

# 3. Save permutation distributions to CSV
df_dist = pd.DataFrame({
    "dev_perm_accuracy": perm_dev_acc,
    "dev_perm_macro_f1": perm_dev_f1,
    "test_perm_accuracy": perm_test_acc,
    "test_perm_macro_f1": perm_test_f1
})
csv_path = OUT_DIR / "exp2_permutation_distributions.csv"
df_dist.to_csv(csv_path, index=False)
print(f"[SAVED] Permutation distributions saved to {csv_path}")

# 4. Generate Publication-Quality Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Subplot 1: Dev Accuracy
ax = axes[0, 0]
ax.hist(perm_dev_acc, bins=35, color="#4C72B0", alpha=0.7, edgecolor="black", density=True, label="Permutation Null (10,000)")
ax.axvline(res_dev["observed"]["accuracy"], color="#C44E52", linestyle="--", linewidth=2.5,
           label=f"Observed OOF Acc: {res_dev['observed']['accuracy']:.2%} (p = {res_dev['permutation_test']['empirical_p_accuracy']:.4f})")
ax.axvline(res_dev["baselines"]["majority_accuracy"], color="#55A868", linestyle=":", linewidth=2,
           label=f"Majority (O+): {res_dev['baselines']['majority_accuracy']:.2%}")
ax.set_title("Dev Cohort (N=80): Accuracy Permutation Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")
ax.legend(loc="upper right", fontsize=9)

# Subplot 2: Dev Macro F1
ax = axes[0, 1]
ax.hist(perm_dev_f1, bins=35, color="#4C72B0", alpha=0.7, edgecolor="black", density=True, label="Permutation Null (10,000)")
ax.axvline(res_dev["observed"]["macro_f1"], color="#C44E52", linestyle="--", linewidth=2.5,
           label=f"Observed OOF F1: {res_dev['observed']['macro_f1']:.4f} (p = {res_dev['permutation_test']['empirical_p_macro_f1']:.4f})")
ax.set_title("Dev Cohort (N=80): Macro F1 Permutation Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Macro F1-Score")
ax.set_ylabel("Density")
ax.legend(loc="upper right", fontsize=9)

# Subplot 3: Test Accuracy
ax = axes[1, 0]
ax.hist(perm_test_acc, bins=25, color="#8172B2", alpha=0.7, edgecolor="black", density=True, label="Permutation Null (10,000)")
ax.axvline(res_test["observed"]["accuracy"], color="#C44E52", linestyle="--", linewidth=2.5,
           label=f"Observed Test Acc: {res_test['observed']['accuracy']:.2%} (p = {res_test['permutation_test']['empirical_p_accuracy']:.4f})")
ax.axvline(res_test["baselines"]["majority_accuracy"], color="#55A868", linestyle=":", linewidth=2,
           label=f"Majority (O+): {res_test['baselines']['majority_accuracy']:.2%}")
ax.set_title("Locked Test Cohort (N=21): Accuracy Permutation Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Density")
ax.legend(loc="upper right", fontsize=9)

# Subplot 4: Test Macro F1
ax = axes[1, 1]
ax.hist(perm_test_f1, bins=25, color="#8172B2", alpha=0.7, edgecolor="black", density=True, label="Permutation Null (10,000)")
ax.axvline(res_test["observed"]["macro_f1"], color="#C44E52", linestyle="--", linewidth=2.5,
           label=f"Observed Test F1: {res_test['observed']['macro_f1']:.4f} (p = {res_test['permutation_test']['empirical_p_macro_f1']:.4f})")
ax.axvline(0.3500, color="orange", linestyle="-.", linewidth=2, label="Pre-registered Target (0.3500)")
ax.set_title("Locked Test Cohort (N=21): Macro F1 Permutation Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Macro F1-Score")
ax.set_ylabel("Density")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig_path = FIG_DIR / "exp2_permutation_null_distribution.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[SAVED] Figure written to {fig_path}")
print("=== EXPERIMENT 2 COMPLETED SUCCESSFULLY ===")
