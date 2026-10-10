"""
scripts/round18_permutation_test.py
Round 18 Item 4: Permutation test with matched statistics on both sides.
- Observed statistic: mean over participant-grouped 5-fold CV (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42))
- Null statistic: mean over the exact same participant-grouped 5-fold CV for each of 200 label shuffles at participant level.
- Evaluated for:
    Variant A: Raw [0, 255] float32 (as E2)
    Variant B: tf.keras.applications.densenet.preprocess_input
- Head architecture:
    Concat 3 scans (3072) -> Dense 512 (ReLU) -> Dropout 0.5 -> Dense 4 (Softmax)
    Plain sparse cross-entropy loss, Adam(lr=1e-3), fixed 20 epochs, batch size 16.
- Metrics reported:
    Observed Accuracy, Observed Balanced Accuracy,
    Null Mean, Null Std, Null 95th Percentile, Null Max,
    Empirical p-value for Accuracy, Empirical p-value for Balanced Accuracy.
"""
import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from collections import Counter
from joblib import Parallel, delayed

sys.stdout.reconfigure(line_buffering=True)

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

class FastNumPyHead:
    def __init__(self, in_dim=3072, hidden=512, out_dim=4, lr=0.001, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(in_dim, hidden) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.randn(hidden, out_dim) * np.sqrt(2.0 / (hidden + out_dim))).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    def fit(self, X, y, epochs=20, batch_size=16, seed=42):
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        for ep in range(epochs):
            perm = rng.permutation(n)
            X_sh = X[perm]; y_sh = y[perm]
            for i in range(0, n, batch_size):
                xb = X_sh[i:i+batch_size]; yb = y_sh[i:i+batch_size]
                bs = xb.shape[0]
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(z1, 0.0)
                drop_mask = (rng.rand(*a1.shape) >= 0.5).astype(np.float32) * 2.0
                a1_drop = a1 * drop_mask
                z2 = a1_drop @ self.W2 + self.b2
                exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
                dz2 = probs.copy()
                dz2[np.arange(bs), yb] -= 1.0
                dz2 /= bs
                dW2 = a1_drop.T @ dz2
                db2 = np.sum(dz2, axis=0)
                da1 = (dz2 @ self.W2.T) * drop_mask
                dz1 = da1 * (z1 > 0.0)
                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0)
                self.t += 1
                b1_pow = 1.0 - 0.9 ** self.t; b2_pow = 1.0 - 0.999 ** self.t
                self.mW1 *= 0.9; self.mW1 += 0.1 * dW1
                self.vW1 *= 0.999; self.vW1 += 0.001 * (dW1 * dW1)
                self.W1 -= self.lr * (self.mW1 / b1_pow) / (np.sqrt(self.vW1 / b2_pow) + 1e-7)
                self.mb1 *= 0.9; self.mb1 += 0.1 * db1
                self.vb1 *= 0.999; self.vb1 += 0.001 * (db1 * db1)
                self.b1 -= self.lr * (self.mb1 / b1_pow) / (np.sqrt(self.vb1 / b2_pow) + 1e-7)
                self.mW2 *= 0.9; self.mW2 += 0.1 * dW2
                self.vW2 *= 0.999; self.vW2 += 0.001 * (dW2 * dW2)
                self.W2 -= self.lr * (self.mW2 / b1_pow) / (np.sqrt(self.vW2 / b2_pow) + 1e-7)
                self.mb2 *= 0.9; self.mb2 += 0.1 * db2
                self.vb2 *= 0.999; self.vb2 += 0.001 * (db2 * db2)
                self.b2 -= self.lr * (self.mb2 / b1_pow) / (np.sqrt(self.vb2 / b2_pow) + 1e-7)

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        return np.argmax(z2, axis=1)

def eval_5fold(X, y_labels, splits, base_seed=42):
    accs, bals = [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        seed = base_seed * 10 + fold_idx
        h = FastNumPyHead(seed=seed)
        h.fit(X[tr_idx], y_labels[tr_idx], epochs=20, batch_size=16, seed=seed)
        preds = h.predict(X[val_idx])
        accs.append(accuracy_score(y_labels[val_idx], preds))
        bals.append(balanced_accuracy_score(y_labels[val_idx], preds))
    return np.mean(accs), np.mean(bals)

def run_single_permutation(X, y_labels, splits, p_idx):
    rng = np.random.RandomState(20000 + p_idx)
    y_perm = rng.permutation(y_labels)
    acc, bal = eval_5fold(X, y_perm, splits, base_seed=100 + p_idx)
    return acc, bal

def run_matched_eval(X, y_labels, groups, name, n_perm=200):
    print("\n" + "="*80)
    print(f"EVALUATING {name}: MATCHED 5-FOLD CV + {n_perm} PERMUTATIONS")
    print("="*80)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y_labels, groups))
    
    # 1. Observed statistic: 5-fold CV mean on true labels
    t0 = time.time()
    obs_acc, obs_bal = eval_5fold(X, y_labels, splits, base_seed=42)
    t_obs = time.time() - t0
    print(f"Observed 5-Fold CV (evaluated in {t_obs:.2f}s):")
    print(f"  Observed Mean Accuracy     : {obs_acc * 100:.2f}%")
    print(f"  Observed Balanced Accuracy : {obs_bal * 100:.2f}%")
    
    # 2. Null statistic: 5-fold CV mean on permuted labels for each shuffle
    print(f"\nRunning {n_perm} label shuffles (each evaluating the identical 5-fold CV)...")
    t0 = time.time()
    null_results = Parallel(n_jobs=8, prefer='threads')(
        delayed(run_single_permutation)(X, y_labels, splits, p)
        for p in range(n_perm)
    )
    t_perm = time.time() - t0
    
    null_accs = np.array([res[0] for res in null_results])
    null_bals = np.array([res[1] for res in null_results])
    
    p_val_acc = (np.sum(null_accs >= obs_acc) + 1.0) / (len(null_accs) + 1.0)
    p_val_bal = (np.sum(null_bals >= obs_bal) + 1.0) / (len(null_bals) + 1.0)
    
    print(f"\nPermutation Test Completed in {t_perm:.2f}s ({n_perm} shuffles):")
    print(f"  Null Accuracy Mean        : {np.mean(null_accs) * 100:.2f}% +/- {np.std(null_accs) * 100:.2f}%")
    print(f"  Null Accuracy 95th Pct    : {np.percentile(null_accs, 95) * 100:.2f}%")
    print(f"  Null Accuracy Max         : {np.max(null_accs) * 100:.2f}%")
    print(f"  Empirical p-value (Acc)   : {p_val_acc:.4f}")
    print(f"  Null Bal Acc Mean         : {np.mean(null_bals) * 100:.2f}% +/- {np.std(null_bals) * 100:.2f}%")
    print(f"  Null Bal Acc 95th Pct     : {np.percentile(null_bals, 95) * 100:.2f}%")
    print(f"  Null Bal Acc Max          : {np.max(null_bals) * 100:.2f}%")
    print(f"  Empirical p-value (Bal)   : {p_val_bal:.4f}")
    
    return {
        "obs_acc": obs_acc,
        "obs_bal": obs_bal,
        "null_acc_mean": np.mean(null_accs),
        "null_acc_std": np.std(null_accs),
        "null_acc_95": np.percentile(null_accs, 95),
        "p_val_acc": p_val_acc,
        "null_bal_mean": np.mean(null_bals),
        "null_bal_std": np.std(null_bals),
        "null_bal_95": np.percentile(null_bals, 95),
        "p_val_bal": p_val_bal
    }

if __name__ == "__main__":
    split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
    df = pd.read_csv(split_csv)

    pids_order = []
    labels_order = []
    grouped = df.groupby("participant_id")
    for pid, grp in grouped:
        assert len(grp) == 3
        pids_order.append(pid)
        labels_order.append(rev_map[grp.iloc[0]["blood_group"]])

    pids_order = np.array(pids_order)
    labels_order = np.array(labels_order)
    N = len(pids_order)
    assert N == 101, f"Expected 101 participants, found {N}"

    o_plus_baseline = np.sum(labels_order == rev_map["O+"]) / N
    print(f"Participant cohort: N = {N}")
    print(f"Class distribution: {dict(Counter([classes[y] for y in labels_order]))}")
    print(f"Always-O+ majority baseline: {np.sum(labels_order == rev_map['O+'])}/{N} = {o_plus_baseline:.4f}")

    cache_dir = os.path.join(project_root, "scratch", "clean_estimate_cache")
    feat_raw_path = os.path.join(cache_dir, "features_variant_a_raw.npy")
    feat_prep_path = os.path.join(cache_dir, "features_variant_b_prep.npy")

    X_raw = np.load(feat_raw_path)
    X_prep = np.load(feat_prep_path)

    res_A = run_matched_eval(X_raw, labels_order, pids_order, "Variant A (Raw [0, 255] as E2)", n_perm=200)
    res_B = run_matched_eval(X_prep, labels_order, pids_order, "Variant B (DenseNet preprocess_input)", n_perm=200)

    print("\n" + "="*85)
    print("FINAL MATCHED PERMUTATION TEST COMPARISON (200 SHUFFLES, 5-FOLD CV ON BOTH SIDES)")
    print("="*85)
    print(f"{'Metric':30s} | {'Variant A (Raw [0,255])':25s} | {'Variant B (preprocess_input)':25s}")
    print("-" * 85)
    print(f"{'Observed 5-Fold Accuracy':30s} | {res_A['obs_acc']*100:.2f}%{' ':19s} | {res_B['obs_acc']*100:.2f}%")
    print(f"{'Observed Balanced Accuracy':30s} | {res_A['obs_bal']*100:.2f}%{' ':19s} | {res_B['obs_bal']*100:.2f}%")
    print(f"{'Always-O+ Baseline':30s} | {o_plus_baseline*100:.2f}%{' ':19s} | {o_plus_baseline*100:.2f}%")
    print(f"{'Null Accuracy Mean +/- SD':30s} | {res_A['null_acc_mean']*100:.2f}% +/- {res_A['null_acc_std']*100:.2f}%{' ':8s} | {res_B['null_acc_mean']*100:.2f}% +/- {res_B['null_acc_std']*100:.2f}%")
    print(f"{'Null Accuracy 95th Percentile':30s} | {res_A['null_acc_95']*100:.2f}%{' ':19s} | {res_B['null_acc_95']*100:.2f}%")
    print(f"{'Empirical p-value (Accuracy)':30s} | {res_A['p_val_acc']:.4f}{' ':20s} | {res_B['p_val_acc']:.4f}")
    print(f"{'Null Balanced Acc Mean +/- SD':30s} | {res_A['null_bal_mean']*100:.2f}% +/- {res_A['null_bal_std']*100:.2f}%{' ':8s} | {res_B['null_bal_mean']*100:.2f}% +/- {res_B['null_bal_std']*100:.2f}%")
    print(f"{'Null Balanced Acc 95th Pct':30s} | {res_A['null_bal_95']*100:.2f}%{' ':19s} | {res_B['null_bal_95']*100:.2f}%")
    print(f"{'Empirical p-value (Bal Acc)':30s} | {res_A['p_val_bal']:.4f}{' ':20s} | {res_B['p_val_bal']:.4f}")
    print("="*85)
