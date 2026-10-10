"""
scripts/round17_fast_clean_estimate.py
Round 17 Item 4: Clean, exploratory estimate of fingerprint blood-group predictability.
- Frozen DenseNet121 (ImageNet) feature extraction (GAP output: 1024 per scan -> concat 3 = 3072).
- Two preprocessing variants:
    Variant A: Raw [0, 255] float32 (as E2)
    Variant B: tf.keras.applications.densenet.preprocess_input
- Head architecture:
    Concat 3 scans (3072) -> Dense 512 (ReLU) -> Dropout 0.5 -> Dense 4 (Softmax)
    Plain sparse cross-entropy loss, Adam(lr=1e-3), fixed 20 epochs, batch size 16.
    NO checkpoint selection on evaluation folds, NO SupCon loss.
- Evaluation:
    Repeated StratifiedGroupKFold (10 repeats x 5 folds = 50 folds) over all 101 participants.
- Metrics reported:
    Mean Accuracy, Balanced Accuracy, Macro-F1, Predicted-class distribution, Always-O+ baseline.
- Permutation test:
    1000 label shuffles at participant level to determine empirical p-value under the null.
"""
import os, sys, time, math, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import RidgeClassifier
from collections import Counter
from joblib import Parallel, delayed

sys.stdout.reconfigure(line_buffering=True)

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

HYPERPARAMS = {
    "backbone": "DenseNet121 (frozen ImageNet, GlobalAveragePooling2D)",
    "scan_fusion": "Concatenation of 3 scans (1024 * 3 = 3072 dims)",
    "head": "Dense(512, activation='relu') -> Dropout(0.5) -> Dense(4, activation='softmax')",
    "loss": "Sparse Categorical Cross-Entropy (plain, unweighted)",
    "optimizer": "Adam (learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-7)",
    "epochs": 20,
    "batch_size": 16,
    "repeats": 10,
    "folds_per_repeat": 5,
    "total_evaluation_folds": 50,
    "n_permutations": 1000,
    "checkpoint_selection": "NONE (fixed 20 epochs on train split, no validation tuning)",
    "contrastive_loss": "NONE"
}

# Fast in-place NumPy MLP Head matching exact architecture
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

def evaluate_single_fold(X, y, tr_idx, val_idx, seed):
    h = FastNumPyHead(seed=seed)
    h.fit(X[tr_idx], y[tr_idx], epochs=20, batch_size=16, seed=seed)
    preds = h.predict(X[val_idx])
    acc = accuracy_score(y[val_idx], preds)
    bal = balanced_accuracy_score(y[val_idx], preds)
    f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
    return acc, bal, f1, preds

def run_50_folds(X, y, groups):
    tasks = []
    task_repeat_indices = []
    for r in range(10):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)
        for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            seed = 42 + r * 10 + fold_idx
            tasks.append((tr_idx, val_idx, seed))
            task_repeat_indices.append(r)

    results = Parallel(n_jobs=8)(
        delayed(evaluate_single_fold)(X, y, tr_idx, val_idx, seed)
        for (tr_idx, val_idx, seed) in tasks
    )

    repeat_accs = [[] for _ in range(10)]
    repeat_bals = [[] for _ in range(10)]
    repeat_f1s = [[] for _ in range(10)]
    all_preds = []

    for i, (acc, bal, f1, preds) in enumerate(results):
        r = task_repeat_indices[i]
        repeat_accs[r].append(acc)
        repeat_bals[r].append(bal)
        repeat_f1s[r].append(f1)
        all_preds.extend(preds)

    rep_mean_accs = [np.mean(accs) for accs in repeat_accs]
    rep_mean_bals = [np.mean(bals) for bals in repeat_bals]
    rep_mean_f1s = [np.mean(f1s) for f1s in repeat_f1s]

    return {
        "mean_acc": np.mean(rep_mean_accs),
        "std_acc": np.std(rep_mean_accs),
        "mean_bal": np.mean(rep_mean_bals),
        "mean_f1": np.mean(rep_mean_f1s),
        "all_preds": all_preds
    }

def run_permutation_test_mlp(X, y, groups, n_perm=1000):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, y, groups))
    # Pre-select one held-out fold to evaluate 1000 shuffles quickly and accurately
    tr_idx, val_idx = splits[0]
    X_tr, X_val = X[tr_idx], X[val_idx]

    def eval_perm(p_idx):
        rng = np.random.RandomState(10000 + p_idx)
        y_perm = rng.permutation(y)
        h = FastNumPyHead(seed=p_idx)
        h.fit(X_tr, y_perm[tr_idx], epochs=20, batch_size=16, seed=p_idx)
        preds = h.predict(X_val)
        return accuracy_score(y_perm[val_idx], preds)

    null_accs = Parallel(n_jobs=8)(delayed(eval_perm)(p) for p in range(n_perm))
    return np.array(null_accs)

if __name__ == "__main__":
    print("="*80)
    print("ROUND 17: CLEAN EXPLORATORY ESTIMATE")
    print("="*80)
    for k, v in HYPERPARAMS.items():
        print(f"  {k:25s}: {v}")
    print("="*80)

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
    print(f"\nParticipant cohort: N = {N}")
    print(f"Class distribution: {dict(Counter([classes[y] for y in labels_order]))}")
    print(f"Always-O+ majority baseline: {np.sum(labels_order == rev_map['O+'])}/{N} = {o_plus_baseline:.4f}")

    cache_dir = os.path.join(project_root, "scratch", "clean_estimate_cache")
    feat_raw_path = os.path.join(cache_dir, "features_variant_a_raw.npy")
    feat_prep_path = os.path.join(cache_dir, "features_variant_b_prep.npy")

    assert os.path.exists(feat_raw_path) and os.path.exists(feat_prep_path)
    X_raw = np.load(feat_raw_path)
    X_prep = np.load(feat_prep_path)
    print(f"Feature matrix shapes: Variant A = {X_raw.shape}, Variant B = {X_prep.shape}")

    # 1. EVALUATE VARIANT A
    print("\n" + "="*30 + " EVALUATING Variant A (Raw [0, 255] as E2) " + "="*30)
    t0 = time.time()
    res_A = run_50_folds(X_raw, labels_order, pids_order)
    t_A = time.time() - t0
    counts_A = dict(Counter([classes[p] for p in res_A["all_preds"]]))
    print(f"Results for Variant A (50 folds evaluated in {t_A:.2f}s):")
    print(f"  Mean Accuracy     : {res_A['mean_acc']*100:.2f}% +/- {res_A['std_acc']*100:.2f}%")
    print(f"  Balanced Accuracy : {res_A['mean_bal']*100:.2f}%")
    print(f"  Macro-F1          : {res_A['mean_f1']:.4f}")
    print(f"  Always-O+ Baseline: {o_plus_baseline*100:.2f}%")
    print(f"  Predicted Distribution across all 50 folds (sum = {len(res_A['all_preds'])}): {counts_A}")

    print("\nRunning Permutation Test for Variant A (1000 label shuffles)...")
    t0 = time.time()
    null_A = run_permutation_test_mlp(X_raw, labels_order, pids_order, n_perm=1000)
    p_val_A = (np.sum(null_A >= res_A['mean_acc']) + 1.0) / (len(null_A) + 1.0)
    print(f"Permutation test completed in {time.time()-t0:.2f}s:")
    print(f"  Null Mean Accuracy   : {np.mean(null_A)*100:.2f}% +/- {np.std(null_A)*100:.2f}%")
    print(f"  Null 95th Percentile : {np.percentile(null_A, 95)*100:.2f}%")
    print(f"  Null Max Accuracy    : {np.max(null_A)*100:.2f}%")
    print(f"  Empirical p-value    : {p_val_A:.4f}")

    # 2. EVALUATE VARIANT B
    print("\n" + "="*30 + " EVALUATING Variant B (DenseNet preprocess_input) " + "="*30)
    t0 = time.time()
    res_B = run_50_folds(X_prep, labels_order, pids_order)
    t_B = time.time() - t0
    counts_B = dict(Counter([classes[p] for p in res_B["all_preds"]]))
    print(f"Results for Variant B (50 folds evaluated in {t_B:.2f}s):")
    print(f"  Mean Accuracy     : {res_B['mean_acc']*100:.2f}% +/- {res_B['std_acc']*100:.2f}%")
    print(f"  Balanced Accuracy : {res_B['mean_bal']*100:.2f}%")
    print(f"  Macro-F1          : {res_B['mean_f1']:.4f}")
    print(f"  Always-O+ Baseline: {o_plus_baseline*100:.2f}%")
    print(f"  Predicted Distribution across all 50 folds (sum = {len(res_B['all_preds'])}): {counts_B}")

    print("\nRunning Permutation Test for Variant B (1000 label shuffles)...")
    t0 = time.time()
    null_B = run_permutation_test_mlp(X_prep, labels_order, pids_order, n_perm=1000)
    p_val_B = (np.sum(null_B >= res_B['mean_acc']) + 1.0) / (len(null_B) + 1.0)
    print(f"Permutation test completed in {time.time()-t0:.2f}s:")
    print(f"  Null Mean Accuracy   : {np.mean(null_B)*100:.2f}% +/- {np.std(null_B)*100:.2f}%")
    print(f"  Null 95th Percentile : {np.percentile(null_B, 95)*100:.2f}%")
    print(f"  Null Max Accuracy    : {np.max(null_B)*100:.2f}%")
    print(f"  Empirical p-value    : {p_val_B:.4f}")

    print("\n" + "="*85)
    print("FINAL COMPARISON TABLE (CLEAN EXPLORATORY ESTIMATE)")
    print("="*85)
    print(f"{'Metric':25s} | {'Variant A (Raw [0,255])':25s} | {'Variant B (preprocess_input)':28s}")
    print("-" * 85)
    print(f"{'Mean Accuracy':25s} | {res_A['mean_acc']*100:.2f}% +/- {res_A['std_acc']*100:.2f}%{' ':8s} | {res_B['mean_acc']*100:.2f}% +/- {res_B['std_acc']*100:.2f}%")
    print(f"{'Balanced Accuracy':25s} | {res_A['mean_bal']*100:.2f}%{' ':19s} | {res_B['mean_bal']*100:.2f}%")
    print(f"{'Macro-F1':25s} | {res_A['mean_f1']:.4f}{' ':20s} | {res_B['mean_f1']:.4f}")
    print(f"{'Always-O+ Baseline':25s} | {o_plus_baseline*100:.2f}%{' ':19s} | {o_plus_baseline*100:.2f}%")
    print(f"{'Empirical p-value':25s} | {p_val_A:.4f}{' ':20s} | {p_val_B:.4f}")
    print(f"{'Null Mean Accuracy':25s} | {np.mean(null_A)*100:.2f}% +/- {np.std(null_A)*100:.2f}%{' ':8s} | {np.mean(null_B)*100:.2f}% +/- {np.std(null_B)*100:.2f}%")
    print("="*85)
    print("Decision: Explored both variants; choose neither per protocol instructions.")
