"""
scripts/round17_clean_estimate.py
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
import os, sys, time, math, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import transform_c5

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

# Hyperparameter declaration
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

print("="*80)
print("ROUND 17: CLEAN ESTIMATE HYPERPARAMETERS")
print("="*80)
for k, v in HYPERPARAMS.items():
    print(f"  {k:25s}: {v}")
print("="*80)

# Load dataset metadata
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

# Feature Extraction
cache_dir = os.path.join(project_root, "scratch", "clean_estimate_cache")
os.makedirs(cache_dir, exist_ok=True)
feat_raw_path = os.path.join(cache_dir, "features_variant_a_raw.npy")
feat_prep_path = os.path.join(cache_dir, "features_variant_b_prep.npy")

if os.path.exists(feat_raw_path) and os.path.exists(feat_prep_path):
    print("\nLoading cached DenseNet121 features...")
    X_raw = np.load(feat_raw_path)
    X_prep = np.load(feat_prep_path)
else:
    print("\nExtracting frozen DenseNet121 features for all 303 scans...")
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(112, 112, 3), pooling="avg")
    base_model.trainable = False

    raw_scans = []
    prep_scans = []
    for pid in pids_order:
        grp = df[df["participant_id"] == pid]
        p_raw, p_prep = [], []
        for _, row in grp.iterrows():
            arr, _, _ = transform_c5(row["scan_path"]) # (112, 112, 3) raw [0, 255]
            p_raw.append(arr)
            p_prep.append(tf.keras.applications.densenet.preprocess_input(arr.copy()))
        raw_scans.append(p_raw)
        prep_scans.append(p_prep)

    raw_scans = np.array(raw_scans) # (101, 3, 112, 112, 3)
    prep_scans = np.array(prep_scans)

    # Flatten to (303, 112, 112, 3) for batched feature extraction
    flat_raw = raw_scans.reshape(303, 112, 112, 3)
    flat_prep = prep_scans.reshape(303, 112, 112, 3)

    feat_flat_raw = base_model.predict(flat_raw, batch_size=32, verbose=0) # (303, 1024)
    feat_flat_prep = base_model.predict(flat_prep, batch_size=32, verbose=0) # (303, 1024)

    X_raw = feat_flat_raw.reshape(101, 3 * 1024)
    X_prep = feat_flat_prep.reshape(101, 3 * 1024)

    np.save(feat_raw_path, X_raw)
    np.save(feat_prep_path, X_prep)
    print("Features extracted and cached.")

print(f"Feature matrix shapes: Variant A = {X_raw.shape}, Variant B = {X_prep.shape}")

# Model builder for the head
def build_head():
    inp = tf.keras.Input(shape=(3072,))
    x = tf.keras.layers.Dense(512, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

# Fast vectorized NumPy MLP for permutation test
class FastNumPyHead:
    def __init__(self, in_dim=3072, hidden=512, out_dim=4, lr=0.001, seed=42):
        rng = np.random.RandomState(seed)
        # He initialization
        self.W1 = rng.randn(in_dim, hidden).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        # Xavier initialization
        self.W2 = rng.randn(hidden, out_dim).astype(np.float32) * np.sqrt(2.0 / (hidden + out_dim))
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        # Adam moments
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
                xb = X_sh[i:i+batch_size]
                yb = y_sh[i:i+batch_size]
                bs = xb.shape[0]

                # Forward
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(z1, 0.0) # ReLU
                # Dropout 0.5 training
                drop_mask = (rng.rand(*a1.shape) >= 0.5).astype(np.float32) * 2.0
                a1_drop = a1 * drop_mask

                z2 = a1_drop @ self.W2 + self.b2
                # Softmax
                exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

                # Gradient cross-entropy
                dz2 = probs.copy()
                dz2[np.arange(bs), yb] -= 1.0
                dz2 /= bs

                dW2 = a1_drop.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1 = (dz2 @ self.W2.T) * drop_mask
                dz1 = da1 * (z1 > 0.0)

                dW1 = xb.T @ dz1
                db1 = np.sum(dz1, axis=0)

                # Adam updates
                self.t += 1
                for param, grad, m_arr, v_arr in [
                    (self.W1, dW1, self.mW1, self.vW1),
                    (self.b1, db1, self.mb1, self.vb1),
                    (self.W2, dW2, self.mW2, self.vW2),
                    (self.b2, db2, self.mb2, self.vb2)
                ]:
                    m_arr[:] = 0.9 * m_arr + 0.1 * grad
                    v_arr[:] = 0.999 * v_arr + 0.001 * (grad ** 2)
                    m_hat = m_arr / (1.0 - 0.9 ** self.t)
                    v_hat = v_arr / (1.0 - 0.999 ** self.t)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)

    def predict_proba(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0) # Inference: no dropout
        z2 = a1 @ self.W2 + self.b2
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

# Run Repeated StratifiedGroupKFold evaluation
def evaluate_pipeline(X, y, groups, name):
    print(f"\n{'='*30} EVALUATING {name} {'='*30}")
    t0 = time.time()
    repeat_accs = []
    repeat_bal_accs = []
    repeat_macro_f1s = []
    all_preds = []

    for r in range(10):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)
        fold_accs = []
        fold_bal = []
        fold_f1 = []
        r_preds = []

        for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            head = FastNumPyHead(seed=42 + r * 10 + fold_idx)
            head.fit(X_tr, y_tr, epochs=20, batch_size=16, seed=100 + fold_idx)
            probs = head.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)

            fold_accs.append(accuracy_score(y_val, preds))
            fold_bal.append(balanced_accuracy_score(y_val, preds))
            fold_f1.append(f1_score(y_val, preds, average="macro", zero_division=0))
            r_preds.extend(preds)

        repeat_accs.append(np.mean(fold_accs))
        repeat_bal_accs.append(np.mean(fold_bal))
        repeat_macro_f1s.append(np.mean(fold_f1))
        all_preds.extend(r_preds)

    elapsed = time.time() - t0
    mean_acc = np.mean(repeat_accs)
    std_acc = np.std(repeat_accs)
    mean_bal = np.mean(repeat_bal_accs)
    mean_f1 = np.mean(repeat_macro_f1s)
    pred_counts = dict(Counter([classes[p] for p in all_preds]))

    print(f"Results for {name} (50 folds evaluated in {elapsed:.2f}s):")
    print(f"  Mean Accuracy     : {mean_acc * 100:.2f}% +/- {std_acc * 100:.2f}%")
    print(f"  Balanced Accuracy : {mean_bal * 100:.2f}%")
    print(f"  Macro-F1          : {mean_f1:.4f}")
    print(f"  Always-O+ Baseline: {o_plus_baseline * 100:.2f}%")
    print(f"  Predicted Distribution across all 50 folds (sum = {len(all_preds)}): {pred_counts}")

    # Permutation test (1000 shuffles at participant level)
    print(f"\nRunning Permutation Test (1000 label shuffles at participant level)...")
    t_perm_start = time.time()
    null_accs = []
    sgkf_single = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf_single.split(X, y, groups))

    for p_idx in range(1000):
        rng = np.random.RandomState(10000 + p_idx)
        y_perm = rng.permutation(y)
        f_accs = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            head = FastNumPyHead(seed=fold_idx)
            head.fit(X[tr_idx], y_perm[tr_idx], epochs=20, batch_size=16, seed=fold_idx)
            probs = head.predict_proba(X[val_idx])
            preds = np.argmax(probs, axis=1)
            f_accs.append(accuracy_score(y_perm[val_idx], preds))
        null_accs.append(np.mean(f_accs))

    null_accs = np.array(null_accs)
    p_value = (np.sum(null_accs >= mean_acc) + 1.0) / (len(null_accs) + 1.0)
    perm_elapsed = time.time() - t_perm_start

    print(f"Permutation test completed in {perm_elapsed:.2f}s (1000 shuffles):")
    print(f"  Null Mean Accuracy   : {np.mean(null_accs) * 100:.2f}% +/- {np.std(null_accs) * 100:.2f}%")
    print(f"  Null 95th Percentile : {np.percentile(null_accs, 95) * 100:.2f}%")
    print(f"  Null Max Accuracy    : {np.max(null_accs) * 100:.2f}%")
    print(f"  Empirical p-value    : {p_value:.4f}")

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_bal": mean_bal,
        "mean_f1": mean_f1,
        "pred_counts": pred_counts,
        "p_value": p_value,
        "null_mean": np.mean(null_accs),
        "null_std": np.std(null_accs)
    }

# Run both variants
res_A = evaluate_pipeline(X_raw, labels_order, pids_order, "Variant A (Raw [0, 255] as E2)")
res_B = evaluate_pipeline(X_prep, labels_order, pids_order, "Variant B (DenseNet preprocess_input)")

print("\n" + "="*80)
print("FINAL COMPARISON TABLE (CLEAN EXPLORATORY ESTIMATE)")
print("="*80)
print(f"{'Metric':25s} | {'Variant A (Raw [0,255])':25s} | {'Variant B (preprocess_input)':28s}")
print("-" * 85)
print(f"{'Mean Accuracy':25s} | {res_A['mean_acc']*100:.2f}% +/- {res_A['std_acc']*100:.2f}%{' ':8s} | {res_B['mean_acc']*100:.2f}% +/- {res_B['std_acc']*100:.2f}%")
print(f"{'Balanced Accuracy':25s} | {res_A['mean_bal']*100:.2f}%{' ':19s} | {res_B['mean_bal']*100:.2f}%")
print(f"{'Macro-F1':25s} | {res_A['mean_f1']:.4f}{' ':20s} | {res_B['mean_f1']:.4f}")
print(f"{'Always-O+ Baseline':25s} | {o_plus_baseline*100:.2f}%{' ':19s} | {o_plus_baseline*100:.2f}%")
print(f"{'Empirical p-value':25s} | {res_A['p_value']:.4f}{' ':20s} | {res_B['p_value']:.4f}")
print(f"{'Null Mean Accuracy':25s} | {res_A['null_mean']*100:.2f}% +/- {res_A['null_std']*100:.2f}%{' ':8s} | {res_B['null_mean']*100:.2f}% +/- {res_B['null_std']*100:.2f}%")
print("="*85)
print("Decision: Explored both variants; choose neither per protocol instructions.")
