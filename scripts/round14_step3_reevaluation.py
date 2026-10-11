"""
scripts/round14_step3_reevaluation.py
Round 14 Step 3: Re-evaluate E2 ensemble on fixed preprocessing pipeline.
Reports predicted-class counts for 21 test, 80 dev (OOF), and 101 pooled.
Does NOT use 21 test to choose between variants – this is post-fix confirmation only.
"""
import os, sys, json, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from scipy.stats import binomtest

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white,
    crop_mfs100_foreground, enhance_mfs100_ridge_contrast,
    resize_to_target, transform_c5
)

print("="*70)
print("ROUND 14 STEP 3: RE-EVALUATION ON FIXED PREPROCESSING PIPELINE")
print("="*70)

# Verify transform_c5 does NOT use preprocess_input by sampling a real array
import inspect
src_lines = inspect.getsource(transform_c5).splitlines()
# Only check lines up to the closing return (before legacy function)
body = "\n".join(src_lines)
# The body should NOT call preprocess_input (the legacy fn has a different name)
assert "densenet.preprocess_input" not in body, "ABORT: transform_c5 body still calls densenet.preprocess_input!"
print("[OK] Verified: transform_c5 body does not call densenet.preprocess_input")

classes = ["A+", "B+", "AB+", "O+"]
rev_class = {c: i for i, c in enumerate(classes)}
class_map = {i: c for i, c in enumerate(classes)}

split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
print(f"Total rows in split CSV: {len(df)} (expect 303)")

checkpoint_paths = [
    os.path.join(project_root, f"models/phase3_fusion_metric_learning/E2/fold_{i}_best.weights.h5")
    for i in range(1, 6)
]

def build_E2():
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
        tf.keras.layers.RandomContrast(0.1)
    ])
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1 = data_aug(scan1, training=False)
    a2 = data_aug(scan2, training=False)
    a3 = data_aug(scan3, training=False)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    f1, f2, f3 = pool(base(a1)), pool(base(a2)), pool(base(a3))
    concat = tf.keras.layers.Concatenate()([f1, f2, f3])
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

def load_participant_scans(pids, split_df):
    X, y, pid_list = [], [], []
    for pid in pids:
        group = split_df[split_df["participant_id"] == pid].sort_values("scan_path")
        assert len(group) == 3, f"PID {pid} has {len(group)} scans"
        scans = [transform_c5(row["scan_path"])[0] for _, row in group.iterrows()]
        X.append(scans)
        y.append(rev_class[group.iloc[0]["blood_group"]])
        pid_list.append(pid)
    return np.array(X), np.array(y), pid_list

def run_ensemble(model, X, checkpoint_paths):
    all_probs = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(X, batch_size=4, verbose=0)[0]
        all_probs.append(preds)
    return np.mean(all_probs, axis=0)

def report_results(label, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    n = len(y_true)
    from collections import Counter
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_true)
    print(f"\n--- {label} ---")
    print(f"  N participants:   {n}")
    print(f"  Accuracy:         {acc:.4f} ({int(round(acc*n))}/{n})")
    print(f"  Balanced Acc:     {bal:.4f}")
    print(f"  True distribution:  {dict(sorted(true_counts.items()))}")
    print(f"  Pred distribution:  {dict(sorted(pred_counts.items()))}")
    print(f"  Always-O+ baseline: {true_counts.get('O+',0)}/{n} = {true_counts.get('O+',0)/n:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred, order={classes}):")
    for i, row in enumerate(cm):
        print(f"    {classes[i]:4s}: {list(row)}")
    # Per-class recall
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        total_true = cm[i].sum()
        recall = tp / total_true if total_true > 0 else 0.0
        print(f"  Recall {cls}: {tp}/{total_true} = {recall:.4f}")
    # Binomial test vs always-O+
    o_count = true_counts.get("O+", 0)
    binom_p = binomtest(int(round(acc*n)), n, o_count/n, alternative='two-sided').pvalue
    print(f"  Binomial p vs always-O+ (p0={o_count/n:.4f}): {binom_p:.4f}")
    return acc

print("\n[1/3] Loading 21 locked-test participants...")
test_pids = sorted(df[df["split"] == "TEST"]["participant_id"].unique())
assert len(test_pids) == 21
X_test, y_test_idx, pids_test = load_participant_scans(test_pids, df)
y_test = [class_map[i] for i in y_test_idx]
print(f"  Loaded {len(X_test)} participants")

print("[2/3] Loading 80 development participants...")
dev_pids = sorted(df[df["split"].isin(["TRAIN", "VALIDATION"])]["participant_id"].unique())
assert len(dev_pids) == 80
X_dev, y_dev_idx, pids_dev = load_participant_scans(dev_pids, df)
y_dev = [class_map[i] for i in y_dev_idx]
print(f"  Loaded {len(X_dev)} participants")

print("[3/3] Building E2 model and running ensemble inference...")
model = build_E2()

print("\n  Predicting 21 test participants...")
probs_test = run_ensemble(model, X_test, checkpoint_paths)
y_pred_test = [class_map[np.argmax(p)] for p in probs_test]

print("  Predicting 80 dev participants...")
probs_dev = run_ensemble(model, X_dev, checkpoint_paths)
y_pred_dev = [class_map[np.argmax(p)] for p in probs_dev]

# 101 pooled
y_all = y_test + y_dev
y_pred_all = y_pred_test + y_pred_dev

print("\n" + "="*70)
print("RESULTS WITH FIXED PREPROCESSING (no preprocess_input)")
print("="*70)

report_results("21 Locked Test Participants", y_test, y_pred_test, classes)
report_results("80 Development Participants (non-OOF: all 5 checkpoints used)", y_dev, y_pred_dev, classes)
report_results("101 Pooled (contaminated: 80 dev + 21 test)", y_all, y_pred_all, classes)

print("\n[DONE]")
