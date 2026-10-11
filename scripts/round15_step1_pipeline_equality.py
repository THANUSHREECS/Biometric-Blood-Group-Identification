"""
scripts/round15_step1_pipeline_equality.py
Round 15 Item 1b: Full-precision pipeline equality check.
Run all 303 scans through:
  (a) experiment pipeline (local transform_c5 identical to training/final_evaluation_e2)
  (b) deployed path: src.preprocessing.transform_c5 + same 5 checkpoints
Compare float32 probabilities directly from model (no rounding).
Report:
  - max abs probability difference per scan
  - % identical top-1
  - whether any difference exists at all
"""
import os, sys, numpy as np, pandas as pd, tensorflow as tf

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground,
    enhance_mfs100_ridge_contrast, resize_to_target, transform_c5 as deployed_transform_c5
)

# ── Experiment pipeline (verbatim copy of experiment6_fusion_supcon.py / final_evaluation_e2.py lines 45-51)
def experiment_transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

# ── Deployed pipeline wrapper (same functions, returns tuple)
def deployed_transform(img_path):
    arr, _, _ = deployed_transform_c5(img_path)
    return arr

# ── Model builder (verbatim from final_evaluation_e2.py)
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_E2():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1, a2, a3 = data_aug(s1, training=False), data_aug(s2, training=False), data_aug(s3, training=False)
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

checkpoint_paths = [
    os.path.join(project_root, f"models/phase3_fusion_metric_learning/E2/fold_{i}_best.weights.h5")
    for i in range(1, 6)
]

# ── Load all 303 scans
split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
print(f"Total rows in split CSV: {len(df)} (expect 303)")

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

print("Building arrays for all 101 participants (3 scans each)...")
exp_X, dep_X, all_pids, all_bg, all_splits = [], [], [], [], []

for pid, group in df.groupby("participant_id"):
    assert len(group) == 3, f"PID {pid} has {len(group)} scans"
    exp_scans, dep_scans = [], []
    for _, row in group.iterrows():
        path = row["scan_path"]
        e = experiment_transform_c5(path)
        d = deployed_transform(path)
        exp_scans.append(e)
        dep_scans.append(d)
    exp_X.append(exp_scans)
    dep_X.append(dep_scans)
    all_pids.append(pid)
    all_bg.append(group.iloc[0]["blood_group"])
    all_splits.append(group.iloc[0]["split"])

exp_X = np.array(exp_X)   # (101, 3, 112, 112, 3)
dep_X = np.array(dep_X)

print(f"\n[Pixel-level check BEFORE model]")
pixel_diff = np.abs(exp_X - dep_X)
print(f"  Max abs pixel diff across all 303x112x112x3 values: {pixel_diff.max():.8f}")
print(f"  Mean abs pixel diff: {pixel_diff.mean():.8f}")
print(f"  Pixel-identical: {np.all(pixel_diff == 0)}")

print("\nBuilding model and running ensemble...")
model = build_E2()

exp_all_probs, dep_all_probs = [], []
for cp in checkpoint_paths:
    model.load_weights(cp)
    ep = model.predict(exp_X, verbose=0)[0]   # (101, 4)
    dp = model.predict(dep_X, verbose=0)[0]   # (101, 4)
    exp_all_probs.append(ep)
    dep_all_probs.append(dp)

exp_ens = np.mean(exp_all_probs, axis=0)   # (101, 4) — float32, no rounding
dep_ens = np.mean(dep_all_probs, axis=0)   # (101, 4)

prob_diff = np.abs(exp_ens - dep_ens)
print(f"\n[Probability-level check (float32, no rounding)]")
print(f"  Max abs prob diff: {prob_diff.max():.10f}")
print(f"  Mean abs prob diff: {prob_diff.mean():.10f}")
print(f"  Max-diff participant idx: {prob_diff.max(axis=1).argmax()}")

exp_top1 = np.argmax(exp_ens, axis=1)
dep_top1 = np.argmax(dep_ens, axis=1)
top1_match = np.sum(exp_top1 == dep_top1)
print(f"  Top-1 identical: {top1_match}/101 ({top1_match/101*100:.2f}%)")

# Split results
print("\n[Per-split breakdown]")
splits = np.array(all_splits)
bg_arr = np.array(all_bg)
pid_arr = np.array(all_pids)

for split_name in ["TEST", "TRAIN", "VALIDATION"]:
    mask = splits == split_name
    if mask.sum() == 0:
        continue
    n = mask.sum()
    y_true = np.array([rev_map[b] for b in bg_arr[mask]])
    exp_pred = exp_top1[mask]
    dep_pred = dep_top1[mask]
    exp_acc = np.sum(exp_pred == y_true) / n
    dep_acc = np.sum(dep_pred == y_true) / n
    match = np.sum(exp_pred == dep_pred)
    print(f"  {split_name} (N={n}): exp_acc={exp_pred.__class__.__name__} exp_acc={exp_acc:.4f}, dep_acc={dep_acc:.4f}, top1_match={match}/{n}")
    from collections import Counter
    print(f"    Exp pred dist: {dict(Counter(classes[i] for i in exp_pred))}")
    print(f"    Dep pred dist: {dict(Counter(classes[i] for i in dep_pred))}")
    # Accuracy
    exp_correct = np.sum(exp_pred == y_true)
    dep_correct = np.sum(dep_pred == y_true)
    print(f"    Exp correct: {exp_correct}/{n} = {exp_correct/n:.4f}")
    print(f"    Dep correct: {dep_correct}/{n} = {dep_correct/n:.4f}")

print("\n[DONE]")
