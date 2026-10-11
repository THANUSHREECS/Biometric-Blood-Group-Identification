"""
scripts/round16_three_pipelines.py
Round 16 Item 2: Three-pipeline comparison for 21 test participants.

P1 = experiment6/final_evaluation_e2 loader: composite+crop+CLAHE+Lanczos+raw float32
P2 = current deployed transform_c5: same as P1 (post-fix)
P3 = old deployed path: P2 + densenet.preprocess_input

NOTE: P1 and P2 call identical functions - this script verifies they agree exactly.
Also verifies determinism by running one participant twice.
Prints one row per participant: id, true, saved_csv, P1, P2, P3 predictions.
"""
import os, sys, numpy as np, pandas as pd, tensorflow as tf
from collections import Counter

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground,
    enhance_mfs100_ridge_contrast, resize_to_target
)

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

# ── Three preprocessing pipelines
def P1(img_path):
    """experiment6/final_evaluation_e2 transform_c5 (verbatim lines 30-36)"""
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

def P2(img_path):
    """Current deployed transform_c5 (post-fix, src/preprocessing.py:247-260)"""
    from src.preprocessing import transform_c5
    arr, _, _ = transform_c5(img_path)
    return arr

def P3(img_path):
    """Old deployed path: P2 + densenet.preprocess_input"""
    arr = P2(img_path)
    return tf.keras.applications.densenet.preprocess_input(arr.copy())

# ── Model builder (verbatim from final_evaluation_e2.py lines 51-72)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])
print("Note: data_augmentation layers called with training=False => identity (deterministic)")

def build_E2():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1 = data_augmentation(s1, training=False)
    a2 = data_augmentation(s2, training=False)
    a3 = data_augmentation(s3, training=False)
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

split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
test_df = df[df["split"] == "TEST"].copy()
saved_csv = pd.read_csv(os.path.join(project_root, "reports/phase3_final_evaluation/final_test_predictions.csv"))
saved_pred = dict(zip(saved_csv["participant_id"], saved_csv["predicted_class"]))
saved_probs = {row["participant_id"]: np.array([
    row["probability_A_plus"], row["probability_B_plus"],
    row["probability_AB_plus"], row["probability_O_plus"]
]) for _, row in saved_csv.iterrows()}

print("\nLoading 21 test participants with all three pipelines...")
pids, trues, p1_arrs, p2_arrs, p3_arrs = [], [], [], [], []
for pid, group in test_df.groupby("participant_id"):
    assert len(group) == 3
    p1_scans, p2_scans, p3_scans = [], [], []
    for _, row in group.iterrows():
        path = row["scan_path"]
        p1_scans.append(P1(path))
        p2_scans.append(P2(path))
        p3_scans.append(P3(path))
    pids.append(pid)
    trues.append(group.iloc[0]["blood_group"])
    p1_arrs.append(p1_scans)
    p2_arrs.append(p2_scans)
    p3_arrs.append(p3_scans)

X1 = np.array(p1_arrs)  # (21,3,112,112,3)
X2 = np.array(p2_arrs)
X3 = np.array(p3_arrs)

print(f"\nPixel diff P1 vs P2: max={np.abs(X1-X2).max():.8f} (should be 0)")

print("\nBuilding model and running ensemble for all 3 pipelines...")
model = build_E2()

def ensemble_predict(X):
    all_probs = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(X, verbose=0)[0]
        all_probs.append(preds)
    return np.mean(all_probs, axis=0)  # (N, 4) float32

probs1 = ensemble_predict(X1)
probs2 = ensemble_predict(X2)
probs3 = ensemble_predict(X3)

# Determinism check: run P1 twice, same participant
print("\n[Determinism check: running participant 0 twice with P1]")
X_single = X1[0:1]
run_a = ensemble_predict(X_single)
run_b = ensemble_predict(X_single)
print(f"  Run A: {run_a[0]}")
print(f"  Run B: {run_b[0]}")
print(f"  Max diff: {np.abs(run_a - run_b).max():.10f}")
print(f"  Deterministic: {np.all(run_a == run_b)}")

print("\n" + "="*120)
print(f"{'PID':8s} {'True':4s} {'Saved':6s} {'P1':6s} {'P2':6s} {'P3':6s} | {'P1=Saved':8s} {'P2=Saved':8s} {'P3=Saved':8s} | MaxDiff P1-Saved  P2-Saved  P3-Saved")
print("-"*120)

p1_top1 = np.argmax(probs1, axis=1)
p2_top1 = np.argmax(probs2, axis=1)
p3_top1 = np.argmax(probs3, axis=1)

agree1, agree2, agree3 = 0, 0, 0
true_labels = []
for i, pid in enumerate(pids):
    t = trues[i]
    true_labels.append(rev_map[t])
    sv = saved_pred.get(pid, "N/A")
    sv_probs = saved_probs.get(pid, np.zeros(4))
    c1 = classes[p1_top1[i]]
    c2 = classes[p2_top1[i]]
    c3 = classes[p3_top1[i]]
    m1 = c1 == sv; m2 = c2 == sv; m3 = c3 == sv
    if m1: agree1 += 1
    if m2: agree2 += 1
    if m3: agree3 += 1
    d1 = np.max(np.abs(probs1[i] - sv_probs))
    d2 = np.max(np.abs(probs2[i] - sv_probs))
    d3 = np.max(np.abs(probs3[i] - sv_probs))
    m1s = "YES" if m1 else "NO"
    m2s = "YES" if m2 else "NO"
    m3s = "YES" if m3 else "NO"
    print(f"{pid:8s} {t:4s} {sv:6s} {c1:6s} {c2:6s} {c3:6s} | {m1s:8s} {m2s:8s} {m3s:8s} | {d1:.6f}  {d2:.6f}  {d3:.6f}")

print("="*120)
print(f"\n[Agreement with saved CSV]")
print(f"  P1 agrees with saved: {agree1}/21")
print(f"  P2 agrees with saved: {agree2}/21")
print(f"  P3 agrees with saved: {agree3}/21")

true_labels = np.array(true_labels)
for name, pred_idx, probs in [("P1", p1_top1, probs1), ("P2", p2_top1, probs2), ("P3", p3_top1, probs3)]:
    acc = np.sum(pred_idx == true_labels) / 21
    dist = dict(Counter(classes[i] for i in pred_idx))
    print(f"\n  {name}: acc={acc:.4f} ({np.sum(pred_idx==true_labels)}/21), pred_dist={dist}")
    for ci, cls in enumerate(classes):
        mask = true_labels == ci
        if mask.sum() > 0:
            recall = np.sum(pred_idx[mask] == ci) / mask.sum()
            print(f"    {cls} recall: {np.sum(pred_idx[mask]==ci)}/{mask.sum()} = {recall:.4f}")

print(f"\nMax prob diff P1 vs P2 (should be 0): {np.abs(probs1-probs2).max():.10f}")
print(f"Max prob diff P2 vs P3 (must be > 0): {np.abs(probs2-probs3).max():.6f}")
print("\n[DONE]")
