"""
scripts/round16_complete_eval.py
Full evaluation for Round 16 Item 2:
- P1_exact: experiment6 / final_evaluation_e2 loader (composite+crop+CLAHE+Lanczos 112x112, raw 0-255 float32)
- P_naive: PIL RGB BILINEAR 112, raw 0-255, no composite/crop/CLAHE
- P2_deployed: current deployed transform_c5 in src/preprocessing.py
- P3_old: P2 + densenet preprocess_input

Runs:
1. 21 test participants comparison against final_test_predictions.csv
2. 80 dev participants (all-5 ensemble and out-of-fold)
3. 101 pooled participants
Reports: accuracy, Wilson 95% CI, predicted-class counts, always-O+ baseline.
"""
import os, sys, math, numpy as np, pandas as pd, tensorflow as tf
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground,
    enhance_mfs100_ridge_contrast, resize_to_target, transform_c5 as deployed_transform_c5
)

classes = ["A+", "B+", "AB+", "O+"]
rev_map = {c: i for i, c in enumerate(classes)}

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95%
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)

# Preprocessing functions
def P1_exact(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

def P_naive(img_path):
    im = Image.open(img_path).convert("RGB")
    im_resized = im.resize((112, 112), Image.Resampling.BILINEAR)
    return np.array(im_resized, dtype=np.float32)

def P2_deployed(img_path):
    arr, _, _ = deployed_transform_c5(img_path)
    return arr

def P3_old(img_path):
    arr, _, _ = deployed_transform_c5(img_path)
    return tf.keras.applications.densenet.preprocess_input(arr.copy())

# Model definition
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

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
saved_csv = pd.read_csv(os.path.join(project_root, "reports/phase3_final_evaluation/final_test_predictions.csv"))
saved_pred = dict(zip(saved_csv["participant_id"], saved_csv["predicted_class"]))
saved_probs = {row["participant_id"]: np.array([
    row["probability_A_plus"], row["probability_B_plus"],
    row["probability_AB_plus"], row["probability_O_plus"]
]) for _, row in saved_csv.iterrows()}

# Pre-load all 101 participants for each pipeline
print("Loading all 101 participants (303 scans)...")
all_pids, all_trues, all_splits = [], [], []
P1_list, Pnaive_list, P2_list, P3_list = [], [], [], []

for pid, group in df.groupby("participant_id"):
    assert len(group) == 3
    s_p1, s_pn, s_p2, s_p3 = [], [], [], []
    for _, row in group.iterrows():
        p = row["scan_path"]
        s_p1.append(P1_exact(p))
        s_pn.append(P_naive(p))
        s_p2.append(P2_deployed(p))
        s_p3.append(P3_old(p))
    all_pids.append(pid)
    all_trues.append(group.iloc[0]["blood_group"])
    all_splits.append(group.iloc[0]["split"])
    P1_list.append(s_p1)
    Pnaive_list.append(s_pn)
    P2_list.append(s_p2)
    P3_list.append(s_p3)

all_pids = np.array(all_pids)
all_trues = np.array(all_trues)
all_splits = np.array(all_splits)
all_trues_idx = np.array([rev_map[b] for b in all_trues])

X_p1 = np.array(P1_list)
X_pn = np.array(Pnaive_list)
X_p2 = np.array(P2_list)
X_p3 = np.array(P3_list)

print("Building model...")
model = build_E2()

# Precompute fold predictions for all 101 for each pipeline
# fold_preds[pipe_name][fold_idx] shape: (101, 4)
pipe_arrays = {
    "P1 (exp6 actual)": X_p1,
    "P_naive (bilinear 112 no crop/clahe)": X_pn,
    "P2 (deployed transform_c5)": X_p2,
    "P3 (old deployed + densenet prep)": X_p3
}

print("Running inferences across all 5 checkpoints...")
pipe_fold_probs = {}
for name, arr in pipe_arrays.items():
    pipe_fold_probs[name] = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(arr, verbose=0)[0]
        pipe_fold_probs[name].append(preds)

# Test mask (21) and Dev mask (80)
test_mask = all_splits == "TEST"
dev_mask = all_splits != "TEST"

# StratifiedGroupKFold on Dev participants
dev_pids_order = all_pids[dev_mask]
dev_trues_idx = all_trues_idx[dev_mask]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Here each participant is one group
fold_splits = list(sgkf.split(np.zeros(len(dev_pids_order)), dev_trues_idx, dev_pids_order))

# Print Results
print("\n" + "="*80)
print("PART 1: 21 TEST PARTICIPANTS VS SAVED CSV")
print("="*80)

test_pids = all_pids[test_mask]
test_trues = all_trues[test_mask]
test_true_idx = all_trues_idx[test_mask]

# Table of 21 test participants
print(f"{'PID':8s} {'True':4s} {'Saved':6s} {'P1':6s} {'Pnaive':6s} {'P2':6s} {'P3':6s} | P1=Sv Pnaive=Sv P2=Sv P3=Sv | MaxDiff P1-Sv  Pn-Sv  P2-Sv  P3-Sv")
print("-" * 115)

# 5-fold ensemble predictions on test set:
pipe_test_ens = {}
for name in pipe_arrays:
    pipe_test_ens[name] = np.mean([pipe_fold_probs[name][k][test_mask] for k in range(5)], axis=0)

for i, pid in enumerate(test_pids):
    t = test_trues[i]
    sv = saved_pred.get(pid, "N/A")
    sv_p = saved_probs.get(pid, np.zeros(4))
    
    c_p1 = classes[np.argmax(pipe_test_ens["P1 (exp6 actual)"][i])]
    c_pn = classes[np.argmax(pipe_test_ens["P_naive (bilinear 112 no crop/clahe)"][i])]
    c_p2 = classes[np.argmax(pipe_test_ens["P2 (deployed transform_c5)"][i])]
    c_p3 = classes[np.argmax(pipe_test_ens["P3 (old deployed + densenet prep)"][i])]
    
    m1 = "YES" if c_p1 == sv else "NO"
    mn = "YES" if c_pn == sv else "NO"
    m2 = "YES" if c_p2 == sv else "NO"
    m3 = "YES" if c_p3 == sv else "NO"
    
    d1 = np.max(np.abs(pipe_test_ens["P1 (exp6 actual)"][i] - sv_p))
    dn = np.max(np.abs(pipe_test_ens["P_naive (bilinear 112 no crop/clahe)"][i] - sv_p))
    d2 = np.max(np.abs(pipe_test_ens["P2 (deployed transform_c5)"][i] - sv_p))
    d3 = np.max(np.abs(pipe_test_ens["P3 (old deployed + densenet prep)"][i] - sv_p))
    
    print(f"{pid:8s} {t:4s} {sv:6s} {c_p1:6s} {c_pn:6s} {c_p2:6s} {c_p3:6s} | {m1:5s} {mn:9s} {m2:5s} {m3:5s} | {d1:.6f} {dn:.6f} {d2:.6f} {d3:.6f}")

print("\n[Summary for 21 Test Participants]")
o_plus_baseline = np.sum(test_trues == "O+") / len(test_trues)
print(f"Always-O+ baseline: {np.sum(test_trues == 'O+')}/{len(test_trues)} = {o_plus_baseline:.4f}")

for name in pipe_arrays:
    ens = pipe_test_ens[name]
    top1 = np.argmax(ens, axis=1)
    acc = np.sum(top1 == test_true_idx) / len(test_trues)
    ci_low, ci_high = wilson_ci(np.sum(top1 == test_true_idx), len(test_trues))
    dist = dict(Counter(classes[c] for c in top1))
    
    # Agreement with saved CSV
    agree = np.sum([classes[top1[i]] == saved_pred[test_pids[i]] for i in range(len(test_pids))])
    max_d = np.max([np.max(np.abs(ens[i] - saved_probs[test_pids[i]])) for i in range(len(test_pids))])
    
    print(f"\n--- {name} ---")
    print(f"  Top-1 agreement with saved CSV: {agree}/21 ({agree/21*100:.1f}%)")
    print(f"  Max abs prob difference vs saved CSV: {max_d:.8f}")
    print(f"  Accuracy: {np.sum(top1 == test_true_idx)}/21 = {acc:.4f} (95% Wilson CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Predicted class distribution: {dist}")
    print(f"  Per-class recall:")
    for ci, cls in enumerate(classes):
        mask_c = test_true_idx == ci
        n_c = np.sum(mask_c)
        rec = np.sum(top1[mask_c] == ci) / n_c if n_c > 0 else 0.0
        print(f"    {cls:4s}: {np.sum(top1[mask_c] == ci)}/{n_c} = {rec:.4f}")

print("\n" + "="*80)
print("PART 2: 80 DEV PARTICIPANTS (ALL-5 ENSEMBLE AND OUT-OF-FOLD)")
print("="*80)
dev_trues = all_trues[dev_mask]
dev_true_idx = all_trues_idx[dev_mask]
dev_o_plus_baseline = np.sum(dev_trues == "O+") / len(dev_trues)
print(f"Always-O+ baseline on Dev (80): {np.sum(dev_trues == 'O+')}/{len(dev_trues)} = {dev_o_plus_baseline:.4f}")

for name in pipe_arrays:
    print(f"\n--- {name} ---")
    # All-5 ensemble
    ens_dev = np.mean([pipe_fold_probs[name][k][dev_mask] for k in range(5)], axis=0)
    top1_ens = np.argmax(ens_dev, axis=1)
    acc_ens = np.sum(top1_ens == dev_true_idx) / len(dev_trues)
    ci_l_ens, ci_h_ens = wilson_ci(np.sum(top1_ens == dev_true_idx), len(dev_trues))
    dist_ens = dict(Counter(classes[c] for c in top1_ens))
    print(f"  [All-5 Ensemble (In-Sample Dev)]")
    print(f"    Accuracy: {np.sum(top1_ens == dev_true_idx)}/80 = {acc_ens:.4f} (95% Wilson CI: [{ci_l_ens:.4f}, {ci_h_ens:.4f}])")
    print(f"    Predicted class distribution: {dist_ens}")
    
    # Out-of-fold (OOF)
    oof_probs = np.zeros((len(dev_trues), 4))
    for fold_i, (train_sub_idx, val_sub_idx) in enumerate(fold_splits):
        # checkpoint index fold_i
        oof_probs[val_sub_idx] = pipe_fold_probs[name][fold_i][dev_mask][val_sub_idx]
    
    top1_oof = np.argmax(oof_probs, axis=1)
    acc_oof = np.sum(top1_oof == dev_true_idx) / len(dev_trues)
    ci_l_oof, ci_h_oof = wilson_ci(np.sum(top1_oof == dev_true_idx), len(dev_trues))
    dist_oof = dict(Counter(classes[c] for c in top1_oof))
    print(f"  [Out-of-Fold (OOF Validation)]")
    print(f"    Accuracy: {np.sum(top1_oof == dev_true_idx)}/80 = {acc_oof:.4f} (95% Wilson CI: [{ci_l_oof:.4f}, {ci_h_oof:.4f}])")
    print(f"    Predicted class distribution: {dist_oof}")

print("\n" + "="*80)
print("PART 3: 101 POOLED PARTICIPANTS (ALL-5 ENSEMBLE)")
print("="*80)
pooled_o_plus_baseline = np.sum(all_trues == "O+") / len(all_trues)
print(f"Always-O+ baseline on Pooled (101): {np.sum(all_trues == 'O+')}/{len(all_trues)} = {pooled_o_plus_baseline:.4f}")

for name in pipe_arrays:
    ens_pooled = np.mean([pipe_fold_probs[name][k] for k in range(5)], axis=0)
    top1_pooled = np.argmax(ens_pooled, axis=1)
    acc_pooled = np.sum(top1_pooled == all_trues_idx) / len(all_trues)
    ci_l_pooled, ci_h_pooled = wilson_ci(np.sum(top1_pooled == all_trues_idx), len(all_trues))
    dist_pooled = dict(Counter(classes[c] for c in top1_pooled))
    print(f"\n--- {name} ---")
    print(f"  Accuracy: {np.sum(top1_pooled == all_trues_idx)}/101 = {acc_pooled:.4f} (95% Wilson CI: [{ci_l_pooled:.4f}, {ci_h_pooled:.4f}])")
    print(f"  Predicted class distribution: {dist_pooled}")

print("\n[DONE COMPLETE EVAL]")
