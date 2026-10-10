import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from scipy.stats import binomtest

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import (
    load_image_to_pil,
    composite_alpha_on_white,
    crop_mfs100_foreground,
    enhance_mfs100_ridge_contrast,
    resize_to_target,
    transform_c5 as deployed_transform_c5
)

def wilson_ci(k, n, confidence=0.95):
    z = 1.95996
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return center - half_width, center + half_width

def experiment_transform_c5(img_path):
    """Experiment6 / final_evaluation pipeline: raw [0, 255] float32 array."""
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

def build_experiment_E2():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    feat1, feat2, feat3 = pool(base(scan1)), pool(base(scan2)), pool(base(scan3))
    concat = tf.keras.layers.Concatenate()([feat1, feat2, feat3])
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

def run_ensemble(model, checkpoints, X):
    probs = []
    for cp in checkpoints:
        model.load_weights(cp)
        preds = model.predict(X, batch_size=32, verbose=0)[0]
        probs.append(preds)
    return np.mean(probs, axis=0)

def main():
    print("="*75, flush=True)
    print("STEP 3, 4, 5: REPRODUCE SAVED RESULTS & PIPELINE COMPARISON", flush=True)
    print("="*75, flush=True)

    classes = ["A+", "B+", "AB+", "O+"]
    rev_class_mapping = {c: i for i, c in enumerate(classes)}
    class_mapping = {i: c for i, c in enumerate(classes)}

    split_csv_path = "reports/phase3_participant_benchmark/participant_split.csv"
    df_splits = pd.read_csv(split_csv_path)

    checkpoint_paths = [
        f"models/phase3_fusion_metric_learning/E2/fold_{i}_best.weights.h5"
        for i in range(1, 6)
    ]

    exp_model = build_experiment_E2()

    # -------------------------------------------------------------
    # 3.1: Reproduce on 21 Locked-Test Participants
    # -------------------------------------------------------------
    print("\n--- 3.1: Evaluating 21 Locked-Test Participants ---", flush=True)
    test_df = df_splits[df_splits["split"] == "TEST"].copy()
    test_X_exp, test_X_dep, test_y, test_pids = [], [], [], []
    for pid, group in test_df.groupby("participant_id"):
        paths = group["scan_path"].tolist()
        scans_exp = [experiment_transform_c5(p) for p in paths]
        scans_dep = [deployed_transform_c5(p)[0] for p in paths]
        test_X_exp.append(scans_exp)
        test_X_dep.append(scans_dep)
        test_y.append(group.iloc[0]["blood_group"])
        test_pids.append(pid)

    test_X_exp = np.array(test_X_exp, dtype=np.float32) # (21, 3, 112, 112, 3)
    test_X_dep = np.array(test_X_dep, dtype=np.float32)
    test_y = np.array(test_y)

    # Score with Experiment Pipeline
    exp_ensemble_probs = run_ensemble(exp_model, checkpoint_paths, test_X_exp)
    exp_preds = [class_mapping[i] for i in np.argmax(exp_ensemble_probs, axis=1)]

    acc_exp_21 = accuracy_score(test_y, exp_preds)
    k_exp_21 = np.sum(test_y == exp_preds)
    print(f"Experiment Pipeline 21 Test Accuracy: {acc_exp_21*100:.2f}% ({k_exp_21}/21)", flush=True)
    print("Predicted Class Counts (Experiment Pipeline):", pd.Series(exp_preds).value_counts().to_dict(), flush=True)
    print("Per-class Recall (Experiment Pipeline):", flush=True)
    for c in classes:
        mask = (test_y == c)
        c_tot = np.sum(mask)
        c_corr = np.sum(np.array(exp_preds)[mask] == c)
        rec = c_corr / c_tot if c_tot > 0 else 0
        print(f"  {c}: {c_corr}/{c_tot} ({rec*100:.2f}%)", flush=True)

    # Score with Deployed Pipeline
    dep_ensemble_probs = run_ensemble(exp_model, checkpoint_paths, test_X_dep)
    dep_preds = [class_mapping[i] for i in np.argmax(dep_ensemble_probs, axis=1)]

    acc_dep_21 = accuracy_score(test_y, dep_preds)
    k_dep_21 = np.sum(test_y == dep_preds)
    print(f"\nDeployed Pipeline 21 Test Accuracy: {acc_dep_21*100:.2f}% ({k_dep_21}/21)", flush=True)
    print("Predicted Class Counts (Deployed Pipeline):", pd.Series(dep_preds).value_counts().to_dict(), flush=True)
    print("Per-class Recall (Deployed Pipeline):", flush=True)
    for c in classes:
        mask = (test_y == c)
        c_tot = np.sum(mask)
        c_corr = np.sum(np.array(dep_preds)[mask] == c)
        rec = c_corr / c_tot if c_tot > 0 else 0
        print(f"  {c}: {c_corr}/{c_tot} ({rec*100:.2f}%)", flush=True)

    # -------------------------------------------------------------
    # 3.2: Rebuild 5 Folds on 80 Dev Participants & Compare fold_results.csv
    # -------------------------------------------------------------
    print("\n--- 3.2: Rebuilding 5 Folds on 80 Dev Participants ---", flush=True)
    dev_df = df_splits[df_splits["split"].isin(["TRAIN", "VALIDATION"])].copy()
    dev_X_exp, dev_X_dep, dev_y, dev_pids = [], [], [], []
    for pid, group in dev_df.groupby("participant_id"):
        paths = group["scan_path"].tolist()
        dev_X_exp.append([experiment_transform_c5(p) for p in paths])
        dev_X_dep.append([deployed_transform_c5(p)[0] for p in paths])
        dev_y.append(group.iloc[0]["blood_group"])
        dev_pids.append(pid)

    dev_X_exp = np.array(dev_X_exp, dtype=np.float32) # (80, 3, 112, 112, 3)
    dev_X_dep = np.array(dev_X_dep, dtype=np.float32)
    dev_y = np.array(dev_y)
    dev_y_idx = np.array([rev_class_mapping[y] for y in dev_y])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(sgkf.split(dev_X_exp, dev_y_idx, dev_pids))

    saved_fold_accs = [0.4375, 0.5000, 0.5000, 0.5625, 0.6875]
    print(f"{'Fold':<6} {'Saved Acc':<12} {'Exp Pipeline Acc':<18} {'Dep Pipeline Acc':<18} {'Match Saved?'}", flush=True)
    oof_preds_exp = [None] * len(dev_y)
    oof_preds_dep = [None] * len(dev_y)

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        cp = checkpoint_paths[fold]
        exp_model.load_weights(cp)

        # Exp pipeline val
        preds_exp_val = exp_model.predict(dev_X_exp[val_idx], verbose=0)[0]
        pred_classes_exp = [class_mapping[i] for i in np.argmax(preds_exp_val, axis=1)]
        acc_exp_val = accuracy_score(dev_y[val_idx], pred_classes_exp)

        # Dep pipeline val
        preds_dep_val = exp_model.predict(dev_X_dep[val_idx], verbose=0)[0]
        pred_classes_dep = [class_mapping[i] for i in np.argmax(preds_dep_val, axis=1)]
        acc_dep_val = accuracy_score(dev_y[val_idx], pred_classes_dep)

        for idx_in_val, orig_idx in enumerate(val_idx):
            oof_preds_exp[orig_idx] = pred_classes_exp[idx_in_val]
            oof_preds_dep[orig_idx] = pred_classes_dep[idx_in_val]

        matches = abs(acc_exp_val - saved_fold_accs[fold]) < 1e-4
        print(f"Fold {fold+1:<2} {saved_fold_accs[fold]*100:<11.2f}% {acc_exp_val*100:<17.2f}% {acc_dep_val*100:<17.2f}% {str(matches)}", flush=True)

    # -------------------------------------------------------------
    # 3.3: Input Divergence Analysis for One Participant
    # -------------------------------------------------------------
    print("\n--- 3.3: Input Divergence Analysis for One Participant ---", flush=True)
    sample_path = test_df.iloc[0]["scan_path"]
    sample_pid = test_df.iloc[0]["participant_id"]
    print(f"Sample Participant: {sample_pid}, Image: {sample_path}", flush=True)

    inp_exp = experiment_transform_c5(sample_path)
    inp_dep, _, meta = deployed_transform_c5(sample_path)

    print(f"Experiment Input: shape={inp_exp.shape}, dtype={inp_exp.dtype}, min={inp_exp.min():.4f}, max={inp_exp.max():.4f}, mean={inp_exp.mean():.4f}, std={inp_exp.std():.4f}", flush=True)
    print(f"Deployed Input:   shape={inp_dep.shape}, dtype={inp_dep.dtype}, min={inp_dep.min():.4f}, max={inp_dep.max():.4f}, mean={inp_dep.mean():.4f}, std={inp_dep.std():.4f}", flush=True)
    max_diff = np.max(np.abs(inp_exp - inp_dep))
    print(f"Max Absolute Difference: {max_diff:.4f}", flush=True)

    # Step trace
    raw_pil = load_image_to_pil(sample_path)
    rgb_exp = composite_alpha_on_white(raw_pil)
    rgb_dep = composite_alpha_on_white(raw_pil)
    diff_comp = np.max(np.abs(np.array(rgb_exp) - np.array(rgb_dep)))

    crop_exp, _ = crop_mfs100_foreground(rgb_exp)
    crop_dep, _ = crop_mfs100_foreground(rgb_dep)
    diff_crop = np.max(np.abs(np.array(crop_exp) - np.array(crop_dep)))

    enh_exp = enhance_mfs100_ridge_contrast(crop_exp)
    enh_dep = enhance_mfs100_ridge_contrast(crop_dep)
    diff_enh = np.max(np.abs(np.array(enh_exp) - np.array(enh_dep)))

    res_exp = resize_to_target(enh_exp, target_size=(112, 112))
    res_dep = resize_to_target(enh_dep, target_size=(112, 112))
    diff_res = np.max(np.abs(np.array(res_exp) - np.array(res_dep)))

    arr_exp = np.array(res_exp, dtype=np.float32)
    arr_dep = tf.keras.applications.densenet.preprocess_input(np.array(res_dep, dtype=np.float32))

    print("\nStep-by-step divergence trace:", flush=True)
    print(f"  1. Composite alpha: max diff = {diff_comp}", flush=True)
    print(f"  2. Crop foreground: max diff = {diff_crop}", flush=True)
    print(f"  3. CLAHE enhance:   max diff = {diff_enh}", flush=True)
    print(f"  4. Resize 112x112:  max diff = {diff_res}", flush=True)
    print(f"  5. Normalisation:   arr_exp (raw [0,255]) vs arr_dep (densenet.preprocess_input) -> max diff = {np.max(np.abs(arr_exp - arr_dep)):.4f}", flush=True)
    print("DIVERGENCE FOUND: Step 5 (Normalisation: raw [0, 255] float32 vs densenet.preprocess_input)", flush=True)

    # -------------------------------------------------------------
    # STEP 4: COLLAPSE CHECK
    # -------------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("STEP 4: COLLAPSE CHECK", flush=True)
    print("="*70, flush=True)

    probs_dev_exp = run_ensemble(exp_model, checkpoint_paths, dev_X_exp)
    probs_dev_dep = run_ensemble(exp_model, checkpoint_paths, dev_X_dep)

    pooled_y = np.concatenate([test_y, dev_y])
    pooled_X_exp = np.concatenate([test_X_exp, dev_X_exp])
    pooled_X_dep = np.concatenate([test_X_dep, dev_X_dep])

    probs_pooled_exp = run_ensemble(exp_model, checkpoint_paths, pooled_X_exp)
    probs_pooled_dep = run_ensemble(exp_model, checkpoint_paths, pooled_X_dep)

    # 415 test images (subsampled or full)
    df_415 = pd.read_csv("outputs/abo_positive_4class/test_predictions.csv")
    X_415_exp = np.array([[experiment_transform_c5(p)]*3 for p in df_415["image_path"]], dtype=np.float32)
    X_415_dep = np.array([[deployed_transform_c5(p)[0]]*3 for p in df_415["image_path"]], dtype=np.float32)

    probs_415_exp = run_ensemble(exp_model, checkpoint_paths, X_415_exp)
    probs_415_dep = run_ensemble(exp_model, checkpoint_paths, X_415_dep)

    def analyze_collapse(name, probs, pipeline_name):
        preds = [class_mapping[i] for i in np.argmax(probs, axis=1)]
        counts = pd.Series(preds).value_counts().to_dict()
        for c in classes:
            counts.setdefault(c, 0)
        n = len(preds)
        a_share = counts["A+"] / n
        stds = {classes[i]: float(np.std(probs[:, i])) for i in range(4)}
        print(f"\n[{name}] - {pipeline_name} (N={n}):", flush=True)
        print(f"  Predicted counts: {counts}", flush=True)
        print(f"  A+ share: {a_share*100:.2f}%", flush=True)
        print(f"  Class prob std: {stds}", flush=True)

    analyze_collapse("21 Locked Test", exp_ensemble_probs, "Experiment Pipeline")
    analyze_collapse("21 Locked Test", dep_ensemble_probs, "Deployed Pipeline")

    analyze_collapse("80 Dev Participants", probs_dev_exp, "Experiment Pipeline")
    analyze_collapse("80 Dev Participants", probs_dev_dep, "Deployed Pipeline")

    analyze_collapse("101 Pooled", probs_pooled_exp, "Experiment Pipeline")
    analyze_collapse("101 Pooled", probs_pooled_dep, "Deployed Pipeline")

    analyze_collapse("415 Dataset Images", probs_415_exp, "Experiment Pipeline")
    analyze_collapse("415 Dataset Images", probs_415_dep, "Deployed Pipeline")

    # -------------------------------------------------------------
    # STEP 5: DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("STEP 5: DESCRIPTIVE STATISTICS", flush=True)
    print("="*70, flush=True)

    # 5.1: 21 Locked Test
    k_test = k_exp_21
    n_test = 21
    acc_test = k_test / n_test
    ci_test_low, ci_test_high = wilson_ci(k_test, n_test)
    btest = binomtest(k_test, n_test, p=8/21, alternative="greater")
    print("\n5.1: 21 Locked Test Participants:", flush=True)
    print(f"  Accuracy: {acc_test*100:.2f}% ({k_test}/{n_test})", flush=True)
    print(f"  Wilson 95% CI: [{ci_test_low*100:.2f}%, {ci_test_high*100:.2f}%]", flush=True)
    print(f"  Always-O+ Baseline: {8/21*100:.2f}% (8/21)", flush=True)
    print(f"  Exact Binomial Test p-value vs Always-O+ (p=8/21): {btest.pvalue:.4f}", flush=True)

    # 5.2: 80 Dev Participants (Out-Of-Fold only)
    k_oof = np.sum(np.array(oof_preds_exp) == dev_y)
    n_oof = len(dev_y)
    acc_oof = k_oof / n_oof
    ci_oof_low, ci_oof_high = wilson_ci(k_oof, n_oof)
    o_dev_baseline = np.sum(dev_y == "O+") / n_oof
    print("\n5.2: 80 Development Participants (Out-Of-Fold Only):", flush=True)
    print(f"  Accuracy: {acc_oof*100:.2f}% ({k_oof}/{n_oof})", flush=True)
    print(f"  Wilson 95% CI: [{ci_oof_low*100:.2f}%, {ci_oof_high*100:.2f}%]", flush=True)
    print(f"  Always-O+ Baseline: {o_dev_baseline*100:.2f}% ({np.sum(dev_y == 'O+')}/{n_oof})", flush=True)
    print("  Note: Checkpoint selection (picking epoch with highest val F1) makes this optimistic.", flush=True)

    # 5.3: 101 Pooled
    k_pooled_exp = np.sum(pooled_y == [class_mapping[i] for i in np.argmax(probs_pooled_exp, axis=1)])
    k_pooled_dep = np.sum(pooled_y == [class_mapping[i] for i in np.argmax(probs_pooled_dep, axis=1)])
    ci_p_exp = wilson_ci(k_pooled_exp, 101)
    ci_p_dep = wilson_ci(k_pooled_dep, 101)
    print("\n5.3: 101 Pooled Participants (CONTAMINATED: includes 80 development participants):", flush=True)
    print(f"  Experiment Pipeline Accuracy: {k_pooled_exp/101*100:.2f}% ({k_pooled_exp}/101), Wilson CI: [{ci_p_exp[0]*100:.2f}%, {ci_p_exp[1]*100:.2f}%]", flush=True)
    print(f"  Deployed Pipeline Accuracy:   {k_pooled_dep/101*100:.2f}% ({k_pooled_dep}/101), Wilson CI: [{ci_p_dep[0]*100:.2f}%, {ci_p_dep[1]*100:.2f}%]", flush=True)
    print(f"  Always-O+ Baseline:           {37/101*100:.2f}% (37/101)", flush=True)

if __name__ == "__main__":
    main()
