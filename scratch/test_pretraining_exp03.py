#!/usr/bin/env python3
"""
scratch/test_pretraining_exp03.py
=================================
Automated Pre-Training Verification Test Suite for EXP-03.
Validates architecture, losses, splits, mapping, and checkpoint hashes WITHOUT training.
"""

import sys
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.exp03.run_exp03 import (
    build_E2_model, build_augmentation_layer, supervised_contrastive_loss,
    MacroF1Checkpoint, NATIVE_CLASSES, CANONICAL_CLASSES, SEED, BATCH_SIZE
)

def run_tests():
    test_results = {}
    print("=" * 70)
    print("RUNNING EXP-03 PRE-TRAINING AUDIT TESTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST 1: Checkpoint Hashes
    # -------------------------------------------------------------
    orig_dir = PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2"
    expected_hashes = {
        "fold_1": "7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d",
        "fold_2": "9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede",
        "fold_3": "8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3",
        "fold_4": "cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519",
        "fold_5": "aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d"
    }
    hash_pass = True
    for i in range(1, 6):
        cp = orig_dir / f"fold_{i}_best.weights.h5"
        if not cp.is_file():
            hash_pass = False
            break
        h = hashlib.sha256(cp.read_bytes()).hexdigest()
        if h != expected_hashes[f"fold_{i}"]:
            hash_pass = False
            break
    test_results["Checkpoint_Hashes_Preserved"] = "PASS" if hash_pass else "FAIL"
    print(f"Test 1: Checkpoint Hashes Preserved: {test_results['Checkpoint_Hashes_Preserved']}")

    # -------------------------------------------------------------
    # TEST 2: Manifest & Cohort Quarantine
    # -------------------------------------------------------------
    split_csv = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
    df = pd.read_csv(split_csv)
    dev_df = df[df["split"].isin(["TRAIN", "VALIDATION"])]
    test_df = df[df["split"] == "TEST"]
    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    c2_pass = (
        len(dev_pids) == 80 and
        len(test_pids) == 21 and
        len(dev_pids & test_pids) == 0 and
        len(dev_df) == 240
    )
    test_results["Cohort_Quarantine_And_Isolation"] = "PASS" if c2_pass else "FAIL"
    print(f"Test 2: Cohort Quarantine & Isolation: {test_results['Cohort_Quarantine_And_Isolation']}")

    # -------------------------------------------------------------
    # TEST 3: Cross-Validation Splits
    # -------------------------------------------------------------
    grouped = dev_df.groupby("participant_id").first().reset_index()
    pids = grouped["participant_id"].values
    y_labels = grouped["blood_group"].values
    native_map = {c: i for i, c in enumerate(NATIVE_CLASSES)}
    y_native = np.array([native_map[lbl] for lbl in y_labels])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dummy_X = np.zeros((len(pids), 1))
    folds = list(sgkf.split(dummy_X, y_native, pids))

    c3_pass = True
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        t_pids = set(pids[t_idx])
        v_pids = set(pids[v_idx])
        if len(t_pids) != 64 or len(v_pids) != 16 or len(t_pids & v_pids) > 0:
            c3_pass = False
        t_counts = pd.Series(y_labels[t_idx]).value_counts().to_dict()
        if any(t_counts.get(c, 0) == 0 for c in NATIVE_CLASSES):
            c3_pass = False
    test_results["Cross_Validation_5_Fold_Splits"] = "PASS" if c3_pass else "FAIL"
    print(f"Test 3: 5-Fold Stratified Group Splits: {test_results['Cross_Validation_5_Fold_Splits']}")

    # -------------------------------------------------------------
    # TEST 4: Class Mapping
    # -------------------------------------------------------------
    # Native: 0: A+, 1: B+, 2: AB+, 3: O+
    # Canonical: 0: A+, 1: AB+, 2: B+, 3: O+
    # Remap: [p[0], p[2], p[1], p[3]]
    c4_pass = True
    test_cases = [
        ([1.0, 0.0, 0.0, 0.0], "A+"),
        ([0.0, 1.0, 0.0, 0.0], "B+"),
        ([0.0, 0.0, 1.0, 0.0], "AB+"),
        ([0.0, 0.0, 0.0, 1.0], "O+")
    ]
    for p_nat, expected_label in test_cases:
        p_canon = np.array([p_nat[0], p_nat[2], p_nat[1], p_nat[3]])
        pred_label = CANONICAL_CLASSES[np.argmax(p_canon)]
        if pred_label != expected_label:
            c4_pass = False
    test_results["Class_Order_Mapping"] = "PASS" if c4_pass else "FAIL"
    print(f"Test 4: Class Order Mapping: {test_results['Class_Order_Mapping']}")

    # -------------------------------------------------------------
    # TEST 5: Model Construction & Shapes for all 3 Arms
    # -------------------------------------------------------------
    c5_pass = True
    for arm in ["arm0", "arm1", "arm2"]:
        m = build_E2_model(arm)
        x_dummy = tf.zeros((2, 3, 112, 112, 3), dtype=tf.float32)
        out = m(x_dummy, training=False)
        if out[0].shape != (2, 4) or out[1].shape != (2, 128):
            c5_pass = False
    test_results["Model_Architecture_All_Arms"] = "PASS" if c5_pass else "FAIL"
    print(f"Test 5: Model Architecture All Arms: {test_results['Model_Architecture_All_Arms']}")

    # -------------------------------------------------------------
    # TEST 6: Augmentation Inactive During Eval
    # -------------------------------------------------------------
    c6_pass = True
    for arm in ["arm0", "arm1", "arm2"]:
        aug = build_augmentation_layer(arm)
        x_img = tf.random.uniform((2, 112, 112, 3), 0, 255)
        out_eval = aug(x_img, training=False)
        max_diff = float(np.max(np.abs(x_img.numpy() - out_eval.numpy())))
        if max_diff > 1e-5:
            c6_pass = False
    test_results["Augmentation_Inactive_In_Eval"] = "PASS" if c6_pass else "FAIL"
    print(f"Test 6: Augmentation Inactive in Eval: {test_results['Augmentation_Inactive_In_Eval']}")

    # -------------------------------------------------------------
    # TEST 7: Supervised Contrastive Loss & Gradients
    # -------------------------------------------------------------
    c7_pass = True
    # Standard batch
    labels_batch = tf.constant([0, 1, 0, 1, 2, 3], dtype=tf.int32)
    feats_batch = tf.random.normal((6, 128))
    with tf.GradientTape() as tape:
        tape.watch(feats_batch)
        loss_val = supervised_contrastive_loss(labels_batch, feats_batch)
    grads = tape.gradient(loss_val, feats_batch)
    if not (np.isfinite(loss_val.numpy()) and np.all(np.isfinite(grads.numpy()))):
        c7_pass = False

    # Edge case: No positive pairs
    labels_no_pos = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    feats_no_pos = tf.random.normal((4, 128))
    with tf.GradientTape() as tape:
        tape.watch(feats_no_pos)
        loss_no_pos = supervised_contrastive_loss(labels_no_pos, feats_no_pos)
    grads_no_pos = tape.gradient(loss_no_pos, feats_no_pos)
    if not (np.isfinite(loss_no_pos.numpy()) and np.all(np.isfinite(grads_no_pos.numpy()))):
        c7_pass = False
    test_results["SupCon_Loss_And_Gradients"] = "PASS" if c7_pass else "FAIL"
    print(f"Test 7: SupCon Loss & Gradients: {test_results['SupCon_Loss_And_Gradients']}")

    # -------------------------------------------------------------
    # TEST 8: Checkpoint Selection & Early Stopping Gate
    # -------------------------------------------------------------
    cb_temp = MacroF1Checkpoint("dummy_path.weights.h5", np.zeros((4, 3, 112, 112, 3)), np.array([0, 1, 2, 3]))
    c8_pass = (cb_temp.min_epochs == 100 and cb_temp.patience == 15)
    test_results["Checkpoint_Selection_And_Gate"] = "PASS" if c8_pass else "FAIL"
    print(f"Test 8: Checkpoint Selection & Gate: {test_results['Checkpoint_Selection_And_Gate']}")

    print("=" * 70)
    all_pass = all(v == "PASS" for v in test_results.values())
    print(f"OVERALL PRE-TRAINING AUDIT STATUS: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)
    return test_results

if __name__ == "__main__":
    results = run_tests()
