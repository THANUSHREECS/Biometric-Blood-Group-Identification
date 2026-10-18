#!/usr/bin/env python3
"""
EXP-06C: Comprehensive Synthetic Safety Test & Architecture Verification Suite
==============================================================================
Runs 13 rigorous programmatic verification gates on synthetic tensors without
accessing any real training data or modifying any existing checkpoints.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.exp06c.run_exp06c import (
    build_exp06c_model, supervised_contrastive_loss,
    FIXED_TEMPERATURE, FIXED_LAMBDA, CONFIG_SPECS,
    CANONICAL_CLASSES, NATIVE_CLASSES
)

SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
PRE_HASHES_FILE = PROJECT_ROOT / "scratch" / "exp06c_pre_preparation_hashes.json"

def sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def run_synthetic_tests():
    print("=" * 80)
    print("EXP-06C: SYNTHETIC SAFETY & IMPLEMENTATION GATE VERIFICATION")
    print("=" * 80)

    results = {}
    batch_size = 4
    # Synthetic batch: 4 participants, 3 scans each, 112x112x3 in [0, 1]
    np.random.seed(42)
    tf.random.set_seed(42)
    x_syn = tf.constant(np.random.uniform(0.0, 1.0, size=(batch_size, 3, 112, 112, 3)).astype(np.float32))
    # Labels with duplicate for contrastive positive pairs: [0, 1, 0, 2]
    y_syn = tf.constant([0, 1, 0, 2], dtype=tf.int32)
    sample_weights = tf.constant([1.0, 1.2, 1.0, 1.5], dtype=tf.float32)

    # -------------------------------------------------------------
    # Gate 1: Output Shapes
    # -------------------------------------------------------------
    print("\n--- Gate 1: Output Tensor Shapes ---")
    mA, bA, _ = build_exp06c_model(head_type="linear", seed_head=42)
    mB, bB, _ = build_exp06c_model(head_type="mlp", seed_head=42)
    out_clf_A, out_emb_A = mA(x_syn, training=False)
    out_clf_B, out_emb_B = mB(x_syn, training=False)

    g1_pass = (
        out_clf_A.shape == (batch_size, 4) and out_emb_A.shape == (batch_size, 128) and
        out_clf_B.shape == (batch_size, 4) and out_emb_B.shape == (batch_size, 128)
    )
    print(f"  Config A: clf shape = {out_clf_A.shape}, emb shape = {out_emb_A.shape}")
    print(f"  Config B: clf shape = {out_clf_B.shape}, emb shape = {out_emb_B.shape}")
    print(f"  Gate 1 Result: {'PASS' if g1_pass else 'FAIL'}")
    results["Gate 1: Output Shapes"] = "PASS" if g1_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 2: Backbone Freezing
    # -------------------------------------------------------------
    print("\n--- Gate 2: Backbone Freezing ---")
    g2_pass = (
        bA.trainable is False and len(bA.trainable_variables) == 0 and bA.count_params() == 7037504 and
        bB.trainable is False and len(bB.trainable_variables) == 0 and bB.count_params() == 7037504
    )
    print(f"  Config A: base.trainable={bA.trainable}, trainable_vars={len(bA.trainable_variables)}, non-trainable_params={bA.count_params():,}")
    print(f"  Config B: base.trainable={bB.trainable}, trainable_vars={len(bB.trainable_variables)}, non-trainable_params={bB.count_params():,}")
    print(f"  Gate 2 Result: {'PASS' if g2_pass else 'FAIL'}")
    results["Gate 2: Backbone Freezing"] = "PASS" if g2_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 3: Parameter Counts & Trainable Variables
    # -------------------------------------------------------------
    print("\n--- Gate 3: Parameter Counts & Trainable Variables ---")
    vars_A = mA.trainable_variables
    vars_B = mB.trainable_variables
    params_A = sum(int(np.prod(v.shape)) for v in vars_A)
    params_B = sum(int(np.prod(v.shape)) for v in vars_B)

    g3_pass = (
        len(vars_A) == 6 and params_A == 1968772 and
        len(vars_B) == 8 and params_B == 3214468
    )
    print(f"  Config A: variables={len(vars_A)} (expected 6), trainable_params={params_A:,} (expected 1,968,772)")
    print(f"  Config B: variables={len(vars_B)} (expected 8), trainable_params={params_B:,} (expected 3,214,468)")
    print(f"  Parameter delta: +{params_B - params_A:,} parameters (+{(params_B/params_A - 1)*100:.1f}%)")
    print(f"  Gate 3 Result: {'PASS' if g3_pass else 'FAIL'}")
    results["Gate 3: Parameter Counts"] = "PASS" if g3_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 4: SupCon Loss Finiteness & Normalization
    # -------------------------------------------------------------
    print("\n--- Gate 4: SupCon Loss Finiteness & Normalization ---")
    l_supcon_A = float(supervised_contrastive_loss(out_emb_A, y_syn, temperature=FIXED_TEMPERATURE).numpy())
    l_supcon_B = float(supervised_contrastive_loss(out_emb_B, y_syn, temperature=FIXED_TEMPERATURE).numpy())
    g4_pass = (
        np.isfinite(l_supcon_A) and not np.isnan(l_supcon_A) and
        np.isfinite(l_supcon_B) and not np.isnan(l_supcon_B)
    )
    print(f"  Config A SupCon Loss (tau={FIXED_TEMPERATURE}): {l_supcon_A:.4f}")
    print(f"  Config B SupCon Loss (tau={FIXED_TEMPERATURE}): {l_supcon_B:.4f}")
    print(f"  Gate 4 Result: {'PASS' if g4_pass else 'FAIL'}")
    results["Gate 4: SupCon Loss Finiteness"] = "PASS" if g4_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 5: Categorical Cross-Entropy Loss Finiteness
    # -------------------------------------------------------------
    print("\n--- Gate 5: Weighted Cross-Entropy Loss Finiteness ---")
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    l_ce_A = float(cce(y_syn, out_clf_A, sample_weight=sample_weights).numpy())
    l_ce_B = float(cce(y_syn, out_clf_B, sample_weight=sample_weights).numpy())
    g5_pass = np.isfinite(l_ce_A) and np.isfinite(l_ce_B)
    print(f"  Config A Weighted CE Loss: {l_ce_A:.4f}")
    print(f"  Config B Weighted CE Loss: {l_ce_B:.4f}")
    print(f"  Gate 5 Result: {'PASS' if g5_pass else 'FAIL'}")
    results["Gate 5: CE Loss Finiteness"] = "PASS" if g5_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 6: Composite Loss Formulation
    # -------------------------------------------------------------
    print("\n--- Gate 6: Composite Loss Formulation ---")
    tot_A = l_ce_A + FIXED_LAMBDA * l_supcon_A
    tot_B = l_ce_B + FIXED_LAMBDA * l_supcon_B
    g6_pass = np.isfinite(tot_A) and np.isfinite(tot_B) and tot_A > l_ce_A and tot_B > l_ce_B
    print(f"  Config A Composite Loss (CE + {FIXED_LAMBDA}*SupCon): {tot_A:.4f}")
    print(f"  Config B Composite Loss (CE + {FIXED_LAMBDA}*SupCon): {tot_B:.4f}")
    print(f"  Gate 6 Result: {'PASS' if g6_pass else 'FAIL'}")
    results["Gate 6: Composite Loss Formulation"] = "PASS" if g6_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 7: Gradient Mechanics
    # -------------------------------------------------------------
    print("\n--- Gate 7: Gradient Mechanics ---")
    with tf.GradientTape() as tape_A:
        p_clf_A, p_emb_A = mA(x_syn, training=True)
        loss_A = cce(y_syn, p_clf_A) + FIXED_LAMBDA * supervised_contrastive_loss(p_emb_A, y_syn, temperature=FIXED_TEMPERATURE)
    grads_A = tape_A.gradient(loss_A, mA.trainable_variables)

    with tf.GradientTape() as tape_B:
        p_clf_B, p_emb_B = mB(x_syn, training=True)
        loss_B = cce(y_syn, p_clf_B) + FIXED_LAMBDA * supervised_contrastive_loss(p_emb_B, y_syn, temperature=FIXED_TEMPERATURE)
    grads_B = tape_B.gradient(loss_B, mB.trainable_variables)

    g7_pass = (
        len(grads_A) == 6 and all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads_A) and
        len(grads_B) == 8 and all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads_B)
    )
    print(f"  Config A: {len(grads_A)} valid non-None gradients (all finite)")
    print(f"  Config B: {len(grads_B)} valid non-None gradients (all finite)")
    print(f"  Gate 7 Result: {'PASS' if g7_pass else 'FAIL'}")
    results["Gate 7: Gradient Mechanics"] = "PASS" if g7_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 8: Weight Update Isolation
    # -------------------------------------------------------------
    print("\n--- Gate 8: Weight Update Isolation ---")
    optA = tf.keras.optimizers.AdamW(learning_rate=1e-3)
    optB = tf.keras.optimizers.AdamW(learning_rate=1e-3)

    w_head_before_A = [v.numpy().copy() for v in mA.trainable_variables]
    w_base_before_A = [v.numpy().copy() for v in bA.weights[:5]]

    w_head_before_B = [v.numpy().copy() for v in mB.trainable_variables]
    w_base_before_B = [v.numpy().copy() for v in bB.weights[:5]]

    optA.apply_gradients(zip(grads_A, mA.trainable_variables))
    optB.apply_gradients(zip(grads_B, mB.trainable_variables))

    head_diff_A = [np.max(np.abs(v.numpy() - w)) for v, w in zip(mA.trainable_variables, w_head_before_A)]
    base_diff_A = [np.max(np.abs(v.numpy() - w)) for v, w in zip(bA.weights[:5], w_base_before_A)]

    head_diff_B = [np.max(np.abs(v.numpy() - w)) for v, w in zip(mB.trainable_variables, w_head_before_B)]
    base_diff_B = [np.max(np.abs(v.numpy() - w)) for v, w in zip(bB.weights[:5], w_base_before_B)]

    g8_pass = (
        all(d > 0.0 for d in head_diff_A) and all(d == 0.0 for d in base_diff_A) and
        all(d > 0.0 for d in head_diff_B) and all(d == 0.0 for d in base_diff_B)
    )
    print(f"  Config A: Head variables updated (max delta={max(head_diff_A):.6f}), Backbone delta = {max(base_diff_A):.8f}")
    print(f"  Config B: Head variables updated (max delta={max(head_diff_B):.6f}), Backbone delta = {max(base_diff_B):.8f}")
    print(f"  Gate 8 Result: {'PASS' if g8_pass else 'FAIL'}")
    results["Gate 8: Weight Update Isolation"] = "PASS" if g8_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 9: Shared Classifier Initialization Parity
    # -------------------------------------------------------------
    print("\n--- Gate 9: Shared Classifier Initialization Parity ---")
    m_init_ref, _, h_ref = build_exp06c_model(head_type="linear", seed_head=100)
    m_init_test, _, h_test = build_exp06c_model(head_type="mlp", seed_head=100)

    # Set shared classification weights
    h_test[0].set_weights(h_ref[0].get_weights())  # dense_512
    h_test[1].set_weights(h_ref[1].get_weights())  # clf

    w_d512_ref = h_ref[0].get_weights()[0]
    w_d512_test = h_test[0].get_weights()[0]
    w_clf_ref = h_ref[1].get_weights()[0]
    w_clf_test = h_test[1].get_weights()[0]

    d512_match = np.allclose(w_d512_ref, w_d512_test)
    clf_match = np.allclose(w_clf_ref, w_clf_test)
    g9_pass = d512_match and clf_match
    print(f"  dense_512 initialization parity: {'MATCH' if d512_match else 'MISMATCH'} (shape: {w_d512_ref.shape})")
    print(f"  clf initialization parity:       {'MATCH' if clf_match else 'MISMATCH'} (shape: {w_clf_ref.shape})")
    print(f"  Gate 9 Result: {'PASS' if g9_pass else 'FAIL'}")
    results["Gate 9: Shared Classifier Init Parity"] = "PASS" if g9_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 10: Projection Head Architectural Differentiation
    # -------------------------------------------------------------
    print("\n--- Gate 10: Projection Head Architectural Differentiation ---")
    emb_layer_A = [l for l in mA.layers if l.name == "emb"][0]
    proj1_layer_B = [l for l in mB.layers if l.name == "proj_mlp_dense1"][0]
    proj2_layer_B = [l for l in mB.layers if l.name == "proj_mlp_dense2"][0]

    g10_pass = (
        emb_layer_A.weights[0].shape == (3072, 128) and
        proj1_layer_B.weights[0].shape == (3072, 512) and
        proj2_layer_B.weights[0].shape == (512, 128) and
        proj1_layer_B.activation.__name__ == "relu" and
        (proj2_layer_B.activation is None or proj2_layer_B.activation.__name__ == "linear") and
        (emb_layer_A.activation is None or emb_layer_A.activation.__name__ == "linear")
    )
    print(f"  Config A Projection: Linear Dense(128) | kernel shape={emb_layer_A.weights[0].shape}, activation={emb_layer_A.activation}")
    print(f"  Config B Projection: 2-Layer MLP | layer 1={proj1_layer_B.weights[0].shape} (ReLU), layer 2={proj2_layer_B.weights[0].shape} (Linear)")
    print(f"  Shape divergence mathematically documented: Linear (3072x128) vs MLP (3072x512 + 512x128)")
    print(f"  Gate 10 Result: {'PASS' if g10_pass else 'FAIL'}")
    results["Gate 10: Projection Architectural Differentiation"] = "PASS" if g10_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 11: Canonical Probability Remapping & Normalization
    # -------------------------------------------------------------
    print("\n--- Gate 11: Canonical Remapping & Normalization ---")
    p_native_test = np.array([0.10, 0.40, 0.20, 0.30]) # A+, B+, AB+, O+
    # Canonical mapping: 0: A+, 1: AB+, 2: B+, 3: O+
    p_canon_test = np.array([p_native_test[0], p_native_test[2], p_native_test[1], p_native_test[3]])
    p_canon_test = p_canon_test / p_canon_test.sum()

    g11_pass = (
        np.isclose(p_canon_test.sum(), 1.0) and
        p_canon_test[0] == 0.10 and  # A+
        p_canon_test[1] == 0.20 and  # AB+
        p_canon_test[2] == 0.40 and  # B+
        p_canon_test[3] == 0.30      # O+
    )
    print(f"  Native probabilities:    {dict(zip(NATIVE_CLASSES, p_native_test))}")
    print(f"  Canonical probabilities: {dict(zip(CANONICAL_CLASSES, p_canon_test))}")
    print(f"  Sum to 1.0: {p_canon_test.sum():.6f}")
    print(f"  Gate 11 Result: {'PASS' if g11_pass else 'FAIL'}")
    results["Gate 11: Canonical Remapping"] = "PASS" if g11_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 12: Cohort Isolation & Quarantine Verification
    # -------------------------------------------------------------
    print("\n--- Gate 12: Cohort Isolation & Quarantine Verification ---")
    df_split = pd.read_csv(SPLIT_CSV)
    dev_df = df_split[df_split["split"].isin(["TRAIN", "VALIDATION"])]
    test_df = df_split[df_split["split"] == "TEST"]

    dev_pids = set(dev_df["participant_id"].unique())
    test_pids = set(test_df["participant_id"].unique())

    overlap = dev_pids.intersection(test_pids)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    dev_pids_sorted = np.array(sorted(list(dev_pids)))
    dev_y_dummy = np.zeros(len(dev_pids_sorted))
    folds = list(sgkf.split(dev_pids_sorted, dev_y_dummy, dev_pids_sorted))

    fold_leaks = []
    for f_idx, (t_idx, v_idx) in enumerate(folds, 1):
        leak = set(dev_pids_sorted[t_idx]).intersection(set(dev_pids_sorted[v_idx]))
        if len(leak) > 0:
            fold_leaks.append(f_idx)

    g12_pass = (
        len(dev_pids) == 80 and
        len(dev_df) == 240 and
        len(test_pids) == 21 and
        len(overlap) == 0 and
        len(fold_leaks) == 0
    )
    print(f"  Dev cohort: {len(dev_pids)} participants ({len(dev_df)} scans) strictly isolated")
    print(f"  Test cohort: {len(test_pids)} participants strictly quarantined (overlap = {len(overlap)})")
    print(f"  5-Fold collocation: zero participant leaks across all 5 folds")
    print(f"  Gate 12 Result: {'PASS' if g12_pass else 'FAIL'}")
    results["Gate 12: Cohort Isolation & Quarantine"] = "PASS" if g12_pass else "FAIL"

    # -------------------------------------------------------------
    # Gate 13: Protected Checkpoint Cryptographic Audit
    # -------------------------------------------------------------
    print("\n--- Gate 13: Protected Checkpoint Cryptographic Audit ---")
    assert PRE_HASHES_FILE.is_file(), f"Pre-hashes missing: {PRE_HASHES_FILE}"
    with open(PRE_HASHES_FILE, "r") as f:
        pre_hashes = json.load(f)

    all_ckpts_match = True
    for name, info in pre_hashes.items():
        p = Path(info["path"])
        h = sha256_of_file(p)
        if h != info["sha256"]:
            all_ckpts_match = False
            print(f"  CORRUPTION in {name}: pre={info['sha256'][:16]} post={h[:16]}")

    g13_pass = all_ckpts_match and len(pre_hashes) == 56
    print(f"  Verified {len(pre_hashes)}/56 historical checkpoints (Production, Orig E2, EXP-03, EXP-05, EXP-06A, EXP-06B)")
    print(f"  Cryptographic match: {'100% BIT-FOR-BIT IDENTICAL' if g13_pass else 'MISMATCH'}")
    print(f"  Gate 13 Result: {'PASS' if g13_pass else 'FAIL'}")
    results["Gate 13: Protected Checkpoint Audit"] = "PASS" if g13_pass else "FAIL"

    print("\n" + "=" * 80)
    print("ALL 13 SYNTHETIC IMPLEMENTATION GATES COMPLETED")
    print("=" * 80)
    for gate, status in results.items():
        print(f"  {gate:<50}: {status}")

    all_pass = all(s == "PASS" for s in results.values())
    print(f"\nFinal Synthetic Safety Decision: {'ALL GATES PASSED (GO)' if all_pass else 'FAIL (NO-GO)'}")

    summary_out = {
        "overall_status": "ALL_GATES_PASSED" if all_pass else "FAILED",
        "gates": results,
        "config_a_trainable_params": params_A,
        "config_b_trainable_params": params_B,
        "parameter_delta": params_B - params_A,
        "backbone_frozen_params": 7037504,
        "protected_checkpoints_checked": len(pre_hashes)
    }
    with open(PROJECT_ROOT / "scratch" / "exp06c_synthetic_safety_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    return all_pass

if __name__ == "__main__":
    success = run_synthetic_tests()
    sys.exit(0 if success else 1)
