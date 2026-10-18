#!/usr/bin/env python3
"""
EXP-07-RES-FIDELITY: Synthetic Data Pipeline & Architecture Test
==============================================================
Validates the complete execution graph, parameter counts, loss computations,
and gradient updates on synthetic dummy data WITHOUT touching real participant images.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp07_resolution_fidelity.config import (
    MODEL_SPEC, CONDITIONS, SEED, BATCH_SIZE, HEAD_LR, WEIGHT_DECAY,
    FIXED_TEMPERATURE, FIXED_LAMBDA, TRAINING_AUTHORIZED
)
from experiments.exp07_resolution_fidelity.train import (
    build_e2_model, supervised_contrastive_loss
)

def run_synthetic_pipeline_test():
    print("=" * 80)
    print("EXP-07: SYNTHETIC DATA PIPELINE & ARCHITECTURE VALIDATION")
    print("=" * 80)

    # 1. Verify Fail-Closed Guard State
    print(f"[Test 1] Fail-Closed Security Guard:")
    assert TRAINING_AUTHORIZED is False, "Security Error: TRAINING_AUTHORIZED must be False during preparation!"
    print("  [PASS] TRAINING_AUTHORIZED is False. Real participant data access is locked.")

    # 2. Test Model Graph & Parameter Consistency across 112x112 and 224x224
    print(f"\n[Test 2] Model Graph & Parameter Invariance Verification:")
    models = {}
    for arm_key, spec in CONDITIONS.items():
        res = spec["input_resolution"]
        print(f"\n  Building Model for {spec['name']} (Resolution: {res})...")
        t0 = time.time()
        model, base, heads = build_e2_model(target_resolution=res, seed_head=SEED)
        build_time = time.time() - t0

        # Assert frozen backbone
        frozen_params = base.count_params()
        assert base.trainable is False, f"[{arm_key}] Backbone must be strictly frozen!"
        assert frozen_params == MODEL_SPEC["expected_frozen_params"], (
            f"[{arm_key}] Frozen params mismatch: expected {MODEL_SPEC['expected_frozen_params']}, got {frozen_params}"
        )

        # Assert trainable head parameters
        trainable_vars = model.trainable_variables
        trainable_params = sum(np.prod(v.shape) for v in trainable_vars)
        assert len(trainable_vars) == MODEL_SPEC["expected_trainable_vars"], (
            f"[{arm_key}] Expected {MODEL_SPEC['expected_trainable_vars']} trainable vars, got {len(trainable_vars)}"
        )
        assert trainable_params == MODEL_SPEC["expected_trainable_params"], (
            f"[{arm_key}] Expected {MODEL_SPEC['expected_trainable_params']} trainable params, got {trainable_params}"
        )

        print(f"    Build time: {build_time:.2f}s")
        print(f"    Frozen backbone params: {frozen_params:,} (100% frozen)")
        print(f"    Trainable head variables: {len(trainable_vars)} tensors")
        print(f"    Trainable head params:    {trainable_params:,}")
        print(f"    [PASS] Model architecture and parameter counts match specifications exactly.")
        models[arm_key] = model

    # Assert parameter parity between both arms (Zero parameter confound)
    arm_a_params = sum(np.prod(v.shape) for v in models["arm_a_112"].trainable_variables)
    arm_b_params = sum(np.prod(v.shape) for v in models["arm_b_224"].trainable_variables)
    assert arm_a_params == arm_b_params, "CONFOUND DETECTED: Trainable parameter count differs between resolutions!"
    print(f"\n  [PASS] Parameter Parity Verified: 112x112 ({arm_a_params:,}) == 224x224 ({arm_b_params:,}). Zero parameter confound.")

    # 3. Test Synthetic Forward Pass & Backward Gradients
    print(f"\n[Test 3] Synthetic Forward & Backward Step Verification:")
    n_dummy = 8
    dummy_labels = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)

    for arm_key, spec in CONDITIONS.items():
        res = spec["input_resolution"]
        h, w = res
        model = models[arm_key]

        # Generate synthetic 3-scan participant data in range [0, 255]
        np.random.seed(SEED)
        dummy_X = np.random.uniform(0.0, 255.0, size=(n_dummy, 3, h, w, 3)).astype(np.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=HEAD_LR, weight_decay=WEIGHT_DECAY)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Forward inference check
        preds_clf, preds_emb = model.predict(dummy_X, verbose=0)
        assert preds_clf.shape == (n_dummy, 4), f"Unexpected clf preds shape: {preds_clf.shape}"
        assert preds_emb.shape == (n_dummy, 128), f"Unexpected emb preds shape: {preds_emb.shape}"
        np.testing.assert_allclose(np.sum(preds_clf, axis=1), 1.0, rtol=1e-5)
        print(f"  {arm_key}: Forward inference verified: clf {preds_clf.shape}, emb {preds_emb.shape}")

        # Training step check (2 epochs on synthetic data)
        history = model.fit(dummy_X, dummy_labels, batch_size=4, epochs=2, verbose=0)
        loss_ep1 = history.history["loss"][0]
        loss_ep2 = history.history["loss"][1]
        assert not np.isnan(loss_ep1) and not np.isnan(loss_ep2), f"NaN loss detected in {arm_key}!"
        print(f"  {arm_key}: 2 Synthetic epochs passed: Loss Ep 1 = {loss_ep1:.4f}, Loss Ep 2 = {loss_ep2:.4f} [PASS]")

    # 4. Verify Quarantine Assertion Guards
    print(f"\n[Test 4] Quarantine Assertion Guards Check:")
    from experiments.exp07_resolution_fidelity.train import train_condition
    try:
        train_condition("arm_a_112")
        raise RuntimeError("FAIL: Guard failed to block unauthorized training!")
    except PermissionError as e:
        print(f"  [PASS] Guard successfully triggered: {e.args[0].splitlines()[1]}")

    print("\n" + "=" * 80)
    print("ALL SYNTHETIC PIPELINE TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    run_synthetic_pipeline_test()
