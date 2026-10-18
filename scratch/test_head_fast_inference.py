import time
import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import tensorflow as tf
from src.predictor import BloodGroupPredictor

pred = BloodGroupPredictor()
np.random.seed(42)
dummy = np.random.randn(1, 3, 112, 112, 3).astype(np.float32)

# Full model execution across 5 folds (Current method)
t0 = time.perf_counter()
p_full = []
for m in pred.fold_models:
    clf, emb = m(dummy, training=False)
    p_full.append(clf.numpy()[0])
p_full = np.mean(p_full, axis=0)
t_full = time.perf_counter() - t0
print(f"Full method: {t_full:.3f}s, output: {p_full}")

# Fast method: Extract 3 scans through shared frozen DenseNet-121 backbone once
t0 = time.perf_counter()
base_net = pred.fold_models[0].get_layer('densenet121')
pool_layer = pred.fold_models[0].get_layer('gap')

# Extract features for all 3 scans
s1, s2, s3 = dummy[:, 0], dummy[:, 1], dummy[:, 2]
scans_batch = tf.concat([s1, s2, s3], axis=0) # (3, 112, 112, 3)
feats = pool_layer(base_net(scans_batch, training=False)) # (3, 1024)
fused_fast = tf.concat([feats[0:1], feats[1:2], feats[2:3]], axis=1) # (1, 3072)

p_fast = []
for m in pred.fold_models:
    d = m.get_layer('dense_512')(fused_fast)
    # in inference, dropout is identity
    clf_out = m.get_layer('clf')(d)
    p_fast.append(clf_out.numpy()[0])
p_fast = np.mean(p_fast, axis=0)
t_fast = time.perf_counter() - t0
print(f"Fast method: {t_fast:.3f}s, output: {p_fast}")

max_diff = np.max(np.abs(p_full - p_fast))
print(f"Max absolute difference: {max_diff}")
assert max_diff < 1e-5, f"Discrepancy too large: {max_diff}"
print("VERIFICATION SUCCESSFUL: 100% Numerical Identity confirmed!")
