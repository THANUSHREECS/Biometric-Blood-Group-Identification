import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "reports" / "efficientnet_e2_ensemble_experiment"
EXP_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import transform_c5
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# Checkpoint paths
EFFNET_CKPT = PROJECT_ROOT / "models" / "abo_positive_4class" / "best_model.keras"
E2_CKPT_DIR = PROJECT_ROOT / "models" / "phase3_fusion_metric_learning" / "E2"
E2_CHECKPOINTS = [E2_CKPT_DIR / f"fold_{i}_best.weights.h5" for i in range(1, 6)]

SPLIT_CSV = PROJECT_ROOT / "reports" / "phase3_participant_benchmark" / "participant_split.csv"
GT_CSV = PROJECT_ROOT / "data" / "mfs100_ground_truth.csv"
SAVED_TEST_CSV = PROJECT_ROOT / "reports" / "phase3_final_evaluation" / "final_test_predictions.csv"

print("--- PHASE C: Creating experiment_config.json ---")
with open(EXP_DIR / "checkpoint_hashes_before_after.json") as f:
    hashes_data = json.load(f)

config = {
    "experiment_name": "fair_comparison_effnet_b0_vs_e2_vs_ensemble",
    "timestamp": "2026-10-13T22:52:00",
    "framework_versions": {
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__
    },
    "models": {
        "efficientnet_b0": {
            "checkpoint_path": str(EFFNET_CKPT),
            "sha256": hashes_data["before_experiment"]["effnet_checkpoint"]["sha256"],
            "input_shape": [224, 224, 3],
            "native_class_order": ["A+", "AB+", "B+", "O+"],
            "preprocessing": "resize(224, 224, Bilinear) -> RGB float32 -> tf.keras.applications.efficientnet.preprocess_input",
            "scan_aggregation": "mean of 3 scan probability vectors"
        },
        "e2_fusion": {
            "checkpoints": [str(cp) for cp in E2_CHECKPOINTS],
            "sha256_hashes": [
                hashes_data["before_experiment"][f"e2_fold{i}_weights"]["sha256"] for i in range(1, 6)
            ],
            "input_shape": [3, 112, 112, 3],
            "native_class_order": ["A+", "B+", "AB+", "O+"],
            "preprocessing": "transform_c5 (composite_alpha -> crop_foreground -> CLAHE_bilateral -> resize(112, 112))",
            "architecture": "3-tower frozen DenseNet-121 -> GAP -> Concat(3072) -> Dense(512, relu) -> Dropout(0.5) -> Dense(4, softmax)",
            "ensemble_aggregation": "mean across 5 folds",
            "canonical_remapping": "[native[0], native[2], native[1], native[3]]"
        }
    },
    "canonical_class_order": ["A+", "AB+", "B+", "O+"],
    "ensemble_formula": "P_combined = 0.5 * P_effnet + 0.5 * P_e2_canonical",
    "cohorts": {
        "development_80": {
            "description": "80 MFS100 development participants (64 TRAIN, 16 VALIDATION)",
            "manifest": str(SPLIT_CSV)
        },
        "locked_test_21": {
            "description": "21 quarantined MFS100 test participants",
            "manifest": str(SPLIT_CSV)
        },
        "separate_cohort_26": {
            "description": "26 MFS100 participants with hospital serological confirmation (S001-S026)",
            "manifest": str(GT_CSV)
        }
    }
}

with open(EXP_DIR / "experiment_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("experiment_config.json successfully created.")

print("\n--- PHASE D: Verifying Inference & Probabilities ---")
print("1. Loading EfficientNet-B0...")
effnet_model = tf.keras.models.load_model(str(EFFNET_CKPT))
print("   EfficientNet-B0 loaded successfully. Expected input:", effnet_model.input_shape)

print("2. Building E2 Fusion model architecture...")
def build_E2():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
        tf.keras.layers.RandomContrast(0.1)
    ])
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    s1, s2, s3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1 = data_augmentation(s1, training=False)
    a2 = data_augmentation(s2, training=False)
    a3 = data_augmentation(s3, training=False)
    base = tf.keras.applications.DenseNet121(include_top=False, weights=None)
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    f1, f2, f3 = pool(base(a1)), pool(base(a2)), pool(base(a3))
    concat = tf.keras.layers.Concatenate()([f1, f2, f3])
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

e2_model = build_E2()
print("   E2 model built. Expected input:", e2_model.input_shape)

# Load first fold weights to test
e2_model.load_weights(str(E2_CHECKPOINTS[0]))
print("   Fold 1 weights loaded successfully.")

# Test on participant P002 (from locked test) to verify alignment against saved_csv
df_split = pd.read_csv(SPLIT_CSV)
p002_rows = df_split[df_split["participant_id"] == "P002"]
p002_scans = [Path(r["scan_path"]) for _, r in p002_rows.iterrows()]
print(f"\n3. Verifying participant P002 (3 scans): {[s.name for s in p002_scans]}")

# Preprocess E2
p002_c5 = [transform_c5(str(p))[0] for p in p002_scans]
p002_tensor = np.stack(p002_c5, axis=0)[np.newaxis, ...]
print("   E2 input tensor shape:", p002_tensor.shape)

# Run 5 fold ensemble
e2_fold_preds = []
for cp in E2_CHECKPOINTS:
    e2_model.load_weights(str(cp))
    clf, _ = e2_model.predict(p002_tensor, verbose=0)
    e2_fold_preds.append(clf[0])

e2_native_prob = np.mean(e2_fold_preds, axis=0)
print("   E2 native output probs [A+, B+, AB+, O+]:", e2_native_prob)

# Compare with saved test CSV for P002
df_saved = pd.read_csv(SAVED_TEST_CSV)
row_p002 = df_saved[df_saved["participant_id"] == "P002"].iloc[0]
saved_p002_native = np.array([
    row_p002["probability_A_plus"],
    row_p002["probability_B_plus"],
    row_p002["probability_AB_plus"],
    row_p002["probability_O_plus"]
])
print("   Saved benchmark probs [A+, B+, AB+, O+]:", saved_p002_native)
max_diff = np.max(np.abs(e2_native_prob - saved_p002_native))
print(f"   Max absolute diff between recomputed E2 and saved benchmark: {max_diff:.8f}")

# Canonical remapping
e2_canonical_prob = np.array([
    e2_native_prob[0], # A+
    e2_native_prob[2], # AB+
    e2_native_prob[1], # B+
    e2_native_prob[3], # O+
])
print("   E2 canonically remapped [A+, AB+, B+, O+]:", e2_canonical_prob)

# Test EfficientNet-B0 on P002
eff_probs = []
for p in p002_scans:
    img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = effnet_preprocess(np.array(img, dtype=np.float32))[np.newaxis, ...]
    eff_probs.append(effnet_model.predict(arr, verbose=0)[0])

eff_mean_prob = np.mean(eff_probs, axis=0)
print("   EfficientNet-B0 3-scan mean [A+, AB+, B+, O+]:", eff_mean_prob)

# Combined
comb_prob = 0.5 * eff_mean_prob + 0.5 * e2_canonical_prob
print("   Combined equal-weight ensemble [A+, AB+, B+, O+]:", comb_prob)
print(f"   Probabilities sum: EffNet={eff_mean_prob.sum():.6f}, E2={e2_canonical_prob.sum():.6f}, Comb={comb_prob.sum():.6f}")
print("   All finite & non-negative:", np.all(comb_prob >= 0) and np.all(np.isfinite(comb_prob)))

print("\n--- PHASE D VALIDATION COMPLETED SUCCESSFULLY ---")
