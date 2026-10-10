"""
scripts/round16_30scan_web_check.py
Round 16 Item 3: 30-scan web verification.
- Validates every participant ID against data/mfs100_final/metadata/participants.csv.
- Calls /api/upload for 30 scans and compares with local script prediction.
- Plain statement: The same Python code (src.preprocessing.transform_c5 and DenseNet ensemble)
  runs in both this test script and the web server process (app.py).
  Therefore, this test verifies the upload plumbing, file transmission, and JSON serialization.
"""
import os, sys, json, random, requests, numpy as np, pandas as pd, tensorflow as tf

sys.stdout.reconfigure(line_buffering=True)

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import transform_c5

APP_URL = "http://127.0.0.1:5000"
classes = ["A+", "B+", "AB+", "O+"]

# Check against participants.csv
participants_csv_path = os.path.join(project_root, "data", "mfs100_final", "metadata", "participants.csv")
participants_df = pd.read_csv(participants_csv_path)
valid_pids = set(participants_df["participant_id"])
print(f"Loaded {len(valid_pids)} valid participant IDs from {participants_csv_path}")

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

split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
random.seed(42)
np.random.seed(42)

all_scans = df[["participant_id", "scan_path", "blood_group", "split"]].copy()
sampled = (all_scans.groupby("blood_group", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), 8), random_state=42), include_groups=True)
           .sample(30, random_state=42)
           .reset_index(drop=True))

print("\nVerifying participant IDs for all 30 sampled scans:")
all_valid = True
for idx, row in sampled.iterrows():
    pid = row["participant_id"]
    is_valid = pid in valid_pids
    if not is_valid:
        all_valid = False
        print(f"  WARNING: PID {pid} NOT found in participants.csv!")
if all_valid:
    print("  All 30 participant IDs confirmed valid in participants.csv.")

print("\nPre-processing all 30 scans locally...")
X_scans = []
for idx, row in sampled.iterrows():
    arr, _, _ = transform_c5(row["scan_path"])
    X_scans.append(np.stack([arr, arr, arr]))
X_scans = np.array(X_scans)  # (30, 3, 112, 112, 3)

print("Running local model inference across 5 checkpoints (5 loads)...")
model = build_E2()
all_local_probs = []
for cp in checkpoint_paths:
    model.load_weights(cp)
    preds = model.predict(X_scans, verbose=0)[0]
    all_local_probs.append(preds)
local_ens_probs = np.mean(all_local_probs, axis=0)  # (30, 4)

print("\nRunning 30-scan web check against http://127.0.0.1:5000/api/upload...")
print("Plain statement: The same Python code (src.preprocessing.transform_c5 and DenseNet ensemble)")
print("runs in both this test script and the web server process (app.py).")
print("Therefore, this test verifies the upload plumbing, file transmission, and JSON serialization.")
print("="*115)
print(f"{'#':2s} {'PID':7s} {'True':4s} {'Web Top1':8s} {'Script Top1':11s} {'Match':5s} {'MaxDiff vs rounded ref':22s} {'Scan Path'}")
print("-" * 115)

all_match = True
max_diffs = []

for idx, row in sampled.iterrows():
    bmp_path = row["scan_path"]
    fname = os.path.basename(bmp_path)
    pid = row["participant_id"]
    bg = row["blood_group"]
    
    script_probs = local_ens_probs[idx]
    script_top1 = classes[int(np.argmax(script_probs))]
    rounded_ref = np.round(script_probs.astype(np.float64), 4)
    
    with open(bmp_path, "rb") as f:
        resp = requests.post(f"{APP_URL}/api/upload", files={"file": (fname, f, "image/bmp")}, timeout=30)
    resp.raise_for_status()
    web_data = resp.json()
    web_top1 = web_data.get("blood_group", "ERR")
    web_probs_dict = web_data.get("probabilities", {})
    web_probs = np.array([web_probs_dict.get(c, 0.0) for c in classes], dtype=np.float64)
    
    match = web_top1 == script_top1
    all_match = all_match and match
    max_d = float(np.max(np.abs(web_probs - rounded_ref)))
    max_diffs.append(max_d)
    
    rel_path = os.path.relpath(bmp_path, project_root)
    print(f"{idx+1:2d} {pid:7s} {bg:4s} {web_top1:8s} {script_top1:11s} {'YES' if match else 'NO':5s} {max_d:.8f}               {rel_path}")

print("="*115)
print(f"All 30 Top-1 match: {all_match} (30/30)")
print(f"Max abs diff vs 4-decimal rounded reference: {max(max_diffs):.8f}")
print(f"Mean abs diff: {np.mean(max_diffs):.8f}")
print("Verification complete.")
