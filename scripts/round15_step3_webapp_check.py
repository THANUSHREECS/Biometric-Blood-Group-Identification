"""
scripts/round15_step3_webapp_check.py
Round 15 Item 3: 30-scan web app verification.
- Picks 30 scans at random (seed 42) across classes from all 303 scans.
- Calls /api/upload for each, compares with local script float32 output.
- States rounding in JSON and reports max diff against rounded reference.
- Tests /api/capture behavior.
"""
import os, sys, json, random, requests, numpy as np, pandas as pd, tensorflow as tf

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import transform_c5

APP_URL = "http://127.0.0.1:5000"
classes = ["A+", "B+", "AB+", "O+"]

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

# ── Load all scans and pick 30 with seed 42
split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
random.seed(42)
np.random.seed(42)

# Sample 30 individual scans (not participants), stratified across classes
all_scans = df[["participant_id", "scan_path", "blood_group", "split"]].copy()
sampled = (all_scans.groupby("blood_group", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), 8), random_state=42))
           .sample(30, random_state=42)
           .reset_index(drop=True))

print("30 selected scans (seed 42):")
print(f"{'#':3s} {'PID':8s} {'BG':4s} {'Split':6s} {'Path'}")
for i, row in sampled.iterrows():
    print(f"{i+1:3d} {row['participant_id']:8s} {row['blood_group']:4s} {row['split']:6s} {row['scan_path']}")

print("\n[Building local model...]")
model = build_E2()

def script_predict_single_scan(bmp_path):
    """Single-scan prediction: replicate x3 (matches /api/upload behaviour)."""
    arr, _, _ = transform_c5(bmp_path)
    X = np.expand_dims(np.stack([arr, arr, arr]), axis=0)  # (1, 3, 112, 112, 3)
    all_probs = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(X, verbose=0)[0]
        all_probs.append(preds[0])  # shape (4,)
    return np.mean(all_probs, axis=0)  # float32, no rounding

print("\n[Calling /api/upload for 30 scans and comparing with local float32 output...]")
print("Note: /api/upload receives ONE image and replicates it x3 for the 3-scan model input.")
print("Note: JSON probabilities are rounded to 4 decimal places (round(..., 4)).")
print("="*100)
print(f"{'#':3s} {'PID':8s} {'BG':4s} {'Web top1':8s} {'Script top1':11s} {'Match':5s} {'MaxDiff vs rounded ref':22s}")
print("-"*100)

all_top1_match = True
max_diffs = []

for idx, row in sampled.iterrows():
    bmp_path = row["scan_path"]
    fname = os.path.basename(bmp_path)
    pid = row["participant_id"]
    bg = row["blood_group"]

    # Script output: float32
    script_probs = script_predict_single_scan(bmp_path)  # shape (4,) float32
    script_top1 = classes[int(np.argmax(script_probs))]

    # Rounded reference (to 4 decimals, same as JSON)
    rounded_ref = np.round(script_probs.astype(np.float64), 4)

    # Web app
    try:
        with open(bmp_path, "rb") as f:
            resp = requests.post(f"{APP_URL}/api/upload",
                                 files={"file": (fname, f, "image/bmp")},
                                 timeout=30)
        resp.raise_for_status()
        web_data = resp.json()
        web_top1 = web_data.get("blood_group", "ERR")
        web_probs_dict = web_data.get("probabilities", {})
        web_probs = np.array([web_probs_dict.get(c, 0.0) for c in classes], dtype=np.float64)
    except Exception as e:
        web_top1 = f"ERR:{e}"
        web_probs = np.zeros(4)

    match = web_top1 == script_top1
    max_diff = float(np.max(np.abs(web_probs - rounded_ref)))
    max_diffs.append(max_diff)
    all_top1_match = all_top1_match and match

    print(f"{idx+1:3d} {pid:8s} {bg:4s} {web_top1:8s} {script_top1:11s} {'YES' if match else 'NO':5s} {max_diff:.8f}")
    if max_diff > 1e-4:
        print(f"    Web : {dict(zip(classes, [f'{x:.4f}' for x in web_probs]))}")
        print(f"    Ref : {dict(zip(classes, [f'{x:.4f}' for x in rounded_ref]))}")

print("="*100)
print(f"All top-1 match: {all_top1_match}")
print(f"Max diff vs 4-decimal rounded reference: {max(max_diffs):.8f}")
print(f"Mean diff: {np.mean(max_diffs):.8f}")
print()
print("JSON rounding statement: The /api/upload endpoint returns probabilities rounded to 4 decimal places")
print("  (via Python's round(..., 4) or equivalent JSON serialisation of 4-dp floats).")
print("  Max observed diff between web JSON and 4-dp rounded script output: see above.")
print()
print("="*60)
print("NOTE: /api/upload replicates one scan three times as input to the 3-scan model.")
print("  The model architecture expects shape (1, 3, 112, 112, 3).")
print("  A single upload image is stacked [img, img, img] — NOT three different scans.")
print()
print("NOTE: /api/capture requires a physical MFS100 scanner. In mock mode it uses")
print("  src.mfs100_bridge.bridge which generates synthetic BMP data and does NOT")
print("  pass three separate real scans to the model. Cannot be verified without hardware.")
