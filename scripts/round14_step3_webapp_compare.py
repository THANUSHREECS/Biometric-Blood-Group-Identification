"""
scripts/round14_step3_webapp_compare.py
Round 14 Step 3: Run 10 stored BMP files through:
  (a) the running web app /api/upload endpoint
  (b) the local script using transform_c5 + ensemble

Compare predicted class and probability vectors.
Prints raw output.
"""
import os, sys, json, time, requests, numpy as np, pandas as pd, tensorflow as tf

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)

from src.preprocessing import transform_c5

APP_URL = "http://127.0.0.1:5000"

classes = ["A+", "B+", "AB+", "O+"]
class_map = {i: c for i, c in enumerate(classes)}

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
    scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    a1 = data_aug(scan1, training=False)
    a2 = data_aug(scan2, training=False)
    a3 = data_aug(scan3, training=False)
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

# Pick 10 BMPs from the raw dataset (first 10 from sorted scan_paths)
split_csv = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv)
# Take one BMP per participant (scan_01.bmp), first 10 participants
sample_rows = df.groupby("participant_id").first().head(10).reset_index()
sample_paths = sample_rows["scan_path"].tolist()
print(f"Selected {len(sample_paths)} BMP files for comparison")

print("\n[1/2] Building local model...")
model = build_E2()

def script_predict(bmp_path):
    """Local script prediction: replicate single scan 3x (same as predictor.predict)."""
    arr, _, _ = transform_c5(bmp_path)
    X = np.expand_dims(np.stack([arr, arr, arr]), axis=0)  # (1, 3, 112, 112, 3)
    all_probs = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(X, verbose=0)[0]
        all_probs.append(preds[0])
    ensemble = np.mean(all_probs, axis=0)
    return class_map[int(np.argmax(ensemble))], {classes[i]: round(float(ensemble[i]), 4) for i in range(4)}

print("\n[2/2] Running web app vs script comparison...")
print("="*80)
print(f"{'File':30s} {'WebApp':6s} {'Script':6s} {'Match':5s}  {'Max prob diff':14s}")
print("-"*80)

all_match = True
for bmp_path in sample_paths:
    fname = os.path.basename(bmp_path)
    
    # Web app via /api/upload
    try:
        with open(bmp_path, "rb") as f:
            resp = requests.post(f"{APP_URL}/api/upload", files={"file": (fname, f, "image/bmp")}, timeout=30)
        resp.raise_for_status()
        web_data = resp.json()
        web_class = web_data.get("blood_group", "ERR")
        web_probs = web_data.get("probabilities", {})
        # Convert to array in class order
        web_arr = np.array([web_probs.get(c, 0.0) for c in classes])
    except Exception as e:
        web_class = f"ERR:{e}"
        web_arr = np.zeros(4)
    
    # Script prediction
    script_class, script_probs_dict = script_predict(bmp_path)
    script_arr = np.array([script_probs_dict.get(c, 0.0) for c in classes])
    
    match = web_class == script_class
    max_diff = float(np.max(np.abs(web_arr - script_arr))) if not web_class.startswith("ERR:") else 999.0
    all_match = all_match and match and (max_diff < 1e-2)
    
    print(f"{fname:30s} {web_class:6s} {script_class:6s} {'YES' if match else 'NO ':5s}  {max_diff:.6f}")
    print(f"  Web  probs: {dict(zip(classes, [round(x,4) for x in web_arr]))}")
    print(f"  Script probs: {script_probs_dict}")

print("="*80)
print(f"All top-1 classes match: {all_match}")
