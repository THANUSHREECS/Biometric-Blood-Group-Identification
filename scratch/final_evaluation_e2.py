import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.utils import resample

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

config_path = os.path.join(project_root, "reports", "phase3_final_evaluation", "frozen_model_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

classes = config["class_order"]
class_mapping = {i: c for i, c in enumerate(classes)}
rev_class_mapping = {c: i for i, c in enumerate(classes)}

split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv_path)
test_df = df[df["split"] == "TEST"].copy()

test_pids = test_df["participant_id"].unique()
assert len(test_pids) == 21, f"Expected 21 test participants, found {len(test_pids)}"
assert len(test_df) == 63, f"Expected 63 scans, found {len(test_df)}"

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

print("Loading 21 test participants...")
test_X, test_y, test_pids_list = [], [], []
for pid, group in test_df.groupby("participant_id"):
    assert len(group) == 3, f"PID {pid} has {len(group)} scans!"
    scans = [transform_c5(row["scan_path"]) for _, row in group.iterrows()]
    test_X.append(scans)
    test_y.append(rev_class_mapping[group.iloc[0]["blood_group"]])
    test_pids_list.append(pid)

test_X = np.array(test_X)
test_y = np.array(test_y)
test_pids_list = np.array(test_pids_list)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_E2():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    
    aug1, aug2, aug3 = data_augmentation(scan1, training=False), data_augmentation(scan2, training=False), data_augmentation(scan3, training=False)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()
    feat1, feat2, feat3 = pool(base(aug1)), pool(base(aug2)), pool(base(aug3))
    concat = tf.keras.layers.Concatenate()([feat1, feat2, feat3])
    
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
    emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
    return tf.keras.Model(inputs, [clf, emb])

model = build_E2()
checkpoint_paths = config["checkpoint_paths"]
assert len(checkpoint_paths) == 5

all_probs = []
for cp in checkpoint_paths:
    cp_abs = os.path.join(project_root, cp.replace("/", os.sep))
    model.load_weights(cp_abs)
    # prediction outputs [clf, emb], we take clf
    preds = model.predict(test_X, verbose=0)[0]
    all_probs.append(preds)

all_probs = np.array(all_probs) # (5, 21, 4)
ensemble_probs = np.mean(all_probs, axis=0) # (21, 4)
y_pred = np.argmax(ensemble_probs, axis=1)

y_true_p = [class_mapping[y] for y in test_y]
y_pred_p = [class_mapping[y] for y in y_pred]

acc = accuracy_score(y_true_p, y_pred_p)
macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
weighted_f1 = f1_score(y_true_p, y_pred_p, average="weighted")
bal_acc = balanced_accuracy_score(y_true_p, y_pred_p)
rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
cm = confusion_matrix(y_true_p, y_pred_p, labels=classes)

# Bootstrap 10k
n_bootstraps = 10000
boot_acc = []
boot_macro = []
np.random.seed(42)

for i in range(n_bootstraps):
    indices = resample(np.arange(len(y_true_p)), replace=True)
    y_t_boot = np.array(y_true_p)[indices]
    y_p_boot = np.array(y_pred_p)[indices]
    
    boot_acc.append(accuracy_score(y_t_boot, y_p_boot))
    boot_macro.append(f1_score(y_t_boot, y_p_boot, average="macro"))

ci_acc = np.percentile(boot_acc, [2.5, 97.5])
ci_macro = np.percentile(boot_macro, [2.5, 97.5])

out_dir = os.path.join(project_root, "reports", "phase3_final_evaluation")

# 2. final_test_predictions.csv
pred_data = []
for i in range(21):
    pred_data.append({
        "participant_id": test_pids_list[i],
        "actual_class": y_true_p[i],
        "predicted_class": y_pred_p[i],
        "probability_A_plus": ensemble_probs[i][0],
        "probability_B_plus": ensemble_probs[i][1],
        "probability_AB_plus": ensemble_probs[i][2],
        "probability_O_plus": ensemble_probs[i][3],
        "confidence": np.max(ensemble_probs[i]),
        "correct": y_true_p[i] == y_pred_p[i]
    })
pd.DataFrame(pred_data).to_csv(os.path.join(out_dir, "final_test_predictions.csv"), index=False)

# 3. confusion matrix
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv(os.path.join(out_dir, "final_test_confusion_matrix.csv"))

# 4. classification report
rep_df = pd.DataFrame(rep).transpose()
rep_df.to_csv(os.path.join(out_dir, "final_test_classification_report.csv"))

# 5. bootstrap intervals
with open(os.path.join(out_dir, "final_test_bootstrap_intervals.json"), "w") as f:
    json.dump({
        "accuracy_mean": np.mean(boot_acc),
        "accuracy_ci_lower": ci_acc[0],
        "accuracy_ci_upper": ci_acc[1],
        "macro_f1_mean": np.mean(boot_macro),
        "macro_f1_ci_lower": ci_macro[0],
        "macro_f1_ci_upper": ci_macro[1],
        "n_bootstraps": n_bootstraps,
        "seed": 42
    }, f, indent=4)

# 6. prediction distribution
pd.Series(y_pred_p).value_counts().to_csv(os.path.join(out_dir, "final_test_prediction_distribution.csv"))

print(f"DONE_MARKER|{acc}|{macro_f1}|{weighted_f1}|{bal_acc}")
