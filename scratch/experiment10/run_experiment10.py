import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv_path)
dev_df = df[df["split"].isin(["TRAIN", "VALIDATION"])].copy()

classes = ["A+", "B+", "AB+", "O+"]
rev_class_mapping = {c: i for i, c in enumerate(classes)}

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

print("Loading raw images...")
X_raw, y_raw, pid_raw = [], [], []
for pid, group in dev_df.groupby("participant_id"):
    for _, row in group.iterrows():
        img = transform_c5(row["scan_path"])
        X_raw.append(img)
        y_raw.append(rev_class_mapping[row["blood_group"]])
        pid_raw.append(pid)

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)
pid_raw = np.array(pid_raw)

# Augmentations
mild_aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Models
def build_model_A():
    inputs = tf.keras.Input(shape=(112, 112, 3))
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")(inputs)
    pool = tf.keras.layers.GlobalAveragePooling2D()(base)
    d = tf.keras.layers.Dense(512, activation="relu")(pool)
    drop = tf.keras.layers.Dropout(0.5)(d)
    out = tf.keras.layers.Dense(4, activation="softmax")(drop)
    model = tf.keras.Model(inputs, out)
    model.layers[1].trainable = False
    return model

def build_model_B():
    inputs = tf.keras.Input(shape=(112, 112, 3))
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")(inputs)
    pool = tf.keras.layers.GlobalAveragePooling2D()(base)
    out = tf.keras.layers.Dense(4, activation="softmax")(pool)
    model = tf.keras.Model(inputs, out)
    model.layers[1].trainable = False
    return model

def build_model_C():
    inputs = tf.keras.Input(shape=(112, 112, 3))
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")(inputs)
    pool = tf.keras.layers.GlobalAveragePooling2D()(base)
    d = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(pool)
    drop = tf.keras.layers.Dropout(0.5)(d)
    out = tf.keras.layers.Dense(4, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop)
    model = tf.keras.Model(inputs, out)
    model.layers[1].trainable = False
    return model

def build_model_D():
    inputs = tf.keras.Input(shape=(112, 112, 3))
    x = mild_aug(inputs)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")(x)
    pool = tf.keras.layers.GlobalAveragePooling2D()(base)
    d = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(pool)
    drop = tf.keras.layers.Dropout(0.5)(d)
    out = tf.keras.layers.Dense(4, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop)
    model = tf.keras.Model(inputs, out)
    model.layers[2].trainable = False
    return model

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

unique_pids = []
unique_y = []
for pid in np.unique(pid_raw):
    idx = np.where(pid_raw == pid)[0]
    unique_pids.append(pid)
    unique_y.append(y_raw[idx[0]])

unique_pids = np.array(unique_pids)
unique_y = np.array(unique_y)

fold_generator = list(sgkf.split(np.zeros(len(unique_pids)), unique_y, unique_pids))

models_dict = {
    "A_E2_Reference": build_model_A,
    "B_Smaller_Head": build_model_B,
    "C_Regularized": build_model_C,
    "D_Reg_Augmented": build_model_D
}

results = []
cm_results = []
class_reports = []

class_counts = np.bincount(y_raw)
total = sum(class_counts)
c_w = {i: total / (4 * count) for i, count in enumerate(class_counts)}

for m_name, build_fn in models_dict.items():
    for fold_idx, (train_pid_idx, val_pid_idx) in enumerate(fold_generator):
        
        train_pids_in_fold = unique_pids[train_pid_idx]
        val_pids_in_fold = unique_pids[val_pid_idx]
        
        train_idx = np.where(np.isin(pid_raw, train_pids_in_fold))[0]
        val_idx = np.where(np.isin(pid_raw, val_pids_in_fold))[0]
        
        X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
        X_v, y_v, p_v = X_raw[val_idx], y_raw[val_idx], pid_raw[val_idx]
        
        model = build_fn()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        
        model.fit(X_tr, y_tr, batch_size=32, epochs=20, class_weight=c_w, verbose=0)
        
        preds_prob = model.predict(X_v, verbose=0)
        
        agg_probs, agg_y_true = [], []
        for pid in np.unique(p_v):
            idx = np.where(p_v == pid)[0]
            agg_probs.append(np.mean(preds_prob[idx], axis=0))
            agg_y_true.append(y_v[idx[0]])
            
        agg_probs = np.array(agg_probs)
        agg_y_true = np.array(agg_y_true)
        y_pred = np.argmax(agg_probs, axis=1)
        
        mac_f1 = f1_score(agg_y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(agg_y_true, y_pred)
        bal_acc = balanced_accuracy_score(agg_y_true, y_pred)
        cm = confusion_matrix(agg_y_true, y_pred, labels=[0,1,2,3])
        rep = classification_report(agg_y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        
        results.append({
            "Model": m_name,
            "Fold": fold_idx+1,
            "Macro_F1": mac_f1,
            "Accuracy": acc,
            "Balanced_Acc": bal_acc
        })
        
        for cls in classes:
            class_reports.append({
                "Model": m_name, "Fold": fold_idx+1,
                "Class": cls, "Precision": rep[cls]["precision"], "Recall": rep[cls]["recall"], "F1": rep[cls]["f1-score"]
            })
        
        for r_idx, r_cls in enumerate(classes):
            for c_idx, c_cls in enumerate(classes):
                cm_results.append({
                    "Model": m_name, "Fold": fold_idx+1,
                    "True_Class": r_cls, "Predicted_Class": c_cls, "Count": cm[r_idx, c_idx]
                })
        
        tf.keras.backend.clear_session()

out_dir = os.path.join(project_root, "reports", "phase4_experiment10")
os.makedirs(out_dir, exist_ok=True)

df_res = pd.DataFrame(results)
print(df_res.groupby("Model")[["Macro_F1", "Accuracy"]].mean())

df_res.to_csv(os.path.join(out_dir, "EXPERIMENT10_FOLD_RESULTS.csv"), index=False)
pd.DataFrame(class_reports).to_csv(os.path.join(out_dir, "EXPERIMENT10_CLASSIFICATION_REPORT.csv"), index=False)
pd.DataFrame(cm_results).to_csv(os.path.join(out_dir, "EXPERIMENT10_CONFUSION_MATRICES.csv"), index=False)

print("Finished!")
