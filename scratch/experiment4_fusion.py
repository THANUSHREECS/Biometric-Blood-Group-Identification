import os
import sys
import glob
import json
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

# Paths
data_dir = os.path.join(project_root, "data", "mfs100_final", "raw")
reports_dir = os.path.join(project_root, "reports", "phase3_feature_fusion")
models_dir = os.path.join(project_root, "models", "phase3_feature_fusion")
split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(reports_dir, "confusion_matrices"), exist_ok=True)

IMAGE_SIZE = (112, 112)
BATCH_SIZE = 16
EPOCHS = 150
MIN_EPOCHS = 100
PATIENCE = 15
SEED = 42
WEIGHT_DECAY = 1e-4

tf.random.set_seed(SEED)
np.random.seed(SEED)

classes = ["A+", "B+", "AB+", "O+"]
class_mapping = {i: c for i, c in enumerate(classes)}
rev_class_mapping = {c: i for i, c in enumerate(classes)}

split_df = pd.read_csv(split_csv_path)

# Extract only DEV (Train + Val)
dev_df = split_df[split_df["split"].isin(["TRAIN", "VALIDATION"])].copy()
test_df = split_df[split_df["split"] == "TEST"].copy()

# Audit: exactly 80 dev participants, 21 test participants
dev_pids = dev_df["participant_id"].unique()
test_pids = test_df["participant_id"].unique()
assert len(dev_pids) == 80, f"Expected 80 dev participants, found {len(dev_pids)}"
assert len(test_pids) == 21, f"Expected 21 test participants, found {len(test_pids)}"
assert len(set(dev_pids).intersection(set(test_pids))) == 0, "Leakage between dev and test!"

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=IMAGE_SIZE)
    return np.array(resized, dtype=np.float32)

print("Pre-loading all development images...")
dev_pids_list = []
dev_X = []
dev_y = []

grouped = dev_df.groupby("participant_id")
for pid, group in grouped:
    scans = []
    if len(group) != 3:
        raise ValueError(f"Participant {pid} has {len(group)} scans. Expected 3.")
    for idx, row in group.iterrows():
        scans.append(transform_c5(row["scan_path"]))
    dev_X.append(scans)
    dev_y.append(rev_class_mapping[group.iloc[0]["blood_group"]])
    dev_pids_list.append(pid)
    
dev_X = np.array(dev_X)
dev_y = np.array(dev_y)
dev_pids_list = np.array(dev_pids_list)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_F0_model():
    inputs = tf.keras.Input(shape=(112, 112, 3))
    x = data_augmentation(inputs)
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False
    f = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    drop = tf.keras.layers.Dropout(0.5)(f)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(drop)
    return tf.keras.Model(inputs, outputs)

def build_F1_model():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1 = inputs[:, 0, :, :, :]
    scan2 = inputs[:, 1, :, :, :]
    scan3 = inputs[:, 2, :, :, :]
    
    aug1 = data_augmentation(scan1)
    aug2 = data_augmentation(scan2)
    aug3 = data_augmentation(scan3)
    
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    
    pool = tf.keras.layers.GlobalAveragePooling2D()
    feat1 = pool(base(aug1))
    feat2 = pool(base(aug2))
    feat3 = pool(base(aug3))
    
    concat = tf.keras.layers.Concatenate()([feat1, feat2, feat3])
    
    d = tf.keras.layers.Dense(512, activation="relu")(concat)
    drop = tf.keras.layers.Dropout(0.5)(d)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(drop)
    return tf.keras.Model(inputs, outputs)

def build_F2_model():
    inputs = tf.keras.Input(shape=(3, 112, 112, 3))
    scan1 = inputs[:, 0, :, :, :]
    scan2 = inputs[:, 1, :, :, :]
    scan3 = inputs[:, 2, :, :, :]
    
    aug1 = data_augmentation(scan1)
    aug2 = data_augmentation(scan2)
    aug3 = data_augmentation(scan3)
    
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
    base.trainable = False
    
    pool = tf.keras.layers.GlobalAveragePooling2D()
    feat1 = pool(base(aug1))
    feat2 = pool(base(aug2))
    feat3 = pool(base(aug3))
    
    avg = tf.keras.layers.Average()([feat1, feat2, feat3])
    
    d = tf.keras.layers.Dense(512, activation="relu")(avg)
    drop = tf.keras.layers.Dropout(0.5)(d)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(drop)
    return tf.keras.Model(inputs, outputs)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_indices = list(sgkf.split(dev_X, dev_y, dev_pids_list))

def eval_F0(model_eval, X_part, y_part, pids_part):
    X_flat = X_part.reshape(-1, 112, 112, 3)
    preds_flat = model_eval.predict(X_flat, verbose=0)
    
    y_true_p, y_pred_p, confs = [], [], []
    for i in range(len(pids_part)):
        start = i * 3
        end = start + 3
        probs = preds_flat[start:end]
        avg_prob = np.mean(probs, axis=0)
        y_true_p.append(class_mapping[y_part[i]])
        y_pred_p.append(class_mapping[np.argmax(avg_prob)])
        confs.append(np.max(avg_prob))
        
    acc = accuracy_score(y_true_p, y_pred_p)
    macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
    rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
    return acc, macro_f1, rep, y_true_p, y_pred_p, np.mean(confs)

def eval_F_fusion(model_eval, X_part, y_part, pids_part):
    preds = model_eval.predict(X_part, verbose=0)
    
    y_true_p, y_pred_p, confs = [], [], []
    for i in range(len(pids_part)):
        prob = preds[i]
        y_true_p.append(class_mapping[y_part[i]])
        y_pred_p.append(class_mapping[np.argmax(prob)])
        confs.append(np.max(prob))
        
    acc = accuracy_score(y_true_p, y_pred_p)
    macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
    rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
    return acc, macro_f1, rep, y_true_p, y_pred_p, np.mean(confs)

class F0Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, X_val, y_val, pids_val):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.pids_val = pids_val
        self.best_f1 = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        acc, val_f1, _, _, _, _ = eval_F0(self.model, self.X_val, self.y_val, self.pids_val)
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.wait = 0
            self.model.save_weights(self.filepath)
        else:
            self.wait += 1
        if (epoch + 1) >= MIN_EPOCHS and self.wait >= PATIENCE:
            self.model.stop_training = True

class FFusionCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, X_val, y_val, pids_val):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.pids_val = pids_val
        self.best_f1 = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        acc, val_f1, _, _, _, _ = eval_F_fusion(self.model, self.X_val, self.y_val, self.pids_val)
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.wait = 0
            self.model.save_weights(self.filepath)
        else:
            self.wait += 1
        if (epoch + 1) >= MIN_EPOCHS and self.wait >= PATIENCE:
            self.model.stop_training = True

results = []
fold_results_all = []
methods = ["F0", "F1", "F2"]

for method in methods:
    print(f"\nEvaluating {method}")
    tf.keras.backend.clear_session()
    
    model_save_dir = os.path.join(models_dir, method)
    os.makedirs(model_save_dir, exist_ok=True)
    
    fold_accs, fold_f1s, fold_train_f1s, fold_train_accs = [], [], [], []
    fold_reps, fold_o_shares = [], []
    
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"  Fold {fold+1}/5")
        tf.keras.backend.clear_session()
        
        X_train, y_train, p_train = dev_X[train_idx], dev_y[train_idx], dev_pids_list[train_idx]
        X_val, y_val, p_val = dev_X[val_idx], dev_y[val_idx], dev_pids_list[val_idx]
        
        train_counts = pd.Series(y_train).value_counts().to_dict()
        total_train = len(y_train)
        cw = {i: total_train / (4 * train_counts.get(i, 1.0)) for i in range(4)}
        
        opt_class = tf.keras.optimizers.AdamW if hasattr(tf.keras.optimizers, "AdamW") else tf.keras.optimizers.Adam
        best_model_path = os.path.join(model_save_dir, f"fold_{fold+1}_best.weights.h5")
        
        if method == "F0":
            model = build_F0_model()
            ckpt = F0Checkpoint(best_model_path, X_val, y_val, p_val)
            model.compile(optimizer=opt_class(learning_rate=1e-3, weight_decay=WEIGHT_DECAY), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            
            X_train_flat = X_train.reshape(-1, 112, 112, 3)
            y_train_flat = np.repeat(y_train, 3)
            
            h = model.fit(X_train_flat, y_train_flat, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=cw, callbacks=[ckpt], verbose=0)
            
            if os.path.exists(best_model_path):
                model.load_weights(best_model_path)
                
            train_acc, train_f1, _, _, _, _ = eval_F0(model, X_train, y_train, p_train)
            val_acc, val_f1, val_rep, v_t, v_p, val_conf = eval_F0(model, X_val, y_val, p_val)
            
        else:
            if method == "F1":
                model = build_F1_model()
            elif method == "F2":
                model = build_F2_model()
                
            ckpt = FFusionCheckpoint(best_model_path, X_val, y_val, p_val)
            model.compile(optimizer=opt_class(learning_rate=1e-3, weight_decay=WEIGHT_DECAY), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            
            h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=cw, callbacks=[ckpt], verbose=0)
            
            if os.path.exists(best_model_path):
                model.load_weights(best_model_path)
                
            train_acc, train_f1, _, _, _, _ = eval_F_fusion(model, X_train, y_train, p_train)
            val_acc, val_f1, val_rep, v_t, v_p, val_conf = eval_F_fusion(model, X_val, y_val, p_val)
            
        params = model.count_params()
        o_share = sum(np.array(v_p) == "O+") / len(v_p)
        gap = train_f1 - val_f1
        
        fold_accs.append(val_acc)
        fold_f1s.append(val_f1)
        fold_train_accs.append(train_acc)
        fold_train_f1s.append(train_f1)
        fold_reps.append(val_rep)
        fold_o_shares.append(o_share)
        
        fold_results_all.append({
            "method": method,
            "fold": fold+1,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_O_share": o_share
        })
        
        if fold == 0:
            cm = confusion_matrix(v_t, v_p, labels=classes)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
            plt.title(f"Val Confusion Matrix - {method} Fold 1")
            plt.savefig(os.path.join(reports_dir, "confusion_matrices", f"{method}_fold1_val_cm.png"))
            plt.close()

    results.append({
        "method": method,
        "mean_val_acc": np.mean(fold_accs),
        "std_val_acc": np.std(fold_accs),
        "mean_val_macro_f1": np.mean(fold_f1s),
        "std_val_macro_f1": np.std(fold_f1s),
        "mean_train_acc": np.mean(fold_train_accs),
        "mean_train_macro_f1": np.mean(fold_train_f1s),
        "generalization_gap": np.mean(fold_train_f1s) - np.mean(fold_f1s),
        "mean_A_recall": np.mean([r.get("A+", {}).get("recall", 0) for r in fold_reps]),
        "mean_B_recall": np.mean([r.get("B+", {}).get("recall", 0) for r in fold_reps]),
        "mean_AB_recall": np.mean([r.get("AB+", {}).get("recall", 0) for r in fold_reps]),
        "mean_O_recall": np.mean([r.get("O+", {}).get("recall", 0) for r in fold_reps]),
        "mean_O_share": np.mean(fold_o_shares),
        "trainable_parameters": params
    })

res_df = pd.DataFrame(results).sort_values(by="mean_val_macro_f1", ascending=False)
res_df.to_csv(os.path.join(reports_dir, "feature_fusion_results.csv"), index=False)
pd.DataFrame(fold_results_all).to_csv(os.path.join(reports_dir, "fold_results.csv"), index=False)
best = res_df.iloc[0]

with open(os.path.join(reports_dir, "EXPERIMENT4_REPORT.md"), "w") as f:
    f.write("# EXPERIMENT 4: FEATURE FUSION\n\n")
    f.write("| Method | Val Macro F1 | Val Acc | Train F1 | Gap | O+ Share | Trainable Params |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for idx, row in res_df.iterrows():
        f.write(f"| {row['method']} | {row['mean_val_macro_f1']:.4f} | {row['mean_val_acc']:.4f} | {row['mean_train_macro_f1']:.4f} | {row['generalization_gap']:.4f} | {row['mean_O_share']:.4f} | {row['trainable_parameters']} |\n")

with open(os.path.join(reports_dir, "LEAKAGE_AUDIT.md"), "w") as f:
    f.write("# LEAKAGE AUDIT\nStatus: PASS\n- Only 80 Dev participants used.\n- Scans are strictly kept together per participant.\n- 21 Test participants completely untouched.")

print(f"DONE_MARKER|{best['method']}|{best['mean_val_macro_f1']}|{best['mean_val_acc']}")
