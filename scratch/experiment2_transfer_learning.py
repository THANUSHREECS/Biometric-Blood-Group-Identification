import os
import sys
import glob
import json
import time
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
reports_dir = os.path.join(project_root, "reports", "phase3_transfer_learning")
models_dir = os.path.join(project_root, "models", "phase3_transfer_learning")
split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(reports_dir, "plots"), exist_ok=True)

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
X_dev, y_dev, pids_dev = [], [], []
for idx, row in dev_df.iterrows():
    X_dev.append(transform_c5(row["scan_path"]))
    y_dev.append(rev_class_mapping[row["blood_group"]])
    pids_dev.append(row["participant_id"])
    
X_dev = np.array(X_dev)
y_dev = np.array(y_dev)
pids_dev = np.array(pids_dev)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_model(arch_name):
    inputs = tf.keras.Input(shape=(112, 112, 3))
    x = data_augmentation(inputs)
    
    if arch_name == "VGG16":
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "ResNet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "EfficientNet-B0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
        
    flat = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    drop = tf.keras.layers.Dropout(0.5)(flat)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="classifier")(drop)
    model = tf.keras.Model(inputs, outputs)
    return model, base

def configure_strategy(model, base, strategy):
    if strategy == "T1": # Frozen backbone
        base.trainable = False
    elif strategy == "T4": # Full fine tuning
        base.trainable = True
    elif strategy == "T2": # Last block
        base.trainable = True
        unfreeze_started = False
        for layer in base.layers:
            if "block5_conv" in layer.name or "conv5_" in layer.name or "block7a" in layer.name:
                unfreeze_started = True
            layer.trainable = unfreeze_started
    elif strategy == "T3": # Partial fine-tuning (~top 25%)
        base.trainable = True
        n_layers = len(base.layers)
        unfreeze_idx = int(n_layers * 0.75)
        for i, layer in enumerate(base.layers):
            layer.trainable = (i >= unfreeze_idx)

def eval_participant_level(model_eval, X, y, pids):
    preds = model_eval.predict(X, verbose=0)
    pid_to_probs = {}
    pid_to_true = {}
    for i, pid in enumerate(pids):
        if pid not in pid_to_probs:
            pid_to_probs[pid] = []
            pid_to_true[pid] = class_mapping[y[i]]
        pid_to_probs[pid].append(preds[i])
        
    y_true_p, y_pred_p = [], []
    for pid, probs in pid_to_probs.items():
        avg_prob = np.mean(probs, axis=0)
        y_true_p.append(pid_to_true[pid])
        y_pred_p.append(class_mapping[np.argmax(avg_prob)])
        
    acc = accuracy_score(y_true_p, y_pred_p)
    macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
    rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
    return acc, macro_f1, rep, y_true_p, y_pred_p

class ParticipantF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, X_val, pids_val, y_val_arr, min_epochs, patience):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.pids_val = pids_val
        self.y_val_arr = y_val_arr
        self.min_epochs = min_epochs
        self.patience = patience
        self.best_f1 = -1.0
        self.wait = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        acc, val_f1, _, _, _ = eval_participant_level(self.model, self.X_val, self.y_val_arr, self.pids_val)
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.wait = 0
            self.best_epoch = epoch + 1
            self.model.save(self.filepath)
        else:
            self.wait += 1
            
        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            self.model.stop_training = True

architectures = ["DenseNet121", "VGG16", "EfficientNet-B0"]
strategies = ["T1", "T2", "T3", "T4"]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_indices = list(sgkf.split(X_dev, y_dev, pids_dev))

results = []
fold_results_all = []

for arch in architectures:
    for strategy in strategies:
        print(f"\nEvaluating {arch} - {strategy}")
        arch_dir = arch.replace("-", "")
        model_save_dir = os.path.join(models_dir, arch_dir, strategy)
        os.makedirs(model_save_dir, exist_ok=True)
        
        fold_accs, fold_f1s, fold_train_f1s, fold_train_accs = [], [], [], []
        fold_reps, fold_o_shares = [], []
        
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            print(f"  Fold {fold+1}/5")
            tf.keras.backend.clear_session()
            
            X_train, y_train, p_train = X_dev[train_idx], y_dev[train_idx], pids_dev[train_idx]
            X_val, y_val, p_val = X_dev[val_idx], y_dev[val_idx], pids_dev[val_idx]
            
            train_counts = pd.Series(y_train).value_counts().to_dict()
            total_train = len(y_train)
            cw = {i: total_train / (4 * train_counts.get(i, 1.0)) for i in range(4)}
            
            model, base = build_model(arch)
            best_model_path = os.path.join(model_save_dir, f"fold_{fold+1}_best.keras")
            ckpt = ParticipantF1Checkpoint(best_model_path, X_val, p_val, y_val, min_epochs=MIN_EPOCHS, patience=PATIENCE)
            
            opt_class = tf.keras.optimizers.AdamW if hasattr(tf.keras.optimizers, "AdamW") else tf.keras.optimizers.Adam
            
            if strategy == "T1":
                configure_strategy(model, base, "T1")
                model.compile(optimizer=opt_class(learning_rate=1e-3, weight_decay=WEIGHT_DECAY), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=cw, callbacks=[ckpt], verbose=0)
            else:
                base.trainable = False
                model.compile(optimizer=opt_class(learning_rate=1e-3, weight_decay=WEIGHT_DECAY), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10, class_weight=cw, verbose=0)
                
                configure_strategy(model, base, strategy)
                model.compile(optimizer=opt_class(learning_rate=1e-5, weight_decay=WEIGHT_DECAY), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS-10, class_weight=cw, callbacks=[ckpt], verbose=0)
                
            if os.path.exists(best_model_path):
                best_model = tf.keras.models.load_model(best_model_path)
            else:
                best_model = model
                
            train_acc, train_f1, _, _, _ = eval_participant_level(best_model, X_train, y_train, p_train)
            val_acc, val_f1, val_rep, _, v_p = eval_participant_level(best_model, X_val, y_val, p_val)
            
            fold_accs.append(val_acc)
            fold_f1s.append(val_f1)
            fold_train_accs.append(train_acc)
            fold_train_f1s.append(train_f1)
            fold_reps.append(val_rep)
            o_share = sum(np.array(v_p) == "O+") / len(v_p)
            fold_o_shares.append(o_share)
            
            fold_results_all.append({
                "architecture": arch,
                "strategy": strategy,
                "fold": fold+1,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "val_O_share": o_share
            })
            
        results.append({
            "architecture": arch,
            "strategy": strategy,
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
            "folds_completed": 5
        })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by="mean_val_macro_f1", ascending=False)
res_df.to_csv(os.path.join(reports_dir, "transfer_learning_results.csv"), index=False)
pd.DataFrame(fold_results_all).to_csv(os.path.join(reports_dir, "fold_results.csv"), index=False)

with open(os.path.join(reports_dir, "LEAKAGE_AUDIT.md"), "w") as f:
    f.write("# LEAKAGE AUDIT\nStatus: PASS\n- Only 80 Dev participants used in 5-fold CV.\n- 21 Test participants completely untouched.")
    
print("EXPERIMENT 2 COMPLETE")
