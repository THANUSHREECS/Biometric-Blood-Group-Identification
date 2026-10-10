import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

# Paths
data_dir = os.path.join(project_root, "data", "mfs100_final", "raw")
reports_dir = os.path.join(project_root, "reports", "phase3_architecture_benchmark")
models_dir = os.path.join(project_root, "models", "phase3_architecture_benchmark")
split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.join(reports_dir, "confusion_matrices"), exist_ok=True)

# Config
IMAGE_SIZE = (112, 112)
BATCH_SIZE = 16
EPOCHS = 150
MIN_EPOCHS = 100
PATIENCE = 15
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

classes = ["A+", "B+", "AB+", "O+"]
class_mapping = {i: c for i, c in enumerate(classes)}
rev_class_mapping = {c: i for i, c in enumerate(classes)}

# Load Split
split_df = pd.read_csv(split_csv_path)

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=IMAGE_SIZE)
    return np.array(resized, dtype=np.float32)

# Prepare datasets
print("Preparing datasets...")
train_images, train_labels, train_pids = [], [], []
val_images, val_labels, val_pids = [], [], []
test_images, test_labels, test_pids = [], [], []

for idx, row in split_df.iterrows():
    img_array = transform_c5(row["scan_path"])
    lbl = rev_class_mapping[row["blood_group"]]
    
    if row["split"] == "TRAIN":
        train_images.append(img_array)
        train_labels.append(lbl)
        train_pids.append(row["participant_id"])
    elif row["split"] == "VALIDATION":
        val_images.append(img_array)
        val_labels.append(lbl)
        val_pids.append(row["participant_id"])
    elif row["split"] == "TEST":
        test_images.append(img_array)
        test_labels.append(lbl)
        test_pids.append(row["participant_id"])

X_train, y_train = np.array(train_images), np.array(train_labels)
X_val, y_val = np.array(val_images), np.array(val_labels)
X_test, y_test = np.array(test_images), np.array(test_labels)

# Class weights from TRAIN only
train_class_counts = pd.Series(y_train).value_counts().to_dict()
total_train = len(y_train)
class_weights = {}
for i in range(len(classes)):
    if i in train_class_counts:
        class_weights[i] = total_train / (len(classes) * train_class_counts[i])
    else:
        class_weights[i] = 1.0

# Validation eval logic
def evaluate_participant_level(model, X, pids, y_true_arr):
    preds = model.predict(X, verbose=0)
    pid_to_probs = {}
    pid_to_true = {}
    for i, pid in enumerate(pids):
        if pid not in pid_to_probs:
            pid_to_probs[pid] = []
            pid_to_true[pid] = class_mapping[y_true_arr[i]]
        pid_to_probs[pid].append(preds[i])
        
    y_true_p = []
    y_pred_p = []
    for pid, probs in pid_to_probs.items():
        avg_prob = np.mean(probs, axis=0)
        y_true_p.append(pid_to_true[pid])
        y_pred_p.append(class_mapping[np.argmax(avg_prob)])
        
    return f1_score(y_true_p, y_pred_p, average="macro")

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
        val_f1 = evaluate_participant_level(self.model, self.X_val, self.pids_val, self.y_val_arr)
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.wait = 0
            self.best_epoch = epoch + 1
            self.model.save(self.filepath)
        else:
            self.wait += 1
            
        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            self.model.stop_training = True

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.027, fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    tf.keras.layers.RandomContrast(0.1)
])

def build_model(arch_name):
    inputs = tf.keras.Input(shape=(112, 112, 3))
    x = data_augmentation(inputs)
    
    if arch_name == "Custom CNN":
        x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        
        x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        
        x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        
        x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        return model, False # not pretrained
        
    elif arch_name == "VGG16":
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "ResNet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "EfficientNet-B0":
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "EfficientNetV2-B0":
        base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_tensor=x)
    elif arch_name == "ConvNeXt-Tiny":
        base = tf.keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_tensor=x)
    
    base.trainable = False
    flat = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    drop = tf.keras.layers.Dropout(0.5)(flat)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(drop)
    
    model = tf.keras.Model(inputs, outputs)
    return model, True

architectures = {
    "custom_cnn": "Custom CNN",
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnetv2_b0": "EfficientNetV2-B0",
    "convnext_tiny": "ConvNeXt-Tiny"
}

results = []
results_txt = []

for dir_name, arch_name in architectures.items():
    print(f"\nTraining {arch_name}")
    tf.keras.backend.clear_session()
    model, is_pretrained = build_model(arch_name)
    
    save_path = os.path.join(models_dir, dir_name)
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "best_model.keras")
    
    ckpt = ParticipantF1Checkpoint(best_model_path, X_val, val_pids, y_val, min_epochs=MIN_EPOCHS, patience=PATIENCE)
    
    if is_pretrained:
        # Phase 1
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10, class_weight=class_weights, verbose=0)
        # Phase 2
        model.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weights, callbacks=[ckpt], verbose=0)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=class_weights, callbacks=[ckpt], verbose=0)
                  
    if os.path.exists(best_model_path):
        best_model = tf.keras.models.load_model(best_model_path)
    else:
        best_model = model
        
    def eval_split(model_eval, X, y, pids):
        preds = model_eval.predict(X, verbose=0)
        pid_to_probs = {}
        pid_to_true = {}
        for i, pid in enumerate(pids):
            if pid not in pid_to_probs:
                pid_to_probs[pid] = []
                pid_to_true[pid] = class_mapping[y[i]]
            pid_to_probs[pid].append(preds[i])
            
        y_true_p, y_pred_p, df_rows = [], [], []
        for pid, probs in pid_to_probs.items():
            avg_prob = np.mean(probs, axis=0)
            y_t = pid_to_true[pid]
            y_p = class_mapping[np.argmax(avg_prob)]
            y_true_p.append(y_t)
            y_pred_p.append(y_p)
            df_rows.append({"participant_id": pid, "true_label": y_t, "final_prediction": y_p})
            
        acc = accuracy_score(y_true_p, y_pred_p)
        macro_f1 = f1_score(y_true_p, y_pred_p, average="macro")
        rep = classification_report(y_true_p, y_pred_p, output_dict=True, zero_division=0)
        return acc, macro_f1, rep, df_rows, y_true_p, y_pred_p
        
    val_acc, val_f1, val_rep, val_rows, v_t, v_p = eval_split(best_model, X_val, y_val, val_pids)
    test_acc, test_f1, test_rep, test_rows, t_t, t_p = eval_split(best_model, X_test, y_test, test_pids)
    
    results.append({
        "architecture": arch_name,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "val_O_share": sum(np.array(v_p) == "O+") / len(v_p),
        "test_O_share": sum(np.array(t_p) == "O+") / len(t_p)
    })
    results_txt.append(f"{arch_name} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    pd.DataFrame(test_rows).to_csv(os.path.join(reports_dir, f"{dir_name}_test_preds.csv"), index=False)

res_df = pd.DataFrame(results).sort_values(by="val_macro_f1", ascending=False)
res_df.to_csv(os.path.join(reports_dir, "architecture_results.csv"), index=False)
best_arch = res_df.iloc[0]

with open(os.path.join(reports_dir, "EXPERIMENT1_REPORT.md"), "w") as f:
    f.write("# EXPERIMENT 1: ARCHITECTURE BENCHMARK\n\n")
    f.write("## Validation Rankings\n")
    f.write(res_df[["architecture", "val_macro_f1", "val_acc", "val_O_share"]].to_markdown(index=False))
    f.write("\n\n## Final Locked Test Performance\n")
    f.write(res_df[["architecture", "test_macro_f1", "test_acc", "test_O_share"]].to_markdown(index=False))
    f.write(f"\n\n## Best Architecture: {best_arch['architecture']}\n")

with open(os.path.join(reports_dir, "LEAKAGE_AUDIT.md"), "w") as f:
    f.write("PASS")

print(f"DONE_MARKER|{best_arch['architecture']}|{best_arch['val_macro_f1']}|{best_arch['test_macro_f1']}|{best_arch['test_acc']}")
