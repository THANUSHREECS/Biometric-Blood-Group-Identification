import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
sys.path.insert(0, project_root)
from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target

# load dataset
split_csv_path = os.path.join(project_root, "reports", "phase3_participant_benchmark", "participant_split.csv")
df = pd.read_csv(split_csv_path)
dev_df = df[df["split"].isin(["TRAIN", "VALIDATION"])].copy()

classes = ["A+", "B+", "AB+", "O+"]
rev_class_mapping = {c: i for i, c in enumerate(classes)}
class_mapping = {i: c for i, c in enumerate(classes)}

def build_feature_extractor():
    base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(112,112,3))
    base.trainable = False
    pool = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return tf.keras.Model(inputs=base.input, outputs=pool)

extractor = build_feature_extractor()

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.float32)

print("Extracting features...")
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

X_feat = extractor.predict(X_raw, batch_size=32, verbose=0)

X_mean, y_mean, pid_mean = [], [], []
for pid in np.unique(pid_raw):
    idx = np.where(pid_raw == pid)[0]
    X_mean.append(np.mean(X_feat[idx], axis=0))
    y_mean.append(y_raw[idx[0]])
    pid_mean.append(pid)

X_mean = np.array(X_mean)
y_mean = np.array(y_mean)
pid_mean = np.array(pid_mean)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_generator = list(sgkf.split(X_mean, y_mean, pid_mean)) 

results = []
classification_reports = []
confusion_matrices = []

for rep_name, X_data, y_data, p_data in [("A_SingleScan", X_feat, y_raw, pid_raw), ("B_ThreeScanMean", X_mean, y_mean, pid_mean)]:
    for fold_idx, (train_pid_idx, val_pid_idx) in enumerate(fold_generator):
        
        val_pids_in_fold = pid_mean[val_pid_idx]
        train_pids_in_fold = pid_mean[train_pid_idx]
        
        train_idx = np.where(np.isin(p_data, train_pids_in_fold))[0]
        val_idx = np.where(np.isin(p_data, val_pids_in_fold))[0]
        
        X_tr, y_tr, p_tr = X_data[train_idx], y_data[train_idx], p_data[train_idx]
        X_v, y_v, p_v = X_data[val_idx], y_data[val_idx], p_data[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)
        
        models = {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), 
                                               param_grid={'C': [0.001, 0.01, 0.1, 1.0]}, 
                                               cv=StratifiedGroupKFold(n_splits=2), 
                                               scoring='f1_macro', n_jobs=-1),
            "LinearSVM": GridSearchCV(SVC(kernel='linear', probability=True, class_weight='balanced'),
                                      param_grid={'C': [0.001, 0.01, 0.1, 1.0]},
                                      cv=StratifiedGroupKFold(n_splits=2),
                                      scoring='f1_macro', n_jobs=-1)
        }
        
        for m_name, model in models.items():
            try:
                if m_name == "Dummy":
                    model.fit(X_tr_s, y_tr)
                    best_c = "N/A"
                else:
                    model.fit(X_tr_s, y_tr, groups=p_tr)
                    best_c = model.best_params_['C']
                
                preds_prob = model.predict_proba(X_v_s)
                
                if rep_name == "A_SingleScan":
                    agg_probs, agg_y_true = [], []
                    for pid in np.unique(p_v):
                        idx = np.where(p_v == pid)[0]
                        agg_probs.append(np.mean(preds_prob[idx], axis=0))
                        agg_y_true.append(y_v[idx[0]])
                    agg_probs = np.array(agg_probs)
                    agg_y_true = np.array(agg_y_true)
                    y_pred = np.argmax(agg_probs, axis=1)
                else:
                    agg_y_true = y_v
                    y_pred = np.argmax(preds_prob, axis=1)
                
                mac_f1 = f1_score(agg_y_true, y_pred, average='macro', zero_division=0)
                acc = accuracy_score(agg_y_true, y_pred)
                bal_acc = balanced_accuracy_score(agg_y_true, y_pred)
                rep = classification_report(agg_y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
                cm = confusion_matrix(agg_y_true, y_pred, labels=[0,1,2,3])
                
                results.append({
                    "Representation": rep_name,
                    "Model": m_name,
                    "Fold": fold_idx+1,
                    "Best_C": best_c,
                    "Macro_F1": mac_f1,
                    "Accuracy": acc,
                    "Balanced_Acc": bal_acc
                })
                
                for cls in classes:
                    classification_reports.append({
                        "Representation": rep_name, "Model": m_name, "Fold": fold_idx+1,
                        "Class": cls, "Precision": rep[cls]["precision"], "Recall": rep[cls]["recall"], "F1": rep[cls]["f1-score"]
                    })
                
                for r_idx, r_cls in enumerate(classes):
                    for c_idx, c_cls in enumerate(classes):
                        confusion_matrices.append({
                            "Representation": rep_name, "Model": m_name, "Fold": fold_idx+1,
                            "True_Class": r_cls, "Predicted_Class": c_cls, "Count": cm[r_idx, c_idx]
                        })
                
            except Exception as e:
                print(f"Failed {rep_name} {m_name} fold {fold_idx+1}: {e}")

out_dir = os.path.join(project_root, "reports", "phase4_experiment8")
os.makedirs(out_dir, exist_ok=True)

df_res = pd.DataFrame(results)
print(df_res.groupby(["Representation", "Model"])[["Macro_F1", "Accuracy"]].mean())

df_res.to_csv(os.path.join(out_dir, "EXPERIMENT8_FOLD_RESULTS.csv"), index=False)
pd.DataFrame(classification_reports).to_csv(os.path.join(out_dir, "EXPERIMENT8_CLASSIFICATION_REPORT.csv"), index=False)
pd.DataFrame(confusion_matrices).to_csv(os.path.join(out_dir, "EXPERIMENT8_CONFUSION_MATRICES.csv"), index=False)

print("Finished!")
