import os
import sys
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from scipy.fftpack import fft2, fftshift

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

def transform_c5(img_path):
    pil_img = load_image_to_pil(img_path)
    rgb_img = composite_alpha_on_white(pil_img)
    cropped, _ = crop_mfs100_foreground(rgb_img)
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    resized = resize_to_target(enhanced, target_size=(112, 112))
    return np.array(resized, dtype=np.uint8)

# Feature extractors
def extract_h0_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    rms_contrast = gray.std()
    return np.array([lap_var, rms_contrast])

def extract_h1_orientation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fd = hog(gray, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, feature_vector=True)
    return fd

def extract_h2_frequency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    return np.array([magnitude_spectrum.mean(), magnitude_spectrum.std(), np.percentile(magnitude_spectrum, 90)])

def extract_h3_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

print("Extracting handcrafted features...")
features = []
y_raw, pid_raw = [], []
for pid, group in dev_df.groupby("participant_id"):
    for _, row in group.iterrows():
        img = transform_c5(row["scan_path"])
        h0 = extract_h0_quality(img)
        h1 = extract_h1_orientation(img)
        h2 = extract_h2_frequency(img)
        h3 = extract_h3_texture(img)
        
        features.append({
            "H0": h0,
            "H1": h1,
            "H2": h2,
            "H3": h3,
            "H4": np.concatenate([h1, h2]),
            "H5": np.concatenate([h0, h1, h2, h3])
        })
        y_raw.append(rev_class_mapping[row["blood_group"]])
        pid_raw.append(pid)

y_raw = np.array(y_raw)
pid_raw = np.array(pid_raw)

# Participant-level averaging
X_dict_mean = {k: [] for k in ["H0", "H1", "H2", "H3", "H4", "H5"]}
y_mean, pid_mean = [], []
for pid in np.unique(pid_raw):
    idx = np.where(pid_raw == pid)[0]
    for k in X_dict_mean.keys():
        f_stack = np.array([features[i][k] for i in idx])
        X_dict_mean[k].append(np.mean(f_stack, axis=0))
    y_mean.append(y_raw[idx[0]])
    pid_mean.append(pid)

for k in X_dict_mean:
    X_dict_mean[k] = np.array(X_dict_mean[k])
y_mean = np.array(y_mean)
pid_mean = np.array(pid_mean)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_generator = list(sgkf.split(X_dict_mean["H0"], y_mean, pid_mean))

results = []
cm_results = []

for feat_group in ["H0", "H1", "H2", "H3", "H4", "H5"]:
    X_data = X_dict_mean[feat_group]
    
    for fold_idx, (train_pid_idx, val_pid_idx) in enumerate(fold_generator):
        val_pids_in_fold = pid_mean[val_pid_idx]
        train_pids_in_fold = pid_mean[train_pid_idx]
        
        train_idx = np.where(np.isin(pid_mean, train_pids_in_fold))[0]
        val_idx = np.where(np.isin(pid_mean, val_pids_in_fold))[0]
        
        X_tr, y_tr, p_tr = X_data[train_idx], y_mean[train_idx], pid_mean[train_idx]
        X_v, y_v, p_v = X_data[val_idx], y_mean[val_idx], pid_mean[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)
        
        models = {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), 
                                               param_grid={'C': [0.001, 0.01, 0.1, 1.0]}, 
                                               cv=StratifiedGroupKFold(n_splits=2), 
                                               scoring='f1_macro', n_jobs=-1),
            "LinearSVM": GridSearchCV(SVC(kernel='linear', class_weight='balanced'),
                                      param_grid={'C': [0.001, 0.01, 0.1, 1.0]},
                                      cv=StratifiedGroupKFold(n_splits=2),
                                      scoring='f1_macro', n_jobs=-1)
        }
        
        for m_name, model in models.items():
            try:
                if m_name == "Dummy":
                    model.fit(X_tr_s, y_tr)
                else:
                    model.fit(X_tr_s, y_tr, groups=p_tr)
                
                y_pred = model.predict(X_v_s)
                
                mac_f1 = f1_score(y_v, y_pred, average='macro', zero_division=0)
                acc = accuracy_score(y_v, y_pred)
                bal_acc = balanced_accuracy_score(y_v, y_pred)
                cm = confusion_matrix(y_v, y_pred, labels=[0,1,2,3])
                
                results.append({
                    "FeatureGroup": feat_group,
                    "Model": m_name,
                    "Fold": fold_idx+1,
                    "Macro_F1": mac_f1,
                    "Accuracy": acc,
                    "Balanced_Acc": bal_acc
                })
                
                for r_idx, r_cls in enumerate(classes):
                    for c_idx, c_cls in enumerate(classes):
                        cm_results.append({
                            "FeatureGroup": feat_group, "Model": m_name, "Fold": fold_idx+1,
                            "True_Class": r_cls, "Predicted_Class": c_cls, "Count": cm[r_idx, c_idx]
                        })
                        
            except Exception as e:
                print(f"Failed {feat_group} {m_name} fold {fold_idx+1}: {e}")

out_dir = os.path.join(project_root, "reports", "phase4_experiment9")
os.makedirs(out_dir, exist_ok=True)

df_res = pd.DataFrame(results)
print(df_res.groupby(["FeatureGroup", "Model"])[["Macro_F1", "Accuracy"]].mean())

df_res.to_csv(os.path.join(out_dir, "EXPERIMENT9_FOLD_RESULTS.csv"), index=False)
pd.DataFrame(cm_results).to_csv(os.path.join(out_dir, "EXPERIMENT9_CONFUSION_MATRICES.csv"), index=False)

print("Finished!")
