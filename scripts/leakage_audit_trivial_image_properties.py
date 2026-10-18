import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {"A+": 0, "B+": 1, "AB+": 2, "O+": 3}
POSITIVE_CLASSES = ["A+", "B+", "AB+", "O+"]

print("Extracting trivial image properties from 2,769 positive images...")

records = []
for cls_name in POSITIVE_CLASSES:
    cls_dir = DATA_DIR / cls_name
    # Deduplicate paths for Windows case-insensitive filesystem
    raw_paths = list(cls_dir.glob("*.[bB][mM][pP]"))
    unique_paths = sorted(list({p.resolve(): p for p in raw_paths}.values()))
    print(f"  Class {cls_name}: found {len(unique_paths)} unique images")
    for p in unique_paths:
        file_size = os.path.getsize(p)
        with Image.open(p) as img:
            w, h = img.size
            mode = img.mode
            bands = img.getbands()
            has_alpha = int("A" in bands)
            
            # Convert to numpy array
            arr = np.array(img)
            
            # Alpha stats
            if has_alpha and arr.ndim == 3 and arr.shape[2] == 4:
                alpha_channel = arr[:, :, 3].astype(np.float32)
                alpha_mean = float(np.mean(alpha_channel))
                alpha_std = float(np.std(alpha_channel))
                alpha_min = float(np.min(alpha_channel))
                alpha_max = float(np.max(alpha_channel))
            else:
                alpha_mean = 255.0
                alpha_std = 0.0
                alpha_min = 255.0
                alpha_max = 255.0
            
            # Grayscale intensity
            if arr.ndim == 3:
                gray = np.mean(arr[:, :, :3], axis=2).astype(np.float32)
            else:
                gray = arr.astype(np.float32)
                
            img_mean = float(np.mean(gray))
            img_std = float(np.std(gray))
            img_min = float(np.min(gray))
            img_max = float(np.max(gray))
            img_median = float(np.median(gray))
            img_p10 = float(np.percentile(gray, 10))
            img_p25 = float(np.percentile(gray, 25))
            img_p75 = float(np.percentile(gray, 75))
            img_p90 = float(np.percentile(gray, 90))
            
            # Border pixels (width 5)
            border_w = 5
            top = gray[:border_w, :]
            bottom = gray[-border_w:, :]
            left = gray[:, :border_w]
            right = gray[:, -border_w:]
            border_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
            
            border_mean = float(np.mean(border_pixels))
            border_std = float(np.std(border_pixels))
            border_min = float(np.min(border_pixels))
            border_max = float(np.max(border_pixels))
            border_to_center_ratio = border_mean / (img_mean + 1e-6)
            
            # 16-bin histogram
            hist, _ = np.histogram(gray, bins=16, range=(0, 256), density=True)
            hist_entropy = float(-np.sum([p * np.log2(p + 1e-12) for p in hist]))
            hist_peak_bin = int(np.argmax(hist))
            
            rec = {
                "filename": p.name,
                "class_name": cls_name,
                "label": CLASS_MAP[cls_name],
                "file_size": file_size,
                "width": w,
                "height": h,
                "aspect_ratio": w / h,
                "total_pixels": w * h,
                "has_alpha": has_alpha,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "img_mean": img_mean,
                "img_std": img_std,
                "img_min": img_min,
                "img_max": img_max,
                "img_median": img_median,
                "img_p10": img_p10,
                "img_p25": img_p25,
                "img_p75": img_p75,
                "img_p90": img_p90,
                "border_mean": border_mean,
                "border_std": border_std,
                "border_min": border_min,
                "border_max": border_max,
                "border_to_center_ratio": border_to_center_ratio,
                "hist_entropy": hist_entropy,
                "hist_peak_bin": hist_peak_bin
            }
            # Add histogram bins
            for b_idx, b_val in enumerate(hist):
                rec[f"hist_bin_{b_idx}"] = float(b_val)
                
            records.append(rec)

df_features = pd.DataFrame(records)
csv_out = OUT_DIR / "exp3_trivial_image_properties_features.csv"
df_features.to_csv(csv_out, index=False)
print(f"Saved {len(df_features)} feature rows to {csv_out}")

# Feature columns
meta_cols = ["filename", "class_name", "label"]
feature_cols = [c for c in df_features.columns if c not in meta_cols]
X = df_features[feature_cols].values
y = df_features["label"].values

print(f"Feature matrix shape: {X.shape}, labels: {y.shape}")

# Stratified 5-Fold CV Evaluation
SEEDS = [42, 43, 44]
N_SPLITS = 5

classifiers = {
    "LogisticRegression": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))
    ]),
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=seed, class_weight="balanced"
    )
}

results = {
    "feature_count": len(feature_cols),
    "feature_names": feature_cols,
    "total_images": len(y),
    "class_counts": {k: int(np.sum(y == v)) for k, v in CLASS_MAP.items()},
    "models": {}
}

for clf_name, clf_builder in classifiers.items():
    print(f"\n--- Evaluating {clf_name} ---")
    results["models"][clf_name] = {
        "true_labels": {"per_seed": {}, "mean_acc": None, "mean_bal_acc": None, "mean_macro_f1": None},
        "shuffled_control": {"per_seed": {}, "mean_acc": None, "mean_bal_acc": None, "mean_macro_f1": None}
    }
    
    # 1. True labels
    seed_accs = []
    seed_bals = []
    seed_f1s = []
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(y), dtype=int)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            model = clf_builder(seed)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X[val_idx])
            
        acc = accuracy_score(y, oof_preds)
        bal = balanced_accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds, average="macro")
        
        seed_accs.append(acc)
        seed_bals.append(bal)
        seed_f1s.append(f1)
        
        results["models"][clf_name]["true_labels"]["per_seed"][str(seed)] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(f1)
        }
        
    results["models"][clf_name]["true_labels"]["mean_acc"] = float(np.mean(seed_accs))
    results["models"][clf_name]["true_labels"]["std_acc"] = float(np.std(seed_accs, ddof=1))
    results["models"][clf_name]["true_labels"]["mean_bal_acc"] = float(np.mean(seed_bals))
    results["models"][clf_name]["true_labels"]["std_bal_acc"] = float(np.std(seed_bals, ddof=1))
    results["models"][clf_name]["true_labels"]["mean_macro_f1"] = float(np.mean(seed_f1s))
    results["models"][clf_name]["true_labels"]["std_macro_f1"] = float(np.std(seed_f1s, ddof=1))
    
    print(f"  True Labels: Acc={np.mean(seed_accs):.4f} +/- {np.std(seed_accs, ddof=1):.4f}, "
          f"Bal={np.mean(seed_bals):.4f} +/- {np.std(seed_bals, ddof=1):.4f}, "
          f"F1={np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s, ddof=1):.4f}")
    
    # 2. Shuffled labels control
    shuff_accs = []
    shuff_bals = []
    shuff_f1s = []
    
    for seed in SEEDS:
        rng = np.random.RandomState(seed + 1000)
        y_shuff = rng.permutation(y)
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_preds_shuff = np.zeros(len(y), dtype=int)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_shuff)):
            model = clf_builder(seed)
            model.fit(X[train_idx], y_shuff[train_idx])
            oof_preds_shuff[val_idx] = model.predict(X[val_idx])
            
        acc_s = accuracy_score(y_shuff, oof_preds_shuff)
        bal_s = balanced_accuracy_score(y_shuff, oof_preds_shuff)
        f1_s = f1_score(y_shuff, oof_preds_shuff, average="macro")
        
        shuff_accs.append(acc_s)
        shuff_bals.append(bal_s)
        shuff_f1s.append(f1_s)
        
        results["models"][clf_name]["shuffled_control"]["per_seed"][str(seed)] = {
            "accuracy": float(acc_s),
            "balanced_accuracy": float(bal_s),
            "macro_f1": float(f1_s)
        }
        
    results["models"][clf_name]["shuffled_control"]["mean_acc"] = float(np.mean(shuff_accs))
    results["models"][clf_name]["shuffled_control"]["std_acc"] = float(np.std(shuff_accs, ddof=1))
    results["models"][clf_name]["shuffled_control"]["mean_bal_acc"] = float(np.mean(shuff_bals))
    results["models"][clf_name]["shuffled_control"]["std_bal_acc"] = float(np.std(shuff_bals, ddof=1))
    results["models"][clf_name]["shuffled_control"]["mean_macro_f1"] = float(np.mean(shuff_f1s))
    results["models"][clf_name]["shuffled_control"]["std_macro_f1"] = float(np.std(shuff_f1s, ddof=1))
    
    print(f"  Shuffled Control: Acc={np.mean(shuff_accs):.4f} +/- {np.std(shuff_accs, ddof=1):.4f}, "
          f"Bal={np.mean(shuff_bals):.4f} +/- {np.std(shuff_bals, ddof=1):.4f}, "
          f"F1={np.mean(shuff_f1s):.4f} +/- {np.std(shuff_f1s, ddof=1):.4f}")

# Save JSON summary
json_out = OUT_DIR / "exp3_trivial_image_properties_results.json"
with open(json_out, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved complete results to {json_out}")
