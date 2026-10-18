import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
EXP_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Part 1: Feature Group Definitions & Stats
# -------------------------------------------------------------
feat_df = pd.read_csv(EXP_DIR / "exp3_trivial_image_properties_features.csv")
y = feat_df["label"].values

feature_groups = {
    "group_i_dims_filesize": [
        "file_size", "width", "height", "aspect_ratio", "total_pixels"
    ],
    "group_ii_alpha_border": [
        "has_alpha", "alpha_mean", "alpha_std", "alpha_min", "alpha_max",
        "border_mean", "border_std", "border_min", "border_max", "border_to_center_ratio"
    ],
    "group_iii_intensity_hist_moments": [
        "img_mean", "img_std", "img_min", "img_max", "img_median",
        "img_p10", "img_p25", "img_p75", "img_p90",
        "hist_entropy", "hist_peak_bin"
    ] + [f"hist_bin_{i}" for i in range(16)],
    "group_iv_all_features": [c for c in feat_df.columns if c not in ["filename", "class_name", "label"]]
}

# Per-class descriptive stats for dimensions, file size, alpha
descriptive_stats = {}
classes = ["A+", "B+", "AB+", "O+"]
for col in ["width", "height", "file_size", "has_alpha", "alpha_mean"]:
    descriptive_stats[col] = {}
    for cls_name in classes:
        sub = feat_df[feat_df["class_name"] == cls_name][col]
        descriptive_stats[col][cls_name] = {
            "count": int(len(sub)),
            "mean": float(sub.mean()),
            "std": float(sub.std(ddof=1)),
            "min": float(sub.min()),
            "max": float(sub.max())
        }

SEEDS = [42, 43, 44]
N_SPLITS = 5

# -------------------------------------------------------------
# Part 2: Feature Group Ablations (5-fold CV, 3 seeds + shuffled control)
# -------------------------------------------------------------
print("Running Feature Group Ablations...")
ablation_results = {
    "descriptive_stats": descriptive_stats,
    "groups": {}
}

for grp_name, grp_cols in feature_groups.items():
    print(f"\nEvaluating {grp_name} (features: {len(grp_cols)})...")
    X_grp = feat_df[grp_cols].values
    
    ablation_results["groups"][grp_name] = {
        "feature_count": len(grp_cols),
        "features": grp_cols,
        "logistic_regression": {"true": {}, "shuffled": {}},
        "random_forest": {"true": {}, "shuffled": {}}
    }
    
    for model_type in ["logistic_regression", "random_forest"]:
        true_seed_metrics = []
        shuff_seed_metrics = []
        
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            oof_true = np.zeros(len(y), dtype=int)
            oof_shuff = np.zeros(len(y), dtype=int)
            
            rng = np.random.RandomState(seed + 1000)
            y_shuff = rng.permutation(y)
            
            for train_idx, val_idx in skf.split(X_grp, y):
                # Model for true
                if model_type == "logistic_regression":
                    m_true = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))])
                    m_shuff = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))])
                else:
                    m_true = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, class_weight="balanced")
                    m_shuff = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed, class_weight="balanced")
                    
                m_true.fit(X_grp[train_idx], y[train_idx])
                oof_true[val_idx] = m_true.predict(X_grp[val_idx])
                
                m_shuff.fit(X_grp[train_idx], y_shuff[train_idx])
                oof_shuff[val_idx] = m_shuff.predict(X_grp[val_idx])
                
            # True metrics
            acc_t = float(accuracy_score(y, oof_true))
            bal_t = float(balanced_accuracy_score(y, oof_true))
            f1_t = float(f1_score(y, oof_true, average="macro"))
            true_seed_metrics.append((acc_t, bal_t, f1_t))
            
            # Shuffled metrics
            acc_s = float(accuracy_score(y_shuff, oof_shuff))
            bal_s = float(balanced_accuracy_score(y_shuff, oof_shuff))
            f1_s = float(f1_score(y_shuff, oof_shuff, average="macro"))
            shuff_seed_metrics.append((acc_s, bal_s, f1_s))
            
            ablation_results["groups"][grp_name][model_type]["true"][str(seed)] = {"accuracy": acc_t, "balanced_accuracy": bal_t, "macro_f1": f1_t}
            ablation_results["groups"][grp_name][model_type]["shuffled"][str(seed)] = {"accuracy": acc_s, "balanced_accuracy": bal_s, "macro_f1": f1_s}
            
        t_accs = [m[0] for m in true_seed_metrics]
        t_bals = [m[1] for m in true_seed_metrics]
        t_f1s = [m[2] for m in true_seed_metrics]
        s_accs = [m[0] for m in shuff_seed_metrics]
        s_bals = [m[1] for m in shuff_seed_metrics]
        s_f1s = [m[2] for m in shuff_seed_metrics]
        
        ablation_results["groups"][grp_name][model_type]["true"]["summary"] = {
            "mean_acc": float(np.mean(t_accs)), "std_acc": float(np.std(t_accs, ddof=1)),
            "mean_bal": float(np.mean(t_bals)), "std_bal": float(np.std(t_bals, ddof=1)),
            "mean_f1": float(np.mean(t_f1s)), "std_f1": float(np.std(t_f1s, ddof=1))
        }
        ablation_results["groups"][grp_name][model_type]["shuffled"]["summary"] = {
            "mean_acc": float(np.mean(s_accs)), "std_acc": float(np.std(s_accs, ddof=1)),
            "mean_bal": float(np.mean(s_bals)), "std_bal": float(np.std(s_bals, ddof=1)),
            "mean_f1": float(np.mean(s_f1s)), "std_f1": float(np.std(s_f1s, ddof=1))
        }
        print(f"  {model_type} True: Acc={np.mean(t_accs):.4f} +/- {np.std(t_accs, ddof=1):.4f}, Bal={np.mean(t_bals):.4f} +/- {np.std(t_bals, ddof=1):.4f}, F1={np.mean(t_f1s):.4f} +/- {np.std(t_f1s, ddof=1):.4f}")
        print(f"  {model_type} Shuff: Acc={np.mean(s_accs):.4f} +/- {np.std(s_accs, ddof=1):.4f}, Bal={np.mean(s_bals):.4f} +/- {np.std(s_bals, ddof=1):.4f}, F1={np.mean(s_f1s):.4f} +/- {np.std(s_f1s, ddof=1):.4f}")

with open(EXP_DIR / "exp3_trivial_ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print("Saved exp3_trivial_ablation_results.json")

# -------------------------------------------------------------
# Part 3: EfficientNet-B0 Head With vs. Without Trivial Features Concatenated
# -------------------------------------------------------------
print("\nRunning EfficientNet-B0 Head (With vs. Without Trivial Features Concatenated)...")
clusters = pd.read_csv(EXP_DIR / "exp1_positive_images_clusters.csv")
clusters['fname'] = clusters['image_path'].apply(lambda x: x.split('/')[-1].split('\\')[-1])

# Map classes to indices exactly as in Phase 3
c2idx = {"A+": 0, "B+": 1, "AB+": 2, "O+": 3}
y_aligned = clusters["class_name"].map(c2idx).values

trivial_feature_cols = [c for c in feat_df.columns if c not in ["filename", "class_name", "label"]]
merged = clusters.merge(feat_df, left_on='fname', right_on='filename')
trivial_aligned = merged[trivial_feature_cols].values # (2769, 42)

eff_features = np.load(EXP_DIR / "original_dataset_raw_effb0_features.npy") # (2769, 1280)

def build_head(input_dim, seed=42, l2_reg=1e-4, dropout=0.30, lr=1e-3):
    inp = tf.keras.layers.Input(shape=(input_dim,), name="features_in")
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer="zeros",
        name="dense_256"
    )(inp)
    x = tf.keras.layers.Dropout(dropout, name="dropout_0.30")(x)
    out = tf.keras.layers.Dense(
        4, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        bias_initializer="zeros",
        name="out_4"
    )(x)
    model = tf.keras.Model(inp, out, name="Historical_Head")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

concat_results = {
    "without_trivial_features": {"per_seed": {}, "summary": {}},
    "with_trivial_features_concatenated": {"per_seed": {}, "summary": {}}
}

EPOCHS = 50
BATCH_SIZE = 128

for condition in ["without_trivial_features", "with_trivial_features_concatenated"]:
    print(f"\nCondition: {condition}")
    seed_accs = []
    seed_bals = []
    seed_f1s = []
    
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(y_aligned), dtype=int)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(eff_features, y_aligned), 1):
            if condition == "without_trivial_features":
                X_tr = eff_features[train_idx]
                X_va = eff_features[val_idx]
            else:
                scaler = StandardScaler()
                triv_tr = scaler.fit_transform(trivial_aligned[train_idx])
                triv_va = scaler.transform(trivial_aligned[val_idx])
                X_tr = np.concatenate([eff_features[train_idx], triv_tr], axis=1)
                X_va = np.concatenate([eff_features[val_idx], triv_va], axis=1)
                
            y_tr = y_aligned[train_idx]
            counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
            cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
            sw = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
            
            head = build_head(input_dim=X_tr.shape[1], seed=seed + fold * 10)
            head.fit(
                X_tr, y_tr,
                sample_weight=sw,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=0
            )
            
            probs = head.predict(X_va, batch_size=BATCH_SIZE, verbose=0)
            oof_preds[val_idx] = np.argmax(probs, axis=1)
            
        acc = float(accuracy_score(y_aligned, oof_preds))
        bal = float(balanced_accuracy_score(y_aligned, oof_preds))
        f1 = float(f1_score(y_aligned, oof_preds, average="macro", zero_division=0))
        
        seed_accs.append(acc)
        seed_bals.append(bal)
        seed_f1s.append(f1)
        
        concat_results[condition]["per_seed"][str(seed)] = {
            "accuracy": acc, "balanced_accuracy": bal, "macro_f1": f1
        }
        print(f"  Seed {seed}: Acc={acc:.4f}, Bal={bal:.4f}, F1={f1:.4f}")
        
    concat_results[condition]["summary"] = {
        "mean_acc": float(np.mean(seed_accs)), "std_acc": float(np.std(seed_accs, ddof=1)),
        "mean_bal": float(np.mean(seed_bals)), "std_bal": float(np.std(seed_bals, ddof=1)),
        "mean_f1": float(np.mean(seed_f1s)), "std_f1": float(np.std(seed_f1s, ddof=1))
    }
    print(f"  Summary: Acc={np.mean(seed_accs):.4f} +/- {np.std(seed_accs, ddof=1):.4f}, Bal={np.mean(seed_bals):.4f} +/- {np.std(seed_bals, ddof=1):.4f}, F1={np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s, ddof=1):.4f}")

with open(EXP_DIR / "exp3_effnet_concat_results.json", "w") as f:
    json.dump(concat_results, f, indent=2)
print("Saved exp3_effnet_concat_results.json")
