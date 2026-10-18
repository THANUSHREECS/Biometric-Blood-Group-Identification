import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
OUT_DIR = PROJECT_ROOT / "experiments" / "leakage_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 3: SIMILARITY-PURGED 5-FOLD CV ON ORIGINAL DATASET (2,769 POSITIVE IMAGES)")
print("="*80)

# 1. Load Data
# DenseNet-121 features for similarity calculation (normalized)
dn_data = np.load(OUT_DIR / "original_dataset_densenet121_features.npz", allow_pickle=True)
dn_feats = dn_data["features"] # (2769, 1024)
dn_norms = np.linalg.norm(dn_feats, axis=1, keepdims=True)
dn_norm = dn_feats / np.clip(dn_norms, 1e-12, None)

# Raw GAP EfficientNet-B0 features for classifier head (UN-NORMALIZED!)
eff_raw = np.load(OUT_DIR / "original_dataset_raw_effb0_features.npy") # (2769, 1280)
print(f"Loaded DenseNet-121 features (for purge): {dn_norm.shape}")
print(f"Loaded Raw GAP EfficientNet-B0 features (classifier input): {eff_raw.shape}")
print(f"  Raw GAP Norms: min={np.min(np.linalg.norm(eff_raw, axis=1)):.2f}, mean={np.mean(np.linalg.norm(eff_raw, axis=1)):.2f}, max={np.max(np.linalg.norm(eff_raw, axis=1)):.2f}")

manifest = pd.read_csv(PROJECT_ROOT / "outputs" / "dataset_manifest.csv")
pos = manifest[manifest["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy().reset_index(drop=True)
classes = ["A+", "B+", "AB+", "O+"]
c2idx = {c: i for i, c in enumerate(classes)}
y_all = pos["class_name"].map(c2idx).values

# Documented Deviations
DEVIATIONS = {
    "feature_input": "Raw GAP EfficientNet-B0 features (un-normalized)",
    "batch_size": 128, # historical was 32
    "fixed_epochs": 50, # historical used 2-stage fine-tuning with early stopping
    "data_augmentation": "None (omitted)",
    "backbone_training": "Strictly frozen (no fine-tuning of top 25 layers)"
}
print(f"\nDocumented Deviations from Historical Recipe: {json.dumps(DEVIATIONS, indent=2)}")

FIXED_EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
PURGE_LEVELS = [0.0, 0.10, 0.25, 0.50]
SEEDS = [42, 43, 44]

def build_head(seed=42):
    inp = tf.keras.Input(shape=(1280,), name="raw_gap_input")
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        bias_initializer="zeros",
        name="dense_256"
    )(inp)
    x = tf.keras.layers.Dropout(0.30, name="dropout_0.30")(x)
    out = tf.keras.layers.Dense(
        4, activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        bias_initializer="zeros",
        name="out_4"
    )(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="Historical_Head")
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# CSV Logger for fold-level results
CSV_PATH = OUT_DIR / "phase3_purged_cv_results.csv"
JSON_PATH = OUT_DIR / "phase3_purged_cv_summary.json"

records = []

# =========================================================================
# RUN PRIMARY ARM: FIXED 50 EPOCHS
# =========================================================================
print("\n" + "="*80)
print("EXECUTING PRIMARY ARM: FIXED 50 EPOCHS")
print("="*80)

skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_splits = list(skf_outer.split(eff_raw, y_all))

for seed in SEEDS:
    print(f"\n>>> Primary Arm (Fixed 50 Epochs) | Seed {seed} <<<")
    t_seed_start = time.time()
    
    # 1. Plain 0% Baseline & Label-Shuffled Control at 0%
    for is_shuffled in [False, True]:
        cond_name = "0%_label_shuffled_control" if is_shuffled else "0%_unpurged_baseline"
        oof_preds = np.zeros(len(y_all), dtype=int)
        
        y_eval = y_all.copy()
        if is_shuffled:
            rng_shuff = np.random.RandomState(seed + 999)
            y_eval = rng_shuff.permutation(y_eval)
            
        for fold, (tr_idx, val_idx) in enumerate(outer_splits, 1):
            X_tr, y_tr = eff_raw[tr_idx], y_eval[tr_idx]
            X_val, y_val = eff_raw[val_idx], y_eval[val_idx]
            
            counts = {i: int(np.sum(y_tr == i)) for i in range(4)}
            cw = {i: float(len(y_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
            sw = np.array([cw[int(lbl)] for lbl in y_tr], dtype=np.float32)
            
            model = build_head(seed=seed + fold * 10)
            hist = model.fit(X_tr, y_tr, sample_weight=sw, batch_size=BATCH_SIZE, epochs=FIXED_EPOCHS, verbose=0)
            final_train_acc = float(hist.history["accuracy"][-1])
            
            preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
            oof_preds[val_idx] = preds
            
            fold_acc = float(accuracy_score(y_val, preds))
            fold_bal = float(balanced_accuracy_score(y_val, preds))
            fold_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
            
            records.append({
                "arm": "Arm1_FixedEpochs",
                "condition": cond_name,
                "purge_fraction": 0.0,
                "seed": seed,
                "fold": fold,
                "train_count": len(X_tr),
                "val_count": len(X_val),
                "final_train_acc": final_train_acc,
                "val_accuracy": fold_acc,
                "val_balanced_accuracy": fold_bal,
                "val_macro_f1": fold_f1,
                "class_counts_train": counts
            })
            
        ov_acc = accuracy_score(y_eval, oof_preds)
        ov_bal = balanced_accuracy_score(y_eval, oof_preds)
        ov_f1 = f1_score(y_eval, oof_preds, average="macro", zero_division=0)
        print(f"  {cond_name:<30}: Acc={ov_acc*100:.2f}%, BalAcc={ov_bal*100:.2f}%, MacroF1={ov_f1:.4f}")

    # 2. Purged Arm vs. Random-Drop Control at 10%, 25%, 50%
    for p_frac in [0.10, 0.25, 0.50]:
        oof_preds_purged = np.zeros(len(y_all), dtype=int)
        oof_preds_random = np.zeros(len(y_all), dtype=int)
        
        for fold, (tr_idx, val_idx) in enumerate(outer_splits, 1):
            X_tr_full, y_tr_full = eff_raw[tr_idx], y_all[tr_idx]
            X_val, y_val = eff_raw[val_idx], y_all[val_idx]
            
            dn_tr = dn_norm[tr_idx]
            dn_val = dn_norm[val_idx]
            
            # Compute max cosine similarity of each training image to ANY held-out image in this fold
            sim_mat = np.dot(dn_tr, dn_val.T)
            max_sims = np.max(sim_mat, axis=1)
            
            n_purge = int(np.round(p_frac * len(tr_idx)))
            
            # --- PURGED ARM ---
            # Drop top n_purge highest similarity images
            purge_order = np.argsort(max_sims)[::-1]
            purged_indices_local = purge_order[:n_purge]
            keep_indices_purged = purge_order[n_purge:]
            
            # Compute per-class purge counts k_c
            purged_classes = y_tr_full[purged_indices_local]
            k_c = {c: int(np.sum(purged_classes == c)) for c in range(4)}
            
            X_tr_p, y_tr_p = X_tr_full[keep_indices_purged], y_tr_full[keep_indices_purged]
            counts_p = {i: int(np.sum(y_tr_p == i)) for i in range(4)}
            cw_p = {i: float(len(y_tr_p) / (4.0 * max(counts_p[i], 1))) for i in range(4)}
            sw_p = np.array([cw_p[int(lbl)] for lbl in y_tr_p], dtype=np.float32)
            
            model_p = build_head(seed=seed + fold * 10)
            hist_p = model_p.fit(X_tr_p, y_tr_p, sample_weight=sw_p, batch_size=BATCH_SIZE, epochs=FIXED_EPOCHS, verbose=0)
            tr_acc_p = float(hist_p.history["accuracy"][-1])
            
            preds_p = np.argmax(model_p.predict(X_val, verbose=0), axis=1)
            oof_preds_purged[val_idx] = preds_p
            
            records.append({
                "arm": "Arm1_FixedEpochs",
                "condition": "similarity_purged",
                "purge_fraction": p_frac,
                "seed": seed,
                "fold": fold,
                "train_count": len(X_tr_p),
                "val_count": len(X_val),
                "final_train_acc": tr_acc_p,
                "val_accuracy": float(accuracy_score(y_val, preds_p)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, preds_p)),
                "val_macro_f1": float(f1_score(y_val, preds_p, average="macro", zero_division=0)),
                "class_counts_train": counts_p,
                "purged_class_counts": k_c
            })
            
            # --- RANDOM-DROP CONTROL ARM ---
            # Drop exactly k_c images randomly from class c
            rng_rd = np.random.RandomState(seed * 100 + fold * 10 + int(p_frac * 100))
            drop_indices_local = []
            for c in range(4):
                c_indices = np.where(y_tr_full == c)[0]
                n_drop_c = k_c[c]
                dropped_c = rng_rd.choice(c_indices, size=n_drop_c, replace=False)
                drop_indices_local.extend(dropped_c)
                
            drop_set = set(drop_indices_local)
            keep_indices_random = [i for i in range(len(tr_idx)) if i not in drop_set]
            
            X_tr_r, y_tr_r = X_tr_full[keep_indices_random], y_tr_full[keep_indices_random]
            counts_r = {i: int(np.sum(y_tr_r == i)) for i in range(4)}
            cw_r = {i: float(len(y_tr_r) / (4.0 * max(counts_r[i], 1))) for i in range(4)}
            sw_r = np.array([cw_r[int(lbl)] for lbl in y_tr_r], dtype=np.float32)
            
            model_r = build_head(seed=seed + fold * 10)
            hist_r = model_r.fit(X_tr_r, y_tr_r, sample_weight=sw_r, batch_size=BATCH_SIZE, epochs=FIXED_EPOCHS, verbose=0)
            tr_acc_r = float(hist_r.history["accuracy"][-1])
            
            preds_r = np.argmax(model_r.predict(X_val, verbose=0), axis=1)
            oof_preds_random[val_idx] = preds_r
            
            records.append({
                "arm": "Arm1_FixedEpochs",
                "condition": "random_drop_control",
                "purge_fraction": p_frac,
                "seed": seed,
                "fold": fold,
                "train_count": len(X_tr_r),
                "val_count": len(X_val),
                "final_train_acc": tr_acc_r,
                "val_accuracy": float(accuracy_score(y_val, preds_r)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, preds_r)),
                "val_macro_f1": float(f1_score(y_val, preds_r, average="macro", zero_division=0)),
                "class_counts_train": counts_r,
                "purged_class_counts": k_c
            })

        bal_p = balanced_accuracy_score(y_all, oof_preds_purged)
        bal_r = balanced_accuracy_score(y_all, oof_preds_random)
        diff = bal_p - bal_r
        print(f"  Level {int(p_frac*100):2d}%: Purged BalAcc={bal_p*100:5.2f}% | Random-Drop BalAcc={bal_r*100:5.2f}% | Delta = {diff*100:+5.2f}%")

    # Save checkpoint after each seed
    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    print(f"  --> Checkpointed results after Seed {seed} ({time.time()-t_seed_start:.1f}s)")

# =========================================================================
# RUN SECONDARY ARM: INNER STRATIFIED VALIDATION SPLIT EARLY STOPPING
# =========================================================================
print("\n" + "="*80)
print("EXECUTING SECONDARY ARM: INNER VALIDATION SPLIT EARLY STOPPING")
print("="*80)

for seed in SEEDS:
    print(f"\n>>> Secondary Arm (Inner Val Early Stopping) | Seed {seed} <<<")
    t_seed_start = time.time()
    
    # 0% Baseline
    oof_preds_base = np.zeros(len(y_all), dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(outer_splits, 1):
        X_outer_tr, y_outer_tr = eff_raw[tr_idx], y_all[tr_idx]
        X_val, y_val = eff_raw[val_idx], y_all[val_idx]
        
        skf_in = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
        in_tr, in_val = next(skf_in.split(X_outer_tr, y_outer_tr))
        
        X_in_tr, y_in_tr = X_outer_tr[in_tr], y_outer_tr[in_tr]
        X_in_val, y_in_val = X_outer_tr[in_val], y_outer_tr[in_val]
        
        counts = {i: int(np.sum(y_in_tr == i)) for i in range(4)}
        cw = {i: float(len(y_in_tr) / (4.0 * max(counts[i], 1))) for i in range(4)}
        sw = np.array([cw[int(lbl)] for lbl in y_in_tr], dtype=np.float32)
        
        model = build_head(seed=seed + fold * 10)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        hist = model.fit(X_in_tr, y_in_tr, sample_weight=sw, batch_size=BATCH_SIZE, epochs=80,
                         validation_data=(X_in_val, y_in_val), callbacks=[es], verbose=0)
        
        preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
        oof_preds_base[val_idx] = preds
        
        records.append({
            "arm": "Arm2_InnerValidation",
            "condition": "0%_unpurged_baseline",
            "purge_fraction": 0.0,
            "seed": seed,
            "fold": fold,
            "train_count": len(X_in_tr),
            "val_count": len(X_val),
            "final_train_acc": float(hist.history["accuracy"][-1]),
            "val_accuracy": float(accuracy_score(y_val, preds)),
            "val_balanced_accuracy": float(balanced_accuracy_score(y_val, preds)),
            "val_macro_f1": float(f1_score(y_val, preds, average="macro", zero_division=0)),
            "class_counts_train": counts
        })
    ov_bal_base = balanced_accuracy_score(y_all, oof_preds_base)
    print(f"  0% Baseline: BalAcc = {ov_bal_base*100:.2f}%")

    # Purged vs Random at 10%, 25%, 50%
    for p_frac in [0.10, 0.25, 0.50]:
        oof_preds_purged = np.zeros(len(y_all), dtype=int)
        oof_preds_random = np.zeros(len(y_all), dtype=int)
        
        for fold, (tr_idx, val_idx) in enumerate(outer_splits, 1):
            X_tr_full, y_tr_full = eff_raw[tr_idx], y_all[tr_idx]
            X_val, y_val = eff_raw[val_idx], y_all[val_idx]
            
            dn_tr = dn_norm[tr_idx]
            dn_val = dn_norm[val_idx]
            sim_mat = np.dot(dn_tr, dn_val.T)
            max_sims = np.max(sim_mat, axis=1)
            
            n_purge = int(np.round(p_frac * len(tr_idx)))
            
            # --- PURGE FIRST ---
            purge_order = np.argsort(max_sims)[::-1]
            purged_indices_local = purge_order[:n_purge]
            keep_indices_purged = purge_order[n_purge:]
            
            purged_classes = y_tr_full[purged_indices_local]
            k_c = {c: int(np.sum(purged_classes == c)) for c in range(4)}
            
            X_tr_p_rem, y_tr_p_rem = X_tr_full[keep_indices_purged], y_tr_full[keep_indices_purged]
            
            # THEN INNER SPLIT FROM REMAINING
            skf_in_p = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            in_tr_p, in_val_p = next(skf_in_p.split(X_tr_p_rem, y_tr_p_rem))
            
            X_in_tr_p, y_in_tr_p = X_tr_p_rem[in_tr_p], y_tr_p_rem[in_tr_p]
            X_in_val_p, y_in_val_p = X_tr_p_rem[in_val_p], y_tr_p_rem[in_val_p]
            
            counts_p = {i: int(np.sum(y_in_tr_p == i)) for i in range(4)}
            cw_p = {i: float(len(y_in_tr_p) / (4.0 * max(counts_p[i], 1))) for i in range(4)}
            sw_p = np.array([cw_p[int(lbl)] for lbl in y_in_tr_p], dtype=np.float32)
            
            model_p = build_head(seed=seed + fold * 10)
            es_p = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            hist_p = model_p.fit(X_in_tr_p, y_in_tr_p, sample_weight=sw_p, batch_size=BATCH_SIZE, epochs=80,
                                 validation_data=(X_in_val_p, y_in_val_p), callbacks=[es_p], verbose=0)
            
            preds_p = np.argmax(model_p.predict(X_val, verbose=0), axis=1)
            oof_preds_purged[val_idx] = preds_p
            
            records.append({
                "arm": "Arm2_InnerValidation",
                "condition": "similarity_purged",
                "purge_fraction": p_frac,
                "seed": seed,
                "fold": fold,
                "train_count": len(X_in_tr_p),
                "val_count": len(X_val),
                "final_train_acc": float(hist_p.history["accuracy"][-1]),
                "val_accuracy": float(accuracy_score(y_val, preds_p)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, preds_p)),
                "val_macro_f1": float(f1_score(y_val, preds_p, average="macro", zero_division=0)),
                "class_counts_train": counts_p,
                "purged_class_counts": k_c
            })
            
            # --- RANDOM-DROP CONTROL ---
            rng_rd = np.random.RandomState(seed * 100 + fold * 10 + int(p_frac * 100))
            drop_indices_local = []
            for c in range(4):
                c_indices = np.where(y_tr_full == c)[0]
                n_drop_c = k_c[c]
                dropped_c = rng_rd.choice(c_indices, size=n_drop_c, replace=False)
                drop_indices_local.extend(dropped_c)
                
            drop_set = set(drop_indices_local)
            keep_indices_random = [i for i in range(len(tr_idx)) if i not in drop_set]
            
            X_tr_r_rem, y_tr_r_rem = X_tr_full[keep_indices_random], y_tr_full[keep_indices_random]
            
            # THEN INNER SPLIT FROM REMAINING
            skf_in_r = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            in_tr_r, in_val_r = next(skf_in_r.split(X_tr_r_rem, y_tr_r_rem))
            
            X_in_tr_r, y_in_tr_r = X_tr_r_rem[in_tr_r], y_tr_r_rem[in_tr_r]
            X_in_val_r, y_in_val_r = X_tr_r_rem[in_val_r], y_tr_r_rem[in_val_r]
            
            counts_r = {i: int(np.sum(y_in_tr_r == i)) for i in range(4)}
            cw_r = {i: float(len(y_in_tr_r) / (4.0 * max(counts_r[i], 1))) for i in range(4)}
            sw_r = np.array([cw_r[int(lbl)] for lbl in y_in_tr_r], dtype=np.float32)
            
            model_r = build_head(seed=seed + fold * 10)
            es_r = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            hist_r = model_r.fit(X_in_tr_r, y_in_tr_r, sample_weight=sw_r, batch_size=BATCH_SIZE, epochs=80,
                                 validation_data=(X_in_val_r, y_in_val_r), callbacks=[es_r], verbose=0)
            
            preds_r = np.argmax(model_r.predict(X_val, verbose=0), axis=1)
            oof_preds_random[val_idx] = preds_r
            
            records.append({
                "arm": "Arm2_InnerValidation",
                "condition": "random_drop_control",
                "purge_fraction": p_frac,
                "seed": seed,
                "fold": fold,
                "train_count": len(X_in_tr_r),
                "val_count": len(X_val),
                "final_train_acc": float(hist_r.history["accuracy"][-1]),
                "val_accuracy": float(accuracy_score(y_val, preds_r)),
                "val_balanced_accuracy": float(balanced_accuracy_score(y_val, preds_r)),
                "val_macro_f1": float(f1_score(y_val, preds_r, average="macro", zero_division=0)),
                "class_counts_train": counts_r,
                "purged_class_counts": k_c
            })

        bal_p = balanced_accuracy_score(y_all, oof_preds_purged)
        bal_r = balanced_accuracy_score(y_all, oof_preds_random)
        diff = bal_p - bal_r
        print(f"  Level {int(p_frac*100):2d}%: Purged BalAcc={bal_p*100:5.2f}% | Random-Drop BalAcc={bal_r*100:5.2f}% | Delta = {diff*100:+5.2f}%")

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    print(f"  --> Checkpointed results after Seed {seed} ({time.time()-t_seed_start:.1f}s)")

# =========================================================================
# SUMMARY ANALYSIS & VALIDITY GATE CHECK
# =========================================================================
df_all = pd.DataFrame(records)
df_all.to_csv(CSV_PATH, index=False)

# Check validity gate on plain 0% baseline (Primary Arm)
base_records = df_all[(df_all["arm"] == "Arm1_FixedEpochs") & (df_all["condition"] == "0%_unpurged_baseline")]
base_bal_accs = base_records.groupby("seed")["val_balanced_accuracy"].mean()
mean_base_bal = float(base_bal_accs.mean())

validity_gate_passed = (mean_base_bal > 0.30) # chance is 0.25; within 5 points is <= 0.30
print("\n" + "="*80)
print(f"VALIDITY GATE AUDIT: Plain 0% Baseline Balanced Accuracy = {mean_base_bal*100:.2f}%")
print(f"Gate Condition: > 30.00% (5 points above chance 25.00%) -> Passed: {validity_gate_passed}")
print("="*80)

# Compute paired differences per fold and seed at 10%, 25%, 50%
summary_dict = {
    "validity_gate": {
        "threshold": 0.30,
        "mean_baseline_balanced_accuracy": mean_base_bal,
        "passed": bool(validity_gate_passed)
    },
    "deviations": DEVIATIONS,
    "primary_arm_fixed_epochs": {},
    "secondary_arm_inner_val": {}
}

for arm_name, arm_key in [("Arm1_FixedEpochs", "primary_arm_fixed_epochs"),
                          ("Arm2_InnerValidation", "secondary_arm_inner_val")]:
    arm_df = df_all[df_all["arm"] == arm_name]
    summary_dict[arm_key] = {}
    
    # Baseline stats
    b_df = arm_df[arm_df["condition"] == "0%_unpurged_baseline"]
    b_seeds = b_df.groupby("seed")["val_balanced_accuracy"].mean()
    summary_dict[arm_key]["0%_baseline"] = {
        "mean": float(b_seeds.mean()),
        "std": float(b_seeds.std()),
        "seeds": {int(s): float(v) for s, v in b_seeds.items()}
    }
    
    # Paired comparisons at 10%, 25%, 50%
    for p_frac in [0.10, 0.25, 0.50]:
        p_df = arm_df[(arm_df["condition"] == "similarity_purged") & (arm_df["purge_fraction"] == p_frac)]
        r_df = arm_df[(arm_df["condition"] == "random_drop_control") & (arm_df["purge_fraction"] == p_frac)]
        
        # Merge on seed and fold for exact paired difference
        paired = pd.merge(
            p_df[["seed", "fold", "val_balanced_accuracy"]].rename(columns={"val_balanced_accuracy": "bal_purged"}),
            r_df[["seed", "fold", "val_balanced_accuracy"]].rename(columns={"val_balanced_accuracy": "bal_random"}),
            on=["seed", "fold"]
        )
        paired["delta"] = paired["bal_purged"] - paired["bal_random"]
        
        # Seed-level averages
        seed_p = p_df.groupby("seed")["val_balanced_accuracy"].mean()
        seed_r = r_df.groupby("seed")["val_balanced_accuracy"].mean()
        seed_delta = seed_p - seed_r
        
        summary_dict[arm_key][f"{int(p_frac*100)}%_purge"] = {
            "purged_mean": float(seed_p.mean()),
            "purged_std": float(seed_p.std()),
            "random_mean": float(seed_r.mean()),
            "random_std": float(seed_r.std()),
            "paired_delta_mean": float(seed_delta.mean()),
            "paired_delta_std": float(seed_delta.std()),
            "per_fold_paired_deltas": paired.to_dict(orient="records")
        }

with open(JSON_PATH, "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"\nPhase 3 execution complete! Summary saved to {JSON_PATH}")
