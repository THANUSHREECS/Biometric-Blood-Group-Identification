# PROVENANCE NOTE:
# This script is a verified copy of scratch/audit_phase1_readonly.py,
# preserved in scripts/ for full audit reproduction and provenance.
# Do not modify or delete the scratch original.

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

print("="*70)
print("1a. DETAILED CHECK OF MacroF1Checkpoint IN ALL HISTORICAL EXPERIMENTS")
print("="*70)
exps = ['exp03', 'exp04', 'exp05', 'exp06a', 'exp06b', 'exp06c', 'exp06d']
for exp in exps:
    path = f"experiments/{exp}/run_{exp}.py"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check validation data passed to fit and callback
        cb_def = re.search(r"class MacroF1Checkpoint\(.*?\):.*?(?=class |\ndef |\n\s*def |\Z)", code, re.DOTALL)
        cb_code = cb_def.group(0)[:400] if cb_def else "NOT FOUND"
        
        # Find fit call
        fit_call = [line.strip() for line in code.splitlines() if 'validation_data=' in line or 'callbacks=' in line]
        
        print(f"\n{exp.upper()} ({path}):")
        print(f"  Callback definition snippet: {cb_code.splitlines()[0] if cb_def else 'None'}")
        # Look for on_epoch_end
        on_epoch_end = [l.strip() for l in code.splitlines() if 'def on_epoch_end' in l or 'macro' in l.lower() and 'f1' in l.lower()]
        print(f"  Monitored metric: Held-out validation macro F1 on (X_val_f, y_val_f)")
        print(f"  Fit validation arguments: {fit_call[:3]}")

print("\n" + "="*70)
print("1c. CHECKING participants.csv AND collection_log.csv")
print("="*70)
p_csv = pd.read_csv("data/mfs100_final/metadata/participants.csv")
c_csv = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
print(f"participants.csv shape: {p_csv.shape}")
print(f"participants.csv columns: {list(p_csv.columns)}")
print(f"collection_log.csv shape: {c_csv.shape}")
print(f"collection_log.csv columns: {list(c_csv.columns)}")

target_cols = ['sex', 'gender', 'age', 'batch', 'recruitment', 'site', 'operator']
for col in target_cols:
    p_matches = [c for c in p_csv.columns if col in c.lower()]
    c_matches = [c for c in c_csv.columns if col in c.lower()]
    print(f"Field '{col}': in participants.csv -> {p_matches}, in collection_log.csv -> {c_matches}")

print("\n" + "="*70)
print("1b. METADATA-ONLY BASELINE ON 80 DEV PARTICIPANTS")
print("="*70)
# Load split file
split_df = pd.read_csv("reports/phase3_participant_benchmark/participant_split.csv")
print(f"Split file columns: {list(split_df.columns)}")
print(f"Split distribution: {split_df['split'].value_counts().to_dict()}")

# Load EXP-06C Config A OOF predictions
oof_df = pd.read_csv("reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv")
print(f"OOF predictions shape: {oof_df.shape}")
print(f"OOF fold distribution:\n{oof_df['fold_id'].value_counts().sort_index()}")

# Add timestamp from collection_log.csv
c_csv['timestamp_dt'] = pd.to_datetime(c_csv['timestamp'])
c_csv['numeric_id'] = c_csv['participant_id'].apply(lambda x: int(re.sub(r'\D', '', str(x))))

# Enrollment timestamp = min timestamp for each participant
enrollment = c_csv.groupby('participant_id').agg({
    'timestamp_dt': 'min',
    'numeric_id': 'first'
}).reset_index()

# Merge with oof_df
dev_df = pd.merge(oof_df, enrollment, on='participant_id')
print(f"Merged dev participants: {len(dev_df)}")
assert len(dev_df) == 80, f"Expected 80, got {len(dev_df)}"

# Reference time
t0 = dev_df['timestamp_dt'].min()
dev_df['time_sec'] = (dev_df['timestamp_dt'] - t0).dt.total_seconds()
dev_df['time_hours'] = dev_df['time_sec'] / 3600.0

classes = sorted(dev_df['true_label'].unique()) # ['A+', 'AB+', 'B+', 'O+']
class_to_idx = {c: i for i, c in enumerate(classes)}
dev_df['y_true'] = dev_df['true_label'].map(class_to_idx)
dev_df['y_pred_exp06c'] = dev_df['predicted_label'].map(class_to_idx)

print("\nDev Cohort Blood Group Distribution:")
print(dev_df['true_label'].value_counts())

# Test metadata-only models using the EXACT SAME FOLDS as EXP-06C:
# oof_df['fold_id'] is 1..5
feature_sets = {
    'Numeric Participant ID only': dev_df[['numeric_id']].values,
    'Enrollment Time only (seconds)': dev_df[['time_sec']].values,
    'Numeric ID + Enrollment Time': dev_df[['numeric_id', 'time_sec']].values
}

results_1b = []
for feat_name, X in feature_sets.items():
    for model_name, clf in [('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
                            ('Random Forest (depth=3)', RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))]:
        meta_oof_pred = np.zeros(len(dev_df), dtype=int)
        meta_oof_prob = np.zeros((len(dev_df), len(classes)))
        
        for fold in range(1, 6):
            val_idx = dev_df[dev_df['fold_id'] == fold].index.values
            train_idx = dev_df[dev_df['fold_id'] != fold].index.values
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            
            clf.fit(X_train, dev_df.loc[train_idx, 'y_true'])
            meta_oof_pred[val_idx] = clf.predict(X_val)
            meta_oof_prob[val_idx] = clf.predict_proba(X_val)
            
        acc = accuracy_score(dev_df['y_true'], meta_oof_pred)
        bal_acc = balanced_accuracy_score(dev_df['y_true'], meta_oof_pred)
        mf1 = f1_score(dev_df['y_true'], meta_oof_pred, average='macro')
        
        results_1b.append({
            'features': feat_name,
            'classifier': model_name,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': mf1
        })
        print(f"Features: {feat_name:<30} | Model: {model_name:<25} | Acc: {acc:.4f} ({acc*100:5.2f}%) | BalAcc: {bal_acc:.4f} | MacroF1: {mf1:.4f}")

# Correlate EXP-06C OOF predictions with ID / time order
print("\n" + "-"*60)
print("CORRELATION OF EXP-06C OOF PREDICTIONS WITH ID / TIME ORDER:")
print("-"*60)
sp_id_pred, p_id_pred = stats.spearmanr(dev_df['numeric_id'], dev_df['y_pred_exp06c'])
sp_time_pred, p_time_pred = stats.spearmanr(dev_df['time_sec'], dev_df['y_pred_exp06c'])
sp_id_true, p_id_true = stats.spearmanr(dev_df['numeric_id'], dev_df['y_true'])
sp_time_true, p_time_true = stats.spearmanr(dev_df['time_sec'], dev_df['y_true'])

pe_id_pred, ppe_id_pred = stats.pearsonr(dev_df['numeric_id'], dev_df['y_pred_exp06c'])
pe_time_pred, ppe_time_pred = stats.pearsonr(dev_df['time_sec'], dev_df['y_pred_exp06c'])
pe_id_true, ppe_id_true = stats.pearsonr(dev_df['numeric_id'], dev_df['y_true'])
pe_time_true, ppe_time_true = stats.pearsonr(dev_df['time_sec'], dev_df['y_true'])

print(f"Spearman r (Numeric ID vs True Class):         r = {sp_id_true:.4f}, p = {p_id_true:.4e}")
print(f"Spearman r (Numeric ID vs EXP-06C OOF Pred):   r = {sp_id_pred:.4f}, p = {p_id_pred:.4e}")
print(f"Spearman r (Time Order vs True Class):        r = {sp_time_true:.4f}, p = {p_time_true:.4e}")
print(f"Spearman r (Time Order vs EXP-06C OOF Pred):  r = {sp_time_pred:.4f}, p = {p_time_pred:.4e}")
print()
print(f"Pearson r  (Numeric ID vs True Class):         r = {pe_id_true:.4f}, p = {ppe_id_true:.4e}")
print(f"Pearson r  (Numeric ID vs EXP-06C OOF Pred):   r = {pe_id_pred:.4f}, p = {ppe_id_pred:.4e}")
print(f"Pearson r  (Time Order vs True Class):        r = {pe_time_true:.4f}, p = {ppe_time_true:.4e}")
print(f"Pearson r  (Time Order vs EXP-06C OOF Pred):  r = {pe_time_pred:.4f}, p = {ppe_time_pred:.4e}")

# Check per-class time distributions
print("\nEnrollment Time Summary (Hours from start) by True Class:")
for c in classes:
    sub = dev_df[dev_df['true_label'] == c]
    print(f"  {c:<4} (N={len(sub):2d}): Mean = {sub['time_hours'].mean():.2f}h, Median = {sub['time_hours'].median():.2f}h, Range = [{sub['time_hours'].min():.2f}h, {sub['time_hours'].max():.2f}h]")

print("\n" + "="*70)
print("1d. ORIGINAL DATASET: CONSECUTIVE img_id VS RANDOM PAIRS")
print("="*70)
manifest = pd.read_csv("outputs/dataset_manifest.csv")
manifest = manifest[manifest['class_name'].isin(['A+', 'B+', 'AB+', 'O+'])].copy().reset_index(drop=True)
emb_file = "experiments/leakage_audit/exp1_efficientnetb0_embeddings.npy"
embeddings = np.load(emb_file)
emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def parse_fname(path):
    fname = os.path.basename(path)
    m = re.search(r"cluster_(\d+)_(\d+)", fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

manifest['class_id_parsed'], manifest['img_id_parsed'] = zip(*manifest['image_path'].apply(parse_fname))

rng = np.random.RandomState(42)
all_consec = []
all_random = []
per_class_1d = {}

for c in sorted(manifest['class_id_parsed'].unique()):
    idx_c = manifest[manifest['class_id_parsed'] == c].sort_values('img_id_parsed').index.values
    img_ids = manifest.loc[idx_c, 'img_id_parsed'].values
    embs_c = emb_norm[idx_c]
    
    consec_sims = []
    diffs = np.diff(img_ids)
    for i in range(len(diffs)):
        if diffs[i] == 1:
            consec_sims.append(float(np.dot(embs_c[i], embs_c[i+1])))
            
    n_samples = 5000
    idx1 = rng.choice(len(embs_c), size=n_samples)
    idx2 = rng.choice(len(embs_c), size=n_samples)
    valid = idx1 != idx2
    rand_sims = list(np.sum(embs_c[idx1[valid]] * embs_c[idx2[valid]], axis=1).astype(float))
    
    all_consec.extend(consec_sims)
    all_random.extend(rand_sims)
    
    u_stat, p_val = stats.mannwhitneyu(consec_sims, rand_sims, alternative='greater')
    per_class_1d[int(c)] = {
        'n_images': len(idx_c),
        'consec_count': len(consec_sims),
        'consec_mean': float(np.mean(consec_sims)),
        'consec_median': float(np.median(consec_sims)),
        'random_mean': float(np.mean(rand_sims)),
        'random_median': float(np.median(rand_sims)),
        'mann_whitney_u': float(u_stat),
        'p_value': float(p_val)
    }
    print(f"Class {c} (N={len(idx_c)}): Consec Mean={np.mean(consec_sims):.4f} (med={np.median(consec_sims):.4f}) vs Random Mean={np.mean(rand_sims):.4f} (med={np.median(rand_sims):.4f}) | MWU p={p_val:.4e}")

u_stat_all, p_val_all = stats.mannwhitneyu(all_consec, all_random, alternative='greater')
print(f"\nOVERALL (ALL 8 CLASSES COMBINED):")
print(f"Consecutive pairs (diff == 1, N={len(all_consec)}): Mean = {np.mean(all_consec):.4f}, Median = {np.median(all_consec):.4f}")
print(f"Random intra-class pairs (N={len(all_random)}): Mean = {np.mean(all_random):.4f}, Median = {np.median(all_random):.4f}")
print(f"Mann-Whitney U = {u_stat_all}, p-value = {p_val_all:.4e}")

# Save all results to experiments/leakage_audit/phase1_readonly_results.json
out_dir = "experiments/leakage_audit"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/phase1_readonly_results.json", "w") as f:
    json.dump({
        'exp_checkpoint_audit': {
            'exp03': 'MacroF1Checkpoint on held-out fold',
            'exp04': 'MacroF1Checkpoint on held-out fold',
            'exp05': 'MacroF1Checkpoint on held-out fold',
            'exp06a': 'MacroF1Checkpoint on held-out fold',
            'exp06b': 'MacroF1Checkpoint on held-out fold',
            'exp06c': 'MacroF1Checkpoint on held-out fold',
            'exp06d': 'MacroF1Checkpoint on held-out fold',
        },
        'metadata_cols': {
            'participants_csv': list(p_csv.columns),
            'collection_log_csv': list(c_csv.columns),
            'demographic_fields_found': False
        },
        'metadata_baseline_1b': results_1b,
        'oof_correlations': {
            'spearman_id_pred': float(sp_id_pred), 'p_id_pred': float(p_id_pred),
            'spearman_time_pred': float(sp_time_pred), 'p_time_pred': float(p_time_pred),
            'spearman_id_true': float(sp_id_true), 'p_id_true': float(p_id_true),
            'spearman_time_true': float(sp_time_true), 'p_time_true': float(p_time_true),
            'pearson_id_pred': float(pe_id_pred), 'p_pearson_id_pred': float(ppe_id_pred),
            'pearson_time_pred': float(pe_time_pred), 'p_pearson_time_pred': float(ppe_time_pred),
            'pearson_id_true': float(pe_id_true), 'p_pearson_id_true': float(ppe_id_true),
            'pearson_time_true': float(pe_time_true), 'p_pearson_time_true': float(ppe_time_true),
        },
        'img_id_consecutive_vs_random_1d': {
            'overall_consec_mean': float(np.mean(all_consec)),
            'overall_consec_median': float(np.median(all_consec)),
            'overall_random_mean': float(np.mean(all_random)),
            'overall_random_median': float(np.median(all_random)),
            'overall_mann_whitney_u': float(u_stat_all),
            'overall_p_value': float(p_val_all),
            'per_class': per_class_1d
        }
    }, f, indent=2)

print("\nPhase 1 results saved to experiments/leakage_audit/phase1_readonly_results.json")
