import os, glob, json
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05']
canonical_classes = ['A+', 'AB+', 'B+', 'O+']

for exp in exps:
    exp_dir = os.path.join('reports', 'accuracy_improvement', exp)
    print(f"\n==========================================")
    print(f"EXPERIMENT: {exp.upper()}")
    print(f"==========================================")
    
    # Check reports/metrics json
    metrics_json = os.path.join(exp_dir, 'metrics.json')
    if os.path.exists(metrics_json):
        with open(metrics_json, 'r') as f:
            data = json.load(f)
            print(f"metrics.json keys: {list(data.keys())}")
            # print snippet
            print(json.dumps(data, indent=2)[:500])
            
    metrics_csv = os.path.join(exp_dir, 'metrics_comparison.csv')
    if os.path.exists(metrics_csv):
        print(f"metrics_comparison.csv:")
        print(pd.read_csv(metrics_csv).to_string())

    csv_files = glob.glob(os.path.join(exp_dir, "*oof*.csv"))
    print(f"OOF CSV Files found ({len(csv_files)}): {[os.path.basename(f) for f in csv_files]}")
    for cf in sorted(csv_files):
        df = pd.read_csv(cf)
        cols = list(df.columns)
        id_col = 'participant_id' if 'participant_id' in cols else ('subject_id' if 'subject_id' in cols else None)
        true_col = 'true_label' if 'true_label' in cols else ('true_blood_group' if 'true_blood_group' in cols else None)
        pred_col = 'predicted_label' if 'predicted_label' in cols else ('pred_label' if 'pred_label' in cols else None)
        fold_col = 'fold' if 'fold' in cols else ('fold_id' if 'fold_id' in cols else None)
        
        n_rows = len(df)
        n_ids = df[id_col].nunique() if id_col else n_rows
        
        if true_col and pred_col:
            acc = accuracy_score(df[true_col], df[pred_col])
            bacc = balanced_accuracy_score(df[true_col], df[pred_col])
            p, r, f1, _ = precision_recall_fscore_support(df[true_col], df[pred_col], average='macro', zero_division=0)
            print(f"\nFile: {os.path.basename(cf)}")
            print(f"  Rows: {n_rows}, Unique IDs ({id_col}): {n_ids}, Folds: {df[fold_col].unique() if fold_col else 'None'}")
            print(f"  Accuracy: {acc*100:.2f}%, Balanced Acc: {bacc*100:.2f}%, Macro F1: {f1:.4f}")
            p_cls, r_cls, f1_cls, s_cls = precision_recall_fscore_support(df[true_col], df[pred_col], labels=canonical_classes, zero_division=0)
            for c, pc, rc, fc, sc in zip(canonical_classes, p_cls, r_cls, f1_cls, s_cls):
                print(f"    Class {c:4s}: Supp={sc:2d}, Rec={rc*100:5.2f}%, Prec={pc*100:5.2f}%, F1={fc:.4f}")
            cm = confusion_matrix(df[true_col], df[pred_col], labels=canonical_classes)
            print(f"  Confusion Matrix:\n{cm}")
            if fold_col:
                fold_f1s = []
                for f_idx in sorted(df[fold_col].unique()):
                    f_df = df[df[fold_col] == f_idx]
                    _, _, f1_f, _ = precision_recall_fscore_support(f_df[true_col], f_df[pred_col], average='macro', zero_division=0)
                    fold_f1s.append(f1_f)
                import numpy as np
                print(f"  Fold F1s: {[round(x, 4) for x in fold_f1s]}")
                print(f"  Fold F1 Mean: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s, ddof=1) if len(fold_f1s)>1 else 0:.4f}")
