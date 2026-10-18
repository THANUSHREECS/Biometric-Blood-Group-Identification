import os, glob, json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

exps = ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06a', 'exp06b', 'exp06c', 'exp06d']

canonical_classes = ['A+', 'AB+', 'B+', 'O+']

for exp in exps:
    exp_dir = os.path.join('reports', 'accuracy_improvement', exp)
    print(f"\n==========================================")
    print(f"EXPERIMENT: {exp.upper()}")
    print(f"==========================================")
    
    # Check prediction files
    csv_files = glob.glob(os.path.join(exp_dir, "*oof*.csv"))
    print(f"OOF CSV Files found ({len(csv_files)}): {[os.path.basename(f) for f in csv_files]}")
    
    for cf in sorted(csv_files):
        arm_name = os.path.basename(cf).replace("_oof_predictions.csv", "")
        df = pd.read_csv(cf)
        print(f"\n--- Arm: {arm_name} ({len(df)} rows) ---")
        cols = list(df.columns)
        print(f"Columns: {cols}")
        
        # Determine ID col, true col, pred col
        id_col = 'participant_id' if 'participant_id' in cols else ('subject_id' if 'subject_id' in cols else None)
        true_col = 'true_label' if 'true_label' in cols else ('true_blood_group' if 'true_blood_group' in cols else None)
        pred_col = 'predicted_label' if 'predicted_label' in cols else ('pred_label' if 'pred_label' in cols else None)
        fold_col = 'fold' if 'fold' in cols else None
        
        unique_ids = df[id_col].nunique() if id_col else len(df)
        print(f"Unique participants/IDs: {unique_ids}")
        
        if true_col and pred_col:
            # Check unique labels
            unique_true = sorted(list(df[true_col].unique()))
            unique_pred = sorted(list(df[pred_col].unique()))
            print(f"Unique true labels: {unique_true}")
            print(f"Unique pred labels: {unique_pred}")
            
            # Recompute metrics
            acc = accuracy_score(df[true_col], df[pred_col])
            bacc = balanced_accuracy_score(df[true_col], df[pred_col])
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(df[true_col], df[pred_col], average='macro', zero_division=0)
            
            print(f"Recomputed Pooled Accuracy: {acc*100:.2f}% ({sum(df[true_col] == df[pred_col])}/{len(df)})")
            print(f"Recomputed Balanced Accuracy: {bacc*100:.2f}%")
            print(f"Recomputed Pooled Macro F1: {f1_macro:.4f}")
            print(f"Recomputed Macro Precision: {p_macro:.4f}, Macro Recall: {r_macro:.4f}")
            
            # Per class recall
            p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(df[true_col], df[pred_col], labels=canonical_classes, zero_division=0)
            for c, p, r, f, s in zip(canonical_classes, p_cls, r_cls, f1_cls, supp_cls):
                print(f"  Class {c:4s}: Support={s:2d}, Prec={p*100:5.2f}%, Rec={r*100:5.2f}%, F1={f:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(df[true_col], df[pred_col], labels=canonical_classes)
            print(f"Confusion Matrix (labels={canonical_classes}):")
            print(cm)
            
            # Fold metrics if fold col present
            if fold_col:
                fold_f1s = []
                for f_idx in sorted(df[fold_col].unique()):
                    f_df = df[df[fold_col] == f_idx]
                    _, _, f1_f, _ = precision_recall_fscore_support(f_df[true_col], f_df[pred_col], average='macro', zero_division=0)
                    fold_f1s.append(f1_f)
                print(f"Fold F1s: {[round(x, 4) for x in fold_f1s]}")
                print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s, ddof=1) if len(fold_f1s)>1 else 0:.4f}")
