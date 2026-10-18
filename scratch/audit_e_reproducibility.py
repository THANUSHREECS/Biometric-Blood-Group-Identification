import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

print("=== AUDIT E: REPRODUCIBILITY OF EXP-06C CONFIG A DEVELOPMENT RESULTS ===")
pred_path = "reports/accuracy_improvement/exp06c/config_a_oof_predictions.csv"
df = pd.read_csv(pred_path)

print(f"Total rows in config_a_oof_predictions.csv: {len(df)}")
print(f"Unique participant IDs: {df['participant_id'].nunique()}")
print(f"Columns: {list(df.columns)}")

# Check for duplicates or missing values
print(f"Missing values: {df.isnull().sum().to_dict()}")
dups = df[df.duplicated(subset=['participant_id'], keep=False)]
print(f"Duplicate participant records: {len(dups)}")

# Compute metrics
y_true = df['true_label'].values
y_pred = df['predicted_label'].values
classes = ["A+", "AB+", "B+", "O+"]

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
macro_p = precision_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
macro_r = recall_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)

print(f"\n--- Recomputed Development Metrics ---")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%) [Reported: 53.75%]")
print(f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%) [Reported: 53.47%]")
print(f"Macro F1: {macro_f1:.4f} [Reported: 0.5487]")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall: {macro_r:.4f}")

print("\n--- Per-Class Performance ---")
p_per_class = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
r_per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

cm = confusion_matrix(y_true, y_pred, labels=classes)
support = cm.sum(axis=1)

for i, c in enumerate(classes):
    print(f"Class {c:4}: Support={support[i]:2}, Recall={r_per_class[i]:.4f} ({r_per_class[i]*100:.2f}%), Precision={p_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

print("\nConfusion Matrix (Rows=True, Cols=Pred [A+, AB+, B+, O+]):")
print(cm)

# Check per-fold metrics if fold_id column exists
if 'fold_id' in df.columns:
    print("\n--- Per-Fold Metrics ---")
    fold_f1s = []
    for f in sorted(df['fold_id'].unique()):
        df_f = df[df['fold_id'] == f]
        f_f1 = f1_score(df_f['true_label'], df_f['predicted_label'], labels=classes, average='macro', zero_division=0)
        f_acc = accuracy_score(df_f['true_label'], df_f['predicted_label'])
        fold_f1s.append(f_f1)
        print(f"Fold {f}: N={len(df_f)}, Macro F1={f_f1:.4f}, Accuracy={f_acc:.4f}")
    print(f"Mean Fold F1: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s, ddof=1):.4f}) [Reported: 0.5331 +/- 0.0895]")
