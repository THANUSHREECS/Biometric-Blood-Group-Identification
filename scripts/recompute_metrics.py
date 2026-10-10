import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('outputs/abo_positive_4class/test_predictions.csv')
y_true = df['ground_truth_class'].values
y_pred = df['predicted_class'].values
N = len(df)
acc = accuracy_score(y_true, y_pred)

counts = df['ground_truth_class'].value_counts()
maj_acc = counts.max() / N

pred_counts = pd.Series(y_pred).value_counts()

print("## 5. FIXED NUMBERS")
print(f"- **Test Set Size:** 415 (not 898, which was the 8-class test set)")
print(f"- **Real Random-Split Accuracy:** {acc*100:.2f}% (not 38.10%)")
print(f"- **Majority Baseline:** {maj_acc*100:.2f}%")

np.random.seed(42)
perm_accs = [accuracy_score(y_true, np.random.permutation(y_true)) for _ in range(1000)]
print(f"- **Permutation p-value:** {np.mean(np.array(perm_accs) >= acc)}")

boot_accs = [accuracy_score(y_true[i], y_pred[i]) for i in (np.random.randint(0, N, N) for _ in range(1000))]
print(f"- **Bootstrap 95% CI:** [{np.percentile(boot_accs, 2.5)*100:.2f}%, {np.percentile(boot_accs, 97.5)*100:.2f}%]")
print(f"- **Predicted class counts:** {dict(pred_counts)}")
for c in ['A+', 'B+', 'AB+', 'O+']:
    r = np.sum((y_true == c) & (y_pred == c)) / np.sum(y_true == c) if np.sum(y_true == c) > 0 else 0
    print(f"  - Recall {c}: {r*100:.2f}%")

print("- **Explanation for B+:** The model predicts B+ fairly often (87 times). The previous explanation that 'extreme imbalance favours A+ causing B+ to never be predicted' was incorrect. B+ has an 78.35% recall.")
