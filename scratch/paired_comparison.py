import pandas as pd
from scipy.stats import binomtest

p3 = pd.read_csv('reports/phase3_final_evaluation/final_test_predictions.csv')
exp06c = pd.read_csv('reports/final_evaluation/primary_evaluation_predictions.csv')

merged = pd.merge(p3, exp06c, on='participant_id')

# Compare correctness
merged['p3_correct'] = merged['actual_class'] == merged['predicted_class']
merged['exp06c_correct'] = merged['true_label'] == merged['predicted_label']

print(f"Total participants: {len(merged)}")
print(f"Phase 3 Model B Correct: {merged['p3_correct'].sum()} / {len(merged)} ({merged['p3_correct'].mean():.4f})")
print(f"EXP-06C Config A Correct: {merged['exp06c_correct'].sum()} / {len(merged)} ({merged['exp06c_correct'].mean():.4f})")

# Contingency table
b_correct = int((merged['p3_correct'] & merged['exp06c_correct']).sum())
p3_only = int((merged['p3_correct'] & ~merged['exp06c_correct']).sum())
exp_only = int((~merged['p3_correct'] & merged['exp06c_correct']).sum())
both_wrong = int((~merged['p3_correct'] & ~merged['exp06c_correct']).sum())

print(f"Both correct: {b_correct}")
print(f"Phase 3 only correct: {p3_only}")
print(f"EXP-06C only correct: {exp_only}")
print(f"Both incorrect: {both_wrong}")

# Discordant pairs:
n_discordant = p3_only + exp_only
print(f"Discordant pairs (n): {n_discordant}")

if n_discordant > 0:
    res = binomtest(exp_only, n_discordant, p=0.5, alternative='two-sided')
    print(f"Binomial test p-value: {res.pvalue:.4f}")
else:
    print("No discordant pairs.")

print("\n--- Detailed Pairwise Participant Comparison ---")
for idx, r in merged.iterrows():
    p = r['participant_id']
    y = r['actual_class']
    p3_p = r['predicted_class']
    exp_p = r['predicted_label']
    print(f"{p}: True={y} | P3={p3_p} (corr={r['p3_correct']}) | EXP06C={exp_p} (corr={r['exp06c_correct']})")
