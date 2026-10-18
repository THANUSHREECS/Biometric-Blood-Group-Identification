import pandas as pd
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

scans = pd.read_csv('experiments/exp10_multi_scan_quality/exp10_scan_quality_audit.csv')
print('Total rows in scan_quality_audit:', len(scans))
print('Unique participants:', scans['participant_id'].nunique())

p_stats = []
for pid, g in scans.groupby('participant_id'):
    assert len(g) == 3
    qs = g['quality_score'].values
    q_min = np.min(qs)
    q_max = np.max(qs)
    ratio = q_max / q_min
    p_stats.append({'participant_id': pid, 'q_min': q_min, 'q_max': q_max, 'ratio': ratio})

df_p = pd.DataFrame(p_stats)
ge_15 = int(np.sum(df_p['ratio'] >= 1.5))
lt_15 = int(np.sum(df_p['ratio'] < 1.5))
ge_20 = int(np.sum(df_p['ratio'] >= 2.0))
print(f'Actual counts: ratio >= 1.5: {ge_15} | ratio < 1.5: {lt_15} | ratio >= 2.0: {ge_20}')
print('Participants with ratio >= 1.5:')
print(df_p[df_p['ratio'] >= 1.5].to_string())

preds = pd.read_csv('experiments/exp10_multi_scan_quality/exp10_participant_oof_predictions.csv')
y = preds['target'].values
ridge_a = preds['pred_ridge_classifier_condition_A_unweighted'].values
ridge_b = preds['pred_ridge_classifier_condition_B_quality_weighted'].values

identical = int(np.sum(ridge_a == ridge_b))
changed = int(np.sum(ridge_a != ridge_b))
corr_a = int(np.sum(ridge_a == y))
corr_b = int(np.sum(ridge_b == y))
corr_only_a = int(np.sum((ridge_a == y) & (ridge_b != y)))
corr_only_b = int(np.sum((ridge_a != y) & (ridge_b == y)))

print(f'\nRidge: identical = {identical} | changed = {changed} | corr_a = {corr_a} | corr_b = {corr_b} | corr_only_a = {corr_only_a} | corr_only_b = {corr_only_b}')
for i in np.where(ridge_a != ridge_b)[0]:
    row = preds.iloc[i]
    bg = row['blood_group']
    print(f"Participant: {row['participant_id']} | True: {y[i]} ({bg}) | Pred A: {ridge_a[i]} | Pred B: {ridge_b[i]}")

# Confusion matrices
cm_a = confusion_matrix(y, ridge_a, labels=[0, 1, 2, 3])
cm_b = confusion_matrix(y, ridge_b, labels=[0, 1, 2, 3])
print('Confusion Matrix Ridge A:\n', cm_a)
print('Confusion Matrix Ridge B:\n', cm_b)
print('Diff in CM (B - A):\n', cm_b - cm_a)

# Random Forest checks
rf_a = preds['pred_random_forest_condition_A_unweighted'].values
rf_b = preds['pred_random_forest_condition_B_quality_weighted'].values

rf_identical = int(np.sum(rf_a == rf_b))
rf_changed = int(np.sum(rf_a != rf_b))
rf_corr_a = int(np.sum(rf_a == y))
rf_corr_b = int(np.sum(rf_b == y))
rf_corr_only_a = int(np.sum((rf_a == y) & (rf_b != y)))
rf_corr_only_b = int(np.sum((rf_a != y) & (rf_b == y)))
print(f'\nRF: identical = {rf_identical} | changed = {rf_changed} | corr_a = {rf_corr_a} | corr_b = {rf_corr_b} | corr_only_a = {rf_corr_only_a} | corr_only_b = {rf_corr_only_b}')

cm_rf_a = confusion_matrix(y, rf_a, labels=[0, 1, 2, 3])
cm_rf_b = confusion_matrix(y, rf_b, labels=[0, 1, 2, 3])
print('Confusion Matrix RF A:\n', cm_rf_a)
print('Confusion Matrix RF B:\n', cm_rf_b)
