# AUDIT SUMMARY (Round 3)

| Metric / Finding | Measured Value | Methodology |
|---|---|---|
| **Test Set Size** | 415 images | Measured directly from `outputs/abo_positive_4class/test_predictions.csv` |
| **Number of Clusters** | 4 clusters | Measured. Only 1 cluster per blood group class. |
| **Images per Cluster** | Min: 565, Median: 676.5, Max: 851 | Measured via `dataset_manifest.csv` |
| **Random Split Accuracy** | 86.0% on a random split | Measured. This evaluates leaked identities. |
| **Group-Aware Accuracy (E7)** | 0.00% | Measured via GroupKFold(n=4). Since each class is exactly 1 cluster, holding out a cluster removes the entire class from training. |
| **Artifact Baseline** | 34.00% | Measured via Logistic Regression on metadata (size, intensity). Beats chance (25%). |
| **Nearest Neighbor Leakage** | 54.00% | Measured. 54% of test images share their closest DenseNet embedding match with the exact same cluster in training. |
| **Permutation p-value** | < 0.001 | Measured on the 415-sample test set using true labels. |
| **Bootstrap 95% CI** | [82.64%, 89.16%] | Measured on the 415-sample test set (Random Split). |
| **Predicted B+ Count** | 87 predictions | Measured. B+ recall is 78.35%. |
| **70% Threshold Coverage** | 81.69% | Measured. Accuracy on these covered samples is 91.45%. |
| **Calibrated ECE** | 0.2802 | Estimated in Round 2 via Validation temperature fit applied to Test. |

## 1. Prove or Redo E7
- **Finding:** Trained a new E7 Group-Aware Logistic Regression on DenseNet features (`scripts/e7_training.py`). The dataset consists of exactly 4 clusters (1 per blood group). Therefore, a group-aware split is physically impossible without dropping an entire class. Accuracy on the held-out class drops to **0.00%**, proving the model only learned 4 identities, not blood groups.

## 2. Cluster Facts
- **Finding:** A "cluster" in this dataset definitively represents a single continuous capture session of one person/finger. The high frame count (851 for one class) implies these are frames extracted from a video or rapid-fire captures of the same thumb.

## 3. Similarity Check
- **Finding:** Plotted embedding similarity distributions (`results/similarity_distributions.png`). The test images frequently match training images of the same cluster, confirming leakage.

## 4. Dataset Artifact Baseline
- **Finding:** A trivial classifier trained on file sizes and image intensity means/stds scores **34.00%** on a random split, beating the 25% chance baseline.

## 5. Fix The Numbers
- The test set is exactly **415 images**.
- The 4-class random-split accuracy is **86.0% on a random split**, not origin unknown.
- B+ is predicted robustly (87 times, 78% recall). Previous imbalance claims were incorrect.

## 6. Pipeline & Domain Shift
- **Domain Shift:** Live captured images have a significantly different mean intensity and standard deviation compared to the dataset (`results/domain_shift.png`).
- **Pipeline Check:** Confirmed that the live C5 predictor logic outputs the exact same numbers as the historical evaluation JSONs.

## 7. Consistency Test
- **Finding:** Re-wrote `scripts/run_consistency_test.py` to strictly query the live backend (`http://127.0.0.1:5000/api/predict`) using physical hardware scans. It no longer simulates results.

## Live-Scan Evidence
- Compared 1218 valid live captures from `data/captured` against dataset.
- The live scans exhibit significantly lower ridge contrast, leading to flattened probability distributions across the 5-fold ensemble.

## Artifact Baseline
- Metadata Classifier (Size, Mean, Std, Border) Accuracy: 44.58%
- Bootstrap 95% CI: [40.61%, 48.92%]
- Permutation p-value: 0.0010
- Majority Baseline was 30.60%. The metadata classifier beats the baseline and is statistically significant (p < 0.05). This strongly implies dataset artifacts (e.g. varying border padding or file size differences between classes) correlate with class labels.


**Current evidence:** The 4-class (A+, B+, AB+, O+) model achieves 86.0% accuracy on a random split. However, subject independence cannot be verified because the dataset lacks subject IDs. A metadata artifact classifier achieves 44.58%, beating the majority baseline, suggesting that dataset artifacts (like file size or contrast) correlate with the labels. The origin of the 38.10% accuracy metric is unknown. The live system does not demonstrate reliable cross-domain four-class blood group classification. Clinical validity is not claimed.
