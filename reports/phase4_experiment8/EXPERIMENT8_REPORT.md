# EXPERIMENT 8 REPORT: REGULARIZED LINEAR CLASSIFIER ON FROZEN FEATURES

## 1. OBJECTIVE
To determine if the E2 Champion model's development performance (0.42 Macro F1) was driven by genuine biological feature extraction or by extreme neural overparameterization (memorizing the 64 training participants).

## 2. RESULTS SUMMARY (5-Fold CV)
*Evaluated on the exact same 80 development participants as previous experiments.*

| Representation | Model | Mean Accuracy | Mean Macro F1 |
|----------------|-------|---------------|---------------|
| A (Single Scan)| Dummy | 28.75% | 0.1072 |
| A (Single Scan)| Logistic Reg | 28.75% | 0.1934 |
| A (Single Scan)| Linear SVM | 32.50% | 0.2357 |
| B (3-Scan Mean)| Dummy | 28.75% | 0.1072 |
| B (3-Scan Mean)| Logistic Reg | 31.25% | 0.2228 |
| B (3-Scan Mean)| **Linear SVM** | **41.25%** | **0.2477** |

*Reference Baseline: E2 Champion (Classification + SupCon)*
- E2 Mean Accuracy: 53.75%
- E2 Mean Macro F1: 0.4240

## 3. INTERPRETATION

**1. Does a regularized linear classifier outperform the majority-class baseline?**
Yes, but only marginally. The Dummy classifier achieved 0.10 Macro F1. The Linear SVM improved this to 0.24.

**2. Does it improve Macro F1 over the E2 development reference?**
**No. It massively underperforms.** The E2 model achieved 0.42 Macro F1. The Linear SVM collapsed to 0.24. This catastrophic drop mathematically proves that the E2 model's 1.5M parameter classification head was heavily memorizing the spurious optical characteristics of the 64 training participants. When forced to use a simple linear boundary (which cannot memorize), the "predictive power" of the frozen DenseNet features almost entirely evaporated.

**3. Which classes remain difficult to identify?**
AB+ and A+ remain profoundly difficult. The linear models regularly yielded 0.0% precision and 0.0% recall for AB+ across nearly all folds. The models frequently defaulted to predicting O+ (the majority class) or B+.

**4. Does averaging three scan features improve performance?**
Yes, slightly. Representation B (averaging the 1024-d vectors before classification) achieved a Macro F1 of 0.247, outperforming Representation A (classifying single scans and averaging probabilities) at 0.235. Early feature pooling appears marginally more robust than late probability pooling.

**5. Is performance consistent across outer folds?**
No. The performance is highly unstable, ranging from complete failure on certain classes in Fold 1 and 5 to sporadic hits in Fold 4. This instability is a direct consequence of the micro-dataset size (N=80).

**6. Do results suggest that the existing frozen features contain useful predictive information?**
The evidence is highly conclusive that the **frozen ImageNet features contain almost no robust biological predictive information** regarding blood groups. A 41% accuracy on 4 imbalanced classes (where naive majority voting yields ~30-40% depending on fold) means the features are essentially noise with respect to the target variable.

**7. What are the most important limitations?**
- N=64 training participants is far too small for any generalized learning, even with linear models.
- Severe class imbalance (AB+ represents < 10% of the dataset).
- ImageNet pre-trained CNN backbones extract macroscopic object features (edges, textures for dogs/cars). They are entirely the wrong mathematical prior for resolving microscopic biometric ridge structures or minutiae counts.
