# EXPERIMENT 10 REPORT: CONTROLLED AUGMENTATION AND REGULARIZATION

## 1. OBJECTIVE
To determine whether regularizing the classification head and applying scientifically justified mild augmentations (rotations, brightness) improves generalization, or if destroying optical artifacts prevents the CNN from "learning" the target.

## 2. RESULTS (5-Fold CV on 80 Development Participants)
*Frozen DenseNet121 Backbone, Participant-Level Grouping*

| Model Condition | Mean Accuracy | Mean Macro F1 |
|-----------------|---------------|---------------|
| A. Existing Reference (`Dense 512`) | 31.25% | 0.2372 |
| B. Smaller Head (`GlobalPool` -> `Dense 4`) | 25.00% | 0.2001 |
| C. Regularized Head (`Dense 64` + L2) | 27.50% | 0.2402 |
| D. Augmented & Regularized (Mild Aug) | **21.25%** | **0.1604** |

## 3. SCIENTIFIC INTERPRETATION

**1. Does the smaller head improve generalization?**
No. Removing the `Dense(512)` hidden layer and connecting the global average pool directly to the 4-class output (Condition B) actually decreased Accuracy to 25.0% and Macro F1 to 0.20.

**2. Does L2 regularization improve Macro F1?**
Marginally, but only relative to the smaller head. The L2-regularized `Dense(64)` model (Condition C) achieved 0.24 Macro F1, putting it practically at a dead heat with the unregularized reference model. Neither model exceeds random baseline bounds.

**3. Does mild augmentation help consistently across folds?**
**No. It catastrophically destroys performance.** Adding mild, mathematically valid augmentations (5% rotation/translation, 10% brightness/contrast variance) dropped Accuracy to 21.25% and Macro F1 to 0.1604. 
*Why?* The CNN was relying entirely on spurious optical artifacts (lighting angles, sensor pressure, background noise) specific to the 64 training individuals to minimize its loss. Augmentation disrupts these optical artifacts. Because there is no genuine biological blood-group signal for the network to fall back on, destroying the optical artifacts forces the network to fail entirely.

**4. Which classes remain difficult to classify?**
The minority classes (A+ and AB+) remain unclassifiable. The Macro F1 scores hovering around ~0.20 mathematically reflect models that are heavily defaulting to the majority O+ and B+ classes while suffering total recall collapse on AB+.

**5. How much does performance vary across folds?**
Performance variance is extreme. Some folds output 0% precision and recall for three out of four classes, heavily predicting majority. This is the hallmark of random guessing on a small-N dataset.

**6. Is there evidence of reproducible improvement?**
There is **absolutely zero evidence** of reproducible biological improvement. Every attempt to mathematically constrain the network to prevent memorization (Linear SVMs in Exp 8, Handcrafted Features in Exp 9, and now Regularization/Augmentation in Exp 10) results in the complete collapse of classification accuracy.

**7. What limitations remain because of the small dataset?**
The 101-participant dataset (80 in development) is too small to overcome deep neural network memorization safely. However, it is large enough to establish that no macroscopic, generalized biological correlation exists between fingerprint ridges and ABO blood types at optical scanner resolutions. If a strong biological signal existed, HOG features or augmented CNNs would have preserved at least partial predictive power, rather than collapsing to ~21% accuracy.
