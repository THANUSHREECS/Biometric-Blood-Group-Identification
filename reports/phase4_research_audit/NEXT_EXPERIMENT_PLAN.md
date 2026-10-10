# NEXT EXPERIMENT PLAN

Based on the historical audit and generalization failure analysis, the next experiments must prioritize extreme regularization and drastically reducing model capacity to prevent the memorization of the 64 training participants.

**Strict Constraint:** All proposed experiments must use *only* the 80 development participants in identical 5-fold cross-validation. The 21 locked test participants must not be accessed.

---

### Priority 1: Regularized Linear Baseline on Frozen Features
- **Hypothesis:** A massively overparameterized 1.5M parameter classification head memorized the 64 participants. A simple linear classifier (e.g., Logistic Regression or SVM with high L2 penalty) directly on the frozen 1024-d DenseNet features will prevent memorization and reveal the true biological signal (if any exists).
- **Justification:** Machine learning theory mandates using models with capacity proportional to dataset size. N=64 demands linear models.
- **Data Used:** 80 development participants (5-fold CV).
- **Primary Metric:** Macro F1.
- **Secondary Metrics:** Balanced Accuracy, Per-class Recall.
- **Baseline:** E2 Champion mean CV Macro F1 (0.42).
- **Stopping Criteria:** Completion of 5-fold CV.
- **Limitations:** Cannot learn complex hierarchical combinations of features.
- **Computational Cost:** Extremely low (seconds).

---

### Priority 2: Hand-Crafted Biometric Feature Extraction
- **Hypothesis:** Deep learning backbones pre-trained on ImageNet (dogs, cars) are poorly suited for fingerprint ridges. Extracting classical biometric features (ridge count, minutiae density, spatial frequency) will provide a biologically interpretable feature space that resists optical domain memorization.
- **Justification:** If blood group affects fingerprints, it must alter physical ridge structure. Traditional biometric features explicitly quantify this structure.
- **Data Used:** 80 development participants (5-fold CV).
- **Primary Metric:** Macro F1.
- **Secondary Metrics:** Balanced Accuracy, Per-class Recall.
- **Baseline:** E2 Champion mean CV Macro F1 (0.42).
- **Stopping Criteria:** Completion of 5-fold CV on classical features.
- **Limitations:** Minutiae extraction on MFS100 captures may be noisy without advanced enhancement.
- **Computational Cost:** Medium (CPU time for ridge extraction).

---

### Priority 3: Extreme Destructive Augmentation
- **Hypothesis:** The network is relying on spurious correlations (e.g., background noise, sensor platen borders). Applying extreme destructive augmentations (Mixup, heavy Cutout, aggressive random Gaussian blur) will destroy spurious artifacts and force the network to look for robust ridge textures.
- **Justification:** Regularization via extreme augmentation is necessary when N=64 to simulate a larger manifold and prevent overfitting.
- **Data Used:** 80 development participants (5-fold CV).
- **Primary Metric:** Macro F1.
- **Secondary Metrics:** Balanced Accuracy, Per-class Recall.
- **Baseline:** E2 Champion mean CV Macro F1 (0.42).
- **Stopping Criteria:** Completion of 5-fold CV.
- **Limitations:** Too much augmentation destroys the fingerprints entirely, leaving the model blind.
- **Computational Cost:** High (requires re-training the CNN heads).
