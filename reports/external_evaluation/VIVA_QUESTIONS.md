# VIVA PREPARATION GUIDE

This document prepares the student for questions during the final-year major project external evaluation.

## 1. Problem Statement & Motivation
**Q: What specific problem were you trying to solve?**
*A: We investigated whether a non-invasive biometric scan (optical fingerprint) could predict a patient's ABO blood group, attempting to map macroscopic ridge phenotypes to genetic traits.*

## 2. Dataset Collection & Preprocessing
**Q: How did you acquire your data?**
*A: We utilized the Mantra MFS100 optical scanner to capture 303 scans across 101 unique participants. We applied a C5 preprocessing pipeline involving foreground cropping and CLAHE contrast enhancement to normalize scanner noise.*

## 3. Deep Learning Architectures & Transfer Learning
**Q: Why did you use DenseNet121, and why was it frozen?**
*A: DenseNet121 provides highly efficient feature extraction. Because our dataset only contained 64 training individuals, attempting to fine-tune the millions of parameters in the CNN backbone would result in catastrophic overfitting. We utilized Transfer Learning by freezing the ImageNet weights to act as a generalized texture extractor.*

## 4. Supervised Contrastive Learning & Feature Fusion
**Q: What is Supervised Contrastive Learning, and why did you use it?**
*A: Standard cross-entropy loss struggles with severe class imbalance (we had only 9 AB+ participants). SupCon pulls representations of the same class closer together in latent space while pushing different classes apart, helping to cluster minority blood groups before classification.*

## 5. Participant-Level Data Splitting
**Q: Why is participant-level splitting critical in biometrics?**
*A: If you split a dataset at the image level, different scans from the same person end up in both the training and test sets. The neural network will simply memorize the identity/fingerprint of the person (data leakage) rather than learning the generalized blood group rule. We enforced strict participant-level isolation.*

## 6. Evaluation Metrics & Generalization Failure
**Q: Your model achieved 38.10% accuracy on the final test set. Is this a failure?**
*A: From a clinical standpoint, yes. However, from a scientific and engineering standpoint, it is a success. We built a rigorous evaluation pipeline that correctly identified that the CNN was memorizing optical sensor artifacts during training. When tested on unseen individuals, it fell back to random guessing (~38%). The project successfully proved that macroscopic optical ridges do not encode generalized blood group traits.*

## 7. Limitations & Ethical Considerations
**Q: What are the main limitations and ethical concerns of this system?**
*A: The primary limitation is the small sample size (101 participants). Ethically, it is critical that this prototype is never used to make medical decisions. Misclassifying a blood type during a transfusion is fatal. The system is strictly a computational research tool.*
