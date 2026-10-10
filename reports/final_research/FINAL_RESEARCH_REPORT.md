# BIOMETRIC BLOODGROUP DETECTION: FINAL RESEARCH REPORT

## 1. ABSTRACT
This project investigated the feasibility of predicting ABO blood groups directly from optical fingerprint captures. A deep learning pipeline was trained on a dataset of 101 participants (303 scans) collected using the Mantra MFS100 optical scanner. While preliminary cross-validation yielded an optimistic Macro F1 of 0.42, independent testing on a locked set of 21 participants collapsed to 38.1% accuracy (random baseline equivalent). Rigorous diagnostic experiments (handcrafted features, linear boundaries, and augmentations) confirmed that the neural network had learned spurious optical and sensor artifacts specific to the training individuals, rather than a generalized biological blood-group signature. The hypothesis that macroscopic fingerprint ridges encode blood group is not supported by this optical dataset.

## 2. INTRODUCTION
The relationship between genetic markers, such as ABO blood types, and phenotypic fingerprint ridge expression (dermatoglyphics) is a subject of historical biometric interest. This research aimed to apply modern convolutional neural networks (CNNs) to discover whether a latent, macroscopic representation of blood group exists within optical fingerprint captures.

## 3. RESEARCH OBJECTIVE
To develop, validate, and independently test a machine learning model capable of classifying the four primary blood groups (A+, B+, AB+, O+) from raw optical fingerprint images, and to scientifically audit whether any learned signal generalizes biologically.

## 4. DATASET AND PROCEDURES
- **Total Participants:** 101
- **Total Scans:** 303 (3 left-thumb scans per participant)
- **Scanner:** Mantra MFS100 (optical, $316 \times 354$ px)
- **Class Distribution:** O+ (37), B+ (31), A+ (24), AB+ (9)
- **Data Split:** Training (64), Validation (16), Locked Test (21). The split was strictly maintained at the participant level to prevent identity leakage.

## 5. PREPROCESSING
A composite preprocessing pipeline (C5) was established via empirical benchmarking. It consists of:
1. Alpha compositing onto a white background.
2. Threshold-based foreground cropping.
3. Adaptive histogram equalization (CLAHE) for ridge contrast enhancement.
4. Resizing to $112 \times 112$ pixels for CNN compatibility.

## 6. EXPERIMENTAL METHODOLOGY
Model development relied entirely on 5-fold participant-level cross-validation over the 80 development participants (Train + Val). All hyperparameter selection was conducted via nested validation to prevent data leakage. 

## 7. SUMMARY OF EXPERIMENTS 0–10
- **Exp 0 (Baseline Audit):** An earlier model trained on 6,000 canonical synthetic images achieved 85.4% accuracy in-domain, but crashed to 38.09% when evaluated zero-shot on the optical MFS100 dataset.
- **Exp 1 (Architecture):** Benchmarked CNNs (VGG, ResNet, DenseNet, EfficientNet). DenseNet121 demonstrated the strongest baseline capacity.
- **Exp 2 (Transfer Learning):** Freezing the ImageNet backbone (T1) outperformed fine-tuning, preventing catastrophic overfitting on the micro-dataset.
- **Exp 3 (Preprocessing):** C5 (Crop + Contrast) yielded the highest validation retention.
- **Exp 4 (Three-Scan Fusion):** Concatenating the three scan feature vectors (F1) improved macro F1 over simple probability averaging.
- **Exp 5 (Metric Learning):** Supervised Contrastive Learning (SupCon) was introduced to improve minority class clustering.
- **Exp 6 (E2 Champion Model):** Combined Fusion and SupCon. **Development CV Result:** 53.75% Accuracy, 0.4240 Macro F1.
- **Exp 7 (Independent Test):** Evaluated E2 on the 21 locked participants. **Test Result:** 38.10% Accuracy, 0.2500 Macro F1. Generalization completely failed.
- **Exp 8 (Regularized Linear Baseline):** Replaced the CNN head with a Linear SVM. **Dev Result:** 0.2477 Macro F1. Proved the E2 CNN was memorizing parameters.
- **Exp 9 (Handcrafted Features):** Extracted HOG, FFT, LBP, and Quality metrics. **Dev Result:** HOG yielded 36.25% Accuracy. Proved CNN was leveraging non-biological optical artifacts.
- **Exp 10 (Augmentation/Regularization):** Applied mild rotations/brightness shifts to disrupt optical noise. **Dev Result:** Accuracy collapsed to 21.25%. 

## 8. FINAL INDEPENDENT TEST EVALUATION
The frozen Exp 6 (E2) model was evaluated strictly zero-shot on the locked 21 participants (63 scans).
- **Test Accuracy:** 38.10%
- **Test Macro F1:** 0.2500

## 9. PER-CLASS PERFORMANCE
During independent testing, the E2 model exhibited a complete recall collapse on the minority classes. 
- **A+ Recall:** 40.0%
- **B+ Recall:** 0.0%
- **AB+ Recall:** 0.0%
- **O+ Recall:** 88.9%
The network merely defaulted to predicting the majority classes (O+ and A+).

## 10. UNCERTAINTY AND LIMITATIONS
1. **Sample Size:** 101 participants (with only 9 AB+ individuals) is vastly inadequate for deep learning.
2. **Domain Artifacts:** The CNN easily memorized lighting, pressure, and background artifacts specific to the training individuals.
3. **Resolution:** 500 DPI optical scanning may lack the microscopic fidelity required to analyze biological minutiae density adequately.

## 11. ETHICAL AND MEDICAL DISCLAIMER
This research is strictly computational and exploratory. The models and methodologies documented here hold absolutely zero clinical validity. Fingerprints cannot and should not be used to medically determine ABO blood groups.

## 12. CONCLUSION
The experiments did not demonstrate reliable generalization to unseen participants. The initially optimistic 53.75% cross-validation accuracy was an artifact of severe neural overparameterization, where the network memorized spurious optical sensor noise rather than learning a generalized biological rule. This was conclusively proven when mathematically constrained models (SVMs, handcrafted HOG features, and augmented CNNs) collapsed to random noise (21-36% accuracy). While this failure does not theoretically rule out a biological connection, it proves that macroscopic fingerprint ridges extracted via standard optical scanners do not possess a detectable, generalizable blood-group signature under these conditions.

## 13. FUTURE RESEARCH DIRECTIONS
1. **Large-Scale Data Collection:** Any future attempt requires thousands of unique participant identities to prevent network memorization.
2. **High-Resolution Scanning:** Utilizing FBI-grade or microscopic ridge scanners (1000+ DPI) to explicitly measure sweat pore density and fine minutiae rather than macroscopic ridge flow.
