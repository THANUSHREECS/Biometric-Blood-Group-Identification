# Viva Preparation Guide

This document contains 30 potential viva questions and technically correct answers based strictly on the project's methodology, architecture, and empirical results.

## Project Objective & Motivation

**1. What is the main objective of your project?**
**Answer:** The objective was to investigate if deep learning models can classify blood groups (A+, B+, AB+, O+) non-invasively using digital fingerprint images, and to build an end-to-end prototype system.

**2. Why did you choose fingerprints to detect blood groups?**
**Answer:** Some past biometric literature suggests weak correlations between dermatoglyphic patterns (like loops, whorls) and blood phenotypes. We wanted to test if CNNs could identify these patterns automatically.

**3. Why is the scope limited to only 4 positive blood groups?**
**Answer:** We limited the scope to A+, B+, AB+, and O+ because Rh-negative blood types were extremely rare in our collected dataset. Including them would cause severe class imbalance and compromise training.

## Hardware & Data Collection

**4. Which scanner did you use and why?**
**Answer:** We used the Mantra MFS100 optical fingerprint scanner. It provides high-resolution (500 DPI) images and is FBI PIV certified, ensuring standard biometric quality.

**5. How did you collect the dataset?**
**Answer:** We collected fingerprint samples from voluntary participants and labeled them with their verified blood groups. We used a standardized protocol capturing the left thumb.

**6. Why is participant-level splitting critical in your dataset?**
**Answer:** If the same participant's fingerprints appear in both the training and test sets, the model might just learn to recognize the person's identity rather than their blood group. Participant-level splitting ensures no data leakage and tests true generalization.

## Preprocessing

**7. What is the C5 Preprocessing Pipeline?**
**Answer:** C5 is our five-step pipeline: 1. Load image, 2. Composite alpha on white, 3. Crop empty MFS100 foreground margins, 4. Enhance ridges using CLAHE (Contrast Limited Adaptive Histogram Equalization), and 5. Resize to 112x112 for the model.

**8. Why did you use CLAHE for ridge enhancement?**
**Answer:** Optical scanners can produce washed-out or low-contrast images. CLAHE normalizes the contrast locally, making the ridge patterns and minutiae much clearer for the CNN to extract features.

**9. What happens if the scanner captures a blank image?**
**Answer:** Our cropping algorithm includes safeguards. If the variance of the image is too low, it detects a blank platen, aborts cropping, and falls back to the original image to prevent crashes.

## Deep Learning Architecture

**10. What model architecture did you use?**
**Answer:** Our E2 Champion Model uses a frozen DenseNet121 as a feature extractor.

**11. Why did you choose DenseNet121?**
**Answer:** DenseNet connects each layer to every other layer in a feed-forward fashion. This mitigates the vanishing gradient problem, encourages feature reuse, and is efficient with parameters.

**12. What does "frozen" mean in this context?**
**Answer:** The weights of the DenseNet121 base, pre-trained on ImageNet, were kept untrainable. We only trained the new dense classification head appended to the network.

**13. What is Feature Fusion?**
**Answer:** Instead of predicting on one scan, the model takes 3 distinct fingerprint scans from the same person. It extracts features from all 3 using DenseNet121 and concatenates them (Feature Fusion) before making a final classification.

**14. What is Supervised Contrastive Learning (SupCon)?**
**Answer:** SupCon is a training technique that pulls representations of the same class closer together in the embedding space and pushes different classes apart, helping the model learn better decision boundaries before standard cross-entropy training.

## Model Evaluation

**15. What is a 5-fold ensemble?**
**Answer:** We trained 5 different checkpoints of the model using 5-fold cross-validation. During inference, we pass the image through all 5 models and average their output probabilities to make a robust, consensus prediction.

**16. How did you handle overfitting?**
**Answer:** We used Dropout layers (0.5) in the classification head, and applied data augmentation and early stopping during training (subject to confirmation from training configuration records).

**17. What metrics did you use to evaluate the model?**
**Answer:** We used Accuracy, Precision, Recall, F1-Score (Macro and Weighted), and Balanced Accuracy.

**18. What is the difference between Macro F1 and Weighted F1?**
**Answer:** Macro F1 calculates the metric independently for each class and averages them equally. Weighted F1 averages them but gives more weight to classes with more samples, accounting for class imbalance.

**19. What is Balanced Accuracy?**
**Answer:** It is the arithmetic mean of sensitivity (true positive rate) for each class, providing a fairer metric than standard accuracy when classes are imbalanced.

## Results & Limitations

**20. What was your final test accuracy?**
**Answer:** The final test accuracy on unseen participants was 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware).

**21. Isn't 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) very low?**
**Answer:** Four-class random guessing has an expected accuracy of 25% only under uniform random prediction. While 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) is higher, observed accuracy alone does not establish meaningful biological information. Therefore, our scientific conclusion is that the tested system did not demonstrate reliable four-class blood group classification on unseen participants.

**22. Why do you think the accuracy is not higher?**
**Answer:** Blood group markers in fingerprints may be too subtle or non-existent at a macro-ridge level. Additionally, our dataset was relatively small, making it difficult for the model to generalize across the entire population.

**23. Why didn't you just train the model more to get higher accuracy?**
**Answer:** We strictly maintained participant-level splitting. If we compromised this to inflate accuracy, the model would simply memorize individuals, which is scientifically invalid.

## Application & UI

**24. What backend framework did you use?**
**Answer:** We used Flask (Python) to serve the API and connect the machine learning models with the frontend.

**25. How is the MFS100 scanner integrated?**
**Answer:** We wrote a custom Python bridge using the Mantra SDK DLL (MFS100Dll.dll) and `ctypes` to interface with the hardware directly from our backend.

**26. Why do you capture 3 scans?**
**Answer:** Fingerprint captures can have noise, smudges, or varying pressure. Capturing 3 distinct scans and fusing their features gives the model more stable data to analyze.

**27. What does the "Mock Mode" do?**
**Answer:** It simulates the scanner using pre-collected authentic dataset images, allowing us to demonstrate the software workflow even if the physical scanner is unavailable.

## Ethics & Future Scope

**28. Is this system ready for hospitals?**
**Answer:** No. It is strictly a research prototype. Our system displays clear medical disclaimers and is not clinically validated. Blood typing for medical decisions requires standard serological testing.

**29. What are the ethical considerations of your project?**
**Answer:** Fingerprints are sensitive PII (Personally Identifiable Information). We anonymized all data during collection and ensured that the application doesn't store permanent biometric records without consent.

**30. What future improvements would you suggest?**
**Answer:** We recommend using multispectral imaging instead of standard optical sensors to capture subsurface blood vessel data. Additionally, collecting a massive, multi-demographic dataset with millions of samples might yield better results.

## Rigorous Evaluation Audit (Final Phase Updates)

**31. Did your final evaluation include a data leakage audit? What were the findings?**
**Answer:** Yes. We audited our dataset splits and found that the raw image clusters lack explicit metadata linking individual images to unique human subject IDs. Therefore, strict subject-level splitting could not be definitively guaranteed. Because of this, our initial 8-class accuracy claims (~85%) are possibly inflated and cannot be verified without subject IDs.

**32. What are the primary "Threats to Validity" in your research?**
**Answer:** There are four major threats to validity:
1. **No Subject IDs:** This prevents true measurement of population generalization.
2. **Domain Shift:** Our live MFS100 scanner captures show different ridge contrast and mean intensity compared to the curated dataset.
3. **Small Live Sample:** We primarily tested live on a single participant (O+ thumb), which drastically limits external validity.
4. **Statistical vs. Scientific Significance:** A 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) accuracy on 4-classes is statistically better than random guessing (25%), but it completely lacks the scientific and clinical significance needed for a medical device.

**33. How does your 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) accuracy compare to statistical baselines?**
**Answer:** We computed the random baseline (25.00%) and majority-class baseline (26.95%). A bootstrap 95% confidence interval placed our accuracy between 34.86% and 41.31%. A permutation test confirmed our result is statistically better than random chance ($p < 0.001$), but the clinical utility remains negligible.

**34. How did you handle model calibration and overconfidence?**
**Answer:** The original model was highly overconfident. We performed calibration analysis, finding an uncalibrated Expected Calibration Error (ECE) of 0.4094. Applying temperature scaling on the validation set reduced the ECE to 0.2802. We also instituted a strict 70% threshold in the UI, below which the system abstains and outputs "Not Reliable" instead of a blood group.

**35. What is the fundamental limitation of this project?**
**Answer:** The primary limitation is biological plausibility: there is no proven genetic mechanism linking friction ridge patterns to ABO blood groups. Our 86.0% on a random split. Subject independence cannot be verified (no subject IDs). Dataset artifacts are possible. | Subject independence cannot be verified. (group-aware) accuracy may capture latent confounding variables in the dataset (like regional demographic distributions) rather than a universal biological truth. Additionally, the data leakage issues prevent claims of broad population generalization. The system is completely unfit for clinical use.
