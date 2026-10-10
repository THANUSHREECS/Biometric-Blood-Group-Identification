# VIVA TECHNICAL FACTS
## Biometric Blood Group Identification System — Q&A Reference

**Document Type:** Viva Examination Technical Reference
**Project:** B.E. Final Year Major Project — Computer Science and Engineering (AI and ML)
**Date:** September 2026
**Purpose:** Accurate factual answers to anticipated viva questions. All answers reflect the actual system — no inflation, no fabrication.

---

## SECTION 1: MOTIVATION AND PROBLEM STATEMENT

**Q: Why do you classify blood group from fingerprint images?**
A: Multiple published research papers (Cummins & Midlo, 1943; Rashad & Abd-Elaziz, 2021; and others) report statistical correlations between dermatoglyphic ridge patterns and ABO blood group phenotype. Our project investigates whether a deep learning model can learn these correlations from a large fingerprint dataset. This is exploratory research — we are not claiming to replace lab testing.

**Q: Why is rapid blood typing important?**
A: In emergency trauma response, battlefield medicine, remote rural clinics, and disaster scenarios, standard serological blood typing (venous blood + lab centrifuge + anti-A/anti-B serum) is unavailable. A rapid non-invasive preliminary estimate could assist triage decisions, pending laboratory confirmation.

**Q: Why 8 classes?**
A: The ABO blood group system has 4 groups (A, B, AB, O). Combined with the Rh factor (positive/negative), there are 8 standard human blood groups: A+, A−, AB+, AB−, B+, B−, O+, O−.

---

## SECTION 2: DATASET

**Q: What dataset did you use?**
A: A publicly sourced fingerprint blood group dataset containing 6,000 raw Windows Bitmap (.BMP) images, 750 per class (approximately). After SHA-256 cryptographic duplicate detection, 5,985 clean images were retained. 15 duplicates were quarantined with full audit mapping.

**Q: How did you split your dataset?**
A: Stratified split: 70% training (4,189), 15% validation (898), 15% testing (898). All 8 classes maintain proportional representation across splits. Cross-split SHA-256 verification confirmed zero data leakage.

**Q: What is data leakage and how did you prevent it?**
A: Data leakage occurs when information from the test set influences training (via duplicate images or overlapping splits). We used SHA-256 file hashing to detect and remove exact duplicates before splitting, ensuring Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.

**Q: Why are some images RGBA (32-bit) with transparency?**
A: The dataset BMPs include alpha transparency channels, with transparent border pixels having raw black RGB values (0,0,0). Naively stripping the alpha would produce artificial black borders that the model has never seen during training. We composite transparent regions onto a white background (255,255,255) matching optical scanner paper backgrounds.

---

## SECTION 3: PREPROCESSING

**Q: What preprocessing do you apply?**
A: Centralized pipeline in `src/preprocessing.py`:
1. Load image (file path / bytes / PIL / numpy — any format)
2. Alpha compositing onto white background (255,255,255) for RGBA images
3. 3-channel RGB conversion
4. Lanczos resizing to 224×224
5. Architecture-specific normalization: ResNet50 gets ImageNet mean subtraction; VGG16 and EfficientNetB0 get their native preprocessing

**Q: Why 224×224?**
A: 224×224 is the standard input size for ImageNet-pretrained models (ResNet50, VGG16, EfficientNetB0). All three architectures expect this resolution.

**Q: What is the special MFS100 preprocessing?**
A: Physical MFS100 captures have wide empty sensor margins. We apply:
- `crop_mfs100_foreground()`: detects active ridge region and crops empty glass margins, with 4 safeguards (blank guard, region check, size floor, margin presence check)
- `enhance_mfs100_ridge_contrast()`: CLAHE (clipLimit=2.5, 8×8 grid) + percentile contrast stretch + bilateral denoising — normalizes the washed-out optical ridge contrast to match training data distribution

**Q: Why does ResNet50 get 1.2× contrast enhancement during inference?**
A: During training, ResNet50 achieved its best performance with mild contrast enhancement. This enhancement was baked into the locked training protocol and must be applied consistently at inference to match the distribution the model learned from.

---

## SECTION 4: TRANSFER LEARNING

**Q: What is transfer learning?**
A: Transfer learning reuses a deep neural network pre-trained on a large dataset (ImageNet, 1.28M images, 1,000 classes) as a feature extractor for a new task. Instead of training from scratch, we freeze the pre-trained convolutional layers (which already detect edges, textures, and shapes) and train only a small classification head on our fingerprint data. This is critical when our dataset (5,985 images) is far smaller than what would be needed for training from scratch.

**Q: What is fine-tuning?**
A: After the classification head converges, we unfreeze the top layers of the convolutional backbone (e.g., `conv5_block` for ResNet50) and train them at a very low learning rate (10⁻⁵). This allows the model to adapt higher-level feature representations to fingerprint ridge patterns while preserving lower-level edge/texture features.

**Q: Why keep BatchNormalization frozen during fine-tuning?**
A: BatchNormalization layers accumulate running mean and variance statistics from ImageNet training. Unfreezing them with a small batch would corrupt these statistics, causing training instability. Keeping them frozen preserves stable normalization statistics.

---

## SECTION 5: MODEL ARCHITECTURES

**Q: What is ResNet50 and why did you use it?**
A: ResNet50 is a 50-layer deep convolutional network using residual (skip) connections. Skip connections solve the vanishing gradient problem by allowing gradients to bypass multiple layers. With 24.1M parameters pre-trained on ImageNet, it provides strong ridge-texture feature extraction. In our ensemble it carries 58% weight because it achieved the highest individual validation accuracy (~81%).

**Q: What is VGG16?**
A: VGG16 is a 16-layer deep convolutional network from Oxford VGGNet (2014). It uses a simple architecture of repeated 3×3 convolution blocks followed by max-pooling. With 14.8M parameters, it is straightforward to fine-tune. In our ensemble it carries 38% weight.

**Q: What is EfficientNetB0?**
A: EfficientNetB0 is the smallest member of the EfficientNet family, designed using compound scaling (depth, width, resolution scaled together optimally). With 4.4M parameters, it is more computationally efficient but achieved lower individual accuracy (~71%) on our dataset. It holds only 4% ensemble weight but contributes positively to AB+ class detection.

**Q: Why use three different architectures in your ensemble?**
A: Different architectures learn different feature representations. ResNet50 excels at deep ridge pattern hierarchies; VGG16 provides broad spatial texture features; EfficientNetB0 captures fine-scale ridge details efficiently. Their errors are partially uncorrelated, so combining them via soft voting averages out individual model mistakes.

---

## SECTION 6: ENSEMBLE LEARNING

**Q: What is ensemble learning?**
A: Ensemble learning combines predictions from multiple models to produce a more accurate and stable result than any single model. It exploits the fact that different models make different errors, so their averaged output cancels individual errors.

**Q: What is soft voting?**
A: Each model outputs a probability distribution over all 8 classes. Soft voting takes a weighted average of these distributions and selects the class with the highest averaged probability. This preserves more information than hard voting (which only uses each model's top-1 prediction).

**Q: How are your ensemble weights determined?**
A: Weights were determined by a grid search on the validation set, testing combinations of ResNet50 (0.5–0.7), VGG16 (0.2–0.4), and EfficientNetB0 (0.04–0.1). The combination ResNet50=58%, VGG16=38%, ENB0=4% produced the best validation accuracy. These weights are now locked.

**Q: Why is EfficientNetB0 only 4%?**
A: EfficientNetB0 achieved 70.6% individual validation accuracy vs. ResNet50's 81% and VGG16's 79%. A higher weight would reduce overall ensemble performance. Despite its low weight, it contributes positively, especially to AB+ class discrimination.

**Q: What is your ensemble's test accuracy?**
A: 85.41% (767/898 correct) on the locked 898-image test set. This is an improvement of +27.84 percentage points over the original ResNet50 baseline (57.57%).

---

## SECTION 7: MFS100 HARDWARE

**Q: What is the Mantra MFS100?**
A: The Mantra MFS100 is a USB optical fingerprint scanner using a 500 DPI optical platen sensor (316×354 pixel resolution). It captures fingerprint ridge impressions as grayscale BMP images. The SDK is distributed as a 32-bit Windows DLL (`MFS100Dll.dll`).

**Q: What is NFIQ?**
A: NFIQ (NIST Fingerprint Image Quality) is a standardized fingerprint quality metric from the US National Institute of Standards and Technology. It ranges from 1 (best) to 5 (worst). NFIQ 1 indicates an excellent scan suitable for reliable matching. NFIQ ≥ 3 triggers our consensus recommendation advisory.

**Q: Why did you need a 32-bit bridge?**
A: The MFS100 SDK DLL is 32-bit (x86). Python and TensorFlow run as 64-bit processes. A 64-bit process cannot load a 32-bit DLL (WinError 193). We solved this by running a separate lightweight 32-bit Python process as an HTTP bridge server on port 5100. The 64-bit Flask backend communicates with it via HTTP REST calls.

**Q: What happens during a hardware capture?**
A:
1. Flask (64-bit) sends HTTP POST to the 32-bit bridge (port 5100)
2. Bridge loads `MFS100Dll.dll` via ctypes
3. MFS100 SDK captures the fingerprint impression
4. Bridge returns raw image bytes + NFIQ score + quality score
5. Flask decodes the bytes, saves the BMP, and passes it to the preprocessing pipeline

---

## SECTION 8: FLASK AND DATABASE

**Q: Why Flask?**
A: Flask is a lightweight Python web framework suitable for research prototypes. It provides REST API routing, template rendering, and static file serving with minimal overhead. The entire application runs as a single process with a clean startup sequence (DB init → model load → scanner init → server start).

**Q: What database do you use?**
A: SQLite via Python's built-in `sqlite3` module. No external database server is required. The database (`data/blood_group.db`) stores prediction audit records including: timestamp, input type, image path, predicted group, confidence, status, notes, and finger_used.

**Q: What does finger_used store?**
A: `finger_used` is a TEXT column storing the acquisition finger as a protocol metadata field. For physical MFS100 captures, it stores `LEFT_THUMB`. For file uploads and mock captures, it is NULL. The scanner does NOT automatically verify finger identity — this is a convention-based audit metadata field.

**Q: Is the migration backward-compatible?**
A: Yes. The `finger_used` column was added via non-destructive `ALTER TABLE ... ADD COLUMN finger_used TEXT`. If the column already exists, the migration is skipped. All 198 pre-existing records retain `NULL` for this field.

---

## SECTION 9: CONFIDENCE THRESHOLD AND SAFETY MECHANISMS

**Q: What is your confidence threshold?**
A: 60% (0.60). If the Champion Ensemble's top-class probability is below 60%, the prediction is flagged as `LOW_CONFIDENCE` and a warning is displayed advising the operator to recapture or confirm via laboratory testing.

**Q: Why 60%?**
A: This threshold was chosen empirically based on validation set analysis. It provides a meaningful safety gate without flagging too many correct predictions. On the validation set, incorrect AB+ predictions had a mean confidence of only 63.94%, and 42.86% of wrong AB+ results were correctly caught at 60%.

**Q: What is the ABO/Rh decomposition?**
A: An analytical decision-support layer (not an additional model). After the 8-class prediction, we compute:
- P(A) = P(A+) + P(A−)
- P(B) = P(B+) + P(B−)
- P(AB) = P(AB+) + P(AB−)
- P(O) = P(O+) + P(O−)

The dominant ABO group and Rh factor are derived from these sums. This provides clinical explainability without modifying the original prediction.

**Q: When does the Rh Uncertainty Advisory trigger?**
A: Three independent rules:
1. Rh probability margin < 20% (P(ABO+) vs P(ABO−) too close)
2. 8-class confidence < 65% (low overall confidence)
3. Predicted Rh-negative (weak dermatoglyphic Rh correlation)

Any one rule triggers the advisory. It does NOT change the prediction.

---

## SECTION 10: LEFT THUMB PROTOCOL

**Q: Why did you standardize on the left thumb?**
A: During real MFS100 testing, left thumb impressions produced more consistent and higher-confidence results for the test subject. However, this is purely an empirical observation from a small number of tests. We standardized on the left thumb for demo reproducibility, not because of any biological claim.

**Q: Does using the left thumb guarantee a correct prediction?**
A: No. A confirmed AB+ subject scanning with the left thumb still received an O− prediction (41.6% confidence, correctly flagged LOW_CONFIDENCE). The model does not know which finger is being used. LEFT_THUMB is metadata only.

**Q: Does the model use finger_used as an input?**
A: Absolutely not. `finger_used` is only stored in the database and printed in the PDF report. The `predict()` function in `src/predictor.py` does not accept a `finger_used` parameter.

---

## SECTION 11: LIMITATIONS AND HONEST REPORTING

**Q: Why do you report 85.41% and not a higher number?**
A: Because 85.41% is the genuine result from a single locked evaluation on unseen test data. We do not cherry-pick, re-run until a higher number appears, or access the test set multiple times. Academic integrity requires honest reporting.

**Q: Why can AB+ be misclassified as O−?**
A: ResNet50 and VGG16, which carry 96% of ensemble weight, occasionally map AB+ ridge textures into the O− feature space. Both AB and O groups can show similar looped ridge patterns. This is a known limitation of dermatoglyphic classification and occurs in 6.60% of AB+ validation samples.

**Q: Is this system ready for clinical use?**
A: No. It is an academic research prototype. It has not undergone:
- Formal clinical validation on diverse real-world subjects
- Independent hospital laboratory comparison studies
- Regulatory approval (FDA/CE/CDSCO)
- Testing across different ethnicities, age groups, and skin conditions

**Q: What is your medical disclaimer?**
A: "This system is an AI-based preliminary estimation tool developed for academic research and proof-of-concept investigation. The results MUST BE CONFIRMED using standard laboratory serological blood typing tests before any clinical or medical administration."

**Q: What is the future scope?**
A:
1. Collect 50–100+ real MFS100 fingerprint scans from known AB+ donors for domain-specific data
2. Increase EfficientNetB0 ensemble weight after rigorous fresh held-out evaluation
3. Multi-subject clinical validation study
4. Grad-CAM explainability to visualize which ridge regions drive predictions
5. Class-conditional Rh factor refinement with a dedicated second-stage classifier

---

*This document is part of the B.E. Final Year Major Project — Biometric Blood Group Identification System.*
*All technical facts reflect the actual implemented system. For viva examination use. September 2026.*
