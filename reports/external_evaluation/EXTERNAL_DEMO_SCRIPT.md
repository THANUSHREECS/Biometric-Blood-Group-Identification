# External Demo Script

**Estimated Time: 5-7 Minutes**

## 1. Introduce the Problem Statement
"Good morning/afternoon, examiners. My major project is titled 'Biometric Blood Group Detection'. Traditional blood typing requires needles, reagents, and laboratory equipment, which can be time-consuming and invasive. This project explores a non-invasive biometric alternative: analyzing the dermatoglyphic patterns of fingerprints to estimate a person's blood group using deep learning."

## 2. Explain the Project Objective
"Our objective was to investigate whether deep learning models could identify subtle structural correlations between fingerprint ridges and blood group phenotypes. We aimed to build a research prototype to capture fingerprints, preprocess them, and run them through a deep learning model to predict the blood group."

## 3. Explain the Four-Class Scope
"To ensure rigorous testing and manage dataset limitations, this project focuses strictly on a four-class classification scope: the positive blood groups **A+, B+, AB+, and O+**. We excluded Rh-negative groups due to severe class imbalance in our collected dataset."

## 4. Explain the MFS100 Scanner
"For data acquisition, we used the Mantra MFS100 optical fingerprint scanner. It provides 500 DPI, FBI PIV certified captures. We developed a custom bridge to integrate this hardware with our Python backend to capture high-quality digital impressions."

## 5. Demonstrate Preprocessing and Inference
"I will now demonstrate the system. *(Perform a capture or upload an image)*. 
As the image is captured, it undergoes our C5 preprocessing pipeline. This involves foreground cropping to remove empty sensor space, alpha compositing, CLAHE ridge enhancement to improve contrast, and resizing. 
It is then passed to our E2 Champion Model—a frozen DenseNet121 feature extractor using a 5-checkpoint ensemble to make the final prediction based on three distinct scans."

## 6. Show the Probability Distribution
"On the dashboard, you can see the 4-Class Probability Distribution. The model assigns a confidence score to A+, B+, AB+, and O+. The ensemble averages these probabilities to produce a consensus prediction."

## 7. Generate the PDF Report
"The system also features automated reporting. By clicking here, we generate a PDF report that includes the biometric impression, the classification breakdown, and the research disclaimers. This can be saved for audit history."

## 8. Explain the Experimental Results Honestly
"It is important to discuss our empirical findings. We evaluated this model on an unseen, participant-level test set of 21 individuals. **The frozen research model achieved an accuracy of 38.10%.** While four-class random guessing has an expected accuracy of 25% only under uniform random prediction, the observed accuracy alone does not establish meaningful biological information. Therefore, the scientific conclusion of our research is that the tested system did not demonstrate reliable four-class blood group classification on unseen participants."

## 8b. Backup Plan (If Physical Scanner Fails)
*(If the Mantra MFS100 scanner fails to connect or capture during the live demo, smoothly transition to Mock Mode or File Upload)*
"As with any hardware demonstration, we have a fallback. I will now switch the system to 'Mock Mode' (or use File Upload) which feeds pre-collected authentic fingerprint images into the exact same preprocessing and inference pipeline. Note that this is running the true model on real data, simulating the scanner input."

## 9. Discuss Limitations and Future Improvements
"The primary limitation is the fundamental challenge of extracting strong biological blood group markers from macro-level ridge patterns. Furthermore, our dataset size restricted the model's ability to generalize. Future improvements would require a significantly larger, multi-demographic dataset, perhaps employing multispectral imaging rather than a standard optical sensor. This remains a research prototype and is not medically validated."
