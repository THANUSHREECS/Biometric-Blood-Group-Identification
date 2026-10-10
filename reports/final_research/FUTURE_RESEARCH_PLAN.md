# FUTURE RESEARCH PLAN

This document outlines the priorities for any future investigations into the biometric bloodgroup detection hypothesis, based on the findings and limitations identified during the MFS100 diagnostic audit.

## 1. INDEPENDENT DATA COLLECTION
The most critical bottleneck in the current research is sample size. 101 participants (and only 9 AB+ participants) mathematically prohibits robust deep learning.
- **Action:** Collect a minimum of 2,000 to 5,000 unique participant identities.
- **Constraint:** Ensure strict demographic, age, and gender balance to prevent the model from learning demographic proxies instead of blood group directly.
- **Constraint:** Ensure biological class balancing. Actively recruit AB+, B+, and A- individuals to prevent catastrophic minority-class collapse.

## 2. STRONGER PARTICIPANT-LEVEL VALIDATION
The diagnostic audit proved that CNNs easily memorize individuals via optical artifacts.
- **Action:** Enforce strict, isolated participant-level splitting at the absolute lowest data-loading levels.
- **Action:** Do not use image-level splits under any circumstances (as was done in the historical 85.4% benchmark).
- **Action:** Implement rigorous Nested Cross-Validation for any hyperparameter tuning to explicitly measure model selection variance.

## 3. ACQUISITION AND DOMAIN-SHIFT ANALYSIS
The optical MFS100 scanner ($316 \times 354$ px at 500 DPI) captures macroscopic ridge flow and heavy platen-pressure artifacts, but struggles to capture microscopic pore structures.
- **Action:** Capture fingerprints from the exact same participant using multiple different scanner hardwares (e.g., Optical vs. Capacitive vs. Ultrasonic).
- **Goal:** Train a model to predict blood group while strictly penalizing the ability to predict the scanner type, forcing the model to ignore hardware artifacts.
- **Hardware Upgrade:** Utilize high-resolution forensic scanners (1000+ DPI) to resolve level-3 fingerprint features (sweat pores, ridge edges), which have a higher probability of containing genetic/biological links than macroscopic ridge whorls.

## 4. VERIFICATION OF THE CANONICAL-DATASET EVALUATION
The original repository claimed an 85.41% accuracy on a canonical 6,000-image dataset (likely SOCOFing).
- **Action:** Audit the historical dataset split. It is highly probable that the split was performed at the image level, allowing different fingers of the exact same synthetic individual into both the training and test sets.
- **Action:** Rerun the historical baseline using strict participant-level splitting to determine if the 85.41% was an artifact of identity leakage.

## 5. FURTHER MODELING
Deep learning modeling should be halted until the data collection and acquisition challenges are solved.
- **Constraint:** Do not deploy more complex architectures (Vision Transformers, larger CNNs) on the current MFS100 dataset. The failure is a data limitation, not a model capacity limitation.
- **Alternative:** Once a high-resolution, multi-thousand participant dataset is established, investigate explicitly neuro-symbolic models that combine learned CNN representations with hard-coded minutiae and sweat-pore graphs.
