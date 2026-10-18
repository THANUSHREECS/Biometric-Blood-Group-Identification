# EXP-08-DERMATOGLYPHIC-MINUTIAE: Handcrafted Biometric Feature Benchmark

## 1. Overview & Scientific Rationale
EXP-08 investigates whether **interpretable, handcrafted dermatoglyphic features** provide measurable predictive information regarding ABO blood groups (A+, AB+, B+, O+) on the Mantra MFS100 optical fingerprint scanner.

While deep neural networks (DenseNet-121, ResNet-50) failed to beat chance-level baselines in EXP-07, published anthropological literature reports weak population-level differences in fingerprint patterns, ridge counts, and minutiae density across blood groups. EXP-08 directly tests this biological hypothesis using transparent, interpretable machine learning models, completely eliminating deep learning black-box shortcut memorization.

---

## 2. Feature Extraction Methodology

### A. Primary Fingerprint Pattern Type
- Evaluates the local orientation field via 2D gradient structure tensor filtering.
- Detects core and delta singular points using the Poincaré Index ($\pm 180^\circ$ rotation).
- Categorizes pattern according to the Henry/Galton forensic classification system:
  - **WHORL:** $\ge 2$ cores or 2 deltas; concentric circular flow.
  - **LEFT LOOP:** 1 core, 1 delta to the right of core (loops open to the left).
  - **RIGHT LOOP:** 1 core, 1 delta to the left of core (loops open to the right).
  - **ARCH:** 0 cores / 0 deltas; continuous laminar transverse flow.

### B. Standardized Sub-Core Ridge Density
- Measures ridge density in a standardized **$5\text{ mm} \times 5\text{ mm}$ ($25\text{ mm}^2 \approx 98 \times 98\text{ pixels}$ at 500 DPI)** region located 2 mm below the primary core.
- Extracts dominant spatial frequency via 2D Fourier Power Spectrum analysis along ridge normals.
- Output: Ridges per $25\text{ mm}^2$ (standard forensic unit; typical human range: 8 to 16 ridges/25 mm²).

### C. Minutiae Extraction (Endings & Bifurcations)
- Applies adaptive contrast thresholding followed by morphological skeletonization (`skimage.morphology.skeletonize`).
- Employs the standard 8-neighbor **Crossing Number (CN)** algorithm:
  $$CN(P) = 0.5 \sum_{i=1}^{8} |P_i - P_{i+1}|$$
  - $CN = 1 \implies$ **Ridge Ending** (termination)
  - $CN = 3 \implies$ **Ridge Bifurcation**
- Discards spurious minutiae within 10 pixels of platen mask boundaries.
- Outputs: Endings count, bifurcations count, total minutiae count, and minutiae density per cm².

### D. Core-to-Delta Physical Distance
- When both singular landmarks are detected: Euclidean distance converted to physical millimeters ($d_{\text{mm}} = d_{\text{px}} / 19.685$).
- Explicit missing value handling: If core or delta is missing, the measurement is marked as `NaN` and flagged via binary missingness indicators.

---

## 3. Preprocessing, Aggregation & Leakage Safeguards

1. **3-Scan Participant Aggregation:**
   - The three independent impressions per participant are aggregated into **exactly ONE row** per participant.
   - Numeric features: Mean and standard deviation across impressions.
   - Categorical pattern: Majority vote across the 3 scans, with a concordance score ($1.0$, $0.67$, or $0.33$).
2. **Strict Cohort Isolation:**
   - 80 Development participants ONLY (240 scans).
   - 21-participant locked test cohort strictly quarantined (0 access).
   - 26-participant hospital cohort strictly quarantined (0 access).
3. **Leakage-Free Cross-Validation:**
   - Evaluated via 5-fold `StratifiedGroupKFold` grouped on `participant_id`.
   - `SimpleImputer(strategy="median")` and `StandardScaler()` are fit **strictly within training folds** via scikit-learn pipelines. Outer validation folds remain completely unobserved during preprocessing fitting.

---

## 4. Models & Evaluation Protocol

### Models Evaluated:
1. **Majority-Class Baseline:** Zero-intelligence predictor (`DummyClassifier(strategy="most_frequent")`).
2. **L2-Regularized Logistic Regression:** Linear classification baseline with balanced class weights (`C=1.0`, `max_iter=1000`).
3. **Random Forest Classifier:** Non-linear decision tree ensemble (`n_estimators=100`, `max_depth=4`, `class_weight="balanced"`).

### Evaluation Metrics:
- **Primary Metric:** Out-of-fold Balanced Accuracy.
- **Secondary Metrics:** Participant accuracy, Macro F1, Cohen's Kappa, per-class precision/recall, and canonical confusion matrices `["A+", "AB+", "B+", "O+"]`.
- **Significance Testing:** 1,000-run participant label permutation test deriving exact non-parametric $p$-values against chance.

---

## 5. Execution Commands

### Validation & Testing (Safe to Run - Zero Real Training)
```bash
# Validate participant splits and safety firewalls
python experiments/exp08_dermatoglyphic_minutiae/split_validator.py

# Run synthetic unit tests on dummy algorithms and pipelines
python experiments/exp08_dermatoglyphic_minutiae/synthetic_test.py

# Verify evaluation engine with synthetic demo
python experiments/exp08_dermatoglyphic_minutiae/evaluate.py --synthetic-demo
```

### Proposed Execution Commands (LOCKED AWAITING AUTHORIZATION)
*Requires `TRAINING_AUTHORIZED = True` in `config.py`:*
```bash
# Extract features and train models under 5-fold cross-validation
python experiments/exp08_dermatoglyphic_minutiae/train.py

# Evaluate out-of-fold predictions and run permutation significance tests
python experiments/exp08_dermatoglyphic_minutiae/evaluate.py
```
