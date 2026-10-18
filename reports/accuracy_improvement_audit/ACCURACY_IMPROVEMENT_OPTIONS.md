# Accuracy Improvement Options & Strategic Roadmap

**Audit Date:** 2026-10-17  
**Scope:** Systematic evaluation of potential research interventions to establish whether genuine, scientifically valid accuracy improvements are achievable.

---

## 1. Evaluation & Prioritization Matrix

| Research Option | Scientific Rationale | Expected Benefit | Risks & Pitfalls | Feasibility & Cost | Priority |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Option 1: Native / Higher Resolution ($224 \times 224$ or $316 \times 354$) with Minimal Preprocessing** | Lanczos downsampling to $112 \times 112$ destroyed ridge structures (down to 2.6 px/period). Restoring native resolution preserves minutiae geometry. | Eliminates spatial aliasing; enables backbones to see true ridge bifurcations. | Backbones may overfit faster if capacity is too high; higher GPU memory. | **High / Low Cost** (Code modification only, runs on current GPU). | **1 (Immediate)** |
| **Option 2: Strict Nested Cross-Validation Protocol (Leakage-Free)** | Replaces outer-fold checkpoint selection with nested inner-validation or fixed 100-epoch training. | Yields mathematically unbiased, trustworthy generalization estimates; ends metric inflation. | Development scores will appear lower (~32%–36%) than historical inflated 53.75%. | **High / Low Cost** (Methodological correction). | **1 (Immediate)** |
| **Option 3: Classical Biometric Handcrafted Feature Baseline** | Biological dermatoglyphic literature links ABO to ridge density, pattern type (loops/whorls), and minutiae density. Extract handcrafted features. | Transparent, biologically grounded, zero deep-learning overfitting risk on small $N=80$. | May confirm that optical fingerprints have weak/negligible predictive correlation with ABO. | **High / Moderate Effort** (Requires minutiae extraction script). | **2 (Near-Term)** |
| **Option 4: Hospital Cohort Identity Audit & Serology Validation** | The 26 hospital participants possess laboratory serology. Resolving cross-cohort identity overlap enables a clean, serologically verified test set. | Validates whether self-reported label noise was the limiting bottleneck. | Participant records may lack unique cross-linking IDs; IRB restrictions may limit use. | **Moderate / Low Compute** (Requires administrative/IRB review). | **2 (Near-Term)** |
| **Option 5: Biometric Domain Pretraining (NIST / FVC Fingerprint SSL)** | Pretrained ImageNet weights (trained on natural objects like dogs and cars) lack domain-specific fingerprint representations. | Backbones learn invariant representations of friction ridges and minutiae flows. | Complex pretraining pipeline; requires substantial external fingerprint data. | **Low / High Compute** (Substantial external data engineering). | **3 (Long-Term)** |

---

## 2. Deep-Dive Analysis of Prioritized Options

### Option 1: Native Resolution & Raw Preprocessing
- **Mechanics:** Replace $112 \times 112$ downsampling with $224 \times 224$ center-crop / resize, or process native $316 \times 354$ scans using patch-based convolutions. Eliminate CLAHE and bilateral smoothing entirely, using only robust min-max normalization.
- **Why It Matters:** Human friction ridges at 500 DPI require at least $5\text{ to }8\text{ pixels}$ per cycle to resolve minutiae endings without severe phase distortion. At $224 \times 224$, the ridge period is $\approx 5.5\text{ pixels}$, comfortably above the Nyquist boundary.
- **Scientific Safeguard:** Must be evaluated strictly using Option 2's leakage-free cross-validation.

### Option 2: Methodological Correction (Nested Inner-Val or Fixed Epochs)
- **Mechanics:**
  - *Approach A (Fixed Epochs):* Fix head training to exactly 80 epochs with cosine decay and evaluate the final epoch weights. No early stopping or callback-based checkpoint selection on the outer fold.
  - *Approach B (Nested Inner-Val):* Inside each 64-participant training fold, split into 48 train and 16 inner-val. Select the best checkpoint on inner-val, then evaluate strictly once on the 16 outer-fold participants.
- **Why It Matters:** Guarantees that any measured accuracy is biologically genuine and not an artifact of random search over 100 evaluation steps.

### Option 3: Classical Dermatoglyphic Feature Extraction
- **Mechanics:** Extract validated forensic biometric features:
  1. Ridge density (ridges per 25 mm² in the sub-core region).
  2. Primary pattern classification (Arch, Left Loop, Right Loop, Whorl).
  3. Total Minutiae Count (Bifurcations and Ridge Endings via skeletonization).
  4. Core-to-delta ridge count.
- Train an $L_2$-regularized Logistic Regression or Random Forest classifier on the 80 Dev participants.
- **Why It Matters:** Directly tests the published anthropological hypothesis that blood group antigens correlate with embryonic ridge formation rate, providing an interpretable scientific benchmark that deep networks cannot obscure.
