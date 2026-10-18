# Chronological Experiment Status & Lineage

**Audit Date:** 2026-10-17  
**Scope:** Complete chronological record of all 15 project development phases from inception through final scientific audit.

---

## 1. Timeline Overview of Experimental Progression

```
[Phase 1: Legacy 6k Dataset]
  ├── Train 7 CNN architectures (Random image split) ──> Historical "86.02%" peak
  └── Legacy ensemble attempts
          │
          ▼
[Phase 2: Domain Shift Crisis]
  ├── Mantra MFS100 Scanner Deployment (101 participants, 303 scans)
  └── Zero-Shot Evaluation of Legacy Models ──> Accuracy collapses to 19%-28% (Below chance)
          │
          ▼
[Phase 3: Participant-Level Partitioning & E2 Three-Scan Architecture]
  ├── Enforce strict participant splits: Train (64), Val (16), Test (21)
  └── E2 Prototype: 3 scans -> DenseNet-121 -> Concat (3072-d) -> Linear Head
          │
          ▼
[Phase 4: Optimization Sweep (EXP-03 to EXP-06D)]
  ├── EXP-03: Augmentation search (Blur, rotation, brightness)
  ├── EXP-04: DenseNet-121 backbone fine-tuning vs frozen
  ├── EXP-05: Feature fusion variations (Concat vs Attention vs Bilinear)
  ├── EXP-06A: Contrastive representation learning (Supervised SimCLR)
  ├── EXP-06C: Linear Head (Config A) vs MLP Head (Config B) -> Config A chosen (53.75% val acc)
  └── EXP-06D: Preprocessing ablation -> Enhanced C5 found to hurt performance
          │
          ▼
[Phase 5: Locked-Cohort Evaluation & Forensic Audits]
  ├── Evaluation of EXP-06C Config A on 21-person locked test set ──> 33.33% (7/21 correct)
  ├── Permutation null test ──> p = 0.5641 (Accuracy within random noise)
  ├── Leakage Audit ──> Checkpoint selection on held-out folds created ~15% artificial inflation
  └── Shortcut Audit ──> Legacy 6k dataset separable by trivial border/intensity statistics (~51% acc)
```

---

## 2. Phase-by-Phase Detailed Chronology

### Phase 1: Original Fingerprint Dataset Preparation & Baseline CNNs
- **Dataset:** ~6,000 synthetic/scraped fingerprint images (2,777 images in 4-class positive subset: A+: 565, B+: 652, AB+: 708, O+: 852).
- **Protocol:** Random image-level train/test split (80/20).
- **Architectures Tested:** Custom CNN (16.93%), MobileNetV2 (41.09%), DenseNet-121 (53.45%), ResNet-50 (78.17%), VGG16 (79.18%), Improved ResNet-50 (85.30%), EfficientNet-B0 (86.02%).
- **Findings:** Apparent breakthrough performance. However, because images from the same fingers appeared in both train and test splits, models learned individual fingerprint identity or scanner acquisition artifacts rather than bloodgroup antigens.

### Phase 2: Mantra MFS100 Collection & Domain Transfer Failure
- **Data:** Primary MFS100 cohort collected (101 participants, 3 scans each = 303 scans at 500 DPI optical resolution).
- **Zero-Shot Test:** Pretrained legacy models evaluated on MFS100 scans without fine-tuning.
- **Results:**
  - EfficientNet-B0: 19.05% to 28.57% accuracy.
  - ResNet-50: 23.81% accuracy.
- **Finding:** Complete failure of domain transfer. Confirmed that legacy models did not learn invariant biological ridge markers.

### Phase 3: Participant-Level Partitioning & E2 Architecture
- **Protocol Formulation:** Participant-level firewall introduced to guarantee zero participant overlap across splits.
  - Train: 64 participants (192 scans).
  - Validation: 16 participants (48 scans).
  - Locked Test: 21 participants (63 scans).
- **E2 Architecture:** Multi-instance three-scan concatenation:
  - Input: Three independent finger impressions per participant $(1, 3, 112, 112, 3)$.
  - Backbone: DenseNet-121 (pretrained ImageNet, frozen).
  - Feature Concatenation: $3 \times 1024 = 3072$-dimensional embedding.
  - Classification Head: Linear projection layer to 4 logits.
- **Baseline Result:** 38.10% accuracy on early test exposure (8/21 participants correct).

### Phase 4: Systematic Development Experiments (EXP-03 to EXP-06D)
- **EXP-03 (Data Augmentation):** Tested affine transformations, elastic distortions, and contrast jitter. Found heavy augmentation degraded validation convergence due to low native resolution ($112 \times 112$).
- **EXP-04 (Backbone Fine-Tuning):** Unfreezing DenseNet-121 blocks led to immediate catastrophic overfitting on the 64 training participants (training accuracy reached 100%, validation collapsed to ~25%). Freezing the backbone was deemed mandatory.
- **EXP-05 (Feature Fusion):** Compared concatenation (3072-d), cross-attention pooling, and elementwise averaging. Simple concatenation demonstrated superior validation stability.
- **EXP-06A (Supervised Contrastive Learning):** Pretrained embeddings with SupCon loss before linear readout. Achieved OOF validation accuracy of 48.75%, but no statistically significant gain over direct training.
- **EXP-06C (Head Architecture Comparison):**
  - **Config A (Linear Head):** 5-fold CV reported 53.75% accuracy, 53.47% balanced accuracy, Macro F1 0.5487.
  - **Config B (Nonlinear MLP Head, 2-layer with Dropout):** Reported 46.25% accuracy, 52.12% balanced accuracy, Macro F1 0.4829.
  - *Selection:* Config A selected as the provisional candidate.
- **EXP-06D (Preprocessing Ablation):**
  - Compared Enhanced C5 (CLAHE + Bilateral Filtering + Normalization) against Raw C5 (Standardized intensity crop).
  - Raw C5 outperformed Enhanced C5 by +0.0622 Macro F1. Proved that CLAHE and bilateral smoothing destroyed minute ridge bifurcation details.

### Phase 5: Locked-Cohort Evaluation & Forensic Audits
- **Final Evaluation Execution:** EXP-06C Config A ensemble evaluated on the 21-participant locked cohort.
- **Empirical Metric:** Accuracy = 33.33% (7/21 correct), Balanced Accuracy = 31.25%, Macro F1 = 0.2083.
- **Majority-Class Baseline Comparison:** Always predicting O+ yields 8/21 = 38.10%. The final model performed worse than zero-intelligence majority voting.
- **Permutation Significance:** 1,000 label permutations produced $p = 0.5641$ (accuracy) and $p = 0.3712$ (Macro F1), establishing that the test result is completely indistinguishable from chance.
- **Forensic Audits:** Identified checkpoint selection leakage in Phase 4 (explained the 53.75% illusion) and discovered 42 trivial shortcut features in the Phase 1 legacy dataset.
