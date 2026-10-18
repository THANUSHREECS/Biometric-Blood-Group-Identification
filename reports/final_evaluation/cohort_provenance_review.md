# Cohort Provenance & Data Boundary Review
## Final Independent Evaluation Protocol Preparation

**Document Type:** Formal Provenance, Linkage, and Exposure Audit  
**Date:** 2026-10-16  
**Auditor:** Independent Technical & Scientific Auditor  
**Audit Boundary:** Metadata & Manifest Review Only — **Zero Protected Images Opened**  
**Execution Guard:** `TRAINING_AUTHORIZED = False`  

---

> [!IMPORTANT]
> ### METADATA-ONLY AUDIT DECLARATION
> This review was conducted strictly through static manifest parsing, filesystem metadata inspection, and historical documentation review. In strict accordance with the evaluation boundary:
> 1. No protected fingerprint image files in `data/mfs100_final` or `data/captured` were loaded, opened, or rendered.
> 2. No protected ground-truth labels were accessed or utilized for hyperparameter search, threshold calibration, or model selection.
> 3. Zero predictions on protected cohorts were generated.

---

## 1. Executive Summary & Provenance Matrix

The repository contains two distinct non-development cohorts designated for evaluation:
1. **The Locked 21-Participant Test Cohort ($N=21$, 63 scans):** Partitioned within `data/mfs100_final` as `split == 'TEST'`.
2. **The Separate 26-Participant Hospital Cohort ($N=26$, 78 scans):** Stored within `data/captured` with manifest in `data/mfs100_ground_truth.csv`.

### Cohort Comparison & Provenance Summary

| Provenance Attribute | Locked 21-Participant Cohort | Separate 26-Participant Hospital Cohort | Evidence Classification |
| :--- | :--- | :--- | :---: |
| **Participant Count ($N$)** | **21 unique participants** | **26 unique participants** | **Verified from source** |
| **Total Scans Evaluated** | **63 scans** (exactly 3 scans/participant) | **78 scans** (exactly 3 scans/participant) | **Verified from source** |
| **Storage Directory** | `data/mfs100_final/` | `data/captured/` | **Verified from source** |
| **Manifest / Source File** | `reports/phase3_participant_benchmark/participant_split.csv` | `data/mfs100_ground_truth.csv` | **Verified from source** |
| **Hardware Sensor** | Mantra MFS100 Optical Fingerprint Scanner | Mantra MFS100 Optical Fingerprint Scanner | **Verified from source** |
| **Acquisition Protocol** | Left Thumb, 3 impressions per participant | Left Thumb, 3 impressions per participant | **Verified from source** |
| **Native Dimensions** | $316 \times 354$ pixels, 8-bit Grayscale (`L`) | $316 \times 354$ pixels, 8-bit Grayscale (`L`) | **Verified from source** |
| **Ground-Truth Source** | Volunteer self-report / institutional registration record | Hospital Laboratory Serological Test Reports (25) + Verified Filename (1) | **Verified from source** |
| **Laboratory Serology Verification** | **NOT VERIFIED** (No physical serology lab sheets in repo) | **VERIFIED FROM SOURCE** (75/78 scans documented as Hospital Serology) | **Verified from source / NOT VERIFIED** |
| **Participant ID Namespace** | `P002`, `P0023`, `P0024`, `P0029`, `P003`... (`P` prefix) | `S001` through `S026` (`S` prefix) | **Verified from source** |
| **Cross-Cohort Identity Overlap** | **NOT VERIFIED** (Minutiae matching across namespaces unperformed) | **NOT VERIFIED** (Minutiae matching across namespaces unperformed) | **NOT VERIFIED** |
| **Historical Exposure Pre-EXP-01** | **EVALUATED HISTORICALLY** in Phase 3 against Baseline Model B | **EVALUATED HISTORICALLY** in Phase 3 against Baseline E2 / EfficientNet | **Reported historically** |
| **Quarantine During EXP-01–06D** | **100% QUARANTINED** (Zero access events throughout EXP-01–06D) | **100% QUARANTINED** (Zero access events throughout EXP-01–06D) | **Verified from source** |
| **Exposure to EXP-06C Config A** | **ZERO EXPOSURE** (Never evaluated on candidate) | **ZERO EXPOSURE** (Never evaluated on candidate) | **Verified from source** |
| **Legitimate Evaluation Role** | **Primary Endpoint:** Internal Held-Out Generalization Benchmark | **Secondary Endpoint:** External Domain-Shifted Serological Verification | **Verified from source** |

---

## 2. In-Depth Provenance Review: Locked 21-Participant Cohort

### 2.1 File Structure & Participant Identification
- **Manifest:** `reports/phase3_participant_benchmark/participant_split.csv` lists 303 total scans across 101 participants. *[Evidence: Verified from source — lines parsed directly]*
- **Partition Distribution:**
  - `TRAIN`: 192 scans (64 participants)
  - `VALIDATION`: 48 scans (16 participants)
  - `TEST`: 63 scans (21 participants)
- **Participant IDs in TEST Split:**
  `P002`, `P0023`, `P0024`, `P0029`, `P003`, `P0035`, `P0039`, `P0041`, `P0042`, `P0047`, `P0048`, `P0054`, `P0071`, `P0073`, `P0075`, `P0082`, `P0083`, `P0088`, `P009`, `P0093`, `P0095`. *[Evidence: Verified from source — exact list extracted]*
- **Class Breakdown in TEST Split:**
  - `A+`: 5 participants (15 scans)
  - `AB+`: 2 participants (6 scans)
  - `B+`: 6 participants (18 scans)
  - `O+`: 8 participants (24 scans)
  - *Total:* 21 participants (63 scans).
  - Always-O+ Trivial Baseline: $8 / 21 = 38.10\%$.

### 2.2 Label Provenance & Serological Status
- The ground-truth blood groups for `data/mfs100_final` were compiled during institutional volunteer collection.
- **Serological Status:** **NOT VERIFIED.** While records state blood groups were obtained during blood donation / health camp records, no scanned laboratory agglutination test reports or phlebotomy requisition logs are stored in the repository. Labels must be classified as **unverified institutional / donor self-reported**.
- **Impact on Claims:** This cohort cannot be used to prove that optical fingerprints can determine serologically validated blood groups. It can only demonstrate algorithmic generalization on held-out human subjects within the same institutional acquisition protocol.

### 2.3 Identity Linkage & Overlap Risks
- The 21 test participants share the exact same ID scheme (`P...`) as the 80 development participants (`TRAIN` and `VALIDATION`).
- The partition was performed via `StratifiedGroupKFold` in Phase 3.
- **Overlap Risk:** While participant IDs are disjoint, no automated biometric minutiae comparison (e.g. via SourceAFIS or Bozorth3) has been executed to prove that no single volunteer enrolled under two different IDs during separate collection sessions. Disjoint IDs provide nominal subject isolation, but **cannot be mathematically certified as biologically disjoint individuals**. *[Evidence: Stated as NOT VERIFIED]*

### 2.4 Historical Exposure Audit
- **Critical Audit Finding:** The 21-participant cohort was **NOT completely untouched** in earlier project history prior to the Accuracy Improvement research cycle.
- **Documented Historic Evaluation:** In Phase 3 (documented in `reports/external_evaluation/FOUR_CLASS_TEST_RESULTS.md` and `PROJECT_EXECUTIVE_SUMMARY.md`), the original Phase 3 Model B (E2 DenseNet-121 ensemble) was evaluated on these 21 participants:
  - Accuracy: **38.10%** (8 / 21; matched the Always-O+ majority baseline).
  - Balanced Accuracy: **30.62%**.
  - Macro F1: **0.2500**.
- **Quarantine During EXP-01 through EXP-06D:** Following Phase 3, this cohort was placed under strict quarantine (`reports/accuracy_improvement/data_split_audit.md`). Throughout the entire ten-experiment accuracy improvement campaign, zero access events occurred.
- **Candidate Exposure:** The candidate model (**EXP-06C Config A**) has **NEVER** been evaluated against this cohort.

---

## 3. In-Depth Provenance Review: Separate 26-Participant Hospital Cohort

### 3.1 File Structure & Participant Identification
- **Manifest:** `data/mfs100_ground_truth.csv` lists 78 records across 26 unique subjects (`S001` through `S026`). *[Evidence: Verified from source — parsed directly]*
- **Directory:** Images reside in `data/captured/`.
- **Participant IDs:** `S001` to `S026`.
- **Scans per Participant:** Exactly 3 scans per participant (78 scans total).
- **Class Breakdown:**
  - `A+`: 4 participants (12 scans)
  - `AB+`: 3 participants (9 scans)
  - `B+`: 8 participants (24 scans)
  - `O+`: 11 participants (33 scans)
  - *Total:* 26 participants (78 scans).
  - Always-O+ Trivial Baseline: $11 / 26 = 42.31\%$.

### 3.2 Label Provenance & Serological Status
- In `data/mfs100_ground_truth.csv`, column `confirmation_source` explicitly documents:
  - 75 scans (25 participants): **"Hospital Serological Test"**
  - 3 scans (1 participant, `S003`): **"Independent ground truth supplied from verified filename"**
- **Serological Status:** **VERIFIED FROM SOURCE (METADATA).** 25 of the 26 participants have documented laboratory serological confirmation, representing the highest ground-truth label fidelity in the entire repository.
- **Impact on Claims:** This cohort represents a genuine external test of whether the model's representations correlate with clinical serology under real-world hospital acquisition.

### 3.3 Identity Linkage & Cross-Cohort Overlap Risks
- The 26 participants utilize an independent ID prefix (`S001`–`S026`).
- **Overlap Risk:** Because both `data/mfs100_final` and `data/captured` were gathered using Mantra MFS100 hardware in Bangalore/Karnataka, there is an unquantified risk that one or more individuals who participated in the 101-volunteer cohort also participated in the hospital capture. No biometric de-duplication across the two cohorts has been performed. Therefore, physical subject disjointness is **NOT VERIFIED**.

### 3.4 Historical Exposure Audit
- **Documented Historic Evaluation:** This cohort was previously evaluated during Phase 3 baseline testing (documented in `reports/efficientnet_e2_ensemble_experiment/separate_26_cohort_metrics.json`):
  - Baseline E2 Fusion: **19.23%** Accuracy (5/26), **0.1548** Macro F1, **20.17%** Balanced Accuracy.
  - EfficientNet-B0: **38.46%** Accuracy (10/26), **0.2788** Macro F1, **29.64%** Balanced Accuracy.
  - Equal-Weight Ensemble: **38.46%** Accuracy (10/26), **0.2778** Macro F1, **31.53%** Balanced Accuracy.
- **Quarantine During EXP-01 through EXP-06D:** Placed under quarantine with 0 access events.
- **Candidate Exposure:** Zero exposure to EXP-06C Config A or any EXP-06 model.

---

## 4. Legitimate Scope of Scientific Claims

Based on the verified provenance, the legitimate scientific scope of each cohort is strictly delimited:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LEGITIMATE SCOPE OF CLAIMS                            │
├──────────────────────────────────────┬───────────────────────────────────────┤
│    Locked 21-Participant Cohort      │  Separate 26-Participant Hospital     │
│       (Primary Test Endpoint)        │      (Secondary External Endpoint)    │
├──────────────────────────────────────┼───────────────────────────────────────┤
│ • Tests held-out generalization on   │ • Tests external generalization under │
│   unseen participants under          │   domain-shifted hospital acquisition │
│   identical sensor conditions.       │ • Evaluates correlation with serology │
│ • Can prove whether EXP-06C Config A │ • Can test whether the model collapses│
│   exceeds baseline E2 (38.10%).      │   under hospital lighting/pressure.   │
│ • CANNOT prove clinical serological  │ • CANNOT be pooled with the 21-cohort │
│   validity (labels are unverified).  │   due to label & sensor differences.  │
└──────────────────────────────────────┴───────────────────────────────────────┘
```

> [!CAUTION]
> ### PROHIBITION ON AUTOMATIC POOLING
> Under no circumstances may the 21-participant locked cohort and the 26-participant hospital cohort be pooled into a single $N=47$ test sample. They possess fundamentally distinct label-verification standards (unverified institutional donor records vs. clinical hospital serology) and distinct acquisition settings. All metrics must be computed, reported, and analyzed **strictly separately**.

---

## 5. Audit Confirmation

- **No Protected Images Opened:** Verified.
- **No Protected Labels Used for Tuning:** Verified.
- **Historical Exposure Acknowledged:** Verified. Neither cohort is described as "never before evaluated" in the broader project lifecycle; their prior exposure to obsolete Phase 3 baselines is fully disclosed.
- **Quarantine Integrity Maintained:** Verified. Both cohorts remain quarantined.
