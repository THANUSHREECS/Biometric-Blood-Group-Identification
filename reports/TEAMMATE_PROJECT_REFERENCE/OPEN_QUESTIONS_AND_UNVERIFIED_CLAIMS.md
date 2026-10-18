# Open Questions & Unverified Claims
**Project:** HemoPrint AI — Biometric Blood Group Detection Research System  
**Audience:** Project Teammates, Project Lead, and Faculty Supervisor  
**Purpose:** Actionable checklist of items requiring empirical confirmation prior to research paper and final project submission  
**Status:** Audit Identified — Confirmation Required

---

## Executive Summary

While the computational codebase, model weights, experiment outputs, and software application have been thoroughly audited, several external questions—primarily concerning **dataset provenance, ethical approvals, cohort isolation, and institutional governance**—cannot be resolved solely through static code inspection.

This document itemizes every unresolved question, explains its scientific significance, identifies the missing documentation, and assigns the responsible team role.

---

## 1. Itemized Open Questions & Unverified Items

### Question 1: Formal Institutional Ethics (IRB) Approval & Protocol Number
* **Current Status:** **UNVERIFIED — Confirmation Required.**
* **What the Codebase Shows:** The software application strictly enforces participant consent via [`tests/test_research_integration.py:31-57`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/tests/test_research_integration.py#L31-L57) and anonymizes participants with pseudonyms (`P-0001` to `P-0080`). However, no formal Institutional Review Board (IRB) or Institutional Ethics Committee (IEC) clearance certificate or protocol approval number was found in the project files.
* **Why It Matters:** Academic journals and IEEE/Springer conferences require an explicit Institutional Ethics statement and protocol ID whenever human biometric data is collected.
* **Action Required:** Teammates must locate and cite the official Institutional Ethics Approval Letter from the institution (e.g., Don Bosco Institute of Technology Ethics Committee).
* **Responsible Role:** Faculty Supervisor / Lead Author.

---

### Question 2: Serological Verification of Ground-Truth Blood Group Labels
* **Current Status:** **UNVERIFIED — Known Limitation.**
* **What the Codebase Shows:** Development-80 participant records in [`reports/phase3_participant_benchmark/participant_split.csv`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/phase3_participant_benchmark/participant_split.csv) are labeled as "self-reported student survey responses."
* **Why It Matters:** Clinical literature indicates that self-reported blood group recall carries an error rate of 5%–10% in student cohorts. If a participant misremembers their blood group as O+ when they are actually B+, supervised classifiers are penalized for correct biological associations or forced to learn noise.
* **Action Required:** Teammates must explicitly state in the paper's *Threats to Validity / Limitations* section that ground-truth labels were self-reported rather than serologically confirmed via forward/reverse hemagglutination.
* **Responsible Role:** Paper Co-Authors (Methodology Section).

---

### Question 3: Identity Overlap Across Cohorts (Dev-80, Test-21, Hospital-26)
* **Current Status:** **UNVERIFIED — Strict Partitioning Assumed.**
* **What the Codebase Shows:**
  * Development Cohort: 80 participants (`P-0001` to `P-0080`).
  * Test-21 Cohort: 21 participants (`S001` to `S021`).
  * Hospital Cohort: 26 participants in `data/quarantine/` and `data/mfs100_final/`.
* **The Missing Evidence:** No automated biometric 1:1 minutiae matcher was executed across these cohorts to mathematically prove that no student participant in Dev-80 was also scanned in Test-21 or Hospital-26.
* **Action Required:** If claims of "completely independent cohort validation" are made in the manuscript, teammates should verify with the data collection team whether any participant was enrolled in multiple collection sessions.
* **Responsible Role:** Data Collection Coordinator.

---

### Question 4: Original Provenance of `dataset_blood_group/` (6,000 Images)
* **Current Status:** **UNVERIFIED — External Dataset Origin.**
* **What the Codebase Shows:** The directory `dataset_blood_group/` contains 6,000 images distributed across 8 blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-). Image names follow patterns like `A+_1.jpg`, `A+_2.jpg`.
* **The Missing Evidence:** It is undocumented whether this folder represents the public Sokoto Blood Group dataset, a Kaggle mirror, or a synthetically augmented image collection.
* **Action Required:** Teammates must cite the exact academic source, DOI, or Kaggle repository URL for `dataset_blood_group/` in the manuscript's *Prior Art / Baseline Datasets* section.
* **Responsible Role:** Primary Researcher / Literature Reviewer.

---

### Question 5: Discrepancies in Legacy EXP-08 Baseline Code
* **Current Status:** **DOCUMENTED & RESOLVED BY AUDIT — Confirmation Required in Text.**
* **What the Codebase Shows:**
  * In the original `experiments/exp08_dermatoglyphic_minutiae/train.py`, the baseline was calculated per-fold without global aggregation, yielding a reported baseline of **28.75%**, and class labels for B+ and AB+ were inverted.
  * The scientific audit in [`reports/accuracy_improvement_audit/EXP08_FINAL_CONSISTENCY_AUDIT.md`](file:///c:/Users/thanu/OneDrive%20-%20Don%20Bosco%20Institute%20of%20Technology/major%20project/reports/accuracy_improvement_audit/EXP08_FINAL_CONSISTENCY_AUDIT.md) recomputed the true global majority baseline as **36.25%** and verified that Random Forest (35.0%) falls within the chance null distribution ($p = 0.2537$).
* **Action Required:** Teammates must ensure that the paper uses the **corrected 36.25% baseline** and the reconciled numbers from `EXP08_FINAL_RESULTS.md`, rather than legacy claims from outdated drafts.
* **Responsible Role:** Manuscript Lead Author.

---

### Question 6: Dual Backend Server Architecture Purpose
* **Current Status:** **DOCUMENTED IN CODEBASE — Team Consensus Recommended.**
* **What the Codebase Shows:** The repository contains two backend files:
  1. `app.py` (Root directory, Flask, port 5000): Serves the active web application, connected to `data/blood_group.db`, tested by `tests/test_research_integration.py`.
  2. `live_app/backend/main.py` (FastAPI, port 8000): An alternative or earlier API harness that exposes architecture benchmarking endpoints.
* **Action Required:** Teammates should decide whether the research paper describes the unified Flask architecture (`app.py`) as the primary system prototype and relegates the FastAPI harness to an experimental utility script.
* **Responsible Role:** Software Systems Author.

---

### Question 7: Publication Metadata & Author Order
* **Current Status:** **UNVERIFIED — Administrative Action Required.**
* **Missing Details:**
  * Complete list of student team member names and student institutional roll numbers.
  * Supervisor / Principal Investigator name, department, and institutional email.
  * Target publication venue (e.g., IEEE Access, Springer Nature Computer Science, International Conference on Biometrics, or Departmental Capstone Proceedings).
  * Funding acknowledgments or institutional project grant codes.
* **Action Required:** Teammates must formalize the title page metadata before compiling the LaTeX or Word submission.
* **Responsible Role:** Entire Project Team.

---

## 2. Summary Action Matrix for Teammates

| Item | Topic | Impact on Paper | Recommended Resolution |
| :---: | :--- | :--- | :--- |
| **Q1** | Ethics / IRB Protocol | Critical for submission | Obtain official IRB approval letter or state institutional capstone review status. |
| **Q2** | Label Noise | High (Scientific validity) | Add explicit caveat: "Ground truth was established via self-reported donor history." |
| **Q3** | Cohort Overlap | High (Generalization claim) | Document cohorts as "separate acquisition sessions" rather than unverified genetic independence. |
| **Q4** | Dataset Source | Moderate (Citation needed) | Add formal citation for `dataset_blood_group` (Sokoto / Kaggle source). |
| **Q5** | EXP-08 Baseline | Moderate (Consistency) | Use 36.25% global majority baseline from `EXP08_FINAL_RESULTS.md`. |
| **Q6** | Dual Backend | Low (Technical documentation) | Present `app.py` as production engine; document `main.py` as benchmark harness. |
| **Q7** | Author Metadata | Administrative | Fill in institutional affiliations and project guide details. |
