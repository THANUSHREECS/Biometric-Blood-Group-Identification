# DATASET AND SPLIT AUDIT

## 1. MFS100 FINAL DATASET VERIFICATION
A programmatic audit script (`scratch/audit_mfs100_dataset.py`) verified the physical files and the manifest (`participant_split.csv`).

- **Total Physical Scans:** 303 (Verified matching the manifest exactly)
- **Total Unique Participants:** 101
- **Scans per Participant:** Exactly 3 for all 101 participants.

**Class Distribution (Participants):**
- **O+:** 37
- **B+:** 31
- **A+:** 24
- **AB+:** 9

## 2. PARTICIPANT-LEVEL SPLIT VERIFICATION
- **TRAIN Participants:** 64
- **VALIDATION Participants:** 16
- **TEST Participants (Locked):** 21

**Integrity Checks:**
- **Participant Leakage:** **PASS.** Set intersection between Train, Validation, and Test participant IDs is strictly empty. No participant crosses partitions.
- **Image Grouping:** **PASS.** All 3 scans for any given participant remain strictly grouped in the same partition.
- **Duplicate Images:** **PASS.** MD5 hashing of all 303 physical images revealed 0 exact duplicates.

## 3. CONCLUSION
The `mfs100_final` dataset and its 64/16/21 split are structurally perfect. There is no data leakage, no participant contamination, and the counts precisely match the intended targets. The generalization failure (Exp 7) was *not* caused by a flawed test split.
