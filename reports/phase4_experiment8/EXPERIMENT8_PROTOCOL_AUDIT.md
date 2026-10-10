# EXPERIMENT 8 PROTOCOL AUDIT

## STATUS: PASS

**1. Data Isolation:** 
- Only the 80 development participants were loaded from `participant_split.csv` via the filter `split.isin(["TRAIN", "VALIDATION"])`. 
- The 21 locked test participants were never loaded, never evaluated, and never used to tune hyperparameters.

**2. Cross-Validation Implementation:**
- The script securely utilizes `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped rigidly by `participant_id`.
- **Participant Grouping:** Every participant belongs to exactly one validation fold, and all 3 of their scans remain together in the single scan evaluation (Rep A) or are mean-pooled beforehand (Rep B).
- **Leakage Prevention:** `GridSearchCV` inside the outer training fold uses `StratifiedGroupKFold(n_splits=2)` purely on the `X_tr` data, guaranteeing hyperparameters $C$ are selected without ever seeing the outer validation fold.

**3. Alignment with Previous Protocols:**
- The 5 folds precisely map to the exact same assignments used in Experiment 4 and Experiment 6, verified by the identical random seed (42) and identical array grouping.
