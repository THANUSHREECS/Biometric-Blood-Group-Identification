# EXPERIMENT 9 PROTOCOL AUDIT

## STATUS: PASS

**1. Data Isolation:** 
- The script securely loads only the 80 development participants from `participant_split.csv`.
- The 21 locked test participants were mathematically excluded from the runtime.

**2. Cross-Validation Conformity:**
- Exactly the same `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` procedure was used on the participant identifiers.
- Nested Cross-Validation (`n_splits=2`) was applied rigorously on the inner training folds to tune the regularization constant $C$ without leaking validation data.

**3. Participant Alignment:**
- Every participant remained fully grouped. The 3 scans for each individual were mathematically pooled (mean) at the feature-extraction stage *before* entering the cross-validation splits. No scan was ever detached from its parent identity.
