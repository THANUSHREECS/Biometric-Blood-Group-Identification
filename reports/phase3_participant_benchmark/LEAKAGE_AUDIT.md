# LEAKAGE AUDIT

Status: PASS

- No participant appears in more than one split.
- No exact image appears in more than one split.
- All 3 scans from each participant are in one split.
- No test participant appears in train or validation.

## Participant Counts
TRAIN participants: 64
VALIDATION participants: 16
TEST participants: 21

## Per-Class Participant Counts
TRAIN: {'O+': 23, 'B+': 20, 'A+': 15, 'AB+': 6}
VALIDATION: {'O+': 6, 'B+': 5, 'A+': 4, 'AB+': 1}
TEST: {'O+': 8, 'B+': 6, 'A+': 5, 'AB+': 2}
