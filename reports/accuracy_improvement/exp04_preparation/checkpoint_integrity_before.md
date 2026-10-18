# EXP-04: Pre-Training Checkpoint & Production Integrity Baseline

**Audit Date:** 2026-10-14T18:21:30+05:30  
**Scope:** Verification and SHA-256 Baseline Recording of Original Production & EXP-03 Checkpoints Prior to EXP-04  
**Audit Finding:** **PASS — ALL PREVIOUS CHECKPOINTS BASELINED AND PROTECTED**  

---

## 1. Executive Summary

Before any code for EXP-04 is executed, this audit establishes the immutable baseline of all existing models and weights in the repository. EXP-04 operates under strict non-destructive isolation:
1. No production model or checkpoint may be modified.
2. Completed EXP-03 artifacts and checkpoints must remain untouched.
3. All EXP-04 scripts, checkpoints, predictions, and reports must reside in dedicated directories (`experiments/exp04/` and `reports/accuracy_improvement/exp04/`).

---

## 2. Original E2 Production Checkpoints Baseline

Target Directory: `models/phase3_fusion_metric_learning/E2/`

| Checkpoint File | File Size (Bytes) | Pre-EXP-04 SHA-256 Checksum | Immutability Status |
|:---|:---:|:---:|:---:|
| `fold_1_best.weights.h5` | 52,910,112 | `7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d` | **PROTECTED / LOCKED** |
| `fold_2_best.weights.h5` | 52,910,112 | `9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede` | **PROTECTED / LOCKED** |
| `fold_3_best.weights.h5` | 52,910,112 | `8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3` | **PROTECTED / LOCKED** |
| `fold_4_best.weights.h5` | 52,910,112 | `cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519` | **PROTECTED / LOCKED** |
| `fold_5_best.weights.h5` | 52,910,112 | `aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d` | **PROTECTED / LOCKED** |

---

## 3. Production Deployment Model

Target File: `models/abo_positive_4class/best_model.keras`
- File Size: 61,023,607 bytes
- Immutability Status: **PROTECTED / LOCKED**. This model will not be re-trained, fine-tuned, or replaced.

---

## 4. Completed EXP-03 Reference Checkpoints Baseline

Target Directory: `experiments/exp03/checkpoints/arm1/` (Leading Development Candidate)

| EXP-03 Checkpoint | File Size (Bytes) | Pre-EXP-04 SHA-256 Checksum | Status |
|:---|:---:|:---:|:---:|
| `arm1/fold_1_best.weights.h5` | 52,910,112 | `0d047b7a10153718b067e84a27a6ad2bc0486ccf1d35057a07da6e2e2d627ac5` | **REFERENCE / LOCKED** |
| `arm1/fold_2_best.weights.h5` | 52,910,112 | `dd94e4c44887399d700ac2de195f513a5b1db73ac561f4ff8c2345d5156f7063` | **REFERENCE / LOCKED** |
| `arm1/fold_3_best.weights.h5` | 52,910,112 | `f763f67068a4411dbd4a405e39248f2ead1b1b1fe228c4b1bca5da807005b3e3` | **REFERENCE / LOCKED** |
| `arm1/fold_4_best.weights.h5` | 52,910,112 | `589f1e52275f68ed02329b1d4a7a9c16f406ed3868b8afeef9706ebace61b496` | **REFERENCE / LOCKED** |
| `arm1/fold_5_best.weights.h5` | 52,910,112 | `848aea5498e469871d5fdcd136d849f971c9c3c485a4c66afb2505d3b0a46776` | **REFERENCE / LOCKED** |

---

## 5. Storage Isolation Rules for EXP-04

1. **Scripts:** Stored exclusively in `experiments/exp04/` (e.g. `experiments/exp04/run_exp04.py`).
2. **Checkpoints:** Stored exclusively in `experiments/exp04/checkpoints/{config_id}/fold_{fold}_best.weights.h5`.
3. **Reports & OOF Predictions:** Stored exclusively in `reports/accuracy_improvement/exp04/`.
4. **Post-Execution Audit:** Pre-experiment hashes will be compared against post-experiment hashes to guarantee zero contamination.

---

## 6. Audit Verdict

All production and EXP-03 checkpoints are baselined, recorded, and protected.
- **Audit Verdict:** **PASS**.
