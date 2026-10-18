# Leakage Root-Cause Analysis & Code Forensic Audit

**Audit Date:** 2026-10-17  
**Scope:** Deep-dive code analysis into data flow, participant splitting firewalls, checkpoint-selection leakage, and optimization bias across development experiments.

---

## 1. Architectural Data Flow & Checkpoint Selection Pipeline

The diagram below traces the exact training, evaluation, and checkpointing workflow used during EXP-03 through EXP-06C:

```
[MFS100 Primary Cohort: 101 Participants (303 Scans)]
                      │
     ┌────────────────┴────────────────┐
     ▼                                 ▼
[Dev-80 Cohort]             [Locked Test: 21 Participants]
(80 Participants)           (63 Scans - Evaluated 17 times)
     │
     ▼ 5-Fold Stratified Group Split (by Participant ID)
┌────────────────────────────────────────────────────────┐
│ For Fold k (1 to 5):                                   │
│   ├── Training Set: 64 Participants (192 scans)        │
│   └── Outer Validation Fold: 16 Participants (48 scans)│
│                                                        │
│   Forward Pass: 3 Scans -> DenseNet-121 -> Concat      │
│   Loss: Categorical Cross-Entropy on Train Set         │
│   Epoch End Callback: `MacroF1Checkpoint`              │
│       │                                                │
│       ├── Evaluates 16 Outer Val Participants          │
│       └── IF val_macro_f1 > best_val_f1:               │
│               SAVE CHECKPOINT (Epoch e*)               │
│                                                        │
│   Final Fold Metric: Report metric at Epoch e*         │
│   ==> SELECTION LEAKAGE OCCURS HERE!                   │
└────────────────────────────────────────────────────────┘
```

---

## 2. The Checkpoint Selection Leakage Mechanism

### Source Code Vulnerability
In `experiments/exp06c/run_exp06c.py` (and inherited from EXP-03, 04, 05):
```python
checkpoint_callback = MacroF1Checkpoint(
    filepath=os.path.join(fold_dir, "best_model.keras"),
    monitor="val_macro_f1",
    mode="max",
    save_best_only=True
)
```
Each fold trained for 100 to 150 epochs. At every single epoch, the 16 held-out validation participants were evaluated. The weights that yielded the maximum `val_macro_f1` were saved as the representative model for that fold.

### Why This Is Flawed (Selection Optimism)
1. **Sample Size per Validation Fold:** $N = 16$ participants (48 scans).
2. **Number of Opportunities to Overfit Noise:** Over 100 epochs, the callback evaluated 100 stochastic realizations of the classification head.
3. In a 4-class problem with only 16 samples, random fluctuations routinely produce temporary spikes where 9 or 10 participants happen to be classified correctly purely by chance ($9/16 = 56.25\%$).
4. The callback deliberately captured this peak noise and saved it as "the model," reporting the peak score as the fold's true out-of-fold generalization.
5. The participants used to *select* the model were the exact same participants used to *score* the model.

---

## 3. Empirical Proof: The Selection-Inclusive Null Experiment

To rigorously prove that the reported 53.75% validation accuracy was an artifact of checkpoint selection rather than biometric ridge learning, a 100-repeat selection-inclusive permutation test was executed in `reports/leakage_audit/`:

1. **Protocol:** Participant labels on the Dev-80 cohort were completely randomized (breaking any possible biological relationship between fingerprints and blood types).
2. **Procedure:** The exact EXP-06C Config A training pipeline and `MacroF1Checkpoint` callback were run 100 times on these pure-noise labels.
3. **Results:**
   - **Mean Validation Accuracy on Pure Noise:** **$46.77\% \pm 4.12\%$**
   - **Mean Validation Macro F1 on Pure Noise:** **$0.4277 \pm 0.0435$**
   - **Maximum Validation Accuracy Observed on Pure Noise:** **$56.25\%$**

### Comparison Table:
| Condition | Reported Val Accuracy | Real Unbiased Val Accuracy |
| :--- | :---: | :---: |
| EXP-06C Config A (True Labels, Selection Leakage) | 53.75% | 31.67% - 36.25% |
| Shuffled Labels (Noise, Selection Leakage) | 46.77% | 25.00% |
| EXP-06C Config A (Fixed 100 Epochs, No Selection) | 36.25% | 36.25% |
| EXP-06C Config A (Nested Inner-Val Split) | 31.67% | 31.67% |

**Scientific Conclusion:** Checkpoint selection on held-out validation folds inflated accuracy by approximately **$+17\%$ to $+20\%$**. When an honest evaluation protocol is enforced (nested inner validation or fixed training epochs), the true development accuracy of EXP-06C is between **31.67% and 36.25%**, which perfectly aligns with the 33.33% obtained on the locked test cohort.

---

## 4. Cross-Fold Contamination & Preprocessing Leakage Audit

1. **Cross-Fold Participant Contamination:**
   - *Audit finding:* **Negative (Clean).** The `GroupKFold` split on `participant_id` strictly prevented any participant's scans from appearing simultaneously in both train and validation folds within the Dev-80 cohort.
2. **Preprocessing Parameter Leakage:**
   - *Audit finding:* **Negative (Clean).** Preprocessing filters (C5 CLAHE, normalization, Lanczos resizing) operated strictly per-image or per-scan without computing global dataset-level mean/variance statistics. No test distribution statistics leaked into training.
3. **Data Augmentation Leakage:**
   - *Audit finding:* **Negative (Clean).** Augmentations were applied online strictly to the training generator and were completely deactivated during validation and test inference.
4. **Architectural Selection Contamination:**
   - *Audit finding:* **Positive (Contaminated).** The decision to freeze DenseNet-121 and adopt 3-scan concatenation (E2 architecture) was influenced by inspecting the 21-participant locked test set across 10 evaluation sessions prior to the formal lock.
