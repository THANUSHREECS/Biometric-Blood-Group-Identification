# EXP-03 Verification: Fold-Level Training & Early Stopping Audit

**Audit Date:** 2026-10-14T18:12:00+05:30  
**Target:** 15 Training Runs (3 Arms $\times$ 5 Folds) in EXP-03  
**Status:** **VERIFIED PASS**  

---

## 1. Executive Summary

This audit addresses the question of why several folds recorded a "best epoch" below epoch 15 (e.g., Arm 0 Fold 3 at epoch 11; Arm 2 Fold 3 at epoch 5) despite the protocol mandating `MIN_EPOCHS = 100`.

### Key Findings:
1. **Zero Premature Terminations:** Every fold across all three arms executed for **at least 100 epochs** (14 folds executed for exactly 100 epochs; Arm 0 Fold 5 executed for 121 epochs).
2. **Mechanics of Best Epoch vs. Training Horizon:** In validation monitoring callbacks, `best_epoch` records the epoch at which the highest validation metric was achieved. Training was strictly forced to continue until at least epoch 100. If no subsequent epoch outperformed that early peak, the callback retained the weights from the early peak while running all 100 epochs.
3. **Weight Restoration Integrity:** At the conclusion of training for each fold, `model.load_weights(str(ckpt_path))` reloaded the weights from `best_epoch` prior to generating out-of-fold validation predictions.

---

## 2. Callback Architecture & Gating Logic

The callback `MacroF1Checkpoint` (`experiments/exp03/run_exp03.py` lines 219–262) implemented the following logic:

```python
class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, X_val, y_val, patience=PATIENCE, min_epochs=MIN_EPOCHS):
        super().__init__()
        self.filepath = filepath
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience        # 15
        self.min_epochs = min_epochs    # 100
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        clf_preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        y_pred = np.argmax(clf_preds, axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)

        # Weight Saving Rule: Only updates if validation macro F1 strictly improves
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.best_epoch = epoch + 1
            self.wait = 0
            self.model.save_weights(self.filepath)
        else:
            self.wait += 1

        # Early Stopping Gate: STRICTLY BLOCKED before min_epochs
        if (epoch + 1) >= self.min_epochs and self.wait >= self.patience:
            self.model.stop_training = True
```

### Explanation of Callback Behavior:
1. When a fold achieves its maximum validation macro F1 early (e.g., at epoch 11 in Arm 0 Fold 3), `self.best_f1 = 0.3556`, `self.best_epoch = 11`, and the weights are saved to disk.
2. In all subsequent epochs (epochs 12 through 100), the validation performance fluctuates below `0.3556`. Consequently, `self.wait` increments by 1 on every epoch, eventually reaching `wait = 89`.
3. In a standard Keras `EarlyStopping` callback without a floor, training would have terminated after 15 epochs of no improvement (at epoch 26).
4. However, the compound condition `(epoch + 1) >= self.min_epochs and self.wait >= self.patience` evaluates to `False` on every epoch where `epoch + 1 < 100`.
5. Early stopping is **strictly gated**. The model is compelled to train through epoch 100.
6. At epoch 100, `(100 >= 100)` is `True` and `(89 >= 15)` is `True`, so training halts at epoch 100.
7. Post-training, `model.load_weights(str(ckpt_path))` loads the best weights preserved from epoch 11.

---

## 3. Comprehensive 15-Fold Verification Table

Extracted directly from execution process logs (`task-2046.log`) and saved training histories:

| Arm | Fold | Total Epochs Trained | Minimum Floor Satisfied? | Best Epoch Recorded | Best Val Macro F1 | Fold Runtime (s) | Premature Stop? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm 0** | Fold 1 | **100** | YES ($\ge 100$) | Epoch 13 | 0.3143 | 244.7s | **NO** |
| **Arm 0** | Fold 2 | **100** | YES ($\ge 100$) | Epoch 18 | 0.4000 | 616.7s | **NO** |
| **Arm 0** | Fold 3 | **100** | YES ($\ge 100$) | Epoch 11 | 0.3556 | 442.8s | **NO** |
| **Arm 0** | Fold 4 | **100** | YES ($\ge 100$) | Epoch 80 | 0.5000 | 465.5s | **NO** |
| **Arm 0** | Fold 5 | **121** | YES ($\ge 100$) | Epoch 106 | 0.3393 | 1680.6s | **NO** |
| **Arm 1** | Fold 1 | **100** | YES ($\ge 100$) | Epoch 27 | 0.3917 | 1082.9s | **NO** |
| **Arm 1** | Fold 2 | **100** | YES ($\ge 100$) | Epoch 27 | 0.5637 | 1017.3s | **NO** |
| **Arm 1** | Fold 3 | **100** | YES ($\ge 100$) | Epoch 21 | 0.2467 | 685.9s | **NO** |
| **Arm 1** | Fold 4 | **100** | YES ($\ge 100$) | Epoch 23 | 0.6230 | 745.8s | **NO** |
| **Arm 1** | Fold 5 | **100** | YES ($\ge 100$) | Epoch 35 | 0.5071 | 754.6s | **NO** |
| **Arm 2** | Fold 1 | **100** | YES ($\ge 100$) | Epoch 38 | 0.3293 | 842.4s | **NO** |
| **Arm 2** | Fold 2 | **100** | YES ($\ge 100$) | Epoch 14 | 0.3833 | 1144.9s | **NO** |
| **Arm 2** | Fold 3 | **100** | YES ($\ge 100$) | Epoch 5 | 0.2889 | 1281.1s | **NO** |
| **Arm 2** | Fold 4 | **100** | YES ($\ge 100$) | Epoch 52 | 0.5000 | 1302.6s | **NO** |
| **Arm 2** | Fold 5 | **100** | YES ($\ge 100$) | Epoch 11 | 0.3051 | 1359.6s | **NO** |

---

## 4. Special Case Analysis: Arm 0 Fold 5 (121 Epochs)

Arm 0 Fold 5 trained for **121 epochs**, which is greater than the 100-epoch floor:
- During epochs 1 through 100, the best validation macro F1 was updated periodically.
- At **Epoch 106**, validation macro F1 reached a new maximum of `0.3393`.
- This reset `self.wait = 0` at epoch 106.
- The callback then allowed training to proceed for another 15 epochs without improvement ($106 + 15 = 121$).
- Training terminated at epoch 121, having fully explored the post-100 trajectory.
- Checkpoint restored: Epoch 106 weights.

---

## 5. Audit Conclusion

The training horizon of EXP-03 was executed with complete fidelity to the experimental protocol:
- Every fold trained for at least 100 epochs.
- No fold suffered premature early termination.
- Checkpoints loaded for OOF prediction were verified to be the true mathematical peaks on validation macro F1.
- **Audit Verdict:** **PASS**.
