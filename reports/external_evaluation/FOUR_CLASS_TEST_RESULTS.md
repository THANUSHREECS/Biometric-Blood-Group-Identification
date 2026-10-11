# FOUR-CLASS TEST RESULTS (FINAL VERIFICATION)

## 1. SCIENTIFIC REPORTING
The active Flask application functions as a faithful inference wrapper around the **Model B: E2 DenseNet121 Ensemble** (deployed). The application preserves the frozen Phase 3 evaluation metrics. 

**Deployed Model Results (Model B: E2 DenseNet121 5-fold ensemble, 21 unseen test participants):**
- **Architecture:** 3-scan feature fusion ensemble (Lanczos 112×112, raw float32 [0, 255])
- **Accuracy:** 38.10% (8/21; matches always-O+ majority baseline)
- **Macro F1 Score:** 0.2500
- **Weighted F1:** 0.3274
- **Balanced Accuracy:** 0.3062

**Comparative Research Baseline (Model A: EfficientNetB0, not deployed):**
- **Dataset / Test:** 415 test images (random image split on public dataset, single scan 224×224)
- **Accuracy:** 86.02% on a random split; subject independence unverifiable; shortcut features possible. Standalone research model; not deployed in application.

## 2. SCIENTIFIC CONCLUSION
Based on the failure of the model to generalize beyond its training set while using strict participant-level splitting and authentic MFS100 optical captures:

**The tested system did not demonstrate reliable four-class blood group classification on unseen participants.**

The integration of the model into this application interface does not alter this scientific conclusion, nor does it provide clinical readiness or medical validity.

## 3. INFERENCE VALIDATION
Through the `test_integration.py` unit testing:
- **Model Load:** E2 checkpoint weights map cleanly without unexpected keys.
- **Data Tensor:** `predict_consensus` dynamically stacks three separate $(112 \times 112 \times 3)$ arrays into the expected $(1, 3, 112, 112, 3)$ input shape for inference.
- **Probabilities:** The unweighted averaging across the 5 models outputs exactly 4 float values that correctly sum to $~1.0$.

There is no data leakage, modification, or test-set peeking performed by the application framework. The reported test accuracy holds absolute.
