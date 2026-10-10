# EXPERIMENT 8 FEATURE AUDIT

## STATUS: PASS

**1. Backbone Configuration:**
- Used: `tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(112,112,3))`
- State: `base.trainable = False`. The backbone was rigorously frozen.

**2. Preprocessing Configuration:**
- Used: `transform_c5` consisting of composite alpha, foreground crop, ridge contrast enhancement, and resize to $112 \times 112$. This identically mirrors the frozen C5 pipeline from previous experiments.

**3. Feature Extraction Layer:**
- Output of `GlobalAveragePooling2D()` yielding a raw 1024-dimensional continuous vector per scan. 

**4. Representations:**
- **A. SINGLE-SCAN:** 1024-d vectors extracted and classified individually. Predictions were then grouped by participant and averaged via probability mean before computing `argmax`.
- **B. THREE-SCAN MEAN:** 1024-d vectors were grouped by participant and mathematically averaged (`np.mean(..., axis=0)`) to create one fixed 1024-d representation per participant prior to any classification or scaling. 
- No arbitrary concatenation (3072-d) occurred in this experiment.
