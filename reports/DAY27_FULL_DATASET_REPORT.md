# DAY 27 FULL DATASET EXPERIMENT REPORT

## Objective
The experiment evaluates whether increasing source-domain training coverage (using the full ~6000 image dataset) improves zero-shot transfer to MFS100.

## Outcomes & Analysis

### 1. Did using the complete positive dataset improve MFS100?
No. Performance on the MFS100 evaluation set dropped from the historical baseline of 51.28% to 46.15%.

### 2. Did using all 8 blood groups improve MFS100?
No. Including the negative blood groups further degraded MFS100 performance down to 34.62%.

### 3. Which model performed better on MFS100?
Model A (the 4-class positive subset model) performed better (46.15%) than Model B (the 8-class model, 34.62%).

### 4. Did A+ recall improve?
No. A+ recall remained at 0.00% for both models.

### 5. Did B+ recall improve?
No meaningful improvement. B+ recall was 12.50% for Model A and 0.00% for Model B.

### 6. Did AB+ recall improve?
No. AB+ recall remained at 0.00% for both models.

### 7. Did O+ dominance decrease?
No. O+ prediction dominance remained extreme, with Model A predicting O+ for 92.31% of the MFS100 images.

### 8. Did the 8-class model incorrectly predict MFS100 samples as negative groups?
Yes. Model B incorrectly predicted 29 out of the 78 positive MFS100 samples as negative blood groups (A-: 7, B-: 3, AB-: 11, O-: 8).

### 9. Did canonical performance change?
Yes. On the canonical source-domain test set, Model A achieved 87.89% accuracy, while Model B achieved 72.44% accuracy. The high canonical performance alongside poor MFS100 transfer strongly indicates domain-specific overfitting.

### 10. Are any apparent improvements large enough to be meaningful given the 78-image dependent evaluation?
No. There were no improvements measured. 

*MFS100 results are exploratory image-level zero-shot measurements and may be statistically dependent because multiple captures can originate from the same burst-capture group. These do NOT establish independent biological-sample uncertainty.*

## Final Conclusion
**OUTCOME D: Performance decreased.**
