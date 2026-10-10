# HemoPrint AI: Final Audit Results Table

| Metric | Random Split (Original) | Pseudo-Group Split (ORB) | Live Validation (External) |
| :--- | :--- | :--- | :--- |
| **Number of Samples** | 898 (test set) | 898 (test set) | 101 Participants (303 scans) |
| **Classes Evaluated** | 8-Class | 8-Class | 4-Class (A+, B+, AB+, O+) |
| **Majority Baseline** | 16.93% | 16.93% | 30.6% |
| **Chance (Prior-Matched)** | ~13.5% | ~13.5% | ~28.5% |
| **Reported Accuracy** | 38.10% (Origin Unknown) | 72.53% | 27.72% |
| **95% Bootstrap CI** | Unknown | Unknown | [19.0%, 37.0%] |
| **Data Leakage / Issues** | Did not isolate by subject | **Invalid:** Grouping failed (87.7% false-link rate) | Strictly Subject-Isolated |

### Additional Artifact Tests (Random Split, 4-Class)
*   **(a) Border/corner-pixel stats only**: 20.82% (95% CI: [18.15%, 23.50%])
*   **(b) BMP header + file size + timestamp only**: 16.93% (95% CI: [14.36%, 19.38%])
*   **(c) Ridge-area stats only**: 19.04% (95% CI: [16.59%, 21.60%])

### Conclusions
1.  **Biological Plausibility:** There is no supported biological or genetic mechanism linking friction ridge patterns to ABO blood groups (no signal detected at n=101; strongly predictive model excluded; weak effects cannot be excluded).
2.  **Subject Independence:** The original dataset's subject independence could not be verified. Attempted pseudo-grouping via ORB feature matching degenerated into a single massive cluster, invalidating the 72.53% result.
3.  **Real-world Performance:** When evaluated on a strictly locked, subject-isolated live validation set of 101 individuals, the model performed at **27.72%**, which fails to beat the constant-class majority baseline of **30.6%**. 
4.  **Model Integrity:** The deployed Champion E2 model (`best_model.keras`) hash precisely matches its earliest backups. No unauthorized modifications or test manipulations were found in the codebase.
