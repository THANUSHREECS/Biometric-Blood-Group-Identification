import numpy as np
from pathlib import Path

p = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project/reports/model_fusion/all_models_participant_probs.npz")
data = np.load(p)
print("Keys in all_models_participant_probs.npz:", data.files)
for k in data.files:
    arr = data[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
