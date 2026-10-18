import hashlib
from pathlib import Path

BASE = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")

baseline_hashes = {
    "Orig E2 fold_1": ("models/phase3_fusion_metric_learning/E2/fold_1_best.weights.h5", "7d9464d18619a94fabd5c85a28c5fd7ef47b071e259bdaada8d05f7aa6db264d"),
    "Orig E2 fold_2": ("models/phase3_fusion_metric_learning/E2/fold_2_best.weights.h5", "9c72809f23152708806a663891bdaa096840cc15e4dee2c485e890db6cc7bede"),
    "Orig E2 fold_3": ("models/phase3_fusion_metric_learning/E2/fold_3_best.weights.h5", "8123b1b86f44719d5cf46341d81b0958665cdc5058844b71fd1590447a0432e3"),
    "Orig E2 fold_4": ("models/phase3_fusion_metric_learning/E2/fold_4_best.weights.h5", "cc122b23e0c6d118cd8ffac836b2ab97d73a419e61ba9d042d735e5f9cd71519"),
    "Orig E2 fold_5": ("models/phase3_fusion_metric_learning/E2/fold_5_best.weights.h5", "aff9e137bed61f641c4f5432ceb91ee40f052bcc63a28a42f55dacac51faf05d"),
    "EXP-03 Arm 1 fold_1": ("experiments/exp03/checkpoints/arm1/fold_1_best.weights.h5", "0d047b7a10153718b067e84a27a6ad2bc0486ccf1d35057a07da6e2e2d627ac5"),
    "EXP-03 Arm 1 fold_2": ("experiments/exp03/checkpoints/arm1/fold_2_best.weights.h5", "dd94e4c44887399d700ac2de195f513a5b1db73ac561f4ff8c2345d5156f7063"),
    "EXP-03 Arm 1 fold_3": ("experiments/exp03/checkpoints/arm1/fold_3_best.weights.h5", "f763f67068a4411dbd4a405e39248f2ead1b1b1fe228c4b1bca5da807005b3e3"),
    "EXP-03 Arm 1 fold_4": ("experiments/exp03/checkpoints/arm1/fold_4_best.weights.h5", "589f1e52275f68ed02329b1d4a7a9c16f406ed3868b8afeef9706ebace61b496"),
    "EXP-03 Arm 1 fold_5": ("experiments/exp03/checkpoints/arm1/fold_5_best.weights.h5", "848aea5498e469871d5fdcd136d849f971c9c3c485a4c66afb2505d3b0a46776"),
    "Production best_model.keras": ("models/abo_positive_4class/best_model.keras", "7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9")
}

all_match = True
for name, (rel_path, expected_hash) in baseline_hashes.items():
    p = BASE / rel_path
    assert p.is_file(), f"Missing {p}"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    curr = h.hexdigest()
    match = (curr == expected_hash)
    if not match:
        all_match = False
    print(f"{name:<28}: {'MATCH (INTACT)' if match else 'FAILED'} [{curr[:16]}...]")

assert all_match, "Integrity failure!"
print("\nALL BASELINES AND PRODUCTION MODEL REMAIN 100% UNTOUCHED AND INTACT.")
