from pathlib import Path

PROJECT_ROOT = Path(r"c:/Users/thanu/OneDrive - Don Bosco Institute of Technology/major project")
orig_dir = PROJECT_ROOT / "dataset_blood_group"

files = list(orig_dir.glob("**/*.*"))[:20]
print("Sample filenames in dataset_blood_group:")
for f in files:
    print(" ", f.relative_to(orig_dir))
