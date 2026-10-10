import os
import shutil

backup_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "major_project_backup"))
print(f"Backing up to {backup_dir}...")

if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

dirs_to_backup = ["models", "results", "outputs"]

for d in dirs_to_backup:
    if os.path.exists(d):
        dest = os.path.join(backup_dir, d)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(d, dest)
        print(f"Backed up {d}")

print("Backup complete.")
