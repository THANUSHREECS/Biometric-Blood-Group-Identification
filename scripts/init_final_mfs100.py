import os
import csv

def main():
    base_dir = "data/mfs100_final"
    dirs = [
        "raw/A+", "raw/B+", "raw/AB+", "raw/O+",
        "metadata"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    participants_csv = os.path.join(base_dir, "metadata", "participants.csv")
    if not os.path.exists(participants_csv):
        with open(participants_csv, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["participant_id", "blood_group", "verification_source", "collection_date", "finger", "consent", "number_of_scans"])

    log_csv = os.path.join(base_dir, "metadata", "collection_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "participant_id", "scan_number", "image_path", "prediction", "confidence", "prob_A_plus", "prob_B_plus", "prob_AB_plus", "prob_O_plus"])

    readme_path = os.path.join(base_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# Final MFS100 Research Data Collection\n\nIndependent, real-world evaluation dataset.\n")

    print("Initialized data/mfs100_final structure.")

if __name__ == "__main__":
    main()
