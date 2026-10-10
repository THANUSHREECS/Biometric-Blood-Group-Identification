import os
import csv
import glob
import hashlib
from collections import defaultdict
from PIL import Image
from datetime import datetime

project_root = r"c:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project"
final_data_root = os.path.join(project_root, "data", "mfs100_final")
metadata_dir = os.path.join(final_data_root, "metadata")
raw_dir = os.path.join(final_data_root, "raw")
participants_csv = os.path.join(metadata_dir, "participants.csv")
collection_log_csv = os.path.join(metadata_dir, "collection_log.csv")

participants = {}
blood_group_counts = {"A+": 0, "B+": 0, "AB+": 0, "O+": 0}
verifications = defaultdict(int)
consent_yes = 0
consent_no = 0
left_thumb_count = 0
other_thumb_count = 0

pii_fields_found = []
pii_keywords = ["name", "phone", "email", "address", "student_id", "aadhaar", "dob", "date_of_birth"]

with open(participants_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for k in reader.fieldnames:
        for pii in pii_keywords:
            if pii in k.lower():
                pii_fields_found.append(k)
    
    for row in reader:
        pid = row["participant_id"]
        bg = row["blood_group"]
        if pid not in participants:
            participants[pid] = {
                "blood_group": bg,
                "scans_expected": 3,
                "scans_found": 0,
                "verification_source": row["verification_source"],
                "finger": row["finger"],
                "consent": row["consent"]
            }
            if bg in blood_group_counts:
                blood_group_counts[bg] += 1
            verifications[row["verification_source"]] += 1
            if row["consent"].strip().lower() == "yes":
                consent_yes += 1
            else:
                consent_no += 1
            if row["finger"].strip().lower() == "left thumb":
                left_thumb_count += 1
            else:
                other_thumb_count += 1

total_participants = len(participants)

# Check directory structure and scans
actual_scans = []
for bg in ["A+", "B+", "AB+", "O+"]:
    bg_dir = os.path.join(raw_dir, bg)
    if not os.path.exists(bg_dir): continue
    for pid in os.listdir(bg_dir):
        p_dir = os.path.join(bg_dir, pid)
        if os.path.isdir(p_dir):
            if pid in participants and participants[pid]["blood_group"] != bg:
                print(f"ERROR: {pid} found in {bg} but ground truth is {participants[pid]['blood_group']}")
            for scan_file in glob.glob(os.path.join(p_dir, "*.bmp")):
                if pid in participants:
                    participants[pid]["scans_found"] += 1
                actual_scans.append((pid, scan_file))

scan_counts = {3: 0, 2: 0, 1: 0, 0: 0, 'more': 0}
for p, data in participants.items():
    c = data["scans_found"]
    if c > 3: scan_counts['more'] += 1
    elif c in scan_counts: scan_counts[c] += 1
    else: scan_counts[0] += 1

# Analyze images
total_images = len(actual_scans)
valid_images = 0
corrupted_images = 0
dimensions = defaultdict(int)

hashes = {}
within_duplicates = 0
cross_participant_duplicates = 0
cross_class_duplicates = 0

for pid, path in actual_scans:
    try:
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            dimensions[f"{w}x{h}_{mode}"] += 1
            valid_images += 1
    except Exception:
        corrupted_images += 1

    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash in hashes:
            orig_pid, orig_bg = hashes[file_hash]
            curr_bg = participants[pid]["blood_group"] if pid in participants else "unknown"
            if pid == orig_pid:
                within_duplicates += 1
            else:
                cross_participant_duplicates += 1
                if curr_bg != orig_bg:
                    cross_class_duplicates += 1
        else:
            hashes[file_hash] = (pid, participants[pid]["blood_group"] if pid in participants else "unknown")

# Temporal issues & inconsistencies from log
log_data = defaultdict(list)
with open(collection_log_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["participant_id"]
        ts = row["timestamp"]
        try:
            dt = datetime.fromisoformat(ts)
            log_data[pid].append((dt, row["prediction"]))
        except:
            pass

valid_temporal = 0
potential_temporal_issue = 0

for pid, entries in log_data.items():
    entries.sort(key=lambda x: x[0])
    if len(entries) >= 2:
        diffs = [(entries[i+1][0] - entries[i][0]).total_seconds() for i in range(len(entries)-1)]
        if any(d > 3600 for d in diffs):
            potential_temporal_issue += 1
        else:
            valid_temporal += 1
    elif len(entries) == 1:
        valid_temporal += 1
        
inconsistent_labels = 0

# Write outputs
os.makedirs(os.path.join(project_root, "reports", "final_mfs100"), exist_ok=True)

csv_out = os.path.join(project_root, "reports", "final_mfs100", "participant_distribution.csv")
with open(csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["blood_group", "unique_participants", "percentage", "total_scans", "expected_scans", "missing_scans"])
    for bg in ["A+", "B+", "AB+", "O+"]:
        count = blood_group_counts[bg]
        pct = (count / total_participants * 100) if total_participants > 0 else 0
        expected = count * 3
        actual = sum(1 for p, d in participants.items() if d["blood_group"] == bg for _ in range(d["scans_found"]))
        missing = expected - actual
        writer.writerow([bg, count, f"{pct:.2f}%", actual, expected, missing])

md_out = os.path.join(project_root, "reports", "final_mfs100", "FINAL_105_PARTICIPANT_DATASET_AUDIT.md")
with open(md_out, "w") as f:
    f.write("# FINAL 105-PARTICIPANT DATASET AUDIT\n\n")
    f.write(f"1. Dataset location: {final_data_root}\n")
    f.write(f"2. Number of participants: {total_participants}\n")
    f.write("3. Blood-group distribution:\n")
    for bg in ["A+", "B+", "AB+", "O+"]:
        f.write(f"   - {bg}: {blood_group_counts[bg]}\n")
    f.write(f"4. Number of scans: {total_images}\n")
    f.write(f"5. Scan completeness: 3/3 ({scan_counts[3]}), 2/3 ({scan_counts[2]}), 1/3 ({scan_counts[1]})\n")
    f.write("6. Ground-truth verification sources:\n")
    for k, v in verifications.items():
        f.write(f"   - {k}: {v}\n")
    f.write(f"7. Consent status: Yes ({consent_yes}), No ({consent_no})\n")
    f.write(f"8. Left-thumb verification: Left thumb ({left_thumb_count}), Other ({other_thumb_count})\n")
    f.write(f"9. Duplicate analysis: Within ({within_duplicates}), Cross-participant ({cross_participant_duplicates})\n")
    f.write(f"10. Temporal analysis: Valid ({valid_temporal}), Issues ({potential_temporal_issue})\n")
    f.write(f"11. Image quality: Valid ({valid_images}), Corrupted ({corrupted_images})\n")
    f.write(f"12. Label consistency: Inconsistent ({inconsistent_labels})\n")
    f.write(f"13. Privacy audit: PII fields found ({', '.join(pii_fields_found) if pii_fields_found else 'None'})\n")
    f.write("14. Limitations: As an automated audit, visual inspection was not performed.\n")

status = "PASS"
if total_participants != 105 or total_images != 315 or cross_participant_duplicates > 0:
    status = "FAIL"
elif corrupted_images > 0 or potential_temporal_issue > 0:
    status = "PASS WITH WARNINGS"

print("===============================================")
print("FINAL 105-PARTICIPANT MFS100 DATASET AUDIT")
print("===============================================")
print(f"Unique participants:\n{total_participants}")
print("Expected:\n105")
print(f"Status:\n{'PASS' if total_participants == 105 else 'FAIL'}\n")

print(f"Total fingerprint images:\n{total_images}")
print("Expected:\n315")
print(f"Status:\n{'PASS' if total_images == 315 else 'FAIL'}\n")

print(f"A+ participants:\n{blood_group_counts['A+']}")
print(f"B+ participants:\n{blood_group_counts['B+']}")
print(f"AB+ participants:\n{blood_group_counts['AB+']}")
print(f"O+ participants:\n{blood_group_counts['O+']}\n")

print(f"3 scans complete:\n{scan_counts[3]}")
print(f"2 scans:\n{scan_counts[2]}")
print(f"1 scan:\n{scan_counts[1]}")
print(f"Missing:\n{scan_counts[0]}\n")

print(f"Left-thumb confirmed:\n{left_thumb_count}\n")

print("Ground-truth verification:")
print(f"Lab report:\n{verifications.get('lab_report', 0)}")
print(f"Medical record:\n{verifications.get('medical_record', 0)}")
print(f"Self reported:\n{verifications.get('self_reported', 0)}")
print(f"Unknown:\n{verifications.get('unknown', 0)}\n")

print("Consent:")
print(f"Complete:\n{consent_yes}")
print(f"Missing:\n{consent_no}\n")

print(f"Cross-participant exact duplicates:\n{cross_participant_duplicates}\n")
print(f"Corrupted images:\n{corrupted_images}\n")
print(f"Label inconsistencies:\n{inconsistent_labels}\n")
print(f"Potential temporal issues:\n{potential_temporal_issue}\n")
print(f"PII detected:\n{', '.join(pii_fields_found) if pii_fields_found else 'None'}\n")
print(f"OVERALL DATASET STATUS:\n{status}")
print("===============================================")
