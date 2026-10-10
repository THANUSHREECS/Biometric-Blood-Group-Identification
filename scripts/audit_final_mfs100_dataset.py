import os
import csv
from collections import defaultdict
from PIL import Image
import hashlib

def get_image_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    base_dir = os.path.join("data", "mfs100_final")
    raw_dir = os.path.join(base_dir, "raw")
    participants_csv = os.path.join(base_dir, "metadata", "participants.csv")
    
    if not os.path.exists(raw_dir) or not os.path.exists(participants_csv):
        print("Dataset not found or incomplete.")
        return

    # Read participants
    participants = {}
    with open(participants_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            participants[row['participant_id']] = row

    print("==================================================")
    print("FINAL MFS100 RESEARCH DATASET AUDIT")
    print("==================================================")
    
    total_participants = len(participants)
    print(f"Total Participants (Metadata): {total_participants}")
    
    participants_per_bg = defaultdict(int)
    for p, data in participants.items():
        participants_per_bg[data['blood_group']] += 1
        
    print("\nParticipants per Blood Group:")
    for bg, count in participants_per_bg.items():
        print(f"  {bg}: {count}")

    # Scan directories
    total_images = 0
    participants_with_3 = 0
    missing_scans = 0
    corrupted_images = 0
    duplicate_images = 0
    dim_mismatch = 0
    
    participant_images = defaultdict(list)
    image_hashes = defaultdict(list)
    
    for bg in ['A+', 'B+', 'AB+', 'O+']:
        bg_dir = os.path.join(raw_dir, bg)
        if not os.path.exists(bg_dir):
            continue
            
        for p_id in os.listdir(bg_dir):
            p_dir = os.path.join(bg_dir, p_id)
            if os.path.isdir(p_dir):
                images = [f for f in os.listdir(p_dir) if f.endswith('.bmp')]
                participant_images[p_id].extend([os.path.join(p_dir, img) for img in images])

    print("\nImage Statistics (By Participant):")
    total_participants_in_raw = len(participant_images)
    print(f"Total Participants (Raw Dir): {total_participants_in_raw}")
    
    for p_id, paths in participant_images.items():
        total_images += len(paths)
        if len(paths) == 3:
            participants_with_3 += 1
        else:
            missing_scans += (3 - len(paths))
            
        p_hashes = set()
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                
                with Image.open(path) as img:
                    if img.size != (316, 354): # Default MFS100 size
                        dim_mismatch += 1
                        
                h = get_image_hash(path)
                if h in p_hashes:
                    duplicate_images += 1
                p_hashes.add(h)
                image_hashes[h].append(path)
            except Exception:
                corrupted_images += 1

    print(f"Total Images: {total_images}")
    print(f"Participants with exactly 3 scans: {participants_with_3}")
    print(f"Missing scans (expected 3 per participant): {missing_scans}")
    print(f"Corrupted images: {corrupted_images}")
    print(f"Duplicate images (same participant): {duplicate_images}")
    print(f"Images with abnormal dimensions: {dim_mismatch}")
    
    # Global duplicates
    global_duplicates = sum(len(paths) - 1 for paths in image_hashes.values() if len(paths) > 1)
    if global_duplicates > duplicate_images:
         print(f"Global Duplicate images (across participants): {global_duplicates - duplicate_images}")

    print("==================================================")

if __name__ == "__main__":
    main()
