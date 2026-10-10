import os
import csv
import argparse
import random
import shutil
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Split Final MFS100 Research Dataset")
    parser.add_argument('--dev', type=float, default=0.7, help="Fraction for development")
    parser.add_argument('--val', type=float, default=0.15, help="Fraction for validation")
    parser.add_argument('--test', type=float, default=0.15, help="Fraction for test")
    parser.add_argument('--output_dir', type=str, default="data/mfs100_final_splits", help="Output directory")
    args = parser.parse_args()

    # Normalize fractions
    total = args.dev + args.val + args.test
    frac_dev = args.dev / total
    frac_val = args.val / total
    frac_test = args.test / total

    base_dir = os.path.join("data", "mfs100_final")
    participants_csv = os.path.join(base_dir, "metadata", "participants.csv")
    
    if not os.path.exists(participants_csv):
        print("Dataset metadata not found.")
        return
        
    participants = []
    with open(participants_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            participants.append(row)
            
    # Stratify by blood group
    stratified = defaultdict(list)
    for p in participants:
        stratified[p['blood_group']].append(p)
        
    splits = {'dev': [], 'val': [], 'test': []}
    
    for bg, p_list in stratified.items():
        random.shuffle(p_list)
        n = len(p_list)
        n_dev = int(n * frac_dev)
        n_val = int(n * frac_val)
        
        splits['dev'].extend(p_list[:n_dev])
        splits['val'].extend(p_list[n_dev:n_dev+n_val])
        splits['test'].extend(p_list[n_dev+n_val:])
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split_name, split_participants in splits.items():
        split_dir = os.path.join(args.output_dir, split_name)
        
        for bg in ['A+', 'B+', 'AB+', 'O+']:
            os.makedirs(os.path.join(split_dir, bg), exist_ok=True)
            
        print(f"Copying {len(split_participants)} participants to {split_name}...")
        for p in split_participants:
            bg = p['blood_group']
            p_id = p['participant_id']
            src_dir = os.path.join(base_dir, "raw", bg, p_id)
            dst_dir = os.path.join(split_dir, bg, p_id)
            
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir)
                
    print("Dataset split successfully.")

if __name__ == "__main__":
    main()
