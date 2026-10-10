import inspect

def print_file_lines(filename, start, end):
    print(f"\n{'='*25} {filename} (Lines {start}-{end}) {'='*25}")
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx in range(start-1, min(end, len(lines))):
        print(f"{idx+1:4d}: {lines[idx]}", end="")

def main():
    print("="*75)
    print("STEP 2: DETAILED CODE COMPARISON")
    print("="*75)
    
    # 1. experiment6_fusion_supcon.py
    exp6 = "scratch/experiment6_fusion_supcon.py"
    # Preprocessing, imports, classes
    print_file_lines(exp6, 12, 70)
    # Model building & fusion
    print_file_lines(exp6, 75, 145)
    # Loss, optimizer, training loop, checkpoint rule
    print_file_lines(exp6, 146, 230)
    
    # 2. src/preprocessing.py
    prep = "src/preprocessing.py"
    print_file_lines(prep, 70, 160)
    
    # 3. src/predictor.py
    pred = "src/predictor.py"
    print_file_lines(pred, 30, 95)

if __name__ == "__main__":
    main()
