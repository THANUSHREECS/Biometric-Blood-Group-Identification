import os
import glob
from datetime import datetime

def get_file_info(path):
    try:
        st = os.stat(path)
        mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size = st.st_size
        return f"{mtime} | {size:>10} bytes | {path}"
    except Exception as e:
        return f"ERROR ({e}) | {path}"

def search_files():
    print("="*75)
    print("ITEM 5 & 6: SEARCHING FOR E2 TRAINING CODE, OUTPUTS, AND WEIGHT PROVENANCE")
    print("="*75)
    
    search_dirs = [
        os.path.abspath("."),
        os.path.abspath("../major_project_backup"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        r"C:\Users\thanu\.gemini\antigravity-ide\brain"
    ]
    
    print("\n--- 1. SEARCH FOR *.ipynb FILES ---")
    notebooks = []
    for d in search_dirs:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "**", "*.ipynb"), recursive=True)
            for f in found:
                # exclude virtual environments or checkpoints
                if ".venv" not in f and ".ipynb_checkpoints" not in f and "node_modules" not in f:
                    notebooks.append(f)
                    
    if notebooks:
        for nb in sorted(notebooks):
            print(f"  {get_file_info(nb)}")
    else:
        print("  None found.")
        
    print("\n--- 2. SEARCH FOR REFERENCES TO E2 STRINGS ---")
    keywords = ["fold_1_best", "SupCon", "phase3_fusion_metric_learning"]
    matching_files = {}
    
    # We will search text files (.py, .json, .md, .txt, .csv, .sh, .bat, .log)
    exts = {".py", ".json", ".md", ".txt", ".csv", ".sh", ".bat", ".log"}
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        print(f"Searching in: {d} ...")
        for root, dirs, files in os.walk(d):
            # prune large unnecessary dirs
            dirs[:] = [sub for sub in dirs if sub not in {".git", ".venv", "node_modules", "__pycache__", ".cache"}]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exts:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        hits = [kw for kw in keywords if kw in content]
                        if hits:
                            matching_files[filepath] = hits
                    except Exception:
                        pass

    print(f"\nTotal matching files found: {len(matching_files)}")
    for p, hits in sorted(matching_files.items()):
        print(f"  [{', '.join(hits)}] -> {get_file_info(p)}")

    print("\n--- 3. SEARCH reports/ AND outputs/ FOR SAVED E2 METRICS/PREDICTIONS ---")
    reports_outputs = []
    for base in ["reports", "outputs"]:
        for root, _, files in os.walk(base):
            for file in files:
                p = os.path.join(root, file)
                reports_outputs.append(p)
                
    e2_artifacts = []
    for p in reports_outputs:
        ext = os.path.splitext(p)[1].lower()
        if ext in {".csv", ".json", ".txt", ".md"}:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if any(k in content for k in ["E2", "fold_1_best", "phase3_fusion"]):
                    e2_artifacts.append(p)
            except Exception:
                pass
                
    print(f"Found {len(e2_artifacts)} artifacts mentioning E2 in reports/ and outputs/:")
    for p in sorted(e2_artifacts):
        print(f"  {get_file_info(p)}")

    print("\n--- 4. DETAILED TIMESTAMPS FOR ALL E2 WEIGHT FILES AND CONFIGS ---")
    e2_dir = "models/phase3_fusion_metric_learning/E2"
    if os.path.exists(e2_dir):
        for f in sorted(os.listdir(e2_dir)):
            full_p = os.path.join(e2_dir, f)
            print(f"  {get_file_info(full_p)}")
            
    print("Frozen config:")
    cfg = "reports/phase3_final_evaluation/frozen_model_config.json"
    if os.path.exists(cfg):
        print(f"  {get_file_info(cfg)}")

if __name__ == "__main__":
    search_files()
