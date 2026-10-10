import os

search_strings = ["38.10", "0.381", "38%"]
found_locations = []

for root, dirs, files in os.walk("."):
    if ".git" in root or "venv" in root or "env" in root or "__pycache__" in root or "scratch" in root:
        continue
    for file in files:
        if file.endswith((".py", ".md", ".json", ".js", ".html", ".csv", ".txt")):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        for s in search_strings:
                            if s in line:
                                found_locations.append(f"{filepath}:{i+1} - {line.strip()}")
            except Exception:
                pass

for loc in found_locations:
    print(loc)
