import os

para = "\n\n**Current evidence:** The 4-class (A+, B+, AB+, O+) model achieves 86.0% accuracy on a random split. However, subject independence cannot be verified because the dataset lacks subject IDs. A metadata artifact classifier achieves 44.58%, beating the majority baseline, suggesting that dataset artifacts (like file size or contrast) correlate with the labels. The origin of the 38.10% accuracy metric is unknown. The live system does not demonstrate reliable cross-domain four-class blood group classification. Clinical validity is not claimed.\n"

docs = [
    "README.md",
    "reports/external_evaluation/VIVA_PREPARATION.md",
    "results/AUDIT_SUMMARY.md"
]

for doc in docs:
    if os.path.exists(doc):
        with open(doc, "a", encoding="utf-8") as f:
            f.write(para)
        print(f"Appended to {doc}")

# For UI footer, we need to inject it into index.html
ui_file = "live_app/frontend/index.html"
if os.path.exists(ui_file):
    with open(ui_file, "r", encoding="utf-8") as f:
        content = f.read()
    if "<footer>" in content:
        content = content.replace("<footer>", f"<footer>{para}")
    else:
        content = content.replace("</body>", f"<footer>{para}</footer></body>")
    with open(ui_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected into UI footer")
