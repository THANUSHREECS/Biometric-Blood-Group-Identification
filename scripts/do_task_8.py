import os
import re

print("Updating src/config.py...")
with open("src/config.py", "r") as f:
    config_data = f.read()

# Update config
if "CLASS_NAMES = [" in config_data:
    config_data = re.sub(r'CLASS_NAMES = \[.*?\]', 'CLASS_NAMES = ["A+", "B+", "AB+", "O+"]', config_data, flags=re.DOTALL)
if "CONFIDENCE_THRESHOLD = 0.60" in config_data:
    config_data = config_data.replace('CONFIDENCE_THRESHOLD = 0.60\n', '')
with open("src/config.py", "w") as f:
    f.write(config_data)

print("Updating app.py...")
with open("app.py", "r") as f:
    app_data = f.read()

config_endpoint = """
@app.route("/api/config", methods=["GET"])
def get_config():
    from src.config import ADVISORY_CONFIRMATION_THRESHOLD, IMAGE_SIZE, CLASS_NAMES
    return jsonify({
        "threshold": ADVISORY_CONFIRMATION_THRESHOLD * 100,
        "image_size": IMAGE_SIZE[0],
        "class_order": CLASS_NAMES
    })
"""
if "/api/config" not in app_data:
    app_data = app_data.replace("# API Endpoints\n# =========================================================================\n", 
                                "# API Endpoints\n# =========================================================================\n" + config_endpoint)

app_data = app_data.replace('pred_res.get("advisory_threshold", 0.70)', 'pred_res.get("advisory_threshold", ADVISORY_CONFIRMATION_THRESHOLD)')

if "ADVISORY_CONFIRMATION_THRESHOLD" not in app_data:
    app_data = app_data.replace("from src.config import DATABASE_PATH", "from src.config import DATABASE_PATH, ADVISORY_CONFIRMATION_THRESHOLD")

with open("app.py", "w") as f:
    f.write(app_data)

print("Updating src/report_generator.py...")
with open("src/report_generator.py", "r") as f:
    report_data = f.read()
if "from src.config import" not in report_data:
    report_data = "from src.config import ADVISORY_CONFIRMATION_THRESHOLD\n" + report_data
report_data = report_data.replace("advisory_threshold=0.70,", "advisory_threshold=ADVISORY_CONFIRMATION_THRESHOLD,")
with open("src/report_generator.py", "w") as f:
    f.write(report_data)

print("Task 8 code replacements executed.")
