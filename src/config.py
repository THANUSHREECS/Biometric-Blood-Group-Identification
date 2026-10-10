import os

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Class Definitions
CLASS_NAMES = ["A+", "B+", "AB+", "O+"]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Preprocessing Constants
IMAGE_SIZE = (112, 112)
INPUT_SHAPE = (112, 112, 3)
BACKGROUND_COLOR = (255, 255, 255) # White background for RGBA compositing

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
QUARANTINE_DATA_DIR = os.path.join(DATA_DIR, "quarantine")
CAPTURED_IMAGE_DIR = os.path.join(DATA_DIR, "captured")
SAMPLE_IMAGE_DIR = os.path.join(DATA_DIR, "sample")
DATABASE_PATH = os.path.join(DATA_DIR, "blood_group.db")

# Fallback / Source dataset directory if raw data not yet populated
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_blood_group")

# Model & Outputs
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "blood_group_resnet50.keras")
RESNET50_IMPROVED_PATH = os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras")
VGG16_PATH = os.path.join(MODELS_DIR, "vgg16_blood_group.keras")
EFFICIENTNETB0_PATH = os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
ENSEMBLE_WEIGHTS_PATH = os.path.join(OUTPUTS_DIR, "ensemble_weights.json")
DATASET_MANIFEST_PATH = os.path.join(OUTPUTS_DIR, "dataset_manifest.csv")
DATASET_REPORT_PATH = os.path.join(OUTPUTS_DIR, "dataset_report.json")
TRAINING_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "training_results.json")
EVALUATION_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_results.json")

# Operational Parameters
ADVISORY_CONFIRMATION_THRESHOLD = 0.70
MFS100_MOCK = os.environ.get("MFS100_MOCK", "true").lower() in ("true", "1", "yes")
MFS100_SDK_PATH = os.environ.get(
    "MFS100_SDK_PATH", 
    r"C:\Program Files\Mantra\MFS100\Driver\MFS100Test\MFS100Dll.dll"
)

