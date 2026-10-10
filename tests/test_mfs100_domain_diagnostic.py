import os
import sys
import json
import hashlib
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Day 1 Model Hash
EXPECTED_DAY1_MODEL_HASH = "7b5ff5d94d4ce3de7d432fcfb3e0ad71b508d69f7e993ca93075bba67ea12de9"

# Locked 8-class production files established in repository suite
EXPECTED_PROTECTED_HASHES = {
    "models/resnet50_improved_blood_group.keras": "a377510597cc216e95f3f4aad7f59c61e2d29ad4723363ffdda767dbc927a1f2",
    "models/vgg16_blood_group.keras": "7f46e1c52000773e98656c87278107c10350095384df1c3bdf12319d6632cc6c",
    "models/efficientnetb0_blood_group.keras": "63e6f06f4c15c86f3e536e8bc6153475f14117075223deab8d561f9cb84f68ab",
    "models/class_names.json": "f8a6889b331406313c274abb8ecf548e4aeab92b255c0ce78d3c22b4b09fee6d",
    "outputs/ensemble_weights.json": "12f23f6ce7562c4852a6e9f96560d58fda4636bc8f40d02149398db46743d604",
    "src/config.py": "84bf90f74fc355d8baa0b806245eb53e4da8d624a0f8dbf4656ff0d91c4d0b6a",  # updated: Round 14 + threshold rename
    "src/preprocessing.py": "4ccbfb0bef96d33054b6de9370f3811a25aed3e54e2ac05648972b3fd6cc71c4",  # updated: Round 14 densenet.preprocess_input removal
    "src/predictor.py": "a868b178648bd64e785d166c4328cf1ed99518d1b9e1440bd7e479052a27bf8d"  # updated: Round 21 text fix
}

EXPECTED_CLASS_MAPPING = {
    "A+": 0,
    "AB+": 1,
    "B+": 2,
    "O+": 3
}

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_raw_mfs100_handling():
    """Test 1: Raw MFS100 316x354 image handling and resizing to 224x224x3."""
    from src.preprocessing import composite_alpha_on_white, resize_to_target
    img = Image.new("L", (316, 354), color=255)
    img_rgb = composite_alpha_on_white(img)
    resized = resize_to_target(img_rgb, (224, 224))
    arr = np.array(resized, dtype=np.float32)
    assert arr.shape == (224, 224, 3)
    assert arr.min() >= 0.0 and arr.max() <= 255.0

def test_crop_coordinates_inside_boundaries():
    """Test 2: Crop coordinates stay inside [0, 316] and [0, 354] boundaries."""
    from src.preprocessing import crop_mfs100_foreground
    arr = np.ones((354, 316), dtype=np.uint8) * 255
    arr[100:250, 80:220] = 50
    patch_img = Image.fromarray(arr)
    cropped_img, metadata = crop_mfs100_foreground(patch_img)
    x1, y1, x2, y2 = metadata["bbox"]
    assert 0 <= x1 < x2 <= 316
    assert 0 <= y1 < y2 <= 354

def test_crop_dimensions_gte_80():
    """Test 3: Crop dimensions >= 80x80."""
    from src.preprocessing import crop_mfs100_foreground
    arr = np.ones((354, 316), dtype=np.uint8) * 255
    arr[100:250, 80:220] = 50
    patch_img = Image.fromarray(arr)
    cropped_img, metadata = crop_mfs100_foreground(patch_img)
    x1, y1, x2, y2 = metadata["bbox"]
    width = x2 - x1
    height = y2 - y1
    assert width >= 80
    assert height >= 80

def test_crop_output_224x224x3():
    """Test 4: Cropped image after Lanczos resize is 224x224x3."""
    from src.preprocessing import crop_mfs100_foreground, composite_alpha_on_white, resize_to_target
    arr = np.ones((354, 316), dtype=np.uint8) * 255
    arr[100:250, 80:220] = 50
    patch_img = Image.fromarray(arr)
    cropped_img, metadata = crop_mfs100_foreground(patch_img)
    comp = composite_alpha_on_white(cropped_img)
    resized = resize_to_target(comp, (224, 224))
    res_arr = np.array(resized, dtype=np.float32)
    assert res_arr.shape == (224, 224, 3)

def test_day1_model_output_dimension():
    """Test 5: Day 1 model output dimension is exactly 4."""
    import tensorflow as tf
    model_path = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class", "best_model.keras")
    assert os.path.exists(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model.output_shape[-1] == 4

def test_softmax_probabilities_sum():
    """Test 6: Softmax probabilities sum to 1 ± 1e-4."""
    import tensorflow as tf
    model_path = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class", "best_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    dummy_input = np.random.uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    for p in preds:
        assert np.isclose(np.sum(p), 1.0, atol=1e-4)

def test_exact_class_mapping():
    """Test 7: Exact class mapping matches Day 1 specification."""
    mapping_path = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class", "class_mapping.json")
    assert os.path.exists(mapping_path)
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    assert mapping["class_to_id"] == EXPECTED_CLASS_MAPPING

def test_no_mfs100_in_training():
    """Test 8: No MFS100 image was used in training/validation splits."""
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(PROJECT_ROOT, "outputs", "abo_positive_4class", f"data_manifest_{split}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                content = f.read()
            assert "data/captured" not in content
            assert "MFS100" not in content
            assert "BMP" not in content

def test_day1_model_sha256_unchanged():
    """Test 9: Day 1 model SHA-256 hash is unchanged."""
    model_path = os.path.join(PROJECT_ROOT, "models", "abo_positive_4class", "best_model.keras")
    assert os.path.exists(model_path)
    actual_hash = compute_sha256(model_path)
    assert actual_hash == EXPECTED_DAY1_MODEL_HASH, f"Day 1 model hash changed! {actual_hash} vs {EXPECTED_DAY1_MODEL_HASH}"

def test_protected_files_sha256_unchanged():
    """Test 10: All protected 8-class production files are unchanged."""
    for rel_path, expected_hash in EXPECTED_PROTECTED_HASHES.items():
        filepath = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(filepath), f"Protected file missing: {filepath}"
        actual_hash = compute_sha256(filepath)
        assert actual_hash == expected_hash, f"Protected file {filepath} altered! {actual_hash} vs {expected_hash}"
