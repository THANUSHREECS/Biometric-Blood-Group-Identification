import os
import io
import unittest
import numpy as np
from PIL import Image
import tensorflow as tf

from src.preprocessing import (
    load_image_to_pil,
    composite_alpha_on_white,
    resize_to_target,
    preprocess_fingerprint,
    load_and_preprocess_for_training
)
from src.config import IMAGE_SIZE

class TestPreprocessingPipeline(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.sample_dir = os.path.join(self.project_root, "data", "sample")

    def test_composite_alpha_on_white(self):
        # Create a test RGBA image with 1 transparent pixel and 1 black pixel
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        # Pixel (0, 0) is fully transparent
        rgba[0, 0] = [0, 0, 0, 0]
        # Pixel (5, 5) is opaque black
        rgba[5, 5] = [0, 0, 0, 255]
        # Pixel (9, 9) is opaque white
        rgba[9, 9] = [255, 255, 255, 255]

        pil_rgba = Image.fromarray(rgba, mode="RGBA")
        rgb_result = composite_alpha_on_white(pil_rgba)

        self.assertEqual(rgb_result.mode, "RGB")
        res_arr = np.array(rgb_result)

        # Transparent pixel should become (255, 255, 255)
        np.testing.assert_array_equal(res_arr[0, 0], [255, 255, 255])
        # Opaque black should remain (0, 0, 0)
        np.testing.assert_array_equal(res_arr[5, 5], [0, 0, 0])
        # Opaque white should remain (255, 255, 255)
        np.testing.assert_array_equal(res_arr[9, 9], [255, 255, 255])

    def test_resize_to_target(self):
        img = Image.new("RGB", (96, 103), color=(128, 128, 128))
        resized = resize_to_target(img, target_size=(224, 224))
        self.assertEqual(resized.size, (224, 224))

    def test_preprocess_fingerprint_from_filepath(self):
        sample_path = os.path.join(self.sample_dir, "A+.bmp")
        if os.path.exists(sample_path):
            batch_tensor, display_pil = preprocess_fingerprint(sample_path)
            self.assertEqual(batch_tensor.shape, (1, 112, 112, 3))
            self.assertEqual(batch_tensor.dtype, np.float32)
            self.assertEqual(display_pil.size, (112, 112))
            self.assertEqual(display_pil.mode, "RGB")

    def test_preprocess_fingerprint_from_bytes(self):
        img = Image.new("RGB", (150, 150), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        img_bytes = buf.getvalue()

        batch_tensor, display_pil = preprocess_fingerprint(img_bytes)
        self.assertEqual(batch_tensor.shape, (1, 112, 112, 3))
        self.assertEqual(display_pil.size, (112, 112))

    def test_load_and_preprocess_for_training(self):
        sample_path = os.path.join(self.sample_dir, "O+.bmp")
        if os.path.exists(sample_path):
            tensor = load_and_preprocess_for_training(sample_path)
            self.assertEqual(tensor.shape, (112, 112, 3))
            self.assertEqual(tensor.dtype, np.float32)

    def test_enhance_mfs100_ridge_contrast(self):
        from src.preprocessing import enhance_mfs100_ridge_contrast, transform_c5
        # Create low-contrast synthetic image simulating MFS100 gray ridges
        arr = np.full((354, 316), 200, dtype=np.uint8)
        arr[50:300:10, 50:250] = 120  # simulated gray ridges
        img = Image.fromarray(arr, mode="L")

        enhanced = enhance_mfs100_ridge_contrast(img)
        self.assertEqual(enhanced.size, (316, 354))
        self.assertEqual(enhanced.mode, "RGB")
        enh_arr = np.array(enhanced)
        # Contrast standard deviation should increase
        self.assertGreater(float(np.std(enh_arr)), float(np.std(arr)))

    def test_transform_c5_source_aware(self):
        from src.preprocessing import transform_c5
        sample_path = os.path.join(self.sample_dir, "A+.bmp")
        if os.path.exists(sample_path):
            # transform_c5 returns (np.ndarray, PIL.Image, metadata_dict) — 3-tuple
            result = transform_c5(sample_path)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            arr, pil_img, meta = result
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(arr.shape, (112, 112, 3))
            self.assertIn("crop_applied", meta)

if __name__ == "__main__":
    unittest.main()
