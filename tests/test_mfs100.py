import os
import unittest
from src.mfs100.scanner import MFS100Scanner

class TestMFS100Scanner(unittest.TestCase):
    def setUp(self):
        self.scanner = MFS100Scanner(mock=True)

    def test_connect_and_info(self):
        connected = self.scanner.connect()
        self.assertTrue(connected)
        self.assertTrue(self.scanner.is_connected())

        info = self.scanner.get_device_info()
        self.assertEqual(info["model"], "MFS100")
        self.assertTrue(info["mock_mode"])
        self.assertTrue(info["is_connected"])

    def test_mock_capture(self):
        self.scanner.connect()
        # Test random capture
        img_path, img_bytes = self.scanner.capture()
        self.assertTrue(os.path.exists(img_path))
        self.assertGreater(len(img_bytes), 1000)

        # Test specific class sample capture
        img_path_b, img_bytes_b = self.scanner.capture(sample_class="B+")
        self.assertTrue(os.path.exists(img_path_b))
        self.assertGreater(len(img_bytes_b), 1000)

    def test_disconnect(self):
        self.scanner.disconnect()
        self.assertFalse(self.scanner.is_connected())

if __name__ == "__main__":
    unittest.main()
