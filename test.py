import unittest
import numpy as np
import cv2
import os
from main import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ImageProcessor()
        self.test_image = np.full((100, 100, 3), 127, dtype=np.uint8)  # Gray dummy image
        self.processor.original_image = self.test_image.copy()
        self.processor.current_image = self.test_image.copy()

    def test_load_image(self):
        # Simulate loading (image is already loaded in setUp)
        self.assertIsNotNone(self.processor.original_image)

    def test_save_image(self):
        path = "temp_test_output.png"
        self.processor.save_image(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)  # Clean up

    def test_apply_sepia(self):
        result = self.processor.effect_sepia(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_negative(self):
        result = self.processor.effect_negative(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_sobel(self):
        result = self.processor.effect_sobel(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_box_blur(self):
        result = self.processor.effect_box_blur(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_sharpening(self):
        result = self.processor.effect_sharpening(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_edge_sketch(self):
        result = self.processor.effect_edge_sketch(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_hdr(self):
        result = self.processor.effect_hdr(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_vignette(self):
        result = self.processor.effect_vignette(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_laplacian(self):
        result = self.processor.effect_laplacian(self.test_image, 1.0)
        self.assertEqual(result.shape, self.test_image.shape)

    def test_apply_adjustments(self):
        adjusted = self.processor.apply_adjustments(brightness=50, contrast=1.2)
        self.assertEqual(adjusted.shape, self.test_image.shape)

    def test_apply_adjustments_to_image(self):
        adjusted = self.processor.apply_adjustments_to_image(self.test_image, brightness=50, contrast=1.5)
        self.assertEqual(adjusted.shape, self.test_image.shape)

    def test_resize_image(self):
        self.processor.resize_image(50, 50)
        self.assertEqual(self.processor.current_image.shape[0:2], (50, 50))

    def test_flip_image_horizontal(self):
        self.processor.flip_image('horizontal')
        self.assertEqual(self.processor.current_image.shape, self.test_image.shape)

    def test_flip_image_vertical(self):
        self.processor.flip_image('vertical')
        self.assertEqual(self.processor.current_image.shape, self.test_image.shape)

    def test_histograms(self):
        b, g, r = self.processor.calculate_histograms()
        self.assertEqual(len(b), 256)
        self.assertEqual(len(g), 256)
        self.assertEqual(len(r), 256)

    def test_reset_image(self):
        self.processor.current_image = np.zeros_like(self.test_image)
        self.processor.reset_image()
        np.testing.assert_array_equal(self.processor.current_image, self.processor.original_image)

if __name__ == '__main__':
    unittest.main()