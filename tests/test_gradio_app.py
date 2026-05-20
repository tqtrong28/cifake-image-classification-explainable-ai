import unittest

import numpy as np
import torch
from PIL import Image

from app import build_heatmap_images, format_prediction, prepare_image


class TestGradioAppHelpers(unittest.TestCase):
    def test_prepare_image_resizes_converts_rgb_and_scales_tensor(self):
        image = Image.new("RGBA", (80, 40), (10, 20, 30, 255))

        tensor, display_image = prepare_image(image, torch.device("cpu"))

        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))
        self.assertEqual(display_image.mode, "RGB")
        self.assertEqual(display_image.size, (32, 32))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_build_heatmap_images_returns_display_sized_rgb_images(self):
        original = Image.new("RGB", (32, 32), (120, 80, 40))
        heatmap = np.zeros((32, 32), dtype=np.float32)
        heatmap[8:24, 8:24] = 1.0

        heatmap_image, overlay_image = build_heatmap_images(original, heatmap)

        self.assertEqual(heatmap_image.mode, "RGB")
        self.assertEqual(overlay_image.mode, "RGB")
        self.assertEqual(heatmap_image.size, (256, 256))
        self.assertEqual(overlay_image.size, (256, 256))

    def test_format_prediction_returns_label_scores_and_summary(self):
        label_scores, summary = format_prediction(0.73)

        self.assertEqual(label_scores["REAL"], 0.73)
        self.assertEqual(label_scores["FAKE"], 0.27)
        self.assertIn("REAL", summary)
        self.assertIn("73.00%", summary)


if __name__ == "__main__":
    unittest.main()
