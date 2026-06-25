import importlib.util
import unittest

import torch

from src.models.cnn import SimpleCNN


class DeiTTinyModelTest(unittest.TestCase):
    def test_deit_tiny_binary_model_is_available_or_reports_missing_dependency(self):
        if importlib.util.find_spec("timm") is None:
            from src.models.deit import DeiTTinyBinary

            with self.assertRaisesRegex(ImportError, "timm"):
                DeiTTinyBinary()
            return

        from src.models.deit import DeiTTinyBinary

        model = DeiTTinyBinary()
        output = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(tuple(output.shape), (2, 1))


class SimpleCNNModelTest(unittest.TestCase):
    def test_simple_cnn_forward_shape(self):
        model = SimpleCNN()
        output = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(tuple(output.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
