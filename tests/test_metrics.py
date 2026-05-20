import unittest

from src.utils.metrics import calculate_binary_metrics


class BinaryMetricsTest(unittest.TestCase):
    def test_calculate_binary_metrics(self):
        metrics = calculate_binary_metrics(
            total_loss=2.0,
            total_samples=4,
            true_positive=1,
            true_negative=1,
            false_positive=1,
            false_negative=1,
        )

        self.assertEqual(metrics["loss"], 0.5)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
