import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dataset import create_dataloaders
from src.data.transforms import get_test_transform, get_train_transform


def create_image(path):
    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    image.save(path)


class DataLoaderSplitTest(unittest.TestCase):
    def test_create_dataloaders_returns_stratified_train_val_test_loaders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)

            for split in ["train", "test"]:
                for class_name in ["FAKE", "REAL"]:
                    class_dir = root_dir / split / class_name
                    class_dir.mkdir(parents=True)

            for class_name in ["FAKE", "REAL"]:
                for index in range(10):
                    create_image(root_dir / "train" / class_name / f"{index}.jpg")

                for index in range(4):
                    create_image(root_dir / "test" / class_name / f"{index}.jpg")

            train_loader, val_loader, test_loader = create_dataloaders(
                root_dir=root_dir,
                train_transform=get_train_transform(),
                val_transform=get_test_transform(),
                test_transform=get_test_transform(),
                batch_size=4,
                val_ratio=0.2,
                seed=42,
                num_workers=0,
            )

            self.assertEqual(len(train_loader.dataset), 16)
            self.assertEqual(len(val_loader.dataset), 4)
            self.assertEqual(len(test_loader.dataset), 8)

            images, labels = next(iter(train_loader))

            self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
            self.assertEqual(tuple(labels.shape), (4,))


if __name__ == "__main__":
    unittest.main()
