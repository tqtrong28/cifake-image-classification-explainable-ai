from pathlib import Path
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class CIFakeDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.class_to_label = {
            "FAKE": 0,
            "REAL": 1,
        }

        self.samples = []

        for class_name, label in self.class_to_label.items():
            class_dir = self.root_dir / split / class_name

            for image_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_train_val_subsets(train_dataset, val_dataset, val_ratio=0.2, seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices_by_label = {}

    for index, (_, label) in enumerate(train_dataset.samples):
        indices_by_label.setdefault(label, []).append(index)

    random_generator = random.Random(seed)
    train_indices = []
    val_indices = []

    for label_indices in indices_by_label.values():
        shuffled_indices = label_indices.copy()
        random_generator.shuffle(shuffled_indices)

        val_count = int(len(shuffled_indices) * val_ratio)

        val_indices.extend(shuffled_indices[:val_count])
        train_indices.extend(shuffled_indices[val_count:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(val_indices)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset


def create_dataloaders(
    root_dir,
    train_transform,
    val_transform,
    test_transform,
    batch_size=64,
    val_ratio=0.2,
    seed=42,
    num_workers=0,
):
    train_dataset = CIFakeDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform,
    )

    val_dataset = CIFakeDataset(
        root_dir=root_dir,
        split="train",
        transform=val_transform,
    )

    test_dataset = CIFakeDataset(
        root_dir=root_dir,
        split="test",
        transform=test_transform,
    )

    train_subset, val_subset = create_train_val_subsets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
