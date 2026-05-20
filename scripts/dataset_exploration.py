from pathlib import Path
from collections import Counter
import random

from PIL import Image
import matplotlib.pyplot as plt


DATASET_DIR = Path("dataset")

SPLITS = ["train", "test"]
CLASSES = ["REAL", "FAKE"]


def count_images():
    print("=== Image Count ===")

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            image_paths = list(folder.glob("*.jpg"))

            print(f"{split}/{class_name}: {len(image_paths)} images")


def check_image_sizes(max_images_per_folder=1000):
    print("\n=== Image Sizes ===")

    sizes = Counter()

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            image_paths = list(folder.glob("*.jpg"))

            for path in image_paths[:max_images_per_folder]:
                image = Image.open(path)
                sizes[image.size] += 1

    for size, count in sizes.items():
        print(f"{size}: {count} images")


def check_corrupted_images():
    print("\n=== Corrupted Image Check ===")

    corrupted = []

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATASET_DIR / split / class_name
            image_paths = list(folder.glob("*.jpg"))

            for path in image_paths:
                try:
                    image = Image.open(path)
                    image.verify()
                except Exception:
                    corrupted.append(path)

    if len(corrupted) == 0:
        print("No corrupted images found.")
    else:
        print(f"Found {len(corrupted)} corrupted images:")
        for path in corrupted[:20]:
            print(path)


def show_samples(split="train", class_name="REAL", num_samples=8):
    folder = DATASET_DIR / split / class_name
    image_paths = list(folder.glob("*.jpg"))

    samples = random.sample(image_paths, num_samples)

    plt.figure(figsize=(12, 3))

    for index, path in enumerate(samples):
        image = Image.open(path)

        plt.subplot(1, num_samples, index + 1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis("off")

    plt.show()


def main():
    count_images()
    check_image_sizes()
    check_corrupted_images()

    show_samples(split="train", class_name="REAL")
    show_samples(split="train", class_name="FAKE")


if __name__ == "__main__":
    main()