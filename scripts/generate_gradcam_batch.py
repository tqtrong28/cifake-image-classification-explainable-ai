from pathlib import Path
import sys
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.explainability.gradcam import GradCAM
from src.models.cnn import SimpleCNN
from src.models.resnet import ResNet18Binary
from src.models.efficientnet import EfficientNetB0Binary
from src.models.densenet import DenseNet121Binary


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MODEL_REGISTRY = {
    "simple_cnn": {
        "cls": SimpleCNN,
        "ckpt": "outputs/checkpoints/simple_cnn_best_model.pt",
        "target_layer": lambda m: m.features,
    },
    "resnet18": {
        "cls": ResNet18Binary,
        "ckpt": "outputs/checkpoints/resnet18_best_model.pt",
        "target_layer": lambda m: m.model.layer2,
    },
    "efficientnet_b0": {
        "cls": EfficientNetB0Binary,
        "ckpt": "outputs/checkpoints/efficientnet_b0_best_model.pt",
        "target_layer": lambda m: m.model.features[3],
    },
    "densenet121": {
        "cls": DenseNet121Binary,
        "ckpt": "outputs/checkpoints/densenet121_best_model.pt",
        "target_layer": lambda m: m.model.features.denseblock2,
    },
}


def collect_samples(dataset_dir: Path, num_per_class: int):
    samples = []
    for class_name in ["REAL", "FAKE"]:
        paths = sorted((dataset_dir / "test" / class_name).glob("*.jpg"))
        for p in paths[:num_per_class]:
            samples.append((class_name, p))
    return samples


def load_image(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img, tensor


def save_overlay(original_image, heatmap, path: Path, title: str):
    plt.figure(figsize=(3.2, 3.2))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title(title, fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main(num_per_class: int = 50):
    dataset_dir = Path("dataset")
    output_root = Path("outputs/gradcam_batch")
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}", flush=True)

    samples = collect_samples(dataset_dir, num_per_class)
    print(f"Samples per model: {len(samples)}", flush=True)

    csv_path = output_root / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "image_path", "true_label", "pred_label",
            "prob_real", "correct", "heatmap_path", "png_path",
        ])

        for model_name, cfg in MODEL_REGISTRY.items():
            print(f"\n=== {model_name} ===", flush=True)
            model_dir = output_root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            model = cfg["cls"]().to(device)
            ckpt = torch.load(cfg["ckpt"], map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            target_layer = cfg["target_layer"](model)
            gradcam = GradCAM(model=model, target_layer=target_layer)

            for class_name, image_path in samples:
                original, tensor = load_image(image_path)
                tensor = tensor.to(device)

                with torch.enable_grad():
                    output = model(tensor)
                    prob_real = float(torch.sigmoid(output)[0, 0].item())
                    heatmap = gradcam.generate(tensor).numpy()

                pred_label = "REAL" if prob_real >= 0.5 else "FAKE"
                correct = (pred_label == class_name)

                stem = f"{class_name.lower()}_{image_path.stem.replace(' ', '_')}"
                png_path = model_dir / f"{stem}.png"
                npy_path = model_dir / f"{stem}.npy"

                np.save(npy_path, heatmap.astype(np.float32))
                save_overlay(
                    original, heatmap, png_path,
                    f"T:{class_name} P:{pred_label} ({prob_real:.2f})",
                )

                writer.writerow([
                    model_name, str(image_path), class_name, pred_label,
                    f"{prob_real:.4f}", int(correct),
                    str(npy_path), str(png_path),
                ])

            gradcam.close()
            print(f"  done {model_name}", flush=True)

    print(f"\nMetadata: {csv_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=50)
    args = parser.parse_args()
    main(num_per_class=args.num)
