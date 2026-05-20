from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from src.explainability.gradcam import GradCAM
from src.models.densenet import DenseNet121Binary


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor


def save_gradcam_result(original_image, heatmap, output_path, title):
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def collect_sample_paths(dataset_dir, class_name, num_samples):
    class_dir = dataset_dir / "test" / class_name
    image_paths = sorted(class_dir.glob("*.jpg"))

    return image_paths[:num_samples]


def main():
    dataset_dir = Path("dataset")
    checkpoint_path = Path("outputs/checkpoints/densenet121_best_model.pt")
    output_dir = Path("outputs/gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples_per_class = 5

    device = get_device()
    print(f"Using device: {device}", flush=True)

    model = DenseNet121Binary().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model.model.features.denseblock4
    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    sample_paths = []

    for class_name in ["REAL", "FAKE"]:
        paths = collect_sample_paths(
            dataset_dir=dataset_dir,
            class_name=class_name,
            num_samples=num_samples_per_class,
        )

        for path in paths:
            sample_paths.append((class_name, path))

    for class_name, image_path in sample_paths:
        original_image, image_tensor = load_image(image_path)
        image_tensor = image_tensor.to(device)

        with torch.enable_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output)[0, 0].item()

            heatmap = gradcam.generate(image_tensor)

        predicted_label = "REAL" if probability >= 0.5 else "FAKE"

        title = (
            f"True: {class_name} | "
            f"Pred: {predicted_label} | "
            f"P(REAL): {probability:.3f}"
        )

        output_name = (
            f"{class_name.lower()}_"
            f"{image_path.stem.replace(' ', '_')}_"
            f"pred_{predicted_label.lower()}.png"
        )

        output_path = output_dir / output_name

        save_gradcam_result(
            original_image=original_image,
            heatmap=heatmap,
            output_path=output_path,
            title=title,
        )

        print(f"Saved {output_path}", flush=True)

    gradcam.close()

    print("Grad-CAM generation finished.", flush=True)


if __name__ == "__main__":
    main()
