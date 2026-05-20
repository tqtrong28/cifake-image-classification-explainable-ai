from pathlib import Path
import errno
import os
import socket

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from src.explainability.gradcam import GradCAM
from src.models.densenet import DenseNet121Binary


CHECKPOINT_PATH = PROJECT_ROOT / "outputs/checkpoints/densenet121_best_model.pt"
MODEL_NAME = "DenseNet121"
IMAGE_SIZE = (32, 32)
DISPLAY_SIZE = (256, 256)

_TO_TENSOR = transforms.ToTensor()
_MODEL_BUNDLE = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model_bundle():
    global _MODEL_BUNDLE

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT_PATH}. "
            "Train DenseNet121 before launching the demo."
        )

    device = get_device()
    model = DenseNet121Binary()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = model.model.features.denseblock2
    gradcam = GradCAM(model=model, target_layer=target_layer)

    _MODEL_BUNDLE = model, gradcam, device
    return _MODEL_BUNDLE


def prepare_image(image, device):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    tensor = _TO_TENSOR(image).unsqueeze(0).to(device)
    return tensor, image


def normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()
    heatmap_min = float(heatmap.min())
    heatmap_max = float(heatmap.max())

    if heatmap_max > heatmap_min:
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    return np.clip(heatmap, 0.0, 1.0)


def build_heatmap_images(original_image, heatmap):
    heatmap = normalize_heatmap(heatmap)
    color_map = colormaps.get_cmap("jet")
    heatmap_rgb = (color_map(heatmap)[:, :, :3] * 255).astype(np.uint8)

    heatmap_image = Image.fromarray(heatmap_rgb)
    heatmap_image = heatmap_image.resize(DISPLAY_SIZE, Image.Resampling.BILINEAR)

    original_display = original_image.resize(DISPLAY_SIZE, Image.Resampling.NEAREST)
    overlay_image = Image.blend(original_display, heatmap_image, alpha=0.45)

    return heatmap_image, overlay_image


def format_prediction(prob_real):
    prob_real = float(prob_real)
    prob_fake = 1.0 - prob_real
    pred_label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if pred_label == "REAL" else prob_fake

    label_scores = {
        "REAL": round(prob_real, 4),
        "FAKE": round(prob_fake, 4),
    }

    summary = (
        f"Dự đoán: {pred_label}\n"
        f"Độ tự tin: {confidence:.2%}\n"
        f"P(REAL): {prob_real:.2%}\n"
        f"P(FAKE): {prob_fake:.2%}"
    )

    return label_scores, summary


def predict_image(image):
    if image is None:
        return {}, "Hãy upload một ảnh trước nhé.", None, None

    model, gradcam, device = load_model_bundle()
    image_tensor, display_image = prepare_image(image, device)

    with torch.no_grad():
        output = model(image_tensor)
        prob_real = float(torch.sigmoid(output)[0, 0].item())

    target_class = "real" if prob_real >= 0.5 else "fake"

    with torch.enable_grad():
        heatmap = gradcam.generate(
            image_tensor,
            target_class=target_class,
        ).numpy()

    heatmap_image, overlay_image = build_heatmap_images(display_image, heatmap)
    label_scores, summary = format_prediction(prob_real)

    return label_scores, summary, heatmap_image, overlay_image


def get_example_images():
    examples = []

    for class_name in ["REAL", "FAKE"]:
        class_dir = PROJECT_ROOT / "dataset/test" / class_name

        if class_dir.exists():
            image_paths = sorted(class_dir.glob("*.jpg"))

            if image_paths:
                examples.append(str(image_paths[0]))

    return examples


def build_demo():
    with gr.Blocks(title="CIFAKE DenseNet121 Demo") as demo:
        gr.Markdown(
            "# CIFAKE Real/Fake Image Detector\n"
            "Demo dùng model DenseNet121 tốt nhất của project và Grad-CAM "
            "để xem model đang chú ý vùng nào."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Ảnh đầu vào",
                    type="pil",
                    height=320,
                )
                predict_button = gr.Button("Dự đoán", variant="primary")

                examples = get_example_images()
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=image_input,
                    )

            with gr.Column():
                label_output = gr.Label(
                    label="Xác suất dự đoán",
                    num_top_classes=2,
                )
                summary_output = gr.Textbox(
                    label="Kết quả",
                    lines=4,
                    interactive=False,
                )

        with gr.Row():
            heatmap_output = gr.Image(
                label="Heatmap Grad-CAM",
                type="pil",
                height=320,
            )
            overlay_output = gr.Image(
                label="Overlay Grad-CAM",
                type="pil",
                height=320,
            )

        predict_button.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[
                label_output,
                summary_output,
                heatmap_output,
                overlay_output,
            ],
        )

    return demo


def find_free_port(start_port=7860, end_port=8059):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EPERM}:
                    raise PermissionError(
                        "Cannot bind a local Gradio port from this sandbox. "
                        "Run python app.py from your normal terminal."
                    ) from exc

                continue

            return port

    raise RuntimeError(f"No free port found from {start_port} to {end_port}.")


def get_launch_port():
    env_port = os.environ.get("GRADIO_SERVER_PORT")

    if env_port is not None:
        return int(env_port)

    return find_free_port()


if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name="127.0.0.1",
        server_port=get_launch_port(),
        show_error=True,
    )
