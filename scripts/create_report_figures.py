from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
sys.path.append(str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.explainability.heatmap_stats import heatmap_stats


FIGURE_DIR = PROJECT_ROOT / "reports/figures"
GRADCAM_DIR = PROJECT_ROOT / "outputs/gradcam_batch"
METADATA_PATH = GRADCAM_DIR / "metadata.csv"

MODELS = ["simple_cnn", "resnet18", "efficientnet_b0", "densenet121"]
MODEL_LABELS = {
    "simple_cnn": "SimpleCNN",
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet121",
}


def save_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    nodes = [
        ("CIFAKE\nDataset", 0.13, 0.72),
        ("Data Check +\nPreprocessing", 0.38, 0.72),
        ("Train / Val / Test\nSplit", 0.63, 0.72),
        ("Train CNN\nModels", 0.88, 0.72),
        ("Compare\nMetrics", 0.63, 0.32),
        ("Grad-CAM\nAnalysis", 0.38, 0.32),
        ("Gradio\nDemo", 0.13, 0.32),
    ]

    for label, x, y in nodes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.85", fc="#eef4ff", ec="#4f6fa8", lw=2.2),
            transform=ax.transAxes,
        )

    arrows = [
        ((0.22, 0.72), (0.29, 0.72)),
        ((0.47, 0.72), (0.54, 0.72)),
        ((0.72, 0.72), (0.79, 0.72)),
        ((0.84, 0.62), (0.68, 0.42)),
        ((0.54, 0.32), (0.47, 0.32)),
        ((0.29, 0.32), (0.22, 0.32)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=2.4, color="#333333"),
        )

    ax.set_title("Overall Project Pipeline", fontsize=19, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pipeline.png", dpi=180)
    plt.close(fig)


def save_dataset_samples():
    rows = []
    for class_name in ["REAL", "FAKE"]:
        paths = sorted((PROJECT_ROOT / "dataset/test" / class_name).glob("*.jpg"))[:6]
        rows.append((class_name, paths))

    fig, axes = plt.subplots(2, 6, figsize=(10, 3.6))

    for row_idx, (class_name, paths) in enumerate(rows):
        for col_idx, path in enumerate(paths):
            ax = axes[row_idx, col_idx]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=12, weight="bold")

    fig.suptitle("CIFAKE Dataset Samples", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dataset_samples.png", dpi=180)
    plt.close(fig)


def save_model_architectures():
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.axis("off")

    rows = [
        ("SimpleCNN", ["Input\n32x32", "Conv\nBlock 1", "Pool", "Conv\nBlock 2", "Pool", "FC\nHead", "Logit"]),
        ("ResNet18", ["Input\n32x32", "Conv\nStem", "Residual\nBlocks", "Global\nPool", "FC\nHead", "Logit"]),
        ("EfficientNet-B0", ["Input\n32x32", "Stem", "MBConv\nBlocks", "Global\nPool", "FC\nHead", "Logit"]),
        ("DenseNet121", ["Input\n32x32", "Conv\nStem", "Dense\nBlocks", "Global\nPool", "FC\nHead", "Logit"]),
    ]

    y_positions = np.linspace(0.8, 0.28, len(rows))

    for y, (model_name, blocks) in zip(y_positions, rows):
        ax.text(
            0.055,
            y,
            model_name,
            ha="left",
            va="center",
            fontsize=16,
            weight="bold",
            transform=ax.transAxes,
        )

        xs = np.linspace(0.24, 0.93, len(blocks))
        for i, (x, block) in enumerate(zip(xs, blocks)):
            ax.text(
                x,
                y,
                block,
                ha="center",
                va="center",
                fontsize=13,
                linespacing=1.15,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#555555", lw=1.6),
                transform=ax.transAxes,
            )
            if i < len(blocks) - 1:
                ax.annotate(
                    "",
                    xy=(xs[i + 1] - 0.052, y),
                    xytext=(x + 0.052, y),
                    xycoords=ax.transAxes,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.6,
                        color="#333333",
                        mutation_scale=18,
                    ),
                )

    ax.text(
        0.5,
        0.1,
        "Note: architecture blocks may include convolution, normalization, activation, and feature aggregation operations depending on the model.",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        transform=ax.transAxes,
    )

    ax.set_title("High-Level Model Architecture Comparison", fontsize=20, weight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_architectures.png", dpi=180)
    plt.close(fig)


def save_model_comparison():
    labels = ["SimpleCNN", "ResNet18", "EfficientNet-B0", "DenseNet121"]
    accuracy = [94.265, 94.800, 94.955, 95.120]
    f1 = [94.198, 94.836, 94.952, 95.159]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#5b8fd9")
    bars2 = ax.bar(x + width / 2, f1, width, label="F1-score", color="#f2a65a")

    ax.set_ylabel("Score (%)")
    ax.set_ylim(93.5, 95.5)
    ax.set_title("Model Performance Comparison on Final Test Set", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def overlay_heatmap(original, heatmap, alpha=0.45):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    colored_img = Image.fromarray(colored).resize(original.size, Image.Resampling.BILINEAR)
    return Image.blend(original, colored_img, alpha=alpha)


def save_gradcam_comparison(class_name, output_name):
    df = pd.read_csv(METADATA_PATH)
    base = df[(df["model"] == "densenet121") & (df["true_label"] == class_name)].iloc[0]
    image_path = str(base["image_path"])

    original = Image.open(PROJECT_ROOT / image_path).convert("RGB")
    fig, axes = plt.subplots(1, len(MODELS) + 1, figsize=(12, 2.8))

    axes[0].imshow(original.resize((128, 128), Image.Resampling.NEAREST))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for ax, model_name in zip(axes[1:], MODELS):
        row = df[(df["model"] == model_name) & (df["image_path"] == image_path)].iloc[0]
        heatmap = np.load(PROJECT_ROOT / row["heatmap_path"])
        overlay = overlay_heatmap(original, heatmap).resize((128, 128), Image.Resampling.NEAREST)
        ax.imshow(overlay)
        ax.set_title(
            f"{MODEL_LABELS[model_name]}\nP:{row['pred_label']} ({float(row['prob_real']):.2f})",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(f"Grad-CAM Comparison on a {class_name} Image", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=180)
    plt.close(fig)


def save_wrong_predictions():
    df = pd.read_csv(METADATA_PATH)
    wrong = df[(df["model"] == "densenet121") & (df["correct"] == 0)].head(4)

    if wrong.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No wrong DenseNet121 cases found in current Grad-CAM sample.", ha="center", va="center")
        fig.savefig(FIGURE_DIR / "gradcam_wrong.png", dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(wrong), figsize=(3.2 * len(wrong), 3.2))
    if len(wrong) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, wrong.iterrows()):
        original = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        heatmap = np.load(PROJECT_ROOT / row["heatmap_path"])
        overlay = overlay_heatmap(original, heatmap).resize((160, 160), Image.Resampling.NEAREST)
        ax.imshow(overlay)
        ax.set_title(
            f"T:{row['true_label']} P:{row['pred_label']}\nP(REAL)={float(row['prob_real']):.2f}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle("Wrong Predictions with DenseNet121 Grad-CAM", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gradcam_wrong.png", dpi=180)
    plt.close(fig)


def save_coverage_histogram():
    df = pd.read_csv(METADATA_PATH)
    records = []

    for _, row in df.iterrows():
        heatmap = np.load(PROJECT_ROOT / row["heatmap_path"])
        stats = heatmap_stats(heatmap)
        records.append({**row.to_dict(), **stats})

    stats_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, model_name in zip(axes, MODELS):
        sub = stats_df[stats_df["model"] == model_name]
        ax.hist(sub[sub["correct"] == 1]["coverage"], bins=20, alpha=0.7, label="Correct", color="#5b8fd9")
        if (sub["correct"] == 0).any():
            ax.hist(sub[sub["correct"] == 0]["coverage"], bins=20, alpha=0.7, label="Wrong", color="#f2a65a")
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle("Distribution of Grad-CAM Coverage Values", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "coverage_histogram.png", dpi=180)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_pipeline()
    save_dataset_samples()
    save_model_architectures()
    save_model_comparison()
    save_gradcam_comparison("REAL", "gradcam_real.png")
    save_gradcam_comparison("FAKE", "gradcam_fake.png")
    save_wrong_predictions()
    save_coverage_histogram()

    print(f"Saved report figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
