from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from torch import nn

from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transform, get_test_transform
from src.models.efficientnet import EfficientNetB0Binary
from src.training.evaluate import evaluate


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    dataset_dir = "dataset"
    batch_size = 64
    val_ratio = 0.2
    seed = 42

    checkpoint_path = Path("outputs/checkpoints/efficientnet_b0_best_model.pt")
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "efficientnet_b0_test_metrics.json"

    device = get_device()
    print(f"Using device: {device}", flush=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, _, test_loader = create_dataloaders(
        root_dir=dataset_dir,
        train_transform=get_train_transform(),
        val_transform=get_test_transform(),
        test_transform=get_test_transform(),
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=0,
    )

    model = EfficientNetB0Binary().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = {
        "model": "EfficientNetB0Binary",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
        "final_test_metrics": test_metrics,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(device),
    }

    print("Test evaluation result:", flush=True)
    print(f"test_loss={test_metrics['loss']:.4f}", flush=True)
    print(f"test_acc={test_metrics['accuracy']:.4f}", flush=True)
    print(f"test_precision={test_metrics['precision']:.4f}", flush=True)
    print(f"test_recall={test_metrics['recall']:.4f}", flush=True)
    print(f"test_f1={test_metrics['f1']:.4f}", flush=True)

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
