from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from torch import nn

from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transform, get_test_transform
from src.models.densenet import DenseNet121Binary
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def format_metrics(prefix, metrics):
    return (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_acc={metrics['accuracy']:.4f} "
        f"{prefix}_precision={metrics['precision']:.4f} "
        f"{prefix}_recall={metrics['recall']:.4f} "
        f"{prefix}_f1={metrics['f1']:.4f}"
    )


def main():
    dataset_dir = "dataset"
    batch_size = 64
    learning_rate = 0.001
    max_epochs = 50
    patience = 5
    val_ratio = 0.2
    seed = 42

    checkpoint_dir = Path("outputs/checkpoints")
    metrics_dir = Path("outputs/metrics")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / "densenet121_best_model.pt"
    metrics_path = metrics_dir / "densenet121_metrics.json"

    device = get_device()
    print(f"Using device: {device}", flush=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=dataset_dir,
        train_transform=get_train_transform(),
        val_transform=get_test_transform(),
        test_transform=get_test_transform(),
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=0,
    )

    model = DenseNet121Binary().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_result)

        print(
            f"Epoch [{epoch}/{max_epochs}] "
            f"{format_metrics('train', train_metrics)} "
            f"{format_metrics('val', val_metrics)}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model": "DenseNet121Binary",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "validation_metrics": val_metrics,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "val_ratio": val_ratio,
                    "seed": seed,
                },
                best_model_path,
            )

            print(f"Saved best model to {best_model_path}", flush=True)
        else:
            epochs_without_improvement += 1

            print(
                f"No validation improvement for "
                f"{epochs_without_improvement}/{patience} epoch(s)",
                flush=True,
            )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.", flush=True)
            break

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics = {
        "model": "DenseNet121Binary",
        "checkpoint_path": str(best_model_path),
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "final_test_metrics": test_metrics,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "val_ratio": val_ratio,
        "seed": seed,
        "device": str(device),
        "history": history,
    }

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Training finished.", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Final test metrics: {format_metrics('test', test_metrics)}", flush=True)
    print(f"Best model path: {best_model_path}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
