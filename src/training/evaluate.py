import torch

from src.utils.metrics import calculate_binary_metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).float()

            true_positive += ((predictions == 1) & (labels == 1)).sum().item()
            true_negative += ((predictions == 0) & (labels == 0)).sum().item()
            false_positive += ((predictions == 1) & (labels == 0)).sum().item()
            false_negative += ((predictions == 0) & (labels == 1)).sum().item()
            total_samples += batch_size

    return calculate_binary_metrics(
        total_loss=total_loss,
        total_samples=total_samples,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )
