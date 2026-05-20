def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_binary_metrics(
    total_loss,
    total_samples,
    true_positive,
    true_negative,
    false_positive,
    false_negative,
):
    average_loss = safe_divide(total_loss, total_samples)
    accuracy = safe_divide(true_positive + true_negative, total_samples)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
