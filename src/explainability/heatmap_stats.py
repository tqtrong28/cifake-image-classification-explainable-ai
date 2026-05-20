import numpy as np


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    if heatmap_max > heatmap_min:
        return (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    return np.zeros_like(heatmap)


def heatmap_stats(heatmap: np.ndarray, threshold: float = 0.5) -> dict:
    heatmap = normalize_heatmap(heatmap)

    h, w = heatmap.shape
    total = heatmap.sum()

    coverage = float((heatmap >= threshold).mean())
    peak_value = float(heatmap.max())

    if total > 0:
        ys, xs = np.mgrid[0:h, 0:w]
        center_mass_y = float((ys * heatmap).sum() / total / (h - 1))
        center_mass_x = float((xs * heatmap).sum() / total / (w - 1))
    else:
        center_mass_x = 0.5
        center_mass_y = 0.5

    flat = heatmap.flatten()
    flat_sum = flat.sum()
    if flat_sum > 0:
        p = flat / flat_sum
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum())
        normalized_entropy = float(entropy / np.log(flat.size))
    else:
        entropy = 0.0
        normalized_entropy = 0.0

    return {
        "coverage": coverage,
        "center_mass_x": center_mass_x,
        "center_mass_y": center_mass_y,
        "peak_value": peak_value,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }
