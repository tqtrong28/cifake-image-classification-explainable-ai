import numpy as np
import pytest
from src.explainability.heatmap_stats import heatmap_stats


def test_uniform_heatmap_has_center_mass_at_middle():
    heatmap = np.ones((32, 32), dtype=np.float32) * 0.7
    stats = heatmap_stats(heatmap)
    assert stats["coverage"] == pytest.approx(1.0)
    assert stats["center_mass_x"] == pytest.approx(0.5, abs=0.02)
    assert stats["center_mass_y"] == pytest.approx(0.5, abs=0.02)
    assert stats["peak_value"] == pytest.approx(0.7)


def test_corner_hotspot_shifts_center_mass():
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[0:4, 0:4] = 1.0
    stats = heatmap_stats(heatmap)
    assert stats["center_mass_x"] < 0.2
    assert stats["center_mass_y"] < 0.2
    assert stats["coverage"] < 0.05


def test_entropy_uniform_higher_than_spike():
    uniform = np.ones((32, 32), dtype=np.float32) * 0.5
    spike = np.zeros((32, 32), dtype=np.float32)
    spike[16, 16] = 1.0
    assert heatmap_stats(uniform)["entropy"] > heatmap_stats(spike)["entropy"]
