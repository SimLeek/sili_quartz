"""The world-model probe's ridge score (no model needed)."""

import importlib.util
from pathlib import Path

import torch


def _probe():
    path = Path(__file__).resolve().parents[1] / "tools" / "world_model_probe.py"
    spec = importlib.util.spec_from_file_location("world_model_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ridge_r2_is_high_for_a_linear_map_and_near_zero_for_noise():
    ridge_r2 = _probe().ridge_r2
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 20, generator=g, dtype=torch.float64)
    w = torch.randn(20, 8, generator=g, dtype=torch.float64)
    y = x @ w + 0.1 * torch.randn(600, 8, generator=g, dtype=torch.float64)
    assert ridge_r2(x[:400], y[:400], x[400:], y[400:]) > 0.95
    noise = torch.randn(600, 8, generator=g, dtype=torch.float64)
    assert abs(ridge_r2(x[:400], noise[:400], x[400:], noise[400:])) < 0.1
