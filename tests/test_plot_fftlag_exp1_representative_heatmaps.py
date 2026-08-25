"""Tests for Exp1 representative band-knockout heatmap plotting."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation_scripts.plot_fftlag_exp1_representative_heatmaps import (
    compute_shared_color_limits,
    extract_idx_exp1_long,
    long_to_pivot,
    long_to_wide,
    plot_exp1_band_knockout_heatmap,
    plot_exp1_representative_combined,
)


def _synthetic_idx_level(idx: int = 35, n_layers: int = 6, n_bands: int = 8) -> pd.DataFrame:
    rows = []
    for layer in range(n_layers):
        for band in range(n_bands):
            rows.append(
                {
                    "idx": idx,
                    "layer": layer,
                    "band": band,
                    "split": "test",
                    "mse_diff_mean": 0.1 * layer + 0.01 * band,
                    "mse_diff_std": 0.01,
                    "n_seeds": 10,
                }
            )
    return pd.DataFrame(rows)


def test_extract_idx_exp1_long_pivot_shape() -> None:
    df = _synthetic_idx_level()
    long_df = extract_idx_exp1_long(df, 35)
    pivot = long_to_pivot(long_df)

    assert len(long_df) == 48
    assert pivot.shape == (6, 8)
    assert list(pivot.columns) == [f"B{i}" for i in range(8)]


def test_long_to_wide_has_layer_column() -> None:
    long_df = extract_idx_exp1_long(_synthetic_idx_level(), 35)
    wide = long_to_wide(long_df)
    assert "layer" in wide.columns
    assert wide.shape == (6, 9)


def test_extract_idx_missing_returns_empty() -> None:
    df = _synthetic_idx_level(idx=35)
    assert extract_idx_exp1_long(df, 999).empty


def test_compute_shared_color_limits_absolute() -> None:
    """Exp1 heatmaps use absolute |ΔMSE|; color limits are [0, vmax]."""
    pivot = long_to_pivot(extract_idx_exp1_long(_synthetic_idx_level(), 35))
    vmin, vmax = compute_shared_color_limits([pivot])
    assert vmin == pytest.approx(0.0)
    assert vmax > 0


def test_plot_single_heatmap_writes_png(tmp_path: Path) -> None:
    long_df = extract_idx_exp1_long(_synthetic_idx_level(), 35)
    pivot = long_to_pivot(long_df)
    out_png = tmp_path / "heatmap.png"

    plot_exp1_band_knockout_heatmap(
        pivot,
        35,
        "e_coli",
        out_png,
        vmin=-1.0,
        vmax=1.0,
        n_seeds=10,
    )
    assert out_png.is_file() and out_png.stat().st_size > 0


def test_plot_combined_grid_writes_png(tmp_path: Path) -> None:
    panels = {
        "e_coli": [
            (35, long_to_pivot(extract_idx_exp1_long(_synthetic_idx_level(35), 35))),
            (1442, long_to_pivot(extract_idx_exp1_long(_synthetic_idx_level(1442), 1442))),
        ],
        "s_aureus": [
            (641, long_to_pivot(extract_idx_exp1_long(_synthetic_idx_level(641), 641))),
        ],
    }
    out_png = tmp_path / "combined.png"
    plot_exp1_representative_combined(
        panels,
        out_png,
        vmin=-1.0,
        vmax=1.0,
        n_seeds=10,
    )
    assert out_png.is_file() and out_png.stat().st_size > 0


def test_plot_combined_empty_skips(tmp_path: Path) -> None:
    out_png = tmp_path / "empty.png"
    plot_exp1_representative_combined({}, out_png, vmin=-1.0, vmax=1.0, n_seeds=10)
    assert not out_png.exists()
