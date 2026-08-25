"""Tests for Exp1 full-test band knockout violin plotting helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation_scripts.plot_fftlag_exp1_fulltest_violin import (
    _aggregate_seed_to_idx,
    build_violin_long_df,
    plot_exp1_band_knockout_violin_layer,
    summarize_violin_long,
)


def _make_idx_level(n_idx: int = 4, layers: list[int] | None = None, bands: list[int] | None = None) -> pd.DataFrame:
    layers = layers or [0, 1]
    bands = bands or [0, 1, 2]
    rows = []
    for idx in range(n_idx):
        for layer in layers:
            for band in bands:
                rows.append(
                    {
                        "idx": idx + 1,
                        "layer": layer,
                        "band": band,
                        "split": "test",
                        "mse_diff_mean": 0.1 * layer + 0.01 * band + 0.001 * idx,
                        "mse_diff_std": 0.01,
                        "n_seeds": 2,
                    }
                )
    return pd.DataFrame(rows)


def test_build_violin_long_df_shape_and_columns() -> None:
    idx_level = _make_idx_level(n_idx=3, layers=[0, 1, 2], bands=list(range(8)))
    long_df = build_violin_long_df(idx_level)

    assert len(long_df) == 3 * 3 * 8
    assert set(long_df["layer"]) == {0, 1, 2}
    assert set(long_df["band"]) == set(range(8))
    assert long_df["band_label"].tolist()[0] == "B0"
    assert "mse_diff" in long_df.columns


def test_build_violin_long_df_filters_non_test_split() -> None:
    idx_level = _make_idx_level(n_idx=2)
    extra = idx_level.copy()
    extra["split"] = "valid"
    combined = pd.concat([idx_level, extra], ignore_index=True)

    long_df = build_violin_long_df(combined)
    assert len(long_df) == len(idx_level)


def test_build_violin_long_df_empty_returns_empty() -> None:
    assert build_violin_long_df(pd.DataFrame()).empty
    assert build_violin_long_df(_make_idx_level()).shape[0] > 0


def test_summarize_violin_long_stats() -> None:
    long_df = build_violin_long_df(_make_idx_level(n_idx=5, layers=[0], bands=[0, 1]))
    summary = summarize_violin_long(long_df)

    assert len(summary) == 2
    assert summary.loc[summary["band"] == 0, "n"].iloc[0] == 5
    assert {"median", "q1", "q3", "mean", "std"}.issubset(summary.columns)


def test_aggregate_seed_to_idx() -> None:
    frames = []
    for seed in (0, 1):
        df = pd.DataFrame(
            [
                {"idx": 1, "layer": 0, "band": 0, "split": "test", "mse_diff": 0.2 + seed * 0.1},
                {"idx": 2, "layer": 0, "band": 0, "split": "test", "mse_diff": 0.4 + seed * 0.1},
            ]
        )
        df["seed"] = seed
        frames.append(df)

    agg = _aggregate_seed_to_idx(frames)
    assert len(agg) == 2
    assert agg["n_seeds"].max() == 2
    row = agg[(agg["idx"] == 1) & (agg["layer"] == 0) & (agg["band"] == 0)]
    assert pytest.approx(row["mse_diff_mean"].iloc[0]) == 0.25


def test_plot_exp1_band_knockout_violin_layer_writes_png(tmp_path: Path) -> None:
    long_df = build_violin_long_df(_make_idx_level(n_idx=6, layers=[0], bands=[0, 1, 2]))
    out_png = tmp_path / "layer0.png"

    ok = plot_exp1_band_knockout_violin_layer(
        long_df,
        layer=0,
        dataset="e_coli",
        out_png=out_png,
        n_seeds=2,
    )
    assert ok is True
    assert out_png.is_file() and out_png.stat().st_size > 0


def test_plot_exp1_band_knockout_violin_layer_empty_skips(tmp_path: Path) -> None:
    long_df = build_violin_long_df(_make_idx_level(layers=[1], bands=[0]))
    out_png = tmp_path / "missing_layer.png"

    ok = plot_exp1_band_knockout_violin_layer(
        long_df,
        layer=99,
        dataset="e_coli",
        out_png=out_png,
        n_seeds=2,
    )
    assert ok is False
    assert not out_png.exists()
