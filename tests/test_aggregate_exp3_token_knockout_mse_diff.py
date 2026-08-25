"""Tests for Exp3 token knockout |ΔMSE| aggregation (mse_diff pipeline)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation_scripts.aggregate_exp3_token_knockout_mse_diff import (
    aggregate_per_peptide_response_std,
    aggregate_seed_mean,
    gather_raw_csvs,
    summarize_distribution,
)

_KO = "knockout_lastlayer_HS.csv"


def _write_seed_csv(path: Path, entries: list[tuple[int, int, float, float]]) -> None:
    """Write CSV with (idx, token_pos, pred_base, pred_ko) rows."""
    rows = []
    for idx, pos, pred_base, pred_ko in entries:
        rows.append(
            {
                "idx": idx,
                "peptide": f"PEP{idx}",
                "seq_len": 10,
                "token_pos": pos,
                "rel_pos": (pos - 1) / 7.0,
                "pred_base": pred_base,
                "pred_ko": pred_ko,
                "layer": 5,
                "pooling": "FLaG",
                "model_version": "esm2_t6",
                "dataset": "s_aureus",
                "split": "test",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def exp3_root(tmp_path: Path) -> Path:
    root = tmp_path / "exp3_fulltest"
    pool = "FLaG"
    ds = "s_aureus"
    positions = [2, 3, 4]
    # y=0 so abs_mse_diff = |pred_ko^2 - pred_base^2|
    for seed, offset in ((0, 0.0), (1, 0.05)):
        entries = []
        for idx, base in ((1, 0.1), (2, 0.2)):
            for pos in positions:
                pred_base = 1.0
                pred_ko = 1.0 + base + 0.01 * pos + offset
                entries.append((idx, pos, pred_base, pred_ko))
        _write_seed_csv(root / pool / f"seed_{seed}" / ds / _KO, entries)
    return root


@pytest.fixture
def labels() -> dict[str, pd.DataFrame]:
    return {
        "s_aureus": pd.DataFrame({"idx": [1, 2], "y": [0.0, 0.0]}),
    }


def test_gather_raw_csvs(exp3_root: Path, labels: dict[str, pd.DataFrame]) -> None:
    raw = gather_raw_csvs(exp3_root, ["s_aureus"], labels)
    assert not raw.empty
    assert set(raw["train_seed"].unique()) == {0, 1}
    assert raw["pooling"].iloc[0] == "FLaG"
    assert "abs_mse_diff" in raw.columns
    assert (raw["abs_mse_diff"] >= 0).all()


def test_aggregate_seed_mean(exp3_root: Path, labels: dict[str, pd.DataFrame]) -> None:
    raw = gather_raw_csvs(exp3_root, ["s_aureus"], labels)
    agg = aggregate_seed_mean(raw)
    assert len(agg) == 2 * 3  # 2 idx × 3 positions
    row = agg[(agg["idx"] == 1) & (agg["token_pos"] == 2)].iloc[0]
    assert row["n_seeds"] == 2
    assert row["abs_mse_diff"] > 0.0


def test_aggregate_per_peptide_response_std(
    exp3_root: Path, labels: dict[str, pd.DataFrame]
) -> None:
    raw = gather_raw_csvs(exp3_root, ["s_aureus"], labels)
    position_long = aggregate_seed_mean(raw)
    pep_df = aggregate_per_peptide_response_std(position_long)
    assert len(pep_df) == 2
    row1 = pep_df[pep_df["idx"] == 1].iloc[0]
    row2 = pep_df[pep_df["idx"] == 2].iloc[0]
    assert row1["n_positions"] == 3
    assert row2["n_positions"] == 3
    assert row1["response_std"] > 0.0
    assert row2["response_std"] > 0.0


def test_summarize_distribution(exp3_root: Path, labels: dict[str, pd.DataFrame]) -> None:
    raw = gather_raw_csvs(exp3_root, ["s_aureus"], labels)
    position_long = aggregate_seed_mean(raw)
    long_df = aggregate_per_peptide_response_std(position_long)
    summary = summarize_distribution(long_df)
    assert len(summary) == 1
    assert summary.iloc[0]["pooling"] == "FLaG"
    assert summary.iloc[0]["n_peptides"] == 2
    assert summary.iloc[0]["n_obs"] == 2
