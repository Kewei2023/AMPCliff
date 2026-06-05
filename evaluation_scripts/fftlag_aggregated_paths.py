"""Path helpers for FFT-LAG mechanism aggregated outputs."""
from __future__ import annotations

from pathlib import Path


def dataset_agg_root(analysis_root: Path, dataset: str) -> Path:
    return analysis_root / "aggregated" / dataset


def exp_agg_dir(analysis_root: Path, dataset: str, exp: str) -> Path:
    return dataset_agg_root(analysis_root, dataset) / exp


def resolve_aggregated_csv(
    agg_dataset_dir: Path,
    exp: str,
    filename: str,
    *,
    legacy_filename: str | None = None,
) -> Path:
    """Prefer new layout ``{exp}/{filename}``, fall back to legacy flat name."""
    new_path = agg_dataset_dir / exp / filename
    if new_path.is_file():
        return new_path
    if legacy_filename:
        legacy = agg_dataset_dir / legacy_filename
        if legacy.is_file():
            return legacy
    return new_path
