# maintained by kewei li
"""Path helpers for FFT-LAG mechanism aggregated outputs and figures."""
from __future__ import annotations

from pathlib import Path


def dataset_agg_root(analysis_root: Path, dataset: str) -> Path:
    return analysis_root / "aggregated" / dataset


def exp_agg_dir(analysis_root: Path, dataset: str, exp: str) -> Path:
    return dataset_agg_root(analysis_root, dataset) / exp


def figures_root(analysis_root: Path) -> Path:
    """Unified figure tree: ``{analysis_root}/figures``."""
    return analysis_root / "figures"


def exp_figures_dir(
    analysis_root: Path,
    exp: str,
    dataset: str | None = None,
) -> Path:
    """
    Figure directory for one experiment.

    - ``dataset is None`` or ``"combined"`` → ``figures/expN/combined``
    - otherwise → ``figures/expN/{dataset}``
    """
    base = figures_root(analysis_root) / exp
    if dataset is None or dataset == "combined":
        return base / "combined"
    return base / dataset


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
