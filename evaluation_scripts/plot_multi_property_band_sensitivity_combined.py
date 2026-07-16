#!/usr/bin/env python3
# maintained by kewei li
"""Combine helix / net charge / hydrophobicity band-sensitivity into one 1x3 subplot figure."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for p in (_REPO_PARENT, _REPO_ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from AMPCliff.dc_property_utils import left_aligned_bucket_chart_title
from analyze_fft_lag_mechanism_by_structure import (
    _BUCKET_LABELS,
    _BUCKET_ORDER,
    _plot_bucket_band_combined,
    _property_bucket_labels,
    _property_bucket_order,
    _property_bucket_xlabel,
)

PROPERTY_DISPLAY = {
    "helix_propensity": "Helix propensity",
    "net_charge": "Net charge",
    "mean_hydrophobicity": "Mean hydrophobicity",
}


@dataclass(frozen=True)
class StratificationPanel:
    key: str
    bucket_col: str
    sample_level: pd.DataFrame
    by_band: pd.DataFrame
    summary: pd.DataFrame


def _sample_level_from_per_sample(sample_band: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    return (
        sample_band.groupby(["idx", bucket_col], as_index=False)
        .agg(mean_abs_mse_diff=("mse_diff_abs_mean", "mean"))
    )


def load_helix_panel(exp5_species_dir: Path) -> StratificationPanel:
    summary = pd.read_csv(exp5_species_dir / "bucketwise_band_sensitivity_summary.csv")
    by_band = pd.read_csv(exp5_species_dir / "bucketwise_band_sensitivity_by_band.csv")
    sample_band = pd.read_csv(exp5_species_dir / "per_sample_band_sensitivity_by_structure.csv")
    return StratificationPanel(
        key="helix_propensity",
        bucket_col="structure_bucket",
        sample_level=_sample_level_from_per_sample(sample_band, "structure_bucket"),
        by_band=by_band,
        summary=summary,
    )


def load_property_panel(intermediate_dir: Path, property_col: str) -> StratificationPanel:
    summary = pd.read_csv(intermediate_dir / f"bucketwise_band_sensitivity_summary_{property_col}.csv")
    by_band = pd.read_csv(intermediate_dir / f"bucketwise_band_sensitivity_by_band_{property_col}.csv")
    sample_band = pd.read_csv(intermediate_dir / f"per_sample_band_sensitivity_by_{property_col}.csv")
    return StratificationPanel(
        key=property_col,
        bucket_col="property_bucket",
        sample_level=_sample_level_from_per_sample(sample_band, "property_bucket"),
        by_band=by_band,
        summary=summary,
    )


def _panel_plot_kwargs(panel: StratificationPanel) -> dict:
    if panel.key == "helix_propensity":
        return {
            "bucket_col": "structure_bucket",
            "bucket_order": _BUCKET_ORDER,
            "label_map": _BUCKET_LABELS,
            "xlabel": "Helix propensity bucket",
        }
    return {
        "bucket_col": "property_bucket",
        "bucket_order": _property_bucket_order(panel.key),
        "label_map": _property_bucket_labels(panel.key),
        "xlabel": _property_bucket_xlabel(panel.key),
    }


def plot_multi_property_bucket_band_combined(
    panels: Sequence[StratificationPanel],
    out_png: Path,
    *,
    species: str,
    title: str | None = None,
) -> Path:
    if not panels:
        raise ValueError("panels must not be empty")

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        is_last = idx == len(panels) - 1
        _plot_bucket_band_combined(
            panel.sample_level,
            panel.by_band,
            bar_y_col="mean_abs_mse_diff",
            line_y_col="mse_diff_abs_mean",
            out_png=None,
            title=PROPERTY_DISPLAY.get(panel.key, panel.key),
            ylabel="Mean |MSE diff|" if idx == 0 else "",
            ax=ax,
            show_legend=is_last,
            **_panel_plot_kwargs(panel),
        )
        if is_last:
            legend = ax.get_legend()
            if legend is not None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                legend.remove()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Band",
            loc="upper left",
            bbox_to_anchor=(0.92, 0.95),
        )

    fig.suptitle(
        title or left_aligned_bucket_chart_title(species, "Band sensitivity by property bucket"),
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.9, 0.94])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def load_panels_for_species(
    species: str,
    *,
    exp5_root: Path,
    property_ko_root: Path,
) -> list[StratificationPanel]:
    exp5_dir = exp5_root / species
    intermediate_dir = property_ko_root / species / "intermediate"
    for path in (exp5_dir, intermediate_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing directory: {path}")
    return [
        load_helix_panel(exp5_dir),
        load_property_panel(intermediate_dir, "net_charge"),
        load_property_panel(intermediate_dir, "mean_hydrophobicity"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=["e_coli", "s_aureus"])
    ap.add_argument(
        "--exp5-dir",
        type=Path,
        default=Path("outputs/analysis/fftlag_mechanism/exp5_structure_fulltest"),
    )
    ap.add_argument(
        "--property-ko-dir",
        type=Path,
        default=Path("outputs/analysis/dc_validation/property_dc_knockout_fulltest"),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/analysis/dc_validation/band_sensitivity_combined"),
    )
    args = ap.parse_args()

    written = []
    for species in args.species:
        panels = load_panels_for_species(
            species,
            exp5_root=args.exp5_dir,
            property_ko_root=args.property_ko_dir,
        )
        out_png = args.output_root / species / "multi_property_bucketwise_band_sensitivity_combined.png"
        plot_multi_property_bucket_band_combined(panels, out_png, species=species)
        written.append(out_png)
        print(f"[OK] {species} -> {out_png}")

    print(f"Wrote {len(written)} combined figures under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
