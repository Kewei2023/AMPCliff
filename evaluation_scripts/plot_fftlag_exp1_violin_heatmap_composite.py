#!/usr/bin/env python3
"""Exp1 composite figure: full-test |ΔP| violins + representative heatmaps (paper layout)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from plot_fftlag_exp1_fulltest_violin import (
    DEFAULT_AGGREGATED_SUBDIRS,
    DEFAULT_DATASETS,
    DEFAULT_EXP1_SUBDIR,
    DEFAULT_SEEDS,
    VALUE_COL as VIOLIN_VALUE_COL,
    build_violin_long_df,
    load_idx_level_exp1,
)
from plot_fftlag_exp1_representative_heatmaps import (
    REPRESENTATIVE_IDX,
    VALUE_COL as HEATMAP_VALUE_COL,
    extract_idx_exp1_long,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

FONT_SCALE = 1.5
COMPOSITE_PNG = "exp1_abs_mse_violin_heatmap_composite.png"
COMPOSITE_SVG = "exp1_abs_mse_violin_heatmap_composite.svg"


def fs(size: float) -> float:
    return size * FONT_SCALE


def _species_label(dataset: str) -> str:
    return r"$\it{E.\ coli}$" if dataset == "e_coli" else r"$\it{S.\ aureus}$"


def _band_labels(n_bands: int = 8) -> list[str]:
    return [rf"$\mathcal{{B}}_{{{i}}}$" for i in range(n_bands)]


def _parse_representative_idx(raw: Optional[str]) -> Dict[str, list[int]]:
    if not raw:
        return {k: list(v) for k, v in REPRESENTATIVE_IDX.items()}
    import json

    parsed = json.loads(raw)
    return {str(k): [int(i) for i in v] for k, v in parsed.items()}


def _load_violin_data(
    analysis_root: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
) -> dict[tuple[str, int, int], list[float]]:
    violin_data: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for dataset in datasets:
        idx_level, _src = load_idx_level_exp1(
            analysis_root, dataset, seeds, exp1_subdir, aggregated_subdirs
        )
        long_df = build_violin_long_df(idx_level)
        if long_df.empty:
            print(f"[WARN] empty Exp1 violin data for {dataset}")
            continue
        for row in long_df.itertuples(index=False):
            violin_data[(dataset, int(row.layer), int(row.band))].append(
                float(getattr(row, VIOLIN_VALUE_COL))
            )
    return violin_data


def _load_heatmap_data(
    analysis_root: Path,
    representative_idx: Mapping[str, Sequence[int]],
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
    n_layers: int = 6,
    n_bands: int = 8,
) -> dict[tuple[str, int], np.ndarray]:
    heatmap_data: dict[tuple[str, int], np.ndarray] = {}
    for dataset, idx_list in representative_idx.items():
        idx_level, _src = load_idx_level_exp1(
            analysis_root, dataset, seeds, exp1_subdir, aggregated_subdirs
        )
        for sample_id in idx_list:
            long_df = extract_idx_exp1_long(idx_level, int(sample_id))
            matrix = np.full((n_layers, n_bands), np.nan, dtype=float)
            if not long_df.empty:
                for row in long_df.itertuples(index=False):
                    layer = int(row.layer)
                    band = int(row.band)
                    if 0 <= layer < n_layers and 0 <= band < n_bands:
                        matrix[layer, band] = float(getattr(row, HEATMAP_VALUE_COL))
            if np.isnan(matrix).any():
                raise ValueError(
                    f"Incomplete representative matrix: dataset={dataset} idx={sample_id}"
                )
            heatmap_data[(dataset, int(sample_id))] = matrix
    return heatmap_data


def plot_exp1_violin_heatmap_composite(
    violin_data: Mapping[tuple[str, int, int], Sequence[float]],
    heatmap_data: Mapping[tuple[str, int], np.ndarray],
    representative_idx: Mapping[str, Sequence[int]],
    out_png: Path,
    *,
    out_svg: Optional[Path] = None,
    species_order: Sequence[str] = DEFAULT_DATASETS,
    n_layers: int = 6,
    n_bands: int = 8,
) -> None:
    layers = list(range(n_layers))
    bands = list(range(n_bands))
    band_labels = _band_labels(n_bands)
    species_labels = {ds: _species_label(ds) for ds in species_order}

    all_violin_values = np.concatenate(
        [
            np.asarray(violin_data[(species, layer, band)], dtype=float)
            for species in species_order
            for layer in layers
            for band in bands
            if (species, layer, band) in violin_data and len(violin_data[(species, layer, band)]) > 0
        ]
    )
    if all_violin_values.size == 0:
        raise ValueError("No violin values available for composite figure")
    violin_ymax = max(10.0, float(np.ceil(all_violin_values.max())))

    all_heatmap_values = np.concatenate([matrix.ravel() for matrix in heatmap_data.values()])
    heatmap_vmax = float(np.quantile(all_heatmap_values, 0.99))
    heatmap_norm = colors.Normalize(vmin=0.0, vmax=heatmap_vmax)
    heatmap_cmap = plt.get_cmap("GnBu")
    violin_blue = "#1f77b4"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": fs(9),
            "axes.titlesize": fs(15.5),
            "axes.labelsize": fs(15.5),
            "xtick.labelsize": fs(8),
            "ytick.labelsize": fs(8.5),
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(22.5, 13.7))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.075, 1.65, 0.18, 1.0],
        hspace=0.16,
    )

    title_a = fig.add_subplot(outer[0])
    title_a.axis("off")
    title_a.text(0.0, 0.55, "(A)", ha="left", va="center", fontsize=fs(17), fontweight="bold")
    title_a.text(
        0.5,
        0.55,
        r"Full-test-set $|\Delta\mathrm{P}|$ distributions across layers and sequence-frequency bands",
        ha="center",
        va="center",
        fontsize=fs(17),
        fontweight="semibold",
    )

    top_grid = outer[1].subgridspec(2, 6, wspace=0.10, hspace=0.13)
    top_axes = np.empty((2, 6), dtype=object)
    positions = np.arange(1, n_bands + 1)

    for row_index, species in enumerate(species_order):
        for column_index, layer in enumerate(layers):
            ax = fig.add_subplot(top_grid[row_index, column_index])
            top_axes[row_index, column_index] = ax
            values = [
                np.asarray(violin_data.get((species, layer, band), []), dtype=float)
                for band in bands
            ]
            if any(len(v) == 0 for v in values):
                raise ValueError(f"Missing violin data for {species} layer={layer}")
            violin = ax.violinplot(
                values,
                positions=positions,
                widths=0.82,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
                points=120,
            )
            for body in violin["bodies"]:
                body.set_facecolor(violin_blue)
                body.set_edgecolor("#303030")
                body.set_linewidth(0.5)
                body.set_alpha(0.85)
            for x_position, band_values in zip(positions, values):
                q1, median, q3 = np.quantile(band_values, [0.25, 0.50, 0.75])
                ax.vlines(x_position, q1, q3, color="#202020", linewidth=1.15, zorder=4)
                ax.scatter(
                    x_position,
                    median,
                    s=9,
                    facecolor="white",
                    edgecolor="#202020",
                    linewidth=0.5,
                    zorder=5,
                )
            ax.set_xlim(0.45, n_bands + 0.55)
            ax.set_ylim(0.0, violin_ymax)
            ax.set_xticks(positions)
            ax.set_xticklabels(band_labels, color="black", fontsize=fs(14), fontweight="bold")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(f"Layer {layer}", pad=6, fontsize=fs(15.5), fontweight="normal")
                ax.tick_params(axis="x", labelbottom=False)
            if column_index != 0:
                ax.tick_params(axis="y", labelleft=False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#555555")
            ax.spines["bottom"].set_color("#555555")

    title_b = fig.add_subplot(outer[2])
    title_b.axis("off")
    title_b.text(0.5, 0.88, "Sequence-frequency band", ha="center", va="center", fontsize=fs(15.5))
    title_b.text(0.0, 0.18, "(B)", ha="left", va="center", fontsize=fs(17), fontweight="bold")
    title_b.text(
        0.5,
        0.18,
        r"Representative peptide-level $|\Delta\mathrm{P}|$ maps",
        ha="center",
        va="center",
        fontsize=fs(17),
        fontweight="semibold",
    )

    bottom_grid = outer[3].subgridspec(
        2,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.045],
        wspace=0.16,
        hspace=0.27,
    )
    bottom_axes = np.empty((2, 5), dtype=object)
    image = None
    for row_index, species in enumerate(species_order):
        sample_ids = list(representative_idx.get(species, []))
        for column_index, sample_id in enumerate(sample_ids):
            ax = fig.add_subplot(bottom_grid[row_index, column_index])
            bottom_axes[row_index, column_index] = ax
            image = ax.imshow(
                heatmap_data[(species, int(sample_id))],
                cmap=heatmap_cmap,
                norm=heatmap_norm,
                interpolation="nearest",
                aspect="auto",
                origin="upper",
            )
            ax.set_title(f"Sample {sample_id}", pad=4, fontsize=fs(14.5), fontweight="normal")
            ax.set_xticks(np.arange(n_bands))
            ax.set_xticklabels(band_labels, color="black", fontsize=fs(14), fontweight="bold")
            ax.set_yticks(np.arange(n_layers))
            ax.set_yticklabels([str(i) for i in range(n_layers)])
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
            if column_index != 0:
                ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(length=2.3, width=0.55, pad=2)
            for spine in ax.spines.values():
                spine.set_visible(False)

    colorbar_axis = fig.add_subplot(bottom_grid[:, 5])
    colorbar = fig.colorbar(image, cax=colorbar_axis, extend="max")
    colorbar.set_label(r"$|\Delta\mathrm{P}|$", rotation=90, labelpad=9, fontsize=fs(15.5))
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(labelsize=fs(8), length=2.5, width=0.55)

    fig.subplots_adjust(left=0.055, right=0.965, top=0.975, bottom=0.065)
    fig.canvas.draw()

    top_row_centers = [
        0.5 * (top_axes[row, 0].get_position().y0 + top_axes[row, 0].get_position().y1)
        for row in range(2)
    ]
    top_center = 0.5 * (top_axes[1, 0].get_position().y0 + top_axes[0, 0].get_position().y1)
    top_left = top_axes[0, 0].get_position().x0
    fig.text(
        top_left - 0.036,
        top_row_centers[0],
        species_labels[species_order[0]],
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(17),
        color="black",
    )
    fig.text(
        top_left - 0.036,
        top_row_centers[1],
        species_labels[species_order[1]],
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(17),
        color="black",
    )
    fig.text(
        top_left - 0.017,
        top_center,
        r"$|\Delta\mathrm{P}|$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(16),
    )

    bottom_row_centers = [
        0.5 * (bottom_axes[row, 0].get_position().y0 + bottom_axes[row, 0].get_position().y1)
        for row in range(2)
    ]
    bottom_center = 0.5 * (
        bottom_axes[1, 0].get_position().y0 + bottom_axes[0, 0].get_position().y1
    )
    bottom_left = bottom_axes[0, 0].get_position().x0
    bottom_y0 = bottom_axes[1, 0].get_position().y0
    fig.text(
        bottom_left - 0.036,
        bottom_row_centers[0],
        species_labels[species_order[0]],
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(17),
        color="black",
    )
    fig.text(
        bottom_left - 0.036,
        bottom_row_centers[1],
        species_labels[species_order[1]],
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(17),
        color="black",
    )
    fig.text(
        bottom_left - 0.017,
        bottom_center,
        "ESM-2 layer",
        rotation=90,
        va="center",
        ha="center",
        fontsize=fs(15.5),
    )
    fig.text(
        0.5,
        bottom_y0 - 0.035,
        "Sequence-frequency band",
        ha="center",
        va="center",
        fontsize=fs(15.5),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
        print(f"[saved] {out_svg}")
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--exp1-subdir", default=DEFAULT_EXP1_SUBDIR)
    ap.add_argument(
        "--aggregated-subdirs",
        nargs="+",
        default=list(DEFAULT_AGGREGATED_SUBDIRS),
    )
    ap.add_argument("--representative-idx-json", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Default: <analysis-root>/aggregated/exp1_abs_mse_violin_heatmap_composite.png",
    )
    args = ap.parse_args()

    out_dir = args.analysis_root / "aggregated"
    out_png = args.out_png or (out_dir / COMPOSITE_PNG)
    out_svg = out_png.with_suffix(".svg") if out_png.suffix.lower() == ".png" else None
    if out_png.is_file() and not args.force:
        print(f"[SKIP] {out_png} exists (use --force to overwrite)")
        return

    rep_idx = _parse_representative_idx(args.representative_idx_json or None)
    violin_data = _load_violin_data(
        args.analysis_root,
        args.datasets,
        args.seeds,
        args.exp1_subdir,
        args.aggregated_subdirs,
    )
    heatmap_data = _load_heatmap_data(
        args.analysis_root,
        rep_idx,
        args.seeds,
        args.exp1_subdir,
        args.aggregated_subdirs,
    )
    plot_exp1_violin_heatmap_composite(
        violin_data,
        heatmap_data,
        rep_idx,
        out_png,
        out_svg=out_svg,
        species_order=tuple(ds for ds in args.datasets if ds in rep_idx),
    )


if __name__ == "__main__":
    main()
