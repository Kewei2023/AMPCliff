#!/usr/bin/env python3
# maintained by kewei li
"""Exp4 combined figure: freq-bin attention violins + representative query×freq heatmaps."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import FormatStrFormatter

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir, exp_figures_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ("e_coli", "s_aureus")
REPRESENTATIVE_IDX: Dict[str, list[int]] = {
    "e_coli": [35, 1442, 1438, 1004, 1043],
    "s_aureus": [641, 379, 1963, 876, 1026],
}

COMPOSITE_PNG = "exp4_frequency_attention_combined_figure.png"
LONG_CSV = "latent_query_freq_distribution_long.csv"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "bluegreen_target957_v4",
    [
        (0.00, "#F7FCF0"),
        (0.10, "#DCEFD2"),
        (0.22, "#A8DDB5"),
        (0.36, "#67C5B6"),
        (0.50, "#2FA6A3"),
        (0.62, "#3A93C8"),
        (0.72, "#2C7FB8"),
        (0.80, "#2B68B1"),
        (0.88, "#3156A6"),
        (0.94, "#273C85"),
        (1.00, "#132B73"),
    ],
    N=256,
)
VIOLIN_FILL = "#86CFC5"
VIOLIN_EDGE = "#247C8A"
VIOLIN_MEDIAN = "#123E5A"


def _species_label(dataset: str) -> str:
    return r"$\it{E.\ coli}$" if dataset == "e_coli" else r"$\it{S.\ aureus}$"


def _parse_representative_idx(raw: Optional[str]) -> Dict[str, list[int]]:
    if not raw:
        return {k: list(v) for k, v in REPRESENTATIVE_IDX.items()}
    import json

    parsed = json.loads(raw)
    return {str(k): [int(i) for i in v] for k, v in parsed.items()}


def _resolve_long_csv(analysis_root: Path, dataset: str) -> Path:
    path = exp_agg_dir(analysis_root, dataset, "exp4") / LONG_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Missing Exp4 long CSV: {path}")
    return path


def _resolve_attn_csv(analysis_root: Path, dataset: str, idx: int) -> Path:
    path = (
        exp_agg_dir(analysis_root, dataset, "exp4")
        / "per_sample"
        / f"idx_{idx}"
        / "plots"
        / "mean_style"
        / "attn_score_raw.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Exp4 attn matrix CSV: {path}")
    return path


def _load_attn_matrix(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    freq_cols = [c for c in df.columns if str(c).startswith("freq_")]
    if not freq_cols:
        raise ValueError(f"No freq_* columns in {csv_path}")
    return df[freq_cols].to_numpy(dtype=float)


def plot_exp4_frequency_attention_combined(
    long_dfs: Mapping[str, pd.DataFrame],
    heatmaps: Mapping[str, np.ndarray],
    representative_idx: Mapping[str, Sequence[int]],
    out_png: Path,
    *,
    out_pdf: Optional[Path] = None,
    out_svg: Optional[Path] = None,
    species_order: Sequence[str] = DEFAULT_DATASETS,
) -> None:
    text_color = "#222222"
    axis_color = "#333333"
    grid_color = "#D9D9D9"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 15,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.edgecolor": axis_color,
            "axes.labelcolor": text_color,
            "xtick.color": text_color,
            "ytick.color": text_color,
            "text.color": text_color,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    all_heat_vals = np.concatenate([mat.ravel() for mat in heatmaps.values()])
    heat_vmin, heat_vmax = np.quantile(all_heat_vals, [0.02, 0.985])
    heat_norm = PowerNorm(
        gamma=1.95, vmin=float(heat_vmin), vmax=float(heat_vmax), clip=True
    )

    fig = plt.figure(figsize=(16.5, 8.4), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.95, 1.25],
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.08,
        hspace=0.42,
    )

    top = outer[0].subgridspec(1, 2, wspace=0.16)
    violin_axes = [
        fig.add_subplot(top[0, 0]),
        fig.add_subplot(top[0, 1]),
    ]

    for ax, dataset in zip(violin_axes, species_order):
        long_df = long_dfs[dataset]
        positions = sorted(long_df["freq_bin"].unique().tolist())
        data_list = [
            long_df.loc[long_df["freq_bin"] == pos, "attn_score"].to_numpy(dtype=float)
            for pos in positions
        ]
        vp = ax.violinplot(
            data_list,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(VIOLIN_FILL)
            body.set_edgecolor(VIOLIN_EDGE)
            body.set_linewidth(0.8)
            body.set_alpha(0.95)
        vp["cmedians"].set_color(VIOLIN_MEDIAN)
        vp["cmedians"].set_linewidth(1.0)
        ax.set_title(
            _species_label(dataset),
            pad=10,
            fontsize=15,
            fontweight="semibold",
            color="black",
        )
        ax.set_xlabel("Frequency bin", fontsize=13.5)
        ax.set_ylabel("Cross-attention score" if ax is violin_axes[0] else "", fontsize=13.5)
        ax.set_xlim(-0.5, max(positions) + 0.5)
        ax.set_xticks(positions)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", color=grid_color, linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    ymin = min(ax.get_ylim()[0] for ax in violin_axes)
    ymax = max(ax.get_ylim()[1] for ax in violin_axes)
    for ax in violin_axes:
        ax.set_ylim(ymin, ymax)

    bottom = outer[1].subgridspec(
        2, 6, width_ratios=[1, 1, 1, 1, 1, 0.042], wspace=0.18, hspace=0.28
    )
    heat_axes = []
    im = None
    for row_idx, dataset in enumerate(species_order):
        for col_idx, sample_id in enumerate(representative_idx[dataset]):
            ax = fig.add_subplot(bottom[row_idx, col_idx])
            heat_axes.append(ax)
            key = f"{dataset}_{sample_id}"
            mat = heatmaps[key]
            im = ax.imshow(
                mat,
                cmap=HEATMAP_CMAP,
                norm=heat_norm,
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_title(f"Sample {sample_id}", pad=3, fontsize=13.0, fontweight="normal")
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(np.arange(mat.shape[1]), fontsize=5.8)
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(np.arange(mat.shape[0]), fontsize=5.8)
            ax.set_xlabel("Frequency bin" if row_idx == 1 else "", labelpad=1.5, fontsize=11.8)
            ax.set_ylabel("Latent query" if col_idx == 0 else "", labelpad=4, fontsize=11.8)
            ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.16, alpha=0.18)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(length=0, pad=0.8)
            for spine in ax.spines.values():
                spine.set_visible(False)

    cax = fig.add_subplot(bottom[:, 5])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.ax.set_title("Score", fontsize=12.8, pad=4)
    cb.ax.tick_params(labelsize=6.8, length=2.2)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    cb.outline.set_linewidth(0.6)

    fig.canvas.draw()
    fig.text(
        0.055,
        violin_axes[0].get_position().y1 + 0.036,
        "(A)",
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    first_bottom_ax = heat_axes[0]
    fig.text(
        first_bottom_ax.get_position().x0 - 0.013,
        first_bottom_ax.get_position().y1 + 0.028,
        "(B)",
        fontsize=15.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.5,
        violin_axes[0].get_position().y1 + 0.036,
        "Cross-attention score distributions across frequency bins",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        first_bottom_ax.get_position().y1 + 0.028,
        "Representative latent-query × frequency-bin attention maps",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
    )

    n_cols = len(list(representative_idx[species_order[0]]))
    row1_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in heat_axes[:n_cols]]
    )
    row2_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in heat_axes[n_cols:]]
    )
    left_x = min(ax.get_position().x0 for ax in heat_axes[:n_cols]) - 0.030
    fig.text(
        left_x,
        row1_y,
        _species_label(species_order[0]),
        rotation=90,
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="semibold",
        color="black",
    )
    fig.text(
        left_x,
        row2_y,
        _species_label(species_order[1]),
        rotation=90,
        ha="center",
        va="center",
        fontsize=14.2,
        fontweight="semibold",
        color="black",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out_png}")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
        print(f"[saved] {out_pdf}")
    if out_svg is not None:
        fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
        print(f"[saved] {out_svg}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--representative-idx-json", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args()

    out_dir = exp_figures_dir(args.analysis_root, "exp4", "combined")
    out_png = args.out_png or (out_dir / COMPOSITE_PNG)
    out_pdf = out_png.with_suffix(".pdf")
    out_svg = out_png.with_suffix(".svg")
    if out_png.is_file() and not args.force:
        print(f"[SKIP] {out_png} exists (use --force to overwrite)")
        return

    rep_idx = _parse_representative_idx(args.representative_idx_json or None)
    species_order = tuple(ds for ds in args.datasets if ds in rep_idx)
    if len(species_order) < 2:
        raise ValueError("Need at least e_coli and s_aureus representative panels")

    long_dfs = {
        dataset: pd.read_csv(_resolve_long_csv(args.analysis_root, dataset))
        for dataset in species_order
    }
    heatmaps: Dict[str, np.ndarray] = {}
    for dataset in species_order:
        for sample_id in rep_idx[dataset]:
            csv_path = _resolve_attn_csv(args.analysis_root, dataset, int(sample_id))
            heatmaps[f"{dataset}_{sample_id}"] = _load_attn_matrix(csv_path)

    plot_exp4_frequency_attention_combined(
        long_dfs,
        heatmaps,
        rep_idx,
        out_png,
        out_pdf=out_pdf,
        out_svg=out_svg,
        species_order=species_order,
    )


if __name__ == "__main__":
    main()
