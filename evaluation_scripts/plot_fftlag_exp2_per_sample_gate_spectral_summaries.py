#!/usr/bin/env python3
# maintained by kewei li
"""Plot Exp2 per-sample gate energy summaries in a 6×5 composite grid."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir, exp_figures_dir
from fftlag_per_sample_panel_order import PANEL_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_NROWS = 6
DEFAULT_NCOLS = 5
OUT_BASENAME = "exp2_per_sample_gate_spectral_summaries"


def _savefig_png_svg(fig: plt.Figure, out_png: Path) -> None:
    out_png = Path(out_png)
    out_svg = out_png.with_suffix(".svg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_svg}")


def _species_label(dataset: str) -> str:
    return "E. coli" if dataset == "e_coli" else "S. aureus"


def _load_idx_level_csv(analysis_root: Path, dataset: str) -> pd.DataFrame:
    csv_path = exp_agg_dir(analysis_root, dataset, "exp2") / "per_sample_gate_by_band_aggregated.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing aggregated CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _global_energy_ylim(df: pd.DataFrame, idx_order: Sequence[int]) -> float:
    sub = df[df["idx"].astype(int).isin([int(i) for i in idx_order])]
    if sub.empty:
        return 1.0
    ymax = float(
        max(
            sub["energy_before_mean"].max(),
            sub["energy_after_mean"].max(),
        )
    )
    return ymax * 1.05 if ymax > 0 else 1.0


def _plot_energy_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    show_xticklabels: bool,
    show_yticklabels: bool,
    show_legend: bool,
    ylim: float,
) -> None:
    sub = sub.sort_values("band")
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(
        x - width / 2,
        sub["energy_before_mean"].to_numpy(),
        width,
        label="Before gate",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        sub["energy_after_mean"].to_numpy(),
        width,
        label="After gate",
        color="darkorange",
        alpha=0.8,
    )
    band_labels = [
        f"B{int(b)}\n[{int(s)},{int(e)})"
        for b, s, e in zip(sub["band"], sub["band_start"], sub["band_end"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels if show_xticklabels else [], fontsize=5)
    ax.set_xlabel("Frequency band" if show_xlabel else "")
    if show_ylabel:
        ax.set_ylabel("Mean energy", fontsize=9)
    ax.set_ylim(0.0, ylim)
    ax.tick_params(axis="y", labelsize=6, labelleft=show_yticklabels)
    if show_legend:
        ax.legend(fontsize=7, loc="upper right")


def plot_per_sample_gate_spectral_summaries(
    df: pd.DataFrame,
    dataset: str,
    idx_order: Sequence[int],
    out_png: Path,
    *,
    nrows: int = DEFAULT_NROWS,
    ncols: int = DEFAULT_NCOLS,
) -> None:
    species = _species_label(dataset)
    n_panels = nrows * ncols
    if len(idx_order) != n_panels:
        raise ValueError(f"{dataset}: expected {n_panels} idx values, got {len(idx_order)}")

    ylim = _global_energy_ylim(df, idx_order)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()
    missing: list[int] = []

    for ax, idx in zip(axes_flat, idx_order):
        sub = df[df["idx"].astype(int) == int(idx)]
        if sub.empty:
            missing.append(int(idx))
            ax.set_visible(False)
            continue
        spec = ax.get_subplotspec()
        row, col = spec.rowspan.start, spec.colspan.start
        show_ylabel = col == 0
        show_xlabel = row == nrows - 1
        show_xticklabels = row == nrows - 1
        show_yticklabels = col == 0
        show_legend = row == 0 and col == ncols - 1
        _plot_energy_panel(
            ax,
            sub,
            show_ylabel=show_ylabel,
            show_xlabel=show_xlabel,
            show_xticklabels=show_xticklabels,
            show_yticklabels=show_yticklabels,
            show_legend=show_legend,
            ylim=ylim,
        )
        ax.text(
            0.0,
            1.06,
            f"sample {idx}",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    if missing:
        plt.close(fig)
        raise ValueError(f"{dataset}: missing idx in CSV: {missing}")

    fig.suptitle(
        f"Per-sample gate spectral summaries ({species}): energy before vs after gate",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig_png_svg(fig, out_png)
    plt.close(fig)


def process_dataset(
    analysis_root: Path,
    dataset: str,
    *,
    nrows: int,
    ncols: int,
    force: bool,
) -> bool:
    out_dir = exp_figures_dir(analysis_root, "exp2", dataset)
    out_png = out_dir / f"{OUT_BASENAME}.png"
    out_svg = out_png.with_suffix(".svg")
    if out_png.is_file() and out_svg.is_file() and not force:
        print(f"[SKIP] {dataset}: outputs exist (use --force to overwrite)")
        return True

    idx_order = PANEL_ORDER.get(dataset)
    if idx_order is None:
        raise ValueError(f"No panel order defined for dataset={dataset}")

    df = _load_idx_level_csv(analysis_root, dataset)
    plot_per_sample_gate_spectral_summaries(
        df,
        dataset,
        idx_order,
        out_png,
        nrows=nrows,
        ncols=ncols,
    )
    print(f"[OK] {dataset}: {len(idx_order)} panels -> {out_png}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--nrows", type=int, default=DEFAULT_NROWS)
    ap.add_argument("--ncols", type=int, default=DEFAULT_NCOLS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    ok = True
    for dataset in args.datasets:
        try:
            if not process_dataset(
                root,
                dataset,
                nrows=args.nrows,
                ncols=args.ncols,
                force=args.force,
            ):
                ok = False
        except (FileNotFoundError, ValueError) as exc:
            print(f"[FAIL] {dataset}: {exc}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
