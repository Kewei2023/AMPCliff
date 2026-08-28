#!/usr/bin/env python3
# maintained by kewei li
"""Plot Exp1 per-sample layer×band knockout heatmaps in a 6×5 composite grid."""
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
import seaborn as sns
from matplotlib.colors import Normalize

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir, exp_figures_dir
from fftlag_per_sample_panel_order import PANEL_ORDER
from plot_fftlag_exp1_representative_heatmaps import extract_idx_exp1_long

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_NROWS = 6
DEFAULT_NCOLS = 5
OUT_BASENAME = "exp1_per_sample_band_knockout_heatmaps"


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
    csv_path = exp_agg_dir(analysis_root, dataset, "exp1") / "per_sample_band_sensitivity_aggregated.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing aggregated CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _pivot_for_heatmap(idx_level: pd.DataFrame, idx: int) -> pd.DataFrame:
    long_df = extract_idx_exp1_long(idx_level, idx, split="test")
    if long_df.empty:
        return pd.DataFrame()
    pivot = long_df.pivot(index="layer", columns="band", values="mse_diff_mean").sort_index()
    pivot.index = pivot.index.astype(int)
    pivot.columns = pivot.columns.astype(int)
    return pivot


def plot_per_sample_knockout_heatmaps(
    idx_level: pd.DataFrame,
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    missing: list[int] = []

    for ax, idx in zip(axes_flat, idx_order):
        pivot = _pivot_for_heatmap(idx_level, int(idx))
        if pivot.empty:
            missing.append(int(idx))
            ax.set_visible(False)
            continue

        vmax = float(np.max(pivot.to_numpy(dtype=float)))
        if vmax <= 0:
            vmax = 1.0

        spec = ax.get_subplotspec()
        row, col = spec.rowspan.start, spec.colspan.start
        show_ylabel = col == 0
        show_xlabel = row == nrows - 1

        sns.heatmap(
            pivot,
            cmap="viridis",
            norm=Normalize(vmin=0.0, vmax=vmax),
            annot=False,
            linewidths=0.5,
            cbar=True,
            cbar_kws={"fraction": 0.046, "pad": 0.04},
            ax=ax,
        )
        ax.set_title(f"sample {idx}", fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel("Frequency band (0 = lowest)" if show_xlabel else "")
        ax.set_ylabel("Transformer layer" if show_ylabel else "")
        ax.tick_params(axis="both", labelsize=6)
        if not show_ylabel:
            ax.set_yticklabels([])
        if not show_xlabel:
            ax.set_xticklabels([])

    if missing:
        plt.close(fig)
        raise ValueError(f"{dataset}: missing idx in CSV: {missing}")

    fig.suptitle(
        f"Per-sample sequence-frequency band-knockout heatmaps ({species})",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
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
    out_dir = exp_figures_dir(analysis_root, "exp1", dataset)
    out_png = out_dir / f"{OUT_BASENAME}.png"
    out_svg = out_png.with_suffix(".svg")
    if out_png.is_file() and out_svg.is_file() and not force:
        print(f"[SKIP] {dataset}: outputs exist (use --force to overwrite)")
        return True

    idx_order = PANEL_ORDER.get(dataset)
    if idx_order is None:
        raise ValueError(f"No panel order defined for dataset={dataset}")

    idx_level = _load_idx_level_csv(analysis_root, dataset)
    plot_per_sample_knockout_heatmaps(
        idx_level,
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
