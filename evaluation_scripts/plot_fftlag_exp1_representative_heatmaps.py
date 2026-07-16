#!/usr/bin/env python3
"""Plot Exp1 representative per-sample layer×band knockout heatmaps (Exp4 ID aligned)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir
from plot_fftlag_exp1_fulltest_violin import load_idx_level_exp1

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_EXP1_SUBDIR = "exp1_band_knockout_fulltest"
DEFAULT_AGGREGATED_SUBDIRS = ("aggregated_fulltest", "aggregated")
DEFAULT_SEEDS = tuple(range(10))

REPRESENTATIVE_IDX: Dict[str, list[int]] = {
    "e_coli": [35, 1442, 1438, 1004, 1043],
    "s_aureus": [641, 379, 1963, 876, 1026],
}

VALUE_COL = "mse_diff_mean"
HEATMAP_PNG = "exp1_band_knockout_heatmap_fulltest.png"
LONG_CSV = "exp1_band_knockout_mse_diff_long.csv"
WIDE_CSV = "exp1_band_knockout_mse_diff_wide.csv"
COMBINED_PNG = "exp1_representative_band_knockout_heatmaps_combined.png"


def _dataset_display(dataset: str) -> str:
    return "E. coli" if dataset == "e_coli" else "S. aureus"


def _band_column_label(band: int) -> str:
    return f"B{int(band)}"


def extract_idx_exp1_long(df: pd.DataFrame, idx: int, *, split: str = "test") -> pd.DataFrame:
    """Return long table for one peptide idx."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "split" in work.columns:
        work = work[work["split"] == split]
    work = work[work["idx"].astype(int) == int(idx)].copy()
    if work.empty:
        return pd.DataFrame()

    if VALUE_COL not in work.columns and "mse_diff" in work.columns:
        work[VALUE_COL] = work["mse_diff"]

    # Plot absolute ΔMSE; underlying CSVs may still store signed values.
    work[VALUE_COL] = work[VALUE_COL].astype(float).abs()

    work["layer"] = work["layer"].astype(int)
    work["band"] = work["band"].astype(int)
    keep = ["idx", "layer", "band", VALUE_COL]
    for col in ("mse_diff_std", "mse_base_mean", "mse_with_hook_mean", "n_seeds"):
        if col in work.columns:
            keep.append(col)
    return work[keep].sort_values(["layer", "band"]).reset_index(drop=True)


def long_to_pivot(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    pivot = long_df.pivot(index="layer", columns="band", values=VALUE_COL).sort_index()
    pivot.columns = [_band_column_label(int(c)) for c in pivot.columns]
    pivot.index = pivot.index.astype(int)
    return pivot


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    pivot = long_to_pivot(long_df)
    if pivot.empty:
        return pivot
    out = pivot.reset_index().rename(columns={"layer": "layer"})
    return out


def compute_shared_color_limits(
    pivots: Iterable[pd.DataFrame],
    *,
    percentile: float = 98.0,
) -> Tuple[float, float]:
    """Return [0, vmax] limits for absolute |ΔMSE| heatmaps."""
    values: list[float] = []
    for pivot in pivots:
        if pivot is not None and not pivot.empty:
            values.extend(pivot.to_numpy(dtype=float).ravel().tolist())
    if not values:
        return 0.0, 1.0
    arr = np.abs(np.asarray(values, dtype=float))
    vmax = float(np.percentile(arr, percentile))
    if vmax <= 0:
        vmax = float(np.max(arr)) or 1.0
    return 0.0, vmax


def plot_exp1_band_knockout_heatmap(
    pivot: pd.DataFrame,
    idx: int,
    dataset: str,
    out_png: Path,
    *,
    vmin: float,
    vmax: float,
    n_seeds: int,
    ax: Optional[plt.Axes] = None,
    show_cbar: bool = True,
    show_title: bool = True,
) -> Figure:
    if pivot.empty:
        raise ValueError(f"empty pivot for idx={idx}")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        annot=False,
        linewidths=0.5,
        cbar=show_cbar,
        cbar_kws={"label": r"$|\Delta\mathrm{MSE}|$"} if show_cbar else None,
        ax=ax,
    )
    ax.set_xlabel("Frequency band")
    ax.set_ylabel("Layer")
    if show_title:
        ds_label = _dataset_display(dataset)
        ax.set_title(
            f"Exp1 band knockout |ΔMSE| — {ds_label} idx={idx} "
            f"(mean across {n_seeds} seeds)"
        )
    if own_fig:
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_png}")
    return fig


def plot_exp1_representative_combined(
    panels: Mapping[str, Sequence[Tuple[int, pd.DataFrame]]],
    out_png: Path,
    *,
    vmin: float,
    vmax: float,
    n_seeds: int,
) -> None:
    species_order = [ds for ds in DEFAULT_DATASETS if ds in panels and panels[ds]]
    if not species_order:
        print(f"[SKIP] no panels for combined plot: {out_png}")
        return

    n_cols = max(len(panels[ds]) for ds in species_order)
    fig, axes = plt.subplots(
        len(species_order),
        n_cols,
        figsize=(3.2 * n_cols + 0.8, 3.0 * len(species_order)),
        squeeze=False,
    )

    last_mesh = None
    for row, dataset in enumerate(species_order):
        row_panels = list(panels[dataset])
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(row_panels):
                ax.set_visible(False)
                continue
            idx, pivot = row_panels[col]
            mesh = sns.heatmap(
                pivot,
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax,
                annot=False,
                linewidths=0.5,
                cbar=False,
                ax=ax,
            )
            last_mesh = mesh.collections[0] if mesh.collections else last_mesh
            ax.set_xlabel("Frequency bin" if row == len(species_order) - 1 else "")
            ax.set_ylabel("Layer" if col == 0 else "")
            ax.set_title(f"Sample {idx}", fontsize=10)
            if col == 0:
                ax.text(
                    -0.28,
                    0.5,
                    _dataset_display(dataset),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

    if last_mesh is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(last_mesh, cax=cbar_ax, label=r"$|\Delta\mathrm{MSE}|$ (absolute)")

    fig.suptitle(
        f"Representative Exp1 band knockout |ΔMSE| heatmaps ({n_seeds} seeds mean)",
        y=1.02,
        fontsize=12,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def _n_seeds_from_long(long_df: pd.DataFrame, default: int) -> int:
    if "n_seeds" in long_df.columns and not long_df["n_seeds"].isna().all():
        return int(long_df["n_seeds"].max())
    return default


def process_dataset(
    analysis_root: Path,
    dataset: str,
    idx_list: Sequence[int],
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    vmin: Optional[float],
    vmax: Optional[float],
    force: bool,
) -> Tuple[bool, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], list[int]]:
    idx_level, source = load_idx_level_exp1(
        analysis_root,
        dataset,
        seeds,
        exp1_subdir,
        aggregated_subdirs,
    )
    if idx_level.empty:
        print(f"[FAIL] {dataset}: no Exp1 idx-level data ({source})")
        return False, {}, {}, list(idx_list)

    long_by_idx: Dict[int, pd.DataFrame] = {}
    pivot_by_idx: Dict[int, pd.DataFrame] = {}
    missing: list[int] = []

    for idx in idx_list:
        long_df = extract_idx_exp1_long(idx_level, idx)
        if long_df.empty:
            missing.append(int(idx))
            print(f"[MISSING] {dataset} idx={idx}")
            continue
        long_by_idx[int(idx)] = long_df
        pivot_by_idx[int(idx)] = long_to_pivot(long_df)

    if not pivot_by_idx:
        return False, long_by_idx, pivot_by_idx, missing

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = compute_shared_color_limits(pivot_by_idx.values())
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    exp1_out = exp_agg_dir(analysis_root, dataset, "exp1")
    for idx, long_df in long_by_idx.items():
        idx_dir = exp1_out / "per_sample" / f"idx_{idx}"
        out_png = idx_dir / HEATMAP_PNG
        out_long = idx_dir / LONG_CSV
        out_wide = idx_dir / WIDE_CSV
        if (
            not force
            and out_png.is_file()
            and out_long.is_file()
            and out_wide.is_file()
        ):
            print(f"[SKIP] {dataset} idx={idx} outputs exist")
            continue

        n_seeds = _n_seeds_from_long(long_df, len(seeds))
        idx_dir.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(out_long, index=False)
        long_to_wide(long_df).to_csv(out_wide, index=False)
        plot_exp1_band_knockout_heatmap(
            pivot_by_idx[idx],
            idx,
            dataset,
            out_png,
            vmin=float(vmin),
            vmax=float(vmax),
            n_seeds=n_seeds,
        )
        print(f"[saved] {out_long}")
        print(f"[saved] {out_wide}")

    print(
        f"[OK] {dataset}: source={source} plotted={len(pivot_by_idx)}/{len(idx_list)} "
        f"missing={missing}"
    )
    return True, long_by_idx, pivot_by_idx, missing


def run_all(
    analysis_root: Path,
    representative_idx: Mapping[str, Sequence[int]],
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    force: bool,
) -> int:
    all_pivots: Dict[str, list[Tuple[int, pd.DataFrame]]] = {}
    all_long: list[pd.DataFrame] = []
    ok = True
    global_vmin: Optional[float] = None
    global_vmax: Optional[float] = None

    pre_pivots: list[pd.DataFrame] = []

    for dataset, idx_list in representative_idx.items():
        idx_level, _ = load_idx_level_exp1(
            analysis_root,
            dataset,
            seeds,
            exp1_subdir,
            aggregated_subdirs,
        )
        for idx in idx_list:
            long_df = extract_idx_exp1_long(idx_level, idx)
            if not long_df.empty:
                pre_pivots.append(long_to_pivot(long_df))

    if pre_pivots:
        global_vmin, global_vmax = compute_shared_color_limits(pre_pivots)

    for dataset, idx_list in representative_idx.items():
        success, long_by_idx, pivot_by_idx, missing = process_dataset(
            analysis_root,
            dataset,
            idx_list,
            seeds,
            exp1_subdir,
            aggregated_subdirs,
            vmin=global_vmin,
            vmax=global_vmax,
            force=force,
        )
        if missing:
            print(f"[WARN] {dataset} missing idx: {missing}")
        if not success or not pivot_by_idx:
            ok = False
        if pivot_by_idx:
            all_pivots[dataset] = [(idx, pivot_by_idx[idx]) for idx in idx_list if idx in pivot_by_idx]
            for idx in idx_list:
                if idx in long_by_idx:
                    part = long_by_idx[idx].copy()
                    part.insert(0, "species", dataset)
                    all_long.append(part)

    if all_pivots:
        combined_png = analysis_root / "aggregated" / COMBINED_PNG
        plot_exp1_representative_combined(
            all_pivots,
            combined_png,
            vmin=float(global_vmin if global_vmin is not None else 0.0),
            vmax=float(global_vmax if global_vmax is not None else 1.0),
            n_seeds=len(seeds),
        )
        combined_long = analysis_root / "aggregated" / "exp1_representative_band_knockout_long.csv"
        if all_long:
            pd.concat(all_long, ignore_index=True).to_csv(combined_long, index=False)
            print(f"[saved] {combined_long}")

        manifest = {
            "representative_idx": {k: list(v) for k, v in representative_idx.items()},
            "combined_png": str(combined_png.resolve()),
            "vmin": global_vmin,
            "vmax": global_vmax,
            "n_seeds": len(seeds),
        }
        manifest_path = analysis_root / "aggregated" / "exp1_representative_band_knockout_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[saved] {manifest_path}")

    return 0 if ok and all_pivots else 1


def _parse_representative_idx(raw: Optional[str]) -> Dict[str, list[int]]:
    if not raw:
        return {k: list(v) for k, v in REPRESENTATIVE_IDX.items()}
    payload = json.loads(raw)
    return {str(k): [int(x) for x in v] for k, v in payload.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--exp1-subdir", default=DEFAULT_EXP1_SUBDIR)
    ap.add_argument(
        "--aggregated-subdirs",
        nargs="*",
        default=list(DEFAULT_AGGREGATED_SUBDIRS),
    )
    ap.add_argument(
        "--representative-idx-json",
        default="",
        help='JSON map species->idx list, default built-in Exp4-aligned IDs',
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    rep_idx = _parse_representative_idx(args.representative_idx_json or None)
    return run_all(
        root,
        rep_idx,
        args.seeds,
        args.exp1_subdir,
        args.aggregated_subdirs,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
