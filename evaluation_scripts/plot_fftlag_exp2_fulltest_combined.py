#!/usr/bin/env python3
# maintained by kewei li
"""Plot Exp2 full-test PSD gate band summary across all peptides (mean ± std)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir, exp_figures_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_EXP2_SUBDIR = "exp2_psd_gate_fulltest"
DEFAULT_SEEDS = tuple(range(10))

VALUE_COLS = ("effective_weight", "energy_before", "energy_after")

TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 12


def _savefig_png_svg(fig: plt.Figure, out_png: Path) -> Path:
    """Save figure as PNG (dpi=200) and sibling SVG; return SVG path."""
    out_png = Path(out_png)
    out_svg = out_png.with_suffix(".svg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_svg}")
    return out_svg


def _dataset_display(dataset: str) -> str:
    return r"$\it{E.\ coli}$" if dataset == "e_coli" else r"$\it{S.\ aureus}$"


def _train_seed_from_dir(seed_dir: Path) -> int:
    name = seed_dir.name
    if not name.startswith("seed_"):
        raise ValueError(f"Expected seed_* dir, got {seed_dir}")
    return int(name.replace("seed_", ""))


def _resolve_idx_level_csv(
    analysis_root: Path,
    dataset: str,
    *,
    aggregated_subdirs: Sequence[str],
) -> Optional[Path]:
    for subdir in aggregated_subdirs:
        path = analysis_root / subdir / dataset / "exp2" / "per_sample_gate_by_band_aggregated.csv"
        if path.is_file():
            return path
    return None


def load_exp2_fulltest_frames(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp2_subdir: str,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = analysis_root / f"seed_{seed}" / dataset / exp2_subdir
        csv_path = seed_dir / "per_sample_gate_by_band.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        frames.append(df)
    return frames


def _aggregate_seed_to_idx(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    group_cols = ["idx", "band", "band_start", "band_end"]
    agg_map: dict = {}
    for col in VALUE_COLS:
        agg_map[f"{col}_mean"] = (col, "mean")
        if col == "effective_weight":
            agg_map[f"{col}_std"] = (col, "std")
    agg_map["n_seeds"] = ("seed", "nunique")
    return all_df.groupby(group_cols, as_index=False).agg(**agg_map)


def _normalize_idx_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either raw per-idx aggregated CSV or seed-aggregated frames."""
    out = df.copy()
    rename_map = {
        "effective_weight": "effective_weight_mean",
        "energy_before": "energy_before_mean",
        "energy_after": "energy_after_mean",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]
    required = ["idx", "band", "band_start", "band_end", "effective_weight_mean", "energy_before_mean", "energy_after_mean"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return out


def aggregate_exp2_all_samples(idx_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-idx seed means to band-level population mean ± std."""
    work = _normalize_idx_level_df(idx_level)
    group_cols = ["band", "band_start", "band_end"]
    agg_map: dict = {
        "effective_weight_mean": ("effective_weight_mean", "mean"),
        "effective_weight_std": ("effective_weight_mean", "std"),
        "energy_before_mean": ("energy_before_mean", "mean"),
        "energy_before_std": ("energy_before_mean", "std"),
        "energy_after_mean": ("energy_after_mean", "mean"),
        "energy_after_std": ("energy_after_mean", "std"),
        "n_samples": ("idx", "nunique"),
    }
    if "n_seeds" in work.columns:
        agg_map["n_seeds"] = ("n_seeds", "max")
    return work.groupby(group_cols, as_index=False).agg(**agg_map).sort_values("band")


def load_idx_level_exp2(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp2_subdir: str,
    *,
    aggregated_subdirs: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    csv_path = _resolve_idx_level_csv(analysis_root, dataset, aggregated_subdirs=aggregated_subdirs)
    if csv_path is not None:
        return pd.read_csv(csv_path), f"csv:{csv_path}"

    frames = load_exp2_fulltest_frames(analysis_root, dataset, seeds, exp2_subdir)
    if not frames:
        return pd.DataFrame(), "missing"
    return _aggregate_seed_to_idx(frames), f"seeds:{len(frames)}"


def _band_stats(band_df: pd.DataFrame, seeds: Sequence[int]) -> tuple[int, int, int]:
    n_samples = int(band_df["n_samples"].max()) if not band_df.empty else 0
    n_bands = int(band_df["band"].nunique()) if not band_df.empty else 0
    if "n_seeds" in band_df.columns and not band_df["n_seeds"].isna().all():
        n_seeds = int(band_df["n_seeds"].max())
    else:
        n_seeds = len(seeds)
    return n_seeds, n_samples, n_bands


def _plot_exp2_gate_band_panel(
    ax: plt.Axes,
    band_df: pd.DataFrame,
    dataset: str,
    *,
    n_seeds: int,
    n_samples: int,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    sub = band_df.sort_values("band")
    ds_label = _dataset_display(dataset)

    x = np.arange(len(sub))
    width = 0.35
    ax.bar(
        x - width / 2,
        sub["energy_before_mean"].to_numpy(),
        width,
        yerr=sub["energy_before_std"].fillna(0.0).to_numpy(),
        capsize=3,
        label="Before gate",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        sub["energy_after_mean"].to_numpy(),
        width,
        yerr=sub["energy_after_std"].fillna(0.0).to_numpy(),
        capsize=3,
        label="After gate",
        color="darkorange",
        alpha=0.8,
    )
    band_labels = [
        rf"$\mathcal{{B}}_{{{int(b)}}}$" + f"\n[{int(s)},{int(e)})"
        for b, s, e in zip(sub["band"], sub["band_start"], sub["band_end"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=TICK_LABEL_FONTSIZE, fontweight="bold", color="black")
    ax.set_xlabel("Frequency band", fontsize=AXIS_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Mean energy", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(ds_label, fontsize=TITLE_FONTSIZE, color="black")
    if show_legend:
        ax.legend(fontsize=TICK_LABEL_FONTSIZE)


def plot_exp2_gate_band_summary_all_samples(
    band_df: pd.DataFrame,
    dataset: str,
    out_png: Path,
    *,
    n_seeds: int,
    n_samples: int,
) -> None:
    sub = band_df.sort_values("band")
    if sub.empty:
        print(f"[SKIP] empty band df for {out_png}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ds_label = _dataset_display(dataset)
    sample_label = f"{ds_label} (mean across {n_samples} peptides, {n_seeds} seeds)"
    _plot_exp2_gate_band_panel(
        ax,
        band_df,
        dataset,
        n_seeds=n_seeds,
        n_samples=n_samples,
        show_ylabel=True,
        show_legend=True,
    )
    ax.set_title(f"Exp2 PSD gate — Energy before vs after gate\n{sample_label}")
    fig.tight_layout()
    _savefig_png_svg(fig, out_png)
    plt.close(fig)


def plot_exp2_gate_band_summary_combined(
    band_dfs: Dict[str, pd.DataFrame],
    meta: Dict[str, Tuple[int, int]],
    out_png: Path,
) -> None:
    datasets = [ds for ds in DEFAULT_DATASETS if ds in band_dfs and not band_dfs[ds].empty]
    if not datasets:
        print(f"[SKIP] no data for combined plot: {out_png}")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        n_seeds, n_samples = meta[dataset]
        _plot_exp2_gate_band_panel(
            ax,
            band_dfs[dataset],
            dataset,
            n_seeds=n_seeds,
            n_samples=n_samples,
            show_ylabel=(dataset == datasets[0]),
            show_legend=(dataset == datasets[-1]),
        )

    fig.tight_layout()
    _savefig_png_svg(fig, out_png)
    plt.close(fig)


def process_dataset(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp2_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    force: bool,
) -> tuple[bool, Optional[pd.DataFrame], int, int]:
    data_dir = exp_agg_dir(analysis_root, dataset, "exp2")
    fig_dir = exp_figures_dir(analysis_root, "exp2", dataset)
    summary_csv = data_dir / "gate_by_band_all_samples_summary.csv"
    out_png = fig_dir / "gate_band_summary_all_samples.png"
    out_svg = out_png.with_suffix(".svg")

    if summary_csv.is_file() and not force:
        band_df = pd.read_csv(summary_csv)
        n_seeds, n_samples, _ = _band_stats(band_df, seeds)
        if out_png.is_file() and out_svg.is_file():
            print(f"[SKIP] {dataset}: outputs exist (use --force to overwrite)")
            return True, band_df, n_seeds, n_samples
        print(f"[RUN] {dataset}: replot from existing summary CSV (PNG/SVG)")
        plot_exp2_gate_band_summary_all_samples(
            band_df,
            dataset,
            out_png,
            n_seeds=n_seeds,
            n_samples=n_samples,
        )
        return True, band_df, n_seeds, n_samples

    idx_level, source = load_idx_level_exp2(
        analysis_root,
        dataset,
        seeds,
        exp2_subdir,
        aggregated_subdirs=aggregated_subdirs,
    )
    if idx_level.empty:
        print(f"[FAIL] {dataset}: no Exp2 idx-level data ({exp2_subdir})")
        return False, None, 0, 0

    band_df = aggregate_exp2_all_samples(idx_level)
    n_seeds, n_samples, n_bands = _band_stats(band_df, seeds)

    print(
        f"[OK] {dataset}: source={source} n_seeds={n_seeds} "
        f"n_samples={n_samples} n_bands={n_bands}"
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    band_df.to_csv(summary_csv, index=False)
    print(f"[saved] {summary_csv}")

    plot_exp2_gate_band_summary_all_samples(
        band_df,
        dataset,
        out_png,
        n_seeds=n_seeds,
        n_samples=n_samples,
    )
    return True, band_df, n_seeds, n_samples


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    seeds: Sequence[int],
    exp2_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    force: bool,
) -> int:
    ok = True
    band_dfs: Dict[str, pd.DataFrame] = {}
    meta: Dict[str, Tuple[int, int]] = {}

    for dataset in datasets:
        success, band_df, n_seeds, n_samples = process_dataset(
            analysis_root,
            dataset,
            seeds,
            exp2_subdir,
            aggregated_subdirs,
            force=force,
        )
        if not success:
            ok = False
            continue
        if band_df is not None and not band_df.empty:
            band_dfs[dataset] = band_df
            meta[dataset] = (n_seeds, n_samples)

    if len(band_dfs) >= 1:
        combined_png = exp_figures_dir(analysis_root, "exp2", "combined") / "exp2_gate_band_summary_all_samples_combined.png"
        combined_svg = combined_png.with_suffix(".svg")
        if combined_png.is_file() and combined_svg.is_file() and not force and len(band_dfs) < 2:
            pass
        elif (
            not combined_png.is_file()
            or not combined_svg.is_file()
            or force
            or len(band_dfs) >= 2
        ):
            plot_exp2_gate_band_summary_combined(band_dfs, meta, combined_png)

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--exp2-subdir", default=DEFAULT_EXP2_SUBDIR)
    ap.add_argument(
        "--aggregated-subdirs",
        nargs="*",
        default=["aggregated_fulltest", "aggregated"],
        help="Search order for per_sample_gate_by_band_aggregated.csv",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    return run_all(
        root,
        args.datasets,
        args.seeds,
        args.exp2_subdir,
        args.aggregated_subdirs,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
