#!/usr/bin/env python3
"""Plot Exp1 full-test band knockout |ΔMSE| distributions as per-layer violin plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_EXP1_SUBDIR = "exp1_band_knockout_fulltest"
DEFAULT_AGGREGATED_SUBDIRS = ("aggregated_fulltest", "aggregated")
DEFAULT_SEEDS = tuple(range(10))

LONG_CSV = "exp1_band_knockout_violin_long.csv"
SUMMARY_CSV = "exp1_band_knockout_violin_summary.csv"
VALUE_COL = "mse_diff"
Y_LABEL = r"$|\Delta\mathrm{MSE}|$ (absolute knockout − baseline)"


def _dataset_display(dataset: str) -> str:
    return "E. coli" if dataset == "e_coli" else "S. aureus"


def _band_tick_label(band: int) -> str:
    return f"B{int(band)}"


def _train_seed_from_dir(seed_dir: Path) -> int:
    name = seed_dir.name
    if not name.startswith("seed_"):
        raise ValueError(f"Expected seed_* dir, got {seed_dir}")
    return int(name.replace("seed_", ""))


def _resolve_idx_level_csv(
    analysis_root: Path,
    dataset: str,
    aggregated_subdirs: Sequence[str],
) -> Optional[Path]:
    for subdir in aggregated_subdirs:
        path = (
            analysis_root
            / subdir
            / dataset
            / "exp1"
            / "per_sample_band_sensitivity_aggregated.csv"
        )
        if path.is_file():
            return path
    return None


def _load_seed_frames(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp1_subdir: str,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        csv_path = analysis_root / f"seed_{seed}" / dataset / exp1_subdir / "per_sample_band_sensitivity.csv"
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
    return (
        all_df.groupby(["idx", "layer", "band", "split"], as_index=False)
        .agg(
            mse_diff_mean=("mse_diff", "mean"),
            mse_diff_std=("mse_diff", "std"),
            n_seeds=("seed", "nunique"),
        )
    )


def load_idx_level_exp1(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    csv_path = _resolve_idx_level_csv(analysis_root, dataset, aggregated_subdirs)
    if csv_path is not None:
        return pd.read_csv(csv_path), f"csv:{csv_path}"

    frames = _load_seed_frames(analysis_root, dataset, seeds, exp1_subdir)
    if not frames:
        return pd.DataFrame(), "missing"
    return _aggregate_seed_to_idx(frames), f"seeds:{len(frames)}"


def build_violin_long_df(idx_level: pd.DataFrame) -> pd.DataFrame:
    """Normalize idx-level Exp1 table to long format for violin plots."""
    if idx_level.empty:
        return pd.DataFrame()

    work = idx_level.copy()
    if "split" in work.columns:
        work = work[work["split"] == "test"].copy()
    if work.empty:
        return pd.DataFrame()

    if VALUE_COL not in work.columns:
        if "mse_diff_mean" in work.columns:
            work[VALUE_COL] = work["mse_diff_mean"]
        else:
            raise ValueError(f"Missing {VALUE_COL} or mse_diff_mean in Exp1 table")

    # Plot absolute ΔMSE; underlying CSVs may still store signed values.
    work[VALUE_COL] = work[VALUE_COL].astype(float).abs()

    work["layer"] = work["layer"].astype(int)
    work["band"] = work["band"].astype(int)
    work["band_label"] = work["band"].map(_band_tick_label)

    if "n_seeds" not in work.columns:
        work["n_seeds"] = 1

    keep = ["idx", "layer", "band", "band_label", VALUE_COL, "n_seeds"]
    optional = [c for c in ("mse_diff_std", "mse_base_mean", "mse_with_hook_mean") if c in work.columns]
    return work[keep + optional].copy()


def summarize_violin_long(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (layer, band), sub in long_df.groupby(["layer", "band"], sort=True):
        values = sub[VALUE_COL].astype(float)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        rows.append(
            {
                "layer": int(layer),
                "band": int(band),
                "band_label": _band_tick_label(int(band)),
                "n": int(len(values)),
                "median": float(med),
                "q1": float(q1),
                "q3": float(q3),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_exp1_band_knockout_violin_layer(
    long_df: pd.DataFrame,
    layer: int,
    dataset: str,
    out_png: Path,
    *,
    n_seeds: int,
) -> bool:
    sub = long_df[long_df["layer"] == layer].copy()
    if sub.empty:
        print(f"[SKIP] empty layer={layer} for {out_png}")
        return False

    band_order = sorted(sub["band"].unique())
    band_labels = [_band_tick_label(b) for b in band_order]
    sub["band_label"] = pd.Categorical(sub["band_label"], categories=band_labels, ordered=True)

    n_samples = int(sub["idx"].nunique())
    ds_label = _dataset_display(dataset)
    title = (
        f"Exp1 band knockout |ΔMSE| — {ds_label} layer {layer} "
        f"(n={n_samples} test samples, {n_seeds} seeds mean)"
    )

    fig_w = max(8, min(16, len(band_order) * 0.9))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    sns.violinplot(
        data=sub,
        x="band_label",
        y=VALUE_COL,
        order=band_labels,
        inner="box",
        cut=0,
        linewidth=0.8,
        ax=ax,
        color="steelblue",
    )
    ax.set_xlabel("Frequency band")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")
    return True


def process_dataset(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    force: bool,
) -> Tuple[bool, Optional[pd.DataFrame], int, int]:
    out_dir = exp_agg_dir(analysis_root, dataset, "exp1")
    long_csv = out_dir / LONG_CSV
    summary_csv = out_dir / SUMMARY_CSV

    layers: list[int] = []
    idx_level, source = load_idx_level_exp1(
        analysis_root,
        dataset,
        seeds,
        exp1_subdir,
        aggregated_subdirs,
    )
    if idx_level.empty:
        print(f"[FAIL] {dataset}: no Exp1 idx-level data ({exp1_subdir})")
        return False, None, 0, 0

    long_df = build_violin_long_df(idx_level)
    if long_df.empty:
        print(f"[FAIL] {dataset}: empty long df after filtering test split")
        return False, None, 0, 0

    summary_df = summarize_violin_long(long_df)
    layers = sorted(long_df["layer"].unique())
    n_samples = int(long_df["idx"].nunique())
    n_seeds = int(long_df["n_seeds"].max()) if "n_seeds" in long_df.columns else len(seeds)
    n_bands = int(long_df["band"].nunique())

    layer_pngs = [out_dir / f"exp1_band_knockout_violin_layer{layer}.png" for layer in layers]
    outputs_ready = (
        long_csv.is_file()
        and summary_csv.is_file()
        and layer_pngs
        and all(p.is_file() for p in layer_pngs)
    )
    if outputs_ready and not force:
        print(f"[SKIP] {dataset}: outputs exist under {out_dir} (use --force to overwrite)")
        return True, long_df, n_samples, n_seeds

    print(
        f"[OK] {dataset}: source={source} n_samples={n_samples} "
        f"n_seeds={n_seeds} n_layers={len(layers)} n_bands={n_bands}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[saved] {long_csv} ({len(long_df)} rows)")
    print(f"[saved] {summary_csv}")

    for layer in layers:
        plot_exp1_band_knockout_violin_layer(
            long_df,
            layer,
            dataset,
            out_dir / f"exp1_band_knockout_violin_layer{layer}.png",
            n_seeds=n_seeds,
        )

    return True, long_df, n_samples, n_seeds


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    seeds: Sequence[int],
    exp1_subdir: str,
    aggregated_subdirs: Sequence[str],
    *,
    force: bool,
) -> int:
    ok = True
    for dataset in datasets:
        success, _, n_samples, n_seeds = process_dataset(
            analysis_root,
            dataset,
            seeds,
            exp1_subdir,
            aggregated_subdirs,
            force=force,
        )
        if not success:
            ok = False
            continue
        print(f"[{dataset}] n_samples={n_samples} n_seeds={n_seeds}")
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
    ap.add_argument("--exp1-subdir", default=DEFAULT_EXP1_SUBDIR)
    ap.add_argument(
        "--aggregated-subdirs",
        nargs="*",
        default=list(DEFAULT_AGGREGATED_SUBDIRS),
        help="Search order for per_sample_band_sensitivity_aggregated.csv",
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
        args.exp1_subdir,
        args.aggregated_subdirs,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
