#!/usr/bin/env python3
"""Aggregate per-sample FFT-LAG mechanism metrics across train seeds and plot line + error bands."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = REPO_ROOT.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from AMPCliff.spectrual_filter.hidden_energy import plot_mse_diff_in_groups
from AMPCliff.utils.fftlag_latent_viz import (
    plot_attn_deviation_heatmap,
    plot_gate_input_band_profile,
    plot_gate_input_latent_vs_freq,
    plot_gate_input_query_summary,
    plot_query_contribution,
    plot_weighted_readout_heatmap,
)


def _expand_malformed_cli_token(arg: str) -> list[str]:
    """Split a single argv token that embeds multiple CLI flags (pasted `\\ --flag` one-liner)."""
    stripped = arg.strip()
    if not stripped:
        return []
    if " --" in arg or (arg.startswith(" ") and "--" in arg):
        try:
            return shlex.split(arg)
        except ValueError:
            return [stripped]
    return [stripped if arg != stripped else arg]


def _sanitize_argv(argv: list[str]) -> list[str]:
    """Normalize argv corrupted by pasted shell line continuations."""
    if not argv:
        return argv
    cleaned = [argv[0]]
    for arg in argv[1:]:
        cleaned.extend(_expand_malformed_cli_token(arg))
    return cleaned


COMPARE_SCALAR_COLS = [
    "cos_latent_freq",
    "gi_latent_l2",
    "gi_freq_l2",
    "cos_gate_latent_vs_freq_cf",
    "cos_gate_latent_vs_cached_raw",
]


def _train_seed_from_dir(dataset_run_dir: Path) -> int:
    parent_name = dataset_run_dir.parent.name
    if not parent_name.startswith("seed_"):
        raise ValueError(f"Expected parent seed_* dir, got {dataset_run_dir}")
    return int(parent_name.replace("seed_", ""))


def _dataset_display(dataset: str) -> str:
    return "E. coli" if dataset == "e_coli" else "S. aureus"


def _collect_seed_dirs(analysis_root: Path, dataset: str, seeds: Sequence[int]) -> list[Path]:
    out: list[Path] = []
    for seed in seeds:
        ds_dir = analysis_root / f"seed_{seed}" / dataset
        if ds_dir.is_dir():
            out.append(ds_dir)
    return out


def _load_seed_csvs(seed_dirs: Sequence[Path], rel_path: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for sd in seed_dirs:
        path = sd / rel_path
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df["seed"] = _train_seed_from_dir(sd)
        frames.append(df)
    return frames


def _add_profile_diff(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_diff"] = out["latent_pooled_readout"] - out["freq_uniform_pool_energy"]
    return out


def _build_gate_band_profile_table(
    latent_profile_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate latent, freq, and per-seed profile_diff across train seeds."""
    if not latent_profile_frames:
        return pd.DataFrame()
    diff_frames = [_add_profile_diff(df) for df in latent_profile_frames]
    group_cols = ["idx", "band", "band_start", "band_end"]
    latent_agg = aggregate_across_seeds(diff_frames, group_cols, ["latent_pooled_readout"])
    freq_agg = aggregate_across_seeds(diff_frames, group_cols, ["freq_uniform_pool_energy"])
    diff_agg = aggregate_across_seeds(diff_frames, group_cols, ["profile_diff"])
    merged = latent_agg.merge(
        freq_agg[
            [
                "idx",
                "band",
                "band_start",
                "band_end",
                "freq_uniform_pool_energy_mean",
                "freq_uniform_pool_energy_std",
            ]
        ],
        on=group_cols,
        how="inner",
    ).merge(
        diff_agg[
            [
                "idx",
                "band",
                "band_start",
                "band_end",
                "profile_diff_mean",
                "profile_diff_std",
            ]
        ],
        on=group_cols,
        how="inner",
    )
    return merged


def aggregate_across_seeds(
    frames: Sequence[pd.DataFrame],
    group_cols: Sequence[str],
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    agg_map: dict = {}
    for col in value_cols:
        agg_map[f"{col}_mean"] = (col, "mean")
        agg_map[f"{col}_std"] = (col, "std")
    return all_df.groupby(list(group_cols), as_index=False).agg(
        **agg_map,
        n_seeds=("seed", "nunique"),
    )


def _plot_line_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    label: str,
    color,
    n_seeds: int,
) -> None:
    ax.plot(x, mean, color=color, linewidth=1.5, alpha=0.92, label=label)
    if n_seeds >= 2:
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.22,
            linewidth=0,
        )


def _metric_series(sub: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, int]:
    mean_col = f"{value_col}_mean"
    std_col = f"{value_col}_std"
    n_seeds = int(sub["n_seeds"].max()) if "n_seeds" in sub.columns and not sub.empty else 1
    mean = sub[mean_col].to_numpy(dtype=float)
    std = sub[std_col].fillna(0.0).to_numpy(dtype=float)
    return mean, std, n_seeds


def plot_exp1_mse_diff_by_band(
    agg: pd.DataFrame,
    idx: int,
    dataset: str,
    out_png: Path,
) -> None:
    sub = agg[(agg["idx"] == idx) & (agg["split"] == "test")].copy()
    if sub.empty:
        return

    layers = sorted(sub["layer"].unique())
    n_layers = len(layers)
    ncols = 3
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    n_seeds = int(sub["n_seeds"].max())

    for i, layer in enumerate(layers):
        ax = axes_flat[i]
        layer_sub = sub[sub["layer"] == layer].sort_values("band")
        x = layer_sub["band"].to_numpy(dtype=float)
        mean, std, _ = _metric_series(layer_sub, "mse_diff")
        _plot_line_band(
            ax,
            x,
            mean,
            std,
            label=f"layer {int(layer)}",
            color="steelblue",
            n_seeds=n_seeds,
        )
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Layer {int(layer)}")
        ax.set_xlabel("Band")
        ax.set_ylabel("MSE diff")
        ax.grid(True, alpha=0.25)

    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Exp1 band knockout — {_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_effective_weight_by_band(
    agg: pd.DataFrame,
    idx: int,
    dataset: str,
    out_png: Path,
) -> None:
    sub = agg[agg["idx"] == idx].sort_values("band")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = sub["band"].to_numpy(dtype=float)
    mean, std, n_seeds = _metric_series(sub, "effective_weight")
    _plot_line_band(
        ax,
        x,
        mean,
        std,
        label="effective_weight",
        color="darkorange",
        n_seeds=n_seeds,
    )
    ax.set_xlabel("Band")
    ax.set_ylabel("Effective gate weight")
    ax.set_title(
        f"Exp2 PSD gate — {_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_query_band_lines(
    agg: pd.DataFrame,
    idx: int,
    dataset: str,
    value_col: str,
    ylabel: str,
    title_suffix: str,
    out_png: Path,
) -> None:
    sub = agg[agg["idx"] == idx].copy()
    if sub.empty:
        return

    queries = sorted(sub["query"].unique())
    n_seeds = int(sub["n_seeds"].max())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.tab10.colors

    for qi, query in enumerate(queries):
        qsub = sub[sub["query"] == query].sort_values("band")
        if qsub.empty:
            continue
        x = qsub["band"].to_numpy(dtype=float)
        mean, std, _ = _metric_series(qsub, value_col)
        c = colors[qi % len(colors)]
        _plot_line_band(
            ax,
            x,
            mean,
            std,
            label=f"q{int(query)}",
            color=c,
            n_seeds=n_seeds,
        )

    ax.set_xlabel("Band")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_suffix} — {_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_query_scalar_lines(
    agg: pd.DataFrame,
    idx: int,
    dataset: str,
    value_col: str,
    ylabel: str,
    title_suffix: str,
    out_png: Path,
) -> None:
    sub = agg[agg["idx"] == idx].sort_values("query")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = sub["query"].to_numpy(dtype=float)
    mean, std, n_seeds = _metric_series(sub, value_col)
    _plot_line_band(
        ax,
        x,
        mean,
        std,
        label=value_col,
        color="steelblue",
        n_seeds=n_seeds,
    )
    ax.set_xlabel("Query")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_suffix} — {_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_gate_band_profile(
    profile_agg: pd.DataFrame,
    idx: int,
    dataset: str,
    out_png: Path,
) -> None:
    sub = profile_agg[profile_agg["idx"] == idx].sort_values("band")
    if sub.empty:
        return

    n_seeds = int(sub["n_seeds"].max())
    fig, ax = plt.subplots(figsize=(8, 4))
    x = sub["band"].to_numpy(dtype=float)
    mean, std, _ = _metric_series(sub, "profile_diff")
    _plot_line_band(
        ax,
        x,
        mean,
        std,
        label="latent − freq",
        color="steelblue",
        n_seeds=n_seeds,
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Band")
    ax.set_ylabel("latent − freq (pooled readout)")
    ax.set_title(
        f"Exp4 gate band profile (latent − freq) — "
        f"{_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_gate_compare_scalars(
    agg: pd.DataFrame,
    idx: int,
    dataset: str,
    out_png: Path,
) -> None:
    sub = agg[agg["idx"] == idx]
    if sub.empty:
        return

    row = sub.iloc[0]
    n_seeds = int(row.get("n_seeds", 1))
    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for col in COMPARE_SCALAR_COLS:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col not in row.index:
            continue
        labels.append(col)
        means.append(float(row[mean_col]))
        stds.append(float(row.get(std_col, 0.0) or 0.0))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Exp4 gate compare — {_dataset_display(dataset)} idx={idx} (n_seeds={n_seeds})"
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mean_cols_to_viz_df(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """Map aggregated *_mean columns to names expected by fftlag_latent_viz."""
    out = df.copy()
    for src, dst in rename_map.items():
        if src in out.columns:
            out[dst] = out[src]
    drop_cols = [c for c in out.columns if c.endswith("_std") or c == "n_seeds"]
    return out.drop(columns=drop_cols, errors="ignore")


def _n_seeds_from_agg(*frames: Optional[pd.DataFrame]) -> int:
    for frame in frames:
        if frame is not None and not frame.empty and "n_seeds" in frame.columns:
            return int(frame["n_seeds"].max())
    return 1


def plot_exp1_mean_style_plots(
    *,
    idx: int,
    dataset: str,
    out_dir: Path,
    exp1_agg: pd.DataFrame,
    k_bands: int = 8,
    group_size: int = 5,
    split: str = "test",
    base: int = 4,
) -> None:
    """Single-seed-style Exp1 band-group line plots and layer×band heatmap (mean only)."""
    sub = exp1_agg[(exp1_agg["idx"] == idx) & (exp1_agg["split"] == split)].copy()
    if sub.empty:
        return

    n_seeds = _n_seeds_from_agg(sub)
    sample_label = f"{_dataset_display(dataset)} idx={idx} (mean across {n_seeds} seeds)"
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = _mean_cols_to_viz_df(sub, {"mse_diff_mean": "mse_diff"})
    plot_df = viz[["layer", "band", "mse_diff"]].copy()
    plot_df["split"] = split

    base_title = (
        f"Exp1 band knockout — {sample_label}\n"
        "MSE difference: (with-filter − baseline) — {{split}} (k={{k}}, base={{base}})"
    )
    plot_mse_diff_in_groups(
        plot_df,
        k_bands=k_bands,
        group_size=group_size,
        base_title=base_title,
        split=split,
        k=k_bands,
        base=base,
        out_prefix=str(out_dir / "mse_diff_bandgroup_"),
        palette_name="tab10",
        set_common_ylim=True,
    )

    pivot = plot_df.pivot(index="layer", columns="band", values="mse_diff").sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        center=0.0,
        cmap="RdBu_r",
        annot=False,
        linewidths=0.5,
        cbar_kws={"label": "MSE diff"},
        ax=ax,
    )
    ax.set_xlabel("Band")
    ax.set_ylabel("Layer")
    ax.set_title(f"Exp1 MSE diff layer×band — {sample_label}")
    fig.tight_layout()
    fig.savefig(out_dir / "mse_diff_layer_band_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_mean_style_plots(
    *,
    idx: int,
    dataset: str,
    out_dir: Path,
    exp2_agg: pd.DataFrame,
) -> None:
    """Single-seed-style Exp2 gate summary at band granularity (mean only)."""
    sub = exp2_agg[exp2_agg["idx"] == idx].sort_values("band")
    if sub.empty:
        return

    n_seeds = _n_seeds_from_agg(sub)
    sample_label = f"{_dataset_display(dataset)} idx={idx} (mean across {n_seeds} seeds)"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bands = sub["band"].to_numpy(dtype=int)
    ax0 = axes[0]
    ax0.bar(bands, sub["effective_weight_mean"].to_numpy(), color="steelblue", alpha=0.8)
    ax0.set_xlabel("Band index")
    ax0.set_ylabel("Effective weight (energy ratio)")
    ax0.set_title("Gate effective weight per band")
    ax0.set_xticks(bands)

    ax1 = axes[1]
    x = np.arange(len(sub))
    width = 0.35
    ax1.bar(
        x - width / 2,
        sub["energy_before_mean"].to_numpy(),
        width,
        label="Before gate",
        color="steelblue",
        alpha=0.8,
    )
    ax1.bar(
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
    ax1.set_xticks(x)
    ax1.set_xticklabels(band_labels, fontsize=7)
    ax1.set_xlabel("Frequency band")
    ax1.set_ylabel("Mean energy")
    ax1.set_title("Energy before vs after gate")
    ax1.legend()

    fig.suptitle(f"Exp2 PSD gate — {sample_label}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_band_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exp4_mean_style_plots(
    *,
    idx: int,
    dataset: str,
    out_dir: Path,
    dev_agg: Optional[pd.DataFrame],
    readout_agg: Optional[pd.DataFrame],
    contrib_agg: Optional[pd.DataFrame],
    profile_agg: Optional[pd.DataFrame],
    compare_agg: Optional[pd.DataFrame],
) -> None:
    """Render single-seed-style figures using cross-seed mean values only (no error bands)."""
    n_seeds = _n_seeds_from_agg(dev_agg, readout_agg, contrib_agg, profile_agg, compare_agg)
    seed_tag = f"(mean across {n_seeds} seeds)"
    sample_label = f"{_dataset_display(dataset)} idx={idx} {seed_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dev_agg is not None:
        sub = dev_agg[dev_agg["idx"] == idx]
        if not sub.empty:
            viz = _mean_cols_to_viz_df(sub, {"deviation_mean": "deviation"})
            plot_attn_deviation_heatmap(
                viz,
                str(out_dir / "latent_attn_deviation_heatmap.png"),
                title=f"Attention deviation from uniform — {sample_label}",
            )

    if readout_agg is not None:
        sub = readout_agg[readout_agg["idx"] == idx]
        if not sub.empty:
            viz = _mean_cols_to_viz_df(sub, {"weighted_mass_mean": "weighted_mass"})
            plot_weighted_readout_heatmap(
                viz,
                str(out_dir / "latent_weighted_readout_heatmap.png"),
                title=f"Value-weighted band readout — {sample_label}",
            )

    if contrib_agg is not None:
        sub = contrib_agg[contrib_agg["idx"] == idx]
        if not sub.empty:
            viz = _mean_cols_to_viz_df(
                sub,
                {
                    "l2_deviation_mean": "l2_deviation",
                    "cos_to_gate_input_mean": "cos_to_gate_input",
                },
            )
            plot_query_contribution(
                viz,
                str(out_dir / "latent_query_contribution.png"),
                title=f"Latent query contribution to gate_input — {sample_label}",
            )

    if profile_agg is not None:
        sub = profile_agg[profile_agg["idx"] == idx]
        if not sub.empty:
            viz = _mean_cols_to_viz_df(
                sub,
                {
                    "latent_pooled_readout_mean": "latent_pooled_readout",
                    "freq_uniform_pool_energy_mean": "freq_uniform_pool_energy",
                },
            )
            plot_gate_input_band_profile(
                viz,
                str(out_dir / "gate_input_band_profile.png"),
                title=f"Gate-input band profiles — {sample_label}",
            )

    if compare_agg is not None:
        sub = compare_agg[compare_agg["idx"] == idx]
        if not sub.empty:
            compare_rename = {
                f"{col}_mean": col for col in COMPARE_SCALAR_COLS if f"{col}_mean" in sub.columns
            }
            viz_cmp = _mean_cols_to_viz_df(sub, compare_rename)
            plot_gate_input_latent_vs_freq(
                viz_cmp,
                str(out_dir / "gate_input_latent_vs_freq.png"),
                title=f"Gate input latent vs freq — {sample_label}",
            )
            if contrib_agg is not None:
                contrib_sub = contrib_agg[contrib_agg["idx"] == idx]
                if not contrib_sub.empty:
                    viz_contrib = _mean_cols_to_viz_df(
                        contrib_sub,
                        {"cos_to_gate_input_mean": "cos_to_gate_input"},
                    )
                    plot_gate_input_query_summary(
                        viz_contrib,
                        viz_cmp,
                        str(out_dir / "gate_input_query_to_mean.png"),
                        title=f"Queries vs gate_input — {sample_label}",
                    )


def _discover_idxs(seed_dirs: Sequence[Path], exps: Sequence[int]) -> list[int]:
    idx_sets: list[set[int]] = []
    if 1 in exps:
        frames = _load_seed_csvs(seed_dirs, "exp1_band_knockout/per_sample_band_sensitivity.csv")
        if frames:
            idx_sets.append(set(int(x) for x in pd.concat(frames)["idx"].unique()))
    if 2 in exps:
        frames = _load_seed_csvs(seed_dirs, "exp2_psd_gate/per_sample_gate_by_band.csv")
        if frames:
            idx_sets.append(set(int(x) for x in pd.concat(frames)["idx"].unique()))
    if 4 in exps:
        frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_query_contribution.csv")
        if frames:
            idx_sets.append(set(int(x) for x in pd.concat(frames)["idx"].unique()))

    if not idx_sets:
        return []
    common = idx_sets[0]
    for s in idx_sets[1:]:
        common &= s
    return sorted(common)


def process_dataset(
    analysis_root: Path,
    dataset: str,
    seeds: Sequence[int],
    exps: Sequence[int],
    idx_filter: Optional[Sequence[int]],
    *,
    skip_csv: bool,
    skip_plots: bool,
    skip_mean_style_plots: bool,
) -> None:
    seed_dirs = _collect_seed_dirs(analysis_root, dataset, seeds)
    if not seed_dirs:
        print(f"[SKIP] No seed dirs for dataset={dataset}")
        return

    idxs = _discover_idxs(seed_dirs, exps)
    if idx_filter:
        idx_filter_set = set(int(i) for i in idx_filter)
        idxs = [i for i in idxs if i in idx_filter_set]
    if not idxs:
        print(f"[SKIP] No common idx for dataset={dataset}")
        return

    exp1_agg: Optional[pd.DataFrame] = None
    exp2_agg: Optional[pd.DataFrame] = None
    exp4_dev_agg: Optional[pd.DataFrame] = None
    exp4_readout_agg: Optional[pd.DataFrame] = None
    exp4_contrib_agg: Optional[pd.DataFrame] = None
    exp4_profile_agg: Optional[pd.DataFrame] = None
    exp4_compare_agg: Optional[pd.DataFrame] = None

    if 1 in exps:
        frames = _load_seed_csvs(seed_dirs, "exp1_band_knockout/per_sample_band_sensitivity.csv")
        exp1_agg = aggregate_across_seeds(
            frames,
            ["idx", "layer", "band", "split"],
            ["mse_diff", "mse_base", "mse_with_hook"],
        )

    if 2 in exps:
        frames = _load_seed_csvs(seed_dirs, "exp2_psd_gate/per_sample_gate_by_band.csv")
        exp2_agg = aggregate_across_seeds(
            frames,
            ["idx", "band", "band_start", "band_end"],
            ["effective_weight", "energy_before", "energy_after"],
        )

    if 4 in exps:
        dev_frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_attn_band_deviation.csv")
        exp4_dev_agg = aggregate_across_seeds(
            dev_frames,
            ["idx", "query", "band", "band_start", "band_end"],
            ["deviation"],
        )
        readout_frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_weighted_band_readout.csv")
        exp4_readout_agg = aggregate_across_seeds(
            readout_frames,
            ["idx", "query", "band", "band_start", "band_end"],
            ["weighted_mass"],
        )
        contrib_frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_query_contribution.csv")
        exp4_contrib_agg = aggregate_across_seeds(
            contrib_frames,
            ["idx", "query"],
            ["l2_deviation", "cos_to_gate_input"],
        )
        latent_profile_frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_gate_input_band_profile.csv")
        if latent_profile_frames:
            exp4_profile_agg = _build_gate_band_profile_table(latent_profile_frames)
        compare_frames = _load_seed_csvs(seed_dirs, "exp4_latent/latent_gate_input_compare.csv")
        compare_num = [
            c for c in COMPARE_SCALAR_COLS
            if compare_frames and c in compare_frames[0].columns
        ]
        if compare_frames and compare_num:
            exp4_compare_agg = aggregate_across_seeds(
                compare_frames,
                ["idx"],
                compare_num,
            )

    exp1_out = exp_agg_dir(analysis_root, dataset, "exp1")
    exp2_out = exp_agg_dir(analysis_root, dataset, "exp2")
    exp4_out = exp_agg_dir(analysis_root, dataset, "exp4")
    for d in (exp1_out, exp2_out, exp4_out):
        d.mkdir(parents=True, exist_ok=True)

    for idx in idxs:
        written: list[str] = []

        if 1 in exps and exp1_agg is not None:
            idx_dir = exp1_out / "per_sample" / f"idx_{idx}"
            plot_dir = idx_dir / "plots"
            idx_dir.mkdir(parents=True, exist_ok=True)
            sub = exp1_agg[exp1_agg["idx"] == idx]
            if not sub.empty:
                if not skip_csv:
                    sub.to_csv(idx_dir / "band_knockout_mse_diff.csv", index=False)
                if not skip_plots:
                    plot_exp1_mse_diff_by_band(
                        exp1_agg, idx, dataset, plot_dir / "mse_diff_by_band.png"
                    )
                    if not skip_mean_style_plots:
                        plot_exp1_mean_style_plots(
                            idx=idx,
                            dataset=dataset,
                            out_dir=plot_dir / "mean_style",
                            exp1_agg=exp1_agg,
                        )
                written.append(str(idx_dir))

        if 2 in exps and exp2_agg is not None:
            idx_dir = exp2_out / "per_sample" / f"idx_{idx}"
            plot_dir = idx_dir / "plots"
            idx_dir.mkdir(parents=True, exist_ok=True)
            sub = exp2_agg[exp2_agg["idx"] == idx]
            if not sub.empty:
                if not skip_csv:
                    sub.to_csv(idx_dir / "effective_weight_by_band.csv", index=False)
                if not skip_plots:
                    plot_exp2_effective_weight_by_band(
                        exp2_agg, idx, dataset, plot_dir / "effective_weight_by_band.png"
                    )
                    if not skip_mean_style_plots:
                        plot_exp2_mean_style_plots(
                            idx=idx,
                            dataset=dataset,
                            out_dir=plot_dir / "mean_style",
                            exp2_agg=exp2_agg,
                        )
                written.append(str(idx_dir))

        if 4 in exps:
            idx_dir = exp4_out / "per_sample" / f"idx_{idx}"
            plot_dir = idx_dir / "plots"
            idx_dir.mkdir(parents=True, exist_ok=True)
            has_exp4 = False
            if exp4_dev_agg is not None:
                sub = exp4_dev_agg[exp4_dev_agg["idx"] == idx]
                if not sub.empty:
                    has_exp4 = True
                    if not skip_csv:
                        sub.to_csv(idx_dir / "attn_band_deviation.csv", index=False)
                    if not skip_plots:
                        plot_exp4_query_band_lines(
                            exp4_dev_agg,
                            idx,
                            dataset,
                            "deviation",
                            "Deviation from uniform",
                            "Exp4 attention deviation",
                            plot_dir / "attn_deviation_by_band.png",
                        )
            if exp4_readout_agg is not None:
                sub = exp4_readout_agg[exp4_readout_agg["idx"] == idx]
                if not sub.empty:
                    has_exp4 = True
                    if not skip_csv:
                        sub.to_csv(idx_dir / "weighted_band_readout.csv", index=False)
                    if not skip_plots:
                        plot_exp4_query_band_lines(
                            exp4_readout_agg,
                            idx,
                            dataset,
                            "weighted_mass",
                            "Weighted mass",
                            "Exp4 weighted band readout",
                            plot_dir / "weighted_readout_by_band.png",
                        )
            if exp4_contrib_agg is not None:
                sub = exp4_contrib_agg[exp4_contrib_agg["idx"] == idx]
                if not sub.empty:
                    has_exp4 = True
                    if not skip_csv:
                        sub.to_csv(idx_dir / "query_contribution.csv", index=False)
                    if not skip_plots:
                        plot_exp4_query_scalar_lines(
                            exp4_contrib_agg,
                            idx,
                            dataset,
                            "l2_deviation",
                            "L2 deviation from mean(latent_out)",
                            "Exp4 query L2 contribution",
                            plot_dir / "query_l2_contribution.png",
                        )
                        plot_exp4_query_scalar_lines(
                            exp4_contrib_agg,
                            idx,
                            dataset,
                            "cos_to_gate_input",
                            "cos(query, gate_input_latent)",
                            "Exp4 query cos to gate_input",
                            plot_dir / "query_cos_to_gate_input.png",
                        )
            if exp4_profile_agg is not None:
                sub = exp4_profile_agg[exp4_profile_agg["idx"] == idx]
                if not sub.empty:
                    has_exp4 = True
                    if not skip_csv:
                        sub.to_csv(idx_dir / "gate_input_band_profile.csv", index=False)
                    if not skip_plots:
                        plot_exp4_gate_band_profile(
                            exp4_profile_agg,
                            idx,
                            dataset,
                            plot_dir / "gate_band_profile.png",
                        )
            if exp4_compare_agg is not None:
                sub = exp4_compare_agg[exp4_compare_agg["idx"] == idx]
                if not sub.empty:
                    has_exp4 = True
                    if not skip_csv:
                        sub.to_csv(idx_dir / "gate_input_compare.csv", index=False)
                    if not skip_plots:
                        plot_exp4_gate_compare_scalars(
                            exp4_compare_agg,
                            idx,
                            dataset,
                            plot_dir / "gate_compare_scalars.png",
                        )
            if has_exp4 and not skip_plots and not skip_mean_style_plots:
                plot_exp4_mean_style_plots(
                    idx=idx,
                    dataset=dataset,
                    out_dir=plot_dir / "mean_style",
                    dev_agg=exp4_dev_agg,
                    readout_agg=exp4_readout_agg,
                    contrib_agg=exp4_contrib_agg,
                    profile_agg=exp4_profile_agg,
                    compare_agg=exp4_compare_agg,
                )
            if has_exp4:
                written.append(str(idx_dir))

        if written:
            print(f"[OK] dataset={dataset} idx={idx} -> {', '.join(written)}")


def main() -> int:
    sys.argv = _sanitize_argv(list(sys.argv))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="e_coli or s_aureus (repeatable; default: both)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Train seeds to aggregate (default: 0..9)",
    )
    ap.add_argument(
        "--exps",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        choices=[1, 2, 4],
        help="Experiments to include (default: 1 2 4)",
    )
    ap.add_argument(
        "--idx",
        type=int,
        nargs="+",
        default=None,
        help="Only process these peptide idx values",
    )
    ap.add_argument("--skip-csv", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument(
        "--skip-mean-style-plots",
        action="store_true",
        help="Skip Exp1/2/4 single-seed-style mean plots under plots/mean_style/",
    )
    args = ap.parse_args()

    datasets = args.datasets or ["s_aureus", "e_coli"]
    for dataset in datasets:
        process_dataset(
            args.analysis_root,
            dataset,
            args.seeds,
            args.exps,
            args.idx,
            skip_csv=args.skip_csv,
            skip_plots=args.skip_plots,
            skip_mean_style_plots=args.skip_mean_style_plots,
        )

    print(f"Done. Outputs under {args.analysis_root / 'aggregated'}/{{dataset}}/exp{{1,2,4}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
