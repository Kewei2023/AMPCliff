#!/usr/bin/env python3
"""Exp5 / DC validation design v2 — Step 3 / 主实验一: DC property decoding probes.
Ridge property probes on DCT coefficient features (DC validation Step 3)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from AMPCliff.dc_property_probe import bootstrap_delta_spearman_ci, run_probe
from AMPCliff.dc_property_utils import (
    COEFFICIENT_KEYS,
    DCT_PROBE_COEFFICIENTS,
    PROBE_PROPERTIES,
    coefficient_label_plain,
    format_ci_range,
    format_mean_pm_std,
    mathfrak_b_label,
    mathfrak_b_norm_label,
    species_display_name,
)


def _parse_seeds(values: Optional[Sequence[str]]) -> List[int]:
    if not values:
        return list(range(10))
    out: List[int] = []
    for v in values:
        out.extend(int(x) for x in str(v).split(","))
    return out


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _feature_matrix(bundle: Dict[str, np.ndarray], coeff_key: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = bundle["idx"].astype(np.int64)
    X = bundle[coeff_key]
    return idx, X


def _split_feature_targets(
    property_df: pd.DataFrame,
    species: str,
    idx_to_pos: Dict[int, int],
    X_all: np.ndarray,
    prop: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub = property_df[property_df["species"] == species]
    split_frames: Dict[str, Tuple[List[np.ndarray], List[float]]] = {
        "train": ([], []),
        "valid": ([], []),
        "test": ([], []),
    }
    for split in split_frames:
        split_sub = sub[sub["split"] == split]
        xs, ys = split_frames[split]
        for _, row in split_sub.iterrows():
            i = int(row["idx"])
            if i not in idx_to_pos:
                continue
            xs.append(X_all[idx_to_pos[i]])
            ys.append(float(row[prop]))
    X_train, y_train = split_frames["train"]
    X_valid, y_valid = split_frames["valid"]
    X_test, y_test = split_frames["test"]
    return (
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_valid),
        np.asarray(y_valid),
        np.asarray(X_test),
        np.asarray(y_test),
    )


def _empty_metrics() -> Dict[str, float]:
    return {
        "spearman": np.nan,
        "r2": np.nan,
        "mae": np.nan,
        "spearman_ci_lo": np.nan,
        "spearman_ci_hi": np.nan,
    }


def analyze_seed(
    feature_path: Path,
    property_df: pd.DataFrame,
    species: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bundle = _load_npz(feature_path)
    rows = []
    delta_rows = []

    coeff_idx_to_key = COEFFICIENT_KEYS
    coeff_mats: Dict[int, Tuple[np.ndarray, Dict[int, int]]] = {}
    for coeff_idx, coeff_key in coeff_idx_to_key.items():
        idx, X_all = _feature_matrix(bundle, coeff_key)
        coeff_mats[coeff_idx] = (X_all, {int(i): p for p, i in enumerate(idx)})

    for prop in PROBE_PROPERTIES:
        c0_X, c0_pos = coeff_mats[0]
        c1_X, c1_pos = coeff_mats[1]
        splits_c0 = _split_feature_targets(property_df, species, c0_pos, c0_X, prop)
        splits_c1 = _split_feature_targets(property_df, species, c1_pos, c1_X, prop)

        c0_pred_out: Optional[Dict[str, object]] = None
        c1_pred_out: Optional[Dict[str, object]] = None

        for coeff_idx, splits in [(0, splits_c0), (1, splits_c1)]:
            X_train, y_train, X_valid, y_valid, X_test, y_test = splits
            if len(X_test) < 2 or len(X_train) < 2 or len(X_valid) < 2:
                metrics = _empty_metrics()
            else:
                metrics = run_probe(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    y_test,
                    bootstrap_seed=seed,
                    return_predictions=coeff_idx in (0, 1),
                )
                if coeff_idx == 0:
                    c0_pred_out = metrics
                elif coeff_idx == 1:
                    c1_pred_out = metrics

            rows.append(
                {
                    "species": species,
                    "seed": seed,
                    "property": prop,
                    "coefficient": coeff_idx,
                    **{k: metrics[k] for k in ("spearman", "r2", "mae", "spearman_ci_lo", "spearman_ci_hi")},
                }
            )

        for coeff_idx in (2, 3):
            X_all, idx_to_pos = coeff_mats[coeff_idx]
            splits = _split_feature_targets(property_df, species, idx_to_pos, X_all, prop)
            X_train, y_train, X_valid, y_valid, X_test, y_test = splits
            if len(X_test) < 2 or len(X_train) < 2 or len(X_valid) < 2:
                metrics = _empty_metrics()
            else:
                metrics = run_probe(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    y_test,
                    bootstrap_seed=seed,
                )
            rows.append(
                {
                    "species": species,
                    "seed": seed,
                    "property": prop,
                    "coefficient": coeff_idx,
                    **{k: metrics[k] for k in ("spearman", "r2", "mae", "spearman_ci_lo", "spearman_ci_hi")},
                }
            )

        if (
            c0_pred_out is not None
            and c1_pred_out is not None
            and "y_test" in c0_pred_out
            and "y_pred" in c0_pred_out
            and "y_pred" in c1_pred_out
        ):
            y_test = np.asarray(c0_pred_out["y_test"])
            pred_c0 = np.asarray(c0_pred_out["y_pred"])
            pred_c1 = np.asarray(c1_pred_out["y_pred"])
            delta, ci_lo, ci_hi = bootstrap_delta_spearman_ci(
                y_test, pred_c0, pred_c1, seed=seed
            )
            delta_rows.append(
                {
                    "species": species,
                    "seed": seed,
                    "property": prop,
                    "delta": delta,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "significant": bool(ci_lo > 0),
                }
            )
        else:
            delta_rows.append(
                {
                    "species": species,
                    "seed": seed,
                    "property": prop,
                    "delta": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "significant": False,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def build_dc_preference_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (species, seed, prop), grp in results.groupby(["species", "seed", "property"]):
        dc = grp[grp["coefficient"] == 0]
        non_dc = grp[grp["coefficient"].isin([1, 2, 3])]
        if dc.empty or non_dc.empty:
            continue
        dc_rho = float(dc["spearman"].iloc[0])
        best_non_dc = float(non_dc["spearman"].max())
        rows.append(
            {
                "species": species,
                "seed": seed,
                "property": prop,
                "dc_spearman": dc_rho,
                "best_non_dc_spearman": best_non_dc,
                "dc_preference": dc_rho - best_non_dc,
                "dc_ci_excludes_zero": bool(
                    dc["spearman_ci_lo"].iloc[0] > 0 or dc["spearman_ci_hi"].iloc[0] < 0
                ),
                "dc_beats_non_dc": bool(dc_rho > best_non_dc),
            }
        )
    return pd.DataFrame(rows)


def build_dc_preference_consensus(preference: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed DC preference into cross-seed consensus (7-8/10 rule)."""
    if preference.empty:
        return pd.DataFrame()

    preferred = preference[
        preference["dc_ci_excludes_zero"]
        & preference["dc_beats_non_dc"]
        & (preference["dc_preference"] > 0)
    ]
    counts = (
        preferred.groupby(["species", "property"], as_index=False)
        .size()
        .rename(columns={"size": "n_seeds_dc_preferred"})
    )
    totals = (
        preference.groupby(["species", "property"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds_total"})
    )
    out = totals.merge(counts, on=["species", "property"], how="left")
    out["n_seeds_dc_preferred"] = out["n_seeds_dc_preferred"].fillna(0).astype(int)
    out["passes_7of10"] = out["n_seeds_dc_preferred"] >= 7
    out["passes_8of10"] = out["n_seeds_dc_preferred"] >= 8
    return out.sort_values(["species", "property"]).reset_index(drop=True)


def build_delta_consensus(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    agg = (
        delta_df.groupby(["species", "property"], as_index=False)
        .agg(
            mean_delta=("delta", "mean"),
            n_seeds_significant=("significant", "sum"),
            n_seeds_total=("seed", "nunique"),
        )
    )
    agg["passes_7of10"] = agg["n_seeds_significant"] >= 7
    agg["passes_8of10"] = agg["n_seeds_significant"] >= 8
    return agg.sort_values(["species", "property"]).reset_index(drop=True)


def build_c0_c1_metric_delta_summary(results: pd.DataFrame, species: str) -> pd.DataFrame:
    """Per-seed C0 vs C1 R2/MAE deltas, aggregated to mean ± std across seeds."""
    sub = results[
        (results["species"] == species) & (results["coefficient"].isin([0, 1]))
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    pivot_r2 = sub.pivot_table(index=["seed", "property"], columns="coefficient", values="r2")
    pivot_mae = sub.pivot_table(index=["seed", "property"], columns="coefficient", values="mae")
    if 0 not in pivot_r2.columns or 1 not in pivot_r2.columns:
        return pd.DataFrame()
    if 0 not in pivot_mae.columns or 1 not in pivot_mae.columns:
        return pd.DataFrame()

    per_seed = pd.DataFrame(
        {
            "property": pivot_r2.index.get_level_values("property"),
            "delta_r2": pivot_r2[0].to_numpy() - pivot_r2[1].to_numpy(),
            "delta_mae": pivot_mae[1].to_numpy() - pivot_mae[0].to_numpy(),
        }
    )
    agg = (
        per_seed.groupby("property", as_index=False)
        .agg(
            delta_r2_mean=("delta_r2", "mean"),
            delta_r2_std=("delta_r2", "std"),
            delta_mae_mean=("delta_mae", "mean"),
            delta_mae_std=("delta_mae", "std"),
        )
        .set_index("property")
        .reindex(PROBE_PROPERTIES)
        .reset_index()
    )
    agg["delta_r2_std"] = agg["delta_r2_std"].fillna(0.0)
    agg["delta_mae_std"] = agg["delta_mae_std"].fillna(0.0)
    return agg


def _summarize_spearman_delta(delta_df: pd.DataFrame, species: str) -> pd.DataFrame:
    sub = delta_df[delta_df["species"] == species].copy()
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby("property", as_index=False)
        .agg(delta=("delta", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
        .set_index("property")
        .reindex(PROBE_PROPERTIES)
        .reset_index()
    )


def _plot_delta_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    ylabel: str,
    show_xticks: bool = False,
    show_ylabel: bool = True,
) -> None:
    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, color="steelblue")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    if show_xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(PROBE_PROPERTIES, rotation=45, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([])


def _plot_delta_ci_3panel_column(
    axes: Sequence[plt.Axes],
    delta_df: pd.DataFrame,
    results: pd.DataFrame,
    species: str,
    *,
    show_ylabel: bool = True,
    show_xticks: bool = True,
) -> bool:
    spearman_summary = _summarize_spearman_delta(delta_df, species)
    metric_summary = build_c0_c1_metric_delta_summary(results, species)
    if spearman_summary.empty or metric_summary.empty:
        return False

    b0 = mathfrak_b_norm_label(0)[1:-1]
    b1 = mathfrak_b_label(1)[1:-1]
    x = np.arange(len(PROBE_PROPERTIES))
    spearman_yerr = np.vstack(
        [
            spearman_summary["delta"] - spearman_summary["ci_lo"],
            spearman_summary["ci_hi"] - spearman_summary["delta"],
        ]
    )
    _plot_delta_errorbar(
        axes[0],
        x,
        spearman_summary["delta"].to_numpy(),
        spearman_yerr,
        ylabel=rf"$\Delta\rho$ (${b0}$ $-$ ${b1}$)",
        show_ylabel=show_ylabel,
    )

    r2_yerr = np.vstack(
        [
            metric_summary["delta_r2_std"].to_numpy(),
            metric_summary["delta_r2_std"].to_numpy(),
        ]
    )
    _plot_delta_errorbar(
        axes[1],
        x,
        metric_summary["delta_r2_mean"].to_numpy(),
        r2_yerr,
        ylabel=rf"$\Delta R^2$ (${b0}$ $-$ ${b1}$)",
        show_ylabel=show_ylabel,
    )

    mae_yerr = np.vstack(
        [
            metric_summary["delta_mae_std"].to_numpy(),
            metric_summary["delta_mae_std"].to_numpy(),
        ]
    )
    _plot_delta_errorbar(
        axes[2],
        x,
        metric_summary["delta_mae_mean"].to_numpy(),
        mae_yerr,
        ylabel=rf"$\Delta$MAE (${b1}$ $-$ ${b0}$)",
        show_ylabel=show_ylabel,
        show_xticks=show_xticks,
    )
    return True


def build_probe_results_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed probe metrics into mean ± std summary rows."""
    if results.empty:
        return pd.DataFrame()

    agg = (
        results.groupby(["species", "property", "coefficient"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            spearman_ci_lo_mean=("spearman_ci_lo", "mean"),
            spearman_ci_lo_std=("spearman_ci_lo", "std"),
            spearman_ci_hi_mean=("spearman_ci_hi", "mean"),
            spearman_ci_hi_std=("spearman_ci_hi", "std"),
        )
    )
    agg["spearman_mean_pm_std"] = agg.apply(
        lambda row: format_mean_pm_std(row["spearman_mean"], row["spearman_std"]),
        axis=1,
    )
    agg["r2_mean_pm_std"] = agg.apply(
        lambda row: format_mean_pm_std(row["r2_mean"], row["r2_std"]),
        axis=1,
    )
    agg["mae_mean_pm_std"] = agg.apply(
        lambda row: format_mean_pm_std(row["mae_mean"], row["mae_std"]),
        axis=1,
    )
    agg["spearman_ci_lo_mean_pm_std"] = agg.apply(
        lambda row: format_mean_pm_std(row["spearman_ci_lo_mean"], row["spearman_ci_lo_std"]),
        axis=1,
    )
    agg["spearman_ci_hi_mean_pm_std"] = agg.apply(
        lambda row: format_mean_pm_std(row["spearman_ci_hi_mean"], row["spearman_ci_hi_std"]),
        axis=1,
    )
    agg["spearman_ci_range"] = agg.apply(
        lambda row: format_ci_range(row["spearman_ci_lo_mean"], row["spearman_ci_hi_mean"]),
        axis=1,
    )
    agg["coefficient_label"] = agg["coefficient"].map(coefficient_label_plain)

    agg["property"] = pd.Categorical(agg["property"], categories=PROBE_PROPERTIES, ordered=True)
    agg["coefficient"] = pd.Categorical(
        agg["coefficient"],
        categories=list(DCT_PROBE_COEFFICIENTS),
        ordered=True,
    )
    agg = agg.sort_values(["species", "property", "coefficient"]).reset_index(drop=True)
    agg["property"] = agg["property"].astype(str)
    agg["coefficient"] = agg["coefficient"].astype(int)

    column_order = [
        "species",
        "property",
        "coefficient",
        "coefficient_label",
        "n_seeds",
        "spearman_mean",
        "spearman_std",
        "spearman_mean_pm_std",
        "r2_mean",
        "r2_std",
        "r2_mean_pm_std",
        "mae_mean",
        "mae_std",
        "mae_mean_pm_std",
        "spearman_ci_lo_mean",
        "spearman_ci_lo_std",
        "spearman_ci_lo_mean_pm_std",
        "spearman_ci_hi_mean",
        "spearman_ci_hi_std",
        "spearman_ci_hi_mean_pm_std",
        "spearman_ci_range",
    ]
    return agg[column_order]


def plot_property_encoding_heatmap(
    results: pd.DataFrame,
    species: str,
    out_png: Path | None = None,
    *,
    ax: plt.Axes | None = None,
    cmap: str = "cividis",
    vmin: float = 0.55,
    vmax: float = 1.00,
    show_cbar: bool = True,
    title: str | None = None,
    cbar_ax: plt.Axes | None = None,
) -> plt.Axes | None:
    sub = results[results["species"] == species].copy()
    if sub.empty:
        return None
    pivot = (
        sub.groupby(["property", "coefficient"], as_index=False)["spearman"]
        .mean()
        .pivot(index="property", columns="coefficient", values="spearman")
    )
    pivot = pivot.reindex(index=PROBE_PROPERTIES, columns=list(DCT_PROBE_COEFFICIENTS))
    pivot.columns = [mathfrak_b_label(c) for c in pivot.columns]

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=show_cbar and cbar_ax is None,
        cbar_ax=cbar_ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("DCT coefficient")
    ax.set_ylabel("Property")
    ax.set_title(title if title is not None else f"DC property encoding ({species_display_name(species)})")

    if owns_figure:
        if out_png is None:
            raise ValueError("out_png is required when ax is not provided")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_delta_ci(delta_df: pd.DataFrame, species: str, out_png: Path) -> None:
    sub = delta_df[delta_df["species"] == species].copy()
    if sub.empty:
        return
    summary = (
        sub.groupby("property", as_index=False)
        .agg(delta=("delta", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"))
        .set_index("property")
        .reindex(PROBE_PROPERTIES)
        .reset_index()
    )
    x = np.arange(len(summary))
    yerr = np.vstack(
        [
            summary["delta"] - summary["ci_lo"],
            summary["ci_hi"] - summary["delta"],
        ]
    )
    plt.figure(figsize=(9, 5))
    plt.errorbar(x, summary["delta"], yerr=yerr, fmt="o", capsize=4, color="steelblue")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xticks(x, summary["property"], rotation=45, ha="right")
    b0_norm = mathfrak_b_norm_label(0)
    b1 = mathfrak_b_label(1)
    plt.ylabel(rf"$\Delta_{{\mathrm{{DC}}}}$ = $\rho({b0_norm[1:-1]}) - \rho({b1[1:-1]})$")
    plt.title(rf"{b0_norm} vs {b1} probe delta ({species_display_name(species)})")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_delta_ci_3panel(
    delta_df: pd.DataFrame,
    results: pd.DataFrame,
    species: str,
    out_png: Path,
) -> None:
    b0_norm = mathfrak_b_norm_label(0)
    b1 = mathfrak_b_label(1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    if not _plot_delta_ci_3panel_column(axes, delta_df, results, species):
        plt.close(fig)
        return

    fig.suptitle(rf"{b0_norm} vs {b1} probe delta ({species_display_name(species)})")
    fig.text(
        0.5,
        0.01,
        "Panel A: bootstrap CI mean across seeds; panels B/C: mean $\\pm$ std of per-seed $\\Delta$",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_ci_3panel_combined(
    delta_df: pd.DataFrame,
    results: pd.DataFrame,
    out_png: Path,
    species_order: Sequence[str] = ("e_coli", "s_aureus"),
) -> None:
    b0_norm = mathfrak_b_norm_label(0)
    b1 = mathfrak_b_label(1)
    row_labels = ("(A)", "(B)", "(C)")

    fig, axes = plt.subplots(3, len(species_order), figsize=(14, 11), sharey="row", sharex="col")
    if len(species_order) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    plotted = False
    for col, species in enumerate(species_order):
        col_axes = [axes[row, col] for row in range(3)]
        if _plot_delta_ci_3panel_column(
            col_axes,
            delta_df,
            results,
            species,
            show_ylabel=(col == 0),
            show_xticks=True,
        ):
            plotted = True
        col_axes[0].set_title(species_display_name(species))
        if col > 0:
            for row in range(3):
                axes[row, col].tick_params(labelleft=False)

    if not plotted:
        plt.close(fig)
        return

    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.16,
            1.0,
            label,
            transform=axes[row, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="right",
            clip_on=False,
        )

    fig.suptitle(rf"{b0_norm} vs {b1} probe delta")
    fig.text(
        0.5,
        0.01,
        "Row A: bootstrap CI mean across seeds; rows B/C: mean $\\pm$ std of per-seed $\\Delta$",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--property-table", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--species", nargs="*", default=["e_coli", "s_aureus"])
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    property_df = pd.read_csv(args.property_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_frames = []
    delta_frames = []
    for species in args.species:
        for seed in seeds:
            feature_path = args.feature_dir / f"{species}_seed{seed}.npz"
            if not feature_path.is_file():
                print(f"[SKIP] missing feature file: {feature_path}")
                continue
            probe_df, delta_df = analyze_seed(feature_path, property_df, species, seed)
            result_frames.append(probe_df)
            delta_frames.append(delta_df)

    if not result_frames:
        print("No probe results generated.")
        return 1

    results = pd.concat(result_frames, ignore_index=True)
    delta_results = pd.concat(delta_frames, ignore_index=True)
    results.to_csv(args.output_dir / "dc_property_probe_results.csv", index=False)
    build_probe_results_summary(results).to_csv(
        args.output_dir / "dc_property_probe_results_summary.csv",
        index=False,
    )
    delta_results.to_csv(args.output_dir / "dc_c0_vs_c1_delta_ci.csv", index=False)

    preference = build_dc_preference_summary(results)
    preference.to_csv(args.output_dir / "dc_preference_summary.csv", index=False)
    consensus = build_dc_preference_consensus(preference)
    consensus.to_csv(args.output_dir / "dc_preference_consensus.csv", index=False)
    delta_consensus = build_delta_consensus(delta_results)
    delta_consensus.to_csv(args.output_dir / "dc_c0_vs_c1_delta_consensus.csv", index=False)

    for species in args.species:
        plot_property_encoding_heatmap(
            results,
            species,
            args.output_dir / f"dc_property_encoding_{species}.png",
        )
        plot_delta_ci(
            delta_results,
            species,
            args.output_dir / f"dc_c0_minus_c1_delta_{species}.png",
        )
        plot_delta_ci_3panel(
            delta_results,
            results,
            species,
            args.output_dir / f"dc_c0_minus_c1_delta_3panel_{species}.png",
        )

    plot_delta_ci_3panel_combined(
        delta_results,
        results,
        args.output_dir / "dc_c0_minus_c1_delta_3panel_combined.png",
        species_order=tuple(args.species),
    )

    print(f"Wrote probe results -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
