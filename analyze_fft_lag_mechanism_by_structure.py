#!/usr/bin/env python3
# maintained by kewei li
"""Structure-bucket secondary analysis for FFT-LAG mechanism experiments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

_EVAL_SCRIPTS = Path(__file__).resolve().parent / "evaluation_scripts"
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import resolve_aggregated_csv

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from spectrual_filter.filter_seq import allocate_prism_bands
from dc_property_utils import (
    BUCKET_LABELS,
    GATE_EFFECT_TASK_NAME,
    KYTE_DOOLITTLE,
    assign_property_buckets as assign_buckets,
    assign_property_buckets_by_species,
    band_sensitivity_task_name,
    helix_propensity,
    hydrophobic_moment,
    infer_species_from_path,
    left_aligned_bucket_chart_title,
    load_knockout_property_proxy_df,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CF_HELIX = set("EALMQKRH")  # noqa: F401 - kept for backward compatibility in docs

DCT_BAND_MAPPING_CAPTION_EN = (
    "Definition of sequence-frequency bands used in Experiment 1 (band knockout) and "
    "stratified in Experiment 5. For each peptide of effective length L, hidden states "
    "at every ESM-2 layer are transformed along the token (sequence) axis by an "
    "orthonormal DCT, yielding L discrete cosine coefficients indexed 0..L-1 "
    "(index 0: lowest sequence frequency; index L-1: highest). The k=8 bands partition "
    "these indices into eight contiguous, non-overlapping intervals using geometric "
    "allocation with base 4: each band receives at least one coefficient, and any "
    "remaining coefficients are assigned in proportion to 4^i for band i. Band 0 captures "
    "the lowest-frequency (slowest along-sequence) components and Band 7 the highest-frequency "
    "(most local) components; band boundaries depend on L (e.g. at L=25, Band 7 spans DCT "
    "indices [11,25) and contains 14 coefficients). Band knockout applies a notch mask on "
    "the target band's DCT coefficients, followed by inverse DCT. "
    "A split requires L >= k (minimum L=8 for k=8). At L=9, Bands 0--6 each span one DCT "
    "coefficient and Band 7 spans [7,9) (two coefficients); at L=25, Band 7 spans DCT "
    "indices [11,25) and contains 14 coefficients)."
)

def load_manifest(manifest_path: Path) -> pd.DataFrame:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for idx, pep in zip(payload["idx_list"], payload["peptides"]):
        rows.append({
            "idx": int(idx),
            "sequence": pep,
            "length": len(pep),
            "helix_propensity": helix_propensity(pep),
            "hydrophobic_moment": hydrophobic_moment(pep),
        })
    df = pd.DataFrame(rows)
    df["structure_bucket"] = assign_buckets(df["helix_propensity"])
    return df


_BUCKET_LABELS = BUCKET_LABELS
_BUCKET_ORDER = ["bottom 30%", "middle 40%", "top 30%"]
_BUCKET_ORDER_RAW = ["bottom_30", "middle_40", "top_30"]
_BAND_LABELS = [rf"$\mathfrak{{B}}_{{{i}}}$" for i in range(8)]


def _map_bucket_labels(
    df: pd.DataFrame,
    col: str = "structure_bucket",
    label_map: Dict[str, str] | None = None,
    bucket_order: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    label_map = label_map or _BUCKET_LABELS
    bucket_order = bucket_order or _BUCKET_ORDER
    out[col] = out[col].map(label_map)
    out[col] = pd.Categorical(out[col], bucket_order, ordered=True)
    return out


def _scale_lines_to_bar_total(
    by_band: pd.DataFrame,
    sample_level: pd.DataFrame,
    line_y_col: str,
    bar_y_col: str,
    bucket_col: str = "structure_bucket",
) -> pd.DataFrame:
    """Scale per-band values so each bucket's 8 bands sum to the bar mean."""
    bar_totals = (
        sample_level.groupby(bucket_col, as_index=False)
        .agg(bar_total=(bar_y_col, "mean"))
    )
    band_sums = (
        by_band.groupby(bucket_col, as_index=False)
        .agg(band_sum=(line_y_col, "sum"))
    )
    scaled = by_band.merge(bar_totals, on=bucket_col).merge(band_sums, on=bucket_col)
    scaled[line_y_col] = np.where(
        scaled["band_sum"] > 0,
        scaled[line_y_col] * scaled["bar_total"] / scaled["band_sum"],
        0.0,
    )
    return scaled[[bucket_col, "band", line_y_col]]


def _plot_bucket_bar(df: pd.DataFrame, y_col: str, out_png: Path, title: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    order = ["bottom_30", "middle_40", "top_30"]
    sns.barplot(data=df, x="structure_bucket", y=y_col, order=order, errorbar="sd")
    plt.xlabel("Helix propensity bucket")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_by_band_lines(
    by_band: pd.DataFrame,
    line_y_col: str,
    out_png: Path,
    title: str,
    ylabel: str,
) -> None:
    plot_df = _map_bucket_labels(by_band)
    plot_df["band_label"] = plot_df["band"].map(lambda b: rf"$\mathfrak{{B}}_{{{int(b)}}}$")

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=plot_df,
        x="structure_bucket",
        y=line_y_col,
        hue="band_label",
        hue_order=_BAND_LABELS,
        marker="o",
        errorbar=None,
    )
    plt.xlabel("Helix propensity bucket")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_band_sensitivity_lines(by_band: pd.DataFrame, out_png: Path, title: str) -> None:
    _plot_by_band_lines(
        by_band,
        "mse_diff_abs_mean",
        out_png,
        title,
        "Mean |MSE diff|",
    )


def _plot_gate_effect_lines(by_band: pd.DataFrame, out_png: Path, title: str) -> None:
    _plot_by_band_lines(
        by_band,
        "effective_weight_mean",
        out_png,
        title,
        "Mean effective weight",
    )


def _plot_bucket_band_combined(
    sample_level: pd.DataFrame,
    by_band: pd.DataFrame,
    bar_y_col: str,
    line_y_col: str,
    out_png: Path | None,
    title: str,
    ylabel: str,
    bucket_col: str = "structure_bucket",
    bucket_order: list[str] | None = None,
    xlabel: str = "Helix propensity bucket",
    label_map: Dict[str, str] | None = None,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
    bar_color: str | None = None,
    bar_edgecolor: str | None = None,
    band_palette: Dict[str, tuple] | None = None,
    line_alpha: float = 0.5,
) -> None:
    bucket_order = bucket_order or _BUCKET_ORDER
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    bar_df = _map_bucket_labels(sample_level, col=bucket_col, label_map=label_map, bucket_order=bucket_order)
    barplot_kwargs: dict = {
        "data": bar_df,
        "x": bucket_col,
        "y": bar_y_col,
        "order": bucket_order,
        "errorbar": None,
        "ax": ax,
        "zorder": 1,
    }
    if bar_color is not None:
        barplot_kwargs["color"] = bar_color
    sns.barplot(**barplot_kwargs)
    for patch in ax.patches:
        patch.set_alpha(0.5 if bar_color is None else 0.85)
        if bar_edgecolor is not None:
            patch.set_edgecolor(bar_edgecolor)
            patch.set_linewidth(0.8)

    line_df = _scale_lines_to_bar_total(
        by_band, sample_level, line_y_col, bar_y_col, bucket_col=bucket_col
    )
    line_df = _map_bucket_labels(line_df, col=bucket_col, label_map=label_map, bucket_order=bucket_order)
    line_df["band_label"] = line_df["band"].map(lambda b: rf"$\mathfrak{{B}}_{{{int(b)}}}$")
    lineplot_kwargs: dict = {
        "data": line_df,
        "x": bucket_col,
        "y": line_y_col,
        "hue": "band_label",
        "hue_order": _BAND_LABELS,
        "marker": "o",
        "markersize": 3,
        "errorbar": None,
        "ax": ax,
        "zorder": 3,
    }
    if band_palette is not None:
        lineplot_kwargs["palette"] = band_palette
    sns.lineplot(**lineplot_kwargs)
    for line in ax.lines:
        line.set_alpha(line_alpha)
        line.set_linewidth(1.1)
    for collection in ax.collections:
        collection.set_alpha(line_alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=12)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if owns_figure:
        if out_png is None:
            raise ValueError("out_png is required when ax is not provided")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_band_sensitivity_combined(
    sample_level: pd.DataFrame,
    by_band: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    _plot_bucket_band_combined(
        sample_level,
        by_band,
        bar_y_col="mean_abs_mse_diff",
        line_y_col="mse_diff_abs_mean",
        out_png=out_png,
        title=title,
        ylabel="Mean |MSE diff|",
    )


def analyze_band_sensitivity(
    proxy_df: pd.DataFrame,
    exp1_csv: Path,
    out_dir: Path,
    species: str | None = None,
) -> None:
    if not exp1_csv.is_file():
        return
    df = pd.read_csv(exp1_csv)
    if "idx" not in df.columns:
        return
    mse_col = "mse_diff_mean" if "mse_diff_mean" in df.columns else "mse_diff"
    merged = df.merge(proxy_df[["idx", "structure_bucket"]], on="idx", how="inner")
    if merged.empty:
        return
    agg = (
        merged.groupby(["structure_bucket", "band"], as_index=False)
        .agg(
            mse_diff_mean=(mse_col, "mean"),
            mse_diff_std=(mse_col, "std"),
            n_samples=("idx", "nunique"),
        )
    )
    agg.to_csv(out_dir / "bucketwise_band_sensitivity.csv", index=False)

    sample_band_level = (
        merged.groupby(["idx", "structure_bucket", "band"], as_index=False)
        .agg(
            mse_diff_abs_mean=(mse_col, lambda s: float(np.mean(np.abs(s)))),
            n_layers=(mse_col, "count"),
        )
    )
    sample_band_level.to_csv(out_dir / "per_sample_band_sensitivity_by_structure.csv", index=False)
    by_band = (
        sample_band_level.groupby(["structure_bucket", "band"], as_index=False)
        .agg(
            mse_diff_abs_mean=("mse_diff_abs_mean", "mean"),
            mse_diff_abs_std=("mse_diff_abs_mean", "std"),
            n_samples=("idx", "nunique"),
        )
    )
    by_band.to_csv(out_dir / "bucketwise_band_sensitivity_by_band.csv", index=False)
    _plot_band_sensitivity_lines(
        by_band,
        out_dir / "bucketwise_band_sensitivity_by_band_lines.png",
        "Band-wise |MSE diff| by helix propensity bucket",
    )

    sample_level = (
        merged.groupby(["idx", "structure_bucket"], as_index=False)
        .agg(mean_abs_mse_diff=(mse_col, lambda s: float(np.mean(np.abs(s)))))
    )
    bucket_summary = (
        sample_level.groupby("structure_bucket", as_index=False)
        .agg(
            mean_abs_mse_diff=("mean_abs_mse_diff", "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    bucket_summary.to_csv(out_dir / "bucketwise_band_sensitivity_summary.csv", index=False)
    _plot_bucket_bar(
        bucket_summary,
        "mean_abs_mse_diff",
        out_dir / "bucketwise_band_sensitivity.png",
        "Mean |MSE diff| by structure bucket",
        "Mean abs MSE diff (avg over layers/bands)",
    )
    _plot_band_sensitivity_combined(
        sample_level,
        by_band,
        out_dir / "bucketwise_band_sensitivity_combined.png",
        left_aligned_bucket_chart_title(
            species or infer_species_from_path(out_dir),
            band_sensitivity_task_name("Helix propensity bucket"),
        ),
    )


def analyze_gate_effect(
    proxy_df: pd.DataFrame,
    exp2_csv: Path,
    out_dir: Path,
    species: str | None = None,
) -> None:
    if not exp2_csv.is_file():
        return
    df = pd.read_csv(exp2_csv)
    if "idx" not in df.columns:
        return
    merged = df.merge(proxy_df, on="idx", how="inner")
    if merged.empty:
        return
    ew_col = "effective_weight_mean" if "effective_weight_mean" in merged.columns else "effective_weight"
    energy_col = "energy_after_mean" if "energy_after_mean" in merged.columns else "energy_after"

    agg = (
        merged.groupby(["structure_bucket", "band"], as_index=False)
        .agg(
            effective_weight_mean=(ew_col, "mean"),
            energy_ratio=(energy_col, "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    agg.to_csv(out_dir / "bucketwise_gate_effect.csv", index=False)

    sample_band_level = (
        merged.groupby(["idx", "structure_bucket", "band"], as_index=False)
        .agg(effective_weight_mean=(ew_col, "mean"))
    )
    sample_band_level.to_csv(out_dir / "per_sample_gate_effect_by_structure.csv", index=False)

    by_band = (
        sample_band_level.groupby(["structure_bucket", "band"], as_index=False)
        .agg(
            effective_weight_mean=("effective_weight_mean", "mean"),
            effective_weight_std=("effective_weight_mean", "std"),
            n_samples=("idx", "nunique"),
        )
    )
    by_band.to_csv(out_dir / "bucketwise_gate_effect_by_band.csv", index=False)
    _plot_gate_effect_lines(
        by_band,
        out_dir / "bucketwise_gate_effect_by_band_lines.png",
        "Band-wise gate effective weight by helix propensity bucket",
    )

    sample_level = (
        sample_band_level.groupby(["idx", "structure_bucket"], as_index=False)
        .agg(gate_spread=("effective_weight_mean", lambda s: float(s.max() - s.min())))
    )
    bucket_summary = (
        sample_level.groupby("structure_bucket", as_index=False)
        .agg(
            gate_spread=("gate_spread", "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    bucket_summary.to_csv(out_dir / "bucketwise_gate_effect_summary.csv", index=False)
    _plot_bucket_bar(
        bucket_summary,
        "gate_spread",
        out_dir / "bucketwise_gate_effect.png",
        "Gate effective-weight spread by structure bucket",
        "max - min effective_weight across bands",
    )
    _plot_bucket_band_combined(
        sample_level,
        by_band,
        bar_y_col="gate_spread",
        line_y_col="effective_weight_mean",
        out_png=out_dir / "bucketwise_gate_effect_combined.png",
        title=left_aligned_bucket_chart_title(
            species or infer_species_from_path(out_dir),
            GATE_EFFECT_TASK_NAME,
        ),
        ylabel="Gate effective weight / spread",
        xlabel="Helix propensity bucket",
    )


def analyze_latent_diversity(proxy_df: pd.DataFrame, exp4_div_csv: Path, out_dir: Path) -> None:
    if not exp4_div_csv.is_file():
        return
    df = pd.read_csv(exp4_div_csv)
    if "idx" not in df.columns:
        return
    merged = df.merge(proxy_df, on="idx", how="inner")
    if merged.empty:
        return
    cos_col = "mean_query_cosine_distance_mean" if "mean_query_cosine_distance_mean" in merged.columns else "mean_query_cosine_distance"
    js_col = "mean_query_js_divergence_mean" if "mean_query_js_divergence_mean" in merged.columns else "mean_query_js_divergence"
    agg = (
        merged.groupby("structure_bucket", as_index=False)
        .agg(
            mean_query_cosine_distance=(cos_col, "mean"),
            mean_query_js_divergence=(js_col, "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    agg.to_csv(out_dir / "bucketwise_latent_diversity.csv", index=False)
    _plot_bucket_bar(
        agg,
        "mean_query_cosine_distance",
        out_dir / "bucketwise_latent_diversity.png",
        "Latent query diversity by structure bucket",
        "Mean query-query cosine distance",
    )


def export_dct_band_mapping_csv(
    out_path: Path,
    example_lengths: Sequence[int] = (9, 25),
    k_bands: int = 8,
    base: int = 4,
) -> None:
    """Write Exp1 DCT band index mapping with English table caption in-row."""
    empty_row = {
        "table_caption_en": "",
        "seq_len": "",
        "band": "",
        "dct_index_range": "",
        "dct_index_start": "",
        "dct_index_end": "",
        "band_size": "",
        "k_bands": "",
        "base": "",
        "allocation_mode": "",
        "transform": "",
        "valid": "",
        "note": "",
    }
    rows = [{**empty_row, "table_caption_en": DCT_BAND_MAPPING_CAPTION_EN}]

    for idx, seq_len in enumerate(example_lengths):
        if idx > 0:
            rows.append({**empty_row, "note": f"--- L={seq_len} ---"})
        if seq_len < k_bands:
            rows.append(
                {
                    **empty_row,
                    "seq_len": int(seq_len),
                    "band": -1,
                    "k_bands": int(k_bands),
                    "base": int(base),
                    "allocation_mode": "geometric_seq_len",
                    "transform": "DCT_along_sequence",
                    "valid": False,
                    "note": f"L < k_bands={k_bands}; eight-band partition undefined",
                }
            )
            continue
        _, sizes, starts, ends = allocate_prism_bands(seq_len, k=k_bands, base=base)
        for band in range(k_bands):
            st = int(starts[band].item())
            ed = int(ends[band].item())
            rows.append(
                {
                    **empty_row,
                    "seq_len": int(seq_len),
                    "band": int(band),
                    "dct_index_range": f"[{st},{ed})",
                    "dct_index_start": st,
                    "dct_index_end": ed,
                    "band_size": int(sizes[band].item()),
                    "k_bands": int(k_bands),
                    "base": int(base),
                    "allocation_mode": "geometric_seq_len",
                    "transform": "DCT_along_sequence",
                    "valid": True,
                }
            )
    columns = [
        "table_caption_en",
        "seq_len",
        "band",
        "dct_index_range",
        "dct_index_start",
        "dct_index_end",
        "band_size",
        "k_bands",
        "base",
        "allocation_mode",
        "transform",
        "valid",
        "note",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[columns].to_csv(out_path, index=False)


def _property_bucket_labels(property_col: str) -> Dict[str, str]:
    if property_col == "net_charge":
        return {
            "bottom_30": "low charge",
            "middle_40": "middle charge",
            "top_30": "high charge",
        }
    if property_col == "mean_hydrophobicity":
        return {
            "bottom_30": "low hydrophobicity",
            "middle_40": "middle hydrophobicity",
            "top_30": "high hydrophobicity",
        }
    if property_col == "hydrophobic_moment":
        return {
            "bottom_30": "low hydrophobic moment",
            "middle_40": "middle hydrophobic moment",
            "top_30": "high hydrophobic moment",
        }
    return _BUCKET_LABELS


def _property_bucket_order(property_col: str) -> list[str]:
    labels = _property_bucket_labels(property_col)
    return [labels[k] for k in _BUCKET_ORDER_RAW if k in labels]


def _property_bucket_xlabel(property_col: str) -> str:
    if property_col == "net_charge":
        return "Net charge bucket"
    if property_col == "mean_hydrophobicity":
        return "Mean hydrophobicity bucket"
    if property_col == "hydrophobic_moment":
        return "Hydrophobic moment bucket"
    return f"{property_col} bucket"


def analyze_property_dc_sensitivity(
    proxy_df: pd.DataFrame,
    exp1_csv: Path,
    property_col: str,
    out_dir: Path,
    bands: Sequence[int] = (0, 1),
    skip_plots: bool = False,
) -> pd.DataFrame:
    if not exp1_csv.is_file():
        return pd.DataFrame()
    df = pd.read_csv(exp1_csv)
    if "idx" not in df.columns:
        return pd.DataFrame()

    mse_col = "mse_diff_mean" if "mse_diff_mean" in df.columns else "mse_diff"
    work = proxy_df.copy()
    work["property_bucket"] = assign_property_buckets_by_species(work, property_col)

    merged_all = df.merge(
        work[["idx", "species", "property_bucket", property_col]],
        on="idx",
        how="inner",
    )
    if merged_all.empty:
        return pd.DataFrame()

    label_map = _property_bucket_labels(property_col)
    bucket_order = _property_bucket_order(property_col)
    bucket_col = "property_bucket"

    sample_band_level = (
        merged_all.groupby(["idx", bucket_col, "band"], as_index=False)
        .agg(
            mse_diff_abs_mean=(mse_col, lambda s: float(np.mean(np.abs(s)))),
            n_layers=(mse_col, "count"),
        )
    )
    sample_band_level.to_csv(
        out_dir / f"per_sample_band_sensitivity_by_{property_col}.csv",
        index=False,
    )
    by_band = (
        sample_band_level.groupby([bucket_col, "band"], as_index=False)
        .agg(
            mse_diff_abs_mean=("mse_diff_abs_mean", "mean"),
            mse_diff_abs_std=("mse_diff_abs_mean", "std"),
            n_samples=("idx", "nunique"),
        )
    )
    by_band.to_csv(
        out_dir / f"bucketwise_band_sensitivity_by_band_{property_col}.csv",
        index=False,
    )

    sample_level = (
        merged_all.groupby(["idx", bucket_col], as_index=False)
        .agg(mean_abs_mse_diff=(mse_col, lambda s: float(np.mean(np.abs(s)))))
    )
    bucket_summary = (
        sample_level.groupby(bucket_col, as_index=False)
        .agg(
            mean_abs_mse_diff=("mean_abs_mse_diff", "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    bucket_summary.to_csv(
        out_dir / f"bucketwise_band_sensitivity_summary_{property_col}.csv",
        index=False,
    )
    if not skip_plots:
        xlabel = _property_bucket_xlabel(property_col)
        plot_species = None
        if "species" in proxy_df.columns and proxy_df["species"].nunique() == 1:
            plot_species = str(proxy_df["species"].iloc[0])
        _plot_bucket_band_combined(
            sample_level,
            by_band,
            bar_y_col="mean_abs_mse_diff",
            line_y_col="mse_diff_abs_mean",
            out_png=out_dir / f"{property_col}_bucketwise_band_sensitivity_combined.png",
            title=left_aligned_bucket_chart_title(
                plot_species or infer_species_from_path(out_dir),
                band_sensitivity_task_name(xlabel),
            ),
            ylabel="Mean |MSE diff|",
            bucket_col=bucket_col,
            bucket_order=bucket_order,
            xlabel=xlabel,
            label_map=label_map,
        )

    merged_bands = merged_all[merged_all["band"].isin(list(bands))]
    if merged_bands.empty:
        return pd.DataFrame()

    agg = (
        merged_bands.groupby(["species", "property_bucket", "band"], as_index=False)
        .agg(
            signed_mse_diff=(mse_col, "mean"),
            abs_mse_diff=(mse_col, lambda s: float(np.mean(np.abs(s)))),
            n=("idx", "nunique"),
        )
    )
    agg["property"] = property_col
    agg.to_csv(out_dir / f"property_dc_knockout_{property_col}.csv", index=False)
    return agg


def load_property_proxy_df(
    manifest_path: Path,
    property_table: Path,
    exp1_csv: Path | None = None,
    species: str | None = None,
) -> pd.DataFrame:
    if exp1_csv is not None and exp1_csv.is_file():
        proxy, _meta = load_knockout_property_proxy_df(
            property_table,
            exp1_csv,
            manifest_path=manifest_path,
            species=species,
        )
        return proxy

    manifest_df = load_manifest(manifest_path)[["idx", "sequence"]]
    props = pd.read_csv(property_table)
    merged = manifest_df.merge(props, on=["idx", "sequence"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between manifest and property table")
    return merged


def run_property_dc_analysis(
    manifest_path: Path,
    property_table: Path,
    aggregated_dir: Path,
    output_dir: Path,
    properties: Sequence[str],
    bands: Sequence[int],
    species: str | None = None,
) -> None:
    exp1_csv = resolve_aggregated_csv(
        aggregated_dir,
        "exp1",
        "per_sample_band_sensitivity_aggregated.csv",
        legacy_filename="exp1_per_sample_band_sensitivity_aggregated.csv",
    )
    proxy_df = load_property_proxy_df(
        manifest_path,
        property_table,
        exp1_csv=exp1_csv,
        species=species,
    )
    proxy_df.to_csv(output_dir / "property_proxy_table.csv", index=False)

    frames = {}
    for prop in properties:
        frames[prop] = analyze_property_dc_sensitivity(
            proxy_df,
            exp1_csv,
            prop,
            output_dir,
            bands=bands,
        )

    combined = []
    for prop, frame in frames.items():
        if not frame.empty:
            combined.append(frame)
    if combined:
        out = pd.concat(combined, ignore_index=True)
        out.to_csv(output_dir / "property_dc_knockout_sensitivity.csv", index=False)


def replot_structure_combined_figures(out_dir: Path, species: str | None = None) -> None:
    """Regenerate combined bucket bar/line figures from saved Exp5 CSVs."""
    species = species or infer_species_from_path(out_dir)

    band_sample_path = out_dir / "per_sample_band_sensitivity_by_structure.csv"
    band_by_band_path = out_dir / "bucketwise_band_sensitivity_by_band.csv"
    if band_sample_path.is_file() and band_by_band_path.is_file():
        sample_band = pd.read_csv(band_sample_path)
        by_band = pd.read_csv(band_by_band_path)
        sample_level = (
            sample_band.groupby(["idx", "structure_bucket"], as_index=False)
            .agg(mean_abs_mse_diff=("mse_diff_abs_mean", "mean"))
        )
        _plot_band_sensitivity_combined(
            sample_level,
            by_band,
            out_dir / "bucketwise_band_sensitivity_combined.png",
            left_aligned_bucket_chart_title(
                species,
                band_sensitivity_task_name("Helix propensity bucket"),
            ),
        )

    gate_sample_path = out_dir / "per_sample_gate_effect_by_structure.csv"
    gate_by_band_path = out_dir / "bucketwise_gate_effect_by_band.csv"
    if gate_sample_path.is_file() and gate_by_band_path.is_file():
        sample_band = pd.read_csv(gate_sample_path)
        by_band = pd.read_csv(gate_by_band_path)
        sample_level = (
            sample_band.groupby(["idx", "structure_bucket"], as_index=False)
            .agg(gate_spread=("effective_weight_mean", lambda s: float(s.max() - s.min())))
        )
        _plot_bucket_band_combined(
            sample_level,
            by_band,
            bar_y_col="gate_spread",
            line_y_col="effective_weight_mean",
            out_png=out_dir / "bucketwise_gate_effect_combined.png",
            title=left_aligned_bucket_chart_title(species, GATE_EFFECT_TASK_NAME),
            ylabel="Gate effective weight / spread",
            xlabel="Helix propensity bucket",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--aggregated-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--analysis-mode", choices=["helix", "property", "both"], default="helix")
    ap.add_argument("--property-table", type=Path, default=None)
    ap.add_argument("--properties", nargs="*", default=["net_charge", "mean_hydrophobicity"])
    ap.add_argument("--bands", nargs="*", type=int, default=[0, 1])
    ap.add_argument(
        "--species",
        type=str,
        default=None,
        help="Filter property proxy to one species (e_coli / s_aureus)",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis_mode in ("helix", "both"):
        if args.manifest is None or args.aggregated_dir is None:
            raise ValueError("helix/both modes require --manifest and --aggregated-dir")
        proxy_df = load_manifest(args.manifest)
        proxy_df.to_csv(args.output_dir / "structure_proxy_table.csv", index=False)

        analyze_band_sensitivity(
            proxy_df,
            resolve_aggregated_csv(
                args.aggregated_dir,
                "exp1",
                "per_sample_band_sensitivity_aggregated.csv",
                legacy_filename="exp1_per_sample_band_sensitivity_aggregated.csv",
            ),
            args.output_dir,
            species=args.species or infer_species_from_path(args.output_dir),
        )
        analyze_gate_effect(
            proxy_df,
            resolve_aggregated_csv(
                args.aggregated_dir,
                "exp2",
                "per_sample_gate_by_band_aggregated.csv",
                legacy_filename="exp2_per_sample_gate_by_band_aggregated.csv",
            ),
            args.output_dir,
            species=args.species or infer_species_from_path(args.output_dir),
        )
        analyze_latent_diversity(
            proxy_df,
            resolve_aggregated_csv(
                args.aggregated_dir,
                "exp4",
                "latent_query_diversity_aggregated.csv",
                legacy_filename="exp4_latent_query_diversity_aggregated.csv",
            ),
            args.output_dir,
        )

        exp5_root = args.output_dir.parent
        export_dct_band_mapping_csv(exp5_root / "exp1_dct_band_mapping.csv")

    if args.analysis_mode in ("property", "both"):
        if args.manifest is None or args.aggregated_dir is None or args.property_table is None:
            raise ValueError("property/both modes require --manifest, --aggregated-dir, --property-table")
        run_property_dc_analysis(
            args.manifest,
            args.property_table,
            args.aggregated_dir,
            args.output_dir,
            properties=args.properties,
            bands=args.bands,
            species=args.species,
        )

    print(f"Exp5 outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
