#!/usr/bin/env python3
"""Exp5 / DC validation design v2 — Step 5 / 主实验二 Part B: property-bucket figures (signed + |ΔMSE|).
Plot property-stratified DC knockout figures from intermediate CSV tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from AMPCliff.dc_property_utils import (
    band_sensitivity_task_name,
    infer_species_from_path,
    left_aligned_bucket_chart_title,
    mathfrak_b_label,
)
from analyze_fft_lag_mechanism_by_structure import (
    _BUCKET_ORDER_RAW,
    _plot_bucket_band_combined,
    _property_bucket_labels,
    _property_bucket_order,
    _property_bucket_xlabel,
)


def _load_required(intermediate_dir: Path, property_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_level_path = intermediate_dir / f"per_sample_band_sensitivity_by_{property_col}.csv"
    by_band_path = intermediate_dir / f"bucketwise_band_sensitivity_by_band_{property_col}.csv"
    summary_path = intermediate_dir / f"bucketwise_band_sensitivity_summary_{property_col}.csv"
    for p in (sample_level_path, by_band_path, summary_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing intermediate table: {p}")

    sample_band = pd.read_csv(sample_level_path)
    by_band = pd.read_csv(by_band_path)
    summary = pd.read_csv(summary_path)

    sample_level = (
        sample_band.groupby(["idx", "property_bucket"], as_index=False)
        .agg(mean_abs_mse_diff=("mse_diff_abs_mean", "mean"))
    )
    return sample_level, by_band, summary


def plot_bucket_combined(
    intermediate_dir: Path,
    figures_dir: Path,
    property_col: str,
) -> Path:
    sample_level, by_band, _ = _load_required(intermediate_dir, property_col)
    label_map = _property_bucket_labels(property_col)
    bucket_order = _property_bucket_order(property_col)
    xlabel = _property_bucket_xlabel(property_col)
    out_png = figures_dir / f"{property_col}_bucketwise_band_sensitivity_combined.png"
    _plot_bucket_band_combined(
        sample_level,
        by_band,
        bar_y_col="mean_abs_mse_diff",
        line_y_col="mse_diff_abs_mean",
        out_png=out_png,
        title=left_aligned_bucket_chart_title(
            infer_species_from_path(intermediate_dir),
            band_sensitivity_task_name(xlabel),
        ),
        ylabel="Mean |MSE diff|",
        bucket_col="property_bucket",
        bucket_order=bucket_order,
        xlabel=xlabel,
        label_map=label_map,
    )
    return out_png


def plot_signed_dc_knockout(
    intermediate_dir: Path,
    figures_dir: Path,
    property_col: str,
    bands: Sequence[int] = (0, 1),
) -> Path | None:
    ko_path = intermediate_dir / f"property_dc_knockout_{property_col}.csv"
    if not ko_path.is_file():
        return None
    df = pd.read_csv(ko_path)
    df = df[df["band"].isin(list(bands))].copy()
    if df.empty:
        return None

    label_map = _property_bucket_labels(property_col)
    bucket_order = _property_bucket_order(property_col)
    df["bucket_label"] = df["property_bucket"].map(label_map)
    df["bucket_label"] = pd.Categorical(df["bucket_label"], categories=[label_map[k] for k in _BUCKET_ORDER_RAW if k in label_map], ordered=True)
    df["band_label"] = df["band"].map(lambda b: mathfrak_b_label(int(b)))

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="bucket_label",
        y="signed_mse_diff",
        hue="band_label",
        order=bucket_order,
        errorbar="sd",
    )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(_property_bucket_xlabel(property_col))
    plt.ylabel("Mean signed MSE diff")
    plt.title(
        rf"Signed MSE diff by {property_col} bucket "
        rf"({', '.join(mathfrak_b_label(b) for b in bands)})"
    )
    plt.tight_layout()
    out_png = figures_dir / f"{property_col}_dc_knockout.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png


def plot_b0_vs_b1(
    intermediate_dir: Path,
    figures_dir: Path,
    properties: Sequence[str],
) -> Path | None:
    sens_path = intermediate_dir / "property_dc_knockout_sensitivity.csv"
    if not sens_path.is_file():
        return None
    df = pd.read_csv(sens_path)
    df = df[df["band"].isin([0, 1]) & df["property"].isin(list(properties))].copy()
    if df.empty:
        return None

    rows = []
    for prop in properties:
        sub = df[df["property"] == prop]
        for band in (0, 1):
            bsub = sub[sub["band"] == band]
            if bsub.empty:
                continue
            rows.append(
                {
                    "property": prop.replace("_", " "),
                    "band": mathfrak_b_label(band),
                    "signed_mse_diff": float(bsub["signed_mse_diff"].mean()),
                    "abs_mse_diff": float(bsub["abs_mse_diff"].mean()),
                }
            )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=plot_df, x="property", y="signed_mse_diff", hue="band", ax=axes[0])
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(r"Mean signed MSE diff ($\mathfrak{B}_0$ vs $\mathfrak{B}_1$)")
    axes[0].set_ylabel("signed MSE diff")
    sns.barplot(data=plot_df, x="property", y="abs_mse_diff", hue="band", ax=axes[1])
    axes[1].set_title(r"Mean |MSE diff| ($\mathfrak{B}_0$ vs $\mathfrak{B}_1$)")
    axes[1].set_ylabel("|MSE diff|")
    fig.suptitle("Property-stratified band sensitivity (pooled across buckets)")
    fig.tight_layout()
    out_png = figures_dir / "b0_vs_b1_property_sensitivity.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intermediate-dir", type=Path, required=True)
    ap.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Default: parent of intermediate-dir / figures",
    )
    ap.add_argument(
        "--properties",
        nargs="*",
        default=["net_charge", "mean_hydrophobicity"],
    )
    ap.add_argument("--bands", nargs="*", type=int, default=[0, 1])
    args = ap.parse_args()

    figures_dir = args.figures_dir or (args.intermediate_dir.parent / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for prop in args.properties:
        try:
            written.append(plot_bucket_combined(args.intermediate_dir, figures_dir, prop))
            signed_png = plot_signed_dc_knockout(
                args.intermediate_dir, figures_dir, prop, bands=args.bands
            )
            if signed_png:
                written.append(signed_png)
        except FileNotFoundError as exc:
            print(f"[WARN] skip {prop}: {exc}")

    b0_png = plot_b0_vs_b1(args.intermediate_dir, figures_dir, args.properties)
    if b0_png:
        written.append(b0_png)

    print(f"Wrote {len(written)} figures -> {figures_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
