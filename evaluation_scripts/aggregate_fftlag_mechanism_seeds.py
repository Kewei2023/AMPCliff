#!/usr/bin/env python3
"""Aggregate FFT-LAG mechanism experiment outputs across train seeds."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _train_seed_from_dir(dataset_run_dir: Path) -> int:
    """Parse train seed from .../seed_{k}/{dataset}/ path."""
    parent_name = dataset_run_dir.parent.name
    if not parent_name.startswith("seed_"):
        raise ValueError(f"Expected parent seed_* dir, got {dataset_run_dir}")
    return int(parent_name.replace("seed_", ""))


def _find_exp1_long_csv(exp_dir: Path) -> Optional[Path]:
    matches = sorted(exp_dir.glob("*-mse_diff-layer_multi-band_long.csv"))
    if matches:
        return matches[0]
    fallback = exp_dir / "per_sample_band_sensitivity.csv"
    return fallback if fallback.is_file() else None


def aggregate_exp1(seed_dirs: list[Path], out_dir: Path, exp1_subdir: str = "exp1_band_knockout") -> None:
    frames = []
    ps_frames = []
    for sd in seed_dirs:
        exp_dir = sd / exp1_subdir
        if not exp_dir.is_dir():
            continue
        seed = _train_seed_from_dir(sd)
        long_csv = _find_exp1_long_csv(exp_dir)
        if long_csv and long_csv.name.endswith("_long.csv"):
            df = pd.read_csv(long_csv)
            df["seed"] = int(seed)
            frames.append(df)
        ps = exp_dir / "per_sample_band_sensitivity.csv"
        if ps.is_file():
            pdf = pd.read_csv(ps)
            pdf["seed"] = int(seed)
            ps_frames.append(pdf)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        agg = (
            all_df.groupby(["layer", "band", "split"], as_index=False)
            .agg(mse_diff_mean=("mse_diff", "mean"), mse_diff_std=("mse_diff", "std"), n_seeds=("seed", "nunique"))
        )
        agg.to_csv(out_dir / "mse_diff_layer_band_aggregated.csv", index=False)

    if ps_frames:
        ps_all = pd.concat(ps_frames, ignore_index=True)
        ps_agg = (
            ps_all.groupby(["idx", "layer", "band", "split"], as_index=False)
            .agg(
                mse_diff_mean=("mse_diff", "mean"),
                mse_diff_std=("mse_diff", "std"),
                mse_base_mean=("mse_base", "mean"),
                mse_with_hook_mean=("mse_with_hook", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        ps_agg.to_csv(out_dir / "per_sample_band_sensitivity_aggregated.csv", index=False)


def aggregate_exp2(seed_dirs: list[Path], out_dir: Path, exp2_subdir: str = "exp2_psd_gate") -> None:
    band_frames = []
    ps_frames = []
    for sd in seed_dirs:
        exp_dir = sd / exp2_subdir
        if not exp_dir.is_dir():
            continue
        seed = _train_seed_from_dir(sd)
        band_csv = exp_dir / "gate_weight_by_band.csv"
        if band_csv.is_file():
            df = pd.read_csv(band_csv)
            df["seed"] = int(seed)
            band_frames.append(df)
        ps_csv = exp_dir / "per_sample_gate_by_band.csv"
        if ps_csv.is_file():
            pdf = pd.read_csv(ps_csv)
            pdf["seed"] = int(seed)
            ps_frames.append(pdf)

    if band_frames:
        all_df = pd.concat(band_frames, ignore_index=True)
        agg = (
            all_df.groupby(["band", "band_start", "band_end"], as_index=False)
            .agg(
                mean_effective_weight=("mean_effective_weight", "mean"),
                mean_effective_weight_std=("mean_effective_weight", "std"),
                energy_before=("energy_before", "mean"),
                energy_after=("energy_after", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        agg.to_csv(out_dir / "gate_weight_by_band_aggregated.csv", index=False)

    if ps_frames:
        ps_all = pd.concat(ps_frames, ignore_index=True)
        ps_agg = (
            ps_all.groupby(["idx", "band", "band_start", "band_end"], as_index=False)
            .agg(
                effective_weight_mean=("effective_weight", "mean"),
                effective_weight_std=("effective_weight", "std"),
                energy_before_mean=("energy_before", "mean"),
                energy_after_mean=("energy_after", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        ps_agg.to_csv(out_dir / "per_sample_gate_by_band_aggregated.csv", index=False)


def _mean_std_agg_dict(cols: list[str]) -> dict:
    out: dict = {}
    for col in cols:
        out[f"{col}_mean"] = (col, "mean")
        out[f"{col}_std"] = (col, "std")
    return out


def _aggregate_exp4_metric_csv(
    seed_dirs: list[Path],
    out_dir: Path,
    filename: str,
    out_name: str,
    group_cols: list[str],
    value_cols: list[str] | str,
    *,
    exp4_subdir: str = "exp4_latent",
    precompute_diff: bool = False,
) -> None:
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    frames = []
    for sd in seed_dirs:
        exp_dir = sd / exp4_subdir
        path = exp_dir / filename
        if not path.is_file():
            continue
        seed = _train_seed_from_dir(sd)
        df = pd.read_csv(path)
        if precompute_diff:
            df = df.copy()
            df["profile_diff"] = df["latent_pooled_readout"] - df["freq_uniform_pool_energy"]
        df["seed"] = int(seed)
        frames.append(df)
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    agg = all_df.groupby(group_cols, as_index=False).agg(
        **_mean_std_agg_dict(value_cols),
        n_seeds=("seed", "nunique"),
    )
    agg.to_csv(out_dir / out_name, index=False)


def aggregate_exp4(seed_dirs: list[Path], out_dir: Path, exp4_subdir: str = "exp4_latent") -> None:
    div_frames = []
    gate_frames = []
    for sd in seed_dirs:
        exp_dir = sd / exp4_subdir
        if not exp_dir.is_dir():
            continue
        seed = _train_seed_from_dir(sd)
        div_csv = exp_dir / "latent_query_diversity.csv"
        if div_csv.is_file():
            ddf = pd.read_csv(div_csv)
            ddf["seed"] = int(seed)
            div_frames.append(ddf)
        gate_csv = exp_dir / "latent_gate_input_stats.csv"
        if gate_csv.is_file():
            gdf = pd.read_csv(gate_csv)
            gdf["seed"] = int(seed)
            gate_frames.append(gdf)

    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_query_band_mass.csv",
        "latent_query_band_mass_aggregated.csv",
        ["idx", "query", "band", "band_start", "band_end"],
        "attention_mass",
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_attn_band_deviation.csv",
        "latent_attn_band_deviation_aggregated.csv",
        ["idx", "query", "band", "band_start", "band_end"],
        "deviation",
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_weighted_band_readout.csv",
        "latent_weighted_band_readout_aggregated.csv",
        ["idx", "query", "band", "band_start", "band_end"],
        "weighted_mass",
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_query_contribution.csv",
        "latent_query_contribution_aggregated.csv",
        ["idx", "query"],
        ["l2_deviation", "cos_to_gate_input"],
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_gate_input_band_profile.csv",
        "latent_gate_input_band_profile_latent_aggregated.csv",
        ["idx", "band", "band_start", "band_end"],
        "latent_pooled_readout",
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_gate_input_band_profile.csv",
        "latent_gate_input_band_profile_freq_aggregated.csv",
        ["idx", "band", "band_start", "band_end"],
        "freq_uniform_pool_energy",
        exp4_subdir=exp4_subdir,
    )
    _aggregate_exp4_metric_csv(
        seed_dirs,
        out_dir,
        "latent_gate_input_band_profile.csv",
        "latent_gate_input_band_profile_diff_aggregated.csv",
        ["idx", "band", "band_start", "band_end"],
        "profile_diff",
        exp4_subdir=exp4_subdir,
        precompute_diff=True,
    )

    compare_frames = []
    for sd in seed_dirs:
        exp_dir = sd / exp4_subdir
        cmp_path = exp_dir / "latent_gate_input_compare.csv"
        if not cmp_path.is_file():
            continue
        seed = _train_seed_from_dir(sd)
        cdf = pd.read_csv(cmp_path)
        cdf["seed"] = int(seed)
        compare_frames.append(cdf)
    if compare_frames:
        cmp_all = pd.concat(compare_frames, ignore_index=True)
        cmp_num = [
            c
            for c in cmp_all.columns
            if c not in {"idx", "seed"}
            and pd.api.types.is_numeric_dtype(cmp_all[c])
            and not c.endswith("_cross_sample_std")
        ]
        cmp_agg = cmp_all.groupby("idx", as_index=False).agg(
            **_mean_std_agg_dict(cmp_num),
            n_seeds=("seed", "nunique"),
        )
        cmp_agg.to_csv(out_dir / "latent_gate_input_compare_aggregated.csv", index=False)

    if div_frames:
        div_all = pd.concat(div_frames, ignore_index=True)
        num_cols = [
            c
            for c in div_all.columns
            if c not in {"idx", "seed", "num_queries"}
            and pd.api.types.is_numeric_dtype(div_all[c])
            and not c.endswith("_cross_sample_std")
        ]
        div_agg = div_all.groupby("idx", as_index=False).agg(
            **_mean_std_agg_dict(num_cols),
            n_seeds=("seed", "nunique"),
        )
        div_agg.to_csv(out_dir / "latent_query_diversity_aggregated.csv", index=False)

    if gate_frames:
        gate_all = pd.concat(gate_frames, ignore_index=True)
        gate_num = [
            c
            for c in gate_all.columns
            if c not in {"idx", "seed"}
            and pd.api.types.is_numeric_dtype(gate_all[c])
            and not c.endswith("_cross_sample_std")
        ]
        gate_agg = gate_all.groupby("idx", as_index=False).agg(
            **_mean_std_agg_dict(gate_num),
            n_seeds=("seed", "nunique"),
        )
        gate_agg.to_csv(out_dir / "latent_gate_input_stats_aggregated.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/analysis/fftlag_mechanism"),
    )
    ap.add_argument("--dataset", required=True, help="e_coli or s_aureus")
    ap.add_argument(
        "--exp1-subdir",
        default="exp1_band_knockout",
        help="Per-seed Exp1 output subdirectory under seed_*/{dataset}/",
    )
    ap.add_argument(
        "--exp2-subdir",
        default="exp2_psd_gate",
        help="Per-seed Exp2 output subdirectory under seed_*/{dataset}/",
    )
    ap.add_argument(
        "--exp4-subdir",
        default="exp4_latent",
        help="Per-seed Exp4 output subdirectory under seed_*/{dataset}/",
    )
    ap.add_argument(
        "--aggregated-subdir",
        default="aggregated",
        help="Top-level aggregated output folder under analysis-root/",
    )
    ap.add_argument(
        "--exp-only",
        choices=["1", "2", "4", "all"],
        default="all",
        help="Which experiments to aggregate (default: all)",
    )
    args = ap.parse_args()

    root = args.analysis_root
    dataset_seed_dirs = []
    for sd in sorted(root.glob("seed_*")):
        ds_dir = sd / args.dataset
        if ds_dir.is_dir():
            dataset_seed_dirs.append(ds_dir)

    if not dataset_seed_dirs:
        print(f"No seed dirs found under {root} for dataset {args.dataset}")
        return 1

    base_out = root / args.aggregated_subdir / args.dataset
    exp1_out = base_out / "exp1"
    exp2_out = base_out / "exp2"
    exp4_out = base_out / "exp4"
    for d in (exp1_out, exp2_out, exp4_out):
        d.mkdir(parents=True, exist_ok=True)

    if args.exp_only in ("1", "all"):
        aggregate_exp1(dataset_seed_dirs, exp1_out, exp1_subdir=args.exp1_subdir)
    if args.exp_only in ("2", "all"):
        aggregate_exp2(dataset_seed_dirs, exp2_out, exp2_subdir=args.exp2_subdir)
    if args.exp_only in ("4", "all"):
        aggregate_exp4(dataset_seed_dirs, exp4_out, exp4_subdir=args.exp4_subdir)

    print(
        f"Aggregated outputs -> {base_out}/exp{{1,2,4}}/ "
        f"(exp1_subdir={args.exp1_subdir}, exp2_subdir={args.exp2_subdir}, exp4_subdir={args.exp4_subdir})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
