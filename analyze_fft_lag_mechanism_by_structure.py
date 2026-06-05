#!/usr/bin/env python3
"""Structure-bucket secondary analysis for FFT-LAG mechanism experiments."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

_EVAL_SCRIPTS = Path(__file__).resolve().parent / "evaluation_scripts"
# #region agent log
try:
    import json as _json, time as _time
    _log_path = Path(__file__).resolve().parent / ".cursor" / "debug-016c6c.log"
    _module_path = _EVAL_SCRIPTS / "fftlag_aggregated_paths.py"
    _payload = {
        "sessionId": "016c6c",
        "runId": "post-fix",
        "hypothesisId": "H1",
        "location": "analyze_fft_lag_mechanism_by_structure.py:import",
        "message": "sys.path bootstrap for fftlag_aggregated_paths",
        "data": {
            "eval_scripts_dir": str(_EVAL_SCRIPTS),
            "module_file_exists": _module_path.is_file(),
            "sys_path_head": sys.path[:3],
        },
        "timestamp": int(_time.time() * 1000),
    }
    with _log_path.open("a", encoding="utf-8") as _lf:
        _lf.write(_json.dumps(_payload) + "\n")
except Exception:
    pass
# #endregion
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import resolve_aggregated_csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CF_HELIX = set("EALMQKRH")

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


def helix_propensity(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in CF_HELIX) / len(seq)


def hydrophobic_moment(seq: str, delta: float = 100.0) -> float:
    if not seq:
        return 0.0
    angles = [math.radians(delta * i) for i in range(len(seq))]
    hx = hy = 0.0
    for a, ang in zip(seq, angles):
        h = KYTE_DOOLITTLE.get(a, 0.0)
        hx += h * math.cos(ang)
        hy += h * math.sin(ang)
    return math.sqrt(hx * hx + hy * hy) / len(seq)


def assign_buckets(scores: pd.Series) -> pd.Series:
    n = len(scores)
    if n == 0:
        return pd.Series(dtype=str)
    order = scores.sort_values().index.tolist()
    n_low = max(1, int(round(n * 0.3)))
    n_high = max(1, int(round(n * 0.3)))
    n_mid = max(0, n - n_low - n_high)
    buckets: Dict[int, str] = {}
    for i, idx in enumerate(order):
        if i < n_low:
            buckets[idx] = "bottom_30"
        elif i < n_low + n_mid:
            buckets[idx] = "middle_40"
        else:
            buckets[idx] = "top_30"
    return scores.index.to_series().map(buckets)


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


def analyze_band_sensitivity(proxy_df: pd.DataFrame, exp1_csv: Path, out_dir: Path) -> None:
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
    sample_level = (
        merged.groupby(["idx", "structure_bucket"], as_index=False)
        .agg(mean_abs_mse_diff=(mse_col, lambda s: float(np.mean(np.abs(s)))))
    )
    band_summary = (
        sample_level.groupby("structure_bucket", as_index=False)
        .agg(max_band_mse_diff=("mean_abs_mse_diff", "mean"))
    )
    _plot_bucket_bar(
        band_summary,
        "max_band_mse_diff",
        out_dir / "bucketwise_band_sensitivity.png",
        "Mean |MSE diff| by structure bucket",
        "Mean abs MSE diff (avg over layers/bands)",
    )


def analyze_gate_effect(proxy_df: pd.DataFrame, exp2_csv: Path, out_dir: Path) -> None:
    if not exp2_csv.is_file():
        return
    df = pd.read_csv(exp2_csv)
    if "idx" not in df.columns:
        return
    merged = df.merge(proxy_df, on="idx", how="inner")
    if merged.empty:
        return
    agg = (
        merged.groupby(["structure_bucket", "band"], as_index=False)
        .agg(
            effective_weight_mean=("effective_weight_mean" if "effective_weight_mean" in merged.columns else "effective_weight", "mean"),
            energy_ratio=("energy_after_mean" if "energy_after_mean" in merged.columns else "energy_after", "mean"),
            n_samples=("idx", "nunique"),
        )
    )
    agg.to_csv(out_dir / "bucketwise_gate_effect.csv", index=False)
    gate_summary = (
        merged.groupby("structure_bucket", as_index=False)
        .agg(
            gate_spread=(
                "effective_weight_mean" if "effective_weight_mean" in merged.columns else "effective_weight",
                lambda s: float(s.max() - s.min()),
            )
        )
    )
    _plot_bucket_bar(
        gate_summary,
        "gate_spread",
        out_dir / "bucketwise_gate_effect.png",
        "Gate effective-weight spread by structure bucket",
        "max - min effective_weight across bands",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--aggregated-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"Exp5 outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
