#!/usr/bin/env python3
# maintained by kewei li
"""
Aggregate Exp3 full-test token knockout CSVs across train seeds.

Expected raw layout:
  {analysis_root}/exp3_fulltest/{pooling}/seed_{k}/{dataset}/knockout_lastlayer_HS.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

_KO_CSV = "knockout_lastlayer_HS.csv"
_SEED_DIR = re.compile(r"^seed_(\d+)$")
DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_EXP3_SUBDIR = "exp3_fulltest"

POOLING_ORDER = [
    "mean",
    "max",
    "attn",
    "last",
    "swe_ot",
    "mltp",
    "latent_attn",
    "fft_latent_attn_gate",
]


def expected_test_size(repo_root: Path, dataset: str, diff: int, threshold: float) -> int:
    test_csv = (
        repo_root
        / "data/blosum62 average"
        / f"diff_{diff}-trd_{threshold}"
        / f"grampa_{dataset}_7_25-test.csv"
    )
    if not test_csv.is_file():
        return 0
    return len(pd.read_csv(test_csv))


def discover_poolings(exp3_root: Path, poolings: Optional[Sequence[str]] = None) -> list[str]:
    if not exp3_root.is_dir():
        return []
    found = sorted(
        d.name for d in exp3_root.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    if poolings:
        want = set(poolings)
        found = [p for p in found if p in want]
    order = {p: i for i, p in enumerate(POOLING_ORDER)}
    return sorted(found, key=lambda p: (order.get(p, 999), p))


def gather_raw_csvs(
    exp3_root: Path,
    datasets: Iterable[str],
    poolings: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    ds_set = set(datasets)
    for pooling in discover_poolings(exp3_root, poolings):
        pool_dir = exp3_root / pooling
        for seed_dir in sorted(pool_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = _SEED_DIR.match(seed_dir.name)
            if not m:
                continue
            train_seed = int(m.group(1))
            for ds_dir in sorted(seed_dir.iterdir()):
                if not ds_dir.is_dir() or ds_dir.name not in ds_set:
                    continue
                csv_path = ds_dir / _KO_CSV
                if not csv_path.is_file():
                    continue
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                df = df.copy()
                df["pooling"] = pooling
                df["dataset"] = ds_dir.name
                df["train_seed"] = train_seed
                df["source_csv"] = str(csv_path)
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_seed_mean(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean abs_delta across train seeds for each (dataset, pooling, idx, token_pos)."""
    if raw.empty:
        return raw
    key = ["dataset", "pooling", "idx", "token_pos"]
    for col in ("peptide", "rel_pos", "seq_len", "layer", "model_version"):
        if col in raw.columns:
            key.append(col)

    per_seed = raw.groupby(key + ["train_seed"], as_index=False)["abs_delta"].mean()
    agg = per_seed.groupby(key, as_index=False).agg(
        abs_delta=("abs_delta", "mean"),
        std_across_seeds=(
            "abs_delta",
            lambda s: float(np.std(s.astype(float), ddof=1)) if len(s) > 1 else 0.0,
        ),
        n_seeds=("abs_delta", "count"),
    )
    return agg


def aggregate_per_peptide_response_std(position_long: pd.DataFrame) -> pd.DataFrame:
    """Std of seed-mean abs_delta across token positions, one row per peptide."""
    if position_long.empty:
        return position_long

    def _position_std(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        return float(np.std(s.astype(float), ddof=1))

    group_cols = ["dataset", "pooling", "idx"]
    agg_spec: dict[str, tuple] = {
        "response_std": ("abs_delta", _position_std),
        "n_positions": ("abs_delta", "count"),
        "n_seeds": ("n_seeds", "first"),
    }
    if "peptide" in position_long.columns:
        agg_spec["peptide"] = ("peptide", "first")

    out = position_long.groupby(group_cols, as_index=False).agg(**agg_spec)
    return out


def summarize_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    value_col = "response_std" if "response_std" in long_df.columns else "abs_delta"
    rows = []
    for (dataset, pooling), grp in long_df.groupby(["dataset", "pooling"], sort=False):
        vals = grp[value_col].astype(float)
        mean_v = float(vals.mean())
        rows.append(
            {
                "dataset": dataset,
                "pooling": pooling,
                "n_obs": int(len(vals)),
                "n_peptides": int(grp["idx"].nunique()),
                "median": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "mean": mean_v,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "cv": float(vals.std(ddof=1) / mean_v) if len(vals) > 1 and mean_v > 0 else 0.0,
                "n_seeds_mean": float(grp["n_seeds"].mean()) if "n_seeds" in grp.columns else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(POOLING_ORDER)}
    out["_ord"] = out["pooling"].map(lambda p: order.get(p, 999))
    return out.sort_values(["dataset", "_ord", "pooling"]).drop(columns=["_ord"]).reset_index(drop=True)


def build_readiness(
    exp3_root: Path,
    datasets: Iterable[str],
    poolings: Optional[Sequence[str]],
    expected_by_dataset: dict[str, int],
) -> pd.DataFrame:
    rows = []
    ds_list = list(datasets)
    for pooling in discover_poolings(exp3_root, poolings):
        for ds in ds_list:
            expected_n = expected_by_dataset.get(ds, 0)
            seed_dirs = sorted((exp3_root / pooling).glob("seed_*"))
            for seed_dir in seed_dirs:
                if not seed_dir.is_dir():
                    continue
                m = _SEED_DIR.match(seed_dir.name)
                if not m:
                    continue
                seed = int(m.group(1))
                csv_path = seed_dir / ds / _KO_CSV
                n_idx = 0
                n_rows = 0
                complete = False
                if csv_path.is_file():
                    df = pd.read_csv(csv_path)
                    n_rows = len(df)
                    n_idx = int(df["idx"].nunique()) if "idx" in df.columns else 0
                    complete = expected_n > 0 and n_idx >= expected_n
                rows.append(
                    {
                        "pooling": pooling,
                        "dataset": ds,
                        "train_seed": seed,
                        "expected_n_idx": expected_n,
                        "n_idx": n_idx,
                        "n_rows": n_rows,
                        "complete": complete,
                        "source_csv": str(csv_path) if csv_path.is_file() else "",
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(POOLING_ORDER)}
    out["_ord"] = out["pooling"].map(lambda p: order.get(p, 999))
    return out.sort_values(["pooling", "dataset", "train_seed"]).drop(columns=["_ord"]).reset_index(drop=True)


def process_dataset(
    analysis_root: Path,
    dataset: str,
    exp3_root: Path,
    *,
    poolings: Optional[Sequence[str]] = None,
    min_seeds: int = 1,
    expected_n: int = 0,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[bool, dict[str, object], Optional[pd.DataFrame]]:
    out_dir = exp_agg_dir(analysis_root, dataset, "exp3")
    long_csv = out_dir / "token_knockout_abs_delta_long.csv"
    summary_csv = out_dir / "token_knockout_summary.csv"
    readiness_csv = out_dir / "token_knockout_readiness.csv"

    if long_csv.is_file() and summary_csv.is_file() and not force:
        long_df = pd.read_csv(long_csv)
        return True, {"skip": 1, "n_obs": len(long_df)}, long_df

    raw = gather_raw_csvs(exp3_root, [dataset], poolings)
    if raw.empty:
        return False, {"error": "no_raw"}, None

    position_long = aggregate_seed_mean(raw)
    position_long = position_long[position_long["n_seeds"] >= min_seeds].copy()
    if position_long.empty:
        return False, {"error": "empty_after_min_seeds"}, None

    long_df = aggregate_per_peptide_response_std(position_long)
    if long_df.empty:
        return False, {"error": "empty_per_peptide_std"}, None

    summary_df = summarize_distribution(long_df)
    readiness_df = build_readiness(
        exp3_root,
        [dataset],
        poolings,
        {dataset: expected_n},
    )

    stats: dict[str, object] = {
        "n_obs": len(long_df),
        "n_peptides": int(long_df["idx"].nunique()),
        "n_poolings": int(long_df["pooling"].nunique()),
        "expected_n": expected_n,
    }

    if dry_run:
        return True, stats, long_df

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    readiness_df.to_csv(readiness_csv, index=False)
    print(f"[saved] {long_csv} ({len(long_df)} rows)")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {readiness_csv}")
    stats["out_dir"] = str(out_dir)
    return True, stats, long_df


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    *,
    exp3_subdir: str = DEFAULT_EXP3_SUBDIR,
    poolings: Optional[Sequence[str]] = None,
    min_seeds: int = 1,
    diff: int = 5,
    threshold: float = 0.9,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    exp3_root = analysis_root / exp3_subdir
    ok_all = True
    long_by_ds: dict[str, pd.DataFrame] = {}

    for dataset in datasets:
        expected_n = expected_test_size(REPO_ROOT, dataset, diff, threshold)
        ok, stats, long_df = process_dataset(
            analysis_root,
            dataset,
            exp3_root,
            poolings=poolings,
            min_seeds=min_seeds,
            expected_n=expected_n,
            force=force,
            dry_run=dry_run,
        )
        ok_all = ok_all and ok
        if long_df is not None and not long_df.empty:
            long_by_ds[dataset] = long_df
        print(
            f"[{dataset}] ok={ok} n_obs={stats.get('n_obs', 0)} "
            f"poolings={stats.get('n_poolings', 0)} expected={stats.get('expected_n', 0)}"
        )

    # Global readiness (all datasets) under aggregated/exp3/
    if not dry_run:
        agg_exp3 = analysis_root / "aggregated" / "exp3"
        agg_exp3.mkdir(parents=True, exist_ok=True)
        expected = {ds: expected_test_size(REPO_ROOT, ds, diff, threshold) for ds in datasets}
        readiness_all = build_readiness(exp3_root, datasets, poolings, expected)
        if not readiness_all.empty:
            path = agg_exp3 / "token_knockout_readiness_all.csv"
            readiness_all.to_csv(path, index=False)
            print(f"[saved] {path}")

    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--poolings", nargs="*", default=None)
    ap.add_argument("--exp3-subdir", default=DEFAULT_EXP3_SUBDIR)
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--diff", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    return run_all(
        root,
        args.datasets,
        exp3_subdir=args.exp3_subdir,
        poolings=args.poolings,
        min_seeds=args.min_seeds,
        diff=args.diff,
        threshold=args.threshold,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
