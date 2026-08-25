#!/usr/bin/env python3
"""Aggregate Exp3 full-test token knockout |ΔMSE| response std across train seeds."""
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
    "attn_structured",
    "last",
    "mltp_paper",
    "latent_attn",
    "FLaG",
]


def label_csv_path(repo_root: Path, dataset: str, diff: int, threshold: float) -> Path:
    return (
        repo_root
        / "data/blosum62 average"
        / f"diff_{diff}-trd_{threshold}"
        / f"grampa_{dataset}_7_25-test.csv"
    )


def load_labels(repo_root: Path, dataset: str, diff: int, threshold: float) -> pd.DataFrame:
    path = label_csv_path(repo_root, dataset, diff, threshold)
    if not path.is_file():
        raise FileNotFoundError(f"Missing label CSV: {path}")
    labels = pd.read_csv(path)
    labels = labels.rename(columns={"Idx": "idx"})
    if "idx" not in labels.columns:
        raise ValueError(f"Label CSV missing Idx column: {path}")
    if "value" not in labels.columns:
        raise ValueError(f"Label CSV missing value column: {path}")
    return labels[["idx", "value"]].rename(columns={"value": "y"})


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
    labels_by_dataset: dict[str, pd.DataFrame],
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
                if "pred_base" not in df.columns or "pred_ko" not in df.columns:
                    raise ValueError(f"Missing pred columns in {csv_path}")

                df = df.copy()
                df["pooling"] = pooling
                df["dataset"] = ds_dir.name
                df["train_seed"] = train_seed
                df = df.merge(labels_by_dataset[ds_dir.name], on="idx", how="left")
                if df["y"].isna().any():
                    missing = int(df["y"].isna().sum())
                    raise ValueError(f"{csv_path}: {missing} rows missing label y after join")

                mse_diff = (df["pred_ko"] - df["y"]) ** 2 - (df["pred_base"] - df["y"]) ** 2
                df["abs_mse_diff"] = mse_diff.abs()
                rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_seed_mean(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    key = ["dataset", "pooling", "idx", "token_pos"]
    for col in ("peptide", "rel_pos", "seq_len", "layer", "model_version"):
        if col in raw.columns:
            key.append(col)

    per_seed = raw.groupby(key + ["train_seed"], as_index=False)["abs_mse_diff"].mean()
    return per_seed.groupby(key, as_index=False).agg(
        abs_mse_diff=("abs_mse_diff", "mean"),
        std_across_seeds=(
            "abs_mse_diff",
            lambda s: float(np.std(s.astype(float), ddof=1)) if len(s) > 1 else 0.0,
        ),
        n_seeds=("abs_mse_diff", "count"),
    )


def aggregate_per_peptide_response_std(position_long: pd.DataFrame) -> pd.DataFrame:
    if position_long.empty:
        return position_long

    def _position_std(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        return float(np.std(s.astype(float), ddof=1))

    group_cols = ["dataset", "pooling", "idx"]
    agg_spec: dict[str, tuple] = {
        "response_std": ("abs_mse_diff", _position_std),
        "n_positions": ("abs_mse_diff", "count"),
        "n_seeds": ("n_seeds", "first"),
    }
    if "peptide" in position_long.columns:
        agg_spec["peptide"] = ("peptide", "first")

    return position_long.groupby(group_cols, as_index=False).agg(**agg_spec)


def summarize_distribution(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, pooling), grp in long_df.groupby(["dataset", "pooling"], sort=False):
        vals = grp["response_std"].astype(float)
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
                "n_seeds_mean": float(grp["n_seeds"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(POOLING_ORDER)}
    out["_ord"] = out["pooling"].map(lambda p: order.get(p, 999))
    return out.sort_values(["dataset", "_ord", "pooling"]).drop(columns=["_ord"]).reset_index(drop=True)


def process_dataset(
    analysis_root: Path,
    dataset: str,
    exp3_root: Path,
    repo_root: Path,
    *,
    poolings: Optional[Sequence[str]] = None,
    min_seeds: int = 1,
    diff: int = 5,
    threshold: float = 0.9,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[bool, pd.DataFrame]:
    out_dir = exp_agg_dir(analysis_root, dataset, "exp3")
    long_csv = out_dir / "token_knockout_mse_diff_long.csv"
    summary_csv = out_dir / "token_knockout_mse_diff_summary.csv"

    if long_csv.is_file() and summary_csv.is_file() and not force:
        long_df = pd.read_csv(long_csv)
        print(f"[SKIP] {long_csv} ({len(long_df)} rows)")
        return True, long_df

    labels = load_labels(repo_root, dataset, diff, threshold)
    raw = gather_raw_csvs(exp3_root, [dataset], {dataset: labels}, poolings)
    if raw.empty:
        print(f"[FAIL] {dataset}: no raw CSVs under {exp3_root}", file=sys.stderr)
        return False, pd.DataFrame()

    position_long = aggregate_seed_mean(raw)
    position_long = position_long[position_long["n_seeds"] >= min_seeds].copy()
    if position_long.empty:
        print(f"[FAIL] {dataset}: empty after min_seeds filter", file=sys.stderr)
        return False, pd.DataFrame()

    long_df = aggregate_per_peptide_response_std(position_long)
    if long_df.empty:
        print(f"[FAIL] {dataset}: empty per-peptide response_std", file=sys.stderr)
        return False, pd.DataFrame()

    summary_df = summarize_distribution(long_df)
    print(
        f"[OK] {dataset}: n_obs={len(long_df)} n_peptides={long_df['idx'].nunique()} "
        f"n_poolings={long_df['pooling'].nunique()} n_seeds={int(long_df['n_seeds'].max())}"
    )

    if dry_run:
        return True, long_df

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[saved] {long_csv}")
    print(f"[saved] {summary_csv}")
    return True, long_df


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

    analysis_root = args.analysis_root.expanduser().resolve()
    exp3_root = analysis_root / args.exp3_subdir
    if not analysis_root.is_dir():
        print(f"Error: analysis root not found: {analysis_root}", file=sys.stderr)
        return 1
    if not exp3_root.is_dir():
        print(f"Error: exp3 root not found: {exp3_root}", file=sys.stderr)
        return 1

    ok_all = True
    for dataset in args.datasets:
        ok, _ = process_dataset(
            analysis_root,
            dataset,
            exp3_root,
            REPO_ROOT,
            poolings=args.poolings,
            min_seeds=args.min_seeds,
            diff=args.diff,
            threshold=args.threshold,
            force=args.force,
            dry_run=args.dry_run,
        )
        ok_all = ok_all and ok

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
