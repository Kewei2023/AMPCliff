#!/usr/bin/env python3
# maintained by kewei li
"""
Aggregate per-seed knockout CSVs: mean/std over train seeds only (not across peptides).

Expected raw CSV from downstream_evaluate_knockout.py:
  <group_root>/<pooling>/seed_<k>/knockout_lastlayer_HS.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

_SEED_DIR = re.compile(r"seed_(\d+)$")
_KO_CSV_NAMES = ("knockout_lastlayer_HS.csv",)


def _infer_train_seed(path: Path) -> int:
    for part in path.parts:
        m = _SEED_DIR.match(part)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer seed from path: {path}")


def gather_raw(group_dir: Path) -> pd.DataFrame:
    rows = []
    for name in _KO_CSV_NAMES:
        for p in sorted(group_dir.rglob(name)):
            if "_aggregated" in p.parts or "_peptide_manifest" in p.parts:
                continue
            df = pd.read_csv(p)
            if df.empty:
                continue
            rel_parts = p.relative_to(group_dir).parts
            if len(rel_parts) < 2:
                continue
            # <pooling>/seed_k/knockout_lastlayer_HS.csv
            pooling = rel_parts[0]
            df["train_seed"] = _infer_train_seed(p.parent)
            df["pooling"] = pooling
            df["group"] = group_dir.name
            df["source_csv"] = str(p)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _bin_rel_pos(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = df.copy()
    out["rel_pos_bin"] = np.minimum(
        (out["rel_pos"].astype(float) * n_bins).astype(int), n_bins - 1
    )
    out["rel_pos_center"] = (out["rel_pos_bin"].astype(float) + 0.5) / n_bins
    return out


def aggregate_seeds(raw: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if raw.empty:
        return raw
    binned = _bin_rel_pos(raw, n_bins)
    base_key = ["group", "pooling", "idx", "peptide", "rel_pos_bin", "rel_pos_center"]
    for c in ("model_version", "dataset"):
        if c in binned.columns:
            base_key.insert(0, c)

    per_seed_key = base_key + ["train_seed"]
    per_seed = (
        binned.groupby(per_seed_key, as_index=False)["abs_delta"]
        .mean()
        .rename(columns={"abs_delta": "abs_delta_seed_mean"})
    )
    agg_df = per_seed.groupby(base_key, as_index=False).agg(
        mean_abs_delta=("abs_delta_seed_mean", "mean"),
        std_across_seeds=(
            "abs_delta_seed_mean",
            lambda s: float(np.std(s.astype(float), ddof=1)) if len(s) > 1 else 0.0,
        ),
        n_seeds=("abs_delta_seed_mean", "count"),
    )
    return agg_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "outputs/ablation_new_data/_amp_knockout_seed_runs",
    )
    ap.add_argument("--n-bins", type=int, default=50)
    ap.add_argument("--group", type=str, default=None, help="Only process one group dir name")
    args = ap.parse_args()

    root = args.input_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    groups = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    if args.group:
        groups = [root / args.group]

    for gdir in groups:
        raw = gather_raw(gdir)
        if raw.empty:
            print(f"[SKIP] no knockout CSVs under {gdir}")
            continue
        summary = aggregate_seeds(raw, args.n_bins)
        out_dir = gdir / "_aggregated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"summary_{gdir.name}.csv"
        summary.to_csv(out_path, index=False)
        raw_path = out_dir / f"all_seeds_long_{gdir.name}.csv"
        raw.to_csv(raw_path, index=False)
        print(f"Wrote {out_path} ({len(summary)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
