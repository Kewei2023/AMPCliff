#!/usr/bin/env python3
"""
Aggregate test/valid Spearman, Pearson, top-K recall, and RMSE across seed_* subdirectories.

Recall matches ``AffinityMetrics`` / ``utils.metrics.cal_recall``: intersection size of
top-K indices by predicted vs true ranking (default topK=10).

RMSE is ``sqrt(mean((pred - true)^2))`` on finite pairs, matching ``sqrt`` of
``AffinityMetrics`` mean squared error.

Expects each seed directory to contain *test_result.csv / *valid_result.csv with one *-pred
and ``true``, as produced by downstream_train.py.

Variance/std in summary use sample formulas (ddof=1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import cal_recall  # noqa: E402


def _find_pred_column(df: pd.DataFrame) -> str:
    preds = [c for c in df.columns if str(c).endswith("-pred")]
    if len(preds) == 1:
        return preds[0]
    if len(preds) == 0:
        raise ValueError("No column ending with '-pred' found")
    raise ValueError(f"Expected exactly one *-pred column, got {preds}")


def _split_metrics_from_df(df: pd.DataFrame, topk: int) -> Tuple[float, float, float, float]:
    pred_col = _find_pred_column(df)
    if "true" not in df.columns:
        raise ValueError("Column 'true' not found")
    y_p = pd.to_numeric(df[pred_col], errors="coerce")
    y_t = pd.to_numeric(df["true"], errors="coerce")
    mask = y_p.notna() & y_t.notna()
    y_p = y_p[mask].to_numpy(dtype=np.float64)
    y_t = y_t[mask].to_numpy(dtype=np.float64)
    n = y_p.size
    if n < 2:
        sp = pe = float("nan")
    else:
        sp, _ = spearmanr(y_p, y_t)
        pe, _ = pearsonr(y_p, y_t)
        sp, pe = float(sp), float(pe)
    if n == 0:
        rec = rmse = float("nan")
    else:
        rec = float(cal_recall(y_p, y_t, topk))
        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
    return sp, pe, rec, rmse


def _load_split_metrics(
    seed_dir: Path, pattern: str, topk: int
) -> Optional[Tuple[float, float, float, float]]:
    files = sorted(seed_dir.rglob(pattern))
    if not files:
        return None
    direct = [f for f in files if f.parent == seed_dir]
    path = direct[0] if direct else files[0]
    df = pd.read_csv(path)
    return _split_metrics_from_df(df, topk)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "var": float("nan"), "std": float("nan"), "n": 0}
    if arr.size == 1:
        v = std = float("nan")
    else:
        v = float(np.var(arr, ddof=1))
        std = float(np.std(arr, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "var": v,
        "std": std,
        "n": int(arr.size),
    }


def collect_exp_dir(exp_dir: Path, topk: int) -> Dict[str, Any]:
    seed_dirs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")],
        key=lambda p: p.name,
    )
    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed_* subdirectories under {exp_dir}. "
            "Expected layout: exp_dir/seed_0/, exp_dir/seed_1/, ..."
        )

    rows: List[Dict[str, Any]] = []
    for sd in seed_dirs:
        seed_label = sd.name.replace("seed_", "", 1)
        test_m = _load_split_metrics(sd, "*test_result.csv", topk)
        valid_m = _load_split_metrics(sd, "*valid_result.csv", topk)
        row: Dict[str, Any] = {"seed": seed_label}
        if test_m:
            (
                row["test_spearman"],
                row["test_pearson"],
                row["test_recall"],
                row["test_rmse"],
            ) = test_m
        else:
            row["test_spearman"] = float("nan")
            row["test_pearson"] = float("nan")
            row["test_recall"] = float("nan")
            row["test_rmse"] = float("nan")
        if valid_m:
            (
                row["valid_spearman"],
                row["valid_pearson"],
                row["valid_recall"],
                row["valid_rmse"],
            ) = valid_m
        else:
            row["valid_spearman"] = float("nan")
            row["valid_pearson"] = float("nan")
            row["valid_recall"] = float("nan")
            row["valid_rmse"] = float("nan")
        rows.append(row)

    metrics = [
        "test_spearman",
        "test_pearson",
        "test_recall",
        "test_rmse",
        "valid_spearman",
        "valid_pearson",
        "valid_recall",
        "valid_rmse",
    ]
    summary: Dict[str, Any] = {}
    for m in metrics:
        vals = [float(r[m]) for r in rows if m in r and np.isfinite(r[m])]
        summary[m] = _summary_stats(vals)

    return {
        "exp_dir": str(exp_dir.resolve()),
        "topk_recall": topk,
        "seeds": rows,
        "summary": summary,
        "note": "var/std are sample statistics (ddof=1); std is nan when n<2.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment group dir, e.g. .../esm2_t12_mean_e_coli_diff5",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K for recall (same as AffinityMetrics topK; default: 10)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Per-seed metrics CSV (default: cwd/seed_metrics_<exp_dir.name>.csv)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Full JSON with summary (default: same stem as out-csv with .json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tables only; do not write files",
    )
    args = ap.parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    if not exp_dir.is_dir():
        print(f"Not a directory: {exp_dir}", file=sys.stderr)
        return 1

    try:
        data = collect_exp_dir(exp_dir, args.topk)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = Path.cwd() / f"seed_metrics_{exp_dir.name}.csv"
    else:
        out_csv = out_csv.expanduser().resolve()

    out_json = args.out_json
    if out_json is None:
        out_json = out_csv.with_suffix(".json")

    df = pd.DataFrame(data["seeds"])
    print(df.to_string(index=False))
    print()
    print("summary (mean / var / std, sample ddof=1):")
    for k, v in data["summary"].items():
        print(f"  {k}: {v}")

    if args.dry_run:
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print()
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
