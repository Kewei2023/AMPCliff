#!/usr/bin/env python3
# maintained by kewei li
"""
Merge four seed_metrics_* CSV files (same pooling family) into one workbook:

  - Four sheets: one per (esm2_t6/t12 × e_coli/s_aureus)
  - Fifth sheet ``metrics_mean_std``: per-config sample mean, sample variance, and sample std (ddof=1)
    across seeds for test/valid spearman, pearson, recall

Pooling kinds:

  - ``mean``, ``max``, ``attn_structured``: files named
    ``seed_metrics_<kind>_pooling_<model>_<dataset>.csv``
  - ``se``: SE pooling files ``seed_metrics_se_pooling_<model>_<dataset>.csv``

Default:
  - read CSV from ``<repo_root>/statics/<pooling>/``
  - write workbook to ``<repo_root>/statics/seed_metrics_<kind>_pooling_merged.xlsx``
    (or ``seed_metrics_se_pooling_merged.xlsx`` when --pooling se)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

GRID: list[tuple[str, str, str]] = [
    ("esm2_t6_e_coli", "esm2_t6", "e_coli"),
    ("esm2_t6_s_aureus", "esm2_t6", "s_aureus"),
    ("esm2_t12_e_coli", "esm2_t12", "e_coli"),
    ("esm2_t12_s_aureus", "esm2_t12", "s_aureus"),
]

METRIC_COLS = [
    "test_spearman",
    "test_pearson",
    "test_recall",
    "valid_spearman",
    "valid_pearson",
    "valid_recall",
]


def _resolve_csv(pooling: str, model: str, dataset: str, statics_root: Path) -> Path:
    pooling_dir = statics_root / pooling
    if pooling == "se":
        return pooling_dir / f"seed_metrics_se_pooling_{model}_{dataset}.csv"
    return pooling_dir / f"seed_metrics_{pooling}_pooling_{model}_{dataset}.csv"


def _metric_stats(df: pd.DataFrame) -> dict:
    row: dict = {}
    for col in METRIC_COLS:
        if col not in df.columns:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_var"] = np.nan
            row[f"{col}_std"] = np.nan
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        arr = s.to_numpy(dtype=np.float64)
        n = int(arr.size)
        if n == 0:
            row[f"{col}_mean"] = np.nan
            row[f"{col}_var"] = np.nan
            row[f"{col}_std"] = np.nan
        else:
            m = float(np.mean(arr))
            row[f"{col}_mean"] = m
            if n > 1:
                v = float(np.var(arr, ddof=1))
                row[f"{col}_var"] = v
                row[f"{col}_std"] = float(np.sqrt(v))
            else:
                row[f"{col}_var"] = float("nan")
                row[f"{col}_std"] = float("nan")
    row["n_seeds"] = len(df)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pooling",
        choices=("attn_structured","last","latent_attn","mltp_paper","FLaG","mean","max"),
        required=True,
        help="Pooling family: mean/max/attn_structured (baseline) or se (SE pooling filenames)",
    )
    ap.add_argument(
        "--statics-root",
        type=Path,
        default=REPO_ROOT / "statics",
        help="Root statics directory (expects CSV under <statics-root>/<pooling>/)",
    )
    ap.add_argument(
        "--out-xlsx",
        type=Path,
        default=None,
        help="Output path (defaults under --statics-root)",
    )
    args = ap.parse_args()

    statics_root = args.statics_root.expanduser().resolve()
    if args.out_xlsx is None:
        if args.pooling == "se":
            out_xlsx = statics_root / "seed_metrics_se_pooling_merged.xlsx"
        else:
            out_xlsx = statics_root / f"seed_metrics_{args.pooling}_pooling_merged.xlsx"
    else:
        out_xlsx = args.out_xlsx.expanduser().resolve()

    summary_rows: list[dict] = []

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sheet_name, model_type, dataset in GRID:
            csv_path = _resolve_csv(args.pooling, model_type, dataset, statics_root)
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

            st = _metric_stats(df)
            summary_rows.append(
                {
                    "sheet": sheet_name,
                    "model_type": model_type,
                    "dataset": dataset,
                    "source_csv": str(csv_path),
                    **st,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="metrics_mean_std", index=False)

    print(f"Wrote {out_xlsx.resolve()}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
