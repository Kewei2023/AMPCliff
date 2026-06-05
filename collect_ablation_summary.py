#!/usr/bin/env python3
"""
Collect ablation experiment results across seeds and compute summary statistics.

Reads per-seed result CSVs (train/valid/test) from experiment directories,
computes MSE, Pearson, RMSE, Spearman for each split, then aggregates
mean / std (ddof=1) / count across seeds.

Usage:
    python collect_ablation_summary.py \
        --model esm2_t6 \
        --config fft_gate_only \
        --dirs \
            outputs/ablation_protein_sa/esm2_t6_fft_gate_only_s_aureus_diff5 \
            outputs/ablation_protein/esm2_t6_fft_gate_only_e_coli_diff5 \
        --output outputs/ablation_summary_fft_gate_only.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PRED_COL_PATTERN = re.compile(r"-pred$")
TRUE_COL = "true"
SPLITS = ["test", "train", "valid"]
METRICS = ["mse", "pearson", "rmse", "spearman", "recall"]
DEFAULT_RECALL_TOPK = 50


def find_pred_col(df: pd.DataFrame) -> str:
    """Find the prediction column (ending with '-pred') in a DataFrame."""
    for col in df.columns:
        if PRED_COL_PATTERN.search(col):
            return col
    raise ValueError(f"No '-pred' column found. Columns: {list(df.columns)}")


def cal_recall(y_pred: np.ndarray, y_true: np.ndarray, top: int) -> int:
    """Top-k index intersection recall (same as utils/metrics.py:cal_recall)."""
    top = min(top, len(y_pred), len(y_true))
    if top <= 0:
        return 0
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    return len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))


def calc_metrics(result_csv: str, recall_topk: int = DEFAULT_RECALL_TOPK) -> dict:
    """Calculate mse, pearson, rmse, spearman, recall from a result CSV."""
    df = pd.read_csv(result_csv)
    pred_col = find_pred_col(df)
    pred = df[pred_col].values.astype(float)
    true_val = df[TRUE_COL].values.astype(float)

    mse = float(np.mean((true_val - pred) ** 2))
    rmse = float(np.sqrt(mse))
    pearson = float(pearsonr(true_val, pred)[0])
    spearman = float(spearmanr(true_val, pred)[0])
    recall = float(cal_recall(pred, true_val, recall_topk))

    return {"mse": mse, "pearson": pearson, "rmse": rmse, "spearman": spearman, "recall": recall}


def discover_dataset_from_dir(dir_path: str) -> str:
    """Infer dataset name from directory name.

    E.g. 'esm2_t6_fft_gate_only_s_aureus_diff5' -> 's_aureus'
         'esm2_t6_fft_gate_only_e_coli_diff5'   -> 'e_coli'
    """
    dirname = Path(dir_path).name
    # Try known dataset suffixes before '_diff'
    for ds in ["s_aureus", "e_coli"]:
        if ds in dirname:
            return ds
    # Fallback: extract the part between config name and '_diff'
    m = re.search(r"_(s_aureus|e_coli)_diff", dirname)
    if m:
        return m.group(1)
    return "unknown"


def collect_one_dir(dir_path: str, model: str, config: str) -> dict | None:
    """Collect metrics from all seeds in one experiment directory."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"[WARN] Directory not found: {dir_path}", file=sys.stderr)
        return None

    dataset = discover_dataset_from_dir(str(dir_path))

    # Discover seed directories
    seed_dirs = sorted(
        [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    )
    if not seed_dirs:
        print(f"[WARN] No seed directories found in: {dir_path}", file=sys.stderr)
        return None

    # Find a sample result file to determine the file name pattern
    result_pattern = None
    for seed_dir in seed_dirs:
        csvs = list(seed_dir.glob("*-test_result.csv"))
        if csvs:
            # e.g. "esm2_t6-blosum62 average-diff5-trd0.9-test_result.csv"
            result_pattern = csvs[0].name.replace("-test_result.csv", "")
            break
    if result_pattern is None:
        print(f"[WARN] No result CSVs found in: {dir_path}", file=sys.stderr)
        return None

    row = {"model": model, "dataset": dataset, "config": config}

    for split in SPLITS:
        csv_name = f"{result_pattern}-{split}_result.csv"
        metrics_list = []

        for seed_dir in seed_dirs:
            csv_path = seed_dir / csv_name
            if not csv_path.exists():
                print(f"  [SKIP] {csv_path} not found", file=sys.stderr)
                continue
            try:
                m = calc_metrics(str(csv_path))
                metrics_list.append(m)
            except Exception as e:
                print(f"  [ERROR] {csv_path}: {e}", file=sys.stderr)

        count = len(metrics_list)
        for metric in METRICS:
            if count > 0:
                vals = [m[metric] for m in metrics_list]
                row[f"{split}_{metric}_mean"] = float(np.mean(vals))
                row[f"{split}_{metric}_std"] = (
                    float(np.std(vals, ddof=1)) if count > 1 else np.nan
                )
                row[f"{split}_{metric}_count"] = count
            else:
                row[f"{split}_{metric}_mean"] = np.nan
                row[f"{split}_{metric}_std"] = np.nan
                row[f"{split}_{metric}_count"] = 0

    print(
        f"  {model} / {dataset} / {config}: "
        f"{count} seeds, "
        f"test_spearman={row.get('test_spearman_mean', 'N/A'):.4f} "
        f"+/- {row.get('test_spearman_std', 'N/A'):.4f}, "
        f"test_recall@{DEFAULT_RECALL_TOPK}={row.get('test_recall_mean', 'N/A'):.2f} "
        f"+/- {row.get('test_recall_std', 'N/A'):.2f}",
        file=sys.stderr,
    )

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Collect ablation experiment summary statistics."
    )
    parser.add_argument(
        "--model", required=True, help="Model name, e.g. esm2_t6"
    )
    parser.add_argument(
        "--config", required=True, help="Config/pooling name, e.g. fft_gate_only"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more experiment directories (each containing seed_*/ subdirs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Build column order matching existing summary CSVs
    columns = ["model", "dataset", "config"]
    for split in SPLITS:
        for metric in METRICS:
            columns.extend(
                [f"{split}_{metric}_mean", f"{split}_{metric}_std", f"{split}_{metric}_count"]
            )

    rows = []
    for d in args.dirs:
        print(f"Processing: {d}", file=sys.stderr)
        row = collect_one_dir(d, args.model, args.config)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[ERROR] No data collected. Exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=columns)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}", file=sys.stderr)
    print(df.to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
