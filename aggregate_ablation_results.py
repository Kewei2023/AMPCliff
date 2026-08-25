# maintained by kewei li
"""
Aggregate ablation experiment results.

Reads per-seed results, computes metrics (pearson, spearman, mse, rmse)
for each (config, seed, split), aggregates along seeds (mean +/- std),
and writes a summary CSV.

Usage:
    python aggregate_ablation_results.py
    python aggregate_ablation_results.py --dataset s_aureus
    python aggregate_ablation_results.py --model esm2_t6 --dataset s_aureus \
        --input_dir outputs/ablation_protein_sa
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


CONFIG_DISPLAY_ORDER = [
    "mean",
    "attn_structured",
    "latent_attn",
    "fft_latent_only",
    "fft_gate_only",
    "fft_gate_only_no_residual",
    "fft_gate_only_tp_mean",
    "fft_gate_only_tp_attn",
    "fft_lag_no_residual",
    "fft_lag_tp_mean",
    "fft_lag_tp_max",
    "fft_lag_tp_attn",
    "fft_lag_L1",
    "fft_lag_L4",
    "fft_lag_L8",
    "fft_lag_L16",
]

SPLITS = ["train", "valid", "test"]


def compute_metrics_from_pred_true(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if len(pred) < 3:
        return {"pearson": np.nan, "spearman": np.nan, "mse": np.nan, "rmse": np.nan}
    pr, _ = pearsonr(true, pred)
    sr, _ = spearmanr(true, pred)
    mse = float(np.mean((true - pred) ** 2))
    return {"pearson": pr, "spearman": sr, "mse": mse, "rmse": np.sqrt(mse)}


def collect_new_format(input_dir, model, dataset):
    """Collect results from {model}_{config}_{dataset}_diff5/seed_{N}/ structure."""
    records = []
    pattern = os.path.join(input_dir, f"{model}_*_{dataset}_diff5")
    for config_dir in sorted(glob.glob(pattern)):
        dirname = os.path.basename(config_dir)
        prefix = f"{model}_"
        suffix = f"_{dataset}_diff5"
        config_tag = dirname[len(prefix):-len(suffix)]

        for seed_dir in sorted(glob.glob(os.path.join(config_dir, "seed_*"))):
            seed = int(os.path.basename(seed_dir).split("_")[1])

            for split in SPLITS:
                result_files = glob.glob(
                    os.path.join(seed_dir, f"*-{split}_result.csv")
                )
                if not result_files:
                    continue
                rf = result_files[0]
                try:
                    df = pd.read_csv(rf)
                except Exception:
                    continue

                pred_col = [c for c in df.columns if c.endswith("-pred")]
                true_col = [c for c in df.columns if c == "true"]
                if not pred_col or not true_col:
                    continue

                metrics = compute_metrics_from_pred_true(
                    df[pred_col[0]].values, df[true_col[0]].values
                )
                for metric_name, val in metrics.items():
                    records.append({
                        "config": config_tag,
                        "seed": seed,
                        "split": split,
                        "metric": metric_name,
                        "value": val,
                    })
    return records


def collect_old_format(input_dir):
    """Collect results from {config}_seed{N}/metrics.csv structure."""
    records = []
    pattern = os.path.join(input_dir, "*_seed*")
    for seed_dir in sorted(glob.glob(pattern)):
        dirname = os.path.basename(seed_dir)
        parts = dirname.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        config_tag = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            continue

        metrics_path = os.path.join(seed_dir, "metrics.csv")
        if not os.path.isfile(metrics_path):
            continue

        df = pd.read_csv(metrics_path)
        if df.empty:
            continue

        row = df.iloc[0]
        for split in SPLITS:
            for metric_name in ["pearson", "spearman", "mse"]:
                col = f"{split}_{split}-{metric_name}"
                if col in row and pd.notna(row[col]):
                    records.append({
                        "config": config_tag,
                        "seed": seed,
                        "split": split,
                        "metric": metric_name,
                        "value": row[col],
                    })
                if metric_name == "mse":
                    col_recall = f"{split}_{split}-recall"
                    if col_recall in row and pd.notna(row[col_recall]):
                        records.append({
                            "config": config_tag,
                            "seed": seed,
                            "split": split,
                            "metric": "recall",
                            "value": row[col_recall],
                        })

    return records


def aggregate(records, model, dataset):
    """Compute mean and std along seeds for each (config, split, metric)."""
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["config", "split", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    rows = []
    for config in CONFIG_DISPLAY_ORDER:
        sub = agg[agg["config"] == config]
        if sub.empty:
            continue
        row = {"model": model, "dataset": dataset, "config": config}
        for _, r in sub.iterrows():
            split = r["split"]
            metric = r["metric"]
            row[f"{split}_{metric}_mean"] = r["mean"]
            row[f"{split}_{metric}_std"] = r["std"]
            row[f"{split}_{metric}_count"] = int(r["count"])
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Root directory of ablation outputs (default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        default="esm2_t6",
        help="Model name (default: esm2_t6)",
    )
    parser.add_argument(
        "--dataset",
        default="e_coli",
        help="Dataset name (default: e_coli)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: auto-generated from model/dataset)",
    )
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    input_dir = args.input_dir
    output_path = args.output

    # Auto-detect input_dir if not specified
    if input_dir is None:
        # Try dataset-specific dir first, then generic
        candidates = [
            f"outputs/ablation_protein_{dataset.replace('_', '')}",
            "outputs/ablation_protein",
        ]
        for c in candidates:
            if os.path.isdir(c):
                input_dir = c
                break
        if input_dir is None:
            input_dir = "outputs/ablation_protein"

    # Auto-generate output filename
    if output_path is None:
        output_path = os.path.join(
            input_dir, f"ablation_summary_{model}_{dataset}.csv"
        )

    print(f"Model:    {model}")
    print(f"Dataset:  {dataset}")
    print(f"Input:    {input_dir}")

    # Collect from both formats, new format takes priority
    new_records = collect_new_format(input_dir, model, dataset)
    old_records = collect_old_format(input_dir)

    print(f"  New format records: {len(new_records)}")
    print(f"  Old format records: {len(old_records)}")

    # Merge: new format overrides old for same (config, seed, split, metric)
    all_df = pd.DataFrame(new_records + old_records)
    if all_df.empty:
        print("No records found!")
        return

    # Deduplicate: keep new format (first occurrence) over old
    all_df = all_df.drop_duplicates(subset=["config", "seed", "split", "metric"], keep="first")

    print(f"  Total unique records: {len(all_df)}")

    # Aggregate
    summary_df = aggregate(all_df.to_dict("records"), model, dataset)

    if summary_df.empty:
        print("No data to aggregate!")
        return

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Print a nice summary table for test split
    print("\n" + "=" * 90)
    print(f"Test Split Summary (mean +/- std, N seeds)  |  Model: {model}  Dataset: {dataset}")
    print("=" * 90)
    print(f"{'Config':<25} {'Pearson':>20} {'Spearman':>20} {'RMSE':>20}")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        cfg = row["config"]
        pr_m = row.get("test_pearson_mean", np.nan)
        pr_s = row.get("test_pearson_std", np.nan)
        sp_m = row.get("test_spearman_mean", np.nan)
        sp_s = row.get("test_spearman_std", np.nan)
        rm_m = row.get("test_rmse_mean", np.nan)
        rm_s = row.get("test_rmse_std", np.nan)
        n = int(row.get("test_pearson_count", 0))
        print(
            f"{cfg:<25} {pr_m:.4f} +/- {pr_s:.4f} ({n})  "
            f"{sp_m:.4f} +/- {sp_s:.4f} ({n})  "
            f"{rm_m:.4f} +/- {rm_s:.4f} ({n})"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
