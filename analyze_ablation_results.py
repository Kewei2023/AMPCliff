# maintained by kewei li
"""
Analyze ablation study results for protein experiments.

Generates:
1. Main ablation table (mean +/- std for RMSE, Pearson, Spearman)
2. LaTeX table output
3. Latent count line plot

Usage:
    python analyze_ablation_results.py
    python analyze_ablation_results.py --input_dir outputs/ablation_protein
    python analyze_ablation_results.py --output_dir paper/ablation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Config name display order and labels ----------

ABLATION_DISPLAY_ORDER = [
    "mean",
    "attn",
    "latent_attn",
    "fft_latent_only",
    "fft_gate_only",
    "fft_lag_no_residual",
    "fft_lag_tp_mean",
    "fft_lag_tp_max",
    "fft_lag_L1",
    "fft_lag_L4",
    "fft_lag_L8",
    "fft_lag_L16",
]

ABLATION_LABELS = {
    "mean": "Mean",
    "attn": "Attn",
    "latent_attn": "Latent Attn",
    "fft_latent_only": "FFT-LATENT",
    "fft_gate_only": "FFT-GATE",
    "fft_lag_no_residual": "FFT-LAG (no residual)",
    "fft_lag_tp_mean": "FFT-LAG / tp\\_mean",
    "fft_lag_tp_max": "FFT-LAG / tp\\_max",
    "fft_lag_L1": "FFT-LAG (L=1)",
    "fft_lag_L4": "FFT-LAG (L=4)",
    "fft_lag_L8": "FFT-LAG (L=8)",
    "fft_lag_L16": "FFT-LAG (L=16)",
}

# Metrics to report (test split)
METRICS = {
    "test_test-mse": "RMSE",
    "test_test-pearson": "Pearson",
    "test_test-spearman": "Spearman",
}


def load_results(input_dir: str) -> pd.DataFrame:
    """Load aggregated_results.csv or scan individual run dirs."""
    agg_path = os.path.join(input_dir, "aggregated_results.csv")
    if os.path.exists(agg_path):
        return pd.read_csv(agg_path)

    # Scan individual run dirs
    all_results = []
    for run_dir in sorted(Path(input_dir).iterdir()):
        metrics_csv = run_dir / "metrics.csv"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            all_results.append(df)

    if not all_results:
        raise FileNotFoundError(f"No results found in {input_dir}")

    return pd.concat(all_results, ignore_index=True)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std per config for each metric."""
    rows = []
    for config_name in ABLATION_DISPLAY_ORDER:
        sub = df[df["config_name"] == config_name]
        if sub.empty:
            continue
        row = {"Config": ABLATION_LABELS.get(config_name, config_name), "config_name": config_name}
        for metric_col, metric_name in METRICS.items():
            if metric_col in sub.columns:
                vals = sub[metric_col].dropna().values
                row[f"{metric_name}_mean"] = np.mean(vals)
                row[f"{metric_name}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row[f"{metric_name}_n"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """Generate LaTeX table for the ablation results."""
    metric_names = list(METRICS.values())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation study on E. coli / ESM2-8M.}")
    lines.append(r"  \label{tab:ablation_protein}")
    lines.append(r"  \begin{tabular}{l" + "c" * len(metric_names) + "}")
    lines.append(r"    \toprule")

    header = "Config"
    for m in metric_names:
        header += f" & {m}"
    header += r" \\"
    lines.append(f"    {header}")
    lines.append(r"    \midrule")

    for _, row in summary_df.iterrows():
        line = f"    {row['Config']}"
        for m in metric_names:
            mean = row.get(f"{m}_mean", float("nan"))
            std = row.get(f"{m}_std", float("nan"))
            if not np.isnan(mean):
                line += f" & {mean:.4f} $\\pm$ {std:.4f}"
            else:
                line += " & -"
        line += r" \\"
        lines.append(line)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latent_count_plot(summary_df: pd.DataFrame, output_path: str):
    """Generate line plot: metric vs num_latents."""
    latent_configs = ["fft_lag_L1", "fft_lag_L4", "fft_lag_L8", "fft_lag_L16"]
    latent_counts = [1, 4, 8, 16]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_keys = ["RMSE", "Pearson", "Spearman"]

    for ax, metric_name in zip(axes, metric_keys):
        means = []
        stds = []
        valid_counts = []
        for config_name in latent_configs:
            row = summary_df[summary_df["config_name"] == config_name]
            if not row.empty:
                mean_val = row.iloc[0].get(f"{metric_name}_mean", float("nan"))
                std_val = row.iloc[0].get(f"{metric_name}_std", float("nan"))
                means.append(mean_val)
                stds.append(std_val)
                valid_counts.append(row.iloc[0].get(f"{metric_name}_n", 0))
            else:
                means.append(float("nan"))
                stds.append(float("nan"))
                valid_counts.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(latent_counts, means, yerr=stds, marker="o", capsize=4, linewidth=2)
        ax.set_xlabel("Number of Latents (L)", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name} vs Latent Count", fontsize=13)
        ax.set_xticks(latent_counts)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Latent count plot saved to {output_path}")


def generate_summary_csv(summary_df: pd.DataFrame, output_path: str):
    """Save summary CSV."""
    summary_df.to_csv(output_path, index=False)
    print(f"Summary CSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--input_dir", type=str, default="outputs/ablation_protein")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_protein/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and aggregate
    df = load_results(args.input_dir)
    print(f"Loaded {len(df)} results from {args.input_dir}")
    print(f"Configs present: {sorted(df['config_name'].unique())}")

    # Compute summary
    summary_df = compute_summary(df)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    # Save outputs
    generate_summary_csv(summary_df, os.path.join(args.output_dir, "ablation_summary.csv"))

    latex = generate_latex_table(summary_df)
    latex_path = os.path.join(args.output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")

    # Latent count plot
    generate_latent_count_plot(summary_df, os.path.join(args.output_dir, "latent_count_plot.png"))

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
