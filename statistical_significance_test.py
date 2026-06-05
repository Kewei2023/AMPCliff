"""
Statistical significance tests for protein (peptide) experiments.

Implements the checklist from statistical_significance_checklist_20260519.md:
- Seed-level paired t-test
- 95% confidence interval (paired t-test CI)
- Benjamini-Hochberg FDR correction within each (dataset, model_key) group
- Long-format per-seed export: task | model | pooling | seed | metric_name | metric_value

Comparisons (P0 + extended baselines):
  - FFT-LAG vs mean
  - FFT-LAG vs attn
  - FFT-LAG vs latent_attn
  - FFT-LAG vs max
  - FFT-LAG vs swe_ot

Usage:
    python statistical_significance_test.py --run_all_protein
    python statistical_significance_test.py \\
        --model esm2_t6 --dataset s_aureus \\
        --input_dirs outputs/ablation-new-data \\
        --output_dir outputs/statistical_significance/s_aureus_esm2_t6 \\
        --fft_lag_config fft_latent_attn_gate
"""

import argparse
import os
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from aggregate_ablation_results import collect_new_format, collect_old_format


COMPARISON_BASELINES = [
    ("mean", "FFT-LAG vs Mean"),
    ("attn", "FFT-LAG vs Attn"),
    ("latent_attn", "FFT-LAG vs Latent Attn"),
    ("max", "FFT-LAG vs Max"),
    ("swe_ot", "FFT-LAG vs SWE-OT"),
]

METRICS_TO_TEST = {
    "test_rmse": ("RMSE", "lower"),
    "test_pearson": ("Pearson", "higher"),
    "test_spearman": ("Spearman", "higher"),
}

DATASET_DISPLAY = {
    "e_coli": "E. coli",
    "s_aureus": "S. aureus",
}

MODEL_DISPLAY = {
    "esm2_t6": "ESM2-8M",
    "esm2_t12": "ESM2-35M",
}

ABLATION_NEW_DATA = "outputs/ablation-new-data"

FFT_LAG_CANDIDATES_BY_MODEL = {
    "esm2_t6": ["fft_latent_attn_gate", "fft_latent_attn_gate_v2"],
    "esm2_t12": ["fft_latent_attn_gate", "fft_latent_attn_gate_v2"],
}

DEFAULT_INPUT_DIRS_BY_KEY = {
    ("esm2_t6", "e_coli"): [ABLATION_NEW_DATA],
    ("esm2_t6", "s_aureus"): [ABLATION_NEW_DATA],
    ("esm2_t12", "e_coli"): [ABLATION_NEW_DATA],
    ("esm2_t12", "s_aureus"): [ABLATION_NEW_DATA],
}

PROTEIN_RUNS = [
    {
        "model": "esm2_t6",
        "dataset": "e_coli",
        "fft_lag_config": "fft_latent_attn_gate",
        "output_dir": "outputs/statistical_significance/e_coli_esm2_t6",
    },
    {
        "model": "esm2_t6",
        "dataset": "s_aureus",
        "fft_lag_config": "fft_latent_attn_gate",
        "output_dir": "outputs/statistical_significance/s_aureus_esm2_t6",
    },
    {
        "model": "esm2_t12",
        "dataset": "e_coli",
        "fft_lag_config": "fft_latent_attn_gate",
        "output_dir": "outputs/statistical_significance/e_coli_esm2_t12",
    },
    {
        "model": "esm2_t12",
        "dataset": "s_aureus",
        "fft_lag_config": "fft_latent_attn_gate",
        "output_dir": "outputs/statistical_significance/s_aureus_esm2_t12",
    },
]


def _parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def display_names(model: str, dataset: str) -> Tuple[str, str]:
    task = DATASET_DISPLAY.get(dataset, dataset)
    model_label = MODEL_DISPLAY.get(model, model)
    return task, model_label


def default_input_dirs(model: str, dataset: str) -> List[str]:
    key = (model, dataset)
    if key not in DEFAULT_INPUT_DIRS_BY_KEY:
        raise ValueError(f"No default input_dirs for model={model} dataset={dataset}")
    return list(DEFAULT_INPUT_DIRS_BY_KEY[key])


def fft_lag_candidates(model: str) -> List[str]:
    if model not in FFT_LAG_CANDIDATES_BY_MODEL:
        raise ValueError(f"Unknown model for FFT-LAG candidates: {model}")
    return FFT_LAG_CANDIDATES_BY_MODEL[model]


def collect_per_seed_records(
    input_dirs: List[str],
    model: str,
    dataset: str,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Collect per-seed test metrics from Hydra ablation directories."""
    records = []
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            warnings.warn(f"Input directory not found, skipping: {input_dir}")
            continue
        records.extend(collect_new_format(input_dir, model, dataset))
        records.extend(collect_old_format(input_dir))

    if not records:
        raise FileNotFoundError(f"No results found under {input_dirs}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["config", "seed", "split", "metric"], keep="first")
    df = df[df["split"] == "test"].copy()
    df = df[df["metric"].isin(["rmse", "pearson", "spearman"])].copy()

    if seeds is not None:
        df = df[df["seed"].isin(seeds)].copy()

    return df


def records_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long records to one row per (config, seed)."""
    wide = (
        df.pivot_table(index=["config", "seed"], columns="metric", values="value")
        .reset_index()
        .rename(columns={"config": "config_name"})
    )
    return wide.rename(
        columns={
            "rmse": "test_rmse",
            "pearson": "test_pearson",
            "spearman": "test_spearman",
        }
    )


def resolve_fft_lag_config(
    wide_df: pd.DataFrame,
    model: str,
    requested: Optional[str],
) -> str:
    """Pick FFT-LAG config, falling back when per-seed data is missing."""
    candidates = fft_lag_candidates(model)
    preferred = candidates[0]

    if requested:
        if requested not in wide_df["config_name"].unique():
            available = sorted(wide_df["config_name"].unique())
            raise ValueError(
                f"Requested FFT-LAG config '{requested}' not found. "
                f"Available configs: {available}"
            )
        return requested

    for candidate in candidates:
        if candidate in wide_df["config_name"].unique():
            if candidate != preferred:
                warnings.warn(
                    f"{preferred} has no per-seed data; using '{candidate}' instead."
                )
            return candidate

    raise ValueError(
        f"No FFT-LAG config with per-seed data found. Tried: {candidates}"
    )


def intersect_seeds(wide_df: pd.DataFrame, configs: List[str]) -> List[int]:
    """Return seeds present for every config."""
    seed_sets = []
    for cfg in configs:
        seeds = set(wide_df.loc[wide_df["config_name"] == cfg, "seed"].tolist())
        if not seeds:
            raise ValueError(f"No seeds found for config '{cfg}'")
        seed_sets.append(seeds)
    common = sorted(set.intersection(*seed_sets))
    if len(common) < 2:
        raise ValueError(f"Need at least 2 common seeds, got {common}")
    return common


def export_long_format(
    wide_df: pd.DataFrame,
    output_dir: str,
    configs: List[str],
    task_name: str,
    model_name: str,
):
    """Export per-seed metrics in checklist long format."""
    rows = []
    metric_map = {
        "test_rmse": "RMSE",
        "test_pearson": "Pearson",
        "test_spearman": "Spearman",
    }
    sub = wide_df[wide_df["config_name"].isin(configs)]
    for _, row in sub.iterrows():
        for col, metric_name in metric_map.items():
            if col in row and pd.notna(row[col]):
                rows.append(
                    {
                        "task": task_name,
                        "model": model_name,
                        "pooling": row["config_name"],
                        "seed": int(row["seed"]),
                        "metric_name": metric_name,
                        "metric_value": row[col],
                    }
                )

    long_df = pd.DataFrame(rows)
    long_path = os.path.join(output_dir, "per_seed_metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"Long-format per-seed metrics saved to {long_path}")
    print(f"  Total rows: {len(long_df)}")
    return long_df


def seed_level_paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Seed-level paired t-test with confidence interval."""
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have same length, got {len(scores_a)} and {len(scores_b)}"
        )

    if len(scores_a) < 2:
        return {
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "mean_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_seeds": len(scores_a),
        }

    diffs = scores_a - scores_b
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    mean_diff = float(np.mean(diffs))
    se_diff = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(diffs) - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_seeds": len(scores_a),
    }


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the mean of scores."""
    rng = np.random.RandomState(seed)
    n = len(scores)

    bootstrap_means = np.array(
        [np.mean(scores[rng.randint(0, n, size=n)]) for _ in range(n_bootstrap)]
    )

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def _paired_scores(
    wide_df: pd.DataFrame,
    config_a: str,
    config_b: str,
    metric_col: str,
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return metric arrays aligned by seed for paired comparison."""
    a = wide_df[wide_df["config_name"] == config_a].set_index("seed")
    b = wide_df[wide_df["config_name"] == config_b].set_index("seed")
    scores_a = []
    scores_b = []
    used_seeds = []
    for seed in seeds:
        if seed not in a.index or seed not in b.index:
            continue
        va, vb = a.at[seed, metric_col], b.at[seed, metric_col]
        if pd.isna(va) or pd.isna(vb):
            continue
        scores_a.append(float(va))
        scores_b.append(float(vb))
        used_seeds.append(seed)
    return np.array(scores_a), np.array(scores_b), used_seeds


def apply_bh_fdr(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR within each (dataset, model_key) group.

    All hypothesis tests in a group (every comparison x metric) are corrected
    jointly using scipy.stats.false_discovery_control(method='bh').
    """
    if results_df.empty:
        return results_df

    out = results_df.copy()
    p_value_fdr = np.full(len(out), np.nan, dtype=float)
    significant_fdr = np.zeros(len(out), dtype=bool)

    for _, idx in out.groupby(["dataset", "model_key"]).groups.items():
        idx_list = list(idx)
        pvals = out.loc[idx_list, "p_value"].to_numpy(dtype=float)
        valid = ~np.isnan(pvals)
        if not valid.any():
            continue

        adjusted = np.full(len(pvals), np.nan, dtype=float)
        adjusted[valid] = stats.false_discovery_control(pvals[valid], method="bh")
        p_value_fdr[idx_list] = adjusted
        significant_fdr[idx_list] = adjusted < alpha

    out["p_value_fdr"] = p_value_fdr
    out["significant_fdr_005"] = significant_fdr
    return out


def available_baselines(wide_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return baseline comparisons whose config exists in wide_df."""
    present = set(wide_df["config_name"].unique())
    available = [(cfg, label) for cfg, label in COMPARISON_BASELINES if cfg in present]
    missing = [cfg for cfg, _ in COMPARISON_BASELINES if cfg not in present]
    for cfg in missing:
        warnings.warn(f"Baseline config '{cfg}' not found in input data; skipping.")
    return available


def run_analysis(
    wide_df: pd.DataFrame,
    fft_lag_config: str,
    output_dir: str,
    task_name: str,
    model_name: str,
    dataset_key: str,
    model_key: str,
) -> pd.DataFrame:
    """Run all statistical tests per the P0 checklist."""
    os.makedirs(output_dir, exist_ok=True)

    baselines = available_baselines(wide_df)
    export_configs = list({fft_lag_config, *[c for c, _ in baselines]})
    export_long_format(wide_df, output_dir, export_configs, task_name, model_name)

    all_results = []

    for baseline_config, label in baselines:
        try:
            comparison_seeds = intersect_seeds(wide_df, [fft_lag_config, baseline_config])
        except ValueError as exc:
            print(f"[SKIP] {label}: {exc}")
            continue

        for metric_col, (metric_name, direction) in METRICS_TO_TEST.items():
            scores_a, scores_b, used_seeds = _paired_scores(
                wide_df, fft_lag_config, baseline_config, metric_col, comparison_seeds
            )

            if len(scores_a) < 2:
                print(f"[SKIP] {label} / {metric_name}: insufficient paired data")
                continue

            ttest_result = seed_level_paired_ttest(scores_a, scores_b)
            bs_diff = bootstrap_ci(scores_a - scores_b)
            n = len(scores_a)

            result = {
                "dataset": dataset_key,
                "model_key": model_key,
                "task": task_name,
                "model": model_name,
                "comparison": label,
                "metric": metric_name,
                "better_direction": direction,
                "method_a": fft_lag_config,
                "method_b": baseline_config,
                "mean_a": float(np.mean(scores_a)),
                "std_a": float(np.std(scores_a, ddof=1)) if n > 1 else 0.0,
                "mean_b": float(np.mean(scores_b)),
                "std_b": float(np.std(scores_b, ddof=1)) if n > 1 else 0.0,
                "mean_diff": ttest_result["mean_diff"],
                "t_stat": ttest_result["t_stat"],
                "p_value": ttest_result["p_value"],
                "significant_005": ttest_result["significant"],
                "ci_lower": ttest_result["ci_lower"],
                "ci_upper": ttest_result["ci_upper"],
                "bootstrap_diff_ci_lower": bs_diff["ci_lower"],
                "bootstrap_diff_ci_upper": bs_diff["ci_upper"],
                "n_seeds": n,
                "seeds_used": ",".join(str(s) for s in used_seeds),
            }
            all_results.append(result)

    if not all_results:
        print("No results to save.")
        return pd.DataFrame()

    results_df = apply_bh_fdr(pd.DataFrame(all_results))

    for _, result in results_df.iterrows():
        sig_mark = (
            "***"
            if result["p_value"] < 0.001
            else "**"
            if result["p_value"] < 0.01
            else "*"
            if result["p_value"] < 0.05
            else "ns"
        )
        fdr_mark = " [FDR]" if result["significant_fdr_005"] else ""
        print(
            f"  {result['comparison']:30s} | {result['metric']:10s} | "
            f"diff={result['mean_diff']:+.4f} | "
            f"p={result['p_value']:.4f} {sig_mark} | "
            f"q={result['p_value_fdr']:.4f}{fdr_mark} | "
            f"CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}] | "
            f"n={int(result['n_seeds'])}"
        )

    csv_path = os.path.join(output_dir, "statistical_significance.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    latex = generate_latex_appendix_table(
        results_df, fft_lag_config, task_name, model_name, dataset_key, model_key
    )
    latex_path = os.path.join(output_dir, "statistical_significance.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"LaTeX table saved to {latex_path}")
    return results_df


def generate_latex_appendix_table(
    results_df: pd.DataFrame,
    fft_lag_config: str,
    task_name: str,
    model_name: str,
    dataset_key: str,
    model_key: str,
) -> str:
    """Generate LaTeX appendix table for statistical significance."""
    n_seeds = int(results_df["n_seeds"].iloc[0]) if not results_df.empty else 0
    label_suffix = f"{dataset_key}_{model_key}".replace(".", "_")
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(
        rf"  \caption{{Statistical significance on {task_name} / {model_name} "
        rf"({fft_lag_config} vs baselines). "
        rf"Seed-level paired $t$-test across {n_seeds} seeds.}}"
    )
    lines.append(rf"  \label{{tab:stat_significance_{label_suffix}}}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Comparison & Metric & Mean Diff & $p$-value & $q$-value (FDR) & 95\% CI (lower) & 95\% CI (upper) \\"
    )
    lines.append(r"    \midrule")

    for _, row in results_df.iterrows():
        sig = ""
        if row["p_value"] < 0.001:
            sig = "$^{***}$"
        elif row["p_value"] < 0.01:
            sig = "$^{**}$"
        elif row["p_value"] < 0.05:
            sig = "$^{*}$"

        fdr_sig = ""
        if row.get("significant_fdr_005", False):
            fdr_sig = "$^{\\dagger}$"

        q_val = row.get("p_value_fdr", float("nan"))
        q_str = f"{q_val:.4f}{fdr_sig}" if pd.notna(q_val) else "---"

        lines.append(
            f"    {row['comparison']} & {row['metric']} & "
            f"{row['mean_diff']:+.4f} & "
            f"{row['p_value']:.4f}{sig} & "
            f"{q_str} & "
            f"{row['ci_lower']:+.4f} & "
            f"{row['ci_upper']:+.4f} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \begin{tablenotes}")
    lines.append(r"    \small")
    lines.append(
        r"    \item Mean diff = FFT-LAG minus baseline. "
        r"For RMSE, negative diff indicates FFT-LAG is better; "
        r"for Pearson/Spearman, positive diff indicates FFT-LAG is better."
    )
    lines.append(
        r"    \item $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$. "
        r"$^{\\dagger}q < 0.05$ after Benjamini-Hochberg FDR correction "
        r"within each dataset/model group. CI from paired $t$-test."
    )
    lines.append(r"  \end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def run_single(
    model: str,
    dataset: str,
    input_dirs: List[str],
    output_dir: str,
    fft_lag_config: Optional[str] = None,
    seed_filter: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run significance analysis for one model/dataset combination."""
    task_name, model_name = display_names(model, dataset)

    print("=" * 60)
    print(f"Statistical Significance: {task_name} / {model_name}")
    print(f"Input dirs: {input_dirs}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    records = collect_per_seed_records(input_dirs, model, dataset, seeds=seed_filter)
    wide_df = records_to_wide(records)
    print(f"Loaded {len(wide_df)} config-seed rows")
    print(f"Configs present: {sorted(wide_df['config_name'].unique())}")

    resolved_fft = resolve_fft_lag_config(wide_df, model, fft_lag_config)
    baselines = available_baselines(wide_df)
    baseline_names = [c for c, _ in baselines]

    print(f"FFT-LAG config: {resolved_fft}")
    print(f"Baselines to test: {baseline_names}")
    print("\n--- Seed-level Paired t-test + 95% CI ---\n")

    return run_analysis(
        wide_df,
        resolved_fft,
        output_dir,
        task_name,
        model_name,
        dataset,
        model,
    )


def run_all_protein(merge_root: str = "outputs/statistical_significance") -> pd.DataFrame:
    """Run all four protein combinations and merge results."""
    os.makedirs(merge_root, exist_ok=True)
    all_frames = []

    for spec in PROTEIN_RUNS:
        print(f"\n{'#' * 60}\n# {spec['dataset']} / {spec['model']}\n{'#' * 60}\n")
        input_dirs = default_input_dirs(spec["model"], spec["dataset"])
        df = run_single(
            model=spec["model"],
            dataset=spec["dataset"],
            input_dirs=input_dirs,
            output_dir=spec["output_dir"],
            fft_lag_config=spec.get("fft_lag_config"),
        )
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    merged_path = os.path.join(merge_root, "statistical_significance_all.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged all runs -> {merged_path} ({len(merged)} rows)")
    return merged


def migrate_legacy_e_coli_t6(merge_root: str) -> None:
    """Copy legacy analysis/ outputs into unified layout if present."""
    legacy = "outputs/ablation_protein/analysis"
    target = os.path.join(merge_root, "e_coli_esm2_t6")
    if os.path.isdir(legacy) and not os.path.isdir(target):
        shutil.copytree(legacy, target)
        print(f"Migrated legacy results: {legacy} -> {target}")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests (protein)")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=None,
        help="Ablation output roots (default: auto by model/dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto by model/dataset)",
    )
    parser.add_argument("--model", type=str, default="esm2_t6")
    parser.add_argument("--dataset", type=str, default="e_coli")
    parser.add_argument(
        "--fft_lag_config",
        type=str,
        default=None,
        help="FFT-LAG config (auto-detect if omitted)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: intersection across configs)",
    )
    parser.add_argument(
        "--run_all_protein",
        action="store_true",
        help="Run all four protein combinations and write merged CSV",
    )
    args = parser.parse_args()

    if args.run_all_protein:
        merge_root = "outputs/statistical_significance"
        migrate_legacy_e_coli_t6(merge_root)
        run_all_protein(merge_root)
        print("\nAll protein analyses complete.")
        return

    input_dirs = args.input_dirs or default_input_dirs(args.model, args.dataset)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"outputs/statistical_significance/{args.dataset}_{args.model}"

    seed_filter = _parse_seeds(args.seeds) if args.seeds else None
    run_single(
        model=args.model,
        dataset=args.dataset,
        input_dirs=input_dirs,
        output_dir=output_dir,
        fft_lag_config=args.fft_lag_config,
        seed_filter=seed_filter,
    )
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
