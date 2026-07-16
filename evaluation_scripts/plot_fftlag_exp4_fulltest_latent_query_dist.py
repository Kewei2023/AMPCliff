#!/usr/bin/env python3
"""Aggregate Exp4 full-test latent-query attention and plot freq-bin distributions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = REPO_ROOT.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from AMPCliff.utils.fftlag_latent_viz import (
    aggregate_attn_by_idx_across_seeds,
    build_pooled_freq_distribution_df,
    plot_latent_query_freq_distribution,
    plot_latent_query_freq_distribution_combined,
    summarize_freq_bin_distribution,
)
from AMPCliff.utils.std_logger import Logger

DEFAULT_DATASETS = ("e_coli", "s_aureus")
DEFAULT_SUBDIR = "exp4_latent_fulltest"


def _parse_seed_dirs(analysis_root: Path, seeds: Optional[Sequence[str]]) -> List[Path]:
    if seeds:
        out = []
        for s in seeds:
            d = analysis_root / f"seed_{s}"
            if d.is_dir():
                out.append(d)
            else:
                Logger.warning(f"seed dir not found: {d}")
        return sorted(out, key=lambda p: int(p.name.replace("seed_", "")))
    return sorted(
        (d for d in analysis_root.iterdir() if d.is_dir() and d.name.startswith("seed_")),
        key=lambda p: int(p.name.replace("seed_", "")),
    )


def _expected_test_size(repo_root: Path, dataset: str, diff: int, threshold: float) -> int:
    test_csv = (
        repo_root
        / "data/blosum62 average"
        / f"diff_{diff}-trd_{threshold}"
        / f"grampa_{dataset}_7_25-test.csv"
    )
    if not test_csv.is_file():
        return 0
    return len(pd.read_csv(test_csv))


def process_dataset(
    analysis_root: Path,
    dataset: str,
    seed_dirs: Sequence[Path],
    *,
    subdir: str = DEFAULT_SUBDIR,
    min_seeds: int = 1,
    expected_n: int = 0,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[bool, Dict[str, object], Optional[pd.DataFrame]]:
    """Return (ok, stats, long_df) for one dataset."""
    out_dir = exp_agg_dir(analysis_root, dataset, "exp4")
    long_csv = out_dir / "latent_query_freq_distribution_long.csv"
    summary_csv = out_dir / "latent_query_freq_distribution_summary.csv"
    box_png = out_dir / "latent_query_freq_boxplot.png"
    violin_png = out_dir / "latent_query_freq_violinplot.png"
    mismatch_csv = out_dir / "latent_query_freq_shape_mismatch.csv"

    outputs = [long_csv, summary_csv, box_png, violin_png]
    if all(p.is_file() for p in outputs) and not force:
        Logger.info(f"[SKIP] dataset={dataset} outputs exist under {out_dir}")
        long_df = pd.read_csv(long_csv)
        return True, {"skip": 1, "n_idx": int(long_df["idx"].nunique())}, long_df

    mean_by_idx, warnings = aggregate_attn_by_idx_across_seeds(
        seed_dirs,
        dataset,
        subdir=subdir,
        min_seeds=min_seeds,
    )

    for w in warnings:
        Logger.warning(w)

    n_idx = len(mean_by_idx)
    n_seeds_used = len(seed_dirs)
    stats: Dict[str, object] = {
        "n_idx": n_idx,
        "n_seeds": n_seeds_used,
        "expected_n": expected_n,
        "warnings": len(warnings),
    }

    if expected_n > 0 and n_idx < expected_n:
        Logger.warning(
            f"[INCOMPLETE] dataset={dataset}: collected {n_idx}/{expected_n} test samples"
        )

    if not mean_by_idx:
        Logger.error(f"[FAIL] dataset={dataset}: no seed-mean attention matrices")
        return False, stats, None

    long_df = build_pooled_freq_distribution_df(mean_by_idx, pool_queries=True)
    summary_df = summarize_freq_bin_distribution(long_df)
    stats["long_rows"] = len(long_df)

    if dry_run:
        Logger.info(
            f"[DRY] dataset={dataset} idx={n_idx} long_rows={len(long_df)} -> {out_dir}"
        )
        return True, stats, long_df

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    Logger.info(f"[saved] {long_csv} ({len(long_df)} rows)")
    Logger.info(f"[saved] {summary_csv}")

    if warnings:
        pd.DataFrame([{"message": w} for w in warnings]).to_csv(mismatch_csv, index=False)
        Logger.info(f"[saved] {mismatch_csv}")

    plot_latent_query_freq_distribution(
        long_df,
        str(box_png),
        kind="box",
        dataset=dataset,
        n_seeds=n_seeds_used,
    )
    plot_latent_query_freq_distribution(
        long_df,
        str(violin_png),
        kind="violin",
        dataset=dataset,
        n_seeds=n_seeds_used,
    )

    stats["out_dir"] = str(out_dir)
    return True, stats, long_df


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    seed_dirs: Sequence[Path],
    *,
    subdir: str = DEFAULT_SUBDIR,
    min_seeds: int = 1,
    diff: int = 5,
    threshold: float = 0.9,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    long_dfs: Dict[str, pd.DataFrame] = {}
    ok_all = True
    n_seeds = len(seed_dirs)

    for dataset in datasets:
        expected_n = _expected_test_size(REPO_ROOT, dataset, diff, threshold)
        ok, stats, long_df = process_dataset(
            analysis_root,
            dataset,
            seed_dirs,
            subdir=subdir,
            min_seeds=min_seeds,
            expected_n=expected_n,
            force=force,
            dry_run=dry_run,
        )
        ok_all = ok_all and ok
        if long_df is not None and not long_df.empty:
            long_dfs[dataset] = long_df
        print(
            f"[{dataset}] ok={ok} n_idx={stats.get('n_idx', 0)} "
            f"expected={stats.get('expected_n', 0)} warnings={stats.get('warnings', 0)}"
        )

    if not dry_run and len(long_dfs) >= 2:
        agg_parent = analysis_root / "aggregated"
        plot_latent_query_freq_distribution_combined(
            long_dfs,
            str(agg_parent / "exp4_latent_query_freq_boxplot_combined.png"),
            kind="box",
            n_seeds=n_seeds,
        )
        plot_latent_query_freq_distribution_combined(
            long_dfs,
            str(agg_parent / "exp4_latent_query_freq_violinplot_combined.png"),
            kind="violin",
            n_seeds=n_seeds,
        )

    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
    )
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--subdir", default=DEFAULT_SUBDIR)
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--diff", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        Logger.error(f"analysis root not found: {root}")
        return 1

    seed_dirs = _parse_seed_dirs(root, args.seeds)
    if not seed_dirs:
        Logger.error("No seed directories found")
        return 1

    return run_all(
        root,
        args.datasets,
        seed_dirs,
        subdir=args.subdir,
        min_seeds=args.min_seeds,
        diff=args.diff,
        threshold=args.threshold,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
