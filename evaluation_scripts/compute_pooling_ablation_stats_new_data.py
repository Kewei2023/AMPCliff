#!/usr/bin/env python3
"""
Aggregate test-set metrics across seeds for pooling ablations under
outputs/ablation-new-data/.

Only ``seed_<id>`` with ``seed_min <= id <= seed_max`` are used (default 0--9;
``seed_10``+ ignored). See ``--seed-min`` / ``--seed-max``.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from extract_ablation_metrics import DEFAULT_RECALL_TOPK, calculate_metrics_from_csv
_EXP_DIR_RE = re.compile(r"^(?P<model>esm2_t6|esm2_t12)_(?P<pooling>.+)_(?P<dataset>e_coli|s_aureus)_diff(?P<diff>\d+)$")
_SEED_RE = re.compile(r"^seed_(\d+)$")
def _default_ablation_new_data_root():
    return (REPO_ROOT / "outputs" / "ablation-new-data").resolve()
def parse_exp_dir_name(dirname: str):
    m = _EXP_DIR_RE.match(dirname)
    if not m: return None
    return (m.group("model"), m.group("pooling"), m.group("dataset"), int(m.group("diff")))
def _find_test_result_csv(seed_dir: Path):
    cands = sorted(seed_dir.glob("*-test_result.csv"))
    return cands[0] if cands else None
def collect_per_seed_rows(
    root: Path,
    recall_topk: int,
    seed_min: int = 0,
    seed_max: int = 9,
    poolings: Optional[set] = None,
):
    rows: List[dict] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"): continue
        parsed = parse_exp_dir_name(exp_dir.name)
        if parsed is None: continue
        model, pooling, dataset, diff = parsed
        if poolings is not None and pooling not in poolings:
            continue
        for seed_sub in sorted(exp_dir.iterdir()):
            if not seed_sub.is_dir(): continue
            sm = _SEED_RE.match(seed_sub.name)
            if not sm: continue
            seed_id = int(sm.group(1))
            if seed_id < seed_min or seed_id > seed_max: continue
            csv_path = _find_test_result_csv(seed_sub)
            if csv_path is None:
                print(f"Warning: no *-test_result.csv under {seed_sub}", file=sys.stderr); continue
            metrics = calculate_metrics_from_csv(csv_path, recall_topk=recall_topk)
            if metrics is None:
                print(f"Warning: could not compute metrics for {csv_path}", file=sys.stderr); continue
            rows.append({"experiment_dir": exp_dir.name, "model_type": model, "pooling": pooling, "dataset": dataset, "diff": diff, "seed_dir": seed_sub.name, "seed": seed_id, "test_result_csv": str(csv_path), "spearman": metrics["spearman"], "pearson": metrics["pearson"], "recall_at_k": metrics["recall"], "recall_topk": metrics["recall_topk"], "rmse": metrics["rmse"]})
    return pd.DataFrame(rows)
def _agg_group(sub: pd.DataFrame, metrics: Tuple[str, ...] = ("spearman", "pearson", "recall_at_k", "rmse")):
    n = len(sub); out: Dict[str, object] = {"n_seeds": n}
    for col in metrics:
        vals = sub[col].astype(float).values
        out[f"{col}_mean"] = float(np.mean(vals)) if n else float("nan")
        out[f"{col}_std_ddof1"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    return out
def summarize_by_pooling(per_seed: pd.DataFrame):
    if per_seed.empty: return pd.DataFrame()
    summaries = []
    for (model, dataset, pooling), grp in per_seed.groupby(["model_type", "dataset", "pooling"], sort=False):
        row = {"model_type": model, "dataset": dataset, "pooling": pooling, "diff": int(grp["diff"].iloc[0]), "recall_topk": float(grp["recall_topk"].iloc[0])}
        row.update(_agg_group(grp)); summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["model_type", "dataset", "pooling"]).reset_index(drop=True)
def _format_pm(mean, std, ndigits=4):
    if mean != mean: return "nan"
    return f"{mean:.{ndigits}f}±{std:.{ndigits}f}"
def print_pretty_table(summary: pd.DataFrame, recall_topk: int):
    if summary.empty: print("(empty summary)"); return
    k = int(recall_topk)
    print(f"pooling | spearman | pearson | recall@{k} (count) | RMSE (mean±std over seeds, std ddof=1)")
    print("-" * 72)
    for _, r in summary.iterrows():
        print(f"{r['pooling']} | {_format_pm(r['spearman_mean'], r['spearman_std_ddof1'])} | {_format_pm(r['pearson_mean'], r['pearson_std_ddof1'])} | {_format_pm(r['recall_at_k_mean'], r['recall_at_k_std_ddof1'], ndigits=2)} | {_format_pm(r['rmse_mean'], r['rmse_std_ddof1'])}")
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--recall-topk", type=int, default=DEFAULT_RECALL_TOPK)
    ap.add_argument(
        "--seed-min",
        type=int,
        default=0,
        help="Minimum seed index (inclusive); default 0",
    )
    ap.add_argument(
        "--seed-max",
        type=int,
        default=9,
        help="Maximum seed index (inclusive); default 9",
    )
    ap.add_argument(
        "--poolings",
        type=str,
        default=None,
        help="Comma-separated pooling names to include (default: all)",
    )
    args = ap.parse_args()
    if args.seed_min > args.seed_max:
        print(f"Error: --seed-min ({args.seed_min}) > --seed-max ({args.seed_max})", file=sys.stderr); return 1
    poolings = None
    if args.poolings:
        poolings = {p.strip() for p in args.poolings.split(",") if p.strip()}
    root = args.root.expanduser().resolve() if args.root else _default_ablation_new_data_root()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (root / "_pooling_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        print(f"Error: root not found: {root}", file=sys.stderr); return 1
    per_seed = collect_per_seed_rows(
        root,
        recall_topk=args.recall_topk,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
        poolings=poolings,
    )
    per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    summary = summarize_by_pooling(per_seed)
    summary.to_csv(out_dir / "summary_by_pooling_all.csv", index=False)
    for (model, dataset), grp in summary.groupby(["model_type", "dataset"], sort=False):
        grp.to_csv(out_dir / f"summary_by_pooling_{model}_{dataset}.csv", index=False)
        print(f"\n{'='*72}\n{model}  {dataset}  (recall_topk={int(args.recall_topk)}, seeds {args.seed_min}--{args.seed_max})\n{'='*72}")
        print_pretty_table(grp.reset_index(drop=True), args.recall_topk)
    print(f"\n{'='*72}\nWrote:\n  {out_dir/'per_seed_metrics.csv'}\n  {out_dir/'summary_by_pooling_all.csv'}\n  {out_dir}/summary_by_pooling_<model>_<dataset>.csv\n{'='*72}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
