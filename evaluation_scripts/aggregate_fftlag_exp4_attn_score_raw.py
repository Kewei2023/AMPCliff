#!/usr/bin/env python3
# maintained by kewei li
"""Aggregate Exp4 raw attention scores across train seeds (per dataset)."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = REPO_ROOT.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from AMPCliff.utils.fftlag_latent_viz import (
    export_attn_matrix_wide_csv,
    plot_raw_attn_heatmap,
    sample_attn_matrix,
)
from AMPCliff.utils.std_logger import Logger

DEFAULT_DATASETS = ("e_coli", "s_aureus")
PT_FILENAME = "latent_attn_weights.pt"
EXP4_SUBDIR = "exp4_latent"


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


def _dataset_display(dataset: str) -> str:
    if dataset == "e_coli":
        return "E. coli"
    if dataset == "s_aureus":
        return "S. aureus"
    return dataset


def _load_attn_by_idx(
    seed_dirs: Sequence[Path],
    dataset: str,
) -> Tuple[Dict[int, List[torch.Tensor]], int]:
    """Load attn matrices grouped by idx for one dataset only."""
    by_idx: DefaultDict[int, List[torch.Tensor]] = defaultdict(list)
    missing_pt = 0

    for seed_dir in seed_dirs:
        exp4_dir = seed_dir / dataset / EXP4_SUBDIR
        pt_path = exp4_dir / PT_FILENAME
        if not pt_path.is_file():
            Logger.warning(f"[MISSING] {pt_path}")
            missing_pt += 1
            continue

        payload = torch.load(pt_path, map_location="cpu")
        samples = payload.get("samples") or []
        if not samples:
            Logger.warning(f"[SKIP] empty samples in {pt_path}")
            continue

        for sample in samples:
            idx = int(sample["idx"])
            aw = sample_attn_matrix(sample["attn_weights"])
            by_idx[idx].append(aw)

    return dict(by_idx), missing_pt


def _mean_attn(mats: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([m.float() for m in mats], dim=0)
    return stacked.mean(dim=0)


def aggregate_dataset(
    analysis_root: Path,
    dataset: str,
    seed_dirs: Sequence[Path],
    *,
    idx_filter: Optional[Sequence[int]] = None,
    min_seeds: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int, int, int]:
    """Return (run, skip, fail, missing_pt, shape_mismatch) for one dataset."""
    by_idx, missing_pt = _load_attn_by_idx(seed_dirs, dataset)
    if not by_idx:
        Logger.warning(f"[SKIP] no attention data for dataset={dataset}")
        return 0, 0, 0, missing_pt, 0

    exp4_out = exp_agg_dir(analysis_root, dataset, "exp4")
    ds_label = _dataset_display(dataset)
    run = skip = fail = shape_mismatch = 0

    idxs = sorted(by_idx.keys())
    if idx_filter is not None:
        allowed = {int(i) for i in idx_filter}
        idxs = [i for i in idxs if i in allowed]

    for idx in idxs:
        mats = by_idx[idx]
        if len(mats) < min_seeds:
            Logger.warning(
                f"[SKIP] dataset={dataset} idx={idx}: only {len(mats)} seeds "
                f"(min_seeds={min_seeds})"
            )
            skip += 1
            continue

        shapes = {tuple(m.shape) for m in mats}
        if len(shapes) != 1:
            Logger.warning(
                f"[SHAPE_MISMATCH] dataset={dataset} idx={idx}: shapes={sorted(shapes)}"
            )
            shape_mismatch += 1
            continue

        out_dir = exp4_out / "per_sample" / f"idx_{idx}" / "plots" / "mean_style"
        out_png = out_dir / "attn_score_raw.png"
        out_csv = out_dir / "attn_score_raw.csv"
        has_png = out_png.is_file()
        has_csv = out_csv.is_file()

        if has_png and has_csv and not force:
            skip += 1
            continue

        n_seeds = len(mats)
        title = f"Cross-attention score — {ds_label} idx={idx} (mean across {n_seeds} seeds)"

        if dry_run:
            if not has_png or force:
                Logger.info(f"[DRY] {out_png} <- {dataset} idx={idx} n_seeds={n_seeds}")
            if not has_csv or force:
                Logger.info(f"[DRY] {out_csv} <- {dataset} idx={idx} n_seeds={n_seeds}")
            run += 1
            continue

        try:
            mean_mat = _mean_attn(mats)
            out_dir.mkdir(parents=True, exist_ok=True)
            if not has_png or force:
                plot_raw_attn_heatmap(mean_mat, str(out_png), title=title, out_csv=str(out_csv))
            elif not has_csv:
                export_attn_matrix_wide_csv(mean_mat, str(out_csv))
            run += 1
        except Exception as exc:
            Logger.error(f"[FAIL] dataset={dataset} idx={idx}: {exc}")
            fail += 1

    return run, skip, fail, missing_pt, shape_mismatch


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    seeds: Optional[Sequence[str]] = None,
    idx_filter: Optional[Sequence[int]] = None,
    min_seeds: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int, int, int]:
    """Return (run, skip, fail, missing_pt, shape_mismatch) totals."""
    seed_dirs = _parse_seed_dirs(analysis_root, seeds)
    if not seed_dirs:
        Logger.warning("No seed directories found")
        return 0, 0, 0, 0, 0

    total_run = total_skip = total_fail = total_missing = total_shape = 0

    for dataset in datasets:
        Logger.info(f"========== Aggregating dataset={dataset} ==========")
        run, skip, fail, missing, shape_mismatch = aggregate_dataset(
            analysis_root,
            dataset,
            seed_dirs,
            idx_filter=idx_filter,
            min_seeds=min_seeds,
            force=force,
            dry_run=dry_run,
        )
        print(
            f"[{dataset}] RUN={run} SKIP={skip} FAIL={fail} "
            f"MISSING_PT={missing} SHAPE_MISMATCH={shape_mismatch}"
        )
        total_run += run
        total_skip += skip
        total_fail += fail
        total_missing += missing
        total_shape += shape_mismatch

    return total_run, total_skip, total_fail, total_missing, total_shape


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
        help="Datasets to aggregate separately (default: e_coli s_aureus)",
    )
    ap.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Train seeds to include (default: all seed_* under analysis-root)",
    )
    ap.add_argument(
        "--idx",
        nargs="*",
        type=int,
        default=None,
        help="Optional peptide idx filter",
    )
    ap.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Minimum number of seeds required to plot one idx",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing PNGs")
    ap.add_argument("--dry-run", action="store_true", help="Print paths only, no plots")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        Logger.error(f"analysis root not found: {root}")
        return 1

    run, skip, fail, missing, shape_mismatch = run_all(
        root,
        datasets=args.datasets,
        seeds=args.seeds,
        idx_filter=args.idx,
        min_seeds=args.min_seeds,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(
        f"Done. RUN={run} SKIP={skip} FAIL={fail} "
        f"MISSING_PT={missing} SHAPE_MISMATCH={shape_mismatch}"
    )
    print(
        "Outputs under: "
        f"{root}/aggregated/{{e_coli|s_aureus}}/exp4/per_sample/idx_*/plots/mean_style/attn_score_raw.{{png,csv}}"
    )
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
