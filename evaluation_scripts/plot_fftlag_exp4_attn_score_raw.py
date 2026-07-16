#!/usr/bin/env python3
# maintained by kewei li
"""Offline CPU replot: Exp4 latent_attn_weights.pt -> per_sample attn_score_raw.png."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = REPO_ROOT.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from AMPCliff.utils.fftlag_latent_viz import export_attn_matrix_wide_csv, plot_raw_attn_heatmap
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
    return "E. coli" if dataset == "e_coli" else "S. aureus"


def process_exp4_dir(
    exp4_dir: Path,
    *,
    dataset: str,
    seed: int,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """Return (run, skip, fail) counts for one exp4_latent directory."""
    pt_path = exp4_dir / PT_FILENAME
    if not pt_path.is_file():
        Logger.warning(f"[MISSING] {pt_path}")
        return 0, 0, 0

    payload = torch.load(pt_path, map_location="cpu")
    samples = payload.get("samples")
    if not samples:
        Logger.warning(f"[SKIP] empty samples in {pt_path}")
        return 0, 0, 0

    run = skip = fail = 0
    ds_label = _dataset_display(dataset)

    for sample in samples:
        idx = int(sample["idx"])
        out_dir = exp4_dir / "per_sample" / f"idx_{idx}"
        out_png = out_dir / "attn_score_raw.png"
        out_csv = out_dir / "attn_score_raw.csv"
        has_png = out_png.is_file()
        has_csv = out_csv.is_file()

        if has_png and has_csv and not force:
            skip += 1
            continue

        if dry_run:
            if not has_png or force:
                Logger.info(f"[DRY] {out_png} <- {pt_path} idx={idx}")
            if not has_csv or force:
                Logger.info(f"[DRY] {out_csv} <- {pt_path} idx={idx}")
            run += 1
            continue

        try:
            aw = sample["attn_weights"]
            out_dir.mkdir(parents=True, exist_ok=True)
            if not has_png or force:
                plot_raw_attn_heatmap(
                    aw,
                    str(out_png),
                    title=f"Cross-attention score — {ds_label} seed={seed} idx={idx}",
                    out_csv=str(out_csv),
                )
            elif not has_csv:
                export_attn_matrix_wide_csv(aw, str(out_csv))
            run += 1
        except Exception as exc:
            Logger.error(f"[FAIL] idx={idx} seed={seed} ds={dataset}: {exc}")
            fail += 1

    return run, skip, fail


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    seeds: Optional[Sequence[str]] = None,
    force: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    """Return (run, skip, fail, missing_pt)."""
    run = skip = fail = missing = 0

    for seed_dir in _parse_seed_dirs(analysis_root, seeds):
        seed = int(seed_dir.name.replace("seed_", ""))
        for ds in datasets:
            exp4_dir = seed_dir / ds / EXP4_SUBDIR
            if not exp4_dir.is_dir():
                Logger.warning(f"[MISSING] exp4 dir: {exp4_dir}")
                missing += 1
                continue
            pt_path = exp4_dir / PT_FILENAME
            if not pt_path.is_file():
                Logger.warning(f"[MISSING] {pt_path}")
                missing += 1
                continue

            r, s, f = process_exp4_dir(
                exp4_dir,
                dataset=ds,
                seed=seed,
                force=force,
                dry_run=dry_run,
            )
            run += r
            skip += s
            fail += f

    return run, skip, fail, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Train seeds to process (default: all seed_* under analysis-root)",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to process (default: e_coli s_aureus)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing PNGs")
    ap.add_argument("--dry-run", action="store_true", help="Print paths only, no plots")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        Logger.error(f"analysis root not found: {root}")
        return 1

    run, skip, fail, missing = run_all(
        root,
        datasets=args.datasets,
        seeds=args.seeds,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"Done. RUN={run} SKIP={skip} FAIL={fail} MISSING_PT={missing}")
    print(f"Outputs under: {root}/seed_*/{{dataset}}/{EXP4_SUBDIR}/per_sample/idx_*/attn_score_raw.{{png,csv}}")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
