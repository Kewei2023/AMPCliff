#!/usr/bin/env python3
"""Select a fixed random subset of test peptides for AMP knockout diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_test_csv(dataset: str, condition: str, diff: int, threshold: float) -> Path:
    base = REPO_ROOT / "data" / condition / f"diff_{diff}-trd_{threshold}"
    return base / f"grampa_{dataset}_7_25-test.csv"


def _load_test_df(test_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    if "Idx" not in df.columns:
        raise ValueError(f"Column 'Idx' not found in {test_csv}")
    return df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)


def _rows_from_df(sub: pd.DataFrame) -> tuple[list[int], list[str], list[Any]]:
    idx_list = [int(x) for x in sub["Idx"].tolist()]
    peptides = sub["Sequence"].tolist()
    ids = sub["ID"].tolist() if "ID" in sub.columns else [str(i) for i in idx_list]
    sample_id_list = [int(x) if str(x).isdigit() else x for x in ids]
    return idx_list, peptides, sample_id_list


def select_all_test(
    test_csv: Path,
    model_version: str,
    dataset: str,
    diff: int,
) -> dict:
    """Include every test-split peptide (full test set manifest for traceability)."""
    df = _load_test_df(test_csv)
    idx_list, peptides, sample_id_list = _rows_from_df(df)
    return {
        "model_version": model_version,
        "dataset": dataset,
        "diff": diff,
        "subset_mode": "all_test",
        "n_peptides": len(idx_list),
        "test_csv": str(test_csv.resolve()),
        "idx_list": idx_list,
        "sample_id_list": sample_id_list,
        "peptides": peptides,
    }


def select_subset(
    test_csv: Path,
    n_peptides: int,
    subset_seed: int,
    model_version: str,
    dataset: str,
    diff: int,
) -> dict:
    df = _load_test_df(test_csv)
    n = min(n_peptides, len(df))
    rng = np.random.default_rng(subset_seed)
    pick = rng.choice(len(df), size=n, replace=False)
    sub = df.iloc[pick]
    idx_list, peptides, sample_id_list = _rows_from_df(sub)
    return {
        "model_version": model_version,
        "dataset": dataset,
        "diff": diff,
        "subset_seed": int(subset_seed),
        "n_peptides": n,
        "test_csv": str(test_csv.resolve()),
        "idx_list": idx_list,
        "sample_id_list": sample_id_list,
        "peptides": peptides,
    }


def extend_subset(
    existing: Dict[str, Any],
    test_csv: Path,
    add_n: int,
    extend_seed: int,
) -> dict:
    """Keep existing idx_list order; append ``add_n`` new samples from the test pool."""
    keep_idx = [int(x) for x in existing["idx_list"]]
    if len(set(keep_idx)) != len(keep_idx):
        raise ValueError("existing manifest idx_list contains duplicates")

    df = _load_test_df(test_csv)
    pool = df[~df["Idx"].isin(keep_idx)].reset_index(drop=True)
    if add_n <= 0:
        raise ValueError(f"add_n must be positive, got {add_n}")
    if add_n > len(pool):
        raise ValueError(
            f"cannot add {add_n} peptides: only {len(pool)} remain after excluding {len(keep_idx)} kept idx"
        )

    rng = np.random.default_rng(extend_seed)
    pick = rng.choice(len(pool), size=add_n, replace=False)
    new_sub = pool.iloc[pick]
    new_idx, new_peptides, new_sample_ids = _rows_from_df(new_sub)

    merged_idx = keep_idx + new_idx
    merged_peptides = list(existing["peptides"]) + new_peptides
    merged_sample_ids = list(existing.get("sample_id_list", keep_idx)) + new_sample_ids

    payload = dict(existing)
    payload.update(
        {
            "n_peptides": len(merged_idx),
            "extend_seed": int(extend_seed),
            "added_n": int(add_n),
            "test_csv": str(test_csv.resolve()),
            "idx_list": merged_idx,
            "sample_id_list": merged_sample_ids,
            "peptides": merged_peptides,
        }
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-version", default=None, help="esm2_t6 or esm2_t12")
    ap.add_argument("--dataset", default=None, help="e_coli or s_aureus")
    ap.add_argument("--diff", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--condition", default="blosum62 average")
    ap.add_argument("--test-csv", type=Path, default=None)
    ap.add_argument("--n-peptides", type=int, default=5)
    ap.add_argument("--subset-seed", type=int, default=42)
    ap.add_argument(
        "--extend-manifest",
        type=Path,
        default=None,
        help="Extend an existing manifest: keep idx_list order, append --add-n new peptides",
    )
    ap.add_argument(
        "--add-n",
        type=int,
        default=None,
        help="Number of peptides to append when using --extend-manifest",
    )
    ap.add_argument(
        "--extend-seed",
        type=int,
        default=43,
        help="RNG seed for appended peptides (default 43; original subset_seed unchanged)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/ablation_new_data/_amp_knockout_seed_runs/_peptide_manifest",
    )
    ap.add_argument(
        "--all-test",
        action="store_true",
        help="Write manifest with all test-split peptides (suffix _fulltest in filename)",
    )
    args = ap.parse_args()

    if args.extend_manifest is not None:
        if args.add_n is None:
            print("Error: --add-n is required with --extend-manifest", file=sys.stderr)
            return 1
        if not args.extend_manifest.is_file():
            print(f"Error: extend manifest not found: {args.extend_manifest}", file=sys.stderr)
            return 1

        existing = json.loads(args.extend_manifest.read_text(encoding="utf-8"))
        model_version = args.model_version or existing["model_version"]
        dataset = args.dataset or existing["dataset"]
        diff = args.diff if args.diff != 5 or "diff" not in existing else existing["diff"]
        test_csv = args.test_csv or Path(existing.get("test_csv", ""))
        if not test_csv.is_file():
            test_csv = _default_test_csv(dataset, args.condition, diff, args.threshold)
        if not test_csv.is_file():
            print(f"Error: test csv not found: {test_csv}", file=sys.stderr)
            return 1

        payload = extend_subset(
            existing=existing,
            test_csv=test_csv,
            add_n=args.add_n,
            extend_seed=args.extend_seed,
        )
        out_path = args.out_dir / f"{model_version}_{dataset}_diff{diff}.json"
    else:
        if not args.model_version or not args.dataset:
            print("Error: --model-version and --dataset are required without --extend-manifest", file=sys.stderr)
            return 1
        test_csv = args.test_csv or _default_test_csv(args.dataset, args.condition, args.diff, args.threshold)
        if not test_csv.is_file():
            print(f"Error: test csv not found: {test_csv}", file=sys.stderr)
            return 1
        if args.all_test:
            payload = select_all_test(
                test_csv=test_csv,
                model_version=args.model_version,
                dataset=args.dataset,
                diff=args.diff,
            )
            out_path = args.out_dir / f"{args.model_version}_{args.dataset}_diff{args.diff}_fulltest.json"
        else:
            payload = select_subset(
                test_csv=test_csv,
                n_peptides=args.n_peptides,
                subset_seed=args.subset_seed,
                model_version=args.model_version,
                dataset=args.dataset,
                diff=args.diff,
            )
            out_path = args.out_dir / f"{args.model_version}_{args.dataset}_diff{args.diff}.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    kept = payload["n_peptides"] - payload.get("added_n", payload["n_peptides"])
    added = payload.get("added_n", payload["n_peptides"])
    print(
        f"Wrote {out_path} ({payload['n_peptides']} peptides: kept={kept} added={added}, "
        f"idx={payload['idx_list']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
