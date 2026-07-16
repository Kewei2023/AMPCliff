#!/usr/bin/env python3
"""Summarize per-seed dc_property_probe_results.csv into mean ± std table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

import pandas as pd

from AMPCliff.analyze_dc_property_encoding import build_probe_results_summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/analysis/dc_validation/dc_property_encoding/dc_property_probe_results.csv"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/dc_validation/dc_property_encoding/dc_property_probe_results_summary.csv"),
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[FAIL] missing input: {args.input}")
        return 1

    results = pd.read_csv(args.input)
    summary = build_probe_results_summary(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"[OK] wrote {len(summary)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
