#!/usr/bin/env python3
"""Exp5 / DC validation design v2 — Step 1: build dc_property_table.csv.
Build dc_property_table.csv for DC validation experiments."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from AMPCliff.dc_property_utils import (
    build_charge_pI_validation,
    build_property_table_from_dirs,
    charge_pI_validation_summary,
    summarize_property_table_qc,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--e-coli-data-dir",
        type=Path,
        default=Path("data/blosum62 average/diff_5-trd_0.9"),
    )
    ap.add_argument(
        "--s-aureus-data-dir",
        type=Path,
        default=Path("data/blosum62 average/diff_5-trd_0.9"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/dc_validation/dc_property_table.csv"),
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = build_property_table_from_dirs(args.e_coli_data_dir, args.s_aureus_data_dir)
    table.to_csv(args.output, index=False)

    qc = summarize_property_table_qc(table)
    qc_path = args.output.parent / "dc_property_table_qc.json"
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2, ensure_ascii=False)

    validation = build_charge_pI_validation(table)
    validation_path = args.output.parent / "charge_pI_validation.csv"
    validation.to_csv(validation_path, index=False)
    validation_summary = charge_pI_validation_summary(validation)
    summary_path = args.output.parent / "charge_pI_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)

    print(f"Wrote {args.output} ({len(table)} rows)")
    print(f"QC -> {qc_path}")
    print(f"Charge/pI validation -> {validation_path}")
    if qc["invalid_sequence_count"]:
        print(f"WARNING: {qc['invalid_sequence_count']} invalid sequences detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
