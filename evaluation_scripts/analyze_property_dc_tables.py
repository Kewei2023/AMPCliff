#!/usr/bin/env python3
# maintained by kewei li
"""Exp5 / DC validation design v2 — Step 5 / 主实验二 Part B: property-bucket tables.
Compute property-stratified DC knockout tables (no plots)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from analyze_fft_lag_mechanism_by_structure import analyze_property_dc_sensitivity
from dc_property_utils import load_knockout_property_proxy_df
from fftlag_aggregated_paths import resolve_aggregated_csv


def run_tables(
    aggregated_dir: Path,
    property_table: Path,
    output_dir: Path,
    properties: Sequence[str],
    bands: Sequence[int],
    species: str | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    exp1_csv = resolve_aggregated_csv(
        aggregated_dir,
        "exp1",
        "per_sample_band_sensitivity_aggregated.csv",
        legacy_filename="exp1_per_sample_band_sensitivity_aggregated.csv",
    )
    if not exp1_csv.is_file():
        raise FileNotFoundError(f"Missing aggregated Exp1 CSV: {exp1_csv}")

    proxy_df, meta = load_knockout_property_proxy_df(
        property_table,
        exp1_csv,
        species=species,
    )
    proxy_df.to_csv(output_dir / "property_proxy_table.csv", index=False)

    frames = []
    bucket_counts: dict[str, dict[str, int]] = {}
    for prop in properties:
        frame = analyze_property_dc_sensitivity(
            proxy_df,
            exp1_csv,
            prop,
            output_dir,
            bands=bands,
            skip_plots=True,
        )
        if frame.empty:
            continue
        frames.append(frame)
        summary_path = output_dir / f"bucketwise_band_sensitivity_summary_{prop}.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            bucket_counts[prop] = {
                str(row["property_bucket"]): int(row["n_samples"])
                for _, row in summary.iterrows()
            }

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_dir / "property_dc_knockout_sensitivity.csv", index=False)

    analysis_meta = {
        "aggregated_exp1_csv": str(exp1_csv.resolve()),
        "property_table": str(property_table.resolve()),
        "species": species,
        "properties": list(properties),
        "bands": list(bands),
        "proxy_meta": meta,
        "bucket_n_samples": bucket_counts,
        "n_properties_computed": len(frames),
    }
    with (output_dir / "analysis_meta.json").open("w", encoding="utf-8") as f:
        json.dump(analysis_meta, f, indent=2, ensure_ascii=False)

    return analysis_meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aggregated-dir",
        type=Path,
        required=True,
        help="e.g. outputs/analysis/fftlag_mechanism/aggregated_fulltest/e_coli",
    )
    ap.add_argument("--property-table", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--properties",
        nargs="*",
        default=["net_charge", "mean_hydrophobicity"],
    )
    ap.add_argument("--bands", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--species", type=str, default=None)
    args = ap.parse_args()

    meta = run_tables(
        aggregated_dir=args.aggregated_dir,
        property_table=args.property_table,
        output_dir=args.output_dir,
        properties=args.properties,
        bands=args.bands,
        species=args.species,
    )
    print(f"Wrote tables -> {args.output_dir}")
    print(f"  proxy_n={meta['proxy_meta']['proxy_n']} properties={meta['n_properties_computed']}")
    for prop, counts in meta.get("bucket_n_samples", {}).items():
        print(f"  {prop} bucket n: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
