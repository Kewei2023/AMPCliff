#!/usr/bin/env python3
# maintained by kewei li
"""Replot DC validation figures from existing CSVs (no probe recomputation)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[2]
_EVAL_SCRIPTS = Path(__file__).resolve().parent
for p in (_REPO_PARENT, _EVAL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from AMPCliff.analyze_dc_property_encoding import (
    build_probe_results_summary,
    plot_delta_ci,
    plot_delta_ci_3panel,
    plot_delta_ci_3panel_combined,
    plot_property_encoding_heatmap,
)
from compare_probe_baselines import plot_comparison


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/analysis/dc_validation"),
        help="DC validation output root (contains dc_property_encoding/)",
    )
    ap.add_argument("--species", nargs="*", default=["e_coli", "s_aureus"])
    args = ap.parse_args()

    probe_dir = args.output_root / "dc_property_encoding"
    probe_csv = probe_dir / "dc_property_probe_results.csv"
    delta_csv = probe_dir / "dc_c0_vs_c1_delta_ci.csv"
    compare_csv = probe_dir / "aac_vs_dc_comparison.csv"

    if not probe_csv.is_file():
        print(f"[FAIL] missing {probe_csv}")
        return 1
    if not delta_csv.is_file():
        print(f"[FAIL] missing {delta_csv}")
        return 1

    results = pd.read_csv(probe_csv)
    delta_results = pd.read_csv(delta_csv)

    summary_csv = probe_dir / "dc_property_probe_results_summary.csv"
    build_probe_results_summary(results).to_csv(summary_csv, index=False)
    print(f"[OK] probe summary -> {summary_csv}")

    for species in args.species:
        plot_property_encoding_heatmap(
            results,
            species,
            probe_dir / f"dc_property_encoding_{species}.png",
        )
        plot_delta_ci(
            delta_results,
            species,
            probe_dir / f"dc_c0_minus_c1_delta_{species}.png",
        )
        plot_delta_ci_3panel(
            delta_results,
            results,
            species,
            probe_dir / f"dc_c0_minus_c1_delta_3panel_{species}.png",
        )
        print(f"[OK] probe heatmap + delta + 3panel -> {species}")

    plot_delta_ci_3panel_combined(
        delta_results,
        results,
        probe_dir / "dc_c0_minus_c1_delta_3panel_combined.png",
        species_order=tuple(args.species),
    )
    print("[OK] 3panel combined -> dc_c0_minus_c1_delta_3panel_combined.png")

    if compare_csv.is_file():
        comparison = pd.read_csv(compare_csv)
        plot_comparison(comparison, probe_dir / "aac_vs_dc_comparison.png")
        for species in args.species:
            sub = comparison[comparison["species"] == species]
            if sub.empty:
                continue
            plot_comparison(
                sub,
                probe_dir / f"aac_vs_dc_comparison_{species}.png",
                facet_species=False,
            )
        print("[OK] AAC comparison figures")
    else:
        print(f"[SKIP] missing {compare_csv}")

    print(f"Replotted figures -> {probe_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
