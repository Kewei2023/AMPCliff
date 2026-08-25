#!/usr/bin/env python3
# maintained by kewei li
"""Export Exp1 full-test band knockout violin CSVs into one XLSX."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
_EVAL_SCRIPTS = Path(__file__).resolve().parent
for p in (_REPO_PARENT, _REPO_ROOT, _EVAL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fftlag_aggregated_paths import exp_agg_dir

_SPECIES_ORDER = ("e_coli", "s_aureus")
_LONG_CSV = "exp1_band_knockout_violin_long.csv"
_SUMMARY_CSV = "exp1_band_knockout_violin_summary.csv"
_PLOT_SCRIPT = "evaluation_scripts/plot_fftlag_exp1_fulltest_violin.py"
_REGENERATE_HINT = f"Run first: python {_PLOT_SCRIPT} --force"

_CAPTION = (
    "Figure X. Per-layer band knockout |ΔMSE| distributions (Exp1). Violin plots show the "
    "cross-peptide distribution of absolute MSE change |knockout − baseline| when spectral "
    "notch filtering removes one frequency band at a given ESM2 layer. Each panel corresponds "
    "to one transformer layer; x-axis bands B0–B7 (B0 = DC). For each peptide, signed ΔMSE is "
    "averaged over 10 training seeds, then absolute value is taken for plotting, before pooling "
    "across the full test set (E. coli: 687 peptides; S. aureus: 635 peptides)."
)


def _load_species_tables(analysis_root: Path, species: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    exp1_dir = exp_agg_dir(analysis_root, species, "exp1")
    long_csv = exp1_dir / _LONG_CSV
    summary_csv = exp1_dir / _SUMMARY_CSV
    for path in (long_csv, summary_csv):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required CSV for species={species}: {path}\n{_REGENERATE_HINT}"
            )

    long_df = pd.read_csv(long_csv)
    summary_df = pd.read_csv(summary_csv)
    long_df.insert(0, "species", species)
    summary_df.insert(0, "species", species)
    return long_df, summary_df, long_csv, summary_csv


def export_exp1_fulltest_violin_data(
    *,
    analysis_root: Path,
    out_xlsx: Path,
    species_list: list[str] | None = None,
) -> Path:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("openpyxl is required. Install: pip install openpyxl") from exc

    species_list = species_list or list(_SPECIES_ORDER)
    long_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    source_paths: dict[str, str] = {}

    for species in species_list:
        long_df, summary_df, long_csv, summary_csv = _load_species_tables(analysis_root, species)
        long_frames.append(long_df)
        summary_frames.append(summary_df)
        source_paths[f"{species}_long_csv"] = str(long_csv.resolve())
        source_paths[f"{species}_summary_csv"] = str(summary_csv.resolve())

    distribution_long = pd.concat(long_frames, ignore_index=True)
    distribution_summary = pd.concat(summary_frames, ignore_index=True)

    meta_rows = [
        {"key": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "caption", "value": _CAPTION},
        {"key": "figure_script", "value": _PLOT_SCRIPT},
        {"key": "export_script", "value": "evaluation_scripts/export_fftlag_exp1_fulltest_violin_data.py"},
        {"key": "analysis_root", "value": str(analysis_root.resolve())},
        {"key": "species", "value": ", ".join(species_list)},
        {"key": "regenerate_hint", "value": _REGENERATE_HINT},
        {"key": "sheet_distribution_long", "value": "Combined long-format ΔMSE values for violin plots"},
        {"key": "sheet_distribution_summary", "value": "Combined per-layer×band summary statistics"},
    ]
    for key, value in source_paths.items():
        meta_rows.append({"key": key, "value": value})
    meta_df = pd.DataFrame(meta_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame([{"caption": _CAPTION}]).to_excel(writer, sheet_name="caption", index=False)
        distribution_long.to_excel(writer, sheet_name="distribution_long", index=False)
        distribution_summary.to_excel(writer, sheet_name="distribution_summary", index=False)
        for species, long_df, summary_df in zip(species_list, long_frames, summary_frames):
            long_df.to_excel(writer, sheet_name=f"{species}_long", index=False)
            summary_df.to_excel(writer, sheet_name=f"{species}_summary", index=False)

    return out_xlsx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/analysis/fftlag_mechanism"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/fftlag_mechanism/aggregated/"
            "exp1_band_knockout_violin_combined_data.xlsx"
        ),
    )
    ap.add_argument("--species", nargs="*", default=list(_SPECIES_ORDER))
    args = ap.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    if not analysis_root.is_dir():
        print(f"[FAIL] analysis root not found: {analysis_root}")
        return 1

    out_xlsx = args.output.expanduser().resolve()
    try:
        path = export_exp1_fulltest_violin_data(
            analysis_root=analysis_root,
            out_xlsx=out_xlsx,
            species_list=list(args.species),
        )
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
