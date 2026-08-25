#!/usr/bin/env python3
"""Export Exp3 token knockout |ΔMSE| CSVs used by the revised combined violin plot into one XLSX."""
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
_LONG_CSV = "token_knockout_mse_diff_long.csv"
_SUMMARY_CSV = "token_knockout_mse_diff_summary.csv"
_PLOT_SCRIPT = "evaluation_scripts/plot_fftlag_exp3_mse_diff_violin_revised.py"
_AGG_SCRIPT = "evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py"
_REGENERATE_HINT = (
    f"Run first: python {_AGG_SCRIPT} --force\n"
    f"Then: python {_PLOT_SCRIPT} --force"
)


def _load_species_tables(analysis_root: Path, species: str) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    exp3_dir = exp_agg_dir(analysis_root, species, "exp3")
    long_csv = exp3_dir / _LONG_CSV
    summary_csv = exp3_dir / _SUMMARY_CSV
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


def export_exp3_token_knockout_data(
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
        {
            "key": "figure_png",
            "value": str(
                (
                    analysis_root
                    / "aggregated"
                    / "exp3_token_knockout_mse_diff_violinplot_combined_no_swe_ot.png"
                ).resolve()
            ),
        },
        {"key": "figure_script", "value": _PLOT_SCRIPT},
        {"key": "aggregate_script", "value": _AGG_SCRIPT},
        {"key": "metric", "value": "response_std of |(pred_ko-y)^2 - (pred_base-y)^2| per peptide"},
        {"key": "export_script", "value": "evaluation_scripts/export_fftlag_exp3_token_knockout_data.py"},
        {"key": "analysis_root", "value": str(analysis_root.resolve())},
        {"key": "species", "value": ", ".join(species_list)},
        {"key": "regenerate_hint", "value": _REGENERATE_HINT},
        {
            "key": "sheet_distribution_long",
            "value": "Combined per-peptide response_std for violin/box plots",
        },
        {"key": "sheet_distribution_summary", "value": "Combined per-pooling summary statistics"},
    ]
    for key, value in source_paths.items():
        meta_rows.append({"key": key, "value": value})
    meta_df = pd.DataFrame(meta_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="meta", index=False)
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
            "exp3_token_knockout_mse_diff_violinplot_combined_data.xlsx"
        ),
    )
    ap.add_argument("--species", nargs="*", default=list(_SPECIES_ORDER))
    args = ap.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    if not analysis_root.is_dir():
        print(f"[FAIL] analysis root not found: {analysis_root}")
        return 1

    try:
        out_xlsx = export_exp3_token_knockout_data(
            analysis_root=analysis_root,
            out_xlsx=args.output.expanduser().resolve(),
            species_list=list(args.species),
        )
    except (FileNotFoundError, ImportError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] exp3 token knockout data -> {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
