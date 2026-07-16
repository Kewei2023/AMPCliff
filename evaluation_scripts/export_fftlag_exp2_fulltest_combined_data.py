#!/usr/bin/env python3
"""Export Exp2 full-test combined gate band summary data and caption into one XLSX."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVAL_SCRIPTS = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _EVAL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fftlag_aggregated_paths import exp_agg_dir

_SPECIES_ORDER = ("e_coli", "s_aureus")
_BAND_SUMMARY_CSV = "gate_by_band_all_samples_summary.csv"
_IDX_LEVEL_CSV = "per_sample_gate_by_band_aggregated.csv"
_PLOT_SCRIPT = "evaluation_scripts/plot_fftlag_exp2_fulltest_combined.py"
_COMBINED_PNG = "exp2_gate_band_summary_all_samples_combined.png"
_COMBINED_XLSX = "exp2_gate_band_summary_all_samples_combined_data.xlsx"
_CAPTION_TXT = "exp2_gate_band_summary_all_samples_combined_caption.txt"
_AGGREGATED_FULLTEST = "aggregated_fulltest"
_REGENERATE_HINT = f"Run first: python {_PLOT_SCRIPT} --force"

EXP2_GATE_BAND_CAPTION_EN = (
    "Figure X. Band-wise spectral energy before and after the PSD gate (Exp2). "
    "Grouped bar charts show mean band energy before (blue) and after (orange) PSD gating "
    "in FFT-LAG, evaluated on the full test sets of E. coli (687 peptides; left) and "
    "S. aureus (635 peptides; right). For each peptide, band statistics are averaged over "
    "10 training seeds; bar heights report the cross-peptide mean and error bars the "
    "cross-peptide standard deviation. Low-frequency band B0 carries the largest absolute "
    "energy; all bands show higher post-gate energy, consistent with frequency-domain "
    "amplification by the learned PSD gate."
)


def _load_band_summary(analysis_root: Path, species: str) -> tuple[pd.DataFrame, Path]:
    exp2_dir = exp_agg_dir(analysis_root, species, "exp2")
    summary_csv = exp2_dir / _BAND_SUMMARY_CSV
    if not summary_csv.is_file():
        raise FileNotFoundError(
            f"Missing required CSV for species={species}: {summary_csv}\n{_REGENERATE_HINT}"
        )
    df = pd.read_csv(summary_csv)
    df.insert(0, "species", species)
    return df, summary_csv


def _load_idx_level(analysis_root: Path, species: str) -> tuple[pd.DataFrame, Path]:
    idx_csv = analysis_root / _AGGREGATED_FULLTEST / species / "exp2" / _IDX_LEVEL_CSV
    if not idx_csv.is_file():
        raise FileNotFoundError(
            f"Missing required idx-level CSV for species={species}: {idx_csv}"
        )
    df = pd.read_csv(idx_csv)
    df.insert(0, "species", species)
    return df, idx_csv


def _write_caption_txt(caption_path: Path) -> None:
    caption_path.parent.mkdir(parents=True, exist_ok=True)
    caption_path.write_text(EXP2_GATE_BAND_CAPTION_EN + "\n", encoding="utf-8")


def export_exp2_fulltest_combined_data(
    *,
    analysis_root: Path,
    out_xlsx: Path,
    caption_path: Path,
    species_list: list[str] | None = None,
) -> Path:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("openpyxl is required. Install: pip install openpyxl") from exc

    species_list = species_list or list(_SPECIES_ORDER)
    band_frames: list[pd.DataFrame] = []
    idx_frames: list[pd.DataFrame] = []
    source_paths: dict[str, str] = {}

    for species in species_list:
        band_df, band_csv = _load_band_summary(analysis_root, species)
        idx_df, idx_csv = _load_idx_level(analysis_root, species)
        band_frames.append(band_df)
        idx_frames.append(idx_df)
        source_paths[f"{species}_band_summary_csv"] = str(band_csv.resolve())
        source_paths[f"{species}_idx_level_csv"] = str(idx_csv.resolve())

    band_summary = pd.concat(band_frames, ignore_index=True)
    _write_caption_txt(caption_path)

    combined_png = (analysis_root / "aggregated" / _COMBINED_PNG).resolve()
    meta_rows = [
        {"key": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "figure_png", "value": str(combined_png)},
        {"key": "figure_script", "value": _PLOT_SCRIPT},
        {"key": "export_script", "value": "evaluation_scripts/export_fftlag_exp2_fulltest_combined_data.py"},
        {"key": "analysis_root", "value": str(analysis_root.resolve())},
        {"key": "species", "value": ", ".join(species_list)},
        {"key": "caption_txt", "value": str(caption_path.resolve())},
        {"key": "regenerate_hint", "value": _REGENERATE_HINT},
        {"key": "sheet_band_summary", "value": "Band-level plot data for both species (mean ± std across peptides)"},
        {"key": "sheet_idx_level", "value": "Per-peptide seed-mean values from aggregated_fulltest"},
    ]
    for key, value in source_paths.items():
        meta_rows.append({"key": key, "value": value})
    meta_df = pd.DataFrame(meta_rows)
    caption_df = pd.DataFrame([{"key": "caption_en", "value": EXP2_GATE_BAND_CAPTION_EN}])

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="meta", index=False)
        caption_df.to_excel(writer, sheet_name="caption", index=False)
        band_summary.to_excel(writer, sheet_name="band_summary", index=False)
        for species, band_df, idx_df in zip(species_list, band_frames, idx_frames):
            band_df.to_excel(writer, sheet_name=f"{species}_band_summary", index=False)
            idx_df.to_excel(writer, sheet_name=f"{species}_idx_level", index=False)

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
            "exp2_gate_band_summary_all_samples_combined_data.xlsx"
        ),
    )
    ap.add_argument(
        "--caption-output",
        type=Path,
        default=Path(
            "outputs/analysis/fftlag_mechanism/aggregated/"
            "exp2_gate_band_summary_all_samples_combined_caption.txt"
        ),
    )
    ap.add_argument("--species", nargs="*", default=list(_SPECIES_ORDER))
    args = ap.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    if not analysis_root.is_dir():
        print(f"[FAIL] analysis root not found: {analysis_root}")
        return 1

    try:
        out_xlsx = export_exp2_fulltest_combined_data(
            analysis_root=analysis_root,
            out_xlsx=args.output.expanduser().resolve(),
            caption_path=args.caption_output.expanduser().resolve(),
            species_list=list(args.species),
        )
    except (FileNotFoundError, ImportError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] exp2 combined data -> {out_xlsx}")
    print(f"[OK] caption -> {args.caption_output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
