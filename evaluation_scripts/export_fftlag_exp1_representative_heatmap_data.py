#!/usr/bin/env python3
"""Export Exp1 representative band-knockout heatmap CSVs into one XLSX."""
from __future__ import annotations

import argparse
import json
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
from plot_fftlag_exp1_representative_heatmaps import (
    COMBINED_PNG,
    LONG_CSV,
    REPRESENTATIVE_IDX,
    WIDE_CSV,
)

_SPECIES_ORDER = ("e_coli", "s_aureus")
_PLOT_SCRIPT = "evaluation_scripts/plot_fftlag_exp1_representative_heatmaps.py"
_REGENERATE_HINT = f"Run first: python {_PLOT_SCRIPT} --force"

_CAPTION = (
    "Figure (Exp1). Representative per-peptide layer×band |ΔMSE| heatmaps from spectral "
    "band knockout (Exp1), aligned with Exp4 Panel B sample IDs. Each heatmap shows "
    "6 ESM2 layers (rows) × 8 frequency bands B0–B7 (columns). Values are absolute MSE "
    "change |knockout − baseline|, averaged over 10 training seeds then taken absolute "
    "for plotting. E. coli samples: 35, 1442, 1438, 1004, 1043; S. aureus samples: "
    "641, 379, 1963, 876, 1026."
)


def _load_idx_tables(analysis_root: Path, species: str, idx: int) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    idx_dir = exp_agg_dir(analysis_root, species, "exp1") / "per_sample" / f"idx_{idx}"
    long_csv = idx_dir / LONG_CSV
    wide_csv = idx_dir / WIDE_CSV
    for path in (long_csv, wide_csv):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required CSV for species={species} idx={idx}: {path}\n{_REGENERATE_HINT}"
            )
    long_df = pd.read_csv(long_csv)
    wide_df = pd.read_csv(wide_csv)
    long_df.insert(0, "species", species)
    wide_df.insert(0, "species", species)
    wide_df.insert(1, "idx", idx)
    return long_df, wide_df, long_csv, wide_csv


def export_exp1_representative_heatmap_data(
    *,
    analysis_root: Path,
    out_xlsx: Path,
    representative_idx: dict[str, list[int]] | None = None,
) -> Path:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("openpyxl is required. Install: pip install openpyxl") from exc

    representative_idx = representative_idx or {k: list(v) for k, v in REPRESENTATIVE_IDX.items()}
    long_frames: list[pd.DataFrame] = []
    wide_frames: list[pd.DataFrame] = []
    source_paths: dict[str, str] = {}

    for species in _SPECIES_ORDER:
        for idx in representative_idx.get(species, []):
            long_df, wide_df, long_csv, wide_csv = _load_idx_tables(analysis_root, species, idx)
            long_frames.append(long_df)
            wide_frames.append(wide_df)
            source_paths[f"{species}_idx_{idx}_long_csv"] = str(long_csv.resolve())
            source_paths[f"{species}_idx_{idx}_wide_csv"] = str(wide_csv.resolve())

    combined_long = pd.concat(long_frames, ignore_index=True)
    combined_wide = pd.concat(wide_frames, ignore_index=True)

    combined_long_path = analysis_root / "aggregated" / "exp1_representative_band_knockout_long.csv"
    combined_png = analysis_root / "aggregated" / COMBINED_PNG
    manifest_path = analysis_root / "aggregated" / "exp1_representative_band_knockout_manifest.json"

    meta_rows = [
        {"key": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "caption", "value": _CAPTION},
        {"key": "figure_png", "value": str(combined_png.resolve())},
        {"key": "figure_script", "value": _PLOT_SCRIPT},
        {"key": "export_script", "value": "evaluation_scripts/export_fftlag_exp1_representative_heatmap_data.py"},
        {"key": "analysis_root", "value": str(analysis_root.resolve())},
        {"key": "representative_idx_json", "value": json.dumps(representative_idx)},
        {"key": "regenerate_hint", "value": _REGENERATE_HINT},
        {"key": "sheet_combined_long", "value": "All representative peptides, long format"},
        {"key": "sheet_combined_wide", "value": "All representative peptides, layer×band wide format"},
    ]
    if combined_long_path.is_file():
        meta_rows.append({"key": "combined_long_csv", "value": str(combined_long_path.resolve())})
    if manifest_path.is_file():
        meta_rows.append({"key": "manifest_json", "value": str(manifest_path.resolve())})
    for key, value in source_paths.items():
        meta_rows.append({"key": key, "value": value})
    meta_df = pd.DataFrame(meta_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        meta_df.to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame([{"caption": _CAPTION}]).to_excel(writer, sheet_name="caption", index=False)
        combined_long.to_excel(writer, sheet_name="combined_long", index=False)
        combined_wide.to_excel(writer, sheet_name="combined_wide", index=False)
        for species in _SPECIES_ORDER:
            for idx in representative_idx.get(species, []):
                long_df, wide_df, _, _ = _load_idx_tables(analysis_root, species, idx)
                sheet_base = f"{species}_idx_{idx}"
                long_df.to_excel(writer, sheet_name=f"{sheet_base}_long", index=False)
                wide_df.to_excel(writer, sheet_name=f"{sheet_base}_wide", index=False)

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
            "exp1_representative_band_knockout_heatmaps_data.xlsx"
        ),
    )
    args = ap.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    if not analysis_root.is_dir():
        print(f"[FAIL] analysis root not found: {analysis_root}")
        return 1

    out_xlsx = args.output.expanduser().resolve()
    try:
        path = export_exp1_representative_heatmap_data(
            analysis_root=analysis_root,
            out_xlsx=out_xlsx,
        )
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[saved] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
