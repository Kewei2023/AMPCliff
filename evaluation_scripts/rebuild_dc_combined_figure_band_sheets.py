#!/usr/bin/env python3
# maintained by kewei li
"""Rebuild band_* sheets in dc_validation_combined_figure_data.xlsx from fresh EXP5 / property KO.

Preserves heatmap_* and probe_* sheets unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = (
    REPO_ROOT / "outputs" / "analysis" / "fftlag_mechanism" / "figures" / "exp5" / "combined"
)
XLSX_PATH = COMBINED_DIR / "dc_validation_combined_figure_data.xlsx"
EXP5_ROOT = REPO_ROOT / "outputs" / "analysis" / "fftlag_mechanism" / "exp5_structure_fulltest"
PROPERTY_KO_ROOT = REPO_ROOT / "outputs" / "analysis" / "dc_validation" / "property_dc_knockout_fulltest"

SPECIES = ("e_coli", "s_aureus")
BUCKET_ORDER = ("bottom_30", "middle_40", "top_30")

# (grid_col, grid_row, panel_key, panel_title, source)
# Matches existing XLSX layout; panel_title kept for data-layer stability.
PANEL_LAYOUT = (
    (0, 0, "helix_propensity", "Helix propensity", "exp5"),
    (1, 0, "mean_hydrophobicity", "Mean hydrophobicity", "property"),
    (0, 1, "net_charge", "Net charge", "property"),
    (1, 1, "hydrophobic_moment", "Hydrophobic moment", "property"),
)

PRESERVE_SHEETS = (
    "heatmap_raw",
    "heatmap_agg",
    "probe_comparison",
    "probe_comparison_plot",
)


def _bucket_sort_key(name: str) -> int:
    try:
        return BUCKET_ORDER.index(name)
    except ValueError:
        return 99


def _load_exp5_summary(species: str) -> pd.DataFrame:
    path = EXP5_ROOT / species / "bucketwise_band_sensitivity_summary.csv"
    df = pd.read_csv(path)
    return df.sort_values("structure_bucket", key=lambda s: s.map(_bucket_sort_key))


def _load_exp5_by_band(species: str) -> pd.DataFrame:
    path = EXP5_ROOT / species / "bucketwise_band_sensitivity_by_band.csv"
    df = pd.read_csv(path)
    df["_bucket_ord"] = df["structure_bucket"].map(_bucket_sort_key)
    return df.sort_values(["_bucket_ord", "band"]).drop(columns="_bucket_ord")


def _load_exp5_sample_level(species: str) -> pd.DataFrame:
    path = EXP5_ROOT / species / "per_sample_band_sensitivity_by_structure.csv"
    df = pd.read_csv(path)
    return (
        df.groupby(["idx", "structure_bucket"], as_index=False)
        .agg(mean_abs_mse_diff=("mse_diff_abs_mean", "mean"))
    )


def _load_property_summary(species: str, prop: str) -> pd.DataFrame:
    path = (
        PROPERTY_KO_ROOT
        / species
        / "intermediate"
        / f"bucketwise_band_sensitivity_summary_{prop}.csv"
    )
    df = pd.read_csv(path)
    return df.sort_values("property_bucket", key=lambda s: s.map(_bucket_sort_key))


def _load_property_by_band(species: str, prop: str) -> pd.DataFrame:
    path = (
        PROPERTY_KO_ROOT
        / species
        / "intermediate"
        / f"bucketwise_band_sensitivity_by_band_{prop}.csv"
    )
    df = pd.read_csv(path)
    df["_bucket_ord"] = df["property_bucket"].map(_bucket_sort_key)
    return df.sort_values(["_bucket_ord", "band"]).drop(columns="_bucket_ord")


def _load_property_sample_level(species: str, prop: str) -> pd.DataFrame:
    path = (
        PROPERTY_KO_ROOT
        / species
        / "intermediate"
        / f"per_sample_band_sensitivity_by_{prop}.csv"
    )
    df = pd.read_csv(path)
    return (
        df.groupby(["idx", "property_bucket"], as_index=False)
        .agg(mean_abs_mse_diff=("mse_diff_abs_mean", "mean"))
    )


def build_band_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for grid_col, grid_row, panel_key, panel_title, source in PANEL_LAYOUT:
        for species in SPECIES:
            if source == "exp5":
                df = _load_exp5_summary(species)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "structure_bucket": r["structure_bucket"],
                            "mean_abs_mse_diff": float(r["mean_abs_mse_diff"]),
                            "n_samples": int(r["n_samples"]),
                            "property_bucket": None,
                        }
                    )
            else:
                df = _load_property_summary(species, panel_key)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "structure_bucket": None,
                            "mean_abs_mse_diff": float(r["mean_abs_mse_diff"]),
                            "n_samples": int(r["n_samples"]),
                            "property_bucket": r["property_bucket"],
                        }
                    )
    return pd.DataFrame(rows)


def build_band_by_band() -> pd.DataFrame:
    rows: list[dict] = []
    for grid_col, grid_row, panel_key, panel_title, source in PANEL_LAYOUT:
        for species in SPECIES:
            if source == "exp5":
                df = _load_exp5_by_band(species)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "structure_bucket": r["structure_bucket"],
                            "band": int(r["band"]),
                            "mse_diff_abs_mean": float(r["mse_diff_abs_mean"]),
                            "mse_diff_abs_std": float(r["mse_diff_abs_std"]),
                            "n_samples": int(r["n_samples"]),
                            "property_bucket": None,
                        }
                    )
            else:
                df = _load_property_by_band(species, panel_key)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "structure_bucket": None,
                            "band": int(r["band"]),
                            "mse_diff_abs_mean": float(r["mse_diff_abs_mean"]),
                            "mse_diff_abs_std": float(r["mse_diff_abs_std"]),
                            "n_samples": int(r["n_samples"]),
                            "property_bucket": r["property_bucket"],
                        }
                    )
    return pd.DataFrame(rows)


def build_band_sample_level() -> pd.DataFrame:
    rows: list[dict] = []
    for grid_col, grid_row, panel_key, panel_title, source in PANEL_LAYOUT:
        for species in SPECIES:
            if source == "exp5":
                df = _load_exp5_sample_level(species)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "idx": int(r["idx"]),
                            "structure_bucket": r["structure_bucket"],
                            "mean_abs_mse_diff": float(r["mean_abs_mse_diff"]),
                            "property_bucket": None,
                        }
                    )
            else:
                df = _load_property_sample_level(species, panel_key)
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "grid_col": grid_col,
                            "grid_row": grid_row,
                            "panel_title": panel_title,
                            "panel_key": panel_key,
                            "species": species,
                            "idx": int(r["idx"]),
                            "structure_bucket": None,
                            "mean_abs_mse_diff": float(r["mean_abs_mse_diff"]),
                            "property_bucket": r["property_bucket"],
                        }
                    )
    return pd.DataFrame(rows)


def build_meta(old_meta: pd.DataFrame) -> pd.DataFrame:
    meta = {str(k): v for k, v in zip(old_meta["key"], old_meta["value"])}
    meta["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["figure_script"] = "evaluation_scripts/plot_dc_validation_combined_figure_v5.py"
    meta["rebuild_script"] = "evaluation_scripts/rebuild_dc_combined_figure_band_sheets.py"
    meta["exp5_dir"] = str(EXP5_ROOT.resolve())
    meta["property_ko_dir"] = str(PROPERTY_KO_ROOT.resolve())
    return pd.DataFrame({"key": list(meta.keys()), "value": list(meta.values())})


def main() -> int:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    legacy_xlsx = (
        REPO_ROOT / "outputs" / "analysis" / "dc_validation" / "combined"
        / "dc_validation_combined_figure_data.xlsx"
    )
    if not XLSX_PATH.is_file():
        if legacy_xlsx.is_file():
            import shutil
            shutil.copy2(legacy_xlsx, XLSX_PATH)
            print(f"Seeded XLSX from legacy: {legacy_xlsx} -> {XLSX_PATH}")
        else:
            raise FileNotFoundError(XLSX_PATH)

    preserved = {name: pd.read_excel(XLSX_PATH, sheet_name=name) for name in PRESERVE_SHEETS}
    old_meta = pd.read_excel(XLSX_PATH, sheet_name="meta")

    band_summary = build_band_summary()
    band_by_band = build_band_by_band()
    band_sample_level = build_band_sample_level()
    meta = build_meta(old_meta)

    # Sanity checks matching historical shapes
    assert len(band_summary) == 24, f"band_summary rows={len(band_summary)} expected 24"
    assert len(band_by_band) == 192, f"band_by_band rows={len(band_by_band)} expected 192"
    assert len(band_sample_level) == 5288, (
        f"band_sample_level rows={len(band_sample_level)} expected 5288"
    )

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="meta", index=False)
        for name in PRESERVE_SHEETS:
            preserved[name].to_excel(writer, sheet_name=name, index=False)
        band_summary.to_excel(writer, sheet_name="band_summary", index=False)
        band_by_band.to_excel(writer, sheet_name="band_by_band", index=False)
        band_sample_level.to_excel(writer, sheet_name="band_sample_level", index=False)

    print(f"Updated: {XLSX_PATH}")
    print(f"  band_summary={len(band_summary)} band_by_band={len(band_by_band)} "
          f"band_sample_level={len(band_sample_level)}")
    print("  preserved sheets:", ", ".join(PRESERVE_SHEETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
