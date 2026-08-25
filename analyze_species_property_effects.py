#!/usr/bin/env python3
# maintained by kewei li
"""Exp5 / DC validation design v2 — Step 4 / 主实验二 Part A: species×property activity effects.
Species-interaction regression of activity vs physicochemical properties."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

FORMULA = (
    "activity_score ~ z_charge + z_hydrophobicity + z_length + z_helix + z_hmoment "
    "+ species_code + z_charge:species_code + z_hydrophobicity:species_code "
    "+ z_hmoment:species_code + z_helix:species_code"
)


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std(ddof=0) + 1e-8)


def prepare_regression_df(property_df: pd.DataFrame) -> pd.DataFrame:
    df = property_df.copy()
    df["species_code"] = (df["species"] == "s_aureus").astype(int)
    df["z_charge"] = _zscore(df["net_charge"])
    df["z_hydrophobicity"] = _zscore(df["mean_hydrophobicity"])
    df["z_length"] = _zscore(df["length"])
    df["z_helix"] = _zscore(df["helix_propensity"])
    df["z_hmoment"] = _zscore(df["hydrophobic_moment"])
    return df


def fit_interaction_model(df: pd.DataFrame):
    return smf.ols(FORMULA, data=df).fit(cov_type="HC3")


def _linear_combo_ci(result, constraint: str) -> Tuple[float, float, float]:
    """Return (estimate, ci_lo, ci_hi) for a statsmodels linear constraint."""
    tt = result.t_test(constraint)
    est = float(tt.effect[0])
    ci = tt.conf_int()[0]
    return est, float(ci[0]), float(ci[1])


def build_effects_table(result, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("net_charge", "z_charge", "z_charge:species_code"),
        ("mean_hydrophobicity", "z_hydrophobicity", "z_hydrophobicity:species_code"),
        ("hydrophobic_moment", "z_hmoment", "z_hmoment:species_code"),
        ("helix_propensity", "z_helix", "z_helix:species_code"),
    ]
    for prop_name, main_term, interaction_term in specs:
        if main_term not in result.params.index or interaction_term not in result.params.index:
            rows.append(
                {
                    "property": prop_name,
                    "effect_e_coli": np.nan,
                    "CI_e_coli_lo": np.nan,
                    "CI_e_coli_hi": np.nan,
                    "effect_s_aureus": np.nan,
                    "CI_s_aureus_lo": np.nan,
                    "CI_s_aureus_hi": np.nan,
                    "interaction_p": np.nan,
                }
            )
            continue

        effect_ecoli, ci_lo_ecoli, ci_hi_ecoli = _linear_combo_ci(result, f"{main_term} = 0")
        effect_saureus, ci_lo_sa, ci_hi_sa = _linear_combo_ci(
            result, f"{main_term} + {interaction_term} = 0"
        )
        p_int = float(result.pvalues.get(interaction_term, np.nan))
        rows.append(
            {
                "property": prop_name,
                "effect_e_coli": effect_ecoli,
                "CI_e_coli_lo": ci_lo_ecoli,
                "CI_e_coli_hi": ci_hi_ecoli,
                "effect_s_aureus": effect_saureus,
                "CI_s_aureus_lo": ci_lo_sa,
                "CI_s_aureus_hi": ci_hi_sa,
                "interaction_p": p_int,
            }
        )
    return pd.DataFrame(rows)


def _prediction_grid(
    df: pd.DataFrame,
    property_col: str,
    z_col: str,
    n_grid: int = 50,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    grid = np.linspace(df[z_col].min(), df[z_col].max(), n_grid)
    other_means = {
        "z_charge": df["z_charge"].mean(),
        "z_hydrophobicity": df["z_hydrophobicity"].mean(),
        "z_length": df["z_length"].mean(),
        "z_helix": df["z_helix"].mean(),
        "z_hmoment": df["z_hmoment"].mean(),
    }
    return grid, other_means, {}


def plot_property_activity_lines(
    df: pd.DataFrame,
    result,
    property_col: str,
    z_col: str,
    out_png: Path,
    title: str,
) -> None:
    grid = np.linspace(df[z_col].min(), df[z_col].max(), 80)
    base = {
        "z_charge": df["z_charge"].mean(),
        "z_hydrophobicity": df["z_hydrophobicity"].mean(),
        "z_length": df["z_length"].mean(),
        "z_helix": df["z_helix"].mean(),
        "z_hmoment": df["z_hmoment"].mean(),
    }
    lines = []
    for species, code in [("e_coli", 0), ("s_aureus", 1)]:
        pred_rows = []
        for val in grid:
            row = dict(base)
            row[z_col] = val
            row["species_code"] = code
            pred_rows.append(row)
        pred_df = pd.DataFrame(pred_rows)
        pred = result.predict(pred_df)
        lines.append((species, grid, pred))

    plt.figure(figsize=(8, 5))
    for species, xvals, yvals in lines:
        plt.plot(xvals, yvals, label=species.replace("_", " "))
    sns.scatterplot(
        data=df,
        x=z_col,
        y="activity_score",
        hue="species",
        alpha=0.15,
        legend=False,
    )
    plt.xlabel(f"z-scored {property_col}")
    plt.ylabel("Predicted / observed activity score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--property-table", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    property_df = pd.read_csv(args.property_table)
    df = prepare_regression_df(property_df)
    result = fit_interaction_model(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effects = build_effects_table(result, df)
    effects.to_csv(args.output_dir / "species_property_activity_effects.csv", index=False)
    with (args.output_dir / "species_property_ols_summary.txt").open("w", encoding="utf-8") as f:
        f.write(result.summary().as_text())

    plot_property_activity_lines(
        df,
        result,
        "net_charge",
        "z_charge",
        args.output_dir / "charge_activity_by_species.png",
        "Activity vs net charge by species",
    )
    plot_property_activity_lines(
        df,
        result,
        "mean_hydrophobicity",
        "z_hydrophobicity",
        args.output_dir / "hydrophobicity_activity_by_species.png",
        "Activity vs mean hydrophobicity by species",
    )
    plot_property_activity_lines(
        df,
        result,
        "hydrophobic_moment",
        "z_hmoment",
        args.output_dir / "hydrophobic_moment_activity_by_species.png",
        "Activity vs hydrophobic moment by species",
    )
    plot_property_activity_lines(
        df,
        result,
        "helix_propensity",
        "z_helix",
        args.output_dir / "helix_propensity_activity_by_species.png",
        "Activity vs helix propensity by species",
    )

    print(f"Wrote species property effects -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
