# maintained by kewei li
"""Shared physicochemical property utilities for DC validation experiments."""
from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_PARENT not in sys.path:
    sys.path.append(_REPO_PARENT)

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from AMPCliff.interpretations.extract_features import approx_net_charge, estimate_pI

CF_HELIX = set("EALMQKRH")
HYDROPHILIC = set("DEKRHNQST")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# Matches features/AAComposition.AALetter order for 20-dim composition vectors.
AAC_AA_ORDER = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

KYTE_DOOLITTLE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

PROPERTY_COLUMNS = [
    "idx",
    "sequence",
    "length",
    "net_charge",
    "charge_density",
    "pI",
    "mean_hydrophobicity",
    "hydrophilic_fraction",
    "helix_propensity",
    "hydrophobic_moment",
    "activity_score",
    "species",
    "split",
]

PROBE_PROPERTIES = [
    "net_charge",
    "charge_density",
    "pI",
    "mean_hydrophobicity",
    "hydrophilic_fraction",
    "helix_propensity",
    "hydrophobic_moment",
]

COEFFICIENT_KEYS = {
    0: "coeff_0_norm",
    1: "coeff_1",
    2: "coeff_2",
    3: "coeff_3",
}

DCT_PROBE_COEFFICIENTS = (0, 1, 2, 3)

SPECIES_DISPLAY_NAMES = {
    "e_coli": "E. coli",
    "s_aureus": "S. aureus",
}

GATE_EFFECT_TASK_NAME = "Gate effect by bucket"


def mathfrak_b_label(i: int) -> str:
    """Display label for DCT coefficient / band index (Fraktur B with subscript)."""
    return rf"$\mathfrak{{B}}_{{{int(i)}}}$"


def mathfrak_b_norm_label(i: int) -> str:
    """Fraktur B with tilde (length-normalized DC, e.g. coeff_0_norm)."""
    return rf"$\tilde{{\mathfrak{{B}}}}_{{{int(i)}}}$"


def coefficient_label_plain(i: int) -> str:
    """Plain-text coefficient label for summary tables (no LaTeX)."""
    if int(i) == 0:
        return "B~0"
    return f"B{int(i)}"


def format_mean_pm_std(mean: float, std: float, decimals: int = 3) -> str:
    if pd.isna(mean) or pd.isna(std):
        return ""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_ci_range(lo_mean: float, hi_mean: float, decimals: int = 3) -> str:
    if pd.isna(lo_mean) or pd.isna(hi_mean):
        return ""
    return f"[{lo_mean:.{decimals}f}, {hi_mean:.{decimals}f}]"


def species_display_name(species: str | None) -> str:
    if not species:
        return ""
    return SPECIES_DISPLAY_NAMES.get(species, species.replace("_", " "))


def infer_species_from_path(path: Path | str) -> str | None:
    for part in Path(path).parts:
        if part in SPECIES_DISPLAY_NAMES:
            return part
    return None


def band_sensitivity_task_name(xlabel: str) -> str:
    """Task label for combined band-sensitivity bar/line charts."""
    if xlabel.endswith(" bucket"):
        prefix = xlabel[: -len(" bucket")]
        return f"Band sensitivity by {prefix.lower()} bucket"
    return f"Band sensitivity by {xlabel.lower()}"


def left_aligned_bucket_chart_title(species: str | None, task_name: str) -> str:
    label = species_display_name(species)
    if label:
        return f"{label} | {task_name}"
    return task_name

BUCKET_ORDER_RAW = ["bottom_30", "middle_40", "top_30"]
BUCKET_LABELS = {
    "bottom_30": "bottom 30%",
    "middle_40": "middle 40%",
    "top_30": "top 30%",
}


def compute_aac_vector(seq: str) -> np.ndarray:
    """Return 20-dim amino-acid composition fractions in AAC_AA_ORDER."""
    seq = str(seq).strip().upper()
    length = len(seq)
    if length == 0:
        return np.zeros(len(AAC_AA_ORDER), dtype=np.float64)
    counts = {aa: 0 for aa in AAC_AA_ORDER}
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
    return np.array([counts[aa] / length for aa in AAC_AA_ORDER], dtype=np.float64)


def helix_propensity(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in CF_HELIX) / len(seq)


def hydrophobic_moment(seq: str, delta: float = 100.0) -> float:
    if not seq:
        return 0.0
    angles = [math.radians(delta * i) for i in range(len(seq))]
    hx = hy = 0.0
    for a, ang in zip(seq, angles):
        h = KYTE_DOOLITTLE.get(a, 0.0)
        hx += h * math.cos(ang)
        hy += h * math.sin(ang)
    return math.sqrt(hx * hx + hy * hy) / len(seq)


def mean_hydrophobicity(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(a, 0.0) for a in seq) / len(seq)


def hydrophilic_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for a in seq if a in HYDROPHILIC) / len(seq)


def biopython_net_charge(seq: str, ph: float = 7.0) -> float:
    if not seq:
        return 0.0
    return float(ProteinAnalysis(seq).charge_at_pH(ph))


def biopython_pI(seq: str) -> float:
    if not seq:
        return 0.0
    return float(ProteinAnalysis(seq).isoelectric_point())


def compute_properties_for_sequence(seq: str) -> Dict[str, float]:
    seq = str(seq).strip().upper()
    length = len(seq)
    net_charge = biopython_net_charge(seq)
    return {
        "length": float(length),
        "net_charge": net_charge,
        "charge_density": net_charge / length if length else 0.0,
        "pI": biopython_pI(seq),
        "mean_hydrophobicity": mean_hydrophobicity(seq),
        "hydrophilic_fraction": hydrophilic_fraction(seq),
        "helix_propensity": helix_propensity(seq),
        "hydrophobic_moment": hydrophobic_moment(seq),
    }


def compute_charge_pI_comparison(seq: str) -> Dict[str, float]:
    seq = str(seq).strip().upper()
    return {
        "net_charge_biopython": biopython_net_charge(seq),
        "net_charge_approx": approx_net_charge(seq),
        "pI_biopython": biopython_pI(seq),
        "pI_approx": estimate_pI(seq),
    }


def normalize_activity_score(df: pd.DataFrame) -> pd.Series:
    if "activity_score" in df.columns:
        return pd.to_numeric(df["activity_score"], errors="coerce")
    if "Activity" in df.columns:
        return pd.to_numeric(df["Activity"], errors="coerce")
    if "value" in df.columns:
        return -pd.to_numeric(df["value"], errors="coerce")
    raise ValueError("Need Activity, activity_score, or value column for activity_score")


def validate_sequences(sequences: Iterable[str]) -> Tuple[List[str], List[str]]:
    invalid = []
    for seq in sequences:
        seq = str(seq).strip().upper()
        bad = sorted(set(re.findall(r"[^A-Z]", seq)) | {c for c in seq if c not in STANDARD_AA})
        if bad:
            invalid.append(f"{seq}: {''.join(bad)}")
    return invalid, list(STANDARD_AA)


def assign_property_buckets(scores: pd.Series) -> pd.Series:
    n = len(scores)
    if n == 0:
        return pd.Series(dtype=str)
    order = scores.sort_values().index.tolist()
    n_low = max(1, int(round(n * 0.3)))
    n_high = max(1, int(round(n * 0.3)))
    n_mid = max(0, n - n_low - n_high)
    buckets: Dict[int, str] = {}
    for i, idx in enumerate(order):
        if i < n_low:
            buckets[idx] = "bottom_30"
        elif i < n_low + n_mid:
            buckets[idx] = "middle_40"
        else:
            buckets[idx] = "top_30"
    return scores.index.to_series().map(buckets)


def assign_property_buckets_by_species(
    df: pd.DataFrame,
    property_col: str,
    species_col: str = "species",
) -> pd.Series:
    buckets = pd.Series(index=df.index, dtype="object")
    for species, sub in df.groupby(species_col):
        buckets.loc[sub.index] = assign_property_buckets(sub[property_col])
    return buckets


def load_species_csvs(
    data_dir: Path,
    species: str,
    splits: Sequence[str] = ("train", "valid", "test"),
) -> pd.DataFrame:
    frames = []
    for split in splits:
        path = data_dir / f"grampa_{species}_7_25-{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing data file: {path}")
        part = pd.read_csv(path)
        part["split"] = split
        part["species"] = species
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Sequence"], keep="first")
    return out


def build_property_table_from_dirs(
    e_coli_data_dir: Path,
    s_aureus_data_dir: Path,
) -> pd.DataFrame:
    frames = [
        load_species_csvs(e_coli_data_dir, "e_coli"),
        load_species_csvs(s_aureus_data_dir, "s_aureus"),
    ]
    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for _, row in raw.iterrows():
        seq = str(row["Sequence"]).strip().upper()
        props = compute_properties_for_sequence(seq)
        rows.append(
            {
                "idx": int(row["Idx"]),
                "sequence": seq,
                "length": int(props["length"]),
                "net_charge": props["net_charge"],
                "charge_density": props["charge_density"],
                "pI": props["pI"],
                "mean_hydrophobicity": props["mean_hydrophobicity"],
                "hydrophilic_fraction": props["hydrophilic_fraction"],
                "helix_propensity": props["helix_propensity"],
                "hydrophobic_moment": props["hydrophobic_moment"],
                "activity_score": float(normalize_activity_score(pd.DataFrame([row])).iloc[0]),
                "species": row["species"],
                "split": row["split"],
            }
        )
    return pd.DataFrame(rows)[PROPERTY_COLUMNS]


def summarize_property_table_qc(df: pd.DataFrame) -> Dict[str, object]:
    invalid, _ = validate_sequences(df["sequence"].tolist())
    dup_idx = int(df["idx"].duplicated().sum())
    dup_seq = int(df["sequence"].duplicated().sum())
    return {
        "n_rows": int(len(df)),
        "n_species": sorted(df["species"].unique().tolist()),
        "n_splits": sorted(df["split"].unique().tolist()),
        "duplicate_idx_count": dup_idx,
        "duplicate_sequence_count": dup_seq,
        "invalid_sequence_count": len(invalid),
        "invalid_sequences_sample": invalid[:20],
        "activity_score_min": float(df["activity_score"].min()),
        "activity_score_max": float(df["activity_score"].max()),
        "activity_score_mean": float(df["activity_score"].mean()),
    }


def build_charge_pI_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cmp = compute_charge_pI_comparison(row["sequence"])
        rows.append(
            {
                "idx": int(row["idx"]),
                "sequence": row["sequence"],
                "species": row["species"],
                **cmp,
                "net_charge_abs_diff": abs(cmp["net_charge_biopython"] - cmp["net_charge_approx"]),
                "pI_abs_diff": abs(cmp["pI_biopython"] - cmp["pI_approx"]),
            }
        )
    out = pd.DataFrame(rows)
    return out


def charge_pI_validation_summary(validation_df: pd.DataFrame) -> Dict[str, float]:
    from scipy.stats import spearmanr

    return {
        "net_charge_max_abs_diff": float(validation_df["net_charge_abs_diff"].max()),
        "net_charge_mean_abs_diff": float(validation_df["net_charge_abs_diff"].mean()),
        "pI_max_abs_diff": float(validation_df["pI_abs_diff"].max()),
        "pI_mean_abs_diff": float(validation_df["pI_abs_diff"].mean()),
        "net_charge_spearman": float(
            spearmanr(validation_df["net_charge_biopython"], validation_df["net_charge_approx"]).correlation
        ),
        "pI_spearman": float(spearmanr(validation_df["pI_biopython"], validation_df["pI_approx"]).correlation),
    }


def exp1_knockout_idx(exp1_csv: Path) -> set[int]:
    """Unique peptide Idx present in aggregated Exp1 band-knockout results."""
    df = pd.read_csv(exp1_csv)
    if "idx" not in df.columns:
        raise ValueError(f"Column 'idx' not found in {exp1_csv}")
    return set(df["idx"].astype(int).unique())


def load_knockout_property_proxy_df(
    property_table: Path,
    exp1_csv: Path,
    *,
    manifest_path: Optional[Path] = None,
    species: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build property proxy aligned to Exp1 knockout peptides (source of truth).

    When the on-disk manifest JSON disagrees with Exp1 aggregated idx (e.g. manifest
    was regenerated after Exp1 ran), Exp1 idx wins so property-stratified knockout
    uses the same peptides as the band-knockout experiment.
    """
    props = pd.read_csv(property_table)
    exp1_idx = exp1_knockout_idx(exp1_csv)

    manifest_idx: set[int] = set()
    if manifest_path is not None and manifest_path.is_file():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_idx = {int(x) for x in payload["idx_list"]}

    overlap = exp1_idx & manifest_idx if manifest_idx else exp1_idx
    use_exp1_as_truth = (not manifest_idx) or (len(overlap) < len(exp1_idx))
    source = "exp1_aggregated" if use_exp1_as_truth else "manifest"

    proxy = props[props["idx"].astype(int).isin(exp1_idx if use_exp1_as_truth else manifest_idx)].copy()
    if species:
        proxy = proxy[proxy["species"] == species].copy()

    missing = exp1_idx - set(proxy["idx"].astype(int))
    meta = {
        "source": source,
        "exp1_n": len(exp1_idx),
        "manifest_n": len(manifest_idx),
        "overlap_n": len(overlap),
        "proxy_n": len(proxy),
        "missing_in_property_table": sorted(missing),
        "manifest_mismatch": bool(manifest_idx and len(overlap) < len(exp1_idx)),
    }
    if missing:
        raise ValueError(
            f"Exp1 knockout idx missing from property table: {sorted(missing)[:10]}"
            + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        )
    if meta["manifest_mismatch"]:
        from AMPCliff.utils.std_logger import Logger

        Logger.info(
            "Manifest idx mismatch with Exp1 aggregated (%d/%d overlap); "
            "using Exp1 idx as knockout peptide set",
            len(overlap),
            len(exp1_idx),
        )
    return proxy, meta
