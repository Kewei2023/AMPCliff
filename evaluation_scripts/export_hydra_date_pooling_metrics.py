#!/usr/bin/env python
# maintained by kewei li
# -*- coding: utf-8 -*-
"""
Extract train/valid/test metrics from flat Hydra day outputs
(e.g. outputs/2026-04-03/*_e_coli_esm2_t12_none_last).

Writes CSV + multi-sheet XLSX (by dataset × model), aligned with
outputs/ablation/local_spectral_anchor_grid_2x2_with_pooling_mean_max_attn.xlsx columns.

Usage:
  python evaluation_scripts/export_hydra_date_pooling_metrics.py \\
    --date-dir outputs/2026-04-03 \\
    --experiment-group 2026-04-03_new_poolings
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extract_ablation_metrics import (  # noqa: E402
    calculate_metrics_from_csv,
    parse_log_file,
)

# HH-MM-SS_dataset_model_apply_pooling — apply is one segment (e.g. none), not \w+
# (underscore is \w in Python, so \w+ would swallow none_latent from none_latent_attn).
_HYDRA_RUN_RE = re.compile(
    r"^(?P<time>\d{2}-\d{2}-\d{2})_"
    r"(?P<dataset>e_coli|s_aureus)_"
    r"(?P<model>esm2_t\d+)_"
    r"(?P<apply>[^_]+)_"
    r"(?P<pooling>.+)$"
)

# Columns in merged spectral/pooling baseline workbook
MERGED_COLUMNS = [
    "experiment_group",
    "experiment_dir",
    "experiment_type",
    "model",
    "dataset",
    "config_name",
    "diff",
    "pooling",
    "num_heads",
    "num_anchor",
    "analysis_dim",
    "stft_n_fft",
    "stft_hop_length",
    "run_slug_type",
    "train_pearson",
    "train_spearman",
    "train_recall",
    "valid_pearson",
    "valid_spearman",
    "valid_recall",
    "test_pearson",
    "test_spearman",
    "test_recall",
    "test_recall_topk",
    "stft_center",
    "use_phase",
    "gated",
    "k_value",
    "use_fft",
    "baseline_pooling",
]


def _sheet_name(dataset: str, model: str) -> str:
    raw = f"{dataset}_{model}"
    return raw[:31] if len(raw) > 31 else raw


def write_xlsx_by_dataset_model(df: pd.DataFrame, path: Path) -> None:
    """One sheet per (dataset, model); same as export_local_spectral_anchor_grid_metrics."""
    if df.empty:
        raise ValueError("DataFrame is empty; cannot write XLSX")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if "dataset" not in df.columns or "model" not in df.columns:
            df.to_excel(writer, sheet_name="all", index=False)
            return

        pairs = df[["dataset", "model"]].drop_duplicates()
        used_sheet_names: set = set()
        for _, row in pairs.iterrows():
            ds, m = row["dataset"], row["model"]
            if pd.isna(ds) or pd.isna(m):
                continue
            ds_s, m_s = str(ds).strip(), str(m).strip()
            if not ds_s or not m_s:
                continue
            sub = df[(df["dataset"] == ds) & (df["model"] == m)].copy()
            sort_cols = [c for c in ("experiment_type", "experiment_dir") if c in sub.columns]
            if sort_cols:
                sub = sub.sort_values(sort_cols)
            base = _sheet_name(ds_s, m_s)
            name = base
            n = 1
            while name in used_sheet_names:
                suffix = f"_{n}"
                name = (base[: max(0, 31 - len(suffix))] + suffix)[:31]
                n += 1
            used_sheet_names.add(name)
            sub.to_excel(writer, sheet_name=name, index=False)

        if not used_sheet_names:
            df.to_excel(writer, sheet_name="all", index=False)


_POOLING_HPARAM_KEYS = (
    "num_heads",
    "num_anchor",
    "analysis_dim",
    "stft_n_fft",
    "stft_hop_length",
    "stft_center",
    "use_phase",
    "gated",
    "use_fft",
)


def _read_pooling_hparams_from_hydra(run_dir: Path) -> Dict[str, Any]:
    """Resolve pooling hyper-params from nested YAML or legacy flat keys in .hydra/config.yaml."""
    if yaml is None:
        return {}
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        with open(cfg_path, encoding="utf-8", errors="ignore") as f:
            d = yaml.safe_load(f)
    except Exception:
        return {}
    mr = (d or {}).get("model", {}).get("regression", {}) or {}
    pooling = mr.get("pooling")
    po = mr.get("pooling_common") or {}
    pc = mr.get("pooling_config") or {}
    method: Dict[str, Any] = {}
    if isinstance(pc, dict) and pooling in pc and isinstance(pc[pooling], dict):
        method = pc[pooling]

    out: Dict[str, Any] = {}
    for key in _POOLING_HPARAM_KEYS:
        v = None
        if key in mr:
            v = mr[key]
        elif isinstance(po, dict) and key in po:
            v = po[key]
        elif isinstance(method, dict) and key in method:
            v = method[key]
        if v is not None:
            out[key] = v
    return out


def _read_diff_from_hydra(run_dir: Path):
    if yaml is None:
        return pd.NA
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return pd.NA
    try:
        with open(cfg_path, encoding="utf-8", errors="ignore") as f:
            d = yaml.safe_load(f)
        diff = (d or {}).get("data", {}).get("diff")
        if isinstance(diff, list) and diff:
            return int(diff[0])
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            return int(diff)
    except Exception:
        return pd.NA
    return pd.NA


def parse_run_dir_name(dirname: str) -> Optional[Dict[str, str]]:
    m = _HYDRA_RUN_RE.match(dirname)
    if not m:
        return None
    return {
        "time": m.group("time"),
        "dataset": m.group("dataset"),
        "model": m.group("model"),
        "apply": m.group("apply"),
        "pooling": m.group("pooling"),
    }


def _metrics_from_csv_split(run_dir: Path, glob_pat: str, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    files = sorted(run_dir.glob(glob_pat))
    if not files:
        return out
    r = calculate_metrics_from_csv(files[0])
    if not r:
        return out
    out[f"{prefix}_pearson"] = r["pearson"]
    out[f"{prefix}_spearman"] = r["spearman"]
    out[f"{prefix}_recall"] = r["recall"]
    if prefix == "test":
        out["test_recall_topk"] = r["recall_topk"]
    return out


def extract_one_run(
    date_dir: Path,
    run_dir: Path,
    experiment_group: str,
    experiment_type: str,
    prefer_log_train_valid: bool,
) -> Optional[Dict[str, Any]]:
    name = run_dir.name
    parsed = parse_run_dir_name(name)
    if not parsed:
        return None
    log_path = run_dir / "downstream_train.log"
    if not log_path.is_file():
        return None

    row: Dict[str, Any] = {c: pd.NA for c in MERGED_COLUMNS}
    row["experiment_group"] = experiment_group
    row["experiment_dir"] = name
    row["experiment_type"] = experiment_type
    row["model"] = parsed["model"]
    row["dataset"] = parsed["dataset"]
    row["config_name"] = name
    row["pooling"] = parsed["pooling"]
    row["baseline_pooling"] = parsed["pooling"]

    for k, v in _read_pooling_hparams_from_hydra(run_dir).items():
        if k in row:
            row[k] = v

    diff_m = re.search(r"diff(\d+)", name)
    row["diff"] = int(diff_m.group(1)) if diff_m else _read_diff_from_hydra(run_dir)

    log_m = parse_log_file(log_path)
    if prefer_log_train_valid:
        for k, v in log_m.items():
            if v is not None:
                row[k] = v

    test_csvs = list(run_dir.glob("*-test_result.csv"))
    if test_csvs:
        test_metrics = calculate_metrics_from_csv(test_csvs[0])
        if test_metrics:
            row["test_pearson"] = test_metrics["pearson"]
            row["test_spearman"] = test_metrics["spearman"]
            row["test_recall"] = test_metrics["recall"]
            row["test_recall_topk"] = test_metrics["recall_topk"]

    if not prefer_log_train_valid:
        row.update(_metrics_from_csv_split(run_dir, "*-train_result.csv", "train"))
        row.update(_metrics_from_csv_split(run_dir, "*-valid_result.csv", "valid"))

    return row


def collect_metrics(
    date_dir: Path,
    experiment_group: str,
    experiment_type: str,
    prefer_log_train_valid: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not date_dir.is_dir():
        return rows
    for child in sorted(date_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        r = extract_one_run(
            date_dir, child, experiment_group, experiment_type, prefer_log_train_valid
        )
        if r:
            rows.append(r)
    return rows


def dataframe_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=MERGED_COLUMNS)
    df = pd.DataFrame(rows)
    for c in MERGED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[MERGED_COLUMNS]


def union_cols(*dfs: pd.DataFrame) -> List[str]:
    seen: set = set()
    order: List[str] = []
    for df in dfs:
        for c in df.columns:
            if c not in seen:
                seen.add(c)
                order.append(c)
    return order


def _sheet_to_dataset_model(sheet: str) -> tuple[str, str]:
    """Map workbook sheet name (e.g. e_coli_esm2_t12) to (dataset, model)."""
    if sheet.startswith("e_coli_"):
        return "e_coli", sheet[len("e_coli_") :]
    if sheet.startswith("s_aureus_"):
        return "s_aureus", sheet[len("s_aureus_") :]
    return "", ""


def merge_xlsx_sheets(
    baseline_path: Path,
    new_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Append new_df rows per (dataset, model) sheet; union columns with baseline."""
    xl = pd.ExcelFile(baseline_path, engine="openpyxl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet in xl.sheet_names:
            base = pd.read_excel(baseline_path, sheet_name=sheet, engine="openpyxl")
            sub = pd.DataFrame()
            if not new_df.empty and "dataset" in new_df.columns and "model" in new_df.columns:
                ds, m = _sheet_to_dataset_model(sheet)
                if ds and m:
                    sub = new_df[
                        (new_df["dataset"].astype(str) == ds)
                        & (new_df["model"].astype(str) == m)
                    ].copy()

            cols = union_cols(base, sub)
            parts = [base.reindex(columns=cols)]
            if not sub.empty:
                parts.append(sub.reindex(columns=cols))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                merged = pd.concat(parts, ignore_index=True)
            merged.to_excel(writer, sheet_name=sheet[:31], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Hydra day-dir pooling metrics and optionally merge with baseline XLSX.",
    )
    parser.add_argument(
        "--date-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "2026-04-03",
        help="Directory containing HH-MM-SS_* run folders",
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        default=None,
        help="Label column experiment_group (default: <date-dir name>)",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="llm_readout_pooling",
        help="experiment_type for new rows",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV (default: <date-dir>/new_poolings_metrics.csv)",
    )
    parser.add_argument(
        "--xlsx-out",
        type=Path,
        default=None,
        help="Output XLSX for new rows only (default: <date-dir>/new_poolings_metrics.xlsx)",
    )
    parser.add_argument(
        "--baseline-xlsx",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "ablation"
        / "local_spectral_anchor_grid_2x2_with_pooling_mean_max_attn.xlsx",
        help="Existing merged workbook to concatenate into",
    )
    parser.add_argument(
        "--merged-xlsx-out",
        type=Path,
        default=None,
        help="If set, write baseline sheets + new rows to this path",
    )
    parser.add_argument(
        "--csv-only-from-csv",
        action="store_true",
        help="Compute train/valid from *-train/valid_result.csv instead of log",
    )
    args = parser.parse_args()

    date_dir = args.date_dir.expanduser().resolve()
    exp_group = args.experiment_group or date_dir.name

    rows = collect_metrics(
        date_dir,
        exp_group,
        args.experiment_type,
        prefer_log_train_valid=not args.csv_only_from_csv,
    )
    df = dataframe_from_rows(rows)

    csv_out = (
        args.csv_out.expanduser().resolve()
        if args.csv_out
        else date_dir / "new_poolings_metrics.csv"
    )
    xlsx_out = (
        args.xlsx_out.expanduser().resolve()
        if args.xlsx_out
        else date_dir / "new_poolings_metrics.xlsx"
    )

    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out} ({len(df)} rows)")

    try:
        write_xlsx_by_dataset_model(df, xlsx_out)
        print(f"Wrote {xlsx_out}")
    except Exception as e:
        print(f"XLSX write failed: {e}")

    merged_out = args.merged_xlsx_out
    if merged_out is None:
        merged_out = date_dir / "local_spectral_anchor_grid_2x2_with_pooling_mean_max_attn_merged.xlsx"

    merged_out = merged_out.expanduser().resolve()
    baseline = args.baseline_xlsx.expanduser().resolve()
    if baseline.is_file() and not df.empty:
        merge_xlsx_sheets(baseline, df, merged_out)
        print(f"Wrote merged: {merged_out}")
    elif not baseline.is_file():
        print(f"Baseline XLSX not found (skip merge): {baseline}")
    else:
        print("No rows extracted; skip merge.")


if __name__ == "__main__":
    main()
