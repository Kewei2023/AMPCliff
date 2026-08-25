#!/usr/bin/env python3
# maintained by kewei li
"""Plot AMP knockout diagnostic figures: 4 poolings x 5 peptides per model x dataset."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

POOLING_ORDER = ["mean", "max", "attn_structured", "FLaG"]
POOLING_LABELS = {
    "mean": "Mean",
    "max": "Max",
    "attn_structured": "attn",
    "FLaG": "FFT-LAG",
}
PEPTIDE_COLORS = plt.cm.tab10.colors[:5]


def _parse_group_name(name: str):
    m = re.match(r"^(esm2_t6|esm2_t12)_(e_coli|s_aureus)_diff(\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _dataset_display(ds: str) -> str:
    return "E. coli" if ds == "e_coli" else "S. aureus"


def _model_display(mv: str) -> str:
    return mv.replace("esm2_t6", "ESM2-8M").replace("esm2_t12", "ESM2-35M")


def _peptide_label(row: pd.DataFrame, idx) -> str:
    sub = row[row["idx"] == idx]
    if sub.empty:
        return f"idx={idx}"
    pep = str(sub["peptide"].iloc[0])
    if len(pep) > 12:
        pep = pep[:10] + "…"
    return f"idx={idx} ({pep})"


def plot_one_group(summary: pd.DataFrame, group_name: str, out_dir: Path, layout: str) -> None:
    parsed = _parse_group_name(group_name)
    if parsed is None:
        print(f"[SKIP] unrecognized group name: {group_name}")
        return
    mv, ds, _diff = parsed

    poolings = [p for p in POOLING_ORDER if p in summary["pooling"].unique()]
    if not poolings:
        print(f"[SKIP] no known poolings in {group_name}")
        return

    idx_order = sorted(summary["idx"].unique(), key=lambda x: (str(type(x)), x))

    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))
        axes_flat = axes.flatten()
    else:
        fig, axes_flat = plt.subplots(1, len(poolings), sharex=True, sharey=True, figsize=(14, 3.8))
        if len(poolings) == 1:
            axes_flat = [axes_flat]

    for j, pooling in enumerate(poolings):
        ax = axes_flat[j]
        sub_pool = summary[summary["pooling"] == pooling]
        for i, idx in enumerate(idx_order):
            sub = sub_pool[sub_pool["idx"] == idx].sort_values("rel_pos_center")
            if sub.empty:
                continue
            x = sub["rel_pos_center"].values
            mean = sub["mean_abs_delta"].values
            std = sub["std_across_seeds"].values
            c = PEPTIDE_COLORS[i % len(PEPTIDE_COLORS)]
            label = _peptide_label(sub_pool, idx)
            ax.plot(x, mean, color=c, linewidth=1.5, alpha=0.92, label=label)
            if (sub["n_seeds"].max() if "n_seeds" in sub.columns else 2) >= 2:
                ax.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=c,
                    alpha=0.22,
                    linewidth=0,
                )
        ax.set_title(POOLING_LABELS.get(pooling, pooling))
        ax.set_xlabel("Relative residue position")
        if j == 0:
            ax.set_ylabel("|Δ prediction|")
        ax.grid(True, alpha=0.25)

    # hide unused axes in 2x2 if fewer than 4 poolings
    for k in range(len(poolings), len(axes_flat)):
        axes_flat[k].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False)
    fig.suptitle(
        f"AMP knockout (last-layer HS) — {_model_display(mv)} / {_dataset_display(ds)}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"amp_knockout_{mv}_{ds}"
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / stem}.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "outputs/ablation_new_data/_amp_knockout_seed_runs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs/ablation_new_data/_amp_knockout_figures",
    )
    ap.add_argument("--layout", choices=("1x4", "2x2"), default="1x4")
    args = ap.parse_args()

    root = args.input_root.expanduser().resolve()
    for gdir in sorted(root.iterdir()):
        if not gdir.is_dir() or gdir.name.startswith("_"):
            continue
        summary_path = gdir / "_aggregated" / f"summary_{gdir.name}.csv"
        if not summary_path.is_file():
            print(f"[SKIP] missing {summary_path}")
            continue
        summary = pd.read_csv(summary_path)
        plot_one_group(summary, gdir.name, args.out_dir.expanduser().resolve(), args.layout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
