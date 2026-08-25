#!/usr/bin/env python3
"""Plot Exp3 revised-pooling token knockout delta|MSE| violin (no swe_ot, MLTP/attn labels)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_EVAL_SCRIPTS = Path(__file__).resolve().parent
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

from fftlag_aggregated_paths import exp_agg_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = ("e_coli", "s_aureus")
LONG_CSV_NAME = "token_knockout_mse_diff_long.csv"
VALUE_COL = "response_std"
Y_LABEL = "Per-peptide response std (delta |P|)"
DEFAULT_TITLE = "Token knockout per-peptide response std (delta |P|) by pooling"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/analysis/fftlag_mechanism/aggregated/"
    "exp3_token_knockout_mse_diff_violinplot_combined_no_swe_ot.png"
)

# Data keys in aggregated CSV (revised poolings replace legacy attn/mltp).
POOLING_ORDER = [
    "mean",
    "max",
    "attn_structured",
    "last",
    "mltp_paper",
    "latent_attn",
    "fft_latent_attn_gate",
]

# X-axis labels match the legacy combined figure.
POOLING_LABELS = {
    "mean": "mean",
    "max": "max",
    "attn_structured": "attn",
    "last": "last",
    "mltp_paper": "MLTP",
    "latent_attn": "latent-Attn",
    "fft_latent_attn_gate": "FLaG",
}

# Spearman ranks among the 7 plotted poolings (esm2_t6 test, revised poolings).
SPEARMAN_RANK_ESM2_T6: Dict[str, Dict[str, int]] = {
    "e_coli": {
        "fft_latent_attn_gate": 1,
        "last": 2,
        "mean": 3,
        "latent_attn": 4,
        "max": 5,
        "attn_structured": 6,
        "mltp_paper": 7,
    },
    "s_aureus": {
        "fft_latent_attn_gate": 1,
        "max": 2,
        "last": 3,
        "mean": 4,
        "latent_attn": 5,
        "mltp_paper": 6,
        "attn_structured": 7,
    },
}

TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 14


def _dataset_display(ds: str) -> str:
    return r"$\it{E.\ coli}$" if ds == "e_coli" else r"$\it{S.\ aureus}$"


def _savefig_png_svg(fig: plt.Figure, out_png: Path) -> None:
    out_png = Path(out_png)
    out_svg = out_png.with_suffix(".svg")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_svg}")


def _prepare_plot_df(long_df: pd.DataFrame) -> pd.DataFrame:
    work = long_df.copy()
    want = set(POOLING_ORDER)
    work = work[work["pooling"].isin(want)].copy()
    present = [p for p in POOLING_ORDER if p in work["pooling"].unique()]
    labels = [POOLING_LABELS[p] for p in present]
    work["pooling_label"] = pd.Categorical(
        work["pooling"].map(POOLING_LABELS),
        categories=labels,
        ordered=True,
    )
    return work


def _add_spearman_rank_badges(
    ax: plt.Axes,
    work: pd.DataFrame,
    dataset: str,
    labels: List[str],
) -> None:
    ranks_by_pool = SPEARMAN_RANK_ESM2_T6.get(dataset, {})
    if not ranks_by_pool:
        return

    label_to_pool = {v: k for k, v in POOLING_LABELS.items()}
    y_max = float(work[VALUE_COL].max())
    y_pad = max(y_max * 0.05, 0.001)

    for i, label in enumerate(labels):
        pool = label_to_pool.get(str(label), "")
        rank = ranks_by_pool.get(pool)
        if rank is None:
            continue
        sub = work.loc[work["pooling_label"] == label, VALUE_COL]
        if sub.empty:
            continue
        sub_max = float(sub.max())
        ax.annotate(
            str(rank),
            xy=(i, sub_max + y_pad),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="red",
            bbox=dict(
                boxstyle="circle,pad=0.28",
                facecolor="white",
                edgecolor="red",
                linewidth=1.2,
            ),
        )

    ax.set_ylim(top=y_max + y_pad * 3.0)


def plot_violin_combined(
    long_dfs: Dict[str, pd.DataFrame],
    out_png: Path,
    *,
    title: Optional[str] = None,
) -> None:
    datasets = [ds for ds, df in long_dfs.items() if df is not None and not df.empty]
    if not datasets:
        print(f"[SKIP] no data for combined plot: {out_png}")
        return

    if title is None:
        title = DEFAULT_TITLE

    fig, axes = plt.subplots(1, len(datasets), figsize=(9 * len(datasets), 5), squeeze=False)
    for i, (ax, dataset) in enumerate(zip(axes[0], datasets)):
        work = _prepare_plot_df(long_dfs[dataset])
        labels = list(work["pooling_label"].cat.categories)
        sns.violinplot(
            data=work,
            x="pooling_label",
            y=VALUE_COL,
            order=labels,
            inner="box",
            cut=0,
            linewidth=0.8,
            ax=ax,
            color="steelblue",
        )
        _add_spearman_rank_badges(ax, work, dataset, labels)
        ax.set_title(_dataset_display(dataset), fontsize=TITLE_FONTSIZE, color="black")
        ax.set_xlabel("Pooling", fontsize=AXIS_LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(title, y=1.02, fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    _savefig_png_svg(fig, out_png)
    plt.close(fig)


def run_all(
    analysis_root: Path,
    datasets: tuple[str, ...],
    *,
    output: Path,
    force: bool = False,
) -> int:
    if output.is_file() and not force:
        print(f"[SKIP] {output}")
        return 0

    long_dfs: Dict[str, pd.DataFrame] = {}
    ok = True
    for dataset in datasets:
        long_csv = exp_agg_dir(analysis_root, dataset, "exp3") / LONG_CSV_NAME
        if not long_csv.is_file():
            print(f"[FAIL] missing {long_csv}", file=sys.stderr)
            print(
                "Run: python evaluation_scripts/aggregate_exp3_token_knockout_mse_diff.py --force",
                file=sys.stderr,
            )
            ok = False
            continue
        long_dfs[dataset] = pd.read_csv(long_csv)

    if not ok or not long_dfs:
        return 1

    plot_violin_combined(long_dfs, output)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    return run_all(
        root,
        tuple(args.datasets),
        output=args.output.expanduser().resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
