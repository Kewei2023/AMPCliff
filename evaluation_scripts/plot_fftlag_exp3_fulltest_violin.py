#!/usr/bin/env python3
# maintained by kewei li
"""Plot Exp3 full-test token knockout |Δ prediction| distributions as grouped violin plots."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
VALUE_COL = "response_std"
Y_LABEL = "Per-peptide response std (|Δ prediction|)"

POOLING_ORDER = [
    "mean",
    "max",
    "attn",
    "last",
    "swe_ot",
    "mltp",
    "latent_attn",
    "fft_latent_attn_gate",
]

POOLING_LABELS = {
    "mean": "mean",
    "max": "max",
    "attn": "attn",
    "last": "last",
    "swe_ot": "SWE-OT",
    "mltp": "MLTP",
    "latent_attn": "latent-Attn",
    "fft_latent_attn_gate": "FLaG",
}

# ESM2-8M test Spearman mean ranks (1 = highest) from ablation_new_data table.
SPEARMAN_RANK_ESM2_T6: Dict[str, Dict[str, int]] = {
    "e_coli": {
        "fft_latent_attn_gate": 1,
        "last": 2,
        "mean": 3,
        "latent_attn": 4,
        "max": 5,
        "swe_ot": 6,
        "attn": 7,
        "mltp": 8,
    },
    "s_aureus": {
        "fft_latent_attn_gate": 1,
        "max": 2,
        "last": 3,
        "swe_ot": 4,
        "mean": 5,
        "latent_attn": 6,
        "attn": 7,
        "mltp": 8,
    },
}


def _dataset_display(ds: str) -> str:
    return "E. coli" if ds == "e_coli" else "S. aureus"


def _prepare_plot_df(long_df: pd.DataFrame) -> pd.DataFrame:
    work = long_df.copy()
    present = [p for p in POOLING_ORDER if p in work["pooling"].unique()]
    labels = [POOLING_LABELS.get(p, p) for p in present]
    work["pooling_label"] = pd.Categorical(
        work["pooling"].map(lambda p: POOLING_LABELS.get(p, p)),
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
    """Annotate each violin with red circled Spearman-mean rank (ESM2-8M)."""
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


def plot_violin_one_dataset(
    long_df: pd.DataFrame,
    out_png: Path,
    *,
    dataset: str = "",
    title: Optional[str] = None,
) -> None:
    if long_df.empty:
        print(f"[SKIP] empty df for {out_png}")
        return

    work = _prepare_plot_df(long_df)
    labels = list(work["pooling_label"].cat.categories)
    n_pool = len(labels)
    fig_w = max(6, min(20, n_pool * 1.4))
    fig, ax = plt.subplots(figsize=(fig_w, 5))

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

    ds_label = _dataset_display(dataset) if dataset else ""
    if title is None:
        title = f"Token knockout response std — {ds_label}"
    ax.set_title(title)
    ax.set_xlabel("Pooling")
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


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
        title = "Token knockout per-peptide response std by pooling"

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
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
        ax.set_title(_dataset_display(dataset))
        ax.set_xlabel("Pooling")
        ax.set_ylabel(Y_LABEL)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def run_all(
    analysis_root: Path,
    datasets: Iterable[str],
    *,
    force: bool = False,
) -> int:
    long_dfs: Dict[str, pd.DataFrame] = {}
    ok = True

    for dataset in datasets:
        out_dir = exp_agg_dir(analysis_root, dataset, "exp3")
        long_csv = out_dir / "token_knockout_abs_delta_long.csv"
        violin_png = out_dir / "token_knockout_violinplot.png"

        if not long_csv.is_file():
            print(f"[FAIL] missing {long_csv} — run aggregate_fftlag_exp3_fulltest.py first")
            ok = False
            continue

        long_df = pd.read_csv(long_csv)
        long_dfs[dataset] = long_df

        if violin_png.is_file() and not force:
            print(f"[SKIP] {violin_png}")
            continue

        plot_violin_one_dataset(long_df, violin_png, dataset=dataset)

    if len(long_dfs) >= 1:
        combined_png = analysis_root / "aggregated" / "exp3_token_knockout_violinplot_combined.png"
        if combined_png.is_file() and not force and len(long_dfs) < 2:
            pass
        elif not combined_png.is_file() or force or len(long_dfs) >= 2:
            plot_violin_combined(long_dfs, combined_png)

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs/analysis/fftlag_mechanism",
    )
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.analysis_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: analysis root not found: {root}", file=sys.stderr)
        return 1

    return run_all(root, args.datasets, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
