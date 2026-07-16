"""Shared color and layout constants for DC validation composite figures."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# Heatmap
heatmap_cmap = "cividis"
heatmap_vmin = 0.55
heatmap_vmax = 1.00

# Probe bar chart (AAC, B0, B1, optional B2)
probe_palette = ["#7A7A7A", "#0072B2", "#009E73", "#E69F00"]

# Small-multiple background bars
small_bar_color = "#D9DEE5"
small_bar_edge = "#A6AFB8"

# Ordered band line colors (B0 .. B7)
band_colors = plt.cm.cividis(np.linspace(0.10, 0.90, 8))

# Layout
width_ratios = [1.05, 1.45, 1.75]
figure_facecolor = "#000000"
subplot_facecolor = "#FFFFFF"

text_color = "#222222"
axis_color = "#333333"

PANEL_LABELS = ("A", "B", "C", "D", "E", "F")
SPECIES_ROW_LABELS = {"e_coli": "E. coli", "s_aureus": "S. aureus"}


def band_color_map(band_labels: Sequence[str] | None = None) -> dict[str, tuple]:
    labels = list(band_labels) if band_labels is not None else [rf"$\mathfrak{{B}}_{{{i}}}$" for i in range(8)]
    return {label: band_colors[i] for i, label in enumerate(labels[: len(band_colors)])}


def probe_color_map(probe_labels: Sequence[str]) -> dict[str, str]:
    return {label: probe_palette[i % len(probe_palette)] for i, label in enumerate(probe_labels)}


def add_panel_label(ax: Axes, label: str, *, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="white",
    )


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(subplot_facecolor)
    ax.tick_params(colors=axis_color, labelsize=7)
    ax.xaxis.label.set_color(axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.title.set_color(text_color)
    for spine in ax.spines.values():
        spine.set_color(axis_color)


def add_shared_probe_legend(fig: Figure, probe_labels: Sequence[str], *, y: float = 0.98) -> None:
    handles = [
        Patch(facecolor=probe_palette[i % len(probe_palette)], edgecolor="none", label=label)
        for i, label in enumerate(probe_labels)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, y),
        labelcolor="white",
    )


def add_band_colorbar(fig: Figure, rect: tuple[float, float, float, float]) -> None:
    """Add horizontal cividis gradient colorbar for B0 (low freq) -> B7 (high freq)."""
    cax = fig.add_axes(rect)
    norm = Normalize(vmin=0, vmax=7)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=norm),
        cax=cax,
        orientation="horizontal",
    )
    cb.set_ticks([0, 7])
    cb.set_ticklabels([r"$\mathfrak{B}_0$", r"$\mathfrak{B}_7$"])
    cb.ax.tick_params(labelsize=6, colors="white")
    cax.set_facecolor(figure_facecolor)
    cax.text(0.5, 1.35, "Band", transform=cax.transAxes, ha="center", va="bottom", fontsize=7, color="white")
