from __future__ import annotations

from pathlib import Path
import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

INPUT_XLSX = Path("dc_validation_combined_figure_data.xlsx")
OUT_PNG = Path("dc_validation_combined_figure_updated_v3.png")
OUT_PDF = Path("dc_validation_combined_figure_updated_v3.pdf")
OUT_SVG = Path("dc_validation_combined_figure_updated_v3.svg")

# Global font-size scaling factor.
# 1.0 keeps the original font sizes.
# Example: 1.2 makes all text 20% larger; 1.5 makes all text 50% larger.
FONT_SCALE = 1.5


def fs(size: float) -> float:
    """Return a font size scaled by the global FONT_SCALE factor."""
    return size * FONT_SCALE

# B1 overlaps B6/B7 near the bottom; draw it last with the highest z-order.
BAND_PLOT_ORDER = [0, 2, 3, 4, 5, 6, 7, 1]

# Excel column / probe keys keep original mathfrak names; display uses mathcal.
COL_B = [rf"$\mathfrak{{B}}_{{{i}}}$" for i in range(4)]
DISP_B = [rf"$\mathcal{{B}}_{{{i}}}$" for i in range(8)]


def _band_line_style(band: int) -> tuple[float, float]:
    if band == 0:
        return 1.45, 2.8
    if band == 1:
        return 1.05, 2.4
    if band == 7:
        return 1.05, 1.9
    return 0.82, 1.9


def _band_line_zorder(draw_rank: int) -> int:
    return 3 + draw_rank


heat_df = pd.read_excel(INPUT_XLSX, sheet_name="heatmap_agg")
probe_df = pd.read_excel(INPUT_XLSX, sheet_name="probe_comparison_plot")
summary_df = pd.read_excel(INPUT_XLSX, sheet_name="band_summary")
band_df = pd.read_excel(INPUT_XLSX, sheet_name="band_by_band")

species_order = ["e_coli", "s_aureus"]
species_names = {"e_coli": r"$\it{E.\ coli}$", "s_aureus": r"$\it{S.\ aureus}$"}

property_order = ["net_charge", "charge_density", "pI", "mean_hydrophobicity", "hydrophilic_fraction", "helix_propensity", "hydrophobic_moment"]
property_labels_heat = {
    "net_charge": "Net charge", "charge_density": "Charge density", "pI": "pI",
    "mean_hydrophobicity": "Mean hydrophobicity", "hydrophilic_fraction": "Hydrophilic fraction",
    "helix_propensity": "Helix propensity", "hydrophobic_moment": "Hydrophobic moment",
}

probe_order = ["AAC", COL_B[0], COL_B[1]]
probe_display = {"AAC": "AAC", COL_B[0]: DISP_B[0], COL_B[1]: DISP_B[1]}
BAND_COLORS = [
    "#0072B2", "#009E73", "#E69F00", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#332288",
]
probe_colors = {"AAC": "#7A7A7A", COL_B[0]: BAND_COLORS[0], COL_B[1]: BAND_COLORS[1]}

heatmap_cmap = plt.get_cmap("GnBu")
heat_vmin, heat_vmax = 0.55, 1.00
small_bar_color = "#D9DEE5"
small_bar_edge = "#A6AFB8"
band_colors = BAND_COLORS
text_color = "#222222"
axis_color = "#333333"
grid_color = "#D9D9D9"

panel_specs = [
    ("helix_propensity", "Helix propensity"),
    ("mean_hydrophobicity", "Mean hydrophobicity"),
    ("net_charge", "Net charge"),
    ("hydrophobic_moment", "Hydrophobic moment"),
]
bucket_order = ["bottom_30", "middle_40", "top_30"]
bucket_labels_compact = ["Low\n30%", "Middle\n40%", "High\n30%"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": fs(8), "axes.titlesize": fs(15), "axes.labelsize": fs(13.5),
    "xtick.labelsize": fs(7), "ytick.labelsize": fs(7), "axes.edgecolor": axis_color,
    "axes.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color,
    "text.color": text_color, "pdf.fonttype": 42, "ps.fonttype": 42,
})

heat_lookup = {}
for _, row in heat_df.iterrows():
    heat_lookup[(row["species"], row["property"])] = [
        float(row[COL_B[0]]), float(row[COL_B[1]]),
        float(row[COL_B[2]]), float(row[COL_B[3]]),
    ]

probe_groups = defaultdict(list)
for _, row in probe_df.iterrows():
    probe_groups[(row["species"], row["property"], row["probe"])].append(float(row["spearman"]))

probe_stats = {}
for key, vals in probe_groups.items():
    mean = statistics.fmean(vals)
    ci95 = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    probe_stats[key] = (mean, ci95)

summary_lookup = {}
for _, row in summary_df.iterrows():
    panel_key = row["panel_key"]
    sp = row["species"]
    bucket = row["structure_bucket"] if pd.notna(row["structure_bucket"]) else row["property_bucket"]
    summary_lookup[(sp, panel_key, bucket)] = float(row["mean_abs_mse_diff"])

raw_band_lookup = {}
for _, row in band_df.iterrows():
    panel_key = row["panel_key"]
    sp = row["species"]
    bucket = row["structure_bucket"] if pd.notna(row["structure_bucket"]) else row["property_bucket"]
    band = int(row["band"])
    raw_band_lookup[(sp, panel_key, bucket, band)] = float(row["mse_diff_abs_mean"])

scaled_band_lookup = {}
for sp in species_order:
    for panel_key, _ in panel_specs:
        for bucket in bucket_order:
            raw = np.array([raw_band_lookup[(sp, panel_key, bucket, band)] for band in range(8)], dtype=float)
            total = summary_lookup[(sp, panel_key, bucket)]
            scaled = raw * total / raw.sum() if raw.sum() > 0 else np.zeros_like(raw)
            for band, value in enumerate(scaled):
                scaled_band_lookup[(sp, panel_key, bucket, band)] = float(value)

fig = plt.figure(figsize=(19.0, 14.2), facecolor="white")

# Two-row layout:
#   Row 0: A | B | C
#   Row 1: D | E | F
# C/F are 2x2 blocks in the right column.
outer = fig.add_gridspec(
    2, 3,
    width_ratios=[1.55, 1.35, 2.00],
    height_ratios=[1, 1],
    left=0.07, right=0.98, top=0.88, bottom=0.10,
    wspace=0.50, hspace=0.70,
)

heat_axes, bar_axes, right_axes_by_species = [], [], {}

for row_idx, sp in enumerate(species_order):
    ax_h = fig.add_subplot(outer[row_idx, 0])
    heat_axes.append(ax_h)
    matrix = np.array([heat_lookup[(sp, prop)] for prop in property_order], dtype=float)
    im = ax_h.imshow(matrix, cmap=heatmap_cmap, norm=Normalize(heat_vmin, heat_vmax), aspect="auto", interpolation="nearest")
    ax_h.set_box_aspect(1)
    ax_h.set_anchor("N")
    ax_h.set_xticks(range(4))
    ax_h.set_xticklabels(DISP_B[:4], color="black", fontsize=fs(12))
    ax_h.set_yticks(range(len(property_order)))
    ax_h.set_yticklabels([property_labels_heat[p] for p in property_order], fontsize=fs(12))
    ax_h.set_xlabel("DCT coefficient", labelpad=5, fontsize=fs(13.5))
    ax_h.set_title(
        f"{species_names[sp]}\nDC property encoding",
        pad=9, fontsize=fs(15), fontweight="semibold", linespacing=1.15,
    )
    ax_h.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, len(property_order), 1), minor=True)
    ax_h.grid(which="minor", color="white", linewidth=0.8)
    ax_h.tick_params(which="minor", bottom=False, left=False)
    ax_h.tick_params(length=0)
    norm = Normalize(heat_vmin, heat_vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            rgba = heatmap_cmap(norm(val))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "#1C1C1C" if luminance > 0.58 else "white"
            ax_h.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=fs(9.5), color=color)
    for spine in ax_h.spines.values():
        spine.set_visible(False)

    ax_b = fig.add_subplot(outer[row_idx, 1])
    bar_axes.append(ax_b)
    x = np.arange(len(property_order), dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    for offset, probe in zip(offsets, probe_order):
        means = [probe_stats[(sp, p, probe)][0] for p in property_order]
        cis = [probe_stats[(sp, p, probe)][1] for p in property_order]
        ax_b.bar(
            x + offset, means, width=width, color=probe_colors[probe], edgecolor="white", linewidth=0.5,
            yerr=cis, error_kw={"ecolor": axis_color, "elinewidth": 0.8, "capsize": 1.8, "capthick": 0.8},
            zorder=3, label=probe_display[probe],
        )
    ax_b.set_ylim(0.0, 1.04)
    ax_b.set_xlim(-0.55, len(property_order) - 0.45)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [property_labels_heat[p] for p in property_order],
        rotation=31, ha="right", rotation_mode="anchor", fontsize=fs(12),
    )
    ax_b.set_ylabel("Spearman's $\\rho$", fontsize=fs(13.5))
    ax_b.set_title(
        f"{species_names[sp]}\nProperty probe comparison",
        pad=12, fontsize=fs(15), fontweight="semibold", linespacing=1.15,
    )
    ax_b.grid(axis="y", color=grid_color, linewidth=0.7, alpha=0.75, zorder=0)
    ax_b.tick_params(axis="x", length=0, pad=6)
    ax_b.tick_params(axis="y", length=3)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_linewidth(0.8)
    ax_b.spines["bottom"].set_linewidth(0.8)
    ax_b.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=fs(9), frameon=False, labelcolor="black", borderaxespad=0.0,
    )

    inner = outer[row_idx, 2].subgridspec(2, 2, wspace=0.22, hspace=0.38)
    right_axes = []
    for panel_idx, (panel_key, panel_title) in enumerate(panel_specs):
        rr, cc = divmod(panel_idx, 2)
        ax = fig.add_subplot(inner[rr, cc])
        right_axes.append(ax)
        bar_values = [summary_lookup[(sp, panel_key, bucket)] for bucket in bucket_order]
        xx = np.arange(3)
        ax.bar(xx, bar_values, width=0.72, color=small_bar_color, edgecolor=small_bar_edge, linewidth=0.7, zorder=1)
        for draw_rank, band in enumerate(BAND_PLOT_ORDER):
            yy = [scaled_band_lookup[(sp, panel_key, bucket, band)] for bucket in bucket_order]
            line_width, marker_size = _band_line_style(band)
            ax.plot(
                xx, yy, color=band_colors[band], linewidth=line_width, marker="o",
                markersize=marker_size, markeredgewidth=0, alpha=0.92,
                zorder=_band_line_zorder(draw_rank),
                label=DISP_B[band],
            )
        ax.set_title(panel_title, pad=3, fontsize=fs(11.4))
        ax.set_xticks(xx)
        if rr == 0:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0, labelbottom=False)
        else:
            ax.set_xticklabels(bucket_labels_compact, fontsize=fs(11.4), linespacing=0.9)
            ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylim(0.0, 0.55)
        ax.set_yticks([0.0, 0.2, 0.4])
        ax.grid(axis="y", color=grid_color, linewidth=0.6, alpha=0.70, zorder=0)
        ax.tick_params(axis="y", length=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        if cc == 0:
            ax.set_ylabel(r"Mean $|\Delta P|$", fontsize=fs(9.2))
    right_axes[1].legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", ncol=1,
        fontsize=fs(6.5), frameon=False, labelcolor="black",
    )
    right_axes_by_species[sp] = right_axes

fig.canvas.draw()

def add_heat_colorbar(ax):
    box = ax.get_position()
    cax = fig.add_axes([box.x1 + 0.010, box.y0, 0.008, box.height])
    cb = fig.colorbar(
        im, cax=cax, orientation="vertical",
        ticks=[0.55, 0.65, 0.75, 0.85, 0.95, 1.00],
    )
    cb.set_label(r"$\rho$", fontsize=fs(13.5), rotation=0, labelpad=10)
    cb.ax.tick_params(labelsize=fs(7), length=2.5)
    cb.outline.set_linewidth(0.6)

add_heat_colorbar(heat_axes[0])
add_heat_colorbar(heat_axes[1])

panel_map = [
    ("A", heat_axes[0]),
    ("B", bar_axes[0]),
    ("C", right_axes_by_species["e_coli"]),
    ("D", heat_axes[1]),
    ("E", bar_axes[1]),
    ("F", right_axes_by_species["s_aureus"]),
]

for letter, obj in panel_map:
    if isinstance(obj, list):
        x0 = min(ax.get_position().x0 for ax in obj)
        y1 = max(ax.get_position().y1 for ax in obj)
        offset = 0.048
        x_shift = 0.042
    else:
        box = obj.get_position()
        x0 = box.x0
        y1 = box.y1
        offset = 0.042
        x_shift = 0.048 if letter in ("A", "D") else 0.026

    fig.text(
        x0 - x_shift, y1 + offset, f"({letter})",
        fontsize=fs(15.5), fontweight="bold",
        ha="right", va="bottom",
    )

for sp in species_order:
    axes = right_axes_by_species[sp]
    left = min(a.get_position().x0 for a in axes)
    right = max(a.get_position().x1 for a in axes)
    top = max(a.get_position().y1 for a in axes)
    fig.text(
        (left + right) / 2, top + 0.034,
        f"{species_names[sp]}\nOverall sensitivity and bandwise allocation",
        ha="center", va="bottom", fontsize=fs(14.4), fontweight="semibold",
        color="black", linespacing=1.15,
    )

fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}, {OUT_PDF}, {OUT_SVG}")