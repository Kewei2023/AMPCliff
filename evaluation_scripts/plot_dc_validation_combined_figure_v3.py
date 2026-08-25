# maintained by kewei li
from __future__ import annotations

from pathlib import Path
import math
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

INPUT_XLSX = Path("dc_validation_combined_figure_data.xlsx")
OUT_PNG = Path("dc_validation_combined_figure_updated_v3.png")
OUT_PDF = Path("dc_validation_combined_figure_updated_v3.pdf")
OUT_SVG = Path("dc_validation_combined_figure_updated_v3.svg")

LEGEND_Y = 0.982
LEGEND_FONTSIZE = 8.5
LEGEND_MARKERSIZE = 8
LEGEND_HANDTEXTPAD = 0.45
LEGEND_HANDLELENGTH = 1.0
# B1 overlaps B6/B7 near the bottom; draw it last with the highest z-order.
BAND_PLOT_ORDER = [0, 2, 3, 4, 5, 6, 7, 1]


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

PROBE_B0_KEY = r"$\mathfrak{B}_{0}$"
PROBE_B0_LABEL = r"$\widetilde{\mathfrak{B}}_{0}$"

probe_specs = [
    ("AAC", "AAC"),
    (PROBE_B0_KEY, PROBE_B0_LABEL),
    (r"$\mathfrak{B}_{1}$", r"$\mathfrak{B}_{1}$"),
]
probe_colors = {"AAC": "#7A7A7A", PROBE_B0_KEY: "#0072B2", r"$\mathfrak{B}_{1}$": "#009E73"}

heatmap_cmap = plt.get_cmap("GnBu")
heat_vmin, heat_vmax = 0.55, 1.00
small_bar_color = "#D9DEE5"
small_bar_edge = "#A6AFB8"
band_cmap = plt.get_cmap("viridis")
band_colors = band_cmap(np.linspace(0.08, 0.92, 8))
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
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": axis_color,
    "axes.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color,
    "text.color": text_color, "pdf.fonttype": 42, "ps.fonttype": 42,
})

heat_lookup = {}
for _, row in heat_df.iterrows():
    heat_lookup[(row["species"], row["property"])] = [
        float(row[r"$\mathfrak{B}_{0}$"]), float(row[r"$\mathfrak{B}_{1}$"]),
        float(row[r"$\mathfrak{B}_{2}$"]), float(row[r"$\mathfrak{B}_{3}$"]),
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

fig = plt.figure(figsize=(17.2, 8.7), facecolor="white")
outer = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.55, 2.08], height_ratios=[1, 1],
                         left=0.065, right=0.985, top=0.90, bottom=0.13, wspace=0.25, hspace=0.31)

heat_axes, bar_axes, right_axes_by_species = [], [], {}

for row_idx, sp in enumerate(species_order):
    ax_h = fig.add_subplot(outer[row_idx, 0])
    heat_axes.append(ax_h)
    matrix = np.array([heat_lookup[(sp, prop)] for prop in property_order], dtype=float)
    im = ax_h.imshow(matrix, cmap=heatmap_cmap, norm=Normalize(heat_vmin, heat_vmax), aspect="auto", interpolation="nearest")
    ax_h.set_xticks(range(4))
    ax_h.set_xticklabels([r"$\widetilde{\mathfrak{B}}_{0}$", r"$\mathfrak{B}_{1}$", r"$\mathfrak{B}_{2}$", r"$\mathfrak{B}_{3}$"])
    ax_h.set_yticks(range(len(property_order)))
    ax_h.set_yticklabels([property_labels_heat[p] for p in property_order])
    ax_h.set_xlabel("DCT coefficient", labelpad=5)
    ax_h.set_ylabel("Physicochemical property", labelpad=6)
    ax_h.set_title(f"{species_names[sp]}  |  DC property encoding", pad=9, fontweight="semibold")
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
            ax_h.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.3, color=color)
    for spine in ax_h.spines.values():
        spine.set_visible(False)

    ax_b = fig.add_subplot(outer[row_idx, 1])
    bar_axes.append(ax_b)
    x = np.arange(len(property_order), dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    for offset, (probe_key, _) in zip(offsets, probe_specs):
        means = [probe_stats[(sp, p, probe_key)][0] for p in property_order]
        cis = [probe_stats[(sp, p, probe_key)][1] for p in property_order]
        ax_b.bar(x + offset, means, width=width, color=probe_colors[probe_key], edgecolor="white", linewidth=0.5,
                 yerr=cis, error_kw={"ecolor": axis_color, "elinewidth": 0.8, "capsize": 1.8, "capthick": 0.8}, zorder=3)
    ax_b.set_ylim(0.0, 1.04)
    ax_b.set_xlim(-0.55, len(property_order) - 0.45)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([property_labels_heat[p] for p in property_order], rotation=31, ha="right", rotation_mode="anchor")
    ax_b.set_ylabel("Spearman's $\\rho$")
    ax_b.set_title("Property probe comparison", pad=9, fontweight="semibold")
    ax_b.grid(axis="y", color=grid_color, linewidth=0.7, alpha=0.75, zorder=0)
    ax_b.tick_params(axis="x", length=0, pad=4)
    ax_b.tick_params(axis="y", length=3)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_linewidth(0.8)
    ax_b.spines["bottom"].set_linewidth(0.8)

    inner = outer[row_idx, 2].subgridspec(2, 2, wspace=0.22, hspace=0.42)
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
            )
        ax.set_title(panel_title, pad=3, fontsize=8.4, fontweight="semibold")
        ax.set_xticks(xx)
        ax.set_xticklabels(bucket_labels_compact, fontsize=6.4, linespacing=0.9)
        ax.set_ylim(0.0, 0.55)
        ax.set_yticks([0.0, 0.2, 0.4])
        ax.grid(axis="y", color=grid_color, linewidth=0.6, alpha=0.70, zorder=0)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        if cc == 0:
            ax.set_ylabel(r"Mean $|\Delta \mathrm{MSE}|$", fontsize=7.2)
    right_axes_by_species[sp] = right_axes

probe_handles = [
    Line2D(
        [0], [0], marker="s", linestyle="none", markersize=LEGEND_MARKERSIZE,
        markerfacecolor=probe_colors[probe_key], markeredgecolor="none", label=probe_label,
    )
    for probe_key, probe_label in probe_specs
]
fig.legend(
    handles=probe_handles,
    labels=[probe_label for _, probe_label in probe_specs],
    loc="upper center",
    bbox_to_anchor=(0.438, LEGEND_Y),
    ncol=len(probe_specs),
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
    handletextpad=LEGEND_HANDTEXTPAD,
    handlelength=LEGEND_HANDLELENGTH,
    columnspacing=1.25,
)

fig.canvas.draw()
top_heat_box = heat_axes[0].get_position()
bottom_heat_box = heat_axes[1].get_position()
cax_heat = fig.add_axes([top_heat_box.x1 + 0.008, bottom_heat_box.y0, 0.008, top_heat_box.y1 - bottom_heat_box.y0])
cb_heat = fig.colorbar(im, cax=cax_heat, orientation="vertical", ticks=[0.55, 0.65, 0.75, 0.85, 0.95, 1.00])
cb_heat.ax.set_title(r"$\rho$", fontsize=8.5, pad=5)
cb_heat.ax.tick_params(labelsize=7, length=2.5)
cb_heat.outline.set_linewidth(0.6)

panel_map = [("A", heat_axes[0]), ("B", bar_axes[0]), ("C", right_axes_by_species["e_coli"][0]),
             ("D", heat_axes[1]), ("E", bar_axes[1]), ("F", right_axes_by_species["s_aureus"][0])]
for letter, ax in panel_map:
    box = ax.get_position()
    fig.text(box.x0, box.y1 + (0.015 if letter in ("C", "F") else 0.010),
             f"({letter})", fontsize=10.5, fontweight="bold", ha="left", va="bottom")

for sp in species_order:
    axes = right_axes_by_species[sp]
    left = min(a.get_position().x0 for a in axes)
    right = max(a.get_position().x1 for a in axes)
    top = max(a.get_position().y1 for a in axes)
    fig.text((left + right) / 2, top + 0.014, f"{species_names[sp]}  |  Overall sensitivity and bandwise allocation",
             ha="center", va="bottom", fontsize=9.4, fontweight="semibold")

all_right_axes = right_axes_by_species["e_coli"] + right_axes_by_species["s_aureus"]
right_left = min(a.get_position().x0 for a in all_right_axes)
right_right = max(a.get_position().x1 for a in all_right_axes)
band_handles = [
    Line2D(
        [0], [0], marker="s", linestyle="none", markersize=LEGEND_MARKERSIZE,
        markerfacecolor=band_colors[i], markeredgecolor="none",
        label=rf"$\mathfrak{{B}}_{{{i}}}$",
    )
    for i in range(8)
]
fig.legend(
    handles=band_handles,
    loc="upper center",
    bbox_to_anchor=((right_left + right_right) / 2, LEGEND_Y),
    ncol=8,
    frameon=False,
    fontsize=LEGEND_FONTSIZE,
    handletextpad=LEGEND_HANDTEXTPAD,
    handlelength=LEGEND_HANDLELENGTH,
    columnspacing=0.88,
)

fig.savefig(OUT_PNG, dpi=350, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}, {OUT_PDF}, {OUT_SVG}")
