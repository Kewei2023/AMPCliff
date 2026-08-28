from pathlib import Path
from collections import defaultdict
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# 与 export_fftlag_exp1_fulltest_violin_data.py 的默认输出保持一致：读写均锚定 aggregated 目录
AGG_DIR = REPO_ROOT / "outputs" / "analysis" / "fftlag_mechanism" / "aggregated"
violin_xlsx = AGG_DIR / "exp1_band_knockout_violin_combined_data.xlsx"
heatmap_xlsx = AGG_DIR / "exp1_representative_band_knockout_heatmaps_data.xlsx"

out_png = AGG_DIR / "exp1_abs_mse_violin_heatmap_composite_allblue_violin.png"
out_svg = AGG_DIR / "exp1_abs_mse_violin_heatmap_composite_allblue_violin.svg"


# Global font-size scaling factor.
# 1.0 keeps the original font sizes.
# Example: 1.2 makes all text 20% larger.
FONT_SCALE = 1.5


def fs(size: float) -> float:
    """Return a font size scaled by FONT_SCALE."""
    return size * FONT_SCALE

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M = f"{{{MAIN_NS}}}"


def column_index(cell_ref):
    index = 0
    for char in re.match(r"([A-Z]+)", cell_ref).group(1):
        index = index * 26 + ord(char) - 64
    return index - 1


def resolve_sheet_xml(xlsx_path, sheet_name):
    with zipfile.ZipFile(xlsx_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_id = None
        for sheet in workbook.find(M + "sheets"):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
                break
        if rel_id is None:
            raise KeyError(f"Worksheet not found: {sheet_name}")

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relationship in relationships:
            if relationship.attrib.get("Id") == rel_id:
                target = relationship.attrib["Target"]
                break
        target = target.lstrip("/")
        return target if target.startswith("xl/") else "xl/" + target


def load_shared_strings(archive):
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(item.itertext()) for item in root.findall(M + "si")]


def iter_sheet_rows(xlsx_path, sheet_name):
    sheet_xml = resolve_sheet_xml(xlsx_path, sheet_name)
    with zipfile.ZipFile(xlsx_path) as archive:
        shared_strings = load_shared_strings(archive)
        with archive.open(sheet_xml) as stream:
            for _, row_element in ET.iterparse(stream, events=("end",)):
                if row_element.tag != M + "row":
                    continue

                values = {}
                for cell in row_element.findall(M + "c"):
                    index = column_index(cell.attrib["r"])
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(M + "v")
                    inline_node = cell.find(M + "is")

                    if inline_node is not None:
                        value = "".join(inline_node.itertext())
                    elif value_node is None:
                        value = ""
                    elif cell_type == "s":
                        value = shared_strings[int(value_node.text)]
                    else:
                        value = value_node.text
                    values[index] = value

                if values:
                    yield [values.get(i, "") for i in range(max(values) + 1)]
                row_element.clear()


# -----------------------------
# Load violin data
# -----------------------------
rows = iter_sheet_rows(violin_xlsx, "distribution_long")
header = next(rows)
columns = {name: i for i, name in enumerate(header)}
delta_column = "mse_diff_mean" if "mse_diff_mean" in columns else "mse_diff"

violin_data = defaultdict(list)
for row in rows:
    species = row[columns["species"]]
    layer = int(float(row[columns["layer"]]))
    band = int(float(row[columns["band"]]))
    violin_data[(species, layer, band)].append(abs(float(row[columns[delta_column]])))

# -----------------------------
# Load heatmap data
# -----------------------------
rows = iter_sheet_rows(heatmap_xlsx, "combined_long")
header = next(rows)
columns = {name: i for i, name in enumerate(header)}
delta_column = "mse_diff_mean" if "mse_diff_mean" in columns else "mse_diff"

sample_order = {
    "e_coli": [35, 1442, 1438, 1004, 1043],
    "s_aureus": [641, 379, 1963, 876, 1026],
}
heatmap_data = {
    (species, sample_id): np.full((6, 8), np.nan)
    for species, sample_ids in sample_order.items()
    for sample_id in sample_ids
}

for row in rows:
    species = row[columns["species"]]
    sample_id = int(float(row[columns["idx"]]))
    key = (species, sample_id)
    if key not in heatmap_data:
        continue
    layer = int(float(row[columns["layer"]]))
    band = int(float(row[columns["band"]]))
    heatmap_data[key][layer, band] = abs(float(row[columns[delta_column]]))

for key, matrix in heatmap_data.items():
    if np.isnan(matrix).any():
        raise ValueError(f"Incomplete representative matrix: {key}")

species_order = ["e_coli", "s_aureus"]
species_labels = {
    "e_coli": r"$\it{E.\ coli}$",
    "s_aureus": r"$\it{S.\ aureus}$",
}
layers = list(range(6))
bands = list(range(8))
band_labels = [rf"$\mathcal{{B}}_{{{i}}}$" for i in bands]

all_violin_values = np.concatenate(
    [
        np.asarray(violin_data[(species, layer, band)])
        for species in species_order
        for layer in layers
        for band in bands
    ]
)
violin_ymax = max(10.0, float(np.ceil(all_violin_values.max())))

all_heatmap_values = np.concatenate([matrix.ravel() for matrix in heatmap_data.values()])
heatmap_vmax = float(np.quantile(all_heatmap_values, 0.99))
heatmap_norm = colors.Normalize(vmin=0.0, vmax=heatmap_vmax)
heatmap_cmap = plt.get_cmap("GnBu")

# All violins use one single blue
violin_blue = "#1f77b4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": fs(9),
        "axes.titlesize": fs(15.5),
        "axes.labelsize": fs(15.5),
        "xtick.labelsize": fs(8),
        "ytick.labelsize": fs(8.5),
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig = plt.figure(figsize=(22.5, 13.7))
outer = fig.add_gridspec(
    4,
    1,
    height_ratios=[0.075, 1.65, 0.18, 1.0],
    hspace=0.16,
)

# Panel A title row
title_a = fig.add_subplot(outer[0])
title_a.axis("off")
title_a.text(0.0, 0.55, "(A)", ha="left", va="center", fontsize=fs(17), fontweight="bold")
title_a.text(
    0.5,
    0.55,
    r"Full-test-set $|\Delta\mathrm{P}|$ distributions across layers and sequence-frequency bands",
    ha="center",
    va="center",
    fontsize=fs(17),
    fontweight="semibold",
)

# Panel A violins
top_grid = outer[1].subgridspec(2, 6, wspace=0.10, hspace=0.13)
top_axes = np.empty((2, 6), dtype=object)
positions = np.arange(1, 9)

for row_index, species in enumerate(species_order):
    for column_index, layer in enumerate(layers):
        ax = fig.add_subplot(top_grid[row_index, column_index])
        top_axes[row_index, column_index] = ax

        values = [np.asarray(violin_data[(species, layer, band)]) for band in bands]
        violin = ax.violinplot(
            values,
            positions=positions,
            widths=0.82,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott",
            points=120,
        )

        for body in violin["bodies"]:
            body.set_facecolor(violin_blue)
            body.set_edgecolor("#303030")
            body.set_linewidth(0.5)
            body.set_alpha(0.85)

        for x_position, band_values in zip(positions, values):
            q1, median, q3 = np.quantile(band_values, [0.25, 0.50, 0.75])
            ax.vlines(x_position, q1, q3, color="#202020", linewidth=1.15, zorder=4)
            ax.scatter(
                x_position,
                median,
                s=9,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.5,
                zorder=5,
            )

        ax.set_xlim(0.45, 8.55)
        ax.set_ylim(0.0, violin_ymax)
        ax.set_xticks(positions)
        ax.set_xticklabels(band_labels, color="black", fontsize=fs(14), fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65)
        ax.set_axisbelow(True)

        if row_index == 0:
            ax.set_title(f"Layer {layer}", pad=6, fontsize=fs(15.5), fontweight="normal")
            ax.tick_params(axis="x", labelbottom=False)
        if column_index != 0:
            ax.tick_params(axis="y", labelleft=False)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

# Panel B title row
title_b = fig.add_subplot(outer[2])
title_b.axis("off")
title_b.text(0.5, 0.88, "Sequence-frequency band", ha="center", va="center", fontsize=fs(15.5))
title_b.text(0.0, 0.18, "(B)", ha="left", va="center", fontsize=fs(17), fontweight="bold")
title_b.text(
    0.5,
    0.18,
    r"Representative peptide-level $|\Delta\mathrm{P}|$ maps",
    ha="center",
    va="center",
    fontsize=fs(17),
    fontweight="semibold",
)

# Panel B heatmaps
bottom_grid = outer[3].subgridspec(
    2,
    6,
    width_ratios=[1, 1, 1, 1, 1, 0.045],
    wspace=0.16,
    hspace=0.27,
)
bottom_axes = np.empty((2, 5), dtype=object)
image = None

for row_index, species in enumerate(species_order):
    for column_index, sample_id in enumerate(sample_order[species]):
        ax = fig.add_subplot(bottom_grid[row_index, column_index])
        bottom_axes[row_index, column_index] = ax

        image = ax.imshow(
            heatmap_data[(species, sample_id)],
            cmap=heatmap_cmap,
            norm=heatmap_norm,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        ax.set_title(f"Sample {sample_id}", pad=4, fontsize=fs(14.5), fontweight="normal")
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(band_labels, color="black", fontsize=fs(14), fontweight="bold")
        ax.set_yticks(np.arange(6))
        ax.set_yticklabels([str(i) for i in range(6)])

        if row_index == 0:
            ax.tick_params(axis="x", labelbottom=False)
        if column_index != 0:
            ax.tick_params(axis="y", labelleft=False)

        ax.tick_params(length=2.3, width=0.55, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)

colorbar_axis = fig.add_subplot(bottom_grid[:, 5])
colorbar = fig.colorbar(image, cax=colorbar_axis, extend="max")
colorbar.set_label(r"$|\Delta\mathrm{P}|$", rotation=90, labelpad=9, fontsize=fs(15.5))
colorbar.outline.set_linewidth(0.6)
colorbar.ax.tick_params(labelsize=fs(8), length=2.5, width=0.55)

fig.subplots_adjust(left=0.055, right=0.965, top=0.975, bottom=0.065)
fig.canvas.draw()

top_row_centers = [
    0.5 * (top_axes[row, 0].get_position().y0 + top_axes[row, 0].get_position().y1)
    for row in range(2)
]
top_center = 0.5 * (top_axes[1, 0].get_position().y0 + top_axes[0, 0].get_position().y1)
top_left = top_axes[0, 0].get_position().x0

fig.text(top_left - 0.036, top_row_centers[0], species_labels["e_coli"],
         rotation=90, va="center", ha="center", fontsize=fs(17), color="black")
fig.text(top_left - 0.036, top_row_centers[1], species_labels["s_aureus"],
         rotation=90, va="center", ha="center", fontsize=fs(17), color="black")
fig.text(top_left - 0.017, top_center, r"$|\Delta\mathrm{P}|$",
         rotation=90, va="center", ha="center", fontsize=fs(16))

bottom_row_centers = [
    0.5 * (bottom_axes[row, 0].get_position().y0 + bottom_axes[row, 0].get_position().y1)
    for row in range(2)
]
bottom_center = 0.5 * (bottom_axes[1, 0].get_position().y0 + bottom_axes[0, 0].get_position().y1)
bottom_left = bottom_axes[0, 0].get_position().x0
bottom_y0 = bottom_axes[1, 0].get_position().y0

fig.text(bottom_left - 0.036, bottom_row_centers[0], species_labels["e_coli"],
         rotation=90, va="center", ha="center", fontsize=fs(17), color="black")
fig.text(bottom_left - 0.036, bottom_row_centers[1], species_labels["s_aureus"],
         rotation=90, va="center", ha="center", fontsize=fs(17), color="black")
fig.text(bottom_left - 0.017, bottom_center, "ESM-2 layer",
         rotation=90, va="center", ha="center", fontsize=fs(15.5))
fig.text(0.5, bottom_y0 - 0.035, "Sequence-frequency band",
         ha="center", va="center", fontsize=fs(15.5))

fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(out_svg, bbox_inches="tight", facecolor="white")

print(f"Created: {out_png}")
print(f"Created: {out_svg}")
print("All violins are now uniformly blue.")
