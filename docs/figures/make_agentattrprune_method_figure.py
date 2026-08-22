#!/usr/bin/env python3
"""Render a compact paper-style method figure for AgentAttrPrune.

Unlike the pipeline overview, this composition treats the dense and sparse
multi-agent graphs as the primary objects and uses attribution/matrix views as
scientific insets.  The small matrices use representative values from the
current four-worker GSM8K calibration artifact.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent

NAVY = "#19354F"
INK = "#2A3B48"
MUTED = "#687987"
BLUE = "#3978B7"
TEAL = "#168F87"
PURPLE = "#7656A8"
ORANGE = "#E68A27"
RED = "#C85151"
GRAY = "#AAB5BD"
PALE = "#EDF2F5"
WHITE = "#FFFFFF"


def text(ax, x, y, value, *, size=8, color=INK, weight="normal", ha="left", va="center", z=10, **kwargs):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=z,
        **kwargs,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=INK,
    lw=1.0,
    alpha=1.0,
    mutation=7,
    connection="arc3",
    linestyle="-",
    style="-|>",
    z=3,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        alpha=alpha,
        connectionstyle=connection,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def worker(ax, x, y, index, *, edge=BLUE, fill=WHITE, halo=None, r=0.019, z=8):
    if halo is not None:
        ax.add_patch(Circle((x, y), r + 0.010, facecolor="none", edgecolor=halo, linewidth=1.2, alpha=0.34, zorder=z - 1))
    ax.add_patch(Circle((x, y), r, facecolor=fill, edgecolor=edge, linewidth=1.55, zorder=z))
    node_label = f"A$_{{{index}}}$" if isinstance(index, str) else f"A{index}"
    text(ax, x, y, node_label, size=7.0, color=NAVY, weight="bold", ha="center", z=z + 1)


def reviewer(ax, x, y, *, width=0.095):
    height = 0.047
    points = [(x, y + height / 2), (x + width / 2, y), (x, y - height / 2), (x - width / 2, y)]
    ax.add_patch(Polygon(points, closed=True, facecolor=WHITE, edgecolor=NAVY, linewidth=1.3, zorder=7))
    text(ax, x, y + 0.003, "reviewer", size=6.8, color=NAVY, weight="bold", ha="center", z=8)


def panel_title(ax, x, y, letter, title, subtitle=None):
    text(ax, x, y, letter, size=12, color=NAVY, weight="bold")
    text(ax, x + 0.028, y, title, size=12, color=NAVY, weight="bold")
    if subtitle:
        text(ax, x + 0.028, y - 0.035, subtitle, size=7.2, color=MUTED)


def graph(
    ax,
    *,
    x0,
    width,
    dense,
    spatial_masks=None,
    temporal_mask=None,
    highlight_target=None,
):
    xs = np.linspace(x0 + 0.040, x0 + width - 0.040, 4)
    y_top, y_bottom, y_review = 0.715, 0.455, 0.210

    # Quiet layer guides make the two reasoning rounds legible without boxes.
    ax.add_line(Line2D([x0 + 0.012, x0 + width - 0.012], [y_top, y_top], color=PALE, lw=7, solid_capstyle="round", zorder=0))
    ax.add_line(Line2D([x0 + 0.012, x0 + width - 0.012], [y_bottom, y_bottom], color=PALE, lw=7, solid_capstyle="round", zorder=0))
    text(ax, x0 + 0.003, y_top, "$r=0$", size=6.8, color=BLUE, weight="bold", ha="right")
    text(ax, x0 + 0.003, y_bottom, "$r=1$", size=6.8, color=BLUE, weight="bold", ha="right")

    if dense:
        spatial_masks = [np.triu(np.ones((4, 4), dtype=int), 1)] * 2
        temporal_mask = np.ones((4, 4), dtype=int)

    # Temporal edges are drawn first; retained weights become visually dominant.
    for source in range(4):
        for target in range(4):
            active = bool(temporal_mask[source, target])
            if not active and not dense:
                continue
            importance = 0.16 + 0.18 * ((source + target) % 4) / 3
            arrow(
                ax,
                (xs[source], y_top - 0.020),
                (xs[target], y_bottom + 0.020),
                color=PURPLE if active else GRAY,
                lw=0.72 + importance if not dense else 0.60,
                alpha=0.33 if dense else 0.55 + importance,
                mutation=4.5,
                linestyle=(0, (2.0, 2.0)),
                z=1,
            )

    for round_index, y in enumerate((y_top, y_bottom)):
        mask = spatial_masks[round_index]
        for source in range(4):
            for target in range(source + 1, 4):
                if not mask[source, target]:
                    continue
                distance = target - source
                arrow(
                    ax,
                    (xs[source] + 0.019, y),
                    (xs[target] - 0.020, y),
                    color=BLUE if dense else TEAL,
                    lw=0.9 if dense else 1.25 + 0.15 * distance,
                    alpha=0.72 if dense else 0.92,
                    mutation=5.0,
                    connection=f"arc3,rad={-0.11 - 0.055 * (distance - 1)}",
                    z=4,
                )

    for round_index, y in enumerate((y_top, y_bottom)):
        for idx, x in enumerate(xs):
            is_target = highlight_target == (round_index, idx)
            worker(
                ax,
                x,
                y,
                idx + 1,
                edge=ORANGE if is_target else (BLUE if dense else TEAL),
                halo=ORANGE if is_target else None,
                fill="#FFF8ED" if is_target else WHITE,
            )

    reviewer(ax, x0 + width / 2, y_review)
    for x in xs:
        arrow(
            ax,
            (x, y_bottom - 0.021),
            (x0 + width / 2, y_review + 0.026),
            color=NAVY,
            lw=0.65,
            alpha=0.34,
            mutation=4.0,
            z=2,
        )
    return xs, (y_top, y_bottom, y_review)


def dense_graph(ax):
    panel_title(
        ax,
        0.030,
        0.925,
        "a",
        "Dense communication graph",
        "ordered spatial DAG + complete temporal $N\\times N$ edges",
    )
    xs, ys = graph(ax, x0=0.032, width=0.270, dense=True, highlight_target=(1, 3))
    text(ax, 0.167, 0.140, "$G_{dense}$", size=10, color=NAVY, weight="bold", ha="center")
    text(ax, 0.167, 0.111, "trace every generation event", size=6.7, color=MUTED, ha="center")
    return xs[3], ys[1]


def sentence_bar(ax, x, y, width, *, selected=False, fixed=False):
    color = BLUE if fixed else (ORANGE if selected else GRAY)
    alpha = 0.96 if (selected or fixed) else 0.40
    ax.add_line(Line2D([x, x + width], [y, y], color=color, lw=6.0, alpha=alpha, solid_capstyle="round", zorder=8))


def attribution_inset(ax, source_point):
    panel_title(
        ax,
        0.345,
        0.925,
        "b",
        "Attribution at a receiving event $e=(b,r)$",
        "mask incoming sentences; keep task/template content fixed",
    )

    center = (0.505, 0.682)
    radius = 0.160
    ax.add_patch(Circle(center, radius, facecolor="#FFFCF7", edgecolor="#E8D6BF", linewidth=1.1, zorder=0))
    arrow(
        ax,
        source_point,
        (center[0] - radius + 0.006, center[1] - 0.010),
        color=ORANGE,
        lw=1.2,
        alpha=0.78,
        mutation=6,
        connection="arc3,rad=-0.16",
        linestyle=(0, (3, 2)),
        z=2,
    )
    text(ax, 0.326, 0.565, "zoom", size=6.2, color=ORANGE, weight="bold", rotation=35)

    source_x = 0.373
    rows = [
        (0.765, "$o_{A_1}$", [0.030, 0.022, 0.036], [True, False, False], BLUE),
        (0.692, "$o_{A_2}$", [0.022, 0.033, 0.025], [False, True, False], BLUE),
        (0.619, "$h_{A_3}$", [0.027, 0.018, 0.032], [False, False, True], PURPLE),
    ]
    target = (0.570, 0.692)
    for y, name, widths, selected, edge_color in rows:
        text(ax, source_x - 0.012, y, name, size=7.3, color=edge_color, weight="bold", ha="right")
        cursor = source_x
        for width, chosen in zip(widths, selected):
            sentence_bar(ax, cursor, y, width, selected=chosen)
            cursor += width + 0.009
        arrow(ax, (cursor + 0.002, y), (target[0] - 0.024, target[1]), color=edge_color, lw=0.9, alpha=0.68, mutation=5.0, z=4)

    worker(ax, target[0], target[1], "b", edge=ORANGE, fill="#FFF7EA", r=0.026)
    text(ax, target[0], target[1] - 0.047, "receiving node", size=6.4, color=MUTED, ha="center")

    # Fixed prompt token bar enters independently and is never part of the ranking.
    sentence_bar(ax, 0.455, 0.585, 0.065, fixed=True)
    text(ax, 0.449, 0.585, "$F$", size=7.0, color=BLUE, weight="bold", ha="right")
    arrow(ax, (0.525, 0.585), (target[0] - 0.006, target[1] - 0.028), color=BLUE, lw=0.95, alpha=0.76, mutation=5.0, z=4)
    text(ax, 0.486, 0.558, "fixed prompt", size=6.0, color=BLUE, ha="center")

    # Teacher-forced output and recovery curve.
    arrow(ax, (target[0] + 0.027, target[1]), (0.624, target[1]), color=TEAL, lw=1.2, mutation=6.5, z=5)
    text(ax, 0.632, target[1] + 0.014, "$y_1,\\ldots,y_L$", size=7.3, color=TEAL, weight="bold")
    text(ax, 0.632, target[1] - 0.015, "teacher forced", size=5.8, color=MUTED)

    plot_x0, plot_x1 = 0.430, 0.600
    plot_y0, plot_y1 = 0.405, 0.505
    ax.add_line(Line2D([plot_x0, plot_x0], [plot_y0, plot_y1], color=GRAY, lw=0.8, zorder=3))
    ax.add_line(Line2D([plot_x0, plot_x1], [plot_y0, plot_y0], color=GRAY, lw=0.8, zorder=3))
    ax.add_line(Line2D([plot_x0, plot_x1], [plot_y0 + 0.095, plot_y0 + 0.095], color=ORANGE, lw=0.8, linestyle=(0, (3, 2)), alpha=0.65, zorder=3))
    recovery = np.array([0.00, 0.47, 0.78, 0.96])
    px = np.linspace(plot_x0 + 0.012, plot_x1 - 0.010, len(recovery))
    py = plot_y0 + 0.010 + recovery * 0.088
    ax.plot(px, py, color=NAVY, lw=1.4, zorder=5)
    ax.scatter(px, py, s=[14, 18, 18, 26], facecolors=[WHITE, ORANGE, ORANGE, TEAL], edgecolors=[NAVY, ORANGE, ORANGE, TEAL], linewidths=1.0, zorder=6)
    text(ax, plot_x0 - 0.010, plot_y1, "$\\rho_k$", size=6.4, color=NAVY, weight="bold", ha="right")
    text(ax, plot_x1, plot_y0 - 0.018, "selected prefix $S_k$", size=6.0, color=MUTED, ha="right")
    text(ax, plot_x1 + 0.004, plot_y0 + 0.095, "0.95", size=5.7, color=ORANGE)
    text(ax, 0.515, 0.371, "exact greedy insertion", size=6.5, color=NAVY, weight="bold", ha="center")

    text(
        ax,
        0.505,
        0.325,
        r"$T_{a\rightarrow b}^{(r)}=|\mathrm{selected\ source\ tokens}|\,/\,|\mathrm{source\ output\ tokens}|$",
        size=7.5,
        color=INK,
        ha="center",
    )
    text(ax, 0.505, 0.294, "repeat for every worker-round event and the final reviewer", size=6.4, color=MUTED, ha="center")


def heatmap(ax, x, y, data, title, *, cell=0.0165, axes=False):
    cmap = LinearSegmentedColormap.from_list("attr", ["#F2F5F7", "#BCDCDD", TEAL, NAVY])
    n = data.shape[0]
    for i in range(n):
        for j in range(n):
            value = float(data[i, j])
            ax.add_patch(
                Rectangle(
                    (x + j * cell, y + (n - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=cmap(min(1, value / 0.55)),
                    edgecolor=WHITE,
                    linewidth=0.55,
                    zorder=5,
                )
            )
    text(ax, x + n * cell / 2, y + n * cell + 0.022, title, size=7.0, color=NAVY, weight="bold", ha="center")
    if axes:
        text(ax, x - 0.010, y + n * cell / 2, "source", size=5.5, color=MUTED, ha="center", rotation=90)
        text(ax, x + n * cell / 2, y - 0.010, "target", size=5.5, color=MUTED, ha="center", va="top")


def matrix_tensor(ax):
    panel_title(
        ax,
        0.345,
        0.245,
        "c",
        "Round-specific edge importance",
        "conditional mean over verified-correct calibration samples $x\\in\\mathcal{C}$",
    )
    spatial0 = np.array(
        [[0.00, 0.45, 0.20, 0.11], [0.00, 0.00, 0.23, 0.15], [0.00, 0.00, 0.00, 0.38], [0.00, 0.00, 0.00, 0.00]]
    )
    spatial1 = np.array(
        [[0.00, 0.08, 0.10, 0.06], [0.00, 0.00, 0.08, 0.05], [0.00, 0.00, 0.00, 0.06], [0.00, 0.00, 0.00, 0.00]]
    )
    temporal1 = np.array(
        [[0.21, 0.06, 0.06, 0.04], [0.13, 0.17, 0.09, 0.05], [0.23, 0.12, 0.30, 0.12], [0.31, 0.12, 0.09, 0.55]]
    )
    heatmap(ax, 0.384, 0.105, spatial0, "$\\bar{M}_s^{(0)}$", axes=True)
    heatmap(ax, 0.478, 0.105, spatial1, "$\\bar{M}_s^{(1)}$")
    heatmap(ax, 0.572, 0.105, temporal1, "$\\bar{M}_t^{(1)}$")
    text(ax, 0.526, 0.073, "low", size=5.8, color=MUTED, ha="right")
    gradient = np.linspace(0, 1, 120).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("attrbar", ["#F2F5F7", "#BCDCDD", TEAL, NAVY])
    ax.imshow(gradient, extent=[0.532, 0.615, 0.067, 0.076], origin="lower", aspect="auto", cmap=cmap, zorder=4)
    text(ax, 0.620, 0.073, "high", size=5.8, color=MUTED)
    text(
        ax,
        0.505,
        0.035,
        r"$\bar{M}[r,a,b]=\sum_{x\in\mathcal{C}}M^x[r,a,b]\,/\,\sum_{x\in\mathcal{C}}O^x[r,a,b]$",
        size=7.2,
        color=INK,
        ha="center",
    )


def sparse_graph(ax):
    panel_title(
        ax,
        0.700,
        0.925,
        "d",
        "Frozen static policy",
        r"remove the lowest-importance $p\%$ edges per round and edge type",
    )
    spatial = np.array(
        [[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=int
    )
    temporal = np.array(
        [[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=int
    )
    graph(
        ax,
        x0=0.698,
        width=0.270,
        dense=False,
        spatial_masks=[spatial, spatial],
        temporal_mask=temporal,
    )
    text(ax, 0.833, 0.140, "$G_p=G_{dense}\\odot P$", size=10, color=NAVY, weight="bold", ha="center")
    text(ax, 0.833, 0.111, "frozen masks $P_s^{(r)},P_t^{(r)}$; reviewer stays dense", size=6.7, color=MUTED, ha="center")

    # Matrix-to-policy link is intentionally thin and structural, not a pipeline chevron.
    arrow(
        ax,
        (0.646, 0.170),
        (0.710, 0.270),
        color=TEAL,
        lw=1.1,
        alpha=0.75,
        mutation=6,
        connection="arc3,rad=-0.12",
        z=3,
    )
    text(ax, 0.684, 0.235, r"bottom $p\%$", size=6.3, color=TEAL, weight="bold", rotation=35, ha="center")

    # Compact legend shared by the graph comparison.
    ax.add_line(Line2D([0.735, 0.767], [0.060, 0.060], color=TEAL, lw=2.0, zorder=7))
    text(ax, 0.773, 0.060, "retained spatial", size=6.0, color=MUTED)
    ax.add_line(Line2D([0.850, 0.882], [0.060, 0.060], color=PURPLE, lw=1.3, linestyle=(0, (2, 2)), zorder=7))
    text(ax, 0.888, 0.060, "retained temporal", size=6.0, color=MUTED)


def render():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 6.6), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    target = dense_graph(ax)
    attribution_inset(ax, target)
    matrix_tensor(ax)
    sparse_graph(ax)

    # A subtle aggregation stem ties the inset to the matrices without creating
    # a sequence of boxed stages.
    arrow(ax, (0.505, 0.282), (0.505, 0.255), color=TEAL, lw=1.0, alpha=0.65, mutation=5, z=3)

    paths = {
        "svg": OUT / "agentattrprune_method_figure.svg",
        "pdf": OUT / "agentattrprune_method_figure.pdf",
        "png": OUT / "agentattrprune_method_figure.png",
    }
    fig.savefig(paths["svg"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(paths["pdf"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(paths["png"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05, dpi=250)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for kind, path in render().items():
        print(f"{kind}: {path}")
