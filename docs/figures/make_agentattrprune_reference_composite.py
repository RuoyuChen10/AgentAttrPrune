#!/usr/bin/env python3
"""Render AgentAttrPrune in the visual language of the earlier paper figures.

The composite uses a real attribution record from GSM8K sample 217
(Programming Expert, round 1) and the current four-worker aggregate matrices.
The purpose is methodological illustration; all text is shortened for layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


HERE = Path(__file__).resolve().parent

BLACK = "#111111"
TEXT = "#202020"
MID = "#686868"
LIGHT = "#E5E7EA"
LIGHTER = "#F4F5F6"
WHITE = "#FFFFFF"
ORANGE = "#F47A20"
ORANGE_MID = "#F6A15E"
ORANGE_LIGHT = "#FCD4B4"
GREEN = "#2A9D78"
RED = "#C8190B"
AGENT_COLORS = ["#2E78C5", "#F28A22", "#6E3CB2", "#23A8B7"]
AGENT_NAMES = ["Math Solver", "Math Analyst", "Programmer", "Inspector"]


def txt(ax, x, y, value, *, size=8, color=TEXT, weight="normal", ha="left", va="center", z=20, **kwargs):
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


def rounded(ax, x, y, w, h, *, fc=WHITE, ec=BLACK, lw=1.0, radius=0.006, z=4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=BLACK,
    lw=1.1,
    alpha=1.0,
    mutation=7,
    connection="arc3",
    linestyle="-",
    z=3,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
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


def panel_heading(ax, x, y, letter, title, *, title_size=12.5):
    txt(ax, x, y, letter, size=17, color=BLACK, weight="bold")
    txt(ax, x + 0.035, y, title, size=title_size, color=BLACK, weight="bold")


def robot(ax, x, y, color, *, scale=1.0, z=10):
    """Small vector robot echoing the agent icons in the reference figures."""
    w = 0.028 * scale
    h = 0.042 * scale
    # Antenna.
    ax.add_line(Line2D([x, x], [y + h * 0.54, y + h * 0.78], color=color, lw=1.4 * scale, zorder=z))
    ax.add_patch(Circle((x, y + h * 0.84), 0.0038 * scale, facecolor=WHITE, edgecolor=color, linewidth=1.2 * scale, zorder=z + 1))
    # Ears and head.
    rounded(ax, x - w * 0.65, y + h * 0.04, w * 1.30, h * 0.63, fc=color, ec=color, lw=1.0, radius=0.006 * scale, z=z)
    rounded(ax, x - w * 0.54, y + h * 0.13, w * 1.08, h * 0.40, fc="#17223A", ec=WHITE, lw=0.7, radius=0.004 * scale, z=z + 1)
    ax.add_patch(Rectangle((x - w * 0.78, y + h * 0.17), w * 0.12, h * 0.27, facecolor=color, edgecolor=color, zorder=z))
    ax.add_patch(Rectangle((x + w * 0.66, y + h * 0.17), w * 0.12, h * 0.27, facecolor=color, edgecolor=color, zorder=z))
    # Face.
    ax.add_patch(Circle((x - w * 0.20, y + h * 0.34), 0.0024 * scale, facecolor=WHITE, edgecolor="none", zorder=z + 2))
    ax.add_patch(Circle((x + w * 0.20, y + h * 0.34), 0.0024 * scale, facecolor=WHITE, edgecolor="none", zorder=z + 2))
    ax.add_line(Line2D([x - w * 0.17, x, x + w * 0.17], [y + h * 0.23, y + h * 0.19, y + h * 0.23], color=WHITE, lw=0.8 * scale, zorder=z + 2))
    # Body.
    rounded(ax, x - w * 0.47, y - h * 0.42, w * 0.94, h * 0.40, fc=color, ec=color, lw=0.8, radius=0.004 * scale, z=z)
    ax.add_patch(Circle((x, y - h * 0.22), 0.005 * scale, facecolor=WHITE, edgecolor="none", alpha=0.92, zorder=z + 1))


def draw_dense_graph(ax):
    panel_heading(ax, 0.020, 0.885, "A", "Dense Multi-Agent Communication")
    txt(ax, 0.055, 0.852, "two reasoning rounds; every candidate worker edge is active", size=7.6, color=MID)

    xs = [0.074, 0.151, 0.228, 0.305]
    y0, y1 = 0.735, 0.555
    for source in range(4):
        for target in range(4):
            arrow(
                ax,
                (xs[source], y0 - 0.023),
                (xs[target], y1 + 0.035),
                color="#B6B8BC",
                lw=0.65,
                alpha=0.42,
                mutation=4.0,
                linestyle=(0, (2, 2)),
                z=1,
            )
    for y in (y0, y1):
        for source in range(4):
            for target in range(source + 1, 4):
                arrow(
                    ax,
                    (xs[source] + 0.018, y),
                    (xs[target] - 0.018, y),
                    color=BLACK,
                    lw=0.95,
                    alpha=0.82,
                    mutation=5.0,
                    connection=f"arc3,rad={-0.08 - 0.04 * (target - source - 1)}",
                    z=5,
                )
    for round_index, y in enumerate((y0, y1)):
        txt(ax, 0.038, y, f"Round {round_index}", size=7.0, color=BLACK, weight="bold", ha="right")
        for idx, (x, color) in enumerate(zip(xs, AGENT_COLORS)):
            robot(ax, x, y, color, scale=0.83)
            txt(ax, x, y - 0.048, f"A{idx + 1}", size=6.8, color=color, weight="bold", ha="center")

    # Dense reviewer is outside the worker policy matrices.
    review_y = 0.425
    for x in xs:
        arrow(ax, (x, y1 - 0.033), (0.190, review_y + 0.035), color=BLACK, lw=0.62, alpha=0.46, mutation=4.0, z=2)
    robot(ax, 0.190, review_y, AGENT_COLORS[3], scale=0.88)
    txt(ax, 0.190, review_y - 0.053, "Dense Reviewer", size=7.4, color=AGENT_COLORS[3], weight="bold", ha="center")

    ax.add_line(Line2D([0.062, 0.092], [0.344, 0.344], color=BLACK, lw=1.1, zorder=9))
    txt(ax, 0.098, 0.344, "spatial DAG", size=6.4, color=MID)
    ax.add_line(Line2D([0.190, 0.220], [0.344, 0.344], color="#A9A9AD", lw=1.0, linestyle=(0, (2, 2)), zorder=9))
    txt(ax, 0.226, 0.344, "temporal $N\\times N$", size=6.4, color=MID)


def sentence_strip(ax, x, y, w, text_value, *, selected=False, saliency=0.0, size=5.2):
    if selected:
        # Saliency 0.4--1.0 controls orange intensity, matching current artifacts.
        t = max(0.0, min(1.0, (saliency - 0.4) / 0.6))
        c0 = np.array(mpl.colors.to_rgb(ORANGE_LIGHT))
        c1 = np.array(mpl.colors.to_rgb(ORANGE))
        fc = tuple(c0 * (1 - t) + c1 * t)
        ec = ORANGE
    else:
        fc, ec = LIGHT, "#C8CACD"
    rounded(ax, x, y, w, 0.018, fc=fc, ec=ec, lw=0.55, radius=0.002, z=7)
    txt(ax, x + 0.004, y + 0.009, text_value, size=size, color=TEXT, weight="bold" if selected else "normal", z=9)


def attribution_row(ax, y, agent_index, ratio, strips, *, historical=False):
    x_robot = 0.432
    x_ratio = 0.458
    x_panel = 0.486
    panel_w = 0.486
    color = AGENT_COLORS[agent_index]
    robot(ax, x_robot, y + 0.026, color, scale=0.55)
    txt(ax, x_robot, y - 0.003, f"A{agent_index + 1}", size=5.5, color=color, weight="bold", ha="center")
    txt(ax, x_ratio, y + 0.026, f"{100 * ratio:.1f}%", size=7.2, color=BLACK, weight="bold", ha="center")
    rounded(ax, x_panel, y, panel_w, 0.052, fc=WHITE, ec=color, lw=1.0, radius=0.002, z=4)
    txt(ax, x_panel + 0.006, y + 0.043, AGENT_NAMES[agent_index], size=5.7, color=color, weight="bold")
    cursor = x_panel + 0.006
    available = panel_w - 0.012
    total_units = sum(item[0] for item in strips) + 0.65 * max(0, len(strips) - 1)
    for units, content, selected, saliency in strips:
        width = available * units / total_units
        sentence_strip(ax, cursor, y + 0.008, width, content, selected=selected, saliency=saliency)
        cursor += width + available * 0.65 / total_units


def draw_attribution(ax):
    panel_heading(ax, 0.390, 0.885, "B", "Node Attribution and Edge Importance")
    txt(
        ax,
        0.425,
        0.852,
        "GSM8K sample 217 · receiver: Programming Expert, round 1 · recovery 96.2%",
        size=7.6,
        color=MID,
    )

    # Fixed task row.
    rounded(ax, 0.405, 0.790, 0.025, 0.055, fc="#DADADA", ec=BLACK, lw=0.8, radius=0.001, z=4)
    txt(ax, 0.4175, 0.8175, "TASK", size=6.2, color=BLACK, weight="bold", ha="center", rotation=90)
    rounded(ax, 0.435, 0.790, 0.537, 0.055, fc=WHITE, ec=BLACK, lw=0.9, radius=0.002, z=4)
    sentence_strip(ax, 0.445, 0.809, 0.510, "55% of Toby's friends are boys; 33 are boys. How many friends are girls?", size=5.7)
    txt(ax, 0.445, 0.798, "fixed task / template — never attributed", size=5.5, color=MID)

    # Group labels deliberately resemble the vertical bands in the earlier figures.
    rounded(ax, 0.405, 0.651, 0.025, 0.128, fc="#DADADA", ec=BLACK, lw=0.8, radius=0.001, z=4)
    txt(ax, 0.4175, 0.715, "CURRENT-ROUND", size=6.4, color=BLACK, weight="bold", ha="center", rotation=90)
    rounded(ax, 0.405, 0.397, 0.025, 0.244, fc="#DADADA", ec=BLACK, lw=0.8, radius=0.001, z=4)
    txt(ax, 0.4175, 0.519, "HISTORICAL", size=6.4, color=BLACK, weight="bold", ha="center", rotation=90)

    attribution_row(
        ax,
        0.718,
        0,
        0.0,
        [(1.0, "The answer is 27.", False, 0.0), (1.8, "No selected source tokens", False, 0.0)],
    )
    attribution_row(
        ax,
        0.657,
        1,
        0.0642201835,
        [
            (1.6, "Let total friends be x.", False, 0.0),
            (1.1, "Solving for x:", True, 0.40),
            (1.5, "x - 33 = 27", False, 0.0),
        ],
    )
    attribution_row(
        ax,
        0.580,
        0,
        0.288,
        [
            (1.6, "Let total friends be x.", True, 0.51),
            (1.6, "55% corresponds to 33", False, 0.0),
            (2.6, "number of girls = total - boys", True, 0.43),
        ],
        historical=True,
    )
    attribution_row(
        ax,
        0.519,
        1,
        0.2830188679,
        [
            (1.0, "Solving for x:", True, 0.82),
            (1.5, "number of girls is:", True, 0.60),
            (1.8, "x - 33 = 60 - 33 = 27", True, 0.84),
        ],
        historical=True,
    )
    attribution_row(
        ax,
        0.458,
        2,
        0.0,
        [(1.5, "Total friends = 60", False, 0.0), (2.3, "Subtract boys to obtain 27", False, 0.0)],
        historical=True,
    )
    attribution_row(
        ax,
        0.397,
        3,
        1.0,
        [(1.0, "The answer is 27.", True, 1.0), (1.8, "top-1 selected sentence", True, 1.0)],
        historical=True,
    )

    # Receiving output and exact-greedy recovery curve.
    rounded(ax, 0.405, 0.332, 0.025, 0.055, fc="#DADADA", ec=BLACK, lw=0.8, radius=0.001, z=4)
    txt(ax, 0.4175, 0.3595, "OUTPUT", size=6.2, color=BLACK, weight="bold", ha="center", rotation=90)
    rounded(ax, 0.435, 0.332, 0.350, 0.055, fc=WHITE, ec=AGENT_COLORS[2], lw=1.0, radius=0.002, z=4)
    robot(ax, 0.455, 0.360, AGENT_COLORS[2], scale=0.47)
    txt(ax, 0.475, 0.369, "Programming Expert", size=5.7, color=AGENT_COLORS[2], weight="bold")
    txt(ax, 0.475, 0.349, "The answer is 27.", size=6.4, color=BLACK, weight="bold")

    plot_x0, plot_x1 = 0.815, 0.968
    plot_y0, plot_y1 = 0.337, 0.382
    ax.add_line(Line2D([plot_x0, plot_x0], [plot_y0, plot_y1], color=BLACK, lw=0.7, zorder=6))
    ax.add_line(Line2D([plot_x0, plot_x1], [plot_y0, plot_y0], color=BLACK, lw=0.7, zorder=6))
    curve = np.array([0.393, 0.554, 0.529, 0.751, 0.846, 0.930, 0.947, 0.962])
    xx = np.linspace(plot_x0 + 0.005, plot_x1 - 0.006, len(curve))
    yy = plot_y0 + (curve - 0.35) / (1.0 - 0.35) * (plot_y1 - plot_y0)
    ax.plot(xx, yy, color=BLACK, lw=1.0, zorder=8)
    ax.scatter(xx, yy, s=9, color=ORANGE, zorder=9)
    ax.add_line(Line2D([plot_x0, plot_x1], [plot_y0 + (0.95 - 0.35) / 0.65 * (plot_y1 - plot_y0)] * 2, color=RED, lw=0.7, linestyle=(0, (2, 2)), zorder=6))
    txt(ax, plot_x0, plot_y1 + 0.007, "Exact greedy recovery", size=5.8, color=BLACK, weight="bold")
    txt(ax, plot_x1, plot_y0 - 0.010, "8 selected sentences", size=5.3, color=MID, ha="right")


def heatmap(ax, x, y, data, title, *, cell=0.027, mask=False):
    cmap = LinearSegmentedColormap.from_list("importance", ["#F4F5F6", "#B7DCE0", "#43A7AA", "#174E70"])
    n = data.shape[0]
    for i in range(n):
        for j in range(n):
            value = float(data[i, j])
            if mask:
                fc = GREEN if value > 0 else WHITE
                value_text = "1" if value > 0 else ""
                value_color = WHITE
            else:
                fc = cmap(min(1.0, value / 0.55))
                value_text = f"{100 * value:.0f}" if value > 0 else ""
                value_color = WHITE if value >= 0.23 else BLACK
            ax.add_patch(Rectangle((x + j * cell, y + (n - 1 - i) * cell), cell, cell, facecolor=fc, edgecolor=BLACK if mask else WHITE, linewidth=0.75, zorder=5))
            if value_text:
                txt(ax, x + (j + 0.5) * cell, y + (n - i - 0.5) * cell, value_text, size=5.7, color=value_color, weight="bold", ha="center")
    txt(ax, x + n * cell / 2, y + n * cell + 0.027, title, size=7.8, color=BLACK, weight="bold", ha="center")
    for i, color in enumerate(AGENT_COLORS):
        txt(ax, x - 0.009, y + (n - i - 0.5) * cell, f"A{i + 1}", size=5.5, color=color, weight="bold", ha="right")
        txt(ax, x + (i + 0.5) * cell, y - 0.010, f"A{i + 1}", size=5.5, color=color, weight="bold", ha="center", va="top")


def draw_matrices(ax):
    panel_heading(ax, 0.020, 0.310, "C", "Round-Specific Edge Importance Matrices")
    txt(
        ax,
        0.055,
        0.276,
        r"$T_{a\rightarrow b}^{(r)}=\dfrac{|\,\mathrm{selected\ source\ tokens}\,|}{|\,\mathrm{visible\ source\ output\ tokens}\,|}$",
        size=8.6,
        color=BLACK,
    )
    txt(ax, 0.055, 0.241, "aggregate 270 verified-correct calibration samples · conditional mean", size=6.8, color=MID)

    spatial0 = np.array([[0, .452, .202, .114], [0, 0, .234, .149], [0, 0, 0, .385], [0, 0, 0, 0]])
    spatial1 = np.array([[0, .077, .098, .057], [0, 0, .085, .050], [0, 0, 0, .060], [0, 0, 0, 0]])
    temporal1 = np.array([[.208, .064, .061, .039], [.128, .166, .087, .050], [.235, .116, .296, .117], [.315, .117, .090, .546]])
    heatmap(ax, 0.080, 0.085, spatial0, "$\\bar{M}_s^{(0)}$  spatial")
    heatmap(ax, 0.220, 0.085, spatial1, "$\\bar{M}_s^{(1)}$  spatial")
    heatmap(ax, 0.360, 0.085, temporal1, "$\\bar{M}_t^{(1)}$  temporal")
    txt(ax, 0.058, 0.139, "source", size=6.0, color=BLACK, weight="bold", ha="center", rotation=90)
    txt(ax, 0.134, 0.055, "target", size=6.0, color=BLACK, weight="bold", ha="center")
    txt(ax, 0.500, 0.201, "matrix orientation", size=6.2, color=MID, ha="center")
    txt(ax, 0.500, 0.178, "[round, source, target]", size=7.2, color=BLACK, weight="bold", ha="center")
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("importance_bar", ["#F4F5F6", "#B7DCE0", "#43A7AA", "#174E70"])
    ax.imshow(gradient, extent=[0.470, 0.548, 0.118, 0.128], origin="lower", aspect="auto", cmap=cmap, zorder=5)
    txt(ax, 0.466, 0.123, "0", size=5.4, color=MID, ha="right")
    txt(ax, 0.552, 0.123, "high", size=5.4, color=MID)
    txt(ax, 0.509, 0.098, "token-ratio importance (%)", size=5.8, color=BLACK, weight="bold", ha="center")


def draw_static_pruning(ax):
    panel_heading(ax, 0.585, 0.310, "D", "Static Pruning from the Frozen Matrix")
    txt(ax, 0.620, 0.277, r"within every $(r,\,edge\ type)$: prune the lowest $p\%$ observed edges", size=7.3, color=MID)

    spatial_mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    temporal_mask = np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    heatmap(ax, 0.622, 0.103, spatial_mask, "$P_s^{(0)}$", cell=0.021, mask=True)
    heatmap(ax, 0.733, 0.103, temporal_mask, "$P_t^{(1)}$", cell=0.021, mask=True)

    # Ranked edge strip and cutoff.
    values = np.linspace(0.02, 0.55, 12)
    cmap = LinearSegmentedColormap.from_list("rank", ["#E8EEF0", "#73BFC0", "#174E70"])
    for index, value in enumerate(values):
        ax.add_patch(Rectangle((0.622 + index * 0.016, 0.220), 0.014, 0.018, facecolor=cmap(value / .55), edgecolor=WHITE, linewidth=0.5, zorder=6))
    cutoff_x = 0.622 + 3 * 0.016 - 0.001
    ax.add_line(Line2D([cutoff_x, cutoff_x], [0.214, 0.244], color=RED, lw=1.5, zorder=8))
    txt(ax, cutoff_x, 0.252, r"$p=25\%$", size=6.2, color=RED, weight="bold", ha="center")
    txt(ax, 0.622, 0.201, "low importance", size=5.6, color=MID)
    txt(ax, 0.812, 0.201, "high", size=5.6, color=MID, ha="right")

    # Compact pruned topology.
    positions = [(0.860, 0.205), (0.925, 0.205), (0.860, 0.116), (0.925, 0.116)]
    retained = [(0, 1, GREEN), (0, 2, "#7556B0"), (1, 2, "#7556B0"), (1, 3, "#7556B0"), (2, 3, GREEN)]
    for source, target, color in retained:
        arrow(ax, positions[source], positions[target], color=color, lw=1.25, mutation=5.0, linestyle=(0, (2, 2)) if color != GREEN else "-", z=5)
    for idx, (x, y) in enumerate(positions):
        robot(ax, x, y, AGENT_COLORS[idx], scale=0.55)
        txt(ax, x, y - 0.034, f"A{idx + 1}", size=5.4, color=AGENT_COLORS[idx], weight="bold", ha="center")
    txt(ax, 0.832, 0.205, "$r=0$", size=5.8, color=MID, weight="bold", ha="right")
    txt(ax, 0.832, 0.116, "$r=1$", size=5.8, color=MID, weight="bold", ha="right")
    txt(ax, 0.8925, 0.260, "Pruned worker graph", size=7.0, color=BLACK, weight="bold", ha="center")
    txt(ax, 0.8925, 0.071, "frozen policy replay on a disjoint split", size=6.1, color=MID, ha="center")
    txt(ax, 0.8925, 0.046, "reviewer remains dense and outside the masks", size=5.8, color=AGENT_COLORS[3], weight="bold", ha="center")


def render():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(18.0, 10.1), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    txt(ax, 0.020, 0.960, "Attribution-Guided Static Pruning for Multi-Agent Systems", size=25, color=BLACK, weight="bold")
    txt(
        ax,
        0.022,
        0.918,
        "Sentence selection recovers node behavior; selected source-token ratios induce round-specific communication masks.",
        size=9.4,
        color=MID,
    )
    ax.add_line(Line2D([0.020, 0.980], [0.902, 0.902], color=BLACK, lw=1.1, zorder=2))

    draw_dense_graph(ax)
    draw_attribution(ax)
    draw_matrices(ax)
    draw_static_pruning(ax)

    # Thin dividers create a composite paper figure, not boxed workflow cards.
    ax.add_line(Line2D([0.372, 0.372], [0.345, 0.892], color="#B8B8B8", lw=0.8, zorder=2))
    ax.add_line(Line2D([0.020, 0.980], [0.325, 0.325], color="#B8B8B8", lw=0.8, zorder=2))
    ax.add_line(Line2D([0.570, 0.570], [0.035, 0.315], color="#B8B8B8", lw=0.8, zorder=2))

    outputs = {
        "svg": HERE / "agentattrprune_reference_composite.svg",
        "pdf": HERE / "agentattrprune_reference_composite.pdf",
        "png": HERE / "agentattrprune_reference_composite.png",
    }
    fig.savefig(outputs["svg"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs["pdf"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs["png"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.04, dpi=230)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for kind, path in render().items():
        print(f"{kind}: {path}")
