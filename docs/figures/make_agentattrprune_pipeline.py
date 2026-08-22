#!/usr/bin/env python3
"""Render the paper-style AgentAttrPrune pipeline overview.

The figure is deliberately self-contained and uses only Matplotlib so that the
SVG/PDF remain vector-editable and the PNG can be used for quick previews.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent

NAVY = "#17324D"
INK = "#263746"
MUTED = "#607181"
BLUE = "#2F6FB0"
BLUE_LIGHT = "#E8F1FA"
TEAL = "#168C83"
TEAL_LIGHT = "#E4F4F1"
PURPLE = "#7256A8"
PURPLE_LIGHT = "#EEEAF7"
ORANGE = "#E58A2B"
ORANGE_LIGHT = "#FFF0DE"
RED = "#C94B4B"
GRAY = "#AAB5BF"
GRAY_LIGHT = "#F2F4F6"
PANEL_EDGE = "#D7E0E7"
WHITE = "#FFFFFF"


def rounded(ax, x, y, w, h, *, fc=WHITE, ec=PANEL_EDGE, lw=1.0, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
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
    color=INK,
    lw=1.2,
    style="-|>",
    mutation=8,
    alpha=1.0,
    connection="arc3",
    linestyle="-",
    z=2,
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


def label(ax, x, y, text, *, size=8, color=INK, weight="normal", ha="left", va="center", z=8, **kwargs):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=z,
        **kwargs,
    )


def node(ax, x, y, text, *, r=0.016, fc=WHITE, ec=BLUE, lw=1.5, text_color=NAVY, z=6):
    circle = Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(circle)
    label(ax, x, y, text, size=7.2, color=text_color, weight="bold", ha="center", z=z + 1)
    return circle


def panel(ax, x0, x1, title, letter, tint):
    rounded(ax, x0, 0.105, x1 - x0, 0.82, fc=tint, ec=PANEL_EDGE, lw=1.0, radius=0.012, z=0)
    rounded(ax, x0 + 0.010, 0.872, 0.028, 0.034, fc=NAVY, ec=NAVY, radius=0.008, z=4)
    label(ax, x0 + 0.024, 0.889, letter, size=10, color=WHITE, weight="bold", ha="center")
    label(ax, x0 + 0.045, 0.889, title, size=11.2, color=NAVY, weight="bold")


def cross(ax, x, y, *, size=0.006, color=RED, lw=1.5, z=9):
    ax.add_line(Line2D([x - size, x + size], [y - size, y + size], color=color, lw=lw, zorder=z))
    ax.add_line(Line2D([x - size, x + size], [y + size, y - size], color=color, lw=lw, zorder=z))


def dense_graph_panel(ax):
    x0, x1 = 0.014, 0.244
    panel(ax, x0, x1, "Dense multi-agent trace", "a", "#F8FBFD")
    label(ax, x0 + 0.016, 0.844, "All candidate communication edges are active", size=7.6, color=MUTED)

    xs = [x0 + 0.055, x0 + 0.100, x0 + 0.145, x0 + 0.190]
    y0, y1 = 0.673, 0.392

    rounded(ax, x0 + 0.014, 0.766, 0.064, 0.044, fc=WHITE, ec=BLUE, radius=0.008, z=4)
    label(ax, x0 + 0.046, 0.788, "Task  $x$", size=8.2, color=NAVY, weight="bold", ha="center")
    arrow(ax, (x0 + 0.078, 0.788), (x0 + 0.126, y0 + 0.058), color=BLUE, lw=1.25, mutation=7, z=3)

    # Complete temporal N x N communication, drawn first so worker nodes remain crisp.
    for source_x in xs:
        for target_x in xs:
            arrow(
                ax,
                (source_x, y0 - 0.018),
                (target_x, y1 + 0.018),
                color=PURPLE,
                lw=0.65,
                alpha=0.22,
                mutation=4.5,
                linestyle=(0, (2.2, 2.0)),
                z=1,
            )

    # Ordered same-round complete DAG: A_i -> A_j for i < j.
    for round_y in (y0, y1):
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                rad = -0.10 - 0.06 * (j - i - 1)
                arrow(
                    ax,
                    (xs[i] + 0.016, round_y),
                    (xs[j] - 0.017, round_y),
                    color=BLUE,
                    lw=0.95,
                    alpha=0.75,
                    mutation=5.2,
                    connection=f"arc3,rad={rad}",
                    z=3,
                )

    for idx, xpos in enumerate(xs, start=1):
        node(ax, xpos, y0, f"A{idx}", fc=WHITE, ec=BLUE)
        node(ax, xpos, y1, f"A{idx}", fc=WHITE, ec=BLUE)

    rounded(ax, x0 + 0.014, y0 - 0.020, 0.031, 0.040, fc=BLUE_LIGHT, ec="none", radius=0.006, z=5)
    label(ax, x0 + 0.0295, y0, "$r=0$", size=6.8, color=BLUE, weight="bold", ha="center")
    rounded(ax, x0 + 0.014, y1 - 0.020, 0.031, 0.040, fc=BLUE_LIGHT, ec="none", radius=0.006, z=5)
    label(ax, x0 + 0.0295, y1, "$r=1$", size=6.8, color=BLUE, weight="bold", ha="center")

    reviewer_y = 0.190
    rounded(ax, x0 + 0.084, reviewer_y - 0.026, 0.092, 0.052, fc=WHITE, ec=NAVY, radius=0.010, z=5)
    label(ax, x0 + 0.130, reviewer_y + 0.007, "Dense reviewer", size=7.7, color=NAVY, weight="bold", ha="center")
    label(ax, x0 + 0.130, reviewer_y - 0.012, "final decision", size=6.7, color=MUTED, ha="center")
    for xpos in xs:
        arrow(ax, (xpos, y1 - 0.018), (x0 + 0.130, reviewer_y + 0.027), color=NAVY, lw=0.65, alpha=0.42, mutation=4.5, z=2)

    ax.add_line(Line2D([x0 + 0.024, x0 + 0.046], [0.130, 0.130], color=BLUE, lw=1.4, zorder=8))
    label(ax, x0 + 0.050, 0.130, "spatial DAG", size=6.4, color=MUTED)
    ax.add_line(Line2D([x0 + 0.118, x0 + 0.140], [0.130, 0.130], color=PURPLE, lw=1.2, linestyle=(0, (2, 2)), zorder=8))
    label(ax, x0 + 0.144, 0.130, "temporal", size=6.4, color=MUTED)


def attribution_panel(ax):
    x0, x1 = 0.257, 0.495
    panel(ax, x0, x1, "Node-wise attribution", "b", "#FFFCF7")
    label(ax, x0 + 0.016, 0.844, "Every worker-round event + the final reviewer", size=7.6, color=MUTED)

    # Incoming source outputs and receiving node.
    source_x = x0 + 0.038
    source_ys = [0.772, 0.708, 0.644]
    source_names = ["A1 output", "A2 output", "A3 history"]
    source_colors = [BLUE, BLUE, PURPLE]
    for sy, name, color in zip(source_ys, source_names, source_colors):
        rounded(ax, source_x, sy - 0.020, 0.068, 0.040, fc=WHITE, ec=color, radius=0.007, z=5)
        label(ax, source_x + 0.034, sy, name, size=6.9, color=INK, weight="bold", ha="center")

    target_x, target_y = x0 + 0.190, 0.708
    node(ax, target_x, target_y, "A$_b$", r=0.023, fc=WHITE, ec=ORANGE, lw=1.8, text_color=NAVY)
    label(ax, target_x, target_y - 0.041, "target  $(b,r)$", size=6.7, color=MUTED, ha="center")
    for sy, color in zip(source_ys, source_colors):
        arrow(ax, (source_x + 0.070, sy), (target_x - 0.025, target_y), color=color, lw=1.0, alpha=0.75, mutation=5.5, z=3)

    # Exact prompt decomposition.
    prompt_x, prompt_y, prompt_w, prompt_h = x0 + 0.018, 0.322, 0.124, 0.246
    rounded(ax, prompt_x, prompt_y, prompt_w, prompt_h, fc=WHITE, ec=PANEL_EDGE, radius=0.009, z=4)
    label(ax, prompt_x + 0.010, prompt_y + prompt_h - 0.024, "Rendered prompt  $P_S$", size=7.5, color=NAVY, weight="bold")
    rows = [
        ("fixed task + template", BLUE_LIGHT, BLUE, "lock"),
        ("q1 : sentence 1", ORANGE_LIGHT, ORANGE, "1"),
        ("q1 : sentence 2", GRAY_LIGHT, GRAY, ""),
        ("q2 : sentence 1", ORANGE_LIGHT, ORANGE, "2"),
        ("q3 : sentence 1", GRAY_LIGHT, GRAY, ""),
    ]
    row_y = prompt_y + prompt_h - 0.060
    for text, fc, ec, badge in rows:
        rounded(ax, prompt_x + 0.009, row_y - 0.015, prompt_w - 0.018, 0.030, fc=fc, ec=ec, lw=0.75, radius=0.005, z=5)
        label(ax, prompt_x + 0.015, row_y, text, size=6.35, color=INK)
        if badge:
            badge_text = "fixed" if badge == "lock" else f"step {badge}"
            label(ax, prompt_x + prompt_w - 0.014, row_y, badge_text, size=5.7, color=ec, weight="bold", ha="right")
        row_y -= 0.037

    arrow(ax, (target_x, target_y - 0.025), (prompt_x + prompt_w * 0.72, prompt_y + prompt_h + 0.010), color=ORANGE, lw=1.1, mutation=6.5, connection="arc3,rad=0.20", z=3)

    # Teacher-forced scoring and recovery.
    model_x, model_y = x0 + 0.162, 0.454
    rounded(ax, model_x, model_y - 0.036, 0.052, 0.072, fc=NAVY, ec=NAVY, radius=0.009, z=5)
    label(ax, model_x + 0.026, model_y + 0.010, "LLM", size=8.2, color=WHITE, weight="bold", ha="center")
    label(ax, model_x + 0.026, model_y - 0.012, "teacher force", size=5.9, color="#D9E5EF", ha="center")
    arrow(ax, (prompt_x + prompt_w + 0.006, model_y), (model_x - 0.006, model_y), color=INK, lw=1.1, mutation=6.0, z=4)

    rounded(ax, model_x + 0.002, 0.328, 0.050, 0.060, fc=TEAL_LIGHT, ec=TEAL, radius=0.008, z=5)
    label(ax, model_x + 0.027, 0.367, "recorded $y$", size=6.2, color=MUTED, ha="center")
    label(ax, model_x + 0.027, 0.345, "$s(S)=\\sum_t z_t[y_t]$", size=7.3, color=TEAL, weight="bold", ha="center")
    arrow(ax, (model_x + 0.026, model_y - 0.038), (model_x + 0.026, 0.391), color=TEAL, lw=1.0, mutation=5.5, z=4)

    rounded(ax, x0 + 0.018, 0.196, 0.202, 0.091, fc=ORANGE_LIGHT, ec="#F0C38E", radius=0.009, z=3)
    label(ax, x0 + 0.028, 0.266, "Exact greedy insertion", size=7.4, color=ORANGE, weight="bold")
    label(ax, x0 + 0.028, 0.237, "choose $j_k$ to minimize $|s(S\\cup\\{v_j\\})-s_*|$", size=6.9, color=INK)
    label(ax, x0 + 0.028, 0.208, "shortest prefix with recovery $\\geq 0.95$", size=6.6, color=MUTED)

    label(ax, x0 + 0.018, 0.150, "orange = selected incoming sentences", size=6.5, color=ORANGE, weight="bold")
    label(ax, x0 + 0.018, 0.129, "fixed prompt content is never masked", size=6.5, color=BLUE, weight="bold")


def draw_matrix(ax, x, y, data, title, *, cell=0.0205):
    cmap = LinearSegmentedColormap.from_list("importance", ["#F3F6F8", "#B9DDE0", TEAL, NAVY])
    n = data.shape[0]
    label(ax, x + n * cell / 2, y + n * cell + 0.025, title, size=7.2, color=NAVY, weight="bold", ha="center")
    for i in range(n):
        for j in range(n):
            value = float(data[i, j])
            rect = Rectangle(
                (x + j * cell, y + (n - 1 - i) * cell),
                cell,
                cell,
                facecolor=cmap(min(1.0, value / 0.55)),
                edgecolor=WHITE,
                linewidth=0.7,
                zorder=5,
            )
            ax.add_patch(rect)
            if value > 0:
                label(
                    ax,
                    x + (j + 0.5) * cell,
                    y + (n - i - 0.5) * cell,
                    f"{value:.2f}",
                    size=4.8,
                    color=WHITE if value > 0.28 else INK,
                    ha="center",
                )
    for idx in range(n):
        label(ax, x - 0.004, y + (n - idx - 0.5) * cell, f"A{idx + 1}", size=5.4, color=MUTED, ha="right")
        label(ax, x + (idx + 0.5) * cell, y - 0.008, f"A{idx + 1}", size=5.4, color=MUTED, ha="center", va="top")


def matrix_panel(ax):
    x0, x1 = 0.508, 0.742
    panel(ax, x0, x1, "Edge-importance matrices", "c", "#F7FCFB")
    label(ax, x0 + 0.016, 0.844, "Aggregate verified-correct calibration samples only", size=7.5, color=MUTED)

    # Sample matrix stack to aggregate matrix.
    for k in range(3):
        rounded(ax, x0 + 0.020 + 0.008 * k, 0.744 - 0.007 * k, 0.052, 0.058, fc=WHITE, ec=TEAL, radius=0.006, z=3 + k)
        label(ax, x0 + 0.046 + 0.008 * k, 0.773 - 0.007 * k, f"$M^{{x_{k+1}}}$", size=7.0, color=TEAL, weight="bold", ha="center", z=7)
    label(ax, x0 + 0.026, 0.716, "correct samples $x\\in\\mathcal{C}$", size=6.4, color=MUTED)
    arrow(ax, (x0 + 0.094, 0.763), (x0 + 0.135, 0.763), color=TEAL, lw=1.3, mutation=7)
    rounded(ax, x0 + 0.141, 0.728, 0.071, 0.070, fc=TEAL, ec=TEAL, radius=0.009, z=4)
    label(ax, x0 + 0.1765, 0.775, "aggregate", size=6.2, color="#DDF4F1", ha="center")
    label(ax, x0 + 0.1765, 0.751, "$\\bar{M}$", size=13, color=WHITE, weight="bold", ha="center")

    # Representative values from a compact four-agent run, used only to make the
    # matrix orientation and ranking visually concrete.
    spatial = np.array(
        [
            [0.00, 0.45, 0.20, 0.11],
            [0.00, 0.00, 0.23, 0.15],
            [0.00, 0.00, 0.00, 0.38],
            [0.00, 0.00, 0.00, 0.00],
        ]
    )
    temporal = np.array(
        [
            [0.21, 0.06, 0.06, 0.04],
            [0.13, 0.17, 0.09, 0.05],
            [0.23, 0.12, 0.30, 0.12],
            [0.31, 0.12, 0.09, 0.55],
        ]
    )
    draw_matrix(ax, x0 + 0.030, 0.499, spatial, "spatial  $\\bar{M}^{(0)}_s$")
    draw_matrix(ax, x0 + 0.139, 0.499, temporal, "temporal  $\\bar{M}^{(1)}_t$")
    label(ax, x0 + 0.012, 0.540, "source", size=5.8, color=MUTED, rotation=90, ha="center")
    label(ax, x0 + 0.071, 0.485, "target", size=5.8, color=MUTED, ha="center", va="top")
    label(ax, x0 + 0.180, 0.485, "target", size=5.8, color=MUTED, ha="center", va="top")

    # Importance scale.
    cmap = LinearSegmentedColormap.from_list("importance_bar", ["#F3F6F8", "#B9DDE0", TEAL, NAVY])
    gradient = np.linspace(0, 1, 120).reshape(1, -1)
    ax.imshow(gradient, extent=[x0 + 0.055, x0 + 0.185, 0.426, 0.440], origin="lower", aspect="auto", cmap=cmap, zorder=4)
    label(ax, x0 + 0.050, 0.433, "low", size=5.8, color=MUTED, ha="right")
    label(ax, x0 + 0.190, 0.433, "high", size=5.8, color=MUTED)
    label(ax, x0 + 0.120, 0.457, "token-ratio importance", size=6.3, color=TEAL, weight="bold", ha="center")

    rounded(ax, x0 + 0.015, 0.256, 0.204, 0.123, fc=WHITE, ec="#B9DDD9", radius=0.009, z=3)
    label(ax, x0 + 0.025, 0.353, r"$T_q=|\mathrm{selected\ source\ tokens}|\,/\,|\mathrm{source\ output\ tokens}|$", size=6.9, color=INK)
    label(ax, x0 + 0.025, 0.315, r"$\bar{M}[r,a,b]=\sum_{x\in\mathcal{C}}M^x[r,a,b]\,/\,\sum_{x\in\mathcal{C}}O^x[r,a,b]$", size=6.9, color=INK)
    label(ax, x0 + 0.025, 0.278, "default: token_ratio / conditional_mean", size=6.4, color=TEAL, weight="bold")

    rounded(ax, x0 + 0.036, 0.151, 0.162, 0.055, fc=NAVY, ec=NAVY, radius=0.009, z=4)
    label(ax, x0 + 0.117, 0.181, "matrix orientation", size=5.9, color="#C8D7E4", ha="center")
    label(ax, x0 + 0.117, 0.161, "[round, source, target]", size=7.5, color=WHITE, weight="bold", ha="center")


def mask_matrix(ax, x, y, mask, *, cell=0.017):
    n = mask.shape[0]
    for i in range(n):
        for j in range(n):
            kept = bool(mask[i, j])
            rect = Rectangle(
                (x + j * cell, y + (n - 1 - i) * cell),
                cell,
                cell,
                facecolor=TEAL if kept else WHITE,
                edgecolor="#CCD7DD",
                linewidth=0.65,
                zorder=5,
            )
            ax.add_patch(rect)
            if kept:
                label(ax, x + (j + 0.5) * cell, y + (n - i - 0.5) * cell, "1", size=5.4, color=WHITE, weight="bold", ha="center")
            elif i < j:
                cross(ax, x + (j + 0.5) * cell, y + (n - i - 0.5) * cell, size=0.004, color=RED, lw=1.0)


def pruning_panel(ax):
    x0, x1 = 0.755, 0.986
    panel(ax, x0, x1, "Static pruning & replay", "d", "#FBFAFD")
    label(ax, x0 + 0.016, 0.844, "Freeze one deterministic policy before evaluation", size=7.5, color=MUTED)

    label(ax, x0 + 0.016, 0.795, "rank within each (round, edge type)", size=7.1, color=NAVY, weight="bold")
    values = np.linspace(0.10, 0.95, 10)
    cmap = LinearSegmentedColormap.from_list("rank", ["#E7ECEF", "#93CAC8", TEAL, NAVY])
    start_x, sq_w = x0 + 0.018, 0.018
    for idx, value in enumerate(values):
        rect = Rectangle((start_x + idx * (sq_w + 0.002), 0.742), sq_w, 0.028, facecolor=cmap(value), edgecolor=WHITE, linewidth=0.6, zorder=5)
        ax.add_patch(rect)
        if idx < 3:
            cross(ax, start_x + idx * (sq_w + 0.002) + sq_w / 2, 0.756, size=0.006, color=RED, lw=1.5)
    label(ax, start_x, 0.724, r"lowest $p\%$ pruned", size=6.3, color=RED, weight="bold")
    label(ax, start_x + 0.197, 0.724, "importance", size=6.1, color=MUTED, ha="right")

    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )
    label(ax, x0 + 0.052, 0.654, "binary mask", size=7.2, color=NAVY, weight="bold", ha="center")
    mask_matrix(ax, x0 + 0.018, 0.549, mask)

    arrow(ax, (x0 + 0.092, 0.585), (x0 + 0.121, 0.585), color=INK, lw=1.2, mutation=7)

    # Replayed sparse graph; gray dashed edges show removed candidates.
    gx = [x0 + 0.141, x0 + 0.187]
    gy = [0.630, 0.548]
    positions = [(gx[0], gy[0]), (gx[1], gy[0]), (gx[0], gy[1]), (gx[1], gy[1])]
    retained_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    pruned_edges = [(0, 3), (1, 2)]
    for s, t in retained_edges:
        arrow(ax, positions[s], positions[t], color=TEAL, lw=1.5, mutation=6, z=3)
    for s, t in pruned_edges:
        arrow(ax, positions[s], positions[t], color=GRAY, lw=1.0, alpha=0.8, mutation=5, linestyle=(0, (2, 2)), z=2)
        mx = (positions[s][0] + positions[t][0]) / 2
        my = (positions[s][1] + positions[t][1]) / 2
        cross(ax, mx, my, size=0.005, color=RED, lw=1.2)
    for idx, (nx, ny) in enumerate(positions, start=1):
        node(ax, nx, ny, f"A{idx}", r=0.015, fc=WHITE, ec=TEAL, lw=1.4, text_color=NAVY)
    label(ax, x0 + 0.164, 0.664, "pruned graph", size=7.2, color=NAVY, weight="bold", ha="center")

    label(ax, x0 + 0.018, 0.489, "retained", size=6.2, color=TEAL, weight="bold")
    ax.add_line(Line2D([x0 + 0.061, x0 + 0.081], [0.489, 0.489], color=TEAL, lw=1.6, zorder=8))
    label(ax, x0 + 0.097, 0.489, "pruned", size=6.2, color=RED, weight="bold")
    ax.add_line(Line2D([x0 + 0.137, x0 + 0.157], [0.489, 0.489], color=GRAY, lw=1.0, linestyle=(0, (2, 2)), zorder=8))
    cross(ax, x0 + 0.147, 0.489, size=0.004, color=RED, lw=1.0)

    rounded(ax, x0 + 0.026, 0.369, 0.179, 0.070, fc=PURPLE_LIGHT, ec="#CFC3E6", radius=0.010, z=4)
    label(ax, x0 + 0.1155, 0.414, "Frozen RoundPolicy", size=8.1, color=PURPLE, weight="bold", ha="center")
    label(ax, x0 + 0.1155, 0.389, "spatial + temporal masks", size=6.7, color=MUTED, ha="center")
    arrow(ax, (x0 + 0.1155, 0.365), (x0 + 0.1155, 0.315), color=PURPLE, lw=1.3, mutation=7)
    rounded(ax, x0 + 0.039, 0.253, 0.153, 0.057, fc=NAVY, ec=NAVY, radius=0.010, z=4)
    label(ax, x0 + 0.1155, 0.287, "Disjoint evaluation split", size=7.6, color=WHITE, weight="bold", ha="center")
    label(ax, x0 + 0.1155, 0.266, "sparse communication replay", size=6.1, color="#C9D8E4", ha="center")

    rounded(ax, x0 + 0.026, 0.148, 0.179, 0.056, fc=WHITE, ec=PANEL_EDGE, radius=0.009, z=4)
    label(ax, x0 + 0.1155, 0.181, "reviewer remains dense", size=7.0, color=NAVY, weight="bold", ha="center")
    label(ax, x0 + 0.1155, 0.161, "outside worker $N\\times N$ masks", size=6.1, color=MUTED, ha="center")


def protocol_ribbon(ax):
    rounded(ax, 0.014, 0.018, 0.972, 0.060, fc=NAVY, ec=NAVY, radius=0.012, z=1)
    stages = [
        (0.040, "1", "dense calibration traces"),
        (0.270, "2", "attribute every node"),
        (0.493, "3", "aggregate correct samples"),
        (0.736, "4", "freeze policy, then evaluate"),
    ]
    for idx, (x, number, text) in enumerate(stages):
        circ = Circle((x, 0.048), 0.013, facecolor=WHITE, edgecolor=WHITE, linewidth=0.8, zorder=5)
        ax.add_patch(circ)
        label(ax, x, 0.048, number, size=7.0, color=NAVY, weight="bold", ha="center")
        label(ax, x + 0.020, 0.048, text, size=7.3, color=WHITE, weight="bold")
        if idx < len(stages) - 1:
            next_x = stages[idx + 1][0]
            arrow(ax, (x + 0.183, 0.048), (next_x - 0.020, 0.048), color="#7890A4", lw=0.9, mutation=5.5, z=3)


def render():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15.6, 5.8), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(
        ax,
        0.5,
        0.967,
        "AgentAttrPrune: from dense communication to a frozen sparse policy",
        size=15.2,
        color=NAVY,
        weight="bold",
        ha="center",
    )

    dense_graph_panel(ax)
    attribution_panel(ax)
    matrix_panel(ax)
    pruning_panel(ax)

    for xpos in (0.249, 0.500, 0.748):
        arrow(ax, (xpos - 0.006, 0.515), (xpos + 0.006, 0.515), color=ORANGE, lw=2.0, mutation=10, z=12)

    protocol_ribbon(ax)

    outputs = {
        "svg": ROOT / "agentattrprune_pipeline_overview.svg",
        "pdf": ROOT / "agentattrprune_pipeline_overview.pdf",
        "png": ROOT / "agentattrprune_pipeline_overview.png",
    }
    fig.savefig(outputs["svg"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs["pdf"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs["png"], facecolor=WHITE, bbox_inches="tight", pad_inches=0.05, dpi=240)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    for kind, output in render().items():
        print(f"{kind}: {output}")
