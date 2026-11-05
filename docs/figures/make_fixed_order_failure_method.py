#!/usr/bin/env python3
"""Draft method figure for fixed-order, correctness-aware AgentAttrPrune.

All numerical values, curves and masks are illustrative. The sparse graph is
computed from the same toy scores displayed in the heatmaps and ranking.
Generate with the agentprune environment; no model or experiment is executed.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK, MUTED = "#193146", "#607180"
BLUE, PURPLE = "#3B76AF", "#8370AA"
TEAL, ORANGE, RED = "#198A79", "#D98130", "#C65857"
PALE, LINE, WHITE = "#F3F6F8", "#D9E1E6", "#FFFFFF"


def txt(ax, x, y, value, size=11, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=20, **kw)


def box(ax, x, y, w, h, fill=WHITE, edge=LINE, radius=.09, lw=1, z=1):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=MUTED, lw=1.3, alpha=1, rad=0, dashed=False, z=3,
          mutation=9):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                        linewidth=lw, alpha=alpha, mutation_scale=mutation,
                        connectionstyle=f"arc3,rad={rad}",
                        linestyle=(0, (3, 3)) if dashed else "-",
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(p)
    return p


def line(ax, xs, ys, color=LINE, lw=1, **kw):
    ax.plot(xs, ys, color=color, lw=lw, **kw)


def node(ax, x, y, i, *, sparse=False, highlight=False):
    color = ORANGE if highlight else (TEAL if sparse else BLUE)
    if highlight:
        ax.add_patch(Circle((x, y), .235, fill=False, edgecolor=ORANGE,
                            alpha=.3, linewidth=4, zorder=6))
    ax.add_patch(Circle((x, y), .18, facecolor="#FFF4E8" if highlight else WHITE,
                        edgecolor=color, linewidth=1.6, zorder=7))
    txt(ax, x, y, f"$A_{i}$", size=11, ha="center")


def toy_scores():
    spatial0 = np.array([[0, .80, .35, .16], [0, 0, .61, .12],
                         [0, 0, 0, .72], [0, 0, 0, 0]])
    spatial1 = np.array([[0, .23, .65, .05], [0, 0, .21, .14],
                         [0, 0, 0, .56], [0, 0, 0, 0]])
    temporal = np.array([[.54, .09, .07, .02], [.26, .58, .03, .10],
                         [.45, .18, .69, .04], [.00, .31, .11, .76]])
    arrays = [spatial0, spatial1, temporal]
    valid = [np.triu(np.ones((4, 4), dtype=bool), 1)] * 2 + [np.ones((4, 4), dtype=bool)]
    edges = sorted((float(a[i, j]), k, i, j)
                   for k, a in enumerate(arrays) for i, j in zip(*np.where(valid[k])))
    masks = [v.copy() for v in valid]
    for _, k, i, j in edges[:14]:
        masks[k][i, j] = False
    return arrays, valid, edges, masks


def graph(ax, x, width, *, masks=None, sparse=False):
    xs = np.linspace(x + .46, x + width - .26, 4)
    y0, y1, yr = 6.55, 4.43, 2.90
    dense_s = np.triu(np.ones((4, 4), dtype=bool), 1)
    sm0, sm1, tm = masks if masks is not None else [dense_s, dense_s, np.ones((4, 4), dtype=bool)]
    for i in range(4):
        for j in range(4):
            if tm[i, j]:
                arrow(ax, (xs[i], y0 - .19), (xs[j], y1 + .19), PURPLE,
                      lw=1.25 if sparse else .85, alpha=.77 if sparse else .24,
                      dashed=True, mutation=6)
    for y, mask in [(y0, sm0), (y1, sm1)]:
        for i in range(4):
            for j in range(i + 1, 4):
                if mask[i, j]:
                    arrow(ax, (xs[i] + .15, y + .09), (xs[j] - .16, y + .09),
                          TEAL if sparse else BLUE, lw=1.6 if sparse else 1.2,
                          rad=-.32 - .11 * (j-i-1), mutation=7)
        for i, px in enumerate(xs):
            node(ax, px, y, i+1, sparse=sparse,
                 highlight=not sparse and y == y1 and i == 3)
    txt(ax, x + .02, y0, "$r=0$", size=10, color=MUTED, ha="center", rotation=90)
    txt(ax, x + .02, y1, "$r=1$", size=10, color=MUTED, ha="center", rotation=90)
    rx = x + width / 2 + .10
    for px in xs:
        arrow(ax, (px, y1 - .19), (rx, yr + .23), INK, lw=.85, alpha=.45, mutation=6)
    box(ax, rx-.56, yr-.23, 1.12, .46, edge=INK)
    txt(ax, rx, yr, "Reviewer", size=10.5, ha="center")
    arrow(ax, (rx, yr-.25), (rx, yr-.65), INK, lw=1.1)
    txt(ax, rx, yr-.85, "$\\hat y$", size=14, ha="center")
    txt(ax, x + width / 2, 7.31, "$A_1 \\prec A_2 \\prec A_3 \\prec A_4$", size=13, ha="center")
    txt(ax, x + width / 2, 7.65, "Same fixed agent order" if sparse else "Fixed agent execution order",
        size=10.5, color=MUTED, ha="center")


def heading(ax, x, y, letter, title, subtitle=None):
    txt(ax, x, y, letter, size=15, weight="bold", color=BLUE)
    txt(ax, x+.29, y, title, size=14, weight="bold")
    if subtitle:
        txt(ax, x, y-.39, subtitle, size=10, color=MUTED)


def sentence_input(ax):
    box(ax, 3.76, 7.63, 6.50, .66, fill=PALE, edge="none")
    txt(ax, 3.95, 8.07, "Incoming source messages", size=10.7, weight="bold")
    for i, color in enumerate([BLUE, PURPLE, BLUE]):
        x = 3.98 + i*1.62
        for j in range(4):
            box(ax, x+j*.28, 7.79, .22, .10,
                fill=color if j in (i, i+1) else "#CCD5DD", edge="none", radius=.025)
    txt(ax, 9.22, 7.99, "Task / prompt\nkept fixed", size=9, color=MUTED)
    arrow(ax, (6.98, 7.61), (6.98, 7.31), INK)
    txt(ax, 6.98, 7.12, "Is the receiving worker's original answer correct?",
        size=11, ha="center", weight="bold")
    line(ax, [5.12, 8.68], [6.88, 6.88], color=LINE, lw=1.3)
    arrow(ax, (5.12, 6.88), (5.12, 6.62), TEAL)
    arrow(ax, (8.68, 6.88), (8.68, 6.62), ORANGE)


def recovery_curve(ax):
    x0, y0, w, h = 4.15, 4.20, 2.03, .99
    line(ax, [x0, x0], [y0, y0+h], color=MUTED, lw=.8)
    line(ax, [x0, x0+w], [y0, y0], color=MUTED, lw=.8)
    line(ax, [x0, x0+w], [y0+.84, y0+.84], color=TEAL, lw=.8, linestyle=(0, (3, 3)))
    px = x0 + np.array([0, .43, .88, 1.35, 1.88])
    py = y0 + np.array([.0, .37, .68, .87, .90])
    line(ax, px[:4], py[:4], color=TEAL, lw=1.9)
    ax.scatter(px[:4], py[:4], s=16, c=TEAL, zorder=7)
    line(ax, px[3:], py[3:], color=LINE, lw=1.2, linestyle="--")
    txt(ax, x0+w, y0+h+.16, "0.95 recovery", size=9, color=TEAL, ha="right")
    txt(ax, x0+w/2, y0-.27, "Greedy insertion steps", size=9, color=MUTED, ha="center")
    txt(ax, x0+1.35, y0+.60, "stop", size=9, color=TEAL, ha="center")


def peak_curve(ax):
    x0, y0, w, h = 7.12, 4.52, 2.74, .75
    line(ax, [x0, x0], [y0, y0+h], color=MUTED, lw=.8)
    line(ax, [x0, x0+w], [y0, y0], color=MUTED, lw=.8)
    vals = np.array([.10, .28, .52, .88, .64, .72, .39, .32])
    px, py = np.linspace(x0, x0+w, len(vals)), y0+vals*.72
    line(ax, px, py, color=ORANGE, lw=1.8)
    ax.scatter(px, py, s=12, c=ORANGE, zorder=7)
    ax.scatter([px[3]], [py[3]], s=53, facecolor=WHITE, edgecolor=ORANGE, lw=1.7, zorder=8)
    line(ax, [px[3], px[3]], [y0, py[3]], color=ORANGE, lw=.9, linestyle="--")
    txt(ax, x0, y0+h+.16, "Gold-prefix logit score", size=9, color=MUTED)
    txt(ax, x0+w/2, y0-.23, "Complete greedy insertion", size=9, color=MUTED, ha="center")


def attribution(ax):
    sentence_input(ax)
    box(ax, 3.78, 2.69, 2.79, 3.88, fill="#F1F9F6", edge="#C8E4DB")
    box(ax, 6.78, 2.69, 3.48, 3.88, fill="#FFF7ED", edge="#EDCEAA")
    txt(ax, 5.18, 6.27, "Correct output", size=13, color=TEAL, weight="bold", ha="center")
    txt(ax, 5.18, 5.85, "Teacher-force the full\nrecorded output", size=11, ha="center", linespacing=1.45)
    recovery_curve(ax)
    txt(ax, 5.18, 3.54, "Select supporting\ninput sentences", size=11, ha="center", linespacing=1.4)
    txt(ax, 5.18, 2.99, "Selected-source token ratio", size=10, color=TEAL, ha="center", weight="bold")

    txt(ax, 8.52, 6.27, "Failure attribution", size=13, color=ORANGE, weight="bold", ha="center")
    txt(ax, 8.52, 5.86, "Teacher-force the gold answer prefix\nin an answer-first scoring prompt", size=10.5, ha="center", linespacing=1.45)
    peak_curve(ax)
    txt(ax, 8.52, 3.93, "Select $S^*$ at the earliest score maximum", size=10.1, ha="center", color=ORANGE)
    arrow(ax, (8.52, 3.69), (8.52, 3.48), ORANGE)
    box(ax, 7.05, 2.89, 2.94, .57, fill=WHITE, edge=ORANGE)
    txt(ax, 8.52, 3.18, "Verify once WITHOUT gold\nOriginal prompt + selected sentences", size=10, ha="center", linespacing=1.5)

    arrow(ax, (5.18, 2.67), (5.18, 2.33), TEAL)
    line(ax, [8.52, 8.52], [2.89, 2.57], color=ORANGE, lw=1.2, zorder=4)
    line(ax, [6.17, 8.52], [2.57, 2.57], color=TEAL, lw=1.2, zorder=4)
    line(ax, [8.52, 9.57], [2.57, 2.57], color=RED, lw=1.2, zorder=4)
    arrow(ax, (6.17, 2.57), (6.17, 2.33), TEAL)
    arrow(ax, (9.57, 2.57), (9.57, 2.33), RED)
    txt(ax, 6.39, 2.43, "pass", size=8.5, color=TEAL)
    txt(ax, 9.08, 2.43, "fail", size=8.5, color=RED)
    box(ax, 3.78, 1.63, 6.48, .69, fill=WHITE, edge=LINE)
    txt(ax, 4.02, 2.08, "Correct / rescued", size=10, color=TEAL, weight="bold")
    txt(ax, 4.02, 1.83, "Token ratio for each incoming edge", size=10)
    line(ax, [6.82, 6.82], [1.76, 2.19], color=LINE)
    txt(ax, 7.02, 2.08, "Rescue failed", size=10, color=RED, weight="bold")
    txt(ax, 7.02, 1.83, "0 for every observed incoming edge", size=10)
    txt(ax, 7.02, 1.20, "Local verification leaves the recorded dense trace unchanged.",
        size=10, color=MUTED, ha="center")


def heatmap(ax, data, valid, x, y, title, cell=.19):
    cmap = LinearSegmentedColormap.from_list("weights", ["#FFFFFF", "#C2E2DC", TEAL, INK])
    for i in range(4):
        for j in range(4):
            p = Rectangle((x+j*cell, y+(3-i)*cell), cell, cell,
                          facecolor=cmap(float(data[i, j])) if valid[i, j] else "#DCE1E6",
                          edgecolor="#D2DBE2", linewidth=.4, zorder=4,
                          hatch=None if valid[i, j] else "//")
            ax.add_patch(p)
    txt(ax, x+2*cell, y+.98, title, size=12, ha="center")


def scoring(ax, arrays, valid, edges):
    txt(ax, 12.15, 7.96, "Sentence evidence → edge score", size=11, ha="center", weight="bold")
    txt(ax, 12.15, 7.46,
        r"$a_e(x)=\frac{|\mathrm{selected\ source\ tokens}|}{|\mathrm{visible\ source\ tokens}|}$",
        size=12.5, ha="center")
    txt(ax, 12.15, 6.98, "Use 0 if local rescue fails", size=10.5, color=RED, ha="center")
    arrow(ax, (12.15, 6.72), (12.15, 6.39), TEAL)
    txt(ax, 12.15, 6.11, r"$\bar a_e=\frac{\sum_{x:\ e\ \mathrm{observed}} a_e(x)}{\#\{x:e\ \mathrm{observed}\}}$",
        size=14, ha="center")
    txt(ax, 12.15, 5.55, "ALL calibration samples", size=11, ha="center", weight="bold")
    txt(ax, 12.15, 5.25, "Failed rescues stay in the denominator", size=9.5, color=RED, ha="center")
    for a, v, x, title in zip(arrays, valid, [10.84, 11.78, 12.72],
                              [r"$\bar M_s^{(0)}$", r"$\bar M_s^{(1)}$", r"$\bar M_t^{(1)}$"]):
        heatmap(ax, a, v, x, 3.91, title)
    txt(ax, 12.15, 3.63, "Rows: source  ·  Columns: target", size=9, color=MUTED, ha="center")
    txt(ax, 12.15, 3.35, "Hatched = absent; white = observed zero", size=9, color=MUTED, ha="center")
    arrow(ax, (12.15, 3.10), (12.15, 2.85), TEAL)
    txt(ax, 12.15, 2.63, "Rank all worker edges globally", size=11, ha="center", weight="bold")
    x0, y0, bw = 10.94, 1.62, .079
    for n, (score, *_rest) in enumerate(edges):
        ax.add_patch(Rectangle((x0+n*.089, y0), bw, .08+score*.68,
                               facecolor=RED if n < 14 else TEAL, edgecolor="none", zorder=4))
    line(ax, [x0+14*.089-.008]*2, [y0-.10, y0+.73], color=INK, lw=.8, linestyle="--")
    txt(ax, 11.47, 1.34, "prune", size=10, color=RED, ha="center")
    txt(ax, 12.75, 1.34, "retain", size=10, color=TEAL, ha="center")
    txt(ax, 12.15, .99, "Bottom p% across rounds + edge types", size=10, ha="center")


def render():
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    arrays, valid, edges, masks = toy_scores()
    fig = plt.figure(figsize=(18, 10.1), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 18), ylim=(0, 10.1), aspect="equal")
    ax.axis("off")

    txt(ax, .35, 9.71, "Fixed-order, failure-aware attribution for communication pruning",
        size=20, weight="bold")
    txt(ax, .35, 9.30, "OFFLINE CALIBRATION", size=10, weight="bold", color=BLUE)
    txt(ax, 3.05, 9.30, "gold available for scoring and correctness checks", size=10, color=MUTED)
    line(ax, [.35, 13.67], [9.08, 9.08], color=LINE, lw=1.2)
    txt(ax, 14.05, 9.30, "INFERENCE  ·  NO GOLD", size=10, weight="bold", color=TEAL)
    line(ax, [14.05, 17.66], [9.08, 9.08], color=LINE, lw=1.2)

    heading(ax, .35, 8.71, "a", "Dense fixed-order MAS", "Collect and freeze the original traces")
    heading(ax, 3.78, 8.71, "b", "Correctness-aware sentence attribution")
    heading(ax, 10.74, 8.71, "c", "Score and prune")
    heading(ax, 14.05, 8.71, "d", "Frozen sparse MAS", "Reuse one static mask on held-out tasks")
    graph(ax, .42, 2.90)
    txt(ax, 1.90, 1.78, "Keep correct AND incorrect traces", size=10.5, ha="center", weight="bold")
    txt(ax, 1.90, 1.34, "Attribute each receiving worker\nin each reasoning round", size=10.5, ha="center", color=MUTED, linespacing=1.5)
    line(ax, [3.35, 3.53, 3.53], [4.43, 4.43, 7.96], color=ORANGE, lw=1.2, zorder=3)
    arrow(ax, (3.53, 7.96), (3.72, 7.96), ORANGE, lw=1.2, mutation=9)
    txt(ax, 3.43, 5.73, "node-level view", size=8.5, color=ORANGE,
        ha="center", rotation=90)
    attribution(ax)
    arrow(ax, (10.32, 1.98), (10.67, 1.98), TEAL, lw=1.7, mutation=11)
    scoring(ax, arrays, valid, edges)
    line(ax, [13.84, 13.84], [.95, 8.98], color=LINE, lw=1.1, linestyle=(0, (4, 4)))
    arrow(ax, (13.54, 2.63), (14.12, 2.63), TEAL, lw=1.7, mutation=11)
    graph(ax, 14.20, 3.20, masks=masks, sparse=True)
    txt(ax, 15.90, 1.78, "Prune whole communication edges", size=10.5, ha="center", weight="bold")
    txt(ax, 15.90, 1.34, "Agent order fixed · reviewer edges retained\nExample: 14 / 28 worker edges pruned", size=10, color=MUTED, ha="center", linespacing=1.5)

    line(ax, [.35, 17.66], [.71, .71], color=LINE)
    line(ax, [.45, .82], [.40, .40], color=BLUE, lw=1.6)
    txt(ax, .92, .40, "same-round messages", size=10, color=MUTED)
    line(ax, [3.23, 3.60], [.40, .40], color=PURPLE, lw=1.4, linestyle=(0, (3, 3)))
    txt(ax, 3.70, .40, "previous-round messages (including self)", size=10, color=MUTED)
    txt(ax, 17.60, .40, "DRAFT  ·  Graphs, scores and curves are illustrative", size=10, color=MUTED, ha="right")

    stem = OUT / "fixed_order_failure_method_draft"
    for ext in ("svg", "pdf", "png"):
        path = stem.with_suffix("."+ext)
        fig.savefig(path, dpi=180 if ext == "png" else 300, facecolor=WHITE)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    render()
