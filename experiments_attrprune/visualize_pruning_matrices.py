"""Render aggregate attribution matrices with a frozen pruning policy overlaid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EDGE_TYPES = ("spatial", "temporal")
PRUNED_FACE = "#f0f1f3"
PRUNED_EDGE = "#a7adb5"
PRUNED_MARK = "#c53b32"
UNAVAILABLE_FACE = "#dfe5ee"
AGENT_DISPLAY_NAMES = {
    "knowledgeable_expert": "Knowl.",
    "critic": "Critic",
    "mathematician": "Math.",
    "psychologist": "Psych.",
    "historian": "Hist.",
    "math_solver": "Solver",
    "mathematical_analyst": "Anal.",
    "programming_expert": "Prog.",
    "inspector": "Insp.",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _inferred_aggregate_path(policy_path: Path) -> Path:
    declared = policy_path_data(policy_path).get("static_sweep", {}).get(
        "aggregate_json"
    )
    if declared:
        declared_path = Path(str(declared)).expanduser()
        if declared_path.is_file():
            return declared_path.resolve()

    # .../attribution/static_pruning/<metric>_<statistic>[_global]/policies/policy.json
    inferred = policy_path.parents[3] / "aggregate_edge_matrices.json"
    if inferred.is_file():
        return inferred.resolve()
    raise FileNotFoundError(
        "Could not find aggregate_edge_matrices.json from the policy metadata "
        f"or relative to {policy_path}"
    )


def policy_path_data(path: Path) -> dict[str, Any]:
    """Read a policy payload; kept separate to make path inference explicit."""

    return _load_json(path)


def _validate_inputs(
    aggregate: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    metric: str,
    statistic: str,
) -> None:
    if list(aggregate.get("agents") or ()) != list(policy.get("agents") or ()):
        raise ValueError("Aggregate and policy use different agent orders")
    if int(aggregate.get("num_rounds", 0)) != int(policy.get("num_rounds", 0)):
        raise ValueError("Aggregate and policy use different round counts")
    if aggregate.get("matrix_orientation") != "[round][source][target]":
        raise ValueError("Aggregate matrix orientation must be [round][source][target]")
    if policy.get("matrix_orientation") != "[round][source][target]":
        raise ValueError("Policy matrix orientation must be [round][source][target]")
    if metric not in (aggregate.get("metrics") or {}):
        raise ValueError(f"Aggregate does not contain metric {metric!r}")
    for edge_type in EDGE_TYPES:
        edge_payload = aggregate["metrics"][metric][edge_type]
        if statistic not in edge_payload:
            raise ValueError(
                f"Aggregate does not contain {metric}/{edge_type}/{statistic}"
            )
        if edge_type not in (policy.get("masks") or {}):
            raise ValueError(f"Policy does not contain a {edge_type!r} mask")


def _requested_percentage(policy: Mapping[str, Any]) -> int:
    sweep = policy.get("static_sweep") or {}
    if "requested_pruning_percent" in sweep:
        return int(sweep["requested_pruning_percent"])
    return int(round(float(policy.get("pruning_rate", 0.0)) * 100))


def _plot_policy(
    aggregate: Mapping[str, Any],
    policy: Mapping[str, Any],
    output_paths: Sequence[Path],
    *,
    dpi: int,
    calibration_label: str | None = None,
    color_max: float | None = None,
    intensity_only: bool = False,
    title_override: str | None = None,
    colorbar_label: str = "Edge importance",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    metric = str(policy.get("metric", "token_ratio"))
    statistic = str(policy.get("statistic", "conditional_mean"))
    _validate_inputs(aggregate, policy, metric=metric, statistic=statistic)

    agents = [str(agent) for agent in aggregate["agents"]]
    agent_labels = [AGENT_DISPLAY_NAMES.get(agent, agent) for agent in agents]
    short_labels = all(len(label) <= 6 for label in agent_labels)
    num_agents = len(agents)
    num_rounds = int(aggregate["num_rounds"])
    matrices = aggregate["metrics"][metric]
    masks = policy["masks"]

    candidate_total = 0
    retained_total = 0
    maximum_value = 0.0
    arrays: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for round_index in range(num_rounds):
        for edge_type in EDGE_TYPES:
            values = np.asarray(
                matrices[edge_type][statistic][round_index], dtype=np.float64
            )
            candidates = np.asarray(
                matrices[edge_type]["candidate_count"][round_index], dtype=np.float64
            ) > 0
            retained = np.asarray(masks[edge_type][round_index], dtype=np.int64) > 0
            if intensity_only:
                retained = candidates.copy()
            if values.shape != (num_agents, num_agents):
                raise ValueError(
                    f"Unexpected {edge_type} matrix shape in round {round_index}: "
                    f"{values.shape}"
                )
            if candidates.shape != values.shape or retained.shape != values.shape:
                raise ValueError(
                    f"Candidate/value/mask shape mismatch for {edge_type} round "
                    f"{round_index}"
                )
            if np.any(retained & ~candidates):
                raise ValueError(
                    f"Policy retains non-candidate {edge_type} cells in round "
                    f"{round_index}"
                )
            candidate_total += int(candidates.sum())
            retained_total += int((retained & candidates).sum())
            if np.any(candidates):
                maximum_value = max(maximum_value, float(values[candidates].max()))
            arrays[(round_index, edge_type)] = (values, candidates, retained)

    vmax = max(0.7, math.ceil(maximum_value * 10.0) / 10.0)
    if color_max is not None:
        if not math.isfinite(color_max) or color_max <= 0:
            raise ValueError("color_max must be finite and positive")
        if color_max < maximum_value:
            raise ValueError("color_max would clip candidate attribution values")
        vmax = color_max
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white" if intensity_only else UNAVAILABLE_FACE)
    norm = Normalize(vmin=0.0, vmax=vmax)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 26,
            "axes.titleweight": "bold",
            "axes.labelsize": 26,
            "axes.labelweight": "bold",
            "xtick.labelsize": 22 if short_labels else 18,
            "ytick.labelsize": 22 if short_labels else 18,
        }
    )
    # Size the canvas around square panels so labels do not create wide gutters.
    panel_size = 4.5
    left_margin = 1.45 if short_labels else 3.2
    bottom_margin = 1.25 if short_labels else 2.0
    if not intensity_only:
        bottom_margin += 0.95
    row_gap = 1.40 + 8 / 72 if short_labels else 2.4
    column_gap = 0.50
    right_margin = 1.65
    top_margin = 2.10 if intensity_only else 1.50
    figure_width = left_margin + 2 * panel_size + column_gap + right_margin
    figure_height = (
        bottom_margin + num_rounds * panel_size
        + (num_rounds - 1) * row_gap + top_margin
    )
    fig, axes = plt.subplots(
        num_rounds,
        len(EDGE_TYPES),
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=left_margin / figure_width,
        right=1 - right_margin / figure_width,
        bottom=bottom_margin / figure_height,
        top=1 - top_margin / figure_height,
        wspace=column_gap / panel_size,
        hspace=row_gap / panel_size,
    )

    image = None
    for round_index in range(num_rounds):
        for column, edge_type in enumerate(EDGE_TYPES):
            axis = axes[round_index][column]
            values, candidates, retained = arrays[(round_index, edge_type)]
            shown = np.where(candidates & retained, values, np.nan)
            image = axis.imshow(
                shown,
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
            axis.set_title(
                f"round {round_index} · {edge_type}",
                pad=12,
            )
            axis.set_xticks(
                range(num_agents), labels=agent_labels,
                rotation=35 if short_labels else 45,
                ha="right", rotation_mode="anchor",
            )
            if column == 0:
                axis.set_yticks(range(num_agents), labels=agent_labels)
                axis.set_ylabel("source", labelpad=8)
            else:
                axis.set_yticks([])
            if round_index == num_rounds - 1:
                axis.set_xlabel("target", labelpad=8)
            axis.tick_params(axis="both", width=1.2, length=5)
            for source in range(num_agents):
                for target in range(num_agents):
                    if not candidates[source, target]:
                        continue
                    value = float(values[source, target])
                    if not retained[source, target]:
                        axis.add_patch(
                            Rectangle(
                                (target - 0.5, source - 0.5),
                                1.0,
                                1.0,
                                facecolor=PRUNED_FACE,
                                edgecolor=PRUNED_EDGE,
                                linewidth=0.8,
                                zorder=2,
                            )
                        )
                    axis.text(
                        target,
                        source if retained[source, target] else source + 0.08,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=(
                            "black"
                            if retained[source, target] and value >= 0.55
                            else "white"
                            if retained[source, target]
                            else "#5f6670"
                        ),
                        fontsize=21 if retained[source, target] else 20,
                        fontweight="bold",
                        zorder=3,
                    )
                    if not retained[source, target]:
                        axis.text(
                            target + 0.31,
                            source - 0.31,
                            "×",
                            ha="center",
                            va="center",
                            color=PRUNED_MARK,
                            fontsize=20,
                            fontweight="bold",
                            zorder=4,
                        )
            for spine in axis.spines.values():
                spine.set_linewidth(1.2)
            axis.set_xlim(-0.5, num_agents - 0.5)
            axis.set_ylim(num_agents - 0.5, -0.5)

    percentage = _requested_percentage(policy)
    allocation = str(policy.get("pruning_allocation", ""))
    scope_label = "global " if allocation == "global" else ""
    aggregate_label = (
        "All-sample aggregate"
        if aggregate.get("aggregation_scope", "correct") == "all"
        else "Correct-only aggregate"
    )
    aggregate_count = int(aggregate.get(
        "num_aggregated_samples", aggregate.get("num_correct_samples", 0)
    ))
    aggregate_label += f" (n={aggregate_count})"
    title = (
        f"{calibration_label or aggregate_label}\n"
        f"{scope_label.capitalize()}{percentage}% pruning · "
        f"retained {retained_total}/{candidate_total}"
    )
    if intensity_only:
        title = (
            f"{calibration_label or aggregate_label}\n"
            "Attribution intensity"
        )
    if title_override is not None:
        title = title_override
    fig.suptitle(
        title,
        fontsize=26,
        fontweight="bold",
        y=1 - 0.10 / figure_height,
    )
    legend_handles = [
        Patch(
            facecolor=cmap(norm(0.35)),
            edgecolor="none",
            label=f"Retained ({retained_total})",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markersize=15,
            markerfacecolor=PRUNED_FACE,
            markeredgecolor=PRUNED_EDGE,
            linestyle="none",
            label=f"Pruned × ({candidate_total - retained_total})",
        ),
    ]
    if intensity_only:
        fig.text(
            0.50, 1 - 1.05 / figure_height, "Blank cells: no candidate edge",
            ha="center", fontsize=20,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.10 / figure_height),
            borderaxespad=0,
            ncol=2,
            frameon=False,
            fontsize=plt.rcParams["xtick.labelsize"],
            handletextpad=0.6,
            columnspacing=2.2,
        )
    if image is not None:
        panel_bottom = bottom_margin / figure_height
        panel_top = 1 - top_margin / figure_height
        colorbar_height = 0.60 * (panel_top - panel_bottom)
        colorbar_axis = fig.add_axes((
            (figure_width - right_margin + 0.35) / figure_width,
            (panel_bottom + panel_top - colorbar_height) / 2,
            0.24 / figure_width,
            colorbar_height,
        ))
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis
        )
        colorbar.set_label(colorbar_label, fontsize=26, labelpad=12)
        colorbar.ax.tick_params(labelsize=20, width=1.1, length=5)
        colorbar.outline.set_linewidth(1.1)

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f"{output_path.stem}.tmp{output_path.suffix}"
        )
        fig.savefig(
            temporary,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.12,
        )
        temporary.replace(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay one or more frozen pruning policies on the calibration "
            "aggregate attribution matrices."
        )
    )
    parser.add_argument("policies", type=Path, nargs="+")
    parser.add_argument(
        "--aggregate-json",
        type=Path,
        default=None,
        help=(
            "Aggregate matrix JSON. By default it is resolved from each policy's "
            "static_sweep metadata or neighboring attribution directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the policy sweep directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg"),
        default=("png", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--calibration-label", default=None)
    parser.add_argument(
        "--intensity-only", action="store_true",
        help="Show all candidate attribution values without a pruning overlay; non-candidates are blank.",
    )
    parser.add_argument(
        "--color-max", type=float, default=None,
        help="Shared color-scale maximum for comparing multiple aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for raw_policy_path in args.policies:
        policy_path = raw_policy_path.expanduser().resolve()
        policy = policy_path_data(policy_path)
        aggregate_path = (
            args.aggregate_json.expanduser().resolve()
            if args.aggregate_json is not None
            else _inferred_aggregate_path(policy_path)
        )
        aggregate = _load_json(aggregate_path)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else policy_path.parent.parent
        )
        metric = str(policy.get("metric", "token_ratio"))
        output_paths = [
            output_dir / f"{policy_path.stem}_{metric}_matrices_viridis.{extension}"
            for extension in dict.fromkeys(args.formats)
        ]
        _plot_policy(
            aggregate, policy, output_paths, dpi=args.dpi,
            calibration_label=args.calibration_label, color_max=args.color_max,
            intensity_only=args.intensity_only,
        )
        for output_path in output_paths:
            print(output_path.resolve())


if __name__ == "__main__":
    main()
