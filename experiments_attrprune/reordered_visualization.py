"""Static scientific figures for the two fixed-order pruning experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments_attrprune.common import load_json, write_json
from experiments_attrprune.random_order_visualization import _read_arrays
from experiments_attrprune.visualize_pruning_matrices import AGENT_DISPLAY_NAMES


def ordered_block(aggregate: Mapping, order_plan: Mapping, edge: str, round_index: int) -> dict:
    """Return a display-only permutation; temporal rows belong to the prior round."""
    arrays = _read_arrays(aggregate)
    agents = list(aggregate["agents"])
    orders = order_plan["round_orders"]
    if list(order_plan["agents"]) != agents or len(orders) != aggregate["num_rounds"]:
        raise ValueError("Order plan and aggregate identity differ")
    if any(len(order) != len(agents) or set(order) != set(agents) for order in orders):
        raise ValueError("Each round order must be an agent permutation")
    source_order = orders[max(0, round_index - 1)] if edge == "temporal" else orders[round_index]
    target_order = orders[round_index]
    source_indices = [agents.index(agent) for agent in source_order]
    target_indices = [agents.index(agent) for agent in target_order]
    indexer = np.ix_(source_indices, target_indices)
    return {"source_agents": list(source_order), "target_agents": list(target_order),
            "source_indices": source_indices, "target_indices": target_indices,
            **{key: value[round_index][indexer] for key, value in arrays[edge].items()}}


def render_matrices(aggregate: Mapping, order_plan: Mapping, policies: Mapping[int, Mapping],
                    output_dir: Path, *, branch: str, comparison: Mapping | None = None) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle

    if branch not in {"direct", "recalibrate"}:
        raise ValueError("Unknown reordered branch")
    arrays = _read_arrays(aggregate)
    blocks = [(0, "spatial"), (1, "spatial"), (1, "temporal")]
    displayed = {(r, edge): ordered_block(aggregate, order_plan, edge, r) for r, edge in blocks}
    agents = list(aggregate["agents"])
    if aggregate["num_rounds"] != 2:
        raise ValueError("This experiment requires two rounds")
    for edge in ("spatial", "temporal"):
        candidates = np.asarray(order_plan["candidate_masks"][edge], dtype=bool)
        if not np.array_equal(arrays[edge]["opportunity_count"] > 0, candidates):
            raise ValueError("Every ordered candidate needs an observation")
        for policy in policies.values():
            mask = np.asarray(policy["masks"][edge], dtype=bool)
            if mask.shape != candidates.shape or np.any(mask & ~candidates):
                raise ValueError("Pruned policy contains a forbidden candidate")
            if list(policy["agents"]) != agents or policy["round_orders"] != order_plan["round_orders"]:
                raise ValueError("Policy identity/order changed")
    if comparison is not None:
        if comparison["agents"] != aggregate["agents"]:
            raise ValueError("Comparison identities differ")
        comparison_blocks = {(r, edge): ordered_block(comparison, order_plan, edge, r) for r, edge in blocks}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eff1f4")
    importance_norm = Normalize(0, 1)
    count_norm = Normalize(0, 40)
    files = {}
    branch_name = "Direct projection" if branch == "direct" else "Re-attributed fixed order"

    def panel(axis, data, edge, r, mask=None, counts=False, prefix=""):
        values = data["opportunity_count" if counts else "conditional_mean"]
        observed = data["opportunity_count"] > 0
        retained = observed if mask is None else mask
        norm = count_norm if counts else importance_norm
        axis.imshow(np.ma.masked_where(~retained, values), cmap=cmap, norm=norm, interpolation="nearest")
        labels = lambda names: [AGENT_DISPLAY_NAMES.get(a, a) for a in names]
        axis.set_xticks(range(len(agents)), labels=labels(data["target_agents"]), rotation=30, ha="right", fontsize=13)
        axis.set_yticks(range(len(agents)), labels=labels(data["source_agents"]), fontsize=13)
        axis.set_xlabel(f"target · round {r}", fontsize=13)
        axis.set_ylabel(f"source · round {r - 1 if edge == 'temporal' else r}", fontsize=13)
        title = f"Round {r} · {edge}"
        if mask is not None:
            title += f" · kept {int(retained.sum())}/{int(observed.sum())}"
        axis.set_title((prefix + "\n" if prefix else "") + title, fontsize=14, weight="bold", pad=12)
        for i in range(len(agents)):
            for j in range(len(agents)):
                if not observed[i, j]:
                    axis.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="#e2e7ef",
                                             edgecolor="#c4cbd6", hatch="//", linewidth=.3))
                    text, color = "0", "#637084"
                elif not retained[i, j]:
                    text, color = "0", "#ad4a42"
                else:
                    value = values[i, j]
                    text = str(int(value)) if counts else f"{value:.2f}"
                    color = "#122034" if norm(value) > .65 else "white"
                axis.text(j, i, text, ha="center", va="center", fontsize=14, weight="bold", color=color)

    def save(stem, rows, *, counts=False, mask_policy=None, compare=False):
        with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none"}):
            fig, axes = plt.subplots(rows, 3, figsize=(18, 7.2 if rows == 1 else 12.5), squeeze=False)
            fig.subplots_adjust(left=.075, right=.88, bottom=.27 if rows == 1 else .19,
                                top=.80 if rows == 1 else .85, wspace=.33, hspace=.58)
            if compare:
                title = "Fixed order · direct projection vs re-attribution"
                for row, data, label in ((0, comparison_blocks, "Direct projection"),
                                         (1, displayed, "Re-attributed fixed order")):
                    for col, (r, edge) in enumerate(blocks):
                        panel(axes[row, col], data[r, edge], edge, r, prefix=label)
            else:
                title = f"{branch_name} · " + ("observation counts" if counts else "token-ratio importance")
                if mask_policy is not None:
                    title += f" · global {round(mask_policy['pruning_rate'] * 100)}% pruning"
                for col, (r, edge) in enumerate(blocks):
                    data = displayed[r, edge]
                    mask = None if mask_policy is None else np.asarray(mask_policy["masks"][edge][r], dtype=bool)[
                        np.ix_(data["source_indices"], data["target_indices"])]
                    panel(axes[0, col], data, edge, r, mask=mask, counts=counts)
            fig.suptitle(title, y=.97, fontsize=23, weight="bold")
            fig.text(.5, .91 if rows == 1 else .92,
                     "Same dev40 · round-specific fixed orders · source rows / target columns", ha="center", fontsize=14)
            bar_axis = fig.add_axes([.905, .36 if rows == 1 else .29, .012, .34 if rows == 1 else .44])
            bar = fig.colorbar(ScalarMappable(norm=count_norm if counts else importance_norm, cmap=cmap), cax=bar_axis)
            bar.set_label("Observed rollouts" if counts else "Conditional mean token ratio", fontsize=12)
            handles = [Patch(facecolor="#e2e7ef", edgecolor="#c4cbd6", hatch="//", label="Excluded by fixed order / self-link")]
            if mask_policy is not None:
                handles.append(Patch(facecolor="#eff1f4", label="Pruned edge (gray 0)"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .085 if rows == 1 else .065),
                       ncol=len(handles), frameon=False, fontsize=12)
            fig.text(.5, .046 if rows == 1 else .035,
                     "Colored 0 = observed zero importance. Round 0 temporal is absent. Reviewer inputs stay dense.",
                     ha="center", fontsize=11)
            files[stem] = {}
            for extension in ("png", "svg"):
                path = output_dir / f"{stem}.{extension}"
                temporary = path.with_name(f"{stem}.tmp.{extension}")
                fig.savefig(temporary, dpi=170, facecolor="white")
                temporary.replace(path)
                files[stem][extension] = str(path.resolve())
            plt.close(fig)

    save("ordered_importance", 1)
    save("ordered_observations", 1, counts=True)
    for percent, policy in policies.items():
        save(f"ordered_global_prune_{percent}pct", 1, mask_policy=policy)
    if comparison is not None:
        save("direct_vs_recalibrated", 2, compare=True)

    data_json = {"agents": agents, "round_orders": order_plan["round_orders"],
                 "matrix_orientation": "[round][source][target]", "canonical_metrics": aggregate["metrics"],
                 "display_blocks": {f"round_{r}_{edge}": {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in data.items()} for (r, edge), data in displayed.items()},
                 "policy_masks": {str(p): policy["masks"] for p, policy in policies.items()}}
    write_json(output_dir / "ordered_matrices.json", data_json)
    rows = []
    for r, edge in blocks:
        data = displayed[r, edge]
        for i, source in enumerate(data["source_agents"]):
            for j, target in enumerate(data["target_agents"]):
                row = {"round": r, "type": edge, "source_round": r - 1 if edge == "temporal" else r,
                       "source": source, "target": target, "source_position": i, "target_position": j,
                       "sum": data["sum"][i, j], "opportunity_count": int(data["opportunity_count"][i, j]),
                       "conditional_mean": data["conditional_mean"][i, j]}
                for p, policy in policies.items():
                    row[f"retained_{p}pct"] = int(policy["masks"][edge][r][agents.index(source)][agents.index(target)])
                rows.append(row)
    with (output_dir / "ordered_edges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "ordering_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "agent", "position", "row_sum", "column_sum", "score"])
        writer.writeheader()
        for r, order in enumerate(order_plan["round_orders"]):
            scores = order_plan["scores"][r]
            for position, agent in enumerate(order):
                index = agents.index(agent)
                writer.writerow({"round": r, "agent": agent, "position": position,
                    **{key: scores[key][index] for key in ("row_sum", "column_sum", "score")}})
    manifest = {"complete": True, "branch": branch, "files": files,
                "importance_color_limits": [0, 1], "count_color_limits": [0, 40],
                "round_orders": order_plan["round_orders"], "matrices_json": str((output_dir / "ordered_matrices.json").resolve()),
                "edges_csv": str((output_dir / "ordered_edges.csv").resolve()),
                "scores_csv": str((output_dir / "ordering_scores.csv").resolve())}
    write_json(output_dir / "visualization_manifest.json", manifest)
    return manifest


def render_run(ctx) -> dict:
    from experiments_attrprune.reordered_mmlu import load_policies, validate_ready_for_evaluation
    aggregate = validate_ready_for_evaluation(ctx)
    policies = load_policies(ctx)
    paths = [ctx.paths.run_config, ctx.paths.run_dir / "order_plan.json",
             ctx.paths.attribution / "aggregate_edge_matrices.json"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    # A direct projection of the same source supplies the comparator without
    # depending on another branch's process, directory, or publication timing.
    comparison = None
    if ctx.branch == "recalibrate":
        from AgentAttrPrune.pruning.reordered import project_aggregate
        comparison = project_aggregate(ctx.source_aggregate, ctx.order_plan)
    output_dir = ctx.paths.attribution / "visualizations" / "ordered"
    result = render_matrices(aggregate, ctx.order_plan, policies, output_dir,
                             branch=ctx.branch, comparison=comparison)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected for path, expected in hashes.items()):
        raise ValueError("Rendering input changed")
    result["source_sha256"] = hashes
    write_json(output_dir / "visualization_manifest.json", result)
    return result
