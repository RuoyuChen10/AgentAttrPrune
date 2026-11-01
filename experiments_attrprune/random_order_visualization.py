"""CPU-only diagnostics for repeated, randomly ordered dense rollouts.

Matrix axes always follow stable agent identities. Missing observations are
masked separately from observed zero importance; pooled means weight rounds by
their edge observation counts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments_attrprune.visualize_pruning_matrices import AGENT_DISPLAY_NAMES


EDGE_TYPES = ("spatial", "temporal")
ORIENTATION = "[round][source][target]"


def _read_arrays(aggregate: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    agents = list(aggregate["agents"])
    rounds = int(aggregate["num_rounds"])
    if not agents or len(set(agents)) != len(agents) or rounds < 1:
        raise ValueError("Aggregate requires unique agents and positive round count")
    if aggregate.get("matrix_orientation") != ORIENTATION:
        raise ValueError(f"Aggregate matrix orientation must be {ORIENTATION}")
    shape = (rounds, len(agents), len(agents))
    arrays = {}
    for edge_type in EDGE_TYPES:
        payload = aggregate["metrics"]["token_ratio"][edge_type]
        item = {
            key: np.asarray(payload[key], dtype=np.float64)
            for key in ("sum", "opportunity_count", "conditional_mean")
        }
        if any(value.shape != shape for value in item.values()):
            raise ValueError(f"Unexpected {edge_type} matrix shape; expected {shape}")
        if any(not np.isfinite(value).all() for value in item.values()):
            raise ValueError(f"Nonfinite values in {edge_type} matrices")
        counts, sums = item["opportunity_count"], item["sum"]
        if (counts < 0).any() or not np.array_equal(counts, np.floor(counts)):
            raise ValueError("Observation counts must be nonnegative integers")
        if (sums < 0).any() or (sums > counts + 1e-9).any():
            raise ValueError("Token-ratio sums must lie between zero and count")
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        if not np.allclose(means, item["conditional_mean"], atol=1e-9, rtol=1e-8):
            raise ValueError("conditional_mean must equal sum / opportunity_count")
        structural = np.ones(shape, dtype=bool)
        if edge_type == "spatial":
            diagonal = np.arange(len(agents))
            structural[:, diagonal, diagonal] = False
        else:
            structural[0] = False
        if counts[~structural].any() or sums[~structural].any():
            raise ValueError("Structurally forbidden edges cannot have observations")
        item["structural_allowed"] = structural
        arrays[edge_type] = item
    return arrays


def _sample_count(aggregate: Mapping[str, Any]) -> int:
    return int(aggregate.get("num_aggregated_samples", aggregate.get("num_correct_samples", 0)))


def _draw_matrix(axis, values, counts, agents, *, title, edge_type, norm, cmap,
                 integer=False, unavailable=False):
    from matplotlib.patches import Rectangle

    axis.set_title(title, fontsize=11)
    if unavailable:
        axis.set_facecolor("#e8edf2")
        axis.text(0.5, 0.5, "N/A\nNo previous round", ha="center", va="center",
                  transform=axis.transAxes, fontsize=15, color="#475569")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    shown = np.ma.masked_where(counts <= 0, values)
    axis.imshow(shown, norm=norm, cmap=cmap)
    labels = [AGENT_DISPLAY_NAMES.get(agent, agent) for agent in agents]
    axis.set_xticks(range(len(agents)), labels=labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(agents)), labels=labels, fontsize=8)
    axis.set_xlabel("Target", fontsize=9)
    axis.set_ylabel("Source", fontsize=9)
    for source in range(len(agents)):
        for target in range(len(agents)):
            if edge_type == "spatial" and source == target:
                axis.add_patch(Rectangle((target - 0.5, source - 0.5), 1, 1,
                                         facecolor="#e8edf2", edgecolor="#bcc6d1", hatch="//"))
                label, color = "0\nforbidden", "#475569"
            elif counts[source, target] == 0:
                label, color = "—", "#64748b"
            else:
                value = float(values[source, target])
                label = str(int(value)) if integer else f"{value:.2f}"
                color = "white" if norm(value) < 0.55 else "#17202a"
            axis.text(target, source, label, ha="center", va="center", color=color, fontsize=8)


def _save_figure(fig, output_dir: Path, stem: str) -> dict[str, str]:
    paths = {}
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        temporary = path.with_name(f"{stem}.tmp.{suffix}")
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        temporary.replace(path)
        paths[suffix] = str(path.resolve())
    return paths


def render_random_order_visualizations(
    aggregate: Mapping[str, Any],
    output_dir: Path,
    reference_aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Write importance/count figures, fixed-order comparison, CSV, and manifest.

    The caller validates run provenance and rollout completeness. This function
    validates matrix compatibility and arithmetic, but imposes no 40/400 budget.
    CSV ``structural_allowed`` describes possible edge types, not whether the
    edge happened to occur. Missing permitted cells have a blank mean in CSV and
    a null mean in pooled JSON; forbidden spatial self edges are explicitly zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    for key in ("agents", "agent_roles"):
        if list(aggregate.get(key) or []) != list(reference_aggregate.get(key) or []):
            raise ValueError(f"Reference and random-order {key} must match in identity order")
    if int(aggregate["num_rounds"]) != int(reference_aggregate["num_rounds"]):
        raise ValueError("Reference and random-order round counts must match")
    arrays = _read_arrays(aggregate)
    reference = _read_arrays(reference_aggregate)
    agents = [str(agent) for agent in aggregate["agents"]]
    rounds = int(aggregate["num_rounds"])
    random_count, reference_count = _sample_count(aggregate), _sample_count(reference_aggregate)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e8edf2")
    intensity_norm = Normalize(vmin=0.0, vmax=1.0)
    count_max = max(1, max(int(item["opportunity_count"].max()) for item in arrays.values()))
    count_norm = Normalize(vmin=0, vmax=count_max)
    footnote = ("Rows: source; columns: target. Spatial diagonal: 0 (forbidden). "
                "Gray —: not observed. Observed zero remains 0.")
    files: dict[str, Any] = {}
    with plt.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        for key, title, norm, integer in (
            ("conditional_mean", "Conditional mean token ratio", intensity_norm, False),
            ("opportunity_count", "Edge observation count", count_norm, True),
        ):
            fig, axes = plt.subplots(rounds, 2, figsize=(12, 5 * rounds), squeeze=False,
                                     constrained_layout=True)
            try:
                for round_index in range(rounds):
                    for column, edge_type in enumerate(EDGE_TYPES):
                        item = arrays[edge_type]
                        _draw_matrix(axes[round_index, column], item[key][round_index],
                                     item["opportunity_count"][round_index], agents,
                                     title=f"Round {round_index} · {edge_type}", edge_type=edge_type,
                                     norm=norm, cmap=cmap, integer=integer,
                                     unavailable=edge_type == "temporal" and round_index == 0)
                fig.suptitle(f"Random order · {random_count} rollouts · {title}\n{footnote}", fontsize=12)
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                             shrink=0.75, label=title)
                files[key] = _save_figure(fig, output_dir, f"random_order_{key}")
            finally:
                plt.close(fig)

        spatial = arrays["spatial"]
        pooled_sum = spatial["sum"].sum(axis=0)
        pooled_count = spatial["opportunity_count"].sum(axis=0)
        pooled_mean = np.divide(pooled_sum, pooled_count, out=np.zeros_like(pooled_sum),
                                where=pooled_count > 0)
        pooled_norm = Normalize(vmin=0, vmax=max(1, int(pooled_count.max())))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        try:
            for axis, values, norm, title, integer in (
                (axes[0], pooled_mean, intensity_norm, "Pooled spatial importance", False),
                (axes[1], pooled_count, pooled_norm, "Pooled spatial observation count", True),
            ):
                _draw_matrix(axis, values, pooled_count, agents, title=title,
                             edge_type="spatial", norm=norm, cmap=cmap, integer=integer)
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, shrink=0.8)
            fig.suptitle("Spatial across rounds: sum of importance / sum of observations\n" + footnote,
                         fontsize=11)
            files["pooled_spatial"] = _save_figure(fig, output_dir, "random_order_pooled_spatial")
        finally:
            plt.close(fig)

        fig, axes = plt.subplots(rounds, 4, figsize=(23, 5 * rounds), squeeze=False,
                                 constrained_layout=True)
        try:
            for round_index in range(rounds):
                for edge_index, edge_type in enumerate(EDGE_TYPES):
                    for variant, (label, values, count) in enumerate((
                        ("Fixed order", reference, reference_count),
                        ("Random order", arrays, random_count),
                    )):
                        item = values[edge_type]
                        _draw_matrix(axes[round_index, edge_index * 2 + variant],
                                     item["conditional_mean"][round_index],
                                     item["opportunity_count"][round_index], agents,
                                     title=f"Round {round_index} · {edge_type}\n{label} · {count} rollouts",
                                     edge_type=edge_type, norm=intensity_norm, cmap=cmap,
                                     unavailable=edge_type == "temporal" and round_index == 0)
            fig.suptitle("Fixed-order reference vs random order · conditional mean token ratio\n" + footnote,
                         fontsize=13)
            fig.colorbar(ScalarMappable(norm=intensity_norm, cmap=cmap), ax=axes.ravel().tolist(),
                         shrink=0.75, label="Conditional mean token ratio (shared scale)")
            files["reference_comparison"] = _save_figure(fig, output_dir, "random_order_vs_fixed_order")
        finally:
            plt.close(fig)

    csv_path = output_dir / "random_order_edges.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "round", "type", "source", "target", "sum", "count", "conditional_mean",
            "observed", "structural_allowed",
        ))
        writer.writeheader()
        for round_index in range(rounds):
            for edge_type in EDGE_TYPES:
                item = arrays[edge_type]
                for source, source_id in enumerate(agents):
                    for target, target_id in enumerate(agents):
                        cell = (round_index, source, target)
                        count = int(item["opportunity_count"][cell])
                        allowed = bool(item["structural_allowed"][cell])
                        mean = float(item["conditional_mean"][cell]) if count else ("" if allowed else 0.0)
                        writer.writerow({
                            "round": round_index, "type": edge_type, "source": source_id,
                            "target": target_id, "sum": float(item["sum"][cell]), "count": count,
                            "conditional_mean": mean, "observed": count > 0,
                            "structural_allowed": allowed,
                        })
    files["edges_csv"] = str(csv_path)
    pooled_allowed = ~np.eye(len(agents), dtype=bool)
    pooled_json = {
        "schema_version": 1, "metric": "token_ratio", "edge_type": "spatial",
        "matrix_orientation": "[source][target]", "agents": agents,
        "agent_roles": list(aggregate.get("agent_roles") or []),
        "rounds_pooled": list(range(rounds)), "num_rollouts": random_count,
        "definition": "sum across rounds and rollouts / observed edge count across rounds and rollouts",
        "sum": pooled_sum.tolist(), "opportunity_count": pooled_count.astype(np.int64).tolist(),
        "conditional_mean": [
            [float(pooled_mean[i, j]) if pooled_count[i, j] or not pooled_allowed[i, j] else None
             for j in range(len(agents))] for i in range(len(agents))
        ],
        "observed": (pooled_count > 0).tolist(), "structural_allowed": pooled_allowed.tolist(),
    }
    pooled_path = output_dir / "pooled_spatial.json"
    pooled_path.write_text(json.dumps(pooled_json, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files["pooled_spatial_json"] = str(pooled_path)
    manifest_path = output_dir / "visualization_manifest.json"
    manifest = {
        "schema_version": 1, "metric": "token_ratio", "statistic": "conditional_mean",
        "matrix_orientation": ORIENTATION, "agents": agents,
        "display_labels": [AGENT_DISPLAY_NAMES.get(agent, agent) for agent in agents],
        "agent_roles": list(aggregate.get("agent_roles") or []), "num_rounds": rounds,
        "num_random_rollouts": random_count, "num_reference_rollouts": reference_count,
        "run_fingerprint": aggregate.get("run_fingerprint"),
        "reference_run_fingerprint": reference_aggregate.get("run_fingerprint"),
        "intensity_color_limits": [0.0, 1.0], "count_color_limits": [0, count_max],
        "pooled_count_color_limits": [0, max(1, int(pooled_count.max()))],
        "missing_observation_display": "masked; distinct from observed zero",
        "spatial_diagonal": "0; structurally forbidden",
        "round_zero_temporal": "N/A; no previous round",
        "pooled_definition": pooled_json["definition"], "files": files,
        "manifest_json": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
