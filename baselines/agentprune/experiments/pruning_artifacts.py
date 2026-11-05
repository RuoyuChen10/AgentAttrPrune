"""Save and visualize the edge policy learned by AgentPrune."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


SCHEMA_VERSION = 1
MATRIX_ORIENTATION = "[source_agent][target_agent]"


def _tensor_matrix(tensor: torch.Tensor, size: int, name: str) -> torch.Tensor:
    values = tensor.detach().cpu().reshape(-1)
    expected = size * size
    if values.numel() != expected:
        raise ValueError(
            f"{name} has {values.numel()} entries, but {size} agents require {expected}."
        )
    return values.reshape(size, size)


def build_pruning_artifact(
    graph: Any,
    training_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a trained Graph's logits and masks into JSON-compatible matrices."""
    nodes = list(graph.nodes.values())
    size = len(nodes)
    if size == 0:
        raise ValueError("Cannot export a graph without worker nodes.")

    workers = [
        {
            "index": index,
            "id": node.id,
            "agent_name": graph.agent_names[index],
            "agent_type": node.node_name,
            "role": node.role,
        }
        for index, node in enumerate(nodes)
    ]

    matrices: Dict[str, Dict[str, Any]] = {}
    checkpoint: Dict[str, torch.Tensor] = {}
    for edge_type in ("spatial", "temporal"):
        logits = _tensor_matrix(
            getattr(graph, f"{edge_type}_logits"), size, f"{edge_type}_logits"
        )
        probabilities = torch.sigmoid(logits)
        masks = _tensor_matrix(
            getattr(graph, f"{edge_type}_masks"), size, f"{edge_type}_masks"
        )
        effective_probabilities = probabilities * masks
        matrices[edge_type] = {
            "optimized": bool(getattr(graph, f"optimized_{edge_type}")),
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "masks": masks.to(dtype=torch.int64).tolist(),
            "effective_probabilities": effective_probabilities.tolist(),
        }
        checkpoint[f"{edge_type}_logits"] = logits.clone()
        checkpoint[f"{edge_type}_masks"] = masks.clone()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "matrix_orientation": MATRIX_ORIENTATION,
        "description": {
            "probabilities": "sigmoid of the learned edge logits",
            "masks": "binary edges retained by iterative pruning",
            "effective_probabilities": (
                "mask-applied sampling probabilities; spatial edges can still be "
                "rejected by the DAG cycle check"
            ),
        },
        "workers": workers,
        "matrices": matrices,
        "training_config": dict(training_config or {}),
    }
    artifact["_checkpoint_tensors"] = checkpoint
    return artifact


def _json_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "_checkpoint_tensors"}


def render_pruning_heatmaps(
    artifact: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Render probabilities, masks, and effective probabilities as a 2x3 heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workers = artifact["workers"]
    labels = [f"{item['index']}:{item['agent_name']}" for item in workers]
    columns = (
        ("probabilities", "Learned probability"),
        ("masks", "Retained mask"),
        ("effective_probabilities", "Effective probability"),
    )
    edge_types = ("spatial", "temporal")
    size = len(labels)
    figure, axes = plt.subplots(
        len(edge_types),
        len(columns),
        figsize=(max(12.0, 2.0 * size), max(7.5, 1.45 * size)),
        constrained_layout=True,
    )

    for row, edge_type in enumerate(edge_types):
        for column, (matrix_name, title) in enumerate(columns):
            axis = axes[row, column]
            matrix = np.asarray(
                artifact["matrices"][edge_type][matrix_name], dtype=float
            )
            image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"{edge_type.capitalize()} — {title}")
            axis.set_xlabel("Target agent")
            axis.set_ylabel("Source agent")
            axis.set_xticks(range(size), labels=labels, rotation=45, ha="right")
            axis.set_yticks(range(size), labels=labels)
            if size <= 10:
                for source in range(size):
                    for target in range(size):
                        value = matrix[source, target]
                        text = f"{int(value)}" if matrix_name == "masks" else f"{value:.2f}"
                        color = "white" if value < 0.55 else "black"
                        axis.text(
                            target,
                            source,
                            text,
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                        )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        "AgentPrune learned edge policy (matrix: source → target)", fontsize=14
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def render_round_edge_heatmaps(
    artifact: Mapping[str, Any],
    output_path: Path,
    metric: str = "binary",
    statistic: str = "conditional_mean",
) -> Path:
    """Render round-aware attribution matrices or a pruned policy JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "masks" in artifact:
        matrices = artifact["masks"]
        value_label = f"{artifact.get('metric', metric)} retained mask"
    elif "metrics" in artifact:
        if metric not in artifact["metrics"]:
            raise ValueError(
                f"Metric {metric!r} is unavailable; choose from {list(artifact['metrics'])}."
            )
        matrices = {
            edge_type: artifact["metrics"][metric][edge_type][statistic]
            for edge_type in ("spatial", "temporal")
        }
        value_label = f"{metric} {statistic}"
    else:
        raise ValueError("Expected an aggregate edge matrix or pruned policy JSON.")

    agent_names = artifact.get("agents") or []
    agent_roles = artifact.get("agent_roles") or []
    labels = [
        f"{index}:{agent_roles[index] if index < len(agent_roles) else name}"
        for index, name in enumerate(agent_names)
    ]
    size = len(labels)
    if size == 0:
        size = len(matrices["spatial"][0])
        labels = [str(index) for index in range(size)]
    num_rounds = len(matrices["spatial"])
    figure, axes = plt.subplots(
        2,
        num_rounds,
        squeeze=False,
        figsize=(max(6.0, 4.2 * num_rounds), max(7.5, 1.35 * size)),
        constrained_layout=True,
    )
    for row, edge_type in enumerate(("spatial", "temporal")):
        if len(matrices[edge_type]) != num_rounds:
            raise ValueError("Spatial and temporal matrices have different round counts.")
        for round_index, values in enumerate(matrices[edge_type]):
            matrix = np.asarray(values, dtype=float)
            axis = axes[row, round_index]
            image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"{edge_type.capitalize()} — round {round_index}")
            axis.set_xlabel("Target agent")
            axis.set_ylabel("Source agent")
            axis.set_xticks(range(size), labels=labels, rotation=45, ha="right")
            axis.set_yticks(range(size), labels=labels)
            if size <= 10:
                for source in range(size):
                    for target in range(size):
                        value = matrix[source, target]
                        text = (
                            f"{int(value)}"
                            if "masks" in artifact
                            else f"{value:.2f}"
                        )
                        axis.text(
                            target,
                            source,
                            text,
                            ha="center",
                            va="center",
                            color="white" if value < 0.55 else "black",
                            fontsize=8,
                        )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"MMLU attribution edge pruning — {value_label} (source → target)",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_pruning_artifacts(
    graph: Any,
    output_dir: Path,
    training_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Save an exact tensor checkpoint, readable JSON matrices, and a PNG heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_pruning_artifact(graph, training_config=training_config)
    json_payload = _json_artifact(artifact)

    json_path = output_dir / "learned_edge_policy.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    checkpoint_path = output_dir / "learned_edge_policy.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "matrix_orientation": MATRIX_ORIENTATION,
            "workers": json_payload["workers"],
            "training_config": json_payload["training_config"],
            **artifact["_checkpoint_tensors"],
        },
        checkpoint_path,
    )

    heatmap_path = render_pruning_heatmaps(
        json_payload, output_dir / "learned_edge_policy.png"
    )
    return {
        "json": json_path,
        "checkpoint": checkpoint_path,
        "heatmap": heatmap_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a learned_edge_policy.json, aggregate_edge_matrices.json, "
            "or pruned_policy_*.json file."
        )
    )
    parser.add_argument("artifact_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metric", choices=("binary", "word_ratio"), default="binary")
    parser.add_argument("--statistic", default="conditional_mean")
    args = parser.parse_args()
    with args.artifact_json.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    output = args.output or args.artifact_json.with_suffix(".png")
    if "matrices" in artifact and "workers" in artifact:
        render_pruning_heatmaps(artifact, output)
    else:
        render_round_edge_heatmaps(
            artifact,
            output,
            metric=args.metric,
            statistic=args.statistic,
        )
    print(f"Saved pruning heatmap to {output}")


if __name__ == "__main__":
    main()
