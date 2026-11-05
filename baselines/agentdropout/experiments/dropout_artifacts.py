"""Serialize and visualize round-specific AgentDropout policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = 1
MATRIX_ORIENTATION = "[round][source_agent][target_agent]"


def _tensor_list(values, size: int) -> list[torch.Tensor]:
    if isinstance(values, torch.nn.ParameterList):
        tensors = list(values)
    elif isinstance(values, (list, tuple)):
        tensors = list(values)
    else:
        tensors = [values]
    return [tensor.detach().cpu().reshape(size, size) for tensor in tensors]


def _matrix_payload(logits, masks, size: int) -> dict[str, Any]:
    logit_matrices = _tensor_list(logits, size)
    mask_matrices = _tensor_list(masks, size)
    if len(logit_matrices) != len(mask_matrices):
        raise ValueError("logit and mask round counts differ")
    probabilities = [torch.sigmoid(matrix) for matrix in logit_matrices]
    effective = [probability * mask for probability, mask in zip(probabilities, mask_matrices)]
    return {
        "logits": [matrix.tolist() for matrix in logit_matrices],
        "probabilities": [matrix.tolist() for matrix in probabilities],
        "masks": [matrix.to(dtype=torch.int64).tolist() for matrix in mask_matrices],
        "effective_probabilities": [matrix.tolist() for matrix in effective],
        "retained_edges": [int(matrix.sum()) for matrix in mask_matrices],
    }


def build_artifact(graph, training_config: Mapping[str, Any]) -> tuple[dict, dict]:
    nodes = list(graph.nodes.values())
    size = len(nodes)
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
    matrices = {
        edge_type: _matrix_payload(
            getattr(graph, f"{edge_type}_logits"),
            getattr(graph, f"{edge_type}_masks"),
            size,
        )
        for edge_type in ("spatial", "temporal")
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "matrix_orientation": MATRIX_ORIENTATION,
        "workers": workers,
        "num_rounds": graph.rounds,
        "node_dropout": {
            "enabled": bool(graph.skip_nodes),
            "dropped_worker_index_by_round": [int(index) for index in graph.skip_nodes],
            "dropped_role_by_round": [workers[int(index)]["role"] for index in graph.skip_nodes],
        },
        "matrices": matrices,
        "training_config": dict(training_config),
        "description": {
            "probabilities": "sigmoid of learned edge logits",
            "masks": "binary policy after node dropout and iterative edge pruning",
            "effective_probabilities": "mask multiplied by learned probability",
            "spatial_rounds": "one source-to-target matrix per discussion round",
            "temporal_rounds": "transition r -> r+1; two rounds therefore have one matrix",
        },
    }
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "matrix_orientation": MATRIX_ORIENTATION,
        "workers": workers,
        "num_rounds": graph.rounds,
        "diff": bool(graph.diff),
        "skip_nodes": [int(index) for index in graph.skip_nodes],
        "training_config": dict(training_config),
    }
    for edge_type in ("spatial", "temporal"):
        checkpoint[f"{edge_type}_logits"] = _tensor_list(
            getattr(graph, f"{edge_type}_logits"), size
        )
        checkpoint[f"{edge_type}_masks"] = _tensor_list(
            getattr(graph, f"{edge_type}_masks"), size
        )
    for name in ("spatial_logits_1", "temporal_logits_1"):
        if hasattr(graph, name):
            checkpoint[name] = _tensor_list(getattr(graph, name), size)
    return payload, checkpoint


def render_heatmap(artifact: Mapping[str, Any], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    spatial_count = len(artifact["matrices"]["spatial"]["masks"])
    temporal_count = len(artifact["matrices"]["temporal"]["masks"])
    for round_index in range(spatial_count):
        panels.append(("spatial", round_index, f"Spatial round {round_index}"))
    for transition in range(temporal_count):
        panels.append(("temporal", transition, f"Temporal {transition}→{transition + 1}"))
    metrics = (
        ("probabilities", "Learned probability"),
        ("masks", "Retained mask"),
        ("effective_probabilities", "Effective probability"),
    )
    labels = [f"{worker['index']}:{worker['role']}" for worker in artifact["workers"]]
    size = len(labels)
    figure, axes = plt.subplots(
        len(panels),
        len(metrics),
        squeeze=False,
        figsize=(max(14, 2.4 * size), max(4.0 * len(panels), 8)),
        constrained_layout=True,
    )
    for row, (edge_type, round_index, panel_title) in enumerate(panels):
        for column, (metric, metric_title) in enumerate(metrics):
            matrix = np.asarray(
                artifact["matrices"][edge_type][metric][round_index], dtype=float
            )
            axis = axes[row, column]
            image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"{panel_title} — {metric_title}")
            axis.set_xlabel("Target worker")
            axis.set_ylabel("Source worker")
            axis.set_xticks(range(size), labels=labels, rotation=40, ha="right")
            axis.set_yticks(range(size), labels=labels)
            for source in range(size):
                for target in range(size):
                    value = matrix[source, target]
                    text = f"{int(value)}" if metric == "masks" else f"{value:.2f}"
                    axis.text(
                        target,
                        source,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < 0.55 else "black",
                    )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    dropped = artifact["node_dropout"]["dropped_role_by_round"]
    figure.suptitle(
        f"AgentDropout {artifact['training_config'].get('benchmark', 'mmlu').upper()} policy (source → target)\n"
        f"Dropped worker by round: {dropped}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_artifacts(graph, output_dir: Path, training_config: Mapping[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact, checkpoint = build_artifact(graph, training_config)
    json_path = output_dir / "learned_dropout_policy.json"
    json_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_path = output_dir / "learned_dropout_policy.pt"
    torch.save(checkpoint, checkpoint_path)
    heatmap_path = render_heatmap(artifact, output_dir / "learned_dropout_policy.png")
    return {"json": json_path, "checkpoint": checkpoint_path, "heatmap": heatmap_path}


def restore_checkpoint(graph, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("matrix_orientation") != MATRIX_ORIENTATION:
        raise ValueError("checkpoint matrix orientation is incompatible")
    if int(checkpoint["num_rounds"]) != graph.rounds:
        raise ValueError("checkpoint and graph round counts differ")
    size = len(graph.nodes)
    if len(checkpoint.get("workers", [])) != size:
        raise ValueError("checkpoint and graph worker counts differ")
    for edge_type in ("spatial", "temporal"):
        for suffix in ("logits", "masks"):
            name = f"{edge_type}_{suffix}"
            targets = getattr(graph, name)
            targets = list(targets) if isinstance(targets, torch.nn.ParameterList) else [targets]
            sources = checkpoint[name]
            if len(targets) != len(sources):
                raise ValueError(f"checkpoint {name} round count differs")
            for target, source in zip(targets, sources):
                with torch.no_grad():
                    target.copy_(torch.as_tensor(source).reshape_as(target))
    graph.skip_nodes = [int(index) for index in checkpoint.get("skip_nodes", [])]
    return checkpoint
