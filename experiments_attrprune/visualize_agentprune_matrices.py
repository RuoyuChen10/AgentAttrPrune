"""Draw AgentPrune's shared learned policy using the round-matrix plot style.

The panels show mask-retained sampling probabilities, not sampled graphs or
attribution scores. Spatial policies repeat every round; temporal starts at 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments_attrprune.visualize_pruning_matrices import _plot_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    source_path = args.policy_json.resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source["matrix_orientation"] != "[source_agent][target_agent]":
        raise ValueError("Expected AgentPrune source-to-target matrices")
    config = source["training_config"]
    if config["mode"] != "FullConnected":
        raise ValueError("Candidate masks must be specified for non-FullConnected runs")
    workers = source["workers"]
    if [w["index"] for w in workers] != list(range(len(workers))):
        raise ValueError("Worker indices must match the serialized matrix order")
    agents = [w["role"].lower().replace(" ", "_") for w in workers]
    count = len(agents)
    rounds = int(config["num_rounds"])
    if rounds < 1:
        raise ValueError("At least one round is required")
    aggregate = {
        "agents": agents,
        "num_rounds": rounds,
        "matrix_orientation": "[round][source][target]",
        "metrics": {"learned_probability": {}},
    }
    policy = {
        "agents": agents,
        "num_rounds": rounds,
        "matrix_orientation": "[round][source][target]",
        "metric": "learned_probability",
        "statistic": "value",
        "pruning_rate": config["pruning_rate"],
        "masks": {},
    }
    unique_counts = {}
    for edge_type in ("spatial", "temporal"):
        saved = source["matrices"][edge_type]
        probabilities = np.asarray(saved["probabilities"], dtype=float)
        mask = np.asarray(saved["masks"])
        if probabilities.shape != (count, count) or mask.shape != probabilities.shape:
            raise ValueError(f"Invalid {edge_type} matrix shape")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("Probabilities must be finite and in [0, 1]")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Masks must be binary")
        if not np.allclose(saved["effective_probabilities"], probabilities * mask):
            raise ValueError("Saved effective probabilities disagree with mask")
        candidates = np.ones((count, count), dtype=int)
        if edge_type == "spatial":
            np.fill_diagonal(candidates, 0)
        if np.any(mask > candidates):
            raise ValueError("Mask retains a non-candidate edge")
        unique_counts[edge_type] = {
            "retained": int(mask.sum()), "candidate": int(candidates.sum())
        }
        values, candidate_rounds, mask_rounds = [], [], []
        for round_index in range(rounds):
            active = edge_type == "spatial" or round_index > 0
            values.append(probabilities.tolist())
            candidate_rounds.append((candidates if active else candidates * 0).tolist())
            mask_rounds.append((mask if active else mask * 0).tolist())
        aggregate["metrics"]["learned_probability"][edge_type] = {
            "value": values, "candidate_count": candidate_rounds
        }
        policy["masks"][edge_type] = mask_rounds

    retained = sum(np.asarray(m).sum() for m in policy["masks"].values())
    candidates = sum(
        np.asarray(m["candidate_count"]).sum()
        for m in aggregate["metrics"]["learned_probability"].values()
    )
    percentage = float(config["pruning_rate"]) * 100
    title = (
        f"AgentPrune · {config['domain'].upper()} train{config['training_questions']}\n"
        f"{percentage:g}% mask pruning · retained {int(retained)}/{int(candidates)}"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"agentprune_prune_{percentage:g}pct_probability_matrices_viridis"
    output_paths = [output_dir / f"{stem}.{extension}" for extension in ("png", "svg")]
    _plot_policy(
        aggregate, policy, output_paths, dpi=args.dpi, color_max=0.7,
        title_override=title, colorbar_label="Learned edge probability",
    )
    (output_dir / "learned_edge_policy.source.json").write_bytes(source_bytes)
    (output_dir / "plot_inputs.json").write_text(
        json.dumps({"aggregate": aggregate, "policy": policy}, indent=2) + "\n"
    )
    manifest = {
        "source_json": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_renderer": str(Path(__file__).with_name("visualize_pruning_matrices.py").resolve()),
        "shared_renderer_sha256": hashlib.sha256(
            Path(__file__).with_name("visualize_pruning_matrices.py").read_bytes()
        ).hexdigest(),
        "matrix_orientation": "[round][source][target]",
        "workers": workers,
        "unique_policy_edge_counts": unique_counts,
        "round_edge_counts": {"retained": int(retained), "candidate": int(candidates)},
        "color_scale": {"cmap": "viridis", "vmin": 0, "vmax": 0.7},
        "semantics": [
            "Colors/numbers show sigmoid learned logits; gray crossed cells are mask-pruned.",
            "Spatial candidate edges include all off-diagonal directed pairs.",
            "One spatial policy is reused across rounds, without retraining or resampling for the plot.",
            "Round 0 has no temporal edges; later rounds reuse the temporal policy.",
            "Retained masks permit Bernoulli sampling; spatial edges also undergo DAG cycle rejection.",
            "The displayed counts include repeated round-edge slots; unique policy counts are separate.",
            "Final reviewer edges are outside these worker matrices.",
        ],
        "outputs": [str(p) for p in output_paths],
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
