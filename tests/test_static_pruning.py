from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.common import build_run_config, write_json
from experiments_attrprune.static_pruning import (
    build_static_sweep,
    summarize_static_sweep,
)


def _metric_payload(values, candidates):
    return {
        "conditional_mean": values,
        "sample_mean": values,
        "sum": values,
        "opportunity_count": candidates,
        "candidate_count": candidates,
    }


def _make_run(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    config = build_run_config(
        model_name="/models/Qwen3-8B",
        num_workers=2,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
    )
    write_json(run_dir / "run_config.json", config)
    zero = [[0.0, 0.0], [0.0, 0.0]]
    spatial_values = [
        [[0.0, 0.8], [0.0, 0.0]],
        [[0.0, 0.2], [0.0, 0.0]],
    ]
    spatial_candidates = [
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ]
    temporal_values = [
        zero,
        [[0.4, 0.3], [0.1, 0.9]],
    ]
    temporal_candidates = [
        zero,
        [[1.0, 1.0], [1.0, 1.0]],
    ]
    aggregate = {
        "schema_version": 1,
        "run_fingerprint": config["fingerprint"],
        "num_correct_samples": 3,
        "agents": [item["agent_id"] for item in config["workers"]],
        "agent_roles": [item["role"] for item in config["workers"]],
        "agent_types": [item["agent_type"] for item in config["workers"]],
        "num_rounds": 2,
        "matrix_orientation": "[round][source][target]",
        "metrics": {
            "token_ratio": {
                "spatial": _metric_payload(spatial_values, spatial_candidates),
                "temporal": _metric_payload(temporal_values, temporal_candidates),
            }
        },
    }
    write_json(run_dir / "attribution" / "aggregate_edge_matrices.json", aggregate)
    return run_dir, config


def test_build_static_sweep_publishes_valid_rate_specific_policies(tmp_path):
    run_dir, config = _make_run(tmp_path)

    manifest = build_static_sweep(run_dir, percentages=(0, 50))

    assert manifest["run_fingerprint"] == config["fingerprint"]
    assert manifest["percentages"] == [0, 50]
    assert Path(manifest["manifest_json"]).is_file()
    dense_entry, half_entry = manifest["policies"]
    assert dense_entry["candidate_edges"] == dense_entry["retained_edges"] == 6
    assert half_entry["candidate_edges"] == 6
    # Pruning follows the established per-round/per-edge-type policy contract:
    # both one-edge spatial rounds lose one edge and temporal loses two of four.
    assert half_entry["pruned_edges"] == 4
    assert half_entry["effective_candidate_pruning_rate"] == pytest.approx(4 / 6)

    policy = RoundPolicy.from_json(half_entry["policy_json"])
    assert int(policy.spatial_masks.sum().item()) == 0
    assert int(policy.temporal_masks.sum().item()) == 2
    assert policy.metadata["metric"] == "token_ratio"
    assert policy.metadata["pruning_rate"] == 0.5
    assert policy.metadata["pruning_allocation"] == "per_round_edge_type"


def test_global_static_sweep_allocates_one_budget_across_all_matrices(tmp_path):
    run_dir, _ = _make_run(tmp_path)

    manifest = build_static_sweep(
        run_dir,
        percentages=(50,),
        pruning_scope="global",
    )

    entry = manifest["policies"][0]
    assert entry["candidate_edges"] == 6
    assert entry["pruned_edges"] == 3
    assert entry["effective_candidate_pruning_rate"] == pytest.approx(0.5)
    assert manifest["pruning_allocation"] == "global"
    assert Path(manifest["manifest_json"]).parent.name.endswith("_global")
    assert "_global_" in entry["evaluation_name"]

    policy = RoundPolicy.from_json(entry["policy_json"])
    # The three globally lowest values are temporal 0.1, spatial 0.2,
    # and temporal 0.3. The strong spatial 0.8 edge is not forced out merely
    # because it is the only edge in its round-specific matrix.
    assert int(policy.spatial_masks.sum().item()) == 1
    assert int(policy.temporal_masks.sum().item()) == 2
    assert policy.spatial_masks[0, 0, 1]
    assert not policy.spatial_masks[1, 0, 1]
    assert policy.metadata["pruning_scope"] == "observed_attributed_edges_global"
    assert policy.metadata["pruning_allocation"] == "global"


def test_static_sweep_rejects_duplicate_percentages(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        build_static_sweep(run_dir, percentages=(5, 5))


def test_summarize_static_sweep_validates_selection_and_writes_csv(tmp_path):
    run_dir, config = _make_run(tmp_path)
    manifest = build_static_sweep(run_dir, percentages=(0, 50))

    for index, entry in enumerate(manifest["policies"]):
        evaluation_dir = Path(entry["evaluation_directory"])
        retained = int(entry["retained_edges"])
        candidate = int(entry["candidate_edges"])
        write_json(
            evaluation_dir / "evaluation_summary.json",
            {
                "schema_version": 1,
                "run_fingerprint": config["fingerprint"],
                "mode": "token_ratio",
                "evaluation_name": entry["evaluation_name"],
                "benchmark": "mmlu",
                "split": "val",
                "start_index": 0,
                "stop_index_exclusive": 3,
                "dataset_size": 3,
                "total": 3,
                "correct": 2 - index,
                "accuracy": (2 - index) / 3,
                "prompt_tokens": 100 - index,
                "generated_tokens": 20,
                "elapsed_seconds": 1.5,
                "retained_edges": retained,
                "candidate_edges": candidate,
                "edge_retention": retained / candidate,
                "source_policy_json": entry["policy_json"],
            },
        )

    summary = summarize_static_sweep(manifest["manifest_json"])

    assert [row["requested_pruning_percent"] for row in summary["rows"]] == [0, 50]
    assert [row["correct"] for row in summary["rows"]] == [2, 1]
    assert Path(summary["summary_json"]).is_file()
    csv_path = Path(summary["summary_csv"])
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["prune_00pct", "prune_50pct"]

    serialized = json.loads(Path(summary["summary_json"]).read_text(encoding="utf-8"))
    assert serialized["selection"]["total"] == 3
