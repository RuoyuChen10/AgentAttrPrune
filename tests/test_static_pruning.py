from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.common import build_run_config, config_fingerprint, write_json
from experiments_attrprune.evaluate_mmlu import _json_fingerprint
from experiments_attrprune.static_pruning import (
    build_static_sweep,
    summarize_static_sweep,
    transfer_static_sweep,
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


def test_transfer_static_sweep_reuses_masks_without_overwriting_source(tmp_path):
    source_run, source_config = _make_run(tmp_path)
    # Reproduce a pre-random-reviewer run: neither the run nor its aggregate
    # declares the current prompt contract.
    source_config.pop("prompt")
    source_config["fingerprint"] = config_fingerprint(source_config)
    write_json(source_run / "run_config.json", source_config)
    aggregate_path = source_run / "attribution" / "aggregate_edge_matrices.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["run_fingerprint"] = source_config["fingerprint"]
    write_json(aggregate_path, aggregate)
    source_manifest = build_static_sweep(
        source_run,
        percentages=(50,),
        pruning_scope="global",
    )
    source_policy_path = Path(source_manifest["policies"][0]["policy_json"])
    source_policy_text = source_policy_path.read_text(encoding="utf-8")
    target_run = tmp_path / "transfer"

    manifest = transfer_static_sweep(
        source_run,
        target_run,
        percentages=(50,),
        pruning_scope="global",
    )

    assert source_policy_path.read_text(encoding="utf-8") == source_policy_text
    target_config = json.loads((target_run / "run_config.json").read_text())
    assert target_config["fingerprint"] != source_config["fingerprint"]
    assert target_config["prompt"]["version"] == "mmlu_v2_random_reviewer_ids"
    assert len(set(target_config["prompt"]["reviewer_agent_ids"].values())) == 2
    target_policy = RoundPolicy.from_json(manifest["policies"][0]["policy_json"])
    source_policy = RoundPolicy.from_json(source_policy_path)
    assert target_policy.spatial_masks.equal(source_policy.spatial_masks)
    assert target_policy.temporal_masks.equal(source_policy.temporal_masks)
    assert target_policy.metadata["run_fingerprint"] == target_config["fingerprint"]
    assert (
        target_policy.metadata["policy_transfer"]["source_run_fingerprint"]
        == source_config["fingerprint"]
    )
    assert manifest["policy_transfer"]["calibration_reused"] is True

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transfer_static_sweep(
            source_run,
            target_run,
            percentages=(50,),
            pruning_scope="global",
        )


def test_static_sweep_rejects_duplicate_percentages(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        build_static_sweep(run_dir, percentages=(5, 5))


@pytest.mark.parametrize("calibration_manifest", [
    {"complete": False, "failures": []},
    {"complete": True, "failures": [{"error": "out of memory"}]},
    {"failures": [{"error": "truncated generation"}]},
    {"complete": True, "aggregation_complete": False, "failures": []},
])
def test_static_sweep_rejects_stale_aggregate_after_incomplete_calibration(
    tmp_path, calibration_manifest,
):
    run_dir, _ = _make_run(tmp_path)
    write_json(run_dir / "attribution" / "manifest.json", calibration_manifest)
    with pytest.raises(ValueError, match="incomplete or has failures"):
        build_static_sweep(run_dir, percentages=(0,), pruning_scope="global")
    assert not (run_dir / "attribution" / "static_pruning").exists()


def test_static_sweep_accepts_legacy_completed_calibration_manifest(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    write_json(run_dir / "attribution" / "manifest.json", {"failures": []})
    manifest = build_static_sweep(run_dir, percentages=(0,))
    assert manifest["calibration_aggregation_scope"] == "correct"


def test_static_sweep_checks_completed_publication_hash(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    aggregate_path = run_dir / "attribution" / "aggregate_edge_matrices.json"
    write_json(run_dir / "attribution" / "manifest.json", {
        "complete": True,
        "aggregation_complete": True,
        "failures": [],
        "aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
    })
    build_static_sweep(run_dir, percentages=(0,))
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["num_correct_samples"] += 1
    write_json(aggregate_path, aggregate)
    with pytest.raises(ValueError, match="completed calibration publication"):
        build_static_sweep(run_dir, percentages=(50,))


def test_all_incorrect_calibration_prepares_full_45_edge_dense_control(tmp_path):
    run_dir, _ = _make_run(tmp_path)
    config = build_run_config(
        model_name="/models/Qwen3-8B", num_workers=5, num_rounds=2,
        temperature=0.0, max_new_tokens=2048, seed=0,
    )
    write_json(run_dir / "run_config.json", config)
    aggregate_path = run_dir / "attribution" / "aggregate_edge_matrices.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate.update({
        "run_fingerprint": config["fingerprint"],
        "agents": [item["agent_id"] for item in config["workers"]],
        "agent_roles": [item["role"] for item in config["workers"]],
        "agent_types": [item["agent_type"] for item in config["workers"]],
        "num_correct_samples": 0,
        "num_aggregated_samples": 40,
        "aggregation_scope": "all",
        "attribution_configuration": {"attribution_mode": "correctness-aware"},
    })
    zero = np.zeros((2, 5, 5)).tolist()
    spatial = np.stack([np.triu(np.ones((5, 5)), k=1)] * 2).tolist()
    temporal = np.stack([np.zeros((5, 5)), np.ones((5, 5))]).tolist()
    aggregate["metrics"]["token_ratio"] = {
        "spatial": _metric_payload(zero, spatial),
        "temporal": _metric_payload(zero, temporal),
    }
    write_json(aggregate_path, aggregate)

    manifest = build_static_sweep(run_dir, percentages=(0, 50), pruning_scope="global")

    assert manifest["calibration_num_aggregated_samples"] == 40
    assert manifest["calibration_num_correct_samples"] == 0
    assert manifest["calibration_aggregation_scope"] == "all"
    assert manifest["calibration_attribution_mode"] == "correctness-aware"
    dense = manifest["policies"][0]
    assert dense["candidate_edges"] == dense["retained_edges"] == 45
    policy = RoundPolicy.from_json(dense["policy_json"])
    assert policy.spatial_masks.sum().item() == 20
    assert policy.temporal_masks.sum().item() == 25
    assert policy.metadata["calibration_num_aggregated_samples"] == 40
    assert policy.metadata["calibration_num_correct_samples"] == 0


def _write_sweep_evaluations(manifest, config, batch_size=None):
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
                **({"eval_batch_size": batch_size} if batch_size is not None else {}),
            },
        )


def test_sweep_summary_keeps_oom_fallback_counts_and_unknown_legacy_coverage(tmp_path):
    run_dir, config = _make_run(tmp_path)
    manifest = build_static_sweep(run_dir, percentages=(0, 50))
    _write_sweep_evaluations(manifest, config, batch_size=4)
    path = Path(manifest["policies"][1]["evaluation_directory"]) / "evaluation_summary.json"
    evaluation = json.loads(path.read_text())
    evaluation["generation_batching"] = {
        "recorded_samples": 2,
        "samples_without_batch_metadata": 1,
        "oom_fallback_samples": 1,
        "oom_fallback_calls": 3,
    }
    write_json(path, evaluation)

    summary = summarize_static_sweep(manifest["manifest_json"])
    legacy, recovered = summary["rows"]
    assert legacy["batch_metadata_samples"] == 0
    assert legacy["samples_without_batch_metadata"] == 3
    assert recovered["batch_metadata_samples"] == 2
    assert recovered["samples_without_batch_metadata"] == 1
    assert recovered["oom_fallback_samples"] == 1
    assert recovered["oom_fallback_calls"] == 3
    with Path(summary["summary_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["oom_fallback_calls"] == "3"
    assert rows[0]["samples_without_batch_metadata"] == "3"


@pytest.mark.parametrize("batch_size", [None, 4])
def test_summarize_static_sweep_validates_selection_and_writes_csv(tmp_path, batch_size):
    run_dir, config = _make_run(tmp_path)
    manifest = build_static_sweep(run_dir, percentages=(0, 50))
    _write_sweep_evaluations(manifest, config, batch_size)
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
    expected_batch_size = batch_size or 1
    assert serialized["eval_batch_size"] == expected_batch_size
    assert {row["eval_batch_size"] for row in summary["rows"]} == {expected_batch_size}
    assert serialized["calibration_num_aggregated_samples"] == 3
    assert serialized["calibration_aggregation_scope"] == "correct"

    mismatched_path = Path(manifest["policies"][1]["evaluation_directory"]) / "evaluation_summary.json"
    mismatched = json.loads(mismatched_path.read_text())
    mismatched["eval_batch_size"] = expected_batch_size + 1
    write_json(mismatched_path, mismatched)
    with pytest.raises(ValueError, match="identical eval_batch_size"):
        summarize_static_sweep(manifest["manifest_json"])


@pytest.mark.parametrize("state_location", ["adjacent", "declared"])
def test_sweep_summary_rejects_same_count_policy_replacement(tmp_path, state_location):
    run_dir, config = _make_run(tmp_path)
    manifest = build_static_sweep(run_dir, percentages=(50,), pruning_scope="global")
    _write_sweep_evaluations(manifest, config, batch_size=4)
    entry = manifest["policies"][0]
    evaluation_dir = Path(entry["evaluation_directory"])
    state_path = (
        evaluation_dir / "evaluation_state.json"
        if state_location == "adjacent" else tmp_path / "saved_evaluation_state.json"
    )
    original_policy = RoundPolicy.from_json(entry["policy_json"])
    write_json(state_path, {"policy_fingerprint": _json_fingerprint(original_policy.to_dict())})
    if state_location == "declared":
        summary_path = evaluation_dir / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["evaluation_state_json"] = str(state_path)
        write_json(summary_path, summary)
    summarize_static_sweep(manifest["manifest_json"])

    policy_path = Path(entry["policy_json"])
    replacement = json.loads(policy_path.read_text())
    # Keep the same retained-edge count and policy path, but change a temporal edge.
    assert replacement["masks"]["temporal"][1][0] == [1, 0]
    replacement["masks"]["temporal"][1][0] = [0, 1]
    changed_policy = RoundPolicy.from_dict(replacement)
    assert changed_policy.temporal_masks.sum() == original_policy.temporal_masks.sum()
    write_json(policy_path, replacement)
    with pytest.raises(ValueError, match="evaluation policy fingerprint"):
        summarize_static_sweep(manifest["manifest_json"])


def test_correctness_aware_sweep_summary_requires_evaluation_state(tmp_path):
    run_dir, config = _make_run(tmp_path)
    aggregate_path = run_dir / "attribution" / "aggregate_edge_matrices.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["attribution_configuration"] = {"attribution_mode": "correctness-aware"}
    write_json(aggregate_path, aggregate)
    manifest = build_static_sweep(run_dir, percentages=(0,))
    _write_sweep_evaluations(manifest, config, batch_size=4)
    with pytest.raises(ValueError, match="Missing evaluation_state.json"):
        summarize_static_sweep(manifest["manifest_json"])

    entry = manifest["policies"][0]
    write_json(Path(entry["evaluation_directory"]) / "evaluation_state.json", {
        "policy_fingerprint": _json_fingerprint(RoundPolicy.from_json(entry["policy_json"]).to_dict()),
    })
    assert summarize_static_sweep(manifest["manifest_json"])["eval_batch_size"] == 4
