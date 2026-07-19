import json

import pytest

from AgentAttrPrune.attribution import (
    AttributionConfig,
    EdgeAttributionCalibrator,
    aggregate_sample_matrices,
    build_sample_matrices,
)
from AgentAttrPrune.pruning import RoundPolicy


def _trace(sample_index=1, schema_version=1):
    trace = {
        "config": {
            "sample_index": sample_index,
            "rounds": 1,
            "fingerprint": "test-run-fingerprint",
        },
        "topology": {
            "workers": [
                {"agent_id": "a", "role": "Critic", "agent_type": "AnalyzeAgent"},
                {"agent_id": "b", "role": "Solver", "agent_type": "AnalyzeAgent"},
            ]
        },
        "final": {"parsed_answer": "A", "gold_answer": "A"},
        "events": [
            {
                "round": 0,
                "agent_id": "b",
                "incoming": [
                    {
                        "agent_id": "a",
                        "round": 0,
                        "edge_type": "spatial",
                        "output": "evidence",
                    }
                ],
            }
        ],
    }
    if schema_version is not None:
        trace["schema_version"] = schema_version
    return trace


def _event_result(binary=1, word_ratio=0.5):
    return {
        "configuration": {"score_type": "logit", "score_reduction": "sum"},
        "edge_scores": [
            {
                "edge_type": "spatial",
                "source_agent": "a",
                "target_agent": "b",
                "target_round": 0,
                "binary": binary,
                "word_ratio": word_ratio,
            }
        ],
    }


def test_aggregate_only_public_api_writes_matrices_visual_and_three_policies(tmp_path):
    output_dir = tmp_path / "attribution"
    matrix_dir = output_dir / "sample_00001"
    matrix_dir.mkdir(parents=True)
    sample = build_sample_matrices(_trace(), [_event_result()])
    sample["selection_configuration"] = {"min_incoming": 2}
    matrix_path = matrix_dir / "sample_edge_matrices.json"
    matrix_path.write_text(json.dumps(sample), encoding="utf-8")

    calibrator = EdgeAttributionCalibrator(
        run_dir=tmp_path / "calibration" / "traces",
        output_dir=output_dir,
        config=AttributionConfig(policy_pruning_rate=0.0),
    )
    result = calibrator.run(aggregate_only=True, aggregate_all_existing=True)

    assert result["manifest"] is None
    assert result["aggregate"]["num_correct_samples"] == 1
    assert isinstance(result["policies"]["binary"], RoundPolicy)
    assert result["policies"]["binary"].metadata["run_fingerprint"] == (
        "test-run-fingerprint"
    )
    assert (output_dir / "aggregate_edge_matrices.json").is_file()
    assert (output_dir / "aggregate_token_ratio_matrices.png").is_file()
    assert (output_dir / "pruned_policy_binary.json").is_file()
    assert (output_dir / "pruned_policy_token_ratio.json").is_file()
    assert (output_dir / "pruned_policy_word_ratio.json").is_file()


def test_aggregate_rejects_legacy_and_v1_trace_matrices_mixed_together():
    current = build_sample_matrices(_trace(schema_version=1), [_event_result()])
    legacy = build_sample_matrices(_trace(sample_index=2, schema_version=None), [_event_result()])

    with pytest.raises(ValueError, match="cannot be mixed"):
        aggregate_sample_matrices([current, legacy])


def test_calibrator_never_publishes_a_legacy_only_aggregate(tmp_path):
    output_dir = tmp_path / "attribution"
    matrix_dir = output_dir / "sample_00002"
    matrix_dir.mkdir(parents=True)
    legacy = build_sample_matrices(
        _trace(sample_index=2, schema_version=None), [_event_result()]
    )
    matrix_path = matrix_dir / "sample_edge_matrices.json"
    matrix_path.write_text(json.dumps(legacy), encoding="utf-8")
    calibrator = EdgeAttributionCalibrator(
        run_dir=tmp_path / "traces",
        output_dir=output_dir,
        config=AttributionConfig(allow_legacy_traces=True),
    )

    with pytest.raises(ValueError, match="debug-only"):
        calibrator.run(aggregate_only=True, aggregate_all_existing=True)


def test_manifest_failures_block_partial_aggregate_and_policy(tmp_path):
    output_dir = tmp_path / "attribution"
    matrix_dir = output_dir / "sample_00001"
    matrix_dir.mkdir(parents=True)
    sample = build_sample_matrices(_trace(), [_event_result()])
    matrix_path = matrix_dir / "sample_edge_matrices.json"
    matrix_path.write_text(json.dumps(sample), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [{"sample_matrix": str(matrix_path)}],
                "failures": [{"sample_id": "sample_00002", "error": "boom"}],
            }
        ),
        encoding="utf-8",
    )
    calibrator = EdgeAttributionCalibrator(
        run_dir=tmp_path / "traces",
        output_dir=output_dir,
    )

    with pytest.raises(ValueError, match="partial formal aggregate"):
        calibrator.run(aggregate_only=True)
    assert not (output_dir / "aggregate_edge_matrices.json").exists()
    assert not (output_dir / "pruned_policy_binary.json").exists()


def test_schema_v1_formal_attribution_requires_exact_token_ids(tmp_path):
    trace = _trace()
    calibrator = EdgeAttributionCalibrator(
        run_dir=tmp_path,
        output_dir=tmp_path / "attribution",
        model_name="stub",
        config=AttributionConfig(min_incoming=1),
    )

    with pytest.raises(ValueError, match="exact input_ids"):
        calibrator._validate_trace(trace, tmp_path / "trace.json")


def test_schema_v1_formal_attribution_rejects_truncated_node_output(tmp_path):
    trace = _trace()
    trace["events"][0].update(
        {
            "event_id": 0,
            "input_ids": [1, 2],
            "generated_ids": [3, 4],
            "finish_reason": "length",
            "hit_max_new_tokens": True,
            "llm_call": {"max_tokens": 2},
        }
    )
    trace["truncated_event_ids"] = [0]
    calibrator = EdgeAttributionCalibrator(
        run_dir=tmp_path,
        output_dir=tmp_path / "attribution",
        model_name="stub",
        config=AttributionConfig(min_incoming=1),
    )

    with pytest.raises(ValueError, match="hit max_new_tokens"):
        calibrator._validate_trace(trace, tmp_path / "trace.json")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"target_recovery": 0}, "target_recovery"),
        ({"sample_limit": 0}, "sample_limit"),
        ({"score_reduction": "none"}, "score_reduction"),
        ({"policy_pruning_rate": 1.1}, "policy_pruning_rate"),
        ({"saliency_floor": 1.0}, "saliency_floor"),
    ],
)
def test_attribution_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AttributionConfig(**kwargs)
