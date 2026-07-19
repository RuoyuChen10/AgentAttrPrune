from __future__ import annotations

from copy import deepcopy
import json

import pytest

from experiments_attrprune.common import build_run_config, write_json
from experiments_attrprune.compare import build_comparison


def _evaluation_summary(config: dict, mode: str) -> dict:
    retained = 55 if mode == "dense" else 40
    return {
        "schema_version": 1,
        "run_fingerprint": config["fingerprint"],
        "mode": mode,
        "split": "val",
        "start_index": 0,
        "stop_index_exclusive": 3,
        "dataset_size": 3,
        "total": 3,
        "correct": 2,
        "accuracy": 2 / 3,
        "prompt_tokens": 100,
        "generated_tokens": 20,
        "elapsed_seconds": 1.5,
        "retained_edges": retained,
        "candidate_edges": 55,
        "edge_retention": retained / 55,
    }


def _comparison_run(tmp_path):
    config = build_run_config(
        model_name="/models/Qwen3-8B",
        num_workers=5,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=256,
        seed=0,
    )
    write_json(tmp_path / "run_config.json", config)
    modes = {
        "dense_val": "dense",
        "binary_val": "binary",
        "token_ratio_val": "token_ratio",
    }
    for directory, mode in modes.items():
        write_json(
            tmp_path / "evaluation" / directory / "evaluation_summary.json",
            _evaluation_summary(config, mode),
        )
    return config


def test_compare_accepts_one_fingerprint_and_identical_selection(tmp_path):
    config = _comparison_run(tmp_path)
    comparison = build_comparison(tmp_path)

    assert comparison["run_fingerprint"] == config["fingerprint"]
    assert [row["evaluation"] for row in comparison["rows"]] == [
        "dense_val",
        "binary_val",
        "token_ratio_val",
    ]
    assert comparison["evaluations"]["binary_val"]["delta_vs_dense"][
        "accuracy"
    ] == 0.0


def test_compare_rejects_different_evaluation_selection(tmp_path):
    _comparison_run(tmp_path)
    path = tmp_path / "evaluation" / "binary_val" / "evaluation_summary.json"
    summary = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    summary["stop_index_exclusive"] = 2
    summary["total"] = 2
    write_json(path, summary)

    with pytest.raises(ValueError, match="selection differs"):
        build_comparison(tmp_path)


def test_compare_rejects_different_candidate_topology(tmp_path):
    _comparison_run(tmp_path)
    path = tmp_path / "evaluation" / "token_ratio_val" / "evaluation_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["candidate_edges"] = 54
    write_json(path, summary)

    with pytest.raises(ValueError, match="candidate topology differs"):
        build_comparison(tmp_path)
