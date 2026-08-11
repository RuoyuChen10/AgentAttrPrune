from __future__ import annotations

from copy import deepcopy
import json

import pytest

from experiments_attrprune.common import (
    build_run_config,
    completed_attempt_indices,
    dataset_identities_match,
    ensure_run_config,
    mmlu_data_identity,
    normalize_answer,
    resolve_run_dir,
)


def test_run_config_fingerprint_is_stable_and_rejects_mixing(tmp_path):
    config = build_run_config(
        model_name="/models/Qwen3-8B",
        num_workers=5,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=256,
        seed=0,
    )
    path = tmp_path / "run_config.json"
    assert ensure_run_config(path, config)["fingerprint"] == config["fingerprint"]
    assert ensure_run_config(path, dict(config))["fingerprint"] == config["fingerprint"]

    changed = build_run_config(
        model_name="/models/Qwen3-8B",
        num_workers=5,
        num_rounds=3,
        temperature=0.0,
        max_new_tokens=256,
        seed=0,
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        ensure_run_config(path, changed)

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["num_rounds"] = 99
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fingerprint"):
        ensure_run_config(path, config)


def test_resume_manifest_skips_correct_and_incorrect_completed_samples(tmp_path):
    path = tmp_path / "attempts.jsonl"
    records = [
        {"sample_index": 1, "status": "completed", "correct": True},
        {"sample_index": 2, "status": "completed", "correct": False},
        {"sample_index": 3, "status": "error"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    assert completed_attempt_indices(path) == {1, 2}


def test_default_output_is_under_agentattrprune_results(tmp_path):
    path = resolve_run_dir(model_name="/models/Qwen3-8B", seed=7, results_root=tmp_path)
    assert path == tmp_path / "mmlu" / "qwen3_8b" / "seed_7"


def test_gsm8k_run_config_uses_benchmark_roles_splits_and_output_root(tmp_path):
    config = build_run_config(
        benchmark="gsm8k",
        model_name="/models/Qwen3-8B",
        num_workers=4,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=256,
        seed=3,
        calibration_split="train",
        evaluation_split="test",
    )
    path = resolve_run_dir(
        benchmark="gsm8k",
        model_name="/models/Qwen3-8B",
        seed=3,
        results_root=tmp_path,
    )

    assert config["benchmark"] == "gsm8k"
    assert config["splits"] == {"calibration": "train", "evaluation": "test"}
    assert [item["role"] for item in config["workers"]] == [
        "Math Solver",
        "Mathematical Analyst",
        "Programming Expert",
        "Inspector",
    ]
    assert path == tmp_path / "gsm8k" / "qwen3_8b" / "seed_3"


def test_mmlu_data_identity_locks_path_content_and_shuffle(tmp_path):
    dev = tmp_path / "data" / "dev" / "topic_dev.csv"
    val = tmp_path / "data" / "val" / "topic_val.csv"
    dev.parent.mkdir(parents=True)
    val.parent.mkdir(parents=True)
    dev.write_text("q,a,b,c,d,A\n", encoding="utf-8")
    val.write_text("q2,a,b,c,d,B\n", encoding="utf-8")

    first = mmlu_data_identity(tmp_path)
    assert first["shuffle_seed"] == 888
    assert first["splits"]["dev"]["record_count"] == 1
    assert first["splits"]["val"]["record_count"] == 1

    val.write_text("q2,a,b,c,d,C\n", encoding="utf-8")
    second = mmlu_data_identity(tmp_path)
    assert first["splits"]["val"]["sha256"] != second["splits"]["val"][
        "sha256"
    ]


def test_dataset_identity_accepts_only_additive_schema_fields():
    current = {
        "benchmark": "mmlu",
        "requested_data_dir": "/data/mmlu",
        "data_root": "/data/mmlu",
        "shuffle_seed": 888,
        "splits": {
            "dev": {
                "file_count": 57,
                "record_count": 285,
                "source_record_count": 285,
                "sha256": "dev-hash",
            },
            "val": {
                "file_count": 57,
                "record_count": 1531,
                "source_record_count": 1531,
                "sha256": "val-hash",
            },
        },
    }
    legacy = deepcopy(current)
    legacy.pop("benchmark")
    for payload in legacy["splits"].values():
        payload.pop("source_record_count")

    assert dataset_identities_match(legacy, current)

    changed_hash = deepcopy(current)
    changed_hash["splits"]["val"]["sha256"] = "changed"
    assert not dataset_identities_match(legacy, changed_hash)

    changed_path = deepcopy(current)
    changed_path["data_root"] = "/different/mmlu"
    assert not dataset_identities_match(legacy, changed_path)

    missing_stored_field = deepcopy(current)
    missing_stored_field["splits"]["val"].pop("sha256")
    assert not dataset_identities_match(current, missing_stored_field)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Decision: B", "B"), ("A\nreason", "A"), ("Option C", "C"), ("", "")],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected
