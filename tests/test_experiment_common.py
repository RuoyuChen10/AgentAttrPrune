from __future__ import annotations

import json

import pytest

from experiments_attrprune.common import (
    build_run_config,
    completed_attempt_indices,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Decision: B", "B"), ("A\nreason", "A"), ("Option C", "C"), ("", "")],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected
