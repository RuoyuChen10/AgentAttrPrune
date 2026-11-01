from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json

import pytest

from AgentAttrPrune.prompt import GSM8KPromptSet
from AgentAttrPrune.prompt.reviewer_ids import AnonymousReviewerIds
from experiments_attrprune import collect_correct, evaluate_mmlu
from experiments_attrprune.common import (
    build_run_config,
    config_fingerprint,
    validate_prompt_contract,
    write_json,
)


def test_gsm8k_reviewer_uses_opaque_labels_and_preserves_output_bodies():
    outputs = [
        {
            "agent_id": "math_solver",
            "reviewer_agent_id": "7vWW",
            "role": "Math Solver",
            "output": "As a Math Solver, I get 2.\nThe answer is 2",
        },
        {
            "agent_id": "inspector",
            "reviewer_agent_id": "5h3K",
            "role": "Inspector",
            "output": "One plus one is two.\nThe answer is 2",
        },
    ]
    original = deepcopy(outputs)
    messages = GSM8KPromptSet.reviewer_messages(task="1 + 1?", worker_outputs=outputs)

    assert messages[0]["content"] == (
        GSM8KPromptSet.get_decision_role()
        + "\n"
        + GSM8KPromptSet.get_decision_constraint()
    )
    assert messages[1]["content"] == (
        "The task is:\n1 + 1?\n\nThe agents' proposed solutions are:\n"
        "7vWW: As a Math Solver, I get 2.\nThe answer is 2\n\n"
        "5h3K: One plus one is two.\nThe answer is 2"
    )
    assert outputs == original
    assert "math_solver:" not in messages[1]["content"]
    assert "inspector:" not in messages[1]["content"]


@pytest.mark.parametrize(
    "aliases",
    [
        {},
        {"math_solver": "same", "mathematical_analyst": "same"},
        {"math_solver": "short", "mathematical_analyst": "abcd"},
        {"math_solver": "中文字符", "mathematical_analyst": "abcd"},
    ],
)
def test_gsm8k_config_rejects_invalid_saved_reviewer_ids(aliases):
    config = _config()
    config["prompt"]["reviewer_agent_ids"] = aliases
    config["fingerprint"] = config_fingerprint(config)
    with pytest.raises(ValueError, match="reviewer_agent_ids"):
        validate_prompt_contract(config)


def _config():
    return build_run_config(
        benchmark="gsm8k",
        model_name="/models/Qwen3-8B",
        num_workers=2,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
        calibration_split="train",
        evaluation_split="test",
    )


def _dataset(data_dir):
    data_dir.mkdir()
    for split in ("train", "test"):
        (data_dir / f"{split}.jsonl").write_text(
            "".join(
                json.dumps({"question": f"Example {index}: 1 + 1?", "answer": "1 + 1 = 2. #### 2"})
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )


def _collect_args(run_dir, data_dir, *, resume=False, limit=1):
    return argparse.Namespace(
        benchmark="gsm8k",
        model_name="/models/Qwen3-8B",
        run_dir=run_dir,
        data_dir=data_dir,
        split="train",
        evaluation_split="test",
        start_index=0,
        limit=limit,
        num_workers=2,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
        resume=resume,
        no_download=True,
    )


def _eval_args(run_dir, data_dir):
    return evaluate_mmlu.parse_args(
        [
            "--benchmark", "gsm8k",
            "--model-name", "/models/Qwen3-8B",
            "--run-dir", str(run_dir),
            "--data-dir", str(data_dir),
            "--mode", "dense",
            "--split", "test",
            "--limit", "2",
            "--save-traces",
            "--no-download",
        ]
    )


class StubChat:
    model_name = "/models/Qwen3-8B"

    async def agen(self, messages, **kwargs):
        return "One plus one is two.\nThe answer is 2"


def _assert_trace_labels(path, aliases):
    trace = json.loads(path.read_text(encoding="utf-8"))
    reviewer = trace["events"][-1]
    assert reviewer["agent_type"] == "FinalRefer"
    assert [item["agent_id"] for item in reviewer["incoming"]] == list(aliases)
    assert [item["reviewer_agent_id"] for item in reviewer["incoming"]] == list(aliases.values())
    prompt = reviewer["input_messages"][-1]["content"]
    for agent_id, alias in aliases.items():
        assert f"{alias}: One plus one is two.\nThe answer is 2" in prompt
        assert f"{agent_id}:" not in prompt
    # Worker-to-worker communication retains the original source metadata.
    worker_prompt = trace["events"][1]["input_messages"][-1]["content"]
    assert "Agent math_solver, role is Math Solver" in worker_prompt


def test_gsm8k_collect_resume_and_evaluate_share_saved_reviewer_ids(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    _dataset(data_dir)
    monkeypatch.setattr(collect_correct, "HFChat", lambda name: StubChat())
    monkeypatch.setattr(evaluate_mmlu, "HFChat", lambda name: StubChat())
    monkeypatch.setattr(collect_correct, "set_reproducible", lambda seed: None)
    monkeypatch.setattr(evaluate_mmlu, "set_reproducible", lambda seed: None)

    first = asyncio.run(collect_correct.collect(_collect_args(run_dir, data_dir)))
    assert first["attempted"] == 1
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["prompt"]["version"] == "gsm8k_v2_random_reviewer_ids"
    aliases = config["prompt"]["reviewer_agent_ids"]
    assert len(set(aliases.values())) == 2
    assert all(len(value) == 4 and value.isascii() and value.isalnum() for value in aliases.values())
    _assert_trace_labels(run_dir / "calibration/traces/sample_00000/trace.json", aliases)

    # A resume or evaluation must never generate another mapping internally.
    prepare = AnonymousReviewerIds.prepare_reviewer_agent_ids.__func__

    def require_saved(cls, agent_ids, reviewer_agent_ids=None):
        assert reviewer_agent_ids == aliases
        return prepare(cls, agent_ids, reviewer_agent_ids)

    monkeypatch.setattr(GSM8KPromptSet, "prepare_reviewer_agent_ids", classmethod(require_saved))
    resumed = asyncio.run(
        collect_correct.collect(_collect_args(run_dir, data_dir, resume=True, limit=2))
    )
    assert resumed["attempted"] == 2
    assert json.loads((run_dir / "run_config.json").read_text()) == config
    _assert_trace_labels(run_dir / "calibration/traces/sample_00001/trace.json", aliases)

    summary = asyncio.run(evaluate_mmlu.evaluate(_eval_args(run_dir, data_dir)))
    assert summary["total"] == summary["correct"] == 2
    for index in range(2):
        _assert_trace_labels(
            run_dir / f"evaluation/dense_test/traces/sample_{index:05d}/trace.json", aliases
        )


@pytest.mark.parametrize("stage", ["collect", "evaluate"])
def test_gsm8k_legacy_run_rejected_before_loading_model(monkeypatch, tmp_path, stage):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    _dataset(data_dir)
    config = _config()
    config.pop("prompt")
    config["fingerprint"] = config_fingerprint(config)
    write_json(run_dir / "run_config.json", config)

    def unexpected_model(name):
        raise AssertionError("Legacy prompt must be rejected before loading a model")

    monkeypatch.setattr(collect_correct, "HFChat", unexpected_model)
    monkeypatch.setattr(evaluate_mmlu, "HFChat", unexpected_model)
    monkeypatch.setattr(collect_correct, "set_reproducible", lambda seed: None)
    monkeypatch.setattr(evaluate_mmlu, "set_reproducible", lambda seed: None)
    with pytest.raises(ValueError, match="prompt contract"):
        if stage == "collect":
            asyncio.run(collect_correct.collect(_collect_args(run_dir, data_dir, resume=True)))
        else:
            asyncio.run(evaluate_mmlu.evaluate(_eval_args(run_dir, data_dir)))
