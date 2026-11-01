from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pytest

from experiments_attrprune import collect_correct


class FakeDataset:
    records = [
        {"task": "q0", "answer": "A", "subject": "s0"},
        {"task": "q1", "answer": "B", "subject": "s1"},
        {"task": "q2", "answer": "D", "subject": "s2"},
    ]

    def __init__(self, *, split, data_dir, download):
        assert split == "dev"
        assert Path(data_dir).is_dir()
        assert download is False

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(self.records[index])

    @staticmethod
    def record_to_sample(record):
        return dict(record)


class PreparingFakeDataset(FakeDataset):
    constructions: list[tuple[str, bool]] = []

    def __init__(self, *, split, data_dir, download):
        assert split in {"dev", "val"}
        assert Path(data_dir).is_dir()
        type(self).constructions.append((split, download))


class FakeResult:
    def __init__(self, answer):
        self.final_answer = answer
        self.metrics = {
            "llm_calls": 1,
            "prompt_tokens": 3,
            "generated_tokens": 1,
            "elapsed_seconds": 0.01,
        }

    def to_trace_final(self):
        return {"output": self.final_answer, "metrics": self.metrics}


class FakeGraph:
    calls: list[int] = []
    answers = {0: "A", 1: "A", 2: "D"}

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def run_sample(self, sample, *, recorder, phase, sample_index):
        del sample
        assert phase == "calibration"
        type(self).calls.append(sample_index)
        invocation_id = recorder.new_invocation_id()
        recorder.record_generation(
            invocation_id,
            {
                "invocation_id": invocation_id,
                "phase": phase,
                "sample_index": sample_index,
                "round": 0,
                "agent_id": "knowledgeable_expert",
                "agent_type": "AnalyzeAgent",
                "role": "Knowlegable Expert",
                "incoming": [],
            },
            {
                "messages": [{"role": "user", "content": f"q{sample_index}"}],
                "rendered_prompt": f"q{sample_index}",
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "generated_ids": [[4]],
                "output_text": [self.answers[sample_index]],
                "generation_kwargs": {},
                "model_name": "stub",
            },
        )
        return FakeResult(self.answers[sample_index])


class FirstSampleTruncatedGraph(FakeGraph):
    async def run_sample(self, sample, *, recorder, phase, sample_index):
        result = await super().run_sample(
            sample,
            recorder=recorder,
            phase=phase,
            sample_index=sample_index,
        )
        if sample_index == 0:
            result.metrics["truncated_events"] = 1
        return result


def _write_data(data_dir: Path) -> None:
    for split in ("dev", "val"):
        path = data_dir / "data" / split / f"topic_{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("q,a,b,c,d,A\n", encoding="utf-8")


def _args(run_dir: Path, data_dir: Path, *, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="/models/Qwen3-8B",
        data_dir=data_dir,
        run_dir=run_dir,
        results_root=None,
        run_name=None,
        split="dev",
        start_index=0,
        limit=3,
        num_workers=2,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
        resume=resume,
        no_download=True,
    )


def test_collect_saves_every_trace_and_resume_skips_wrong_too(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "mmlu"
    _write_data(data_dir)
    FakeGraph.calls.clear()
    monkeypatch.setattr(collect_correct, "MMLUDataset", FakeDataset)
    monkeypatch.setattr(collect_correct, "HFChat", lambda name: {"model": name})
    monkeypatch.setattr(collect_correct, "MultiAgentGraph", FakeGraph)
    monkeypatch.setattr(collect_correct, "set_reproducible", lambda seed: None)

    summary = asyncio.run(
        collect_correct.collect(_args(run_dir, data_dir, resume=False))
    )

    assert summary["attempted"] == 3
    assert summary["correct"] == 2
    assert FakeGraph.calls == [0, 1, 2]
    attempts = [
        json.loads(line)
        for line in (run_dir / "calibration" / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["correct"] for item in attempts] == [True, False, True]
    correct = [
        json.loads(line)
        for line in (run_dir / "calibration" / "correct_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["sample_index"] for item in correct] == [0, 2]
    assert (run_dir / "calibration" / "traces" / "sample_00000" / "trace.json").is_file()
    wrong_trace = (
        run_dir / "calibration" / "traces" / "sample_00001" / "trace.json"
    )
    assert wrong_trace.is_file()
    wrong_payload = json.loads(wrong_trace.read_text(encoding="utf-8"))
    assert wrong_payload["final"]["correct"] is False
    assert (run_dir / "calibration" / "traces" / "sample_00002" / "trace.json").is_file()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["dataset"]["splits"]["dev"]["record_count"] == 1
    assert config["dataset"]["splits"]["val"]["record_count"] == 1
    aliases = config["prompt"]["reviewer_agent_ids"]
    assert len(aliases) == 2
    assert len(set(aliases.values())) == 2
    assert all(len(alias) == 4 for alias in aliases.values())

    resumed = asyncio.run(
        collect_correct.collect(_args(run_dir, data_dir, resume=True))
    )
    assert resumed["attempted"] == 3
    assert FakeGraph.calls == [0, 1, 2]
    assert len(
        (run_dir / "calibration" / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3
    resumed_config = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert resumed_config["prompt"]["reviewer_agent_ids"] == aliases


def test_collect_prepares_both_splits_before_dataset_fingerprinting(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "mmlu"
    _write_data(data_dir)
    PreparingFakeDataset.constructions.clear()
    FakeGraph.calls.clear()
    monkeypatch.setattr(collect_correct, "MMLUDataset", PreparingFakeDataset)
    monkeypatch.setattr(collect_correct, "HFChat", lambda name: {"model": name})
    monkeypatch.setattr(collect_correct, "MultiAgentGraph", FakeGraph)
    monkeypatch.setattr(collect_correct, "set_reproducible", lambda seed: None)
    args = _args(run_dir, data_dir, resume=False)
    args.no_download = False
    args.limit = 1

    asyncio.run(collect_correct.collect(args))

    assert PreparingFakeDataset.constructions == [("dev", True), ("val", True)]


def test_collect_saves_but_rejects_truncated_sample(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "mmlu"
    _write_data(data_dir)
    FakeGraph.calls.clear()
    monkeypatch.setattr(collect_correct, "MMLUDataset", FakeDataset)
    monkeypatch.setattr(collect_correct, "HFChat", lambda name: {"model": name})
    monkeypatch.setattr(
        collect_correct,
        "MultiAgentGraph",
        FirstSampleTruncatedGraph,
    )
    monkeypatch.setattr(collect_correct, "set_reproducible", lambda seed: None)

    with pytest.raises(RuntimeError, match="truncated traces"):
        asyncio.run(
            collect_correct.collect(_args(run_dir, data_dir, resume=False))
        )

    attempts = [
        json.loads(line)
        for line in (run_dir / "calibration" / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["status"] for item in attempts] == [
        "truncated",
        "completed",
        "completed",
    ]
    assert attempts[0]["prediction_correct"] is True
    assert (
        run_dir / "calibration" / "traces" / "sample_00000" / "trace.json"
    ).is_file()
