from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from AgentAttrPrune.pruning import RoundPolicy
from experiments_attrprune.common import (
    build_run_config,
    mmlu_data_identity,
    write_json,
)
from experiments_attrprune import evaluate_mmlu


MODEL = "/models/Qwen3-8B"


class FakeDataset:
    records = [
        {"task": "q0", "answer": "A", "subject": "s0"},
        {"task": "q1", "answer": "B", "subject": "s1"},
        {"task": "q2", "answer": "D", "subject": "s2"},
    ]
    construction = None

    def __init__(self, *, split, data_dir, download):
        type(self).construction = {
            "split": split,
            "data_dir": Path(data_dir),
            "download": download,
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(self.records[index])

    @staticmethod
    def record_to_sample(record):
        return dict(record)


class FakeResult:
    def __init__(self, answer, metrics, edge_counts):
        self.final_answer = answer
        self.metrics = metrics
        self.edge_counts = edge_counts

    def to_trace_final(self):
        return {
            "output": self.final_answer,
            "metrics": self.metrics,
            "edge_counts": self.edge_counts,
        }


class FakeGraph:
    instances = []
    answers = {0: "Decision: A", 1: "Decision: A", 2: "Decision: D"}

    def __init__(
        self,
        llm,
        worker_specs,
        num_rounds,
        policy,
        temperature,
        max_new_tokens,
    ):
        self.llm = llm
        self.worker_specs = list(worker_specs)
        self.num_rounds = num_rounds
        self.policy = policy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        type(self).instances.append(self)

        count = len(self.worker_specs)
        candidate = (
            num_rounds * count * (count - 1) // 2
            + max(0, num_rounds - 1) * count * count
        )
        active = sum(
            int(policy.mask(kind, round_index).sum().item())
            for kind in ("spatial", "temporal")
            for round_index in range(num_rounds)
        )
        self.edge_counts = {
            "worker_active": active,
            "worker_candidate": candidate,
            "worker_retention": active / candidate if candidate else 1.0,
            "reviewer_dense_edges": count,
        }

    async def run_sample(
        self, sample, *, recorder, phase, sample_index
    ):
        assert phase == "evaluation"
        return FakeResult(
            self.answers[sample_index],
            {
                "llm_calls": 5,
                "prompt_tokens": 10 + sample_index,
                "generated_tokens": 2,
                "elapsed_seconds": 0.25,
            },
            self.edge_counts,
        )


def make_run(tmp_path: Path, *, num_workers: int = 2) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    config = build_run_config(
        model_name=MODEL,
        num_workers=num_workers,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
    )
    write_json(run_dir / "run_config.json", config)
    return run_dir, config


def make_args(run_dir: Path, *, mode="dense", save_traces=False, limit=3):
    return argparse.Namespace(
        model_name=MODEL,
        data_dir=run_dir / "data",
        run_dir=run_dir,
        results_root=None,
        run_name=None,
        mode=mode,
        split="val",
        start_index=0,
        limit=limit,
        seed=0,
        save_traces=save_traces,
        no_download=True,
    )


def install_fakes(monkeypatch):
    FakeGraph.instances.clear()
    monkeypatch.setattr(evaluate_mmlu, "MMLUDataset", FakeDataset)
    monkeypatch.setattr(evaluate_mmlu, "HFChat", lambda name: {"model": name})
    monkeypatch.setattr(evaluate_mmlu, "MultiAgentGraph", FakeGraph)
    monkeypatch.setattr(evaluate_mmlu, "set_reproducible", lambda seed: None)


def write_half_policy(run_dir: Path, config: dict, mode: str) -> Path:
    workers = config["workers"]
    policy = RoundPolicy.from_dict(
        {
            "schema_version": 1,
            "run_fingerprint": config["fingerprint"],
            "metric": mode,
            "agents": [item["agent_id"] for item in workers],
            "agent_roles": [item["role"] for item in workers],
            "agent_types": [item["agent_type"] for item in workers],
            "num_rounds": 2,
            "matrix_orientation": "[round][source][target]",
            "masks": {
                "spatial": [
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                ],
                "temporal": [
                    [[0, 0], [0, 0]],
                    [[1, 0], [0, 1]],
                ],
            },
        }
    )
    path = run_dir / "attribution" / f"pruned_policy_{mode}.json"
    return policy.to_json(path)


def test_pruned_evaluation_covers_correct_and_incorrect_samples(monkeypatch, tmp_path):
    run_dir, config = make_run(tmp_path)
    source_policy = write_half_policy(run_dir, config, "binary")
    install_fakes(monkeypatch)

    summary = asyncio.run(
        evaluate_mmlu.evaluate(make_args(run_dir, mode="binary"))
    )

    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["llm_calls"] == 15
    assert summary["prompt_tokens"] == 33
    assert summary["generated_tokens"] == 6
    assert summary["graph_elapsed_seconds"] == pytest.approx(0.75)
    assert summary["retained_edges"] == 3
    assert summary["candidate_edges"] == 6
    assert summary["edge_retention"] == pytest.approx(0.5)
    assert summary["source_policy_json"] == str(source_policy)
    assert summary["run_fingerprint"] == config["fingerprint"]

    evaluation_dir = run_dir / "evaluation" / "binary_val"
    assert json.loads(
        (evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )["total"] == 3
    prediction_records = [
        json.loads(line)
        for line in (evaluation_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["correct"] for item in prediction_records] == [True, False, True]
    resolved = RoundPolicy.from_json(evaluation_dir / "resolved_policy.json")
    assert resolved.spatial_masks.tolist() == RoundPolicy.from_json(
        source_policy
    ).spatial_masks.tolist()

    graph = FakeGraph.instances[-1]
    assert graph.num_rounds == config["num_rounds"]
    assert graph.temperature == config["decoding"]["temperature"]
    assert graph.max_new_tokens == config["decoding"]["max_new_tokens"]
    assert FakeDataset.construction["split"] == "val"
    assert FakeDataset.construction["download"] is False


def test_dense_evaluation_writes_resolved_policy_and_optional_traces(
    monkeypatch, tmp_path
):
    run_dir, config = make_run(tmp_path)
    install_fakes(monkeypatch)

    summary = asyncio.run(
        evaluate_mmlu.evaluate(
            make_args(run_dir, mode="dense", save_traces=True, limit=1)
        )
    )

    assert summary["source_policy_json"] is None
    assert summary["retained_edges"] == summary["candidate_edges"] == 6
    assert summary["edge_retention"] == 1.0
    assert summary["traces_saved"] is True
    evaluation_dir = run_dir / "evaluation" / "dense_val"
    resolved = RoundPolicy.from_json(evaluation_dir / "resolved_policy.json")
    assert resolved.metadata["kind"] == "dense"
    trace = json.loads(
        (evaluation_dir / "traces" / "sample_00000" / "trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace["schema_version"] == 1
    assert trace["final"]["gold_answer"] == "A"
    assert trace["final"]["correct"] is True
    assert trace["config"]["fingerprint"] == config["fingerprint"]


def test_evaluation_rejects_config_or_missing_policy_before_model_load(
    monkeypatch, tmp_path
):
    run_dir, _ = make_run(tmp_path)
    model_loaded = False

    def fail_if_loaded(name):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model should not load")

    monkeypatch.setattr(evaluate_mmlu, "HFChat", fail_if_loaded)
    args = make_args(run_dir, mode="binary")
    with pytest.raises(FileNotFoundError, match="run the attribution stage"):
        asyncio.run(evaluate_mmlu.evaluate(args))
    assert model_loaded is False

    split_args = make_args(run_dir)
    split_args.split = "dev"
    with pytest.raises(ValueError, match="split must match"):
        asyncio.run(evaluate_mmlu.evaluate(split_args))
    assert model_loaded is False


def test_evaluation_rejects_policy_from_another_run_or_metric(tmp_path):
    run_dir, config = make_run(tmp_path)
    policy_path = write_half_policy(run_dir, config, "binary")
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload["run_fingerprint"] = "another-run"
    write_json(policy_path, payload)

    with pytest.raises(ValueError, match="run fingerprint"):
        asyncio.run(evaluate_mmlu.evaluate(make_args(run_dir, mode="binary")))

    payload["run_fingerprint"] = config["fingerprint"]
    payload["metric"] = "word_ratio"
    write_json(policy_path, payload)
    with pytest.raises(ValueError, match="policy metric"):
        asyncio.run(evaluate_mmlu.evaluate(make_args(run_dir, mode="binary")))


def test_evaluation_rejects_changed_mmlu_snapshot_before_writing_outputs(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    data_dir = run_dir / "data"
    for split in ("dev", "val"):
        path = data_dir / "data" / split / f"topic_{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("q,a,b,c,d,A\n", encoding="utf-8")
    identity = mmlu_data_identity(data_dir)
    config = build_run_config(
        model_name=MODEL,
        num_workers=2,
        num_rounds=2,
        temperature=0.0,
        max_new_tokens=64,
        seed=0,
        dataset_identity=identity,
    )
    write_json(run_dir / "run_config.json", config)
    install_fakes(monkeypatch)

    (data_dir / "data" / "val" / "topic_val.csv").write_text(
        "changed,a,b,c,d,B\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="data identity differs"):
        asyncio.run(evaluate_mmlu.evaluate(make_args(run_dir, mode="dense")))
    assert not (run_dir / "evaluation").exists()


def test_evaluation_cli_requires_a_named_mode():
    with pytest.raises(SystemExit):
        evaluate_mmlu.parse_args([])
    assert evaluate_mmlu.parse_args(["--mode", "word_ratio"]).mode == "word_ratio"
    assert evaluate_mmlu.parse_args(["--mode", "token_ratio"]).mode == "token_ratio"
