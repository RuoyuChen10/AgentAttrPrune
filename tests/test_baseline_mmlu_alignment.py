"""CPU checks for the cross-repository MMLU question/prompt contract."""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from AgentAttrPrune.datasets import MMLUDataset
from AgentAttrPrune.prompt import MMLUPromptSet
from experiments_attrprune import baseline_mmlu_alignment as align
from experiments_attrprune.common import benchmark_data_identity, build_run_config, write_json


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "CALIBRATION_SIZE", 2)
    monkeypatch.setattr(align, "EVALUATION_SIZE", 3)
    data = tmp_path / "data"
    for split, count in (("dev", 8), ("val", 3)):
        directory = data / split
        directory.mkdir(parents=True)
        with (directory / f"arithmetic_{split}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows([[f"{split} question {i}", "one", "two", "three", "four", "A"] for i in range(count)])
    config = build_run_config(benchmark="mmlu", model_name="stub", num_workers=5, num_rounds=2,
        temperature=0, max_new_tokens=2048, seed=0, calibration_split="dev", evaluation_split="val",
        dataset_identity=benchmark_data_identity("mmlu", data, splits=("dev", "val")),
        reviewer_agent_ids={agent: f"Id{i:02d}" for i, (agent, _) in enumerate(align.WORKER_ROLES[:5])})
    root = tmp_path / "reference"
    write_json(root / "run_config.json", config)
    dataset = MMLUDataset(split="dev", data_dir=data)
    selected = []
    for index in range(2):
        sample = dataset.record_to_sample(dataset[index])
        events, prior = [], {}
        for round_index in range(2):
            for j, worker in enumerate(config["workers"]):
                incoming = []
                for edge, source_round, sources in (
                    ("spatial", round_index, config["workers"][:j]),
                    ("temporal", round_index - 1, config["workers"] if round_index else []),
                ):
                    incoming.extend({"agent_id": s["agent_id"], "role": s["role"], "edge_type": edge,
                                     "round": source_round, "output": prior[(source_round, s["agent_id"])]} for s in sources)
                messages = MMLUPromptSet.worker_messages(role=worker["role"], task=sample["task"],
                    spatial_incoming=[x for x in incoming if x["edge_type"] == "spatial"],
                    temporal_incoming=[x for x in incoming if x["edge_type"] == "temporal"])
                events.append({"round": round_index, "agent_id": worker["agent_id"], "incoming": incoming,
                               "input_messages": messages, "hit_max_new_tokens": False})
                prior[(round_index, worker["agent_id"])] = f"Reasoning: source {worker['agent_id']} round {round_index}. Decision: A"
        incoming = [{"agent_id": w["agent_id"], "role": w["role"], "edge_type": "spatial", "round": 1,
                     "output": prior[(1, w["agent_id"])], "reviewer_agent_id": config["prompt"]["reviewer_agent_ids"][w["agent_id"]]}
                    for w in config["workers"]]
        events.append({"round": 2, "agent_id": "dense_reviewer", "incoming": incoming,
                       "input_messages": MMLUPromptSet.reviewer_messages(task=sample["task"], worker_outputs=incoming)})
        trace = {"generation_complete": True, "token_ids_complete": True, "truncated_event_ids": [],
                 "config": {**config, "sample_index": index}, "sample": sample, "events": events,
                 "final": {"correct": index == 0}}
        path = root / f"calibration/traces/sample_{index:05d}/trace.json"
        write_json(path, trace)
        selected.append({"sample_id": f"sample_{index:05d}", "sample_index": index,
                         "trace_json": str(path), "sha256": align.file_sha(path)})
    write_json(root / "calibration/selection_manifest.json", {
        "expected_sample_count": 2, "expected_start_index": 0, "samples": selected})
    return root, data, config


def test_exact_cohort_includes_incorrect_and_full_validation(reference):
    root, data, config = reference
    contract = align.load_alignment(root)
    for split, name, indices in (("dev", "calibration_samples", range(2)), ("val", "evaluation_samples", range(3))):
        dataset = MMLUDataset(split=split, data_dir=data)
        assert contract[name] == [dataset.record_to_sample(dataset[i]) for i in indices]
    assert contract["reference_run_config"] == config
    assert contract["reviewer_agent_ids"] == config["prompt"]["reviewer_agent_ids"]
    without_fingerprint = {k: v for k, v in contract.items() if k != "fingerprint"}
    assert contract["fingerprint"] == align.digest(without_fingerprint)


def test_frozen_contract_rejects_cohort_change(reference, tmp_path):
    contract = align.load_alignment(reference[0])
    target = tmp_path / "new_baseline"
    path = align.freeze_alignment(target, contract)
    original = path.read_bytes()
    align.freeze_alignment(target, contract)
    assert path.read_bytes() == original
    changed = copy.deepcopy(contract)
    changed["calibration_samples"].reverse()
    with pytest.raises(ValueError):
        align.freeze_alignment(target, changed)
    assert path.read_bytes() == original


def test_source_change_rejected(reference):
    contract = align.load_alignment(reference[0])
    path = reference[0] / "calibration/traces/sample_00001/trace.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        align.verify_sources(contract)
    with pytest.raises(ValueError, match="SHA changed"):
        align.load_alignment(reference[0])


def test_prompt_drift_rejected(reference, monkeypatch):
    original = MMLUPromptSet.get_analyze_constraint
    monkeypatch.setattr(MMLUPromptSet, "get_analyze_constraint", staticmethod(lambda role: original(role) + "Changed instruction."))
    with pytest.raises(ValueError, match="prompt differs"):
        align.load_alignment(reference[0])


def test_validation_data_change_rejected(reference):
    path = reference[1] / "val/arithmetic_val.csv"
    path.write_text(path.read_text().replace("val question 0", "different question"))
    with pytest.raises(ValueError, match="dataset differs"):
        align.load_alignment(reference[0])


def test_missing_calibration_record_rejected(reference):
    path = reference[0] / "calibration/selection_manifest.json"
    value = align.load_json(path)
    value["samples"].pop()
    write_json(path, value)
    with pytest.raises(ValueError, match="complete original dev40"):
        align.load_alignment(reference[0])
