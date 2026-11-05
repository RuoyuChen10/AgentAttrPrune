"""CPU checks for the new prompt/data-aligned AgentDropout training run."""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT", ROOT.parent.parent))):
    sys.path.insert(0, str(directory))

from experiments import run_mmlu_attrprune_dev40 as runner

runner._local_dataset_namespace()
from AgentAttrPrune.prompt import MMLUPromptSet
from AgentAttrPrune.prompt.mmlu import WORKER_ROLES
from AgentDropout.graph.graph import Graph
from AgentDropout.llm.llm_registry import LLMRegistry
from experiments import mmlu_dev40_prompt_adapter as adapter
from experiments.aligned_train_mmlu import _infinite_loader, train
from experiments.evaluate_mmlu_dev40_sampled import realize_policy, evaluate
from experiments.dropout_artifacts import restore_checkpoint, save_artifacts


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def __deepcopy__(self, memo):
        return self

    async def agen(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return "D" if "top decision-maker" in messages[0]["content"] else "Reasoning: stub\nDecision: D"


@pytest.fixture
def alignment():
    def sample(i):
        return {"task": f"Question {i}\nOption A: a\nOption B: b\nOption C: c\nOption D: d\n",
                "question": f"Question {i}", "options": ["a", "b", "c", "d"], "answer": "D", "subject": "test"}
    return {"reference_run_config": {}, "workers": [
        {"agent_id": identity, "role": role, "node_index": i} for i, (identity, role) in enumerate(WORKER_ROLES[:5])],
        "reviewer_agent_ids": {identity: f"ID{i:02d}" for i, (identity, _) in enumerate(WORKER_ROLES[:5])},
        "calibration_samples": [sample(i) for i in range(40)],
        "evaluation_samples": [sample(i + 1000) for i in range(1531)],
        "source_hashes": {}, "fingerprint": "fixture-alignment", "prompt_version": MMLUPromptSet.VERSION}


@pytest.fixture
def context(tmp_path, monkeypatch, alignment):
    monkeypatch.setattr(runner, "load_alignment", lambda *_: deepcopy(alignment))
    args = runner.parse_args(["--run-dir", str(tmp_path / "run"), "--dry-run", "--model-path", "cpu-stub"])
    return args, runner.prepare_context(args)


@pytest.fixture
def graph(monkeypatch, alignment):
    llm = RecordingLLM()
    monkeypatch.setattr(LLMRegistry, "get", lambda *_: llm)
    return adapter.build_graph(alignment, "cpu-stub"), llm


def test_exact_pool_continuous_original_loader_and_preserved_budget(context):
    args, ctx = context
    dataset = runner.FrozenDataset(ctx["alignment"]["calibration_samples"], "dev")
    loader = _infinite_loader(dataset, np.random.default_rng(0))
    actual = [next(loader)[0] for _ in range(320)]
    planned = [i for step in ctx["plan"] for i in step["sample_indices"]]
    assert actual == planned
    assert Counter(actual[:160]) == Counter({i: 4 for i in range(40)})
    assert Counter(actual[160:]) == Counter({i: 4 for i in range(40)})
    assert ctx["config"]["training"]["batch_size"] == 16
    assert ctx["config"]["evaluation"]["batch_size"] == 4
    assert ctx["config"]["llm_batch_size"] == 4
    assert not ctx["run_dir"].exists()


def test_prompt_contract_and_dropout_sentinel(graph, alignment):
    policy, llm = graph
    worker = list(policy.nodes.values())[1]
    spatial = {"historian": {"role": "Historian", "output": "history"},
               "mathematician": {"role": "Mathematician", "output": "None."}}
    temporal = {"historian": {"role": "Historian", "output": "older history"},
                "critic": {"role": "Critic", "output": "previous"}}
    messages = worker._process_inputs({"task": "task"}, spatial, temporal)
    assert messages == MMLUPromptSet.worker_messages(
        role="Critic", task="task",
        spatial_incoming=[{"agent_id": "historian", "role": "Historian", "output": "history"}],
        temporal_incoming=[{"agent_id": "critic", "role": "Critic", "output": "previous"},
                           {"agent_id": "historian", "role": "Historian", "output": "older history"}])
    # Supply reversed input insertion order: reviewer still uses canonical IDs.
    info = {w["agent_id"]: {"role": w["role"], "output": "None." if i == 2 else "D"}
            for i, w in reversed(list(enumerate(alignment["workers"])))}
    expected = [{"agent_id": w["agent_id"], "role": w["role"],
                 "reviewer_agent_id": alignment["reviewer_agent_ids"][w["agent_id"]],
                 "output": info[w["agent_id"]]["output"]} for w in alignment["workers"]]
    assert policy.decision_node._process_inputs({"task": "task"}, info, {}) == \
        MMLUPromptSet.reviewer_messages(task="task", worker_outputs=expected)
    asyncio.run(worker._async_execute({"task": "task"}, {}, {}))
    assert llm.calls[-1][1] == {"max_tokens": 2048, "temperature": 0.0}


@pytest.mark.parametrize("seed", [0, 7, 1530])
def test_original_graph_sampling_execution_and_gradient_unchanged(graph, seed):
    aligned, llm = graph
    original = deepcopy(aligned)
    original.__class__ = Graph
    outputs = []
    for candidate in (aligned, original):
        candidate.generator = torch.Generator().manual_seed(seed)
        llm.calls.clear()
        result = asyncio.run(candidate.arun({"task": "task"}, 2, skip=True))
        result[1].backward()
        outputs.append((deepcopy(llm.calls), result[0], result[1].detach(),
                        [p.grad.clone() for p in candidate.spatial_logits_1],
                        candidate.spatial_adj_matrix.copy(), candidate.temporal_adj_matrix.copy()))
    assert outputs[0][:2] == outputs[1][:2]
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert all(torch.equal(a, b) for a, b in zip(outputs[0][3], outputs[1][3]))
    assert np.array_equal(outputs[0][4], outputs[1][4])
    assert np.array_equal(outputs[0][5], outputs[1][5])


def test_full_original_training_checkpoint_and_resumable_evaluation(context, graph):
    args, ctx = context
    policy, llm = graph
    runner.freeze_context(ctx)
    asyncio.run(train(policy, runner.FrozenDataset(ctx["alignment"]["calibration_samples"], "dev"),
                      num_iters=10, num_rounds=2, lr=0.1, batch_size=16, prune_every=5,
                      pruning_rate=0.25, delta=0.1, seed=0,
                      metrics_path=ctx["run_dir"] / "training_metrics.jsonl"))
    runner.validate_training(ctx)
    assert len(policy.skip_nodes) == 2
    artifacts = save_artifacts(policy, ctx["run_dir"] / "pruning_artifacts", ctx["config"])
    runner._training_completion(ctx, save=True)
    fresh = adapter.build_graph(ctx["alignment"], "cpu-stub")
    restore_checkpoint(fresh, artifacts["checkpoint"])
    assert fresh.skip_nodes == policy.skip_nodes
    for edge in ("spatial", "temporal"):
        for suffix in ("logits", "masks"):
            assert all(torch.equal(a, b) for a, b in zip(getattr(policy, f"{edge}_{suffix}"), getattr(fresh, f"{edge}_{suffix}")))
    dataset = runner.FrozenDataset(ctx["alignment"]["evaluation_samples"], "val")
    kwargs = dict(num_rounds=2, output_dir=ctx["run_dir"] / "evaluation",
                  start_index=0, limit=4, batch_size=4, seed=0, run_config=ctx["config"])
    asyncio.run(evaluate(fresh, dataset, resume=False, **kwargs))
    before = len(llm.calls)
    asyncio.run(evaluate(fresh, dataset, resume=True, **kwargs))
    assert len(llm.calls) == before
    assert runner.validate_predictions(ctx) == 4
    runner._training_completion(ctx)
    # Duplicate predictions cannot be silently collapsed by the inherited reader.
    path = ctx["run_dir"] / "evaluation/predictions.jsonl"
    with path.open("a") as handle:
        handle.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_predictions(ctx)


@pytest.mark.parametrize("filename", ["run_config.json", "training_draw_plan.json", "prompt_data_alignment.json"])
def test_frozen_metadata_changes_rejected(context, filename):
    args, ctx = context
    runner.freeze_context(ctx)
    path = ctx["run_dir"] / filename
    value = json.loads(path.read_text())
    if isinstance(value, list):
        value[0]["sample_indices"][0] = 999
    else:
        value["tampered"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Frozen experiment changed"):
        runner.prepare_context(args)


def test_failed_generation_never_becomes_valid_training_example(graph):
    policy, llm = graph
    async def fail(*args, **kwargs):
        raise RuntimeError("simulated OOM")
    llm.agen = fail
    with pytest.raises(RuntimeError, match="simulated OOM"):
        asyncio.run(policy.arun({"task": "task"}, 2, max_tries=1))


def test_evaluation_batching_preserves_sampled_graphs_and_answers(tmp_path, alignment, graph):
    policy, llm = graph
    policy.skip_nodes = [2, 3]
    dataset = runner.FrozenDataset(alignment["evaluation_samples"], "val")
    results = []
    for batch_size in (1, 4):
        destination = tmp_path / str(batch_size)
        asyncio.run(evaluate(policy, dataset, num_rounds=2, output_dir=destination,
                             start_index=0, limit=4, batch_size=batch_size, seed=0,
                             resume=False, run_config={}))
        results.append(runner._jsonl(destination / "predictions.jsonl"))
    assert results[0] == results[1]
    a, b = realize_policy(policy, 0), realize_policy(policy, 1)
    for realized in (a, b):
        assert realized.optimized_spatial and realized.optimized_temporal
        asyncio.run(realized.arun({"task": "task"}, 2, case=True))
    assert a.execution_trace[0]["spatial_edges"] != b.execution_trace[0]["spatial_edges"]


@pytest.mark.parametrize("seed", [0, 7, 1530])
def test_native_sampler_temporal_self_edges_and_log_probability(graph, seed):
    policy, llm = graph
    policy.skip_nodes = [2, 3]
    # Make temporal self-edges certain, exposing the legacy fixed-mask path's
    # extra cycle filtering if the stochastic flags are accidentally disabled.
    for logits in policy.temporal_logits:
        with torch.no_grad():
            logits.fill_(100.0)
    original = deepcopy(policy)
    original.__class__ = Graph
    original.generator = torch.Generator().manual_seed(seed)
    expected = asyncio.run(original.arun({"task": "task"}, 2, case=True))
    realized = realize_policy(policy, seed)
    actual = asyncio.run(realized.arun({"task": "task"}, 2, case=True))
    assert actual[0] == expected[0]
    assert torch.equal(actual[1].detach(), expected[1].detach())
    assert actual[2] == expected[2]
    assert np.array_equal(realized.spatial_adj_matrix, original.spatial_adj_matrix)
    assert np.array_equal(realized.temporal_adj_matrix, original.temporal_adj_matrix)
    assert len(realized.execution_trace[1]["temporal_edges"]) == 25
    assert all([i, i] in realized.execution_trace[1]["temporal_edges"] for i in range(5))


def test_real_reference_question_sets_and_saved_prompt_bytes(monkeypatch):
    if not runner.DEFAULT_REFERENCE.is_dir():
        pytest.skip("Local original seed0 reference is unavailable")
    alignment = runner.load_alignment(runner.DEFAULT_REFERENCE)
    assert len(alignment["calibration_samples"]) == 40
    assert len(alignment["evaluation_samples"]) == 1531
    assert alignment["calibration_samples_sha256"] == "bd48c1f61dc3ed42d786e260d1bbce14bf9077c93b8c29a1b0e53d2c10dc1885"
    assert alignment["evaluation_samples_sha256"] == "e751c91405516b27f46b873d628ee5fd8f51ea61fefd6a3af18cdab74654cd4a"
    monkeypatch.setattr(LLMRegistry, "get", lambda *_: RecordingLLM())
    policy = adapter.build_graph(alignment, "cpu-stub")
    checked = 0
    for index, sample in enumerate(alignment["calibration_samples"]):
        path = runner.DEFAULT_REFERENCE / f"calibration/traces/sample_{index:05d}/trace.json"
        trace = json.loads(path.read_text())
        assert trace["sample"] == sample
        for event in trace["events"]:
            reviewer = event["agent_id"] == "dense_reviewer"
            node = policy.decision_node if reviewer else policy.nodes[event["agent_id"]]
            spatial = {item["agent_id"]: item for item in event["incoming"]
                       if reviewer or item["edge_type"] == "spatial"}
            temporal = {item["agent_id"]: item for item in event["incoming"]
                        if not reviewer and item["edge_type"] == "temporal"}
            assert node._process_inputs({"task": sample["task"]}, spatial, temporal) == event["input_messages"]
            checked += 1
    assert checked == 440
