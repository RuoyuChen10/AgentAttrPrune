"""Offline checks for original-GSM8K protocol with the exact AgentAttrPrune seed0 cohort."""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT", ROOT.parent.parent))):
    sys.path.insert(0, str(directory))
from experiments import run_gsm8k_attrprune_seed0 as runner
runner._local_dataset_namespace()
from AgentAttrPrune.prompt import GSM8KPromptSet
from AgentAttrPrune.prompt.gsm8k import WORKER_ROLES
from AgentDropout.graph.graph import Graph
from AgentDropout.llm.llm_registry import LLMRegistry
from experiments import gsm8k_seed0_prompt_adapter as adapter
from experiments.run_gsm8k_aligned import train
from experiments.evaluate_gsm8k_seed0_sampled import realize_policy, evaluate
from experiments.dropout_artifacts import restore_checkpoint, save_artifacts


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def __deepcopy__(self, memo):
        return self

    async def agen(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), kwargs))
        return "The answer is 42"


@pytest.fixture
def alignment():
    def sample(i):
        return {"task": f"Question {i}\u0085with Unicode separator", "question": f"Question {i}\u0085with Unicode separator",
                "benchmark": "gsm8k", "answer": "42"}
    return {"reference_run_config": {}, "workers": [
        {"agent_id": identity, "role": role, "node_index": i} for i, (identity, role) in enumerate(WORKER_ROLES)],
        "reviewer_agent_ids": {identity: f"ID{i:02d}" for i, (identity, _) in enumerate(WORKER_ROLES)},
        "calibration_samples": [sample(i) for i in range(40)],
        "evaluation_samples": [sample(i + 1000) for i in range(1319)],
        "source_hashes": {}, "fingerprint": "fixture-alignment", "prompt_version": GSM8KPromptSet.VERSION}


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


def original_args():
    return SimpleNamespace(node_iters=2, node_batch_size=20, edge_iters=4, edge_batch_size=10,
        num_rounds=2, learning_rate=0.1, prune_every=2, pruning_rate=0.1, delta=0.1, seed=0)


def test_original_budget_order_and_reward(context):
    _, ctx = context
    plan = ctx["plan"]
    assert len(plan) == 6
    assert [len(x["sample_indices"]) for x in plan] == [20, 20, 10, 10, 10, 10]
    assert [i for step in plan[:2] for i in step["sample_indices"]] == list(range(40))
    assert [i for step in plan[2:] for i in step["sample_indices"]] == list(range(40))
    dataset = runner.FrozenDataset(ctx["alignment"]["calibration_samples"], "train")
    assert dataset.postprocess_answer(["The answer is 42"]) == dataset.record_to_target_answer(dataset[0]) == "42.0"
    assert ctx["config"]["training"]["total_draws"] == 80
    assert ctx["config"]["llm_batch_size"] == ctx["config"]["evaluation"]["batch_size"] == 4


def test_exact_prompt_order_and_original_gsm_dropout_sentinel(graph, alignment):
    policy, llm = graph
    worker = policy.nodes["programming_expert"]
    spatial = {"math_solver": {"role": "Math Solver", "output": "None."}}
    temporal = {"inspector": {"role": "Inspector", "output": "old"},
                "math_solver": {"role": "Math Solver", "output": "previous"}}
    assert worker._process_inputs({"task": "task"}, spatial, temporal) == GSM8KPromptSet.worker_messages(
        role="Programming Expert", task="task", spatial_incoming=[
            {"agent_id": "math_solver", "role": "Math Solver", "output": "None."}], temporal_incoming=[
            {"agent_id": "math_solver", "role": "Math Solver", "output": "previous"},
            {"agent_id": "inspector", "role": "Inspector", "output": "old"}])
    outputs = {w["agent_id"]: {"role": w["role"], "output": "None." if i == 2 else "42"}
               for i, w in reversed(list(enumerate(alignment["workers"])))}
    expected = [{"agent_id": w["agent_id"], "role": w["role"],
                 "reviewer_agent_id": alignment["reviewer_agent_ids"][w["agent_id"]],
                 "output": outputs[w["agent_id"]]["output"]} for w in alignment["workers"]]
    assert policy.decision_node._process_inputs({"task": "task"}, outputs, {}) == GSM8KPromptSet.reviewer_messages(
        task="task", worker_outputs=expected)
    assert asyncio.run(worker._async_execute({"task": "task"}, {}, {})) == "The answer is 42"
    assert llm.calls[-1][1] == {"max_tokens": 2048, "temperature": 0.0}


def test_full_80_draw_training_matches_original_graph_then_resume(context, graph):
    _, ctx = context
    policy, llm = graph
    original = deepcopy(policy)
    original.__class__ = Graph
    dataset = runner.FrozenDataset(ctx["alignment"]["calibration_samples"], "train")
    runner.freeze_context(ctx)
    asyncio.run(train(policy, dataset, original_args(), ctx["run_dir"] / "training_metrics.jsonl"))
    asyncio.run(train(original, dataset, original_args(), None))
    runner.validate_training(ctx)
    assert len(policy.skip_nodes) == 2 and policy.skip_nodes == original.skip_nodes
    for name in ("spatial_logits", "temporal_logits", "spatial_logits_1", "temporal_logits_1", "spatial_masks", "temporal_masks"):
        assert all(torch.equal(a, b) for a, b in zip(getattr(policy, name), getattr(original, name)))
    metrics = runner._jsonl(ctx["run_dir"] / "training_metrics.jsonl")
    assert Counter(i for row in metrics for i in row["sample_indices"]) == Counter({i: 2 for i in range(40)})
    assert [r["pruned_this_iteration"] for r in metrics if r["stage"] == "edge_dropout"] == [False, True, False, True]
    artifacts = save_artifacts(policy, ctx["run_dir"] / "pruning_artifacts", ctx["config"])
    runner._training_completion(ctx, save=True)
    restored = adapter.build_graph(ctx["alignment"], "cpu-stub")
    restore_checkpoint(restored, artifacts["checkpoint"])
    kwargs = dict(num_rounds=2, output_dir=ctx["run_dir"] / "evaluation", start_index=0,
                  limit=4, batch_size=4, seed=0, run_config=ctx["config"])
    dataset = runner.FrozenDataset(ctx["alignment"]["evaluation_samples"], "test")
    asyncio.run(evaluate(restored, dataset, resume=False, **kwargs))
    before = len(llm.calls)
    asyncio.run(evaluate(restored, dataset, resume=True, **kwargs))
    assert len(llm.calls) == before
    assert runner.validate_predictions(ctx) == 4
    runner._training_completion(ctx)
    # The valid U+0085 character must not split any JSON record during summary.
    assert len(runner._jsonl(ctx["run_dir"] / "evaluation/predictions.jsonl")) == 4
    p = ctx["run_dir"] / "evaluation/predictions.jsonl"
    with p.open("a") as handle:
        handle.write(p.read_text().split("\n")[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_predictions(ctx)


@pytest.mark.parametrize("seed", [0, 7, 1318])
def test_original_sampler_temporal_self_edges_and_log_probability(graph, seed):
    policy, llm = graph
    policy.skip_nodes = [2, 3]
    for logits in policy.temporal_logits:
        with torch.no_grad():
            logits.fill_(100.0)
    original = deepcopy(policy)
    original.__class__ = Graph
    original.generator = torch.Generator().manual_seed(seed)
    expected = asyncio.run(original.arun({"task": "task"}, 2, case=True))
    realized = realize_policy(policy, seed)
    actual = asyncio.run(realized.arun({"task": "task"}, 2, case=True))
    assert actual[0] == expected[0] and actual[2] == expected[2]
    assert torch.equal(actual[1].detach(), expected[1].detach())
    assert np.array_equal(realized.spatial_adj_matrix, original.spatial_adj_matrix)
    assert np.array_equal(realized.temporal_adj_matrix, original.temporal_adj_matrix)
    assert len(realized.execution_trace[1]["temporal_edges"]) == 16
    assert all([i, i] in realized.execution_trace[1]["temporal_edges"] for i in range(4))


def test_evaluation_batch_size_keeps_identical_graphs_and_answers(context, graph):
    _, ctx = context
    policy, llm = graph
    policy.skip_nodes = [2, 3]
    save_artifacts(policy, ctx["run_dir"] / "pruning_artifacts", ctx["config"])
    dataset = runner.FrozenDataset(ctx["alignment"]["evaluation_samples"], "test")
    outputs = []
    for size in (1, 4):
        destination = ctx["run_dir"] / f"eval_batch{size}"
        asyncio.run(evaluate(policy, dataset, num_rounds=2, output_dir=destination,
            start_index=0, limit=4, batch_size=size, seed=0, resume=False, run_config=ctx["config"]))
        outputs.append(runner._jsonl(destination / "predictions.jsonl"))
    assert outputs[0] == outputs[1]


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


def test_failed_generation_is_not_a_valid_zero_reward(graph):
    policy, llm = graph
    async def fail(*args, **kwargs):
        raise RuntimeError("simulated OOM")
    llm.agen = fail
    with pytest.raises(RuntimeError, match="simulated OOM"):
        asyncio.run(policy.arun({"task": "task"}, 2, max_tries=1))


def test_real_seed0_cohort_and_all_360_saved_prompts(monkeypatch):
    if not runner.DEFAULT_REFERENCE.is_dir():
        pytest.skip("Local original seed0 reference is unavailable")
    alignment = runner.load_alignment()
    assert alignment["calibration_samples_sha256"] == "99e8b7b2a5b061af367985403e5d48072b1b879f566dcef9d8c1fd4ab80ebece"
    assert alignment["evaluation_samples_sha256"] == "90456244d9ef7af6a0eaa0c3d86d12c5de0cf34c3c92efb891b4c097e65fde00"
    monkeypatch.setattr(LLMRegistry, "get", lambda *_: RecordingLLM())
    policy = adapter.build_graph(alignment, "cpu-stub")
    checked = 0
    for index, sample in enumerate(alignment["calibration_samples"]):
        trace = json.loads((runner.DEFAULT_REFERENCE / f"calibration/traces/sample_{index:05d}/trace.json").read_text())
        for event in trace["events"]:
            reviewer = event["agent_id"] == "dense_reviewer"
            node = policy.decision_node if reviewer else policy.nodes[event["agent_id"]]
            spatial = {i["agent_id"]: i for i in event["incoming"] if reviewer or i["edge_type"] == "spatial"}
            temporal = {i["agent_id"]: i for i in event["incoming"] if not reviewer and i["edge_type"] == "temporal"}
            assert node._process_inputs({"task": sample["task"]}, spatial, temporal) == event["input_messages"]
            checked += 1
    assert checked == 360


def test_real_hf_backend_batches_four_requests_on_cpu():
    import queue
    import threading
    from AgentDropout.llm.hf_chat import HFChat, _GenerationRequest
    from transformers import BatchEncoding
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, prompts, **kwargs):
            assert kwargs == {"return_tensors": "pt", "padding": True}
            return BatchEncoding({"input_ids": torch.ones((len(prompts), 3), dtype=torch.long),
                                  "attention_mask": torch.ones((len(prompts), 3), dtype=torch.long)})
        def batch_decode(self, output, **kwargs):
            return ["The answer is 42"] * output.shape[0]
    class Model:
        device = torch.device("cpu")
        calls = []
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["input_ids"].shape[0] == 4
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 2048
            return torch.cat((kwargs["input_ids"], torch.full((4, 1), 42)), dim=1)
    backend = object.__new__(HFChat)
    backend.tokenizer = Tokenizer()
    model = Model()
    backend.models = [model]
    backend._requests = queue.Queue()
    backend._batch_size = 4
    backend._batch_wait_seconds = 0.01
    requests = [_GenerationRequest(str(i), 2048, 0.0) for i in range(4)]
    for request in requests:
        backend._requests.put(request)
    thread = threading.Thread(target=backend._batch_worker, args=(0,), daemon=True)
    thread.start()
    for request in requests:
        assert request.event.wait(5)
        assert request.error is None and request.result == "The answer is 42"
    assert len(model.calls) == 1
