"""CPU coverage for exact prompts, original graph sampling, and safe resume."""
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import attrprune_gsm8k_seed0_adapter as adapter
from experiments import run_gsm8k_attrprune_seed0 as runner
from AgentAttrPrune.prompt.gsm8k import WORKER_ROLES
from AgentPrune.graph.graph import Graph


class FakeBackend:
    def __init__(self):
        self.calls, self.batch_sizes = [], []

    def __deepcopy__(self, memo):
        return self

    @asynccontextmanager
    async def async_batching(self, batch_size):
        self.batch_sizes.append(batch_size)
        yield

    async def agen(self, messages, *, temperature, max_tokens):
        await asyncio.sleep(0)
        self.calls.append(copy.deepcopy(messages))
        observer = adapter.generation_observer.get()
        if observer:
            observer({"messages": messages, "output_text": ["42"],
                      "rendered_prompt": str(messages), "model_name": "fake",
                      "input_ids": torch.tensor([[1, 2]]), "generated_ids": torch.tensor([[3]]),
                      "attention_mask": torch.tensor([[1, 1]]),
                      "generation_kwargs": {"temperature": temperature, "max_tokens": max_tokens,
                                            "effective_batch_size": 4}})
        return "42"


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(adapter, "_settings", None)
    monkeypatch.setattr(adapter, "_backend", None)
    workers = [{"agent_id": agent_id, "role": role, "node_index": index}
               for index, (agent_id, role) in enumerate(WORKER_ROLES[:4])]
    def sample(index):
        return {"question": f"Question {index}", "task": f"Question {index}",
                "answer": "42", "benchmark": "gsm8k"}
    alignment = {"fingerprint": "fixture-alignment", "workers": workers,
        "reviewer_agent_ids": {row["agent_id"]: f"W{index:03d}" for index, row in enumerate(workers)},
        "reference_run_config": {"model_name": "/unused/model", "decoding": {
            "temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}},
        "calibration_samples": [sample(index) for index in range(40)],
        "evaluation_samples": [sample(index + 100) for index in range(1319)], "source_hashes": {}}
    args = SimpleNamespace(protocol_reference=runner.DEFAULT_PROTOCOL, run_dir=tmp_path, stage="train")
    manifest = runner.make_manifest(args, alignment)
    llm = FakeBackend()
    adapter.configure(alignment, model_name=manifest["model_name"], backend=llm)

    def save_without_plot(graph, directory, training_config):
        artifact = runner.build_pruning_artifact(graph, training_config=training_config)
        runner.write_json(directory / "learned_edge_policy.json", {
            key: value for key, value in artifact.items() if key != "_checkpoint_tensors"})
    monkeypatch.setattr(runner, "save_pruning_artifacts", save_without_plot)
    return args, alignment, manifest, llm


def test_protocol_reuses_original_updates_on_exact_40_pool(context):
    _, _, manifest, _ = context
    plan = manifest["training_plan"]
    assert len(plan) == 400 and Counter(plan) == Counter({index: 10 for index in range(40)})
    assert all(plan[index * 10:(index + 1) * 10] == [index] * 10 for index in range(40))
    assert manifest["protocol"]["train_batch_size"] == 1
    assert manifest["protocol"]["samples_per_query"] == 10
    assert manifest["protocol"]["llm_microbatch_size"] == 4
    assert manifest["protocol"]["pruning_rate"] == 0.5
    assert manifest["protocol"]["low_rank_regularization"] is True
    from experiments.regularization import graph_regularization
    assert runner.graph_regularization is graph_regularization


def test_exact_worker_and_anonymous_reviewer_prompts(context):
    _, alignment, manifest, _ = context
    graph = runner.new_graph(manifest)
    workers = list(graph.nodes.values())
    info = {node.id: {"role": node.role, "output": f"Advice {index}"}
            for index, node in reversed(list(enumerate(workers)))}
    task = {"task": "Question?"}
    items = [dict(agent_id=source, **value) for source, value in info.items()]
    canonical = sorted(items, key=lambda row: list(graph.nodes).index(row["agent_id"]))
    for node in workers:
        assert node.build_messages(task, info, info) == adapter.GSM8KPromptSet.worker_messages(
            role=node.role, task=task["task"], spatial_incoming=items, temporal_incoming=canonical)
    for item in canonical:
        item["reviewer_agent_id"] = alignment["reviewer_agent_ids"][item["agent_id"]]
    assert graph.decision_node.build_messages(task, info, {}) == adapter.GSM8KPromptSet.reviewer_messages(
        task=task["task"], worker_outputs=canonical)


@pytest.mark.parametrize("prune_updates", [0, 1])
def test_original_graph_policy_sampling_and_gradient_are_preserved(context, monkeypatch, prune_updates):
    _, _, manifest, llm = context
    matched = runner.new_graph(manifest)
    original = Graph(domain="gsm8k", llm_name="unused", agent_names=[adapter.WORKER_NAME] * 4,
        decision_method=adapter.REVIEWER_NAME, optimized_spatial=True, optimized_temporal=True,
        node_kwargs=[{"id": row["agent_id"], "role": row["role"]} for row in manifest["workers"]])
    assert type(matched) is type(original) is Graph
    for graph in (original, matched):
        with torch.no_grad():
            graph.spatial_logits.copy_(torch.linspace(-1, 1, 16))
            graph.temporal_logits.copy_(torch.linspace(1, -1, 16))
        for _ in range(prune_updates):
            graph.update_masks(0.5)
    assert runner.policy_digest(original) == runner.policy_digest(matched)
    counts = [(12, 16), (6, 8)]
    assert (int(matched.spatial_masks.sum()), int(matched.temporal_masks.sum())) == counts[prune_updates]
    values = []
    for graph in (original, matched):
        torch.manual_seed(42)
        probability = 0
        matrices = []
        for round_index in range(2):
            probability += graph.construct_spatial_connection()
            probability += graph.construct_temporal_connection(round_index)
            matrices.append((graph.spatial_adj_matrix.copy(), graph.temporal_adj_matrix.copy()))
        (-probability).backward()
        values.append((probability, matrices, graph.spatial_logits.grad, graph.temporal_logits.grad))
    assert torch.equal(values[0][0], values[1][0])
    np.testing.assert_equal(values[0][1], values[1][1])
    assert torch.equal(values[0][2], values[1][2]) and torch.equal(values[0][3], values[1][3])


def test_training_resume_restores_adam_rng_and_skips_committed_rollouts(context, monkeypatch):
    args, alignment, manifest, llm = context
    uninterrupted = args.run_dir / "uninterrupted"
    uninterrupted.mkdir()
    args.run_dir = uninterrupted
    asyncio.run(runner.execute(args, alignment, manifest, None))
    expected = runner.load_checkpoint(uninterrupted, manifest, alignment)
    assert expected["iteration"] == 40 and len(expected["training"]) == 400
    assert len(llm.calls) == 400 * 9 and llm.batch_sizes == [4] * 40
    assert Counter(row["sample_index"] for row in expected["training"]) == Counter({index: 10 for index in range(40)})

    interrupted = uninterrupted.parent / "interrupted"
    interrupted.mkdir()
    args.run_dir = interrupted
    original_commit = runner.commit
    class StopAfterCommit(Exception):
        pass
    def stop_after_two(run_dir, state, graph, optimizer=None):
        original_commit(run_dir, state, graph, optimizer)
        if state["iteration"] == 2:
            raise StopAfterCommit
    monkeypatch.setattr(runner, "commit", stop_after_two)
    with pytest.raises(StopAfterCommit):
        asyncio.run(runner.execute(args, alignment, manifest, None))
    saved = runner.load_checkpoint(interrupted, manifest, alignment)
    assert saved["iteration"] == 2 and len(saved["training"]) == 20
    first_traces = {row["trace"]: row["trace_sha256"] for row in saved["training"]}
    calls_before = len(llm.calls)
    monkeypatch.setattr(runner, "commit", original_commit)
    asyncio.run(runner.execute(args, alignment, manifest, saved))
    resumed = runner.load_checkpoint(interrupted, manifest, alignment)
    assert len(llm.calls) - calls_before == 380 * 9
    assert resumed["policy_sha256"] == expected["policy_sha256"]
    assert all(runner.file_sha(interrupted / path) == sha for path, sha in first_traces.items())

    # Evaluation commits groups of four questions and retains the sampled policy.
    args.stage = "eval"
    def stop_after_eval_batch(run_dir, state, graph, optimizer=None):
        original_commit(run_dir, state, graph, optimizer)
        if len(state["evaluation"]) >= 4:
            raise StopAfterCommit
    monkeypatch.setattr(runner, "commit", stop_after_eval_batch)
    with pytest.raises(StopAfterCommit):
        asyncio.run(runner.execute(args, alignment, manifest, resumed))
    evaluated = runner.load_checkpoint(interrupted, manifest, alignment)
    assert len(evaluated["evaluation"]) == 4 and evaluated["policy_sha256"] == resumed["policy_sha256"]
    before = len(llm.calls)
    with pytest.raises(StopAfterCommit):
        asyncio.run(runner.execute(args, alignment, manifest, evaluated))
    evaluated_again = runner.load_checkpoint(interrupted, manifest, alignment)
    assert len(evaluated_again["evaluation"]) == 8
    assert len(llm.calls) - before == 4 * 9
    assert evaluated_again["evaluation"][:4] == evaluated["evaluation"]

    # The final-round graph is observed from actual incoming messages, including
    # temporal self-edges; no additional sampling is used for visualization.
    trace_path = interrupted / resumed["training"][-1]["trace"]
    trace = json.loads(trace_path.read_text())
    assert len(trace["events"]) == 9
    assert np.asarray(trace["realized_graph"]["spatial"]).shape == (2, 4, 4)
    assert not np.asarray(trace["realized_graph"]["temporal"])[0].any()
    trace["sample"]["question"] = "modified"
    runner.write_json(trace_path, trace)
    with pytest.raises(ValueError, match="trace.*modified"):
        runner.load_checkpoint(interrupted, manifest, alignment)


def test_bad_source_and_protocol_changes_are_rejected(context, tmp_path):
    args, _, manifest, _ = context
    bad = copy.deepcopy(json.loads(runner.DEFAULT_PROTOCOL.read_text()))
    bad["train_batch_size"] = 4
    path = tmp_path / "bad.json"
    runner.write_json(path, bad)
    with pytest.raises(ValueError, match="train_batch_size"):
        runner.protocol_from_artifact(path)
    runner.write_json(path, {"source": 1})
    altered = copy.deepcopy(manifest)
    altered["source_hashes"] = {str(path): runner.file_sha(path)}
    runner.write_json(path, {"source": 2})
    with pytest.raises(ValueError, match="Source changed"):
        runner.verify_manifest_sources(altered)


def test_real_async_batch_session_groups_four_model_requests(context, monkeypatch):
    args, alignment, manifest, _ = context
    # Use the production HFChat.agen/async_batching/_AsyncBatchSession code;
    # replace only actual model generation so this remains a CPU test.
    llm = object.__new__(adapter.HFChat)
    llm.logits_processors = []
    llm.model = SimpleNamespace(generation_config=SimpleNamespace(num_beams=1, min_length=0))
    sizes = []
    def generation(messages, max_tokens):
        return {"messages": messages, "output_text": ["42"], "rendered_prompt": str(messages),
                "model_name": "fake", "input_ids": torch.tensor([[1, 2]]),
                "generated_ids": torch.tensor([[3]]), "attention_mask": torch.tensor([[1, 1]]),
                "generation_kwargs": {"temperature": 0.0, "max_tokens": max_tokens}}
    def generate_batch(requests):
        sizes.append(len(requests))
        return [("42", generation(request.messages, request.max_tokens)) for request in requests]
    def generate_single(messages, *, max_tokens, temperature, num_comps):
        sizes.append(1)
        adapter.generation_observer.get()(generation(messages, max_tokens))
        return "42"
    llm._generate_batch, llm._generate = generate_batch, generate_single
    monkeypatch.setattr(adapter, "_backend", llm)
    graph = runner.new_graph(manifest)
    identities = [{"phase": "evaluation", "sample_index": index, "ordinal": index} for index in range(4)]
    results, _ = asyncio.run(runner.run_batch(graph, alignment["evaluation_samples"][:4], identities,
                                             args.run_dir / "real_batch_session", manifest))
    assert len(results) == 4 and sum(sizes) == 36 and max(sizes) == 4
    events = [event for row, _ in results for event in json.loads(open(row["trace"]).read())["events"]]
    assert any(event["generation"]["generation_kwargs"]["effective_batch_size"] == 4 for event in events)
    assert all(event["generation"]["generation_kwargs"]["requested_batch_size"] == 4 for event in events)


def test_original_graph_retry_cannot_hide_a_generation_failure(context, monkeypatch):
    args, alignment, manifest, llm = context
    original = llm.agen
    failures = []
    async def fail_once(*args, **kwargs):
        if not failures:
            failures.append(True)
            raise RuntimeError("Synthetic generation failure")
        return await original(*args, **kwargs)
    monkeypatch.setattr(llm, "agen", fail_once)
    graph = runner.new_graph(manifest)
    identity = {"phase": "evaluation", "sample_index": 0, "ordinal": 0}
    with pytest.raises(ValueError, match="Incomplete, failed"):
        asyncio.run(runner.run_batch(graph, alignment["evaluation_samples"][:1], [identity],
                                    args.run_dir / "failed_batch", manifest))
    assert not (args.run_dir / "checkpoint.json").exists()
    trace = json.loads((args.run_dir / "failed_batch/rollout_00000.json").read_text())
    assert any("Synthetic generation failure" in error for error in trace["errors"])
