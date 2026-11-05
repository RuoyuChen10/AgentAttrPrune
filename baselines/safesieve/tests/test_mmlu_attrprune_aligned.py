"""Offline checks for the aligned SafeSieve experiment; no model weights/API calls."""
import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_mmlu_attrprune_aligned as run
from experiments import attrprune_mmlu_adapter as adapter
from AgentAttrPrune.prompt.mmlu import WORKER_ROLES


@pytest.fixture
def alignment(monkeypatch):
    workers = [{"agent_id": agent, "role": role, "node_index": i}
               for i, (agent, role) in enumerate(WORKER_ROLES[:5])]
    def samples(prefix, count):
        return [{"question": f"{prefix} {i}", "options": ["one", "two", "three", "four"],
                 "answer": "ABCD"[i % 4], "subject": "test",
                 "task": f"{prefix} {i}\nOption A: one\nOption B: two\nOption C: three\nOption D: four\n"}
                for i in range(count)]
    result = {"workers": workers, "reviewer_agent_ids": {w["agent_id"]: f"Ab{i:02}" for i, w in enumerate(workers)},
              "calibration_samples": samples("dev", 40), "evaluation_samples": samples("val", 1531),
              "reference_run_config": {"model_name": "/model/Qwen3-8B", "decoding": {
                  "temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}},
              "reference_run": "/original/reference", "source_hashes": {}, "fingerprint": "frozen-test"}
    monkeypatch.setattr(adapter, "_alignment", None)
    monkeypatch.setattr(run, "load_alignment", lambda _: copy.deepcopy(result))
    monkeypatch.setattr(run, "verify_sources", lambda _: None)
    return result


@pytest.fixture
def ctx(tmp_path, alignment):
    log = tmp_path / "old.log"
    log.write_text("MMLU roles: " + repr([w["role"] for w in alignment["workers"]]) + "\n"
        ">>> Using LLM expert scoring for similarity calculation (model: gpt-5.6-sol)\n"
        ">>> Initial Similarity Matrix:\n" + str(np.array([
            [0,.44,.68,.45,.65], [.44,0,.65,.58,.65], [.68,.65,0,.58,.52],
            [.45,.58,.58,0,.7], [.65,.65,.52,.7,0]])))
    args = run.parse_args(["--run-dir", str(tmp_path / "run"), "--similarity-log", str(log)])
    result = run.prepare_context(args)
    result["args"] = args
    return result


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def __call__(self, text, return_tensors):
        ids = torch.tensor([[3] * (len(text) % 11 + 2)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    def batch_decode(self, tokens, skip_special_tokens):
        return ["Reasoning: preserved raw output. Decision: A" for _ in tokens]


def make_llm():
    """Exercise the real HFChat queue, padding and model.generate batch path."""
    llm = object.__new__(run.HFChat)
    llm.model_name = "local:/model/Qwen3-8B"
    llm.model_path = "/model/Qwen3-8B"
    llm.logits_processors = []
    llm.tokenizer = FakeTokenizer()
    llm._generation_lock = threading.Lock()
    llm.render_messages = lambda messages: "\n".join(m["content"] for m in messages)
    llm.calls = []
    def generate(input_ids, attention_mask, **kwargs):
        llm.calls.append({"batch_size": len(input_ids), "kwargs": kwargs})
        suffix = input_ids.new_tensor([[9, 2]] * len(input_ids))
        return SimpleNamespace(sequences=torch.cat([input_ids, suffix], dim=1))
    llm.model = SimpleNamespace(generation_config=SimpleNamespace(num_beams=1, min_length=0),
                                device="cpu", generate=generate)
    def single(messages, *, max_tokens, temperature, num_comps):
        assert temperature == 0 and num_comps == 1
        response, event = llm._generate_batch([SimpleNamespace(messages=messages, max_tokens=max_tokens)])[0]
        observer = run.generation_observer.get()
        if observer:
            observer(event)
        return response
    llm._generate = single
    return llm


def factory(llm):
    def get(name):
        assert run.HFChat.resolve_model_name(name) == "/model/Qwen3-8B"
        return llm
    return get


@pytest.mark.parametrize("role_index", range(5))
@pytest.mark.parametrize("with_neighbors", [False, True])
def test_exact_worker_prompts_for_all_roles_and_neighbor_orders(alignment, role_index, with_neighbors):
    adapter.configure(alignment, llm_factory=factory(make_llm()))
    worker = alignment["workers"][role_index]
    node = adapter.AttrPruneMMLUWorker(id=worker["agent_id"], role=worker["role"], llm_name="local:/model/Qwen3-8B")
    chosen = alignment["workers"][::-1] if with_neighbors else []
    info = {w["agent_id"]: {"role": w["role"], "output": f"Complete raw answer of {w['agent_id']}"} for w in chosen}
    spatial = [{"agent_id": key, **item} for key, item in info.items()]
    temporal = list(reversed(spatial))
    actual = node.build_messages({"task": "Frozen task\n"}, info, info)
    expected = adapter.MMLUPromptSet.worker_messages(role=worker["role"], task="Frozen task\n",
                                                     spatial_incoming=spatial, temporal_incoming=temporal)
    assert actual == expected


def test_reviewer_preserves_frozen_aliases_and_canonical_order_after_node_removal(alignment):
    adapter.configure(alignment, llm_factory=factory(make_llm()))
    node = adapter.AttrPruneMMLUReviewer(llm_name="local:/model/Qwen3-8B")
    remaining = [alignment["workers"][4], alignment["workers"][1]]
    info = {w["agent_id"]: {"role": w["role"], "output": w["agent_id"] + " full text"} for w in remaining}
    outputs = [{"agent_id": w["agent_id"], **info[w["agent_id"]],
                "reviewer_agent_id": alignment["reviewer_agent_ids"][w["agent_id"]]} for w in remaining[::-1]]
    assert node.build_messages({"task": "Question"}, info, {}) == adapter.MMLUPromptSet.reviewer_messages(
        task="Question", worker_outputs=outputs)
    with pytest.raises(ValueError, match="final-round"):
        node.build_messages({"task": "Question"}, info, info)


def test_identity_and_role_mismatches_are_rejected(alignment):
    adapter.configure(alignment, llm_factory=factory(make_llm()))
    w = alignment["workers"][0]
    with pytest.raises(ValueError, match="identity/role"):
        adapter.AttrPruneMMLUWorker(id="newid", role=w["role"])
    with pytest.raises(ValueError, match="role"):
        adapter._items({w["agent_id"]: {"role": "wrong", "output": "A"}})
    with pytest.raises(ValueError, match="canonical"):
        adapter._items({"newid": {"role": w["role"], "output": "A"}})


def test_budget_is_285_draws_of_40_unique_with_18_logical_batches(ctx):
    cfg = ctx["config"]
    assert cfg["training"]["draw_plan"] == [i % 40 for i in range(285)]
    assert cfg["training"]["logical_batch_size"] == 16
    assert cfg["training"]["logical_batch_count"] == 18
    assert cfg["training"]["possible_pruning_batches"] == [10, 15]
    assert cfg["llm_microbatch_size"] == cfg["eval_batch_size"] == 4
    assert cfg["decoding"] == {"temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}
    assert cfg["backend_name"] == "local:/model/Qwen3-8B"
    assert cfg["similarity"]["precision"]["bitwise_original_internal_matrix_verified"] is False
    assert not ctx["directory"].exists()


def test_dry_run_has_no_output_or_model_calls(ctx, monkeypatch, capsys):
    monkeypatch.setattr(run, "build_graph", lambda *a, **k: pytest.fail("Dry run loaded a model"))
    run.main(["--run-dir", str(ctx["directory"]), "--similarity-log", str(ctx["args"].similarity_log), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not ctx["directory"].exists()


def test_frozen_source_and_cohort_changes_rejected(ctx, monkeypatch):
    run.prepare_run(ctx)
    args = copy.copy(ctx["args"])
    args.resume = True
    assert run.prepare_context(args)["config"] == ctx["config"]
    changed = copy.deepcopy(ctx["alignment"])
    changed["fingerprint"] = "different-cohort"
    monkeypatch.setattr(run, "load_alignment", lambda _: changed)
    with pytest.raises(ValueError, match="configuration changed"):
        run.prepare_context(args)
    ctx["args"].similarity_log.write_text("altered original matrix")
    with pytest.raises(ValueError, match="similarity log changed"):
        run.verify_context(ctx)


def test_dry_run_refuses_foreign_output_without_modifying_it(ctx):
    ctx["directory"].mkdir()
    target = ctx["directory"] / "old_results.json"
    target.write_text("old")
    with pytest.raises(FileExistsError):
        run.prepare_context(ctx["args"])
    assert target.read_text() == "old"
    assert list(ctx["directory"].iterdir()) == [target]


def test_real_graph_calls_real_hf_microbatch_path_and_keeps_prompt_identities(ctx):
    llm = make_llm()
    graph = run.build_graph(ctx, llm_factory=factory(llm))
    async def work():
        async with llm.async_batching(4):
            return await run.run_batch(graph, ctx["alignment"]["calibration_samples"][:4])
    results = asyncio.run(work())
    assert len(results) == 4
    assert max(call["batch_size"] for call in llm.calls) == 4
    assert all(call["batch_size"] <= 4 for call in llm.calls)
    assert all(not call["kwargs"]["do_sample"] and call["kwargs"]["max_new_tokens"] == 2048 for call in llm.calls)
    expected_ids = set(ctx["config"]["worker_ids"]) | {"dense_reviewer"}
    for result in results:
        assert len(result["events"]) == 11
        assert {e["agent_id"] for e in result["events"]} == expected_ids
        assert result["events"][-1]["round"] == 2
        assert result["response"] == "Reasoning: preserved raw output. Decision: A"
        assert all(e["generation_kwargs"]["temperature"] == 0 for e in result["events"])
    # Inference runs only on copies; training policy/base connections stay intact.
    assert not graph.spatial_adj_matrix.any()


def fake_result(graph, sample):
    index = int(sample["question"].split()[-1])
    active = graph.spatial_masks.detach().cpu().numpy().reshape(graph.num_nodes, graph.num_nodes)
    adj = np.zeros_like(active)
    for i in range(graph.num_nodes):
        for j in range(i + 1, graph.num_nodes):
            adj[i, j] = int(bool(active[i, j] and (i + j + index) % 3))
    response = "unparseable" if index % 7 == 0 else "Decision: " + (sample["answer"] if index % 3 else "D")
    return {"response": response, "spatial_adjacency": adj.tolist(), "events": [], "elapsed_seconds": 0.0}




def test_policy_and_statistics_restore_keep_dtypes_after_node_removal(ctx):
    graph = run.build_graph(ctx, llm_factory=factory(make_llm()))
    matrix = np.ones((5,5)) - np.eye(5)
    matrix[4, :] = matrix[:, 4] = 0
    graph.spatial_masks = torch.nn.Parameter(torch.tensor(matrix.flatten()), requires_grad=False)
    removed = graph.check_and_remove_isolated_nodes(2)
    assert removed == [ctx["config"]["worker_ids"][4]]
    policy = run.policy_state(graph)
    restored = run.build_graph(ctx, policy, llm_factory=factory(make_llm()))
    assert run.policy_state(restored) == policy
    manager = run.EdgeMatrixManager(5, np.array(ctx["config"]["similarity"]["matrix"]), alpha=.5)
    manager.remove_nodes_and_update_matrix([4], [0,1,2,3])
    state = run.manager_state(manager)
    assert state["dtypes"]["similarity_matrix"] == "float32"
    assert run.manager_state(run.restore_manager(state)) == state


def test_training_resume_keeps_completed_logical_batches_and_final_policy(ctx, monkeypatch):
    calls = []
    async def batches(graph, samples):
        calls.append([s["question"] for s in samples])
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        return [fake_result(graph, sample) for sample in samples]
    monkeypatch.setattr(run, "run_batch", batches)
    llm = make_llm()
    with pytest.raises(RuntimeError, match="interruption"):
        asyncio.run(run.train(ctx, llm_factory=factory(llm)))
    checkpoint = run._load_checkpoint(ctx["directory"] / "training_checkpoint.json", ctx, "training")
    assert len(checkpoint["records"]) == 32
    first_sha = checkpoint["records"][0]["trace_sha256"]
    completed = asyncio.run(run.train(ctx, resume=True, llm_factory=factory(llm)))
    assert completed["complete"] and len(completed["records"]) == 285
    assert len(completed["batches"]) == 18
    assert calls[3][0] == "dev 32"
    assert completed["records"][0]["trace_sha256"] == first_sha
    assert len({r["sample_sha256"] for r in completed["records"]}) == 40
    assert [m["batch_index"] + 1 for m in completed["batches"] if m["pruned"]] == [10, 15]
    assert not list((ctx["directory"] / "training/traces").glob("*.previous_*.json"))
    before = len(calls)
    asyncio.run(run.train(ctx, resume=True, llm_factory=factory(llm)))
    assert len(calls) == before
    # Verify final policy/statistics match an uninterrupted run with the same draws.
    fresh = {**ctx, "directory": ctx["directory"].with_name("fresh")}
    other = asyncio.run(run.train(fresh, llm_factory=factory(llm)))
    assert completed["policy"] == other["policy"]
    assert completed["manager"] == other["manager"]


def test_modified_trace_missing_record_and_incomplete_evaluation_block_publication(ctx):
    sample = ctx["alignment"]["evaluation_samples"][0]
    result = {"response": "Decision: A", "events": [], "elapsed_seconds": 1.0}
    saved = run._save_trace(ctx, "evaluation", 0, sample, result)
    state = {"phase": "evaluation", "run_fingerprint": ctx["config"]["fingerprint"], "complete": False,
             "policy_sha256": "policy", "records": [{"sample_index": 0, "sample_sha256": run.digest(sample),
                 "response": result["response"], "correct": True, **saved}]}
    path = ctx["directory"] / "evaluation_checkpoint.json"
    run._save_checkpoint(path, state)
    with pytest.raises(ValueError, match="incomplete"):
        run.summarize(ctx)
    state["records"][0]["sample_index"] = 1
    with pytest.raises(ValueError, match="identity"):
        run.validate_records(ctx, state, [sample])
    state["records"][0]["sample_index"] = 0
    Path(saved["trace_json"]).write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        run.validate_records(ctx, state, [sample])
    assert not (ctx["directory"] / "summary.json").exists()


def test_missing_generation_audit_refuses_completed_answer(ctx, monkeypatch):
    graph = run.build_graph(ctx, llm_factory=factory(make_llm()))
    async def no_events(self, inputs, num_rounds):
        return ["Decision: A"], torch.tensor(0.)
    monkeypatch.setattr(run.Graph, "arun", no_events)
    with pytest.raises(ValueError, match="missing a scheduled node"):
        asyncio.run(run.run_question(graph, ctx["alignment"]["calibration_samples"][0]))


def test_evaluation_resume_keeps_completed_answers_and_uses_four_question_batches(ctx, monkeypatch):
    # Keep all 285 training draws; shorten only this isolated synthetic evaluation.
    ctx["alignment"]["evaluation_samples"] = ctx["alignment"]["evaluation_samples"][:6]
    async def training_batch(graph, samples):
        return [fake_result(graph, sample) for sample in samples]
    monkeypatch.setattr(run, "run_batch", training_batch)
    llm = make_llm()
    asyncio.run(run.train(ctx, llm_factory=factory(llm)))
    calls = []
    async def evaluation_batch(graph, samples):
        calls.append([sample["question"] for sample in samples])
        if len(calls) == 2:
            raise RuntimeError("interruption")
        return [{"response": "Decision: A", "spatial_adjacency": [], "elapsed_seconds": .2,
                 "events": [{"input_ids": [[1,2,3]], "generated_ids": [[4,2]],
                             "generation_kwargs": {"effective_batch_size": len(samples)}}]} for sample in samples]
    monkeypatch.setattr(run, "run_batch", evaluation_batch)
    with pytest.raises(RuntimeError, match="interruption"):
        asyncio.run(run.evaluate(ctx, llm_factory=factory(llm)))
    state = run._load_checkpoint(ctx["directory"] / "evaluation_checkpoint.json", ctx, "evaluation")
    assert len(state["records"]) == 4 and not state["complete"]
    original_sha = state["records"][0]["trace_sha256"]
    result = asyncio.run(run.evaluate(ctx, resume=True, llm_factory=factory(llm)))
    assert calls == [["val 0", "val 1", "val 2", "val 3"], ["val 4", "val 5"], ["val 4", "val 5"]]
    assert result["complete"] and result["total"] == 6 and result["correct"] == 2
    assert result["generation_batching"]["max_effective_batch_size"] == 4
    state = run._load_checkpoint(ctx["directory"] / "evaluation_checkpoint.json", ctx, "evaluation")
    assert state["records"][0]["trace_sha256"] == original_sha
    assert asyncio.run(run.evaluate(ctx, resume=True, llm_factory=factory(llm))) == result
    (ctx["directory"] / "policy.json").write_text("{}")
    with pytest.raises(ValueError, match="policy changed"):
        run.summarize(ctx)


