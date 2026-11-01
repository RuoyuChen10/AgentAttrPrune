"""Calibration resampling never changes val data or fixed graph execution rules."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import numpy as np
import pytest

from AgentAttrPrune.attribution.edge import build_pruned_masks
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.graph.fixed_order_skip_empty import SkipEmptyFinalRoundGraph
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import resampled_evaluation as evaluation
from experiments_attrprune.common import load_json, write_json
from test_random_order_graph import RecordingStub, SPECS, ALIASES
from test_reordered_evaluation import BatchStub, CompleteStub, setup as original_setup


@pytest.fixture
def setup(original_setup, monkeypatch):
    setup = original_setup
    ctx = setup.ctx
    ctx.config.pop("reordered_experiment")
    ctx.config["resampled_experiment"] = {"calibration_seed": 1, "num_questions": 40}
    ctx.config["fingerprint"] = evaluation.config_fingerprint(ctx.config)
    ctx.calibration_seed = 1
    agents = [worker["agent_id"] for worker in ctx.config["workers"]]
    ctx.round_orders = [agents[:], agents[:]]
    write_json(ctx.paths.run_config, ctx.config)
    ctx.source_hashes = {str(ctx.paths.run_config): evaluation.file_sha(ctx.paths.run_config)}
    aggregate = {
        "agents": agents, "agent_roles": [worker["role"] for worker in ctx.config["workers"]],
        "agent_types": [worker["agent_type"] for worker in ctx.config["workers"]],
        "num_rounds": 2, "matrix_orientation": "[round][source][target]",
        "run_fingerprint": ctx.config["fingerprint"], "num_aggregated_samples": 40,
        "aggregation_scope": "all", "metrics": {"token_ratio": {}},
    }
    for edge in ("spatial", "temporal"):
        counts = np.zeros((2, 5, 5))
        means = np.zeros_like(counts)
        if edge == "spatial":
            for r in (0, 1):
                counts[r] = np.triu(np.full((5, 5), 40), 1)
                means[r] = (0.7 if r == 0 else 0.2) + np.arange(25).reshape(5, 5) / 200
        else:
            counts[1] = 40
            means[1] = .05 + np.arange(25).reshape(5, 5) / 1000
        means[counts == 0] = 0
        aggregate["metrics"]["token_ratio"][edge] = {
            "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
            "conditional_mean": means.tolist(), "candidate_count": counts.tolist(),
        }
    setup.policies = {rate: build_pruned_masks(aggregate, metric="token_ratio", statistic="conditional_mean",
        threshold=None, pruning_rate=rate / 100, unobserved="keep", pruning_scope="global")
        for rate in (25, 50, 60, 70, 80, 90)}
    from experiments_attrprune import resampled_mmlu as runner
    def load_policies(context):
        for path, expected in context.source_hashes.items():
            if evaluation.file_sha(Path(path)) != expected:
                raise ValueError("Frozen source changed")
        return copy.deepcopy(setup.policies)
    monkeypatch.setattr(runner, "load_policies", load_policies)
    return setup


def output(setup, rate=80):
    return setup.ctx.paths.evaluation / f"static_token_ratio_conditional_mean_global_prune_{rate}pct_val"


def run(setup, llm=None, rate=80, **kwargs):
    options = {"pruning_percent": rate, "eval_batch_size": 4, "limit": 5}
    options.update(kwargs)
    return asyncio.run(evaluation.evaluate_policy(setup.ctx, llm=llm, **options))


@pytest.mark.parametrize("rate,retained,expected_calls", [(25, 34, 11), (50, 23, 11), (60, 18, 11),
                                                       (70, 14, 11), (80, 9, 6), (90, 5, 6)])
def test_all_six_rates_retain_original_budget_and_execute_static_graph(setup, rate, retained, expected_calls):
    llm = BatchStub(setup.tokenizer)
    summary = run(setup, llm, rate=rate)
    assert summary["total"] == summary["correct"] == 5
    assert summary["llm_calls"] == llm.calls == expected_calls * 5
    assert summary["retained_edges"] == retained and summary["candidate_edges"] == 45
    assert summary["calibration_seed"] == 1
    assert llm.batches == [4] and llm.max_active == 4
    rows = evaluation._read_jsonl(output(setup, rate) / "predictions.jsonl")
    for row in rows:
        trace = load_json(row["trace_json"])
        assert len(trace["events"]) == expected_calls
        assert trace["config"]["fixed_order_masks"] == setup.policies[rate]["masks"]
        assert trace["config"]["seed"] == 0
        assert trace["config"]["round_orders"] == setup.ctx.round_orders
        reviewer_round = 0 if expected_calls == 6 else 1
        assert [item["round"] for item in trace["events"][-1]["incoming"]] == [reviewer_round] * 5
        assert [item["agent_id"] for item in trace["events"][-1]["incoming"]] == setup.ctx.round_orders[0]
    assert evaluation.validate_completed_policy(setup.ctx, pruning_percent=rate, limit=5) == summary


def test_canonical_dense_graph_matches_original_prompts_outputs_and_sources():
    old_llm, new_llm = RecordingStub(), RecordingStub()
    old = MultiAgentGraph(old_llm, SPECS, reviewer_agent_ids=ALIASES)
    new = SkipEmptyFinalRoundGraph(new_llm, SPECS, round_orders=[["a", "b", "c"]] * 2,
                                  reviewer_agent_ids=ALIASES)
    old_trace, new_trace = GenerationTraceRecorder(), GenerationTraceRecorder()
    before = old.run("Question", recorder=old_trace)
    after = new.run("Question", recorder=new_trace)
    assert old_llm.calls == new_llm.calls
    assert old_trace.events == new_trace.events
    assert before.worker_outputs == after.worker_outputs
    assert before.reviewer_output == after.reviewer_output


@pytest.mark.parametrize("edge", ["spatial", "temporal"])
def test_any_remaining_round_one_edge_prevents_skip(setup, edge):
    setup.policies[80]["masks"][edge][1][0][1] = 1
    summary = run(setup, BatchStub(setup.tokenizer), limit=1)
    assert summary["llm_calls"] == 11
    assert summary["execution_contract"]["reviewer_source_round"] == 1
    assert summary["execution_contract"]["skipped_worker_rounds"] == []


def test_readonly_validation_and_completed_resume_do_not_touch_files_or_model(setup, monkeypatch):
    first = run(setup, BatchStub(setup.tokenizer), limit=2)
    before = {p: p.read_bytes() for p in setup.ctx.paths.run_dir.rglob("*") if p.is_file()}
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Loaded model for complete run"))
    assert evaluation.validate_completed_policy(setup.ctx, pruning_percent=80, limit=2) == first
    assert {p: p.read_bytes() for p in before} == before
    assert run(setup, limit=2) == first
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError, match="resume"):
        run(setup, limit=2, resume=False)


def test_failure_and_orphan_recovery_never_repeat_finished_question(setup):
    broken = CompleteStub(setup.tokenizer, fail_after=6)
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup, broken, eval_batch_size=1, limit=3)
    assert len(evaluation._read_jsonl(output(setup) / "predictions.jsonl")) == 1
    with pytest.raises(FileNotFoundError, match="not complete"):
        evaluation.validate_completed_policy(setup.ctx, pruning_percent=80, eval_batch_size=1, limit=3)
    good = CompleteStub(setup.tokenizer)
    result = run(setup, good, eval_batch_size=1, limit=3)
    assert good.calls == 12 and result["failed_attempts"] == 1
    assert evaluation.validate_completed_policy(setup.ctx, pruning_percent=80, eval_batch_size=1, limit=3) == result


def test_interrupted_checkpoint_recovers_orphan_trace(setup, monkeypatch):
    original = evaluation._write_jsonl_atomic
    stopped = False
    def interrupted(path, rows):
        nonlocal stopped
        if path.name == "predictions.jsonl" and rows and not stopped:
            stopped = True
            raise RuntimeError("checkpoint interrupted")
        return original(path, rows)
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", interrupted)
    first = CompleteStub(setup.tokenizer)
    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        run(setup, first, eval_batch_size=1, limit=2)
    assert first.calls == 6
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", original)
    fresh = CompleteStub(setup.tokenizer)
    assert run(setup, fresh, eval_batch_size=1, limit=2)["total"] == 2
    assert fresh.calls == 6


@pytest.mark.parametrize("mutation", ["reviewer_source", "contract", "extra_node", "duplicate", "missing", "policy", "summary", "attempt", "foreign_trace", "token_usage"])
def test_incomplete_or_changed_results_are_rejected(setup, mutation):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    directory = output(setup)
    path = directory / "traces/sample_00000/trace.json"
    if mutation in {"reviewer_source", "contract", "extra_node"}:
        trace = load_json(path)
        if mutation == "reviewer_source":
            trace["events"][-1]["incoming"][0]["round"] = 1
        elif mutation == "contract":
            trace["config"]["execution_contract"]["reviewer_source_round"] = 1
        else:
            trace["events"].insert(5, copy.deepcopy(trace["events"][0]))
        write_json(path, trace)
    elif mutation == "duplicate":
        rows = evaluation._read_jsonl(directory / "predictions.jsonl")
        evaluation._write_jsonl_atomic(directory / "predictions.jsonl", rows + rows)
    elif mutation == "missing":
        path.unlink()
    elif mutation == "policy":
        value = load_json(directory / "resolved_policy.json")
        value["extra"] = True
        write_json(directory / "resolved_policy.json", value)
    elif mutation == "attempt":
        evaluation._write_jsonl_atomic(directory / "attempts.jsonl", [])
    elif mutation == "foreign_trace":
        write_json(directory / "traces/foreign/trace.json", load_json(path))
    elif mutation == "token_usage":
        trace = load_json(path)
        trace["usage"]["generated_tokens"] += 1
        trace["final"]["metrics"]["generated_tokens"] += 1
        trace["evaluation_record"]["metrics"]["generated_tokens"] += 1
        state = evaluation.prepare_evaluation(setup.ctx, pruning_percent=80, eval_batch_size=4, limit=1)
        with pytest.raises(ValueError, match="usage differs"):
            evaluation._validate_trace(state, 0, trace)
        return
    else:
        value = load_json(directory / "evaluation_summary.json")
        value["correct"] = 0
        write_json(directory / "evaluation_summary.json", value)
    with pytest.raises(ValueError):
        evaluation.validate_completed_policy(setup.ctx, pruning_percent=80, limit=1)


def test_changing_selection_batch_order_or_source_cannot_reuse_checkpoint(setup):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=2)
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=1, eval_batch_size=1)
    setup.ctx.round_orders[0].reverse()
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=1)
    setup.ctx.round_orders[0].reverse()
    setup.ctx.paths.run_config.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        run(setup, limit=1)


def test_calibration_seed_does_not_change_validation_selection_or_generation_seed(setup):
    a = evaluation.prepare_evaluation(setup.ctx, pruning_percent=80, eval_batch_size=4, limit=5)
    second = copy.deepcopy(setup.ctx)
    second.calibration_seed = 2
    second.config["resampled_experiment"]["calibration_seed"] = 2
    second.config["fingerprint"] = evaluation.config_fingerprint(second.config)
    b = evaluation.prepare_evaluation(second, pruning_percent=80, eval_batch_size=4, limit=5)
    assert a.selection == b.selection and a.samples == b.samples
    assert a.config["seed"] == b.config["seed"] == 0
    assert a.config["fingerprint"] != b.config["fingerprint"]


def test_invalid_rate_never_loads_model(setup):
    with pytest.raises(ValueError, match="Choose a frozen pruning rate"):
        run(setup, rate=30)
