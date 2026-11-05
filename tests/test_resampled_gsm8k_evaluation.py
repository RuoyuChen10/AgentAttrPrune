"""GSM8K resampling preserves numeric prompts, fixed masks, and test selection."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AgentAttrPrune.attribution.edge import build_pruned_masks
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.graph.fixed_order_skip_empty import SkipEmptyFinalRoundGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.prompt import GSM8KPromptSet
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import resampled_gsm8k_evaluation as evaluation
from experiments_attrprune.common import (
    ExperimentPaths, benchmark_data_identity, build_run_config, load_json,
    reviewer_agent_ids_from_config, worker_specs_from_config, write_json,
)
from test_pipeline_stub import CharacterTokenizer


class CompleteStub:
    model_name = "stub"

    def __init__(self, tokenizer, *, fail_after=None, truncate=False, answer="12.5"):
        self.tokenizer = tokenizer
        self.calls = 0
        self.fail_after = fail_after
        self.truncate = truncate
        self.answer = answer
        self.messages = []

    async def agen(self, messages, *, max_tokens=2048, temperature=0, num_comps=1):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated inference failure")
        self.calls += 1
        self.messages.append(copy.deepcopy(messages))
        output = f"Compute half of twenty-five.\nThe answer is {self.answer}"
        rendered = self.tokenizer.apply_chat_template(messages)
        ids = self.tokenizer(rendered)["input_ids"]
        observer = generation_observer.get()
        if observer:
            observer({
                "model_name": "stub", "messages": messages, "rendered_prompt": rendered,
                "input_ids": [ids], "attention_mask": [[1] * len(ids)],
                "generated_ids": [self.tokenizer(output)["input_ids"]], "output_text": [output],
                "finish_reasons": ["length" if self.truncate else "eos"],
                "hit_max_new_tokens": [self.truncate],
                "generation_kwargs": {"max_tokens": max_tokens, "temperature": temperature, "num_completions": num_comps},
            })
        return output


class BatchStub(CompleteStub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []
        self.active = self.max_active = 0

    @asynccontextmanager
    async def async_batching(self, size):
        self.batches.append(size)
        yield

    async def agen(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await asyncio.sleep(.002)
            return await super().agen(*args, **kwargs)
        finally:
            self.active -= 1


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "test"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps({
            "question": f"Problem {index}: what is half of 25?",
            "answer": "Divide twenty-five by two. #### 12.500",
        }) + "\n" for index in range(6)))
    aliases = {"math_solver": "Ab12", "mathematical_analyst": "Cd34",
               "programming_expert": "Ef56", "inspector": "Gh78"}
    config = build_run_config(benchmark="gsm8k", model_name="stub", num_workers=4,
        num_rounds=2, temperature=0, max_new_tokens=2048, seed=0,
        calibration_split="train", evaluation_split="test",
        dataset_identity=benchmark_data_identity("gsm8k",data,splits=("train","test")),
        reviewer_agent_ids=aliases)
    config["resampled_experiment"] = {"calibration_seed": 1, "num_questions": 40}
    config["fingerprint"] = evaluation.config_fingerprint(config)
    agents = [worker["agent_id"] for worker in config["workers"]]
    paths = ExperimentPaths(tmp_path / "run")
    write_json(paths.run_config, config)
    ctx = SimpleNamespace(paths=paths,config=config,calibration_seed=1,
        round_orders=[agents[:],agents[:]],
        source_hashes={str(paths.run_config):evaluation.file_sha(paths.run_config)})
    aggregate = {
        "agents": agents, "agent_roles": [worker["role"] for worker in config["workers"]],
        "agent_types": [worker["agent_type"] for worker in config["workers"]],
        "num_rounds": 2, "matrix_orientation": "[round][source][target]",
        "run_fingerprint": config["fingerprint"], "num_aggregated_samples": 40,
        "aggregation_scope": "all", "metrics": {"token_ratio": {}},
    }
    for edge in ("spatial", "temporal"):
        counts = np.zeros((2, 4, 4))
        means = np.zeros_like(counts)
        if edge == "spatial":
            for r in (0, 1):
                counts[r] = np.triu(np.full((4, 4), 40), 1)
                means[r] = (0.7 if r == 0 else 0.2) + np.arange(16).reshape(4, 4) / 200
        else:
            counts[1] = 40
            means[1] = .05 + np.arange(16).reshape(4, 4) / 1000
        means[counts == 0] = 0
        aggregate["metrics"]["token_ratio"][edge] = {
            "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
            "conditional_mean": means.tolist(), "candidate_count": counts.tolist(),
        }
    policies = {rate: build_pruned_masks(aggregate, metric="token_ratio", statistic="conditional_mean",
        threshold=None, pruning_rate=rate / 100, unobserved="keep", pruning_scope="global")
        for rate in (25, 50, 60, 70, 80, 90)}
    from experiments_attrprune import resampled_gsm8k as runner
    def load_policies(context):
        for path, expected in context.source_hashes.items():
            if evaluation.file_sha(Path(path)) != expected:
                raise ValueError("Frozen source changed")
        return copy.deepcopy(policies)
    monkeypatch.setattr(runner, "load_policies", load_policies)
    return SimpleNamespace(ctx=ctx,policies=policies,tokenizer=CharacterTokenizer(),data=data)


def output(setup, rate=80):
    return setup.ctx.paths.evaluation / f"static_token_ratio_conditional_mean_global_prune_{rate}pct_test"


def run(setup, llm=None, rate=80, **kwargs):
    options = {"pruning_percent": rate, "eval_batch_size": 4, "limit": 5}
    options.update(kwargs)
    return asyncio.run(evaluation.evaluate_policy(setup.ctx, llm=llm, **options))


@pytest.mark.parametrize("rate,retained,expected_calls", [(25, 21, 9), (50, 14, 9), (60, 11, 9),
                                                       (70, 8, 9), (80, 6, 5), (90, 3, 5)])
def test_all_six_rates_retain_original_budget_and_execute_static_graph(setup, rate, retained, expected_calls):
    llm = BatchStub(setup.tokenizer)
    summary = run(setup, llm, rate=rate)
    assert summary["total"] == summary["correct"] == 5
    assert summary["llm_calls"] == llm.calls == expected_calls * 5
    assert summary["retained_edges"] == retained and summary["candidate_edges"] == 28
    assert summary["calibration_seed"] == 1
    assert llm.batches == [4] and llm.max_active == 4
    rows = evaluation._read_jsonl(output(setup, rate) / "predictions.jsonl")
    for row in rows:
        trace = load_json(row["trace_json"])
        assert len(trace["events"]) == expected_calls
        assert trace["config"]["fixed_order_masks"] == setup.policies[rate]["masks"]
        assert trace["config"]["seed"] == 0
        assert trace["config"]["round_orders"] == setup.ctx.round_orders
        reviewer_round = 0 if expected_calls == 5 else 1
        assert [item["round"] for item in trace["events"][-1]["incoming"]] == [reviewer_round] * 4
        assert [item["agent_id"] for item in trace["events"][-1]["incoming"]] == setup.ctx.round_orders[0]
    assert evaluation.validate_completed_policy(setup.ctx, pruning_percent=rate, limit=5) == summary


@pytest.mark.parametrize("rate", [None, 50, 70])
def test_canonical_dense_and_pruned_graph_match_original_numeric_prompts_outputs_sources(setup, rate):
    old_llm, new_llm = CompleteStub(setup.tokenizer), CompleteStub(setup.tokenizer)
    kwargs={"reviewer_agent_ids":reviewer_agent_ids_from_config(setup.ctx.config),
            "prompt_set":GSM8KPromptSet,"max_new_tokens":2048}
    if rate is not None: kwargs["policy"] = setup.policies[rate]
    specs=worker_specs_from_config(setup.ctx.config)
    old = MultiAgentGraph(old_llm, specs, **kwargs)
    new = SkipEmptyFinalRoundGraph(new_llm, specs, round_orders=setup.ctx.round_orders, **kwargs)
    old_trace, new_trace = GenerationTraceRecorder(), GenerationTraceRecorder()
    before = old.run("What is half of 25?", recorder=old_trace)
    after = new.run("What is half of 25?", recorder=new_trace)
    assert old_llm.messages == new_llm.messages
    assert old_trace.events == new_trace.events
    assert before.worker_outputs == after.worker_outputs
    assert before.reviewer_output == after.reviewer_output
    assert before.parsed_answer == after.parsed_answer == "12.5"
    assert all("Decision:" not in message[0]["content"] for message in new_llm.messages)


@pytest.mark.parametrize("edge", ["spatial", "temporal"])
def test_any_remaining_round_one_edge_prevents_skip(setup, edge):
    setup.policies[80]["masks"][edge][1][0][1] = 1
    summary = run(setup, BatchStub(setup.tokenizer), limit=1)
    assert summary["llm_calls"] == 9
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
    broken = CompleteStub(setup.tokenizer, fail_after=5)
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup, broken, eval_batch_size=1, limit=3)
    assert len(evaluation._read_jsonl(output(setup) / "predictions.jsonl")) == 1
    with pytest.raises(FileNotFoundError, match="not complete"):
        evaluation.validate_completed_policy(setup.ctx, pruning_percent=80, eval_batch_size=1, limit=3)
    good = CompleteStub(setup.tokenizer)
    result = run(setup, good, eval_batch_size=1, limit=3)
    assert good.calls == 10 and result["failed_attempts"] == 1
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
    assert first.calls == 5
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", original)
    fresh = CompleteStub(setup.tokenizer)
    assert run(setup, fresh, eval_batch_size=1, limit=2)["total"] == 2
    assert fresh.calls == 5


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
