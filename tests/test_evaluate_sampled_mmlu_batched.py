"""CPU checks for fresh 80% sampled-DAG evaluation with four concurrent questions."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from AgentAttrPrune.pruning.sampled import sample_graph
from experiments_attrprune import evaluate_sampled_mmlu as legacy
from experiments_attrprune import evaluate_sampled_mmlu_batched as evaluation
from experiments_attrprune.common import load_json, write_json
from test_evaluate_sampled_mmlu import setup as sampled_setup
from test_random_order_mmlu import CompleteStub


class BatchStub(CompleteStub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []
        self.active = 0
        self.max_active = 0

    @asynccontextmanager
    async def async_batching(self, size):
        self.batches.append(size)
        yield

    async def agen(self, *args, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.002)
            return await super().agen(*args, **kwargs)
        finally:
            self.active -= 1


def args(setup, *extra):
    return evaluation.parse_args(["--run-dir", str(setup.source), "--pruning-scope", "global",
        "--pruning-percent", "80", "--eval-batch-size", "4", "--limit", "6", *extra])


def run(arguments, stub=None):
    return asyncio.run(evaluation.evaluate(arguments, llm=stub))


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_prepare80_retains13_of65_candidates_and_freezes_batch4(sampled_setup, monkeypatch, scope):
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("CPU stage loaded a model"))
    arguments = args(sampled_setup, "--pruning-scope", scope)
    ctx = evaluation.prepare_context(arguments)
    assert ctx.output_dir.name == "prune_80pct"
    assert ctx.config["eval_batch_size"] == ctx.config["generation_batch_size"] == 4
    assert ctx.policy["pruning"]["candidate_parameter_count"] == 65
    assert ctx.policy["pruning"]["pruned_parameter_count"] == 52
    spatial = np.asarray(ctx.policy["masks"]["spatial"])
    temporal = np.asarray(ctx.policy["masks"]["temporal"])
    assert spatial.sum() + temporal.sum() == 13
    assert temporal[0].sum() == 0
    if scope == "uniform":
        assert spatial.sum(axis=(1, 2)).tolist() == [4, 4]
        assert temporal.sum(axis=(1, 2)).tolist() == [0, 5]
    assert run(args(sampled_setup, "--pruning-scope", scope, "--dry-run"))["dry_run"]
    assert not ctx.output_dir.exists()
    assert run(args(sampled_setup, "--pruning-scope", scope, "--prepare-only"))["prepared"]
    frozen = snapshot(ctx.output_dir)
    run(args(sampled_setup, "--pruning-scope", scope, "--prepare-only", "--resume"))
    assert snapshot(ctx.output_dir) == frozen


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_four_question_concurrency_preserves_masks_rounds_and_resume(sampled_setup, monkeypatch, scope):
    llm = BatchStub(sampled_setup.tokenizer)
    summary = run(args(sampled_setup, "--pruning-scope", scope), llm)
    assert summary["complete"] and summary["total"] == summary["correct"] == 6
    assert summary["eval_batch_size"] == 4
    assert summary["llm_calls"] == llm.calls == 66
    assert llm.max_active == 4 and llm.active == 0 and llm.batches == [4]
    output = Path(summary["predictions_jsonl"]).parent
    records = legacy._read_jsonl(output / "predictions.jsonl")
    policy = load_json(output / "sampling_policy.json")
    for record in records:
        seed = legacy.sampling_seed(0, record["sample_index"])
        assert record["sampling_seed"] == seed
        assert record["realized_graph"] == sample_graph(policy, seed=seed)
        trace = load_json(record["trace_json"])
        assert trace["config"]["eval_batch_size"] == 4
        assert trace["config"]["evaluation_mode"] == "agentprune_sampling_batched"
        assert len(trace["events"]) == 11
        assert all(item["round"] == 1 for item in trace["events"][-1]["incoming"])
        assert trace["config"]["prompt"] == sampled_setup.config["prompt"]
    before = snapshot(output)
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Completed resume loaded a model"))
    assert run(args(sampled_setup, "--pruning-scope", scope, "--resume")) == summary
    assert snapshot(output) == before
    with pytest.raises(FileExistsError):
        run(args(sampled_setup, "--pruning-scope", scope))


def test_global80_empty_round1_still_executes_all_workers(sampled_setup):
    for edge in ("spatial", "temporal"):
        block = sampled_setup.aggregate["metrics"]["token_ratio"][edge]
        counts = np.asarray(block["opportunity_count"])
        means = np.full(counts.shape, 0.01)
        if edge == "spatial":
            means[0] = 0.9
        means[counts == 0] = 0
        block["conditional_mean"] = means.tolist()
        block["sum"] = (counts * means).tolist()
    ctx = evaluation.prepare_context(args(sampled_setup))
    assert np.asarray(ctx.policy["masks"]["spatial"])[1].sum() == 0
    assert np.asarray(ctx.policy["masks"]["temporal"])[1].sum() == 0
    summary = run(args(sampled_setup), BatchStub(sampled_setup.tokenizer))
    assert summary["llm_calls"] == 66
    for record in legacy._read_jsonl(ctx.output_dir / "predictions.jsonl"):
        trace = load_json(record["trace_json"])
        assert sum(event["round"] == 1 for event in trace["events"]) == 5
        assert len(trace["events"][-1]["incoming"]) == 5
        assert all(item["round"] == 1 for item in trace["events"][-1]["incoming"])


def test_failure_preserves_first_batch_and_resumes_only_remaining(sampled_setup):
    stub = BatchStub(sampled_setup.tokenizer, fail_after=44)
    with pytest.raises(RuntimeError):
        run(args(sampled_setup), stub)
    assert stub.active == 0
    ctx = evaluation.prepare_context(args(sampled_setup))
    records = legacy._read_jsonl(ctx.output_dir / "predictions.jsonl")
    assert len(records) == 4
    assert not (ctx.output_dir / "evaluation_summary.json").exists()
    before = {Path(r["trace_json"]): Path(r["trace_json"]).read_bytes() for r in records}
    resumed = BatchStub(sampled_setup.tokenizer)
    summary = run(args(sampled_setup, "--resume"), resumed)
    assert summary["total"] == 6 and resumed.calls == 22 and summary["failed_attempts"] >= 1
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("target", ["duplicate", "missing_trace", "changed_trace_batch", "changed_config"])
def test_incomplete_or_changed_checkpoint_cannot_resume(sampled_setup, target):
    summary = run(args(sampled_setup), BatchStub(sampled_setup.tokenizer))
    output = Path(summary["predictions_jsonl"]).parent
    records = legacy._read_jsonl(output / "predictions.jsonl")
    if target == "duplicate":
        legacy._write_jsonl_atomic(output / "predictions.jsonl", records + [records[-1]])
    elif target == "missing_trace":
        Path(records[-1]["trace_json"]).unlink()
    elif target == "changed_trace_batch":
        p = Path(records[-1]["trace_json"])
        trace = load_json(p)
        trace["config"]["eval_batch_size"] = 1
        write_json(p, trace)
    else:
        p = output / "evaluation_config.json"
        cfg = load_json(p)
        cfg["eval_batch_size"] = 1
        write_json(p, cfg)
    llm = BatchStub(sampled_setup.tokenizer)
    with pytest.raises(ValueError):
        run(args(sampled_setup, "--resume"), llm)
    assert llm.calls == 0


def test_legacy50_and60_checkpoints_remain_byte_identical(sampled_setup):
    old_paths = []
    for rate in (50, 60):
        old_args = sampled_setup.args("--pruning-percent", str(rate), "--prepare-only")
        asyncio.run(legacy.evaluate(old_args))
        old_paths.append(legacy.prepare_context(old_args).output_dir)
    frozen = {p: snapshot(p) for p in old_paths}
    run(args(sampled_setup), BatchStub(sampled_setup.tokenizer))
    assert {p: snapshot(p) for p in old_paths} == frozen
    with pytest.raises(ValueError):
        run(args(sampled_setup, "--output-dir", str(old_paths[0]), "--resume"))
