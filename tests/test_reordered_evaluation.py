"""CPU integration coverage for fixed-order, batched evaluation checkpoints."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AgentAttrPrune.pruning.reordered import make_order_plan, build_reordered_policy
from experiments_attrprune import reordered_evaluation as evaluation
from experiments_attrprune.common import ExperimentPaths, benchmark_data_identity, build_run_config, load_json, write_json
from test_pipeline_stub import CharacterTokenizer
from test_random_order_mmlu import CompleteStub


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
    for split in ("dev", "val"):
        folder = data / split
        folder.mkdir(parents=True)
        (folder / f"test_{split}.csv").write_text("\n".join(
            f"Question {index},Correct,Wrong,Wrong,Wrong,A" for index in range(6)) + "\n")
    config = build_run_config(model_name="stub", num_workers=5, num_rounds=2,
        temperature=0, max_new_tokens=512, seed=0,
        dataset_identity=benchmark_data_identity("mmlu", data, splits=("dev", "val")),
        reviewer_agent_ids={"knowledgeable_expert": "Ab12", "critic": "Cd34", "mathematician": "Ef56",
                            "psychologist": "Gh78", "historian": "Ij90"})
    agents = [worker["agent_id"] for worker in config["workers"]]
    aggregate = {"agents": agents,
                 "agent_roles": [worker["role"] for worker in config["workers"]],
                 "agent_types": [worker["agent_type"] for worker in config["workers"]],
                 "num_rounds": 2, "metrics": {"token_ratio": {}},
                 "matrix_orientation": "[round][source][target]", "num_aggregated_samples": 40,
                 "aggregation_scope": "all", "run_fingerprint": config["fingerprint"]}
    for edge in ("spatial", "temporal"):
        counts = np.full((2, 5, 5), 20 if edge == "spatial" else 40, dtype=float)
        if edge == "spatial":
            for matrix in counts:
                np.fill_diagonal(matrix, 0)
        else:
            counts[0] = 0
        means = (np.arange(50).reshape(2, 5, 5) + 1) / 70
        means[counts == 0] = 0
        aggregate["metrics"]["token_ratio"][edge] = {
            "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
            "conditional_mean": means.tolist(), "candidate_count": counts.tolist()}
    order_plan = make_order_plan(aggregate)
    config["reordered_experiment"] = {"branch": "direct", "order_plan_sha256": evaluation.digest(order_plan)}
    config["fingerprint"] = evaluation.config_fingerprint(config)
    paths = ExperimentPaths(tmp_path / "run")
    write_json(paths.run_config, config)
    hashes = {str(paths.run_config): evaluation.file_sha(paths.run_config)}
    policies = {percent: build_reordered_policy(aggregate, order_plan, percent) for percent in (80, 60, 50)}
    ctx = SimpleNamespace(paths=paths, config=config, order_plan=order_plan, source_config=config,
                          source_hashes=hashes, branch="direct", resume=True, percentages=(80, 60, 50))
    from experiments_attrprune import reordered_mmlu as runner
    def validate(context):
        for path, expected in context.source_hashes.items():
            if evaluation.file_sha(Path(path)) != expected:
                raise ValueError("Frozen source changed")
        return aggregate
    monkeypatch.setattr(runner, "validate_ready_for_evaluation", validate)
    def load_policies(context):
        validate(context)
        return copy.deepcopy(policies)
    monkeypatch.setattr(runner, "load_policies", load_policies)
    return SimpleNamespace(ctx=ctx, policies=policies, tokenizer=CharacterTokenizer(), data=data,
                           output=paths.evaluation / "global_prune_50pct_val")


def run(setup, llm=None, **kwargs):
    options = {"pruning_percent": 50, "eval_batch_size": 4, "limit": 5}
    options.update(kwargs)
    return asyncio.run(evaluation.evaluate_policy(setup.ctx, llm=llm, **options))


def test_batch_four_uses_fixed_edges_and_nonoverlapping_elapsed(setup):
    llm = BatchStub(setup.tokenizer)
    summary = run(setup, llm)
    assert summary["complete"] and summary["total"] == summary["correct"] == 5
    assert summary["llm_calls"] == llm.calls == 55
    assert llm.batches == [4] and llm.max_active == 4
    assert summary["edge_counts"]["worker_candidate"] == 45
    assert summary["edge_counts"]["worker_active"] == 23
    assert summary["elapsed_seconds"] < summary["sample_elapsed_seconds_sum"]
    records = evaluation._read_jsonl(setup.output / "predictions.jsonl")
    assert len(records) == 5
    for record in records:
        trace = load_json(record["trace_json"])
        assert trace["config"]["round_orders"] == setup.ctx.order_plan["round_orders"]
        assert trace["config"]["fixed_order_masks"] == setup.policies[50]["masks"]
        assert len(trace["events"]) == 11
        assert record["edge_counts"] == summary["edge_counts"]
    assert summary["elapsed_seconds"] == sum(r["evaluation_elapsed_seconds"] for r in records)


def test_completed_resume_does_not_load_model_or_replay(setup, monkeypatch):
    summary = run(setup, BatchStub(setup.tokenizer))
    originals = {p: p.read_bytes() for p in setup.output.rglob("*") if p.is_file()}
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Completed run loaded model"))
    assert run(setup) == summary
    assert {p: p.read_bytes() for p in originals} == originals
    with pytest.raises(FileExistsError, match="resume"):
        run(setup, resume=False)


def test_runtime_failure_keeps_successes_and_resumes_only_missing(setup):
    broken = CompleteStub(setup.tokenizer, fail_after=11)
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup, broken, eval_batch_size=1, limit=3)
    records = evaluation._read_jsonl(setup.output / "predictions.jsonl")
    assert len(records) == 1 and records[0]["correct"]
    assert not (setup.output / "evaluation_summary.json").exists()
    assert load_json(setup.output / "progress.json")["complete"] is False
    llm = CompleteStub(setup.tokenizer)
    summary = run(setup, llm, eval_batch_size=1, limit=3)
    assert summary["accuracy"] == 1 and llm.calls == 22
    assert summary["failed_attempts"] == 1


def test_failed_batched_cohort_cancels_cleanly_without_false_predictions(setup):
    broken = BatchStub(setup.tokenizer, fail_after=8)
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup, broken)
    assert broken.active == 0
    assert not (setup.output / "evaluation_summary.json").exists()
    assert evaluation._read_jsonl(setup.output / "predictions.jsonl") == []
    assert all(a["status"] != "completed" for a in evaluation._read_jsonl(setup.output / "attempts.jsonl"))
    assert run(setup, BatchStub(setup.tokenizer))["total"] == 5


def test_orphan_trace_recovers_without_replaying_success(setup, monkeypatch):
    original = evaluation._write_jsonl_atomic
    stopped = False
    def interrupt(path, records):
        nonlocal stopped
        if path.name == "predictions.jsonl" and records and not stopped:
            stopped = True
            raise RuntimeError("checkpoint interrupted")
        return original(path, records)
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", interrupt)
    first = CompleteStub(setup.tokenizer)
    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        run(setup, first, eval_batch_size=1, limit=3)
    assert first.calls == 11
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", original)
    fresh = CompleteStub(setup.tokenizer)
    summary = run(setup, fresh, eval_batch_size=1, limit=3)
    assert fresh.calls == 22 and summary["total"] == 3
    assert any(a["status"] == "recovered" for a in evaluation._read_jsonl(setup.output / "attempts.jsonl"))


@pytest.mark.parametrize("field", ["trace", "mask", "duplicate", "missing", "policy", "summary"])
def test_tampering_rejected_before_generation(setup, field):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    trace_path = setup.output / "traces/sample_00000/trace.json"
    if field in {"trace", "mask"}:
        trace = load_json(trace_path)
        if field == "trace":
            trace["events"][-1]["incoming"][0]["source_output"] = "changed"
        else:
            trace["config"]["fixed_order_masks"]["spatial"][0][0][0] = 1
        write_json(trace_path, trace)
    elif field == "duplicate":
        records = evaluation._read_jsonl(setup.output / "predictions.jsonl")
        evaluation._write_jsonl_atomic(setup.output / "predictions.jsonl", records + records)
    elif field == "missing":
        trace_path.unlink()
    elif field == "policy":
        value = load_json(setup.output / "resolved_policy.json")
        value["extra"] = True
        write_json(setup.output / "resolved_policy.json", value)
    else:
        value = load_json(setup.output / "evaluation_summary.json")
        value["correct"] = 0
        write_json(setup.output / "evaluation_summary.json", value)
    llm = BatchStub(setup.tokenizer)
    with pytest.raises(ValueError):
        run(setup, llm, limit=1)
    assert llm.calls == 0


def test_config_batch_limit_order_and_source_changes_rejected(setup):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    for kwargs in ({"eval_batch_size": 1, "limit": 1}, {"limit": 2}):
        with pytest.raises(ValueError, match="configuration changed"):
            run(setup, **kwargs)
    setup.ctx.order_plan["round_orders"][0].reverse()
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=1)
    setup.ctx.order_plan["round_orders"][0].reverse()
    setup.ctx.paths.run_config.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        run(setup, limit=1)


def test_truncated_validation_answers_are_scored_and_reported(setup):
    summary = run(setup, BatchStub(setup.tokenizer, truncate=True), limit=2)
    assert summary["total"] == summary["correct"] == 2
    assert summary["truncated_samples"] == 2 and summary["truncated_events"] == 22


def test_validation_selection_is_exact_and_frozen(setup):
    state = evaluation.prepare_evaluation(setup.ctx, pruning_percent=50, eval_batch_size=4, limit=5)
    assert state.selection["num_samples"] == 5 and state.selection["dataset_size"] == 6
    assert [s["sample_index"] for s in state.selection["samples"]] == list(range(5))
    assert not state.output.exists()
    with pytest.raises(ValueError, match="validation contains"):
        evaluation.prepare_evaluation(setup.ctx, pruning_percent=50, eval_batch_size=4, limit=7)
    (setup.data / "val/test_val.csv").write_text("Changed,Yes,No,No,No,A\n")
    with pytest.raises(ValueError, match="dataset fingerprint"):
        evaluation.prepare_evaluation(setup.ctx, pruning_percent=50, eval_batch_size=4, limit=5)
