"""Independent empty-final-round reevaluation and durable checkpoint checks."""
from __future__ import annotations

import asyncio
import copy

import pytest

from experiments_attrprune import reordered_evaluation as baseline
from experiments_attrprune import reordered_skip_empty_evaluation as evaluation
from experiments_attrprune.common import load_json, write_json
from test_reordered_evaluation import BatchStub, CompleteStub, setup as base_setup


@pytest.fixture
def setup(base_setup):
    policy = base_setup.policies[80]
    n = len(policy["agents"])
    policy["masks"] = {edge: [[[0] * n for _ in range(n)] for _ in range(2)]
                       for edge in ("spatial", "temporal")}
    order = base_setup.ctx.order_plan["round_orders"][0]
    edges = [(policy["agents"].index(source), policy["agents"].index(target))
             for i, source in enumerate(order) for target in order[i + 1:]]
    for source, target in edges[:9]:
        policy["masks"]["spatial"][0][source][target] = 1
    base_setup.output = base_setup.ctx.paths.evaluation / "global_prune_80pct_val_skip_empty_round1"
    return base_setup


def run(setup, llm=None, **kwargs):
    options = {"pruning_percent": 80, "eval_batch_size": 4, "limit": 5}
    options.update(kwargs)
    return asyncio.run(evaluation.evaluate_policy(setup.ctx, llm=llm, **options))


def test_empty_round_skips_five_calls_and_reviewer_uses_round_zero(setup):
    llm = BatchStub(setup.tokenizer)
    summary = run(setup, llm)
    assert summary["complete"] and summary["total"] == summary["correct"] == 5
    assert summary["llm_calls"] == llm.calls == 30
    assert summary["edge_counts"]["worker_active"] == 9
    assert summary["edge_counts"]["worker_candidate"] == 45
    assert llm.batches == [4] and llm.max_active == 4
    contract = summary["execution_contract"]
    assert contract["executed_worker_rounds"] == [0]
    assert contract["skipped_worker_rounds"] == [1]
    assert contract["reviewer_source_round"] == 0
    for record in evaluation._read_jsonl(setup.output / "predictions.jsonl"):
        trace = load_json(record["trace_json"])
        assert len(trace["events"]) == 6
        assert [e["round"] for e in trace["events"]] == [0] * 5 + [2]
        assert [x["round"] for x in trace["events"][-1]["incoming"]] == [0] * 5
        assert [x["agent_id"] for x in trace["events"][-1]["incoming"]] == setup.policies[80]["agents"]
        assert record["execution_contract"] == trace["config"]["execution_contract"] == contract
        assert trace["config"]["fixed_order_masks"] == setup.policies[80]["masks"]
        assert trace["config"]["round_orders"] == setup.ctx.order_plan["round_orders"]


def test_old_evaluation_and_source_preserved_in_separate_directory(setup):
    old = asyncio.run(baseline.evaluate_policy(setup.ctx, pruning_percent=80,
        eval_batch_size=4, limit=1, llm=BatchStub(setup.tokenizer)))
    old_dir = setup.ctx.paths.evaluation / "global_prune_80pct_val"
    before = {p: p.read_bytes() for p in old_dir.rglob("*") if p.is_file()}
    source_before = setup.ctx.paths.run_config.read_bytes()
    new = run(setup, BatchStub(setup.tokenizer), limit=1)
    assert old["llm_calls"] == 11 and new["llm_calls"] == 6
    assert old["evaluation_fingerprint"] != new["evaluation_fingerprint"]
    assert old["policy_sha256"] == new["policy_sha256"]
    assert old["selection_sha256"] == new["selection_sha256"]
    assert old["round_orders"] == new["round_orders"]
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert setup.ctx.paths.run_config.read_bytes() == source_before
    assert new["baseline_evaluation_dir"] == str(old_dir)


def test_completed_resume_avoids_model_and_replay(setup, monkeypatch):
    first = run(setup, BatchStub(setup.tokenizer), limit=2)
    before = {p: p.read_bytes() for p in setup.output.rglob("*") if p.is_file()}
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("loaded model"))
    assert run(setup, limit=2) == first
    assert all(p.read_bytes() == contents for p, contents in before.items())
    with pytest.raises(FileExistsError, match="resume"):
        run(setup, limit=2, resume=False)


def test_failure_resumes_only_unfinished_questions(setup):
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup, CompleteStub(setup.tokenizer, fail_after=6), eval_batch_size=1, limit=3)
    assert len(evaluation._read_jsonl(setup.output / "predictions.jsonl")) == 1
    assert not (setup.output / "evaluation_summary.json").exists()
    llm = CompleteStub(setup.tokenizer)
    result = run(setup, llm, eval_batch_size=1, limit=3)
    assert llm.calls == 12 and result["total"] == 3 and result["failed_attempts"] == 1


def test_orphan_trace_recovers_without_new_calls_for_saved_question(setup, monkeypatch):
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
    result = run(setup, fresh, eval_batch_size=1, limit=2)
    assert fresh.calls == 6 and result["total"] == 2


@pytest.mark.parametrize("mutation", ["reviewer_source", "contract", "extra_node", "duplicate", "missing", "policy", "summary"])
def test_modified_or_incomplete_results_cannot_resume_as_complete(setup, mutation):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    path = setup.output / "traces/sample_00000/trace.json"
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
        rows = evaluation._read_jsonl(setup.output / "predictions.jsonl")
        evaluation._write_jsonl_atomic(setup.output / "predictions.jsonl", rows + rows)
    elif mutation == "missing":
        path.unlink()
    elif mutation == "policy":
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


@pytest.mark.parametrize("edge", ["spatial", "temporal"])
def test_one_nonempty_round_one_matrix_keeps_round_one(setup, edge):
    agents = setup.policies[80]["agents"]
    order = setup.ctx.order_plan["round_orders"][1]
    i, j = agents.index(order[0]), agents.index(order[1])
    setup.policies[80]["masks"][edge][1][i][j] = 1
    result = run(setup, BatchStub(setup.tokenizer), limit=1)
    assert result["llm_calls"] == 11
    assert result["execution_contract"]["executed_worker_rounds"] == [0, 1]
    assert result["execution_contract"]["skipped_worker_rounds"] == []
    assert result["execution_contract"]["reviewer_source_round"] == 1


def test_truncation_is_scored_and_reported_for_six_events(setup):
    result = run(setup, BatchStub(setup.tokenizer, truncate=True), limit=2)
    assert result["total"] == 2 and result["truncated_samples"] == 2
    assert result["truncated_events"] == 12


def test_dry_run_does_not_write_or_load_model(setup, monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "prepare_context", lambda args: setup.ctx)
    evaluation.main(["--branch", "direct", "--dry-run", "--limit", "5"])
    assert not setup.output.exists()
    assert '"reviewer_source_round": 0' in capsys.readouterr().out


def test_changed_order_source_or_selection_rejected(setup):
    run(setup, BatchStub(setup.tokenizer), limit=1)
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=2)
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=1, eval_batch_size=1)
    setup.ctx.order_plan["round_orders"][0].reverse()
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup, limit=1)
    setup.ctx.order_plan["round_orders"][0].reverse()
    setup.ctx.paths.run_config.write_text("{}")
    with pytest.raises(ValueError, match="source changed"):
        run(setup, limit=1)


def test_other_pruning_rates_are_rejected(setup):
    with pytest.raises(ValueError, match="restricted to 80"):
        run(setup, pruning_percent=60)
