"""CPU checks for changing only the GSM8K calibration cohort to baseline train40."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AgentAttrPrune.datasets import GSM8KDataset
from experiments_attrprune import baseline_aligned_gsm8k as run
from experiments_attrprune import resampled_gsm8k as core
from experiments_attrprune.common import write_json
from test_pipeline_stub import VisibleTokenScorer
from test_resampled_gsm8k import CompleteStub, setup as reference_setup


@pytest.fixture
def setup(reference_setup, tmp_path, monkeypatch):
    reference = reference_setup
    baseline_data = tmp_path / "baseline_train.jsonl"
    baseline_data.write_bytes((reference.data / "train.jsonl").read_bytes())
    audit_dir = tmp_path / "cohort_audit"
    audit_record = audit_dir / "evidence.json"
    write_json(audit_record, {"verified": ["AgentPrune", "AgentDropout", "SafeSieve"]})
    source = GSM8KDataset("train", reference.data, shuffle_seed=None)
    cohort = [source.record_to_sample(source[index]) for index in range(core.NUM_QUESTIONS)]

    def load_cohort(data, audit):
        assert data.resolve() == baseline_data.resolve()
        assert audit.resolve() == audit_dir.resolve()
        hashes = {str(path.resolve()): core.file_sha(path) for path in (baseline_data, audit_record)}
        provenance = {
            "cohort_type": run.COHORT_TYPE,
            "source_indices": list(range(core.NUM_QUESTIONS)),
            "baseline_artifact_sha256": hashes,
            "ordered_question_answer_sha256": core.digest(cohort),
        }
        return copy.deepcopy(cohort), provenance, hashes

    monkeypatch.setattr(run, "load_baseline_cohort", load_cohort)
    output = tmp_path / "baseline_aligned"

    def command(*extra):
        return ["--reference-run", str(reference.source), "--baseline-data", str(baseline_data),
                "--cohort-audit-dir", str(audit_dir), "--run-dir", str(output),
                "--stage", "prepare", "--resume", "--limit", "3", *extra]

    return SimpleNamespace(
        reference=reference, baseline_data=baseline_data, audit_dir=audit_dir,
        audit_record=audit_record, cohort=cohort, output=output,
        command=command, args=lambda *extra: run.parse_args(command(*extra)),
        tokenizer=reference.tokenizer,
    )


def test_defaults_keep_the_requested_static_sweep():
    args = run.parse_args([])
    assert run.PERCENTAGES == (80, 70, 60)
    assert args.eval_batch_size == 4 and args.limit == 1319
    assert args.stage == "all" and args.pruning_percent is None
    with pytest.raises(SystemExit):
        run.parse_args(["--stage", "eval", "--pruning-percent", "50"])
    with pytest.raises(SystemExit):
        run.parse_args(["--stage", "prepare", "--pruning-percent", "80"])


def test_source_order_maps_to_original_shuffled_dataset_without_changing_test_order(setup):
    before = {path: path.read_bytes() for path in setup.reference.source.rglob("*") if path.is_file()}
    test_before = (setup.reference.data / "test.jsonl").read_bytes()
    ctx = run.prepare_context(setup.args())
    shuffled = GSM8KDataset("train", setup.reference.data, shuffle_seed=888)
    positions = {row["question"]: index for index, row in enumerate(shuffled)}
    expected = [positions[sample["question"]] for sample in setup.cohort]
    assert expected != [0, 1], "Fixture must distinguish source order from shuffled indices"
    assert ctx.plan["selected_dataset_indices"] == expected
    assert ctx.plan["selected_source_indices"] == [0, 1]
    assert list(ctx.samples.values()) == setup.cohort
    assert [item["dataset_index"] for item in ctx.plan["rollouts"]] == expected
    assert [item["source_index"] for item in ctx.plan["rollouts"]] == [0, 1]
    for rollout, sample in zip(ctx.plan["rollouts"], setup.cohort):
        assert rollout["sample_sha256"] == core.digest(sample)
    inherited = {key: value for key, value in ctx.config.items()
                 if key not in {"fingerprint", "resampled_experiment"}}
    assert inherited == {key: value for key, value in setup.reference.config.items() if key != "fingerprint"}
    assert ctx.config["dataset"]["shuffle_seed"] == 888
    assert ctx.config["dataset"]["splits"]["test"] == setup.reference.config["dataset"]["splits"]["test"]
    assert ctx.config["seed"] == 0
    assert ctx.plan["selection_seed"] is None
    assert ctx.percentages == (80, 70, 60) and ctx.eval_batch_size == 4
    meta = ctx.config["resampled_experiment"]
    assert meta["generation_batch_size"] == 1 and meta["attribution_batch_size"] == 4
    assert meta["pruning_scope"] == "global"
    assert meta["sampling"] == "none; execute all retained edges"
    assert "skip Round1 iff both spatial and temporal masks are empty" in meta["empty_round_rule"]
    agents = [worker["agent_id"] for worker in setup.reference.config["workers"]]
    assert ctx.round_orders == [agents, agents]
    masks = ctx.order_plan["candidate_masks"]
    assert np.asarray(masks["spatial"]).sum(axis=(1, 2)).tolist() == [6, 6]
    assert np.asarray(masks["temporal"]).sum(axis=(1, 2)).tolist() == [0, 16]
    assert core.attribution_config(ctx).to_dict() == setup.reference.aggregate["attribution_configuration"]
    assert test_before == (setup.reference.data / "test.jsonl").read_bytes()
    assert before == {path: path.read_bytes() for path in before}
    assert not ctx.paths.run_dir.exists()


@pytest.mark.parametrize("mutation", ["reordered_cohort", "changed_question", "changed_answer", "duplicate_question"])
def test_changed_or_invalid_cohort_cannot_resume(setup, mutation):
    ctx = run.prepare_context(setup.args())
    core.prepare_run(ctx)
    frozen = {path: path.read_bytes() for path in ctx.paths.run_dir.rglob("*") if path.is_file()}
    if mutation == "reordered_cohort":
        setup.cohort.reverse()
    elif mutation == "changed_question":
        setup.cohort[0]["question"] += " changed"
        setup.cohort[0]["task"] = setup.cohort[0]["question"]
    elif mutation == "changed_answer":
        setup.cohort[0]["answer"] = "999"
    else:
        setup.cohort[1] = copy.deepcopy(setup.cohort[0])
    with pytest.raises(ValueError):
        run.prepare_context(setup.args())
    assert frozen == {path: path.read_bytes() for path in frozen}


@pytest.mark.parametrize("mutation", ["source", "audit_evidence", "plan", "reference"])
def test_changed_frozen_evidence_or_plan_blocks_resume(setup, mutation):
    ctx = run.prepare_context(setup.args())
    core.prepare_run(ctx)
    if mutation == "plan":
        path = ctx.paths.run_dir / "calibration_plan.json"
        plan = core.load_json(path)
        plan["selected_dataset_indices"].reverse()
        write_json(path, plan)
    else:
        path = {"source": setup.baseline_data, "audit_evidence": setup.audit_record,
                "reference": setup.reference.source / "run_config.json"}[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        core.prepare_run(ctx)
    with pytest.raises(ValueError):
        run.prepare_context(setup.args())


def test_cpu_collection_attribution_static_policies_and_resume_keep_source_identity(setup):
    baseline_before = setup.baseline_data.read_bytes()
    reference_before = {path: path.read_bytes() for path in setup.reference.source.rglob("*") if path.is_file()}
    ctx = run.prepare_context(setup.args())
    llm = CompleteStub(setup.tokenizer)
    summary = asyncio.run(core.collect_run(ctx, llm=llm))
    assert summary["num_rollouts"] == 2 and llm.calls == 18
    traces = {path: path.read_bytes() for path in ctx.paths.traces.rglob("trace.json")}
    for rollout in ctx.plan["rollouts"]:
        trace = core.load_json(core.trace_path_for(ctx, rollout))
        assert trace["sample"] == setup.cohort[rollout["source_index"]]
        assert trace["config"]["source_index"] == rollout["source_index"]
        assert trace["config"]["dataset_index"] == rollout["dataset_index"]
    core.attribute_run(ctx, scorer=VisibleTokenScorer(setup.tokenizer))
    aggregate = core.aggregate_run(ctx)
    assert aggregate["num_aggregated_samples"] == 2
    assert sum(aggregate["node_status_counts"].values()) == 18
    assert "baseline-aligned source-order" in aggregate["calibration"]["source"]
    for edge in ("spatial", "temporal"):
        expected = np.asarray(ctx.order_plan["candidate_masks"][edge]) * 2
        np.testing.assert_array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"], expected)
    policies = core.load_policies(ctx)
    assert list(policies) == [80, 70, 60]
    assert [sum(np.asarray(policy["masks"][edge]).sum() for edge in ("spatial", "temporal"))
            for policy in policies.values()] == [6, 8, 11]
    assert {path.name for path in (ctx.paths.run_dir / "policies").glob("prune_*pct.json")} == {
        "prune_80pct.json", "prune_70pct.json", "prune_60pct.json"}
    resumed_ctx = run.prepare_context(setup.args())
    second = CompleteStub(setup.tokenizer)
    asyncio.run(core.collect_run(resumed_ctx, llm=second))
    assert second.calls == 0
    assert core.validate_ready_for_evaluation(resumed_ctx) == aggregate
    assert traces == {path: path.read_bytes() for path in traces}
    assert baseline_before == setup.baseline_data.read_bytes()
    assert reference_before == {path: path.read_bytes() for path in reference_before}


def test_dry_run_never_loads_model_or_writes_output(setup, monkeypatch, capsys):
    import AgentAttrPrune.llm as llms
    monkeypatch.setattr(llms, "HFChat", lambda *args, **kwargs: pytest.fail("CPU dry-run loaded a model"))
    run.main(setup.command("--dry-run"))
    result = json.loads(capsys.readouterr().out)
    assert result["percentages"] == [80, 70, 60]
    assert result["generation_seed"] == 0
    assert result["eval_batch_size"] == 4
    assert result["decoding"] == setup.reference.config["decoding"]
    assert not setup.output.exists()
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        run.main(setup.command("--stage", "collect"))
    assert not setup.output.exists()


def test_all_runs_requested_rates_in_order_with_separate_stage_processes(setup, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    run.main(setup.command("--stage", "all"))
    assert [cmd[cmd.index("--stage") + 1] for cmd, _ in calls] == [
        "prepare", "collect", "attribute", "aggregate", "eval", "eval", "eval", "summarize"]
    assert [cmd[cmd.index("--pruning-percent") + 1] for cmd, _ in calls if "--pruning-percent" in cmd] == [
        "80", "70", "60"]
    for cmd, options in calls:
        assert options["check"] is True and "--resume" in cmd
        assert cmd[cmd.index("--baseline-data") + 1] == str(setup.baseline_data)
        assert cmd[cmd.index("--cohort-audit-dir") + 1] == str(setup.audit_dir)
        assert cmd[cmd.index("--eval-batch-size") + 1] == "4"


def test_reference_directory_cannot_be_reused(setup):
    with pytest.raises(ValueError, match="independent"):
        run.prepare_context(setup.args("--run-dir", str(setup.reference.source)))
