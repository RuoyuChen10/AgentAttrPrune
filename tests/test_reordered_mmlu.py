"""CPU integration checks for frozen-order transfer and recalibration."""
from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments_attrprune import reordered_mmlu as run
from experiments_attrprune.common import benchmark_data_identity, build_run_config, write_json
from test_pipeline_stub import CharacterTokenizer, VisibleTokenScorer
from test_random_order_mmlu import CompleteStub, FailingRescueScorer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(run, "NUM_QUESTIONS", 2)
    source = tmp_path / "source"
    data = tmp_path / "data"
    for split in ("dev", "val"):
        folder = data / split
        folder.mkdir(parents=True)
        (folder / f"test_{split}.csv").write_text("\n".join(
            f"Question {i},Correct,Wrong,Wrong,Wrong,A" for i in range(4)) + "\n")
    config = build_run_config(model_name="stub", num_workers=5, num_rounds=2,
        temperature=0, max_new_tokens=512, seed=0,
        dataset_identity=benchmark_data_identity("mmlu", data, splits=("dev", "val")),
        reviewer_agent_ids={"knowledgeable_expert":"Ab12", "critic":"Cd34", "mathematician":"Ef56",
                            "psychologist":"Gh78", "historian":"Ij90"})
    config["random_order_experiment"] = {"reference": {"run_dir": str(tmp_path / "reference")}}
    config["fingerprint"] = run.config_fingerprint(config)
    write_json(source / "run_config.json", config)
    records = []
    for index in range(2):
        sample = {"task": f"Question {index}. Option A is correct. Options B C D are wrong.", "answer": "A"}
        path = source / "calibration/traces" / f"sample_{index:05d}_repeat_00/trace.json"
        write_json(path, {"config": {**config, "sample_index": index}, "sample": sample})
        records.append({"sample_index": index, "trace_json": str(path), "sha256": run.file_sha(path)})
    write_json(source / "calibration/selection_manifest.json", {"samples": records})
    aggregate = {"agents": [w["agent_id"] for w in config["workers"]],
                 "agent_roles": [w["role"] for w in config["workers"]],
                 "agent_types": [w["agent_type"] for w in config["workers"]],
                 "num_rounds": 2, "matrix_orientation": "[round][source][target]",
                 "metrics": {}, "run_fingerprint": config["fingerprint"],
                 "num_aggregated_samples": 40, "num_correct_samples": 30,
                 "aggregation_scope": "all", "attribution_configuration": {
                     "attribution_mode": "correctness-aware", "aggregation_scope": "all"}}
    for metric in ("token_ratio", "word_ratio", "binary"):
        aggregate["metrics"][metric] = {}
        for edge in ("spatial", "temporal"):
            counts = np.full((2,5,5), 20 if edge == "spatial" else 40, dtype=float)
            if edge == "spatial":
                for matrix in counts:
                    np.fill_diagonal(matrix, 0)
            else:
                counts[0] = 0
            means = (np.arange(50).reshape(2,5,5) + 1) / 70
            means[counts == 0] = 0
            aggregate["metrics"][metric][edge] = {
                "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
                "conditional_mean": means.tolist(), "candidate_count": counts.tolist(),
                "sample_mean": (means * counts / 40).tolist()}
    write_json(source / "attribution/aggregate_edge_matrices.json", aggregate)
    paths = [source / "run_config.json", source / "calibration/selection_manifest.json",
             source / "attribution/aggregate_edge_matrices.json"]
    hashes = {str(path): run.file_sha(path) for path in paths}
    monkeypatch.setattr(run, "_source", lambda path: (copy.deepcopy(config), copy.deepcopy(aggregate), dict(hashes)))
    def args(branch="recalibrate", *extra):
        return run.parse_args(["--branch", branch, "--source-run", str(source), "--limit", "3",
                               "--stage", "prepare", "--resume", *extra])
    return SimpleNamespace(source=source, config=config, aggregate=aggregate, args=args,
                           tokenizer=CharacterTokenizer(), monkeypatch=monkeypatch)


def test_direct_prepare_preserves_denominators_and_source_without_new_traces(setup):
    before = {p: p.read_bytes() for p in setup.source.rglob("*") if p.is_file()}
    ctx = run.prepare_context(setup.args("direct"))
    assert not ctx.paths.run_dir.exists()
    run.prepare_run(ctx)
    aggregate = run.validate_ready_for_evaluation(ctx)
    spatial = aggregate["metrics"]["token_ratio"]["spatial"]
    allowed = np.asarray(ctx.order_plan["candidate_masks"]["spatial"], dtype=bool)
    original = setup.aggregate["metrics"]["token_ratio"]["spatial"]
    for key in original:
        actual = np.asarray(spatial[key])
        assert np.array_equal(actual[allowed], np.asarray(original[key])[allowed])
        assert not np.any(actual[~allowed])
    assert np.all(np.asarray(spatial["opportunity_count"])[allowed] == 20)
    assert aggregate["metrics"]["token_ratio"]["temporal"] == setup.aggregate["metrics"]["token_ratio"]["temporal"]
    assert not ctx.paths.traces.exists()
    assert not (ctx.paths.attribution / "manifest.json").exists()
    assert run.load_json(ctx.paths.attribution / "projection_manifest.json")["new_attributions"] == 0
    policies = run.load_policies(ctx)
    assert list(policies) == [80,60,50]
    assert [sum(np.asarray(p["masks"][edge]).sum() for edge in ("spatial", "temporal"))
            for p in policies.values()] == [9,18,23]
    assert before == {p: p.read_bytes() for p in before}
    frozen = {p: p.read_bytes() for p in ctx.paths.run_dir.rglob("*") if p.is_file()}
    run.prepare_run(run.prepare_context(setup.args("direct")))
    assert frozen == {p: p.read_bytes() for p in frozen}


def test_branches_share_order_questions_and_freeze_parameters(setup):
    direct = run.prepare_context(setup.args("direct"))
    recalibrate = run.prepare_context(setup.args())
    assert direct.order_plan == recalibrate.order_plan
    assert direct.plan == recalibrate.plan and direct.samples == recalibrate.samples
    assert direct.config["fingerprint"] != recalibrate.config["fingerprint"]
    run.prepare_run(recalibrate)
    for changes in (("--eval-batch-size", "1"), ("--limit", "2"), ("--attr-batch-size", "2"),
                    ("--percentages", "50", "60", "80")):
        with pytest.raises(ValueError, match="configuration or plan"):
            run.prepare_context(setup.args("recalibrate", *changes))


def test_recalibration_collect_attribute_aggregate_end_to_end_and_resume(setup):
    ctx = run.prepare_context(setup.args())
    llm = CompleteStub(setup.tokenizer)
    result = asyncio.run(run.collect_run(ctx, llm=llm))
    assert result["num_rollouts"] == 2 and llm.calls == 22
    traces = {p: p.read_bytes() for p in ctx.paths.traces.rglob("trace.json")}
    for rollout in ctx.plan["rollouts"]:
        trace = run.load_json(run.trace_path_for(ctx, rollout))
        assert trace["config"]["round_orders"] == ctx.order_plan["round_orders"]
    run.attribute_run(ctx, scorer=VisibleTokenScorer(setup.tokenizer))
    aggregate = run.aggregate_run(ctx)
    assert aggregate["num_aggregated_samples"] == 2
    assert sum(aggregate["node_status_counts"].values()) == 22
    for edge in ("spatial", "temporal"):
        expected = np.asarray(ctx.order_plan["candidate_masks"][edge]) * 2
        assert np.array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"], expected)
    assert run.validate_ready_for_evaluation(ctx) == aggregate
    assert list(run.load_policies(ctx)) == [80,60,50]
    fresh = CompleteStub(setup.tokenizer)
    asyncio.run(run.collect_run(ctx, llm=fresh))
    assert fresh.calls == 0
    run.attribute_run(ctx, scorer=VisibleTokenScorer(setup.tokenizer))
    assert run.aggregate_run(ctx) == aggregate
    assert traces == {p: p.read_bytes() for p in traces}


def test_failed_rescues_count_as_observed_zero_without_changing_traces(setup):
    ctx = run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer, answer="B")))
    before = {p: p.read_bytes() for p in ctx.paths.traces.rglob("trace.json")}
    run.attribute_run(ctx, scorer=FailingRescueScorer(setup.tokenizer))
    aggregate = run.aggregate_run(ctx)
    assert aggregate["num_aggregated_samples"] == 2 and aggregate["num_correct_samples"] == 0
    assert aggregate["node_status_counts"]["rescue_failed"] > 0
    for edge in ("spatial", "temporal"):
        assert not np.asarray(aggregate["metrics"]["token_ratio"][edge]["sum"]).any()
        assert np.array_equal(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"],
                              np.asarray(ctx.order_plan["candidate_masks"][edge]) * 2)
    assert before == {p: p.read_bytes() for p in before}


def test_collection_failure_and_orphan_recovery_only_runs_missing_question(setup):
    ctx = run.prepare_context(setup.args())
    with pytest.raises(RuntimeError, match="failed after"):
        asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer, fail_after=11)))
    path = ctx.paths.calibration / "collection_state.json"
    state = run.load_json(path)
    assert len(state["completed"]) == 1 and not state["complete"]
    original = Path(state["completed"][0]["trace_json"])
    before = original.read_bytes()
    state["completed"] = []
    write_json(path, state)
    llm = CompleteStub(setup.tokenizer)
    asyncio.run(run.collect_run(ctx, llm=llm))
    assert llm.calls == 11 and original.read_bytes() == before
    assert run.load_json(path)["failures"] == {}


def test_truncation_blocks_formal_collection(setup):
    ctx = run.prepare_context(setup.args())
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer, truncate=True)))
    assert not list(ctx.paths.traces.glob("sample_*/trace.json"))
    assert not (ctx.paths.calibration / "selection_manifest.json").exists()
    assert list((ctx.paths.calibration / "diagnostics").rglob("failed_trace.json"))


@pytest.mark.parametrize("target", ["order_plan", "trace", "duplicate_selection", "missing_trace", "foreign_trace"])
def test_changed_plan_or_trace_identity_blocks_resume(setup, target):
    ctx = run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer)))
    path = next(ctx.paths.traces.glob("sample_*/trace.json"))
    if target == "order_plan":
        plan_path = ctx.paths.run_dir / "order_plan.json"
        value = run.load_json(plan_path)
        value["round_orders"][0].reverse()
        write_json(plan_path, value)
        with pytest.raises(ValueError, match="plan"):
            run.prepare_context(setup.args())
        return
    if target == "trace":
        path.write_text(path.read_text() + "\n")
    elif target == "duplicate_selection":
        selection_path = ctx.paths.calibration / "selection_manifest.json"
        value = run.load_json(selection_path)
        value["samples"][-1] = value["samples"][0]
        write_json(selection_path, value)
    elif target == "missing_trace":
        path.unlink()
    else:
        write_json(ctx.paths.traces / "sample_foreign/trace.json", run.load_json(path))
    llm = CompleteStub(setup.tokenizer)
    with pytest.raises((ValueError, FileNotFoundError)):
        asyncio.run(run.collect_run(ctx, llm=llm))
    assert llm.calls == 0


@pytest.mark.parametrize("target", ["missing_node", "incomplete_index", "changed_matrix", "duplicate_manifest", "failed_manifest"])
def test_incomplete_attribution_blocks_publication_and_policies(setup, target):
    ctx = run.prepare_context(setup.args())
    asyncio.run(run.collect_run(ctx, llm=CompleteStub(setup.tokenizer)))
    run.attribute_run(ctx, scorer=VisibleTokenScorer(setup.tokenizer))
    manifest_path = ctx.paths.attribution / "manifest.json"
    manifest = run.load_json(manifest_path)
    entry = manifest["samples"][0]
    if target == "missing_node":
        index = run.load_json(Path(entry["node_attribution_index"]))
        Path(index["events"][0]["json"]).unlink()
    elif target == "incomplete_index":
        path = Path(entry["node_attribution_index"])
        value = run.load_json(path)
        value["complete"] = False
        write_json(path, value)
    elif target == "changed_matrix":
        path = Path(entry["sample_matrix"])
        value = run.load_json(path)
        value["matrices"]["token_ratio"]["temporal"][1][0][0] = 0.987
        write_json(path, value)
    elif target == "duplicate_manifest":
        manifest["samples"].append(manifest["samples"][0])
        write_json(manifest_path, manifest)
    else:
        manifest["failures"] = [{"error": "OOM"}]
        write_json(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        run.aggregate_run(ctx)
    assert run.load_json(manifest_path)["aggregation_complete"] is False
    assert not (ctx.paths.run_dir / "policies").exists()


def test_source_and_policy_changes_are_rejected(setup):
    ctx = run.prepare_context(setup.args("direct"))
    run.prepare_run(ctx)
    path = ctx.paths.run_dir / "policies/prune_80pct.json"
    value = run.load_json(path)
    value["extra"] = True
    write_json(path, value)
    with pytest.raises(ValueError, match="policy changed"):
        run.load_policies(ctx)
    (setup.source / "run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="source artifact changed"):
        run.validate_ready_for_evaluation(ctx)


def test_cpu_dry_run_and_gpu_guard_do_not_write_or_load_models(setup, monkeypatch, capsys):
    import AgentAttrPrune.llm as llms
    monkeypatch.setattr(llms, "HFChat", lambda *a, **kw: pytest.fail("CPU check loaded GPU model"))
    command = ["--branch", "direct", "--source-run", str(setup.source), "--limit", "3"]
    run.main([*command, "--dry-run"])
    value = json.loads(capsys.readouterr().out)
    assert value["percentages"] == [80,60,50] and value["eval_batch_size"] == 4
    assert not Path(value["run_dir"]).exists()
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        run.main(command)
    assert not Path(value["run_dir"]).exists()


def test_all_uses_separate_processes_in_requested_order(setup, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    calls = []
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    run.main(["--branch", "recalibrate", "--source-run", str(setup.source), "--limit", "3", "--resume"])
    stages = [command[command.index("--stage") + 1] for command, _ in calls]
    assert stages == ["prepare", "collect", "attribute", "aggregate", "visualize", "eval", "eval", "eval", "summarize"]
    assert [command[command.index("--pruning-percent") + 1] for command, _ in calls if "--pruning-percent" in command] == ["80","60","50"]
    assert all(options["check"] and "--resume" in command for command, options in calls)


def test_cli_and_output_isolation(setup):
    args = run.parse_args(["--branch", "direct"])
    assert args.percentages == [80,60,50] and args.limit == 1531 and args.eval_batch_size == 4
    for extra in (["--percentages", "80", "80"], ["--percentages", "30"],
                  ["--eval-batch-size", "0"], ["--stage", "collect"], ["--pruning-percent", "80"]):
        with pytest.raises(SystemExit):
            run.parse_args(["--branch", "direct", *extra])
    with pytest.raises(ValueError, match="independent"):
        run.prepare_context(setup.args("direct", "--run-dir", str(setup.source / "nested")))
    foreign = setup.source.parent / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="Nonempty"):
        run.prepare_context(setup.args("direct", "--run-dir", str(foreign)))


def test_real_evaluator_summary_rejects_tampered_accuracy_and_compares_branches(setup, monkeypatch):
    from experiments_attrprune.reordered_evaluation import evaluate_policy

    class BatchedStub(CompleteStub):
        @asynccontextmanager
        async def async_batching(self, size):
            assert size == 4
            yield self

    direct = run.prepare_context(setup.args("direct"))
    run.prepare_run(direct)
    for percent in direct.percentages:
        asyncio.run(evaluate_policy(direct, pruning_percent=percent, limit=3,
                    eval_batch_size=4, llm=BatchedStub(setup.tokenizer)))
    summary = run.summarize_run(direct)
    assert summary["complete"] and summary["comparison_status"] != "complete"
    path = direct.paths.evaluation / "global_prune_80pct_val/evaluation_summary.json"
    original = run.load_json(path)
    altered = copy.deepcopy(original)
    altered["correct"] = 0
    altered["accuracy"] = 0.0
    write_json(path, altered)
    with pytest.raises(ValueError, match="validated trace checkpoints"):
        run.summarize_run(direct)
    write_json(path, original)
    recalibrate = run.prepare_context(setup.args())
    asyncio.run(run.collect_run(recalibrate, llm=CompleteStub(setup.tokenizer)))
    run.attribute_run(recalibrate, scorer=VisibleTokenScorer(setup.tokenizer))
    run.aggregate_run(recalibrate)
    for percent in recalibrate.percentages:
        asyncio.run(evaluate_policy(recalibrate, pruning_percent=percent, limit=3,
                    eval_batch_size=4, llm=BatchedStub(setup.tokenizer)))
    before_other = {p: p.read_bytes() for p in direct.paths.run_dir.rglob("*") if p.is_file()}
    combined = run.summarize_run(recalibrate)
    assert combined["comparison_status"] == "complete"
    comparison = run.load_json(Path(combined["comparison_json"]))
    assert len(comparison["results"]) == 6
    assert [row["pruning_percent"] for row in comparison["results"]] == [80,80,60,60,50,50]
    assert {row["branch"] for row in comparison["results"]} == {"direct", "recalibrate"}
    assert before_other == {p: p.read_bytes() for p in before_other}
    # Changed counterpart summaries must never produce a successful comparison.
    write_json(path, altered)
    assert run.summarize_run(recalibrate)["comparison_status"] != "complete"
    assert run.load_json(Path(combined["comparison_json"]))["complete"] is False
