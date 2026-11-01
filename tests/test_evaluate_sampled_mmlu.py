"""CPU-only integration tests for the independent sampled-graph evaluator."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AgentAttrPrune.pruning.sampled import sample_graph
from experiments_attrprune import evaluate_sampled_mmlu as evaluation
from experiments_attrprune.common import benchmark_data_identity, build_run_config, load_json, write_json
from test_pipeline_stub import CharacterTokenizer
from test_random_order_mmlu import CompleteStub


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    source = tmp_path / "source"
    data = tmp_path / "data"
    for split in ("dev", "val"):
        folder = data / split
        folder.mkdir(parents=True)
        (folder / f"test_{split}.csv").write_text("\n".join(
            f"Question {i},Correct,Wrong,Wrong,Wrong,A" for i in range(6)) + "\n")
    config = build_run_config(model_name="stub", num_workers=5, num_rounds=2,
        temperature=0, max_new_tokens=512, seed=0,
        dataset_identity=benchmark_data_identity("mmlu", data, splits=("dev", "val")),
        reviewer_agent_ids={"knowledgeable_expert":"Ab12", "critic":"Cd34", "mathematician":"Ef56",
                            "psychologist":"Gh78", "historian":"Ij90"})
    config["random_order_experiment"] = {"reference": {"run_dir": str(tmp_path / "reference")}}
    config["fingerprint"] = evaluation.config_fingerprint(config)
    write_json(source / "run_config.json", config)
    aggregate = {"agents": [worker["agent_id"] for worker in config["workers"]],
                 "num_rounds": 2, "metrics": {"token_ratio": {}},
                 "matrix_orientation": "[round][source][target]",
                 "run_fingerprint": config["fingerprint"], "num_aggregated_samples": 40,
                 "aggregation_scope": "all"}
    for edge in ("spatial", "temporal"):
        counts = np.full((2,5,5), 20 if edge == "spatial" else 40, dtype=float)
        if edge == "spatial":
            for matrix in counts:
                np.fill_diagonal(matrix, 0)
        else:
            counts[0] = 0
        means = (np.arange(50).reshape(2,5,5) + 1) / 70
        means[counts == 0] = 0
        aggregate["metrics"]["token_ratio"][edge] = {
            "sum": (means * counts).tolist(), "opportunity_count": counts.tolist(),
            "conditional_mean": means.tolist(), "candidate_count": counts.tolist()}
    hashes = {str(source / "run_config.json"): evaluation.source_runner.file_sha(source / "run_config.json")}
    real_source = evaluation._source
    monkeypatch.setattr(evaluation, "_source", lambda path: (copy.deepcopy(config), copy.deepcopy(aggregate), dict(hashes)))
    def args(*extra):
        return evaluation.parse_args(["--run-dir",str(source),"--pruning-scope","global",
                                      "--pruning-percent","50","--limit","3",*extra])
    return SimpleNamespace(source=source, data=data, config=config, aggregate=aggregate,
        hashes=hashes, args=args, tokenizer=CharacterTokenizer(), real_source=real_source)


def run(args, stub=None):
    return asyncio.run(evaluation.evaluate(args, llm=stub))


def test_cpu_dry_run_prepare_and_configuration_freeze(setup, monkeypatch):
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("CPU preparation loaded a model"))
    overview = run(setup.args("--dry-run"))
    output = Path(overview["output_dir"])
    assert overview["num_samples"] == 3
    assert not output.exists()
    result = run(setup.args("--prepare-only"))
    assert result["prepared"]
    policy = load_json(output / "sampling_policy.json")
    assert policy["round_policy"] == "per_round"
    assert policy["pruning"]["candidate_parameter_count"] == 65
    assert policy["pruning"]["pruned_parameter_count"] == 32
    assert not (output / "predictions.jsonl").exists()
    frozen = {p: p.read_bytes() for p in output.glob("*.json")}
    run(setup.args("--prepare-only", "--resume"))
    assert {p: p.read_bytes() for p in frozen} == frozen
    with pytest.raises(ValueError, match="configuration changed"):
        run(setup.args("--prepare-only", "--seed", "1"))


def _shift_inspection_probabilities(policy, *, direction, steps=1):
    """Emulate float32 vector-math differences without changing sampling inputs."""
    for edge in ("spatial", "temporal"):
        probabilities = np.asarray(policy["parameters"][edge]["probabilities"], dtype=np.float32)
        for _ in range(steps):
            probabilities = np.nextafter(probabilities, np.float32(direction))
        policy["parameters"][edge]["probabilities"] = probabilities.tolist()
        policy["parameters"][edge]["effective_probabilities"] = (
            probabilities * np.asarray(policy["masks"][edge])
        ).tolist()
    return policy


def _emulate_policy_rebuild(monkeypatch, transform):
    from AgentAttrPrune.pruning import sampled
    original = sampled.prepare_sampling_policy

    def rebuild(*args, **kwargs):
        return transform(original(*args, **kwargs))

    monkeypatch.setattr(sampled, "prepare_sampling_policy", rebuild)


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_cpu_probability_rounding_preserves_prepared_policy_and_fingerprint(setup, monkeypatch, direction):
    run(setup.args("--prepare-only"))
    original = evaluation.prepare_context(setup.args())
    frozen = {path: path.read_bytes() for path in original.output_dir.glob("*.json")}
    _emulate_policy_rebuild(monkeypatch, lambda policy: _shift_inspection_probabilities(
        policy, direction=direction))

    resumed = evaluation.prepare_context(setup.args("--resume"))
    assert resumed.policy == original.policy
    assert resumed.config == original.config
    assert resumed.config["fingerprint"] == original.config["fingerprint"]
    assert run(setup.args("--prepare-only", "--resume"))["prepared"]
    assert {path: path.read_bytes() for path in frozen} == frozen
    with pytest.raises(ValueError, match="configuration changed: evaluation_config.json"):
        evaluation.prepare_context(setup.args("--resume", "--seed", "1"))


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_probability_drift_beyond_one_float32_ulp_is_rejected(setup, monkeypatch, direction):
    run(setup.args("--prepare-only"))
    _emulate_policy_rebuild(monkeypatch, lambda policy: _shift_inspection_probabilities(
        policy, direction=direction, steps=2))
    with pytest.raises(ValueError, match="sampling_policy.json.*beyond float32 rounding"):
        evaluation.prepare_context(setup.args("--resume"))


@pytest.mark.parametrize("field", ["logits", "masks"])
def test_policy_reuse_does_not_allow_operational_parameter_changes(setup, monkeypatch, field):
    run(setup.args("--prepare-only"))

    def change(policy):
        if field == "logits":
            policy["parameters"]["spatial"]["logits"][1][0][1] += 0.1
        else:
            policy["masks"]["spatial"][1][0][1] = 1 - policy["masks"]["spatial"][1][0][1]
        return policy

    _emulate_policy_rebuild(monkeypatch, change)
    with pytest.raises(ValueError, match="configuration changed: sampling_policy.json"):
        evaluation.prepare_context(setup.args("--resume"))


def test_saved_probability_tampering_is_rejected_even_within_one_ulp(setup):
    run(setup.args("--prepare-only"))
    context = evaluation.prepare_context(setup.args())
    policy_path = context.output_dir / "sampling_policy.json"
    tampered = _shift_inspection_probabilities(load_json(policy_path), direction=np.inf)
    write_json(policy_path, tampered)
    with pytest.raises(ValueError, match="sampling_policy.json.*hash mismatch"):
        evaluation.prepare_context(setup.args("--resume"))


def test_completed_evaluation_resumes_without_replay_across_cpu_probability_rounding(setup, monkeypatch):
    summary = run(setup.args(), CompleteStub(setup.tokenizer))
    output = Path(summary["predictions_jsonl"]).parent
    frozen = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    _emulate_policy_rebuild(monkeypatch, lambda policy: _shift_inspection_probabilities(
        policy, direction=np.inf))
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Completed resume loaded model"))

    assert run(setup.args("--resume")) == summary
    assert {path: path.read_bytes() for path in frozen} == frozen


def test_sampled_evaluation_trace_counts_summary_and_no_recompute_resume(setup, monkeypatch):
    llm = CompleteStub(setup.tokenizer)
    summary = run(setup.args(), llm)
    assert summary["complete"] and summary["total"] == summary["correct"] == 3
    assert llm.calls == 33 and summary["llm_calls"] == 33
    assert summary["generated_tokens"] > 0
    assert summary["edge_counts"]["worker_candidate_per_sample"] == 65
    output = Path(summary["predictions_jsonl"]).parent
    records = evaluation._read_jsonl(output / "predictions.jsonl")
    assert summary["edge_counts"]["worker_active_total"] == sum(r["sampled_edge_counts"]["worker_active"] for r in records)
    assert summary["edge_counts"]["worker_active_maximum"] >= summary["edge_counts"]["worker_active_minimum"]
    policy = load_json(output / "sampling_policy.json")
    for record in records:
        assert record["sampling_seed"] == evaluation.sampling_seed(0, record["sample_index"])
        assert record["realized_graph"] == sample_graph(policy, seed=record["sampling_seed"])
        trace = load_json(record["trace_json"])
        assert trace["config"]["prompt"] == setup.config["prompt"]
        assert len(trace["events"]) == 11
    original_traces = {Path(r["trace_json"]): Path(r["trace_json"]).read_bytes() for r in records}
    import AgentAttrPrune.llm as llm_module
    monkeypatch.setattr(llm_module, "HFChat", lambda *a, **k: pytest.fail("Completed resume loaded model"))
    again = run(setup.args("--resume"))
    assert again == summary
    assert {p: p.read_bytes() for p in original_traces} == original_traces
    assert len(evaluation._read_jsonl(output / "attempts.jsonl")) == 6
    with pytest.raises(FileExistsError, match="resume"):
        run(setup.args(), CompleteStub(setup.tokenizer))


def test_runtime_failure_is_not_incorrect_and_resume_only_missing(setup):
    broken = CompleteStub(setup.tokenizer, fail_after=11)
    with pytest.raises(RuntimeError, match="failed after"):
        run(setup.args(), broken)
    output = evaluation.prepare_context(setup.args()).output_dir
    assert not (output / "evaluation_summary.json").exists()
    records = evaluation._read_jsonl(output / "predictions.jsonl")
    assert len(records) == 1 and records[0]["correct"]
    assert load_json(output / "progress.json")["complete"] is False
    attempts = evaluation._read_jsonl(output / "attempts.jsonl")
    assert attempts[-1]["status"] == "failed"
    assert Path(attempts[-1]["partial_trace_json"]).exists()
    fresh = CompleteStub(setup.tokenizer)
    summary = run(setup.args("--resume"), fresh)
    assert fresh.calls == 22 and summary["accuracy"] == 1
    assert summary["failed_attempts"] == 1


def test_completed_orphan_trace_recovers_without_generation(setup, monkeypatch):
    original = evaluation._write_jsonl_atomic
    failed_once = False
    def interrupt(path, records):
        nonlocal failed_once
        if path.name == "predictions.jsonl" and records and not failed_once:
            failed_once = True
            raise RuntimeError("checkpoint interruption")
        return original(path, records)
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", interrupt)
    llm = CompleteStub(setup.tokenizer)
    with pytest.raises(RuntimeError, match="checkpoint interruption"):
        run(setup.args(), llm)
    assert llm.calls == 11
    monkeypatch.setattr(evaluation, "_write_jsonl_atomic", original)
    resumed = CompleteStub(setup.tokenizer)
    summary = run(setup.args("--resume"), resumed)
    assert resumed.calls == 22 and summary["total"] == 3


def test_truncation_is_recorded_and_follows_existing_evaluation_semantics(setup):
    summary = run(setup.args("--limit","1"), CompleteStub(setup.tokenizer, truncate=True))
    assert summary["total"] == 1 and summary["correct"] == 1
    assert summary["truncated_samples"] == 1 and summary["truncated_events"] == 11
    record = evaluation._read_jsonl(Path(summary["predictions_jsonl"]))[0]
    trace = load_json(record["trace_json"])
    assert trace["generation_complete"] is False
    assert len(record["truncated_event_ids"]) == 11


@pytest.mark.parametrize("target", ["trace", "policy", "selection", "prediction"])
def test_checkpoint_tampering_rejected_before_model_call(setup, target):
    summary = run(setup.args("--limit","1"), CompleteStub(setup.tokenizer))
    output = Path(summary["predictions_jsonl"]).parent
    if target == "trace":
        path = output / "traces/sample_00000/trace.json"
        value = load_json(path)
        value["events"][-1]["output"] = "B"
        write_json(path, value)
    elif target in {"policy", "selection"}:
        path = output / ("sampling_policy.json" if target == "policy" else "selection_manifest.json")
        value = load_json(path)
        value["extra"] = True
        write_json(path, value)
    else:
        path = output / "predictions.jsonl"
        records = evaluation._read_jsonl(path)
        records[0]["trace_sha256"] = "changed"
        evaluation._write_jsonl_atomic(path, records)
    llm = CompleteStub(setup.tokenizer)
    with pytest.raises(ValueError):
        run(setup.args("--limit","1","--resume"), llm)
    assert llm.calls == 0


def test_rate_and_scope_share_question_seeds_but_keep_separate_policies(setup):
    first = evaluation.prepare_context(setup.args())
    second = evaluation.prepare_context(setup.args("--pruning-percent","60"))
    uniform = evaluation.prepare_context(setup.args("--pruning-scope","uniform"))
    assert first.selection == second.selection == uniform.selection
    assert first.output_dir != second.output_dir != uniform.output_dir
    assert first.policy != second.policy != uniform.policy
    assert first.policy["pruning"]["pruned_parameter_count"] == 32
    assert second.policy["pruning"]["pruned_parameter_count"] == 39


def test_foreign_destination_dataset_change_and_source_change_rejected(setup):
    foreign = setup.source.parent / "foreign"
    foreign.mkdir()
    (foreign / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        run(setup.args("--output-dir",str(foreign),"--prepare-only"))
    assert (foreign / "user.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="independent"):
        run(setup.args("--output-dir",str(setup.source / "attribution"),"--prepare-only"))
    ctx = evaluation.prepare_context(setup.args())
    (setup.source / "run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="source artifact changed"):
        evaluation._prepare(ctx)
    assert not ctx.output_dir.exists()
    (setup.data / "val/test_val.csv").write_text("Changed,Yes,No,No,No,A\n")
    with pytest.raises(ValueError, match="dataset fingerprint"):
        evaluation.prepare_context(setup.args())


def test_prepare_metadata_interruption_is_recoverable(setup, monkeypatch):
    original = evaluation.source_runner.freeze_json
    def interrupt(path, value):
        if path.name == "sampling_policy.json":
            raise RuntimeError("stopped after ownership marker")
        original(path, value)
    monkeypatch.setattr(evaluation.source_runner, "freeze_json", interrupt)
    with pytest.raises(RuntimeError, match="ownership"):
        run(setup.args("--prepare-only"))
    monkeypatch.setattr(evaluation.source_runner, "freeze_json", original)
    assert run(setup.args("--prepare-only","--resume"))["prepared"]


def test_source_verifier_checks_complete_aggregate_hash_before_attribution(setup, monkeypatch):
    config = copy.deepcopy(setup.config)
    config["random_order_experiment"].update(num_questions=40,repeats_per_question=1,
        attribution_mode="correctness-aware",order_seed=0,attribution_batch_size=4)
    write_json(setup.source / "run_config.json", config)
    paths = evaluation.source_runner.ExperimentPaths(setup.source)
    write_json(setup.source / "rollout_plan.json", {})
    write_json(paths.calibration / "selection_manifest.json", {})
    write_json(paths.attribution / "aggregate_edge_matrices.json", setup.aggregate)
    write_json(paths.attribution / "manifest.json", {"complete":True,"aggregation_complete":True,
        "aggregate_sha256":"stale","failures":[]})
    monkeypatch.setattr(evaluation.source_runner, "prepare_context", lambda args: SimpleNamespace(paths=paths))
    monkeypatch.setattr(evaluation.source_runner, "verified_sample_matrices", lambda *args: pytest.fail("stale source reached verification"))
    with pytest.raises(ValueError, match="incomplete, stale or failed"):
        setup.real_source(setup.source)


def test_cli_defaults_and_invalid_arguments():
    args = evaluation.parse_args(["--pruning-scope","global","--pruning-percent","50"])
    assert args.limit == 1531 and args.round_policy == "per_round" and args.split == "val"
    for extra in (["--limit","0"], ["--round-policy","shared"], ["--dry-run","--prepare-only"]):
        with pytest.raises(SystemExit):
            evaluation.parse_args(["--pruning-scope","global","--pruning-percent","50",*extra])
