"""CPU end-to-end coverage of the isolated repeated-rollout experiment."""

from __future__ import annotations

import asyncio
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from AgentAttrPrune.attribution.edge import aggregate_sample_matrices, build_sample_matrices, empty_sample_matrices
from AgentAttrPrune.datasets import evaluate_prediction
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import generation_observer
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import random_order_mmlu as experiment
from experiments_attrprune.common import build_run_config, reviewer_agent_ids_from_config, worker_specs_from_config, write_json
from test_pipeline_stub import CharacterTokenizer, VisibleTokenScorer


class CompleteStub:
    model_name = "stub"

    def __init__(self, tokenizer, *, fail_after=None, truncate=False, answer="A"):
        self.tokenizer = tokenizer
        self.calls = 0
        self.fail_after = fail_after
        self.truncate = truncate
        self.answer = answer

    async def agen(self, messages, *, max_tokens=512, temperature=0, num_comps=1):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("simulated inference failure")
        self.calls += 1
        reviewer = "top decision-maker" in messages[0]["content"]
        output = self.answer if reviewer else f"Reasoning: Evidence supports option {self.answer}.\nDecision: {self.answer}"
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


class FailingRescueScorer(VisibleTokenScorer):
    def score_teacher_forced_targets(self, *args, **kwargs):
        result = super().score_teacher_forced_targets(*args, **kwargs)
        indices = kwargs["selected_output_token_indices"]
        result["target_logprobs"] = torch.full((len(result["score"]), len(indices)), -0.1)
        result["logprob_sum"] = result["target_logprobs"].sum(dim=1)
        result["logprob_mean"] = result["target_logprobs"].mean(dim=1)
        return result

    def generate_from_rendered_chat(self, rendered, *, max_new_tokens, do_sample):
        assert do_sample is False
        text = "Reasoning: Still incorrect.\nDecision: B"
        return {"input_ids": torch.tensor(self.tokenizer(rendered)["input_ids"]),
                "generated_answer_ids": torch.tensor(self.tokenizer(text)["input_ids"] + [0]),
                "output_text": text}


@pytest.fixture
def small_context(tmp_path, monkeypatch):
    # Smaller counts exercise the same production functions; the CLI has no
    # budget override and the independent plan test checks the full 40 x 10.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(experiment, "NUM_QUESTIONS", 2)
    monkeypatch.setattr(experiment, "REPEATS", 2)
    monkeypatch.setattr(experiment, "NUM_WORKERS", 2)
    reference = tmp_path / "fixed"
    tokenizer = CharacterTokenizer()
    config = build_run_config(model_name="stub", num_workers=2, num_rounds=2,
                              temperature=0, max_new_tokens=512, seed=0,
                              calibration_split="dev", evaluation_split="val",
                              reviewer_agent_ids={"knowledgeable_expert": "Ab12", "critic": "Cd34"})
    write_json(reference / "run_config.json", config)
    selection, matrices = [], []
    for sample_index in range(2):
        sample = {"task": f"Question {sample_index}. Option A is correct. Options B C D are wrong.", "answer": "A"}
        path = reference / "calibration/traces" / f"sample_{sample_index:05d}" / "trace.json"
        recorder = GenerationTraceRecorder(path, config={**config, "sample_index": sample_index}, sample=sample)
        graph = MultiAgentGraph(CompleteStub(tokenizer), worker_specs_from_config(config),
                                num_rounds=2, max_new_tokens=512,
                                reviewer_agent_ids=reviewer_agent_ids_from_config(config))
        result = asyncio.run(graph.run_sample(sample, recorder=recorder, sample_index=sample_index))
        recorder.write({**result.to_trace_final(), **evaluate_prediction("mmlu", result.final_answer, sample)})
        selection.append({"sample_id": path.parent.name, "sample_index": sample_index,
                          "trace_json": str(path), "sha256": experiment.file_sha(path)})
        trace = experiment.load_json(path)
        event_results = [{"configuration": {"attribution_mode": "correctness-aware"},
                          "edge_scores": [{"edge_type": inc["edge_type"], "source_agent": inc["agent_id"],
                                           "target_agent": e["agent_id"], "target_round": e["round"],
                                           "token_ratio": 0.2, "word_ratio": 0.2, "binary": 1}
                                          for inc in e["incoming"]]}
                         for e in trace["events"]]
        matrices.append(build_sample_matrices(trace, event_results))
    write_json(reference / "calibration/selection_manifest.json", {"samples": selection})
    aggregate_path = reference / "attribution/aggregate_edge_matrices.json"
    write_json(aggregate_path, aggregate_sample_matrices(matrices, aggregation_scope="all"))
    write_json(reference / "attribution/manifest.json", {
        "complete": True, "aggregation_complete": True, "failures": [],
        "aggregate_sha256": experiment.file_sha(aggregate_path),
    })
    args = experiment.parse_args(["--stage", "collect", "--reference-run", str(reference),
                                  "--run-dir", str(tmp_path / "random"), "--resume"])
    return experiment.prepare_context(args), tokenizer, args


def test_full_budget_plan_is_independent_of_global_rng_and_has_full_coverage():
    agents = list("abcde")
    plan = experiment.make_rollout_plan(agents, list(range(40)))
    random.seed(135)
    for _ in range(100):
        random.random()
    assert plan == experiment.make_rollout_plan(agents, list(range(40)))
    assert plan != experiment.make_rollout_plan(agents, list(range(40)), order_seed=1)
    assert len(plan["rollouts"]) == 400
    assert len({r["rollout_id"] for r in plan["rollouts"]}) == 400
    counts = np.zeros((2, 5, 5), dtype=int)
    for rollout in plan["rollouts"]:
        for r, order in enumerate(rollout["round_orders"]):
            assert sorted(order) == agents
            for a in agents:
                for b in agents:
                    counts[r, agents.index(a), agents.index(b)] += order.index(a) < order.index(b)
    off_diagonal = ~np.eye(5, dtype=bool)
    assert np.all(counts[:, off_diagonal] > 0)
    assert np.all((counts + counts.transpose(0, 2, 1))[:, off_diagonal] == 400)
    assert any(r["round_orders"][0] != r["round_orders"][1] for r in plan["rollouts"])


def test_collect_attribute_aggregate_visualize_end_to_end(small_context):
    ctx, tokenizer, _ = small_context
    reference = Path(ctx.config["random_order_experiment"]["reference"]["run_dir"])
    original_hashes = {p: experiment.file_sha(p) for p in reference.rglob("*") if p.is_file()}
    llm = CompleteStub(tokenizer)
    summary = asyncio.run(experiment.collect_run(ctx, llm=llm))
    assert summary["num_questions"] == 2
    assert summary["num_rollouts"] == 4
    assert llm.calls == 20
    trace_hashes = {p: experiment.file_sha(p) for p in ctx.paths.traces.rglob("trace.json")}
    manifest = experiment.attribute_run(ctx, scorer=VisibleTokenScorer(tokenizer))
    assert manifest["complete"] is True
    assert len(manifest["samples"]) == 4
    aggregate = experiment.aggregate_run(ctx)
    assert aggregate["num_rollouts"] == aggregate["num_aggregated_samples"] == 4
    assert aggregate["aggregation_scope"] == "all"
    spatial = np.asarray(aggregate["metrics"]["token_ratio"]["spatial"]["opportunity_count"])
    temporal = np.asarray(aggregate["metrics"]["token_ratio"]["temporal"]["opportunity_count"])
    assert np.all((spatial + spatial.transpose(0, 2, 1))[:, 0, 1] == 4)
    assert np.all(temporal[1] == 4)
    assert np.all(temporal[0] == 0)
    visualization = experiment.visualize_run(ctx)
    assert Path(visualization["files"]["reference_comparison"]["png"]).is_file()
    assert not list(ctx.paths.run_dir.rglob("*policy*.json"))
    assert not ctx.paths.evaluation.exists()
    assert {p: experiment.file_sha(p) for p in trace_hashes} == trace_hashes
    assert {p: experiment.file_sha(p) for p in original_hashes} == original_hashes
    # Successful resume uses the frozen schedules and existing traces exactly.
    resumed_llm = CompleteStub(tokenizer)
    asyncio.run(experiment.collect_run(ctx, llm=resumed_llm))
    assert resumed_llm.calls == 0
    experiment.attribute_run(ctx, scorer=VisibleTokenScorer(tokenizer))
    assert experiment.aggregate_run(ctx)["metrics"] == aggregate["metrics"]


def test_partial_failure_resume_and_orphan_trace_recovery(small_context):
    ctx, tokenizer, _ = small_context
    with pytest.raises(RuntimeError):
        asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer, fail_after=5)))
    state_path = ctx.paths.calibration / "collection_state.json"
    state = experiment.load_json(state_path)
    assert len(state["completed"]) == 1
    assert len(state["failures"]) == 1
    assert not (ctx.paths.calibration / "selection_manifest.json").exists()
    first = Path(state["completed"][0]["trace_json"])
    initial_sha = experiment.file_sha(first)
    # Simulate a crash between atomic trace write and checkpoint update.
    state["completed"] = []
    write_json(state_path, state)
    llm = CompleteStub(tokenizer)
    asyncio.run(experiment.collect_run(ctx, llm=llm))
    assert llm.calls == 15
    assert experiment.file_sha(first) == initial_sha
    state = experiment.load_json(state_path)
    assert len(state["completed"]) == 4
    assert state["failures"] == {}
    assert state["complete"] is True


def test_failed_rescues_and_incorrect_reviewers_remain_in_new_aggregate(small_context):
    ctx, tokenizer, _ = small_context
    asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer, answer="B")))
    hashes = {p: experiment.file_sha(p) for p in ctx.paths.traces.rglob("trace.json")}
    experiment.attribute_run(ctx, scorer=FailingRescueScorer(tokenizer))
    aggregate = experiment.aggregate_run(ctx)
    assert aggregate["num_aggregated_samples"] == 4
    assert aggregate["num_correct_samples"] == 0
    assert aggregate["node_status_counts"]["rescue_failed"] == 12
    assert aggregate["node_status_counts"]["no_attributable_input"] == 4
    for edge_type in ("spatial", "temporal"):
        metric = aggregate["metrics"]["token_ratio"][edge_type]
        assert np.asarray(metric["sum"]).sum() == 0
        assert np.asarray(metric["conditional_mean"]).sum() == 0
        expected_observations = 8 if edge_type == "spatial" else 16
        assert np.asarray(metric["opportunity_count"]).sum() == expected_observations
    assert {p: experiment.file_sha(p) for p in hashes} == hashes


def test_truncation_is_diagnostic_not_completed_or_zero(small_context):
    ctx, tokenizer, _ = small_context
    with pytest.raises(ValueError, match="incomplete"):
        asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer, truncate=True)))
    state = experiment.load_json(ctx.paths.calibration / "collection_state.json")
    assert state["completed"] == []
    assert state["complete"] is False
    assert not list(ctx.paths.traces.glob("sample_*/trace.json"))
    assert list((ctx.paths.calibration / "diagnostics").rglob("failed_trace.json"))


@pytest.mark.parametrize("mutation", ["changed_trace", "missing_trace", "extra_trace", "duplicate_selection", "changed_plan"])
def test_frozen_selection_and_plan_reject_tampering(small_context, mutation):
    ctx, tokenizer, args = small_context
    asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer)))
    path = next(ctx.paths.traces.glob("sample_*/trace.json"))
    if mutation == "changed_trace":
        path.write_text(path.read_text() + "\n")
    elif mutation == "missing_trace":
        path.unlink()
    elif mutation == "extra_trace":
        write_json(ctx.paths.traces / "sample_extra/trace.json", experiment.load_json(path))
    elif mutation == "duplicate_selection":
        selection_path = ctx.paths.calibration / "selection_manifest.json"
        selection = experiment.load_json(selection_path)
        selection["samples"][-1] = selection["samples"][0]
        write_json(selection_path, selection)
    else:
        plan_path = ctx.paths.run_dir / "rollout_plan.json"
        plan = experiment.load_json(plan_path)
        plan["rollouts"][0]["round_orders"][0].reverse()
        write_json(plan_path, plan)
        with pytest.raises(ValueError, match="plan"):
            experiment.prepare_context(args)
        return
    with pytest.raises(ValueError):
        experiment.validate_selection(ctx)


@pytest.mark.parametrize("mutation", ["duplicate_checkpoint", "changed_checkpoint_trace", "no_resume"])
def test_collection_does_not_overwrite_completed_rollouts(small_context, mutation):
    ctx, tokenizer, _ = small_context
    asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer)))
    state_path = ctx.paths.calibration / "collection_state.json"
    state = experiment.load_json(state_path)
    if mutation == "duplicate_checkpoint":
        state["completed"].append(copy.deepcopy(state["completed"][0]))
        write_json(state_path, state)
    elif mutation == "changed_checkpoint_trace":
        path = Path(state["completed"][0]["trace_json"])
        path.write_text(path.read_text() + "\n")
    else:
        ctx.resume = False
    llm = CompleteStub(tokenizer)
    with pytest.raises((ValueError, FileExistsError)):
        asyncio.run(experiment.collect_run(ctx, llm=llm))
    assert llm.calls == 0


@pytest.mark.parametrize("mutation", ["missing_matrix", "duplicate_manifest", "incomplete_index", "altered_matrix", "stale_event", "failure"])
def test_aggregate_rejects_incomplete_or_changed_artifacts(small_context, mutation):
    ctx, tokenizer, _ = small_context
    asyncio.run(experiment.collect_run(ctx, llm=CompleteStub(tokenizer)))
    experiment.attribute_run(ctx, scorer=VisibleTokenScorer(tokenizer))
    experiment.aggregate_run(ctx)
    manifest_path = ctx.paths.attribution / "manifest.json"
    manifest = experiment.load_json(manifest_path)
    entry = manifest["samples"][0]
    if mutation == "missing_matrix":
        Path(entry["sample_matrix"]).unlink()
    elif mutation == "duplicate_manifest":
        manifest["samples"].append(copy.deepcopy(entry))
        write_json(manifest_path, manifest)
    elif mutation == "incomplete_index":
        path = Path(entry["node_attribution_index"])
        index = experiment.load_json(path)
        index["complete"] = False
        write_json(path, index)
    elif mutation == "altered_matrix":
        path = Path(entry["sample_matrix"])
        matrix = experiment.load_json(path)
        matrix["matrices"]["token_ratio"]["spatial"][0][0][1] = 0.98765
        write_json(path, matrix)
    elif mutation == "stale_event":
        index = experiment.load_json(Path(entry["node_attribution_index"]))
        path = Path(index["events"][0]["json"])
        event = experiment.load_json(path)
        event["configuration"]["event_sha256"] = "changed"
        write_json(path, event)
    else:
        manifest["failures"] = [{"error": "OOM"}]
        write_json(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        experiment.aggregate_run(ctx)
    assert experiment.load_json(manifest_path)["aggregation_complete"] is False
    with pytest.raises(ValueError):
        experiment.visualize_run(ctx)


def test_absent_edge_excluded_but_failed_observed_zero_counts():
    matrices = []
    for value in [0.6, 0.3, 0.0, None]:
        values = empty_sample_matrices(1, 2)
        if value is not None:
            values["opportunity"]["spatial"][0][0][1] = 1
            values["candidate"]["spatial"][0][0][1] = 1
            values["token_ratio"]["spatial"][0][0][1] = value
        matrices.append({"agents": ["a", "b"], "num_rounds": 1, "matrices": values,
                         "correct_verified": False, "run_fingerprint": "test"})
    result = aggregate_sample_matrices(matrices, aggregation_scope="all")["metrics"]["token_ratio"]["spatial"]
    assert result["opportunity_count"][0][0][1] == 3
    assert result["sum"][0][0][1] == pytest.approx(0.9)
    assert result["conditional_mean"][0][0][1] == pytest.approx(0.3)


def test_dry_run_validates_without_writing_or_loading_models(small_context, monkeypatch, capsys):
    ctx, _, args = small_context
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    experiment.main(["--reference-run", str(args.reference_run), "--run-dir", str(args.run_dir), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["num_rollouts"] == 4
    assert not ctx.paths.run_dir.exists()
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        experiment.main(["--stage", "collect", "--reference-run", str(args.reference_run), "--run-dir", str(args.run_dir)])
    assert not ctx.paths.run_dir.exists()


def test_calibrate_uses_separate_stage_processes(small_context, monkeypatch):
    _, _, args = small_context
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    commands = []
    monkeypatch.setattr(experiment.subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    experiment.main(["--reference-run", str(args.reference_run), "--run-dir", str(args.run_dir), "--resume"])
    assert [command[command.index("--stage") + 1] for command, _ in commands] == ["collect", "attribute", "aggregate", "visualize"]
    assert all(options["check"] and "--resume" in command for command, options in commands)


def test_reference_and_output_must_be_separate(small_context):
    _, _, args = small_context
    args.run_dir = args.reference_run / "new_run"
    with pytest.raises(ValueError, match="independent"):
        experiment.prepare_context(args)
