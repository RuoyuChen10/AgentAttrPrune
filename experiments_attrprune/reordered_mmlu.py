"""Frozen row-minus-column worker ordering: direct transfer or recalibration.

Each branch preserves canonical identities and the same two fixed round orders.
The direct branch projects existing attribution statistics; the recalibration
branch generates forty new dense traces before correctness-aware attribution.
All evaluation policies are deterministic global pruning over 45 candidates.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import csv
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np

from experiments_attrprune import random_order_mmlu as source_runner
from experiments_attrprune.common import (
    DEFAULT_RESULTS_ROOT, ExperimentPaths, config_fingerprint, hf_model_reference,
    load_json, reviewer_agent_ids_from_config, validate_prompt_contract,
    worker_specs_from_config, write_json,
)
from experiments_attrprune.evaluate_sampled_mmlu import _source, _check_hashes


DEFAULT_SOURCE = DEFAULT_RESULTS_ROOT / "mmlu/qwen3_8b/dev40_random_order1_correctness_aware"
NUM_QUESTIONS, NUM_ROUNDS, NUM_WORKERS = 40, 2, 5
EXPERIMENT_VERSION = 1
STAGES = ("prepare", "collect", "attribute", "aggregate", "visualize", "eval", "summarize", "all")
digest, file_sha = source_runner.digest, source_runner.file_sha
freeze_json, run_lock = source_runner.freeze_json, source_runner.run_lock
# These validators depend on the context's frozen plan, not on how its orders
# were chosen. Keeping them shared preserves the established trace contract.
validate_trace = source_runner.validate_trace
selection_snapshot = source_runner.selection_snapshot
validate_selection = source_runner.validate_selection
trace_path_for = source_runner.trace_path_for
event_digest = source_runner.event_digest


@dataclass
class ReorderedContext:
    paths: ExperimentPaths
    config: dict
    order_plan: dict
    plan: dict
    samples: dict[int, dict]
    source_config: dict
    source_hashes: dict[str, str]
    source_aggregate: dict
    branch: str
    resume: bool
    percentages: tuple[int, ...]
    eval_batch_size: int
    limit: int


def _frozen_files(ctx: ReorderedContext) -> dict[Path, dict]:
    return {
        ctx.paths.run_config: ctx.config,
        ctx.paths.run_dir / "order_plan.json": ctx.order_plan,
        ctx.paths.run_dir / "rollout_plan.json": ctx.plan,
        ctx.paths.run_dir / "source_questions.json": {
            "schema_version": 1, "source_run": ctx.config["reordered_experiment"]["source_run"],
            "samples": [{"sample_index": index, "sample": sample, "sample_sha256": digest(sample)}
                        for index, sample in sorted(ctx.samples.items())],
        },
    }


def _validate_frozen(ctx: ReorderedContext) -> None:
    root = ctx.paths.run_dir
    if not root.exists():
        return
    files = _frozen_files(ctx)
    entries = {path.name for path in root.iterdir()} - {".random_order.lock"}
    if entries and not ctx.paths.run_config.exists():
        raise FileExistsError("Nonempty output directory has no reordered configuration")
    names = {path.name for path in files}
    for path, expected in files.items():
        if path.exists() and load_json(path) != expected:
            raise ValueError(f"Frozen reordered configuration or plan changed: {path.name}")
        if not path.exists() and entries - names:
            raise ValueError(f"Incomplete frozen reordered metadata: {path.name}")


def prepare_context(args: argparse.Namespace) -> ReorderedContext:
    """Read and verify all calibration inputs without writing or loading a model."""
    from AgentAttrPrune.pruning.reordered import make_order_plan
    source = args.source_run.resolve()
    source_config, aggregate, source_hashes = _source(source)
    validate_prompt_contract(source_config)
    if (len(source_config["workers"]) != NUM_WORKERS
            or source_config["num_rounds"] != NUM_ROUNDS
            or source_config["decoding"]["temperature"] != 0
            or source_config["decoding"]["enable_thinking"] is not False):
        raise ValueError("Expected the frozen five-worker/two-round greedy MMLU source")
    source_selection = load_json(source / "calibration/selection_manifest.json")
    samples = {}
    for record in source_selection["samples"]:
        index = record["sample_index"]
        path = Path(record["trace_json"])
        if index in samples or file_sha(path) != record["sha256"]:
            raise ValueError("Source selection contains duplicate samples or changed traces")
        trace = load_json(path)
        samples[index] = trace["sample"]
        source_hashes[str(path)] = record["sha256"]
    if set(samples) != set(range(NUM_QUESTIONS)):
        raise ValueError("Source selection must contain exactly the original dev40 questions")
    order_plan = make_order_plan(aggregate)
    plan = {"schema_version": 1, "experiment_version": EXPERIMENT_VERSION,
            "agents": order_plan["agents"], "num_rounds": NUM_ROUNDS,
            "num_questions": len(samples), "repeats_per_question": 1,
            "num_rollouts": len(samples), "ordering": "fixed row-minus-column per round",
            "rollouts": [{"rollout_id": f"sample_{index:05d}_repeat_00", "sample_index": index,
                          "repeat_index": 0, "round_orders": copy.deepcopy(order_plan["round_orders"])}
                         for index in sorted(samples)]}
    suffix = "direct" if args.branch == "direct" else "recalibrated"
    output = (args.run_dir or source.with_name(f"dev40_rowminuscol_{suffix}_correctness_aware")).resolve()
    reference = Path(source_config["random_order_experiment"]["reference"]["run_dir"]).resolve()
    for protected in (source, reference):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Use an independent output directory outside the source/reference runs")
    percentages = tuple(args.percentages)
    config = copy.deepcopy(source_config)
    config.pop("random_order_experiment", None)
    config["topology"]["worker_spatial"] = "frozen per-round row-minus-column order; forward dense candidates"
    config["topology"]["worker_temporal"] = "round>0 previous-to-current all N x N including self"
    config["reordered_experiment"] = {
        "version": EXPERIMENT_VERSION, "branch": args.branch,
        "source_run": str(source), "source_run_fingerprint": source_config["fingerprint"],
        "source_artifact_sha256": source_hashes,
        "order_plan_sha256": digest(order_plan), "plan_sha256": digest(plan),
        "num_questions": len(samples), "repeats_per_question": 1,
        "order_rule": "descending raw spatial conditional_mean row sum minus column sum; canonical ties",
        "round_orders": order_plan["round_orders"], "reorder_after_attribution": False,
        "spatial_input_order": "filtered execution prefix",
        "temporal_input_order": "canonical agent identity", "reviewer_input_order": "canonical agent identity",
        "generation_batch_size": 1, "attribution_batch_size": args.attr_batch_size,
        "attribution_mode": "correctness-aware", "aggregation_scope": "all",
        "pruning_scope": "global", "candidate_worker_edges": 45,
        "percentages": list(percentages), "sampling": "none; execute all retained edges",
        "eval_batch_size": args.eval_batch_size, "evaluation_limit": args.limit,
        "save_traces": True,
    }
    config["fingerprint"] = config_fingerprint(config)
    ctx = ReorderedContext(ExperimentPaths(output), config, order_plan, plan, samples,
                           source_config, source_hashes, aggregate, args.branch, args.resume,
                           percentages, args.eval_batch_size, args.limit)
    _validate_frozen(ctx)
    _check_hashes(ctx.source_hashes)
    return ctx


def _projected_aggregate(ctx: ReorderedContext) -> dict:
    from AgentAttrPrune.pruning.reordered import project_aggregate
    value = project_aggregate(ctx.source_aggregate, ctx.order_plan)
    value["source_run_fingerprint"] = ctx.source_config["fingerprint"]
    value["run_fingerprint"] = ctx.config["fingerprint"]
    value["reordered_experiment"] = ctx.config["reordered_experiment"]
    value["calibration"] = {
        "source": "derived projection of existing attribution; no new traces or attribution",
        "source_run": ctx.config["reordered_experiment"]["source_run"],
        "original_observation_counts_preserved": True,
    }
    return value


def prepare_run(ctx: ReorderedContext) -> dict:
    _check_hashes(ctx.source_hashes)
    _validate_frozen(ctx)
    for path, value in _frozen_files(ctx).items():
        freeze_json(path, value)
    if ctx.branch == "direct":
        aggregate = _projected_aggregate(ctx)
        path = ctx.paths.attribution / "aggregate_edge_matrices.json"
        freeze_json(path, aggregate)
        freeze_json(ctx.paths.attribution / "projection_manifest.json", {
            "schema_version": 1, "kind": "derived_order_projection", "complete": True,
            "run_fingerprint": ctx.config["fingerprint"], "source_artifact_sha256": ctx.source_hashes,
            "order_plan_sha256": digest(ctx.order_plan), "aggregate_sha256": file_sha(path),
            "new_traces": 0, "new_attributions": 0,
        })
        prepare_policies(ctx)
    return {"prepared": True, "branch": ctx.branch, "run_dir": str(ctx.paths.run_dir),
            "fingerprint": ctx.config["fingerprint"], "round_orders": ctx.order_plan["round_orders"]}


async def collect_run(ctx: ReorderedContext, *, llm: Any = None) -> dict:
    from AgentAttrPrune.datasets import evaluate_prediction
    from AgentAttrPrune.graph.fixed_order import FixedOrderGraph
    from AgentAttrPrune.tracing import GenerationTraceRecorder
    from experiments_attrprune.collect_correct import set_reproducible
    from experiments_attrprune.common import hf_model_reference

    _require_recalibrate(ctx)
    prepare_run(ctx)
    freeze_json(ctx.paths.run_config, ctx.config)
    freeze_json(ctx.paths.run_dir / "rollout_plan.json", ctx.plan)
    state_path = ctx.paths.calibration / "collection_state.json"
    state = load_json(state_path) if state_path.exists() else {
        "schema_version": 1, "run_fingerprint": ctx.config["fingerprint"],
        "completed": [], "failures": {}, "complete": False,
    }
    completed = {item["rollout_id"]: item for item in state["completed"]}
    planned = {r["rollout_id"]: r for r in ctx.plan["rollouts"]}
    if (len(completed) != len(state["completed"]) or set(completed) - set(planned)
            or state.get("run_fingerprint") != ctx.config["fingerprint"]):
        raise ValueError("Collection checkpoint contains duplicate or foreign rollout identities")
    existing = list(ctx.paths.traces.glob("sample_*/trace.json"))
    if (state["completed"] or state["failures"] or existing) and not ctx.resume:
        raise FileExistsError("Collection already has attempts; use --resume or a new run directory")
    if {p.parent.name for p in existing} - set(planned):
        raise ValueError("Unexpected trace in the random-order collection")
    for rollout_id, item in completed.items():
        path = trace_path_for(ctx, planned[rollout_id])
        if item["trace_json"] != str(path.resolve()) or file_sha(path) != item["sha256"]:
            raise ValueError(f"Completed rollout changed or is missing: {rollout_id}")
        validate_trace(ctx, planned[rollout_id], load_json(path))
    frozen_selection = ctx.paths.calibration / "selection_manifest.json"
    if frozen_selection.exists():
        validate_selection(ctx)
    os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
    os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for rollout in ctx.plan["rollouts"]:
        rollout_id = rollout["rollout_id"]
        if rollout_id in completed:
            continue
        path = trace_path_for(ctx, rollout)
        started = time.perf_counter()
        recorder = None
        try:
            if path.exists():
                # Recover an atomic trace written immediately before a checkpoint interruption.
                trace = load_json(path)
                validate_trace(ctx, rollout, trace)
            else:
                if llm is None:
                    from AgentAttrPrune.llm import HFChat
                    llm = HFChat(hf_model_reference(ctx.config["model_name"]))
                set_reproducible(int(ctx.config["seed"]) + rollout["sample_index"])
                graph = FixedOrderGraph(
                    llm, worker_specs_from_config(ctx.config), num_rounds=NUM_ROUNDS,
                    round_orders=rollout["round_orders"], temperature=0,
                    max_new_tokens=ctx.config["decoding"]["max_new_tokens"],
                    reviewer_agent_ids=reviewer_agent_ids_from_config(ctx.config),
                )
                sample = ctx.samples[rollout["sample_index"]]
                recorder = GenerationTraceRecorder(
                    path, config={**ctx.config, **rollout, "phase": "calibration"}, sample=sample,
                )
                result = await graph.run_sample(sample, recorder=recorder, phase="calibration",
                                                sample_index=rollout["sample_index"])
                outcome = evaluate_prediction("mmlu", result.final_answer, sample)
                trace = recorder.payload({**result.to_trace_final(), **outcome})
                validate_trace(ctx, rollout, trace)
                write_json(path, trace)
            completed[rollout_id] = {
                **rollout, "trace_json": str(path.resolve()), "sha256": file_sha(path),
                "correct": trace["final"]["correct"], "metrics": trace["final"]["metrics"],
            }
            state["failures"].pop(rollout_id, None)
            state["completed"] = list(completed.values())
            write_json(state_path, state)
            print(f"Collected {len(completed)}/{len(planned)}: {rollout_id}", flush=True)
        except Exception as error:
            state["complete"] = False
            state["failures"][rollout_id] = {
                "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if recorder is not None:
                write_json(ctx.paths.calibration / "diagnostics" / rollout_id / "failed_trace.json",
                           recorder.payload({"error": str(error)}))
            write_json(state_path, state)
            raise
    freeze_json(frozen_selection, selection_snapshot(ctx))
    state["complete"] = True
    write_json(state_path, state)
    summary = {
        "complete": True, "run_fingerprint": ctx.config["fingerprint"],
        "num_questions": ctx.plan["num_questions"],
        "repeats_per_question": ctx.plan["repeats_per_question"],
        "num_rollouts": len(completed), "correct": sum(v["correct"] for v in completed.values()),
        "accuracy": sum(v["correct"] for v in completed.values()) / len(completed),
        "metrics": {k: sum(v["metrics"].get(k, 0) for v in completed.values())
                    for k in ("llm_calls", "prompt_tokens", "generated_tokens", "elapsed_seconds")},
    }
    write_json(ctx.paths.calibration / "summary.json", summary)
    _check_hashes(ctx.source_hashes)
    return summary


def attribution_config(ctx: ReorderedContext):
    from AgentAttrPrune.attribution.calibrator import AttributionConfig
    return AttributionConfig(attribution_mode="correctness-aware", aggregation_scope="all",
                             batch_size=ctx.config["reordered_experiment"]["attribution_batch_size"])


def _require_recalibrate(ctx: ReorderedContext) -> None:
    if ctx.branch != "recalibrate":
        raise ValueError("The direct branch does not generate or reattribute calibration traces")


def attribute_run(ctx: ReorderedContext, *, scorer: Any = None) -> dict:
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator
    _require_recalibrate(ctx)
    _validate_frozen(ctx)
    _check_hashes(ctx.source_hashes)
    validate_selection(ctx)
    if any(ctx.paths.attribution.glob("sample_*/event_*.json")) and not ctx.resume:
        raise FileExistsError("Attribution exists; use --resume")
    calibrator = EdgeAttributionCalibrator(
        ctx.paths.traces, ctx.paths.attribution, model_name=ctx.config["model_name"],
        config=attribution_config(ctx), scorer=scorer,
    )
    result = calibrator.run(attribute_only=True)
    validate_selection(ctx)
    _check_hashes(ctx.source_hashes)
    return result["manifest"]


def verified_sample_matrices(ctx: ReorderedContext, selection: dict, manifest: dict) -> tuple[list[dict], dict]:
    from AgentAttrPrune.attribution.calibrator import ALGORITHM_VERSION, CORRECTNESS_ALGORITHM_VERSION
    from AgentAttrPrune.attribution.edge import build_sample_matrices, METRICS
    if (manifest.get("complete") is not True or manifest.get("failures")
            or manifest.get("skipped_incorrect") or manifest.get("configuration") != attribution_config(ctx).to_dict()):
        raise ValueError("Attribution manifest is incomplete, failed or uses different settings")
    entries = manifest.get("samples", [])
    by_id = {e["sample_id"]: e for e in entries}
    expected = {r["rollout_id"] for r in ctx.plan["rollouts"]}
    actual = {p.parent.name for p in ctx.paths.attribution.glob("sample_*/sample_edge_matrices.json")}
    if len(by_id) != len(entries) or set(by_id) != expected or actual != expected:
        raise ValueError("Attribution requires exactly one matrix per frozen rollout")
    matrices, statuses = [], Counter()
    for record in selection["samples"]:
        rollout_id = record["rollout_id"]
        entry = by_id[rollout_id]
        folder = ctx.paths.attribution / rollout_id
        matrix_path, index_path = folder / "sample_edge_matrices.json", folder / "node_attribution_index.json"
        if (entry.get("sample_matrix") != str(matrix_path.resolve())
                or entry.get("node_attribution_index") != str(index_path.resolve())
                or entry.get("trace_json") != record["trace_json"]):
            raise ValueError("Attribution manifest points outside its rollout")
        matrix, index, trace = load_json(matrix_path), load_json(index_path), load_json(Path(record["trace_json"]))
        count = NUM_ROUNDS * NUM_WORKERS + 1
        if (index.get("complete") is not True or index.get("sample_id") != rollout_id
                or index.get("sample_index") != record["sample_index"]
                or index.get("actual_event_count") != count or index.get("expected_event_count") != count
                or len(index.get("events", [])) != count):
            raise ValueError("Incomplete node attribution index")
        events, event_paths = [], []
        for source, indexed in zip(trace["events"], index["events"]):
            from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator
            event_path = folder / f"{EdgeAttributionCalibrator._event_slug(source)}.json"
            if indexed.get("event_id") != source["event_id"] or indexed.get("json") != str(event_path.resolve()):
                raise ValueError("Node index identity/path does not match its trace")
            event = load_json(event_path)
            configuration = event.get("configuration", {})
            if (event.get("sample_id") != rollout_id or event.get("trace_json") != record["trace_json"]
                    or event.get("event", {}).get("event_id") != source["event_id"]
                    or configuration.get("event_sha256") != event_digest(source, trace)
                    or configuration.get("algorithm_version") != ALGORITHM_VERSION
                    or configuration.get("correctness_algorithm_version") != CORRECTNESS_ALGORITHM_VERSION):
                raise ValueError("Stale or mismatched event attribution")
            for key in ("attribution_mode", "aggregation_scope", "target_mode", "target_token_limit",
                        "score_type", "score_reduction", "target_recovery", "batch_size", "enable_thinking"):
                if configuration.get(key) != attribution_config(ctx).to_dict()[key]:
                    raise ValueError(f"Event attribution setting changed: {key}")
            status = event.get("status")
            if status not in {"attributed", "rescued", "rescue_failed", "no_attributable_input"}:
                raise ValueError("Failed or incomplete event cannot enter the aggregate")
            if indexed.get("status") != status:
                raise ValueError("Node status differs from its index")
            if status == "rescue_failed" and any(
                edge.get(metric, 0) != 0 for edge in event.get("edge_scores", []) for metric in METRICS
            ):
                raise ValueError("Failed rescue must contribute observed zero edge scores")
            statuses[status] += 1
            events.append(event)
            event_paths.append(str(event_path.resolve()))
        rebuilt = build_sample_matrices(trace, events)
        if (any(matrix.get(k) != value for k, value in rebuilt.items())
                or matrix.get("event_attribution_files") != event_paths
                or matrix.get("node_attribution_index") != str(index_path.resolve())):
            raise ValueError("Sample matrix does not match its node attributions")
        if (matrix.get("selection_configuration", {}).get("node_scope") != "all"
                or matrix.get("selection_configuration", {}).get("event_ids") is not None
                or matrix.get("selection_configuration", {}).get("event_limit") is not None):
            raise ValueError("All nodes must be attributed")
        for edge_type in ("spatial", "temporal"):
            opportunities = np.asarray(matrix["matrices"]["opportunity"][edge_type])
            candidate = np.asarray(matrix["matrices"]["candidate"][edge_type])
            if not np.array_equal(opportunities, candidate):
                raise ValueError("An actual edge is missing its attribution observation")
            for metric in METRICS:
                values = np.asarray(matrix["matrices"][metric][edge_type])
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError("Edge importance must be a finite ratio in [0, 1]")
                if np.any(values[opportunities == 0] != 0):
                    raise ValueError("Absent edges cannot contribute importance")
        matrices.append(matrix)
    return matrices, dict(statuses)


def _aggregate_from_artifacts(ctx: ReorderedContext, manifest: dict) -> dict:
    from AgentAttrPrune.attribution.edge import aggregate_sample_matrices
    selection = validate_selection(ctx)
    matrices, statuses = verified_sample_matrices(ctx, selection, manifest)
    if len(matrices) != NUM_QUESTIONS or sum(statuses.values()) != NUM_QUESTIONS * (NUM_ROUNDS * NUM_WORKERS + 1):
        raise ValueError("Recalibration requires exactly 40 traces and 440 node records")
    aggregate = aggregate_sample_matrices(matrices, aggregation_scope="all")
    for edge in ("spatial", "temporal"):
        expected = np.asarray(ctx.order_plan["candidate_masks"][edge], dtype=int) * NUM_QUESTIONS
        for metric in aggregate["metrics"].values():
            if (not np.array_equal(np.asarray(metric[edge]["opportunity_count"]), expected)
                    or not np.array_equal(np.asarray(metric[edge]["candidate_count"]), expected)):
                raise ValueError("Fixed-order recalibration counts must be 40 on every allowed edge and zero elsewhere")
    aggregate.update({"num_questions": NUM_QUESTIONS, "repeats_per_question": 1,
                      "num_rollouts": NUM_QUESTIONS, "node_status_counts": statuses,
                      "reordered_experiment": ctx.config["reordered_experiment"],
                      "selection_sha256": file_sha(ctx.paths.calibration / "selection_manifest.json"),
                      "calibration": {"source": "all new frozen-order dev40 rollouts, including final-incorrect",
                                      "run_dir": str(ctx.paths.run_dir), "selection": selection}})
    return aggregate


def aggregate_run(ctx: ReorderedContext) -> dict:
    _require_recalibrate(ctx)
    _validate_frozen(ctx)
    _check_hashes(ctx.source_hashes)
    manifest_path = ctx.paths.attribution / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["aggregation_complete"] = False
    manifest.pop("aggregate_sha256", None)
    write_json(manifest_path, manifest)
    aggregate = _aggregate_from_artifacts(ctx, manifest)
    path = ctx.paths.attribution / "aggregate_edge_matrices.json"
    write_json(path, aggregate)
    manifest.update(aggregation_complete=True, aggregate_sha256=file_sha(path))
    write_json(manifest_path, manifest)
    prepare_policies(ctx)
    return aggregate


def validate_ready_for_evaluation(ctx: ReorderedContext) -> dict:
    """Validate publication against exact source statistics or all node artifacts."""
    _validate_frozen(ctx)
    _check_hashes(ctx.source_hashes)
    path = ctx.paths.attribution / "aggregate_edge_matrices.json"
    aggregate = load_json(path)
    if ctx.branch == "direct":
        manifest = load_json(ctx.paths.attribution / "projection_manifest.json")
        if (manifest.get("kind") != "derived_order_projection" or manifest.get("complete") is not True
                or manifest.get("aggregate_sha256") != file_sha(path)
                or manifest.get("order_plan_sha256") != digest(ctx.order_plan)
                or aggregate != _projected_aggregate(ctx)):
            raise ValueError("Derived projection differs from its frozen source or order")
    else:
        manifest = load_json(ctx.paths.attribution / "manifest.json")
        if (manifest.get("complete") is not True or manifest.get("aggregation_complete") is not True
                or manifest.get("failures") or manifest.get("aggregate_sha256") != file_sha(path)):
            raise ValueError("Recalibrated attribution is incomplete or stale")
        if aggregate != _aggregate_from_artifacts(ctx, manifest):
            raise ValueError("Recalibrated aggregate differs from its complete node attributions")
    if aggregate.get("run_fingerprint") != ctx.config["fingerprint"]:
        raise ValueError("Aggregate belongs to another reordered run")
    return aggregate


def _policy_payloads(ctx: ReorderedContext, aggregate: dict) -> dict[int, dict]:
    from AgentAttrPrune.pruning.reordered import build_reordered_policy
    result = {}
    for percent in ctx.percentages:
        policy = build_reordered_policy(aggregate, ctx.order_plan, percent)
        policy["reordered_source"] = {"branch": ctx.branch,
            "run_fingerprint": ctx.config["fingerprint"], "order_plan_sha256": digest(ctx.order_plan),
            "aggregate_sha256": file_sha(ctx.paths.attribution / "aggregate_edge_matrices.json")}
        result[percent] = policy
    return result


def prepare_policies(ctx: ReorderedContext) -> dict[int, dict]:
    aggregate = validate_ready_for_evaluation(ctx)
    policies = _policy_payloads(ctx, aggregate)
    root = ctx.paths.run_dir / "policies"
    entries = []
    for percent, policy in policies.items():
        path = root / f"prune_{percent}pct.json"
        freeze_json(path, policy)
        entries.append({"pruning_percent": percent, "policy_json": str(path),
                        "policy_sha256": digest(policy), "file_sha256": file_sha(path)})
    freeze_json(root / "manifest.json", {"schema_version": 1, "complete": True,
        "run_fingerprint": ctx.config["fingerprint"], "branch": ctx.branch,
        "order_plan_sha256": digest(ctx.order_plan), "policies": entries})
    return policies


def load_policies(ctx: ReorderedContext) -> dict[int, dict]:
    aggregate = validate_ready_for_evaluation(ctx)
    expected = _policy_payloads(ctx, aggregate)
    root = ctx.paths.run_dir / "policies"
    manifest = load_json(root / "manifest.json")
    entries = []
    for percent, payload in expected.items():
        path = root / f"prune_{percent}pct.json"
        if load_json(path) != payload:
            raise ValueError("Frozen reordered pruning policy changed")
        entries.append({"pruning_percent": percent, "policy_json": str(path),
                        "policy_sha256": digest(payload), "file_sha256": file_sha(path)})
    if manifest != {"schema_version": 1, "complete": True,
                    "run_fingerprint": ctx.config["fingerprint"], "branch": ctx.branch,
                    "order_plan_sha256": digest(ctx.order_plan), "policies": entries}:
        raise ValueError("Frozen reordered policy manifest changed")
    return expected


def _summary_rows(ctx: ReorderedContext, policies: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    from experiments_attrprune import reordered_evaluation as evaluation
    rows, summaries = [], []
    for percent in ctx.percentages:
        directory = ctx.paths.evaluation / f"global_prune_{percent:02d}pct_val"
        path = directory / "evaluation_summary.json"
        summary = load_json(path)
        if (summary.get("complete") is not True or summary.get("branch") != ctx.branch
                or summary.get("run_fingerprint") != ctx.config["fingerprint"]
                or summary.get("total") != ctx.limit or summary.get("eval_batch_size") != ctx.eval_batch_size
                or summary.get("pruning_percent") != percent
                or summary.get("policy_sha256") != digest(policies[percent])
                or summary.get("order_plan_sha256") != digest(ctx.order_plan)):
            raise ValueError(f"Evaluation summary is incomplete or incompatible: {path}")
        predictions = directory / "predictions.jsonl"
        if summary.get("predictions_sha256") != file_sha(predictions):
            raise ValueError("Evaluation predictions changed after publication")
        state = evaluation.prepare_evaluation(ctx, pruning_percent=percent,
            eval_batch_size=ctx.eval_batch_size, limit=ctx.limit)
        records = evaluation._resume_records(state, resume=True)
        attempts = evaluation._read_jsonl(directory / "attempts.jsonl")
        if (len(records) != ctx.limit or any(
            item.get("evaluation_fingerprint") != state.config["fingerprint"]
            or item.get("sample_index") not in state.samples
            or item.get("status") not in {"started", "completed", "failed", "recovered"}
            for item in attempts)):
            raise ValueError("Evaluation coverage or attempt identity is incomplete")
        if summary != evaluation._summary(state, records, attempts):
            raise ValueError("Evaluation summary differs from its validated trace checkpoints")
        summaries.append(summary)
        rows.append({key: summary[key] for key in (
            "branch", "pruning_percent", "total", "correct", "accuracy", "llm_calls",
            "prompt_tokens", "generated_tokens", "elapsed_seconds", "eval_batch_size")})
        rows[-1]["worker_active_edges"] = summary["edge_counts"]["worker_active"]
    return rows, summaries


def _write_comparison_if_ready(ctx: ReorderedContext, rows: list[dict]) -> dict:
    json_path = ctx.paths.run_dir / "branch_comparison.json"
    def unavailable(reason: str) -> dict:
        if json_path.exists():
            write_json(json_path, {"schema_version": 1, "complete": False, "error": reason})
        return {"comparison_status": reason}
    other_branch = "recalibrate" if ctx.branch == "direct" else "direct"
    suffix = "recalibrated" if other_branch == "recalibrate" else "direct"
    other_root = ctx.paths.run_dir.parent / f"dev40_rowminuscol_{suffix}_correctness_aware"
    other_summary = other_root / "sweep_summary.json"
    if not other_summary.exists():
        return unavailable("other branch has no complete sweep summary")
    expected_config = copy.deepcopy(ctx.config)
    expected_config["reordered_experiment"]["branch"] = other_branch
    expected_config["fingerprint"] = config_fingerprint(expected_config)
    try:
        if load_json(other_root / "run_config.json") != expected_config:
            raise ValueError("Branch source, ordering or evaluation configuration differs")
        other = replace(ctx, paths=ExperimentPaths(other_root), config=expected_config, branch=other_branch)
        other_rows, summaries = _summary_rows(other, load_policies(other))
        published = load_json(other_summary)
        if (published.get("complete") is not True or published.get("run_fingerprint") != other.config["fingerprint"]
                or published.get("round_orders") != ctx.order_plan["round_orders"]
                or published.get("percentages") != list(ctx.percentages)
                or published.get("results") != summaries):
            raise ValueError("Other branch sweep summary is stale or incomplete")
        # Each rate must use exactly the same frozen validation question list.
        for percent, summary in zip(ctx.percentages, summaries):
            current = load_json(ctx.paths.evaluation / f"global_prune_{percent:02d}pct_val/evaluation_summary.json")
            if summary.get("selection_sha256") != current.get("selection_sha256"):
                raise ValueError("The branches evaluated different validation selections")
    except (ValueError, FileNotFoundError, KeyError) as error:
        return unavailable(f"other branch not ready or compatible: {error}")
    paired_rows = sorted(rows + other_rows, key=lambda row: (
        ctx.percentages.index(row["pruning_percent"]), row["branch"]))
    csv_path = ctx.paths.run_dir / "branch_comparison.csv"
    write_json(json_path, {"schema_version": 1, "complete": True,
        "source_run_fingerprint": ctx.source_config["fingerprint"],
        "order_plan_sha256": digest(ctx.order_plan), "round_orders": ctx.order_plan["round_orders"],
        "eval_batch_size": ctx.eval_batch_size, "evaluation_limit": ctx.limit,
        "branch_run_directories": {ctx.branch: str(ctx.paths.run_dir), other_branch: str(other_root)},
        "results": paired_rows})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    return {"comparison_status": "complete", "comparison_json": str(json_path), "comparison_csv": str(csv_path)}


def summarize_run(ctx: ReorderedContext) -> dict:
    policies = load_policies(ctx)
    rows, summaries = _summary_rows(ctx, policies)
    result = {"schema_version": 1, "complete": True, "branch": ctx.branch,
              "run_fingerprint": ctx.config["fingerprint"], "round_orders": ctx.order_plan["round_orders"],
              "percentages": list(ctx.percentages), "results": summaries}
    write_json(ctx.paths.run_dir / "sweep_summary.json", result)
    with (ctx.paths.run_dir / "sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {**result, **_write_comparison_if_ready(ctx, rows)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("direct", "recalibrate"), required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--percentages", type=int, nargs="+", default=[80, 60, 50])
    parser.add_argument("--pruning-percent", type=int, choices=(80, 60, 50))
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1531)
    parser.add_argument("--attr-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (len(set(args.percentages)) != len(args.percentages)
            or any(percent not in {80, 60, 50} for percent in args.percentages)):
        parser.error("Choose distinct percentages from 80, 60, 50")
    if min(args.eval_batch_size, args.limit, args.attr_batch_size) < 1:
        parser.error("Batch sizes and evaluation limit must be positive")
    if args.pruning_percent is not None and (args.stage != "eval" or args.pruning_percent not in args.percentages):
        parser.error("--pruning-percent selects one configured percentage for --stage eval")
    if args.branch == "direct" and args.stage in {"collect", "attribute", "aggregate"}:
        parser.error("The direct branch has no collection or attribution stages")
    return args


def _child_args(args: argparse.Namespace, stage: str, *, percent: int | None = None) -> list[str]:
    command = [sys.executable, "-u", "-m", "experiments_attrprune.reordered_mmlu",
               "--branch", args.branch, "--stage", stage, "--source-run", str(args.source_run),
               "--percentages", *map(str, args.percentages), "--eval-batch-size", str(args.eval_batch_size),
               "--limit", str(args.limit), "--attr-batch-size", str(args.attr_batch_size)]
    if args.run_dir is not None:
        command += ["--run-dir", str(args.run_dir)]
    if args.resume:
        command += ["--resume"]
    if percent is not None:
        command += ["--pruning-percent", str(percent)]
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "branch": ctx.branch, "run_dir": str(ctx.paths.run_dir),
                          "fingerprint": ctx.config["fingerprint"], "num_questions": len(ctx.samples),
                          "round_orders": ctx.order_plan["round_orders"], "candidate_edges": 45,
                          "percentages": list(ctx.percentages), "eval_batch_size": ctx.eval_batch_size,
                          "evaluation_limit": ctx.limit}, indent=2))
        return
    if args.stage in {"collect", "attribute", "eval", "all"} and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for GPU stages")
    if args.stage == "all":
        stages = ["prepare"] + (["collect", "attribute", "aggregate"] if args.branch == "recalibrate" else [])
        for stage in stages + ["visualize"]:
            subprocess.run(_child_args(args, stage), check=True)
        for percent in ctx.percentages:
            subprocess.run(_child_args(args, "eval", percent=percent), check=True)
        subprocess.run(_child_args(args, "summarize"), check=True)
        return
    if args.stage == "eval" and args.pruning_percent is None:
        for percent in ctx.percentages:
            subprocess.run(_child_args(args, "eval", percent=percent), check=True)
        return
    with run_lock(ctx.paths.run_dir):
        if args.stage == "prepare":
            result = prepare_run(ctx)
        elif args.stage == "collect":
            result = asyncio.run(collect_run(ctx))
        elif args.stage == "attribute":
            result = attribute_run(ctx)
        elif args.stage == "aggregate":
            result = aggregate_run(ctx)
        elif args.stage == "visualize":
            from experiments_attrprune.reordered_visualization import render_run
            result = render_run(ctx)
        elif args.stage == "eval":
            from experiments_attrprune.reordered_evaluation import evaluate_policy
            result = asyncio.run(evaluate_policy(ctx, pruning_percent=args.pruning_percent,
                resume=ctx.resume, eval_batch_size=ctx.eval_batch_size, limit=ctx.limit))
        else:
            result = summarize_run(ctx)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
