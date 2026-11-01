"""Independent dev40 calibration resamples with the original fixed worker order.

Only the calibration selection varies. Generation/evaluation seeds, prompts,
roles, decoding and attribution settings are inherited from the reference run.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np

from AgentAttrPrune.datasets import MMLUDataset
from experiments_attrprune.common import (
    DEFAULT_RESULTS_ROOT, ExperimentPaths, benchmark_data_identity, config_fingerprint,
    dataset_identities_match, load_json, reviewer_agent_ids_from_config,
    validate_prompt_contract, worker_specs_from_config, write_json,
)
from experiments_attrprune import random_order_mmlu as source_runner

DEFAULT_REFERENCE = DEFAULT_RESULTS_ROOT / "mmlu/qwen3_8b/dev40_correctness_aware"
NUM_QUESTIONS, NUM_ROUNDS, NUM_WORKERS = 40, 2, 5
PERCENTAGES = (25, 50, 60, 70, 80, 90)
STAGES = ("prepare", "collect", "attribute", "aggregate", "policies", "eval", "summarize", "all")
digest, file_sha = source_runner.digest, source_runner.file_sha
freeze_json, run_lock = source_runner.freeze_json, source_runner.run_lock
validate_trace = source_runner.validate_trace
selection_snapshot, validate_selection = source_runner.selection_snapshot, source_runner.validate_selection
trace_path_for, event_digest = source_runner.trace_path_for, source_runner.event_digest


@dataclass
class ResampledContext:
    paths: ExperimentPaths
    config: dict
    plan: dict
    samples: dict[int, dict]
    round_orders: list[list[str]]
    order_plan: dict
    source_config: dict
    source_hashes: dict[str, str]
    calibration_seed: int
    reference_run: Path
    resume: bool
    percentages: tuple[int, ...]
    eval_batch_size: int
    limit: int


def _check_hashes(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if file_sha(Path(name)) != expected:
            raise ValueError(f"Frozen reference artifact changed: {name}")


def _reference(path: Path) -> tuple[dict, dict, dict, dict[str, str]]:
    paths = [path / "run_config.json", path / "attribution/aggregate_edge_matrices.json",
             path / "attribution/manifest.json", path / "calibration/selection_manifest.json"]
    config, aggregate, manifest, selection = [load_json(p) for p in paths]
    hashes = {str(p): file_sha(p) for p in paths}
    from experiments_attrprune.static_pruning import _validate_sources
    _validate_sources(config, aggregate, metric="token_ratio", statistic="conditional_mean")
    validate_prompt_contract(config)
    if (len(config["workers"]) != NUM_WORKERS or config["num_rounds"] != NUM_ROUNDS
            or config["seed"] != 0 or config["decoding"]["temperature"] != 0
            or config["decoding"]["enable_thinking"] is not False
            or config["decoding"]["max_new_tokens"] != 2048):
        raise ValueError("Reference must be the original five-worker/two-round Qwen dev40 setup")
    if (aggregate.get("num_aggregated_samples") != NUM_QUESTIONS
            or aggregate.get("aggregation_scope") != "all"
            or aggregate.get("attribution_configuration", {}).get("attribution_mode") != "correctness-aware"
            or manifest.get("complete") is not True or manifest.get("aggregation_complete") is not True
            or manifest.get("failures") or manifest.get("skipped_incorrect")
            or manifest.get("aggregate_sha256") != hashes[str(paths[1])]
            or len(manifest.get("samples", [])) != NUM_QUESTIONS):
        raise ValueError("Reference attribution must be the complete correctness-aware all-dev40 publication")
    expected_ids = {f"sample_{i:05d}" for i in range(NUM_QUESTIONS)}
    if {s.get("sample_id") for s in manifest["samples"]} != expected_ids:
        raise ValueError("Reference attribution has duplicate/missing samples")
    selected = selection.get("samples", [])
    if len(selected) != NUM_QUESTIONS or {s["sample_index"] for s in selected} != set(range(NUM_QUESTIONS)):
        raise ValueError("Reference calibration selection must be the original dev indices 0..39")
    from AgentAttrPrune.attribution.calibrator import AttributionConfig
    cfg = AttributionConfig(**manifest["configuration"])
    if (cfg.batch_size != 4 or cfg.sample_scope != "all" or cfg.node_scope != "all"
            or cfg.sample_limit is not None or cfg.event_limit is not None or cfg.event_ids is not None
            or cfg.attribution_mode != "correctness-aware" or cfg.aggregation_scope != "all"
            or cfg.accept_incomplete or cfg.allow_legacy_traces):
        raise ValueError("Reference attribution settings are incompatible with the all-node experiment")
    n = len(config["workers"])
    expected_spatial = np.broadcast_to(np.triu(np.ones((n,n), dtype=int),1),(NUM_ROUNDS,n,n))*NUM_QUESTIONS
    expected_temporal = np.zeros((NUM_ROUNDS,n,n), dtype=int); expected_temporal[1:] = NUM_QUESTIONS
    for edge, expected in (("spatial",expected_spatial),("temporal",expected_temporal)):
        if not np.array_equal(np.asarray(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"]),expected):
            raise ValueError("Reference must use the original canonical fixed order")
    return config, aggregate, manifest["configuration"], hashes


def _frozen_files(ctx: ResampledContext) -> dict[Path, dict]:
    return {ctx.paths.run_config: ctx.config,
            ctx.paths.run_dir / "calibration_plan.json": ctx.plan,
            ctx.paths.run_dir / "order_plan.json": ctx.order_plan}


def _validate_frozen(ctx: ResampledContext) -> None:
    root = ctx.paths.run_dir
    if not root.exists(): return
    entries = {p.name for p in root.iterdir()} - {".random_order.lock"}
    files = _frozen_files(ctx)
    if entries and not ctx.paths.run_config.exists():
        raise FileExistsError("Nonempty resampling directory has no frozen configuration")
    for path, expected in files.items():
        if path.exists() and load_json(path) != expected:
            raise ValueError(f"Frozen resampling configuration or plan changed: {path.name}")
        if not path.exists() and entries - {p.name for p in files}:
            raise ValueError(f"Incomplete frozen resampling metadata: {path.name}")


def prepare_context(args: argparse.Namespace) -> ResampledContext:
    """Read-only planning; calibration and evaluation random seeds stay distinct."""
    reference = args.reference_run.resolve()
    base, aggregate, attr_settings, hashes = _reference(reference)
    identity = base["dataset"]
    data_dir = Path(identity.get("requested_data_dir") or identity["data_root"])
    actual = benchmark_data_identity("mmlu",data_dir,splits=identity["splits"],shuffle_seed=identity["shuffle_seed"])
    if not dataset_identities_match(identity,actual): raise ValueError("MMLU data fingerprint changed")
    dataset = MMLUDataset(split="dev",data_dir=data_dir,download=False,shuffle_seed=identity["shuffle_seed"])
    if len(dataset) < NUM_QUESTIONS: raise ValueError("Too few dev questions")
    indices = sorted(random.Random(args.calibration_seed).sample(range(len(dataset)),NUM_QUESTIONS))
    if indices == list(range(NUM_QUESTIONS)):
        raise ValueError("Resampling unexpectedly reproduced the baseline cohort")
    samples = {i:dataset.record_to_sample(dataset[j]) for i,j in enumerate(indices)}
    agents = [w["agent_id"] for w in base["workers"]]
    orders = [list(agents) for _ in range(NUM_ROUNDS)]
    n = len(agents)
    spatial = np.broadcast_to(np.triu(np.ones((n,n),dtype=int),1),(NUM_ROUNDS,n,n)).copy()
    temporal = np.zeros((NUM_ROUNDS,n,n),dtype=int);temporal[1:]=1
    order_plan = {"schema_version":1,"agents":agents,"round_orders":orders,
                  "order_rule":"original canonical fixed order",
                  "candidate_masks":{"spatial":spatial.tolist(),"temporal":temporal.tolist()}}
    plan = {"schema_version":1,"calibration_seed":args.calibration_seed,
            "sampling_rule":"random.Random(seed).sample(dev_indices,40); sorted; no replacement within cohort; independent cohorts may overlap",
            "dataset_size":len(dataset),"dataset_identity":identity,
            "selected_dataset_indices":indices,"agents":agents,"num_rounds":NUM_ROUNDS,
            "num_questions":NUM_QUESTIONS,"repeats_per_question":1,"num_rollouts":NUM_QUESTIONS,
            "rollouts":[{"rollout_id":f"sample_{i:05d}","sample_index":i,"dataset_index":j,
                "sample_sha256":digest(samples[i]),"round_orders":orders} for i,j in enumerate(indices)],
            "questions":[{"sample_index":i,"dataset_index":j,"sample":samples[i],"sample_sha256":digest(samples[i])}
                         for i,j in enumerate(indices)]}
    output = (args.run_dir or reference.with_name(f"dev40_resample_seed{args.calibration_seed}_correctness_aware")).resolve()
    if output==reference or reference in output.parents or output in reference.parents:
        raise ValueError("Use an independent resampling directory outside the reference")
    config = copy.deepcopy(base)
    config["resampled_experiment"] = {
        "version":1,"calibration_seed":args.calibration_seed,"reference_run":str(reference),
        "reference_run_fingerprint":base["fingerprint"],"reference_artifact_sha256":hashes,
        "sampling_rule":plan["sampling_rule"],"selected_dataset_indices":indices,
        "calibration_plan_sha256":digest(plan),"order_plan_sha256":digest(order_plan),
        "num_questions":NUM_QUESTIONS,"generation_seed":base["seed"],"generation_batch_size":1,
        "attribution_configuration":copy.deepcopy(attr_settings),"attribution_batch_size":attr_settings["batch_size"],
        "eval_batch_size":args.eval_batch_size,"evaluation_limit":args.limit,"save_traces":True,
        "percentages":list(args.percentages),"pruning_scope":"global","sampling":"none; execute all retained edges",
        "empty_round_rule":"skip Round1 iff both spatial and temporal masks are empty; reviewer reads Round0",
    }
    config["fingerprint"] = config_fingerprint(config)
    ctx = ResampledContext(ExperimentPaths(output),config,plan,samples,orders,order_plan,base,hashes,
        args.calibration_seed,reference,args.resume,tuple(args.percentages),args.eval_batch_size,args.limit)
    _validate_frozen(ctx);_check_hashes(hashes)
    return ctx


def prepare_run(ctx: ResampledContext) -> dict:
    _validate_frozen(ctx);_check_hashes(ctx.source_hashes)
    for path,payload in _frozen_files(ctx).items(): freeze_json(path,payload)
    return {"prepared":True,"run_dir":str(ctx.paths.run_dir),"calibration_seed":ctx.calibration_seed,
            "generation_seed":ctx.config["seed"],"fingerprint":ctx.config["fingerprint"],
            "selected_dataset_indices":ctx.plan["selected_dataset_indices"],"round_orders":ctx.round_orders}


async def collect_run(ctx: ResampledContext, *, llm: Any = None) -> dict:
    from AgentAttrPrune.datasets import evaluate_prediction
    from AgentAttrPrune.graph.fixed_order import FixedOrderGraph
    from AgentAttrPrune.tracing import GenerationTraceRecorder
    from experiments_attrprune.collect_correct import set_reproducible
    from experiments_attrprune.common import hf_model_reference

    prepare_run(ctx)
    freeze_json(ctx.paths.run_config, ctx.config)
    freeze_json(ctx.paths.run_dir / "calibration_plan.json", ctx.plan)
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
                set_reproducible(int(ctx.config["seed"]) + rollout["dataset_index"])
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




def attribution_config(ctx: ResampledContext):
    from AgentAttrPrune.attribution.calibrator import AttributionConfig
    return AttributionConfig(**ctx.config["resampled_experiment"]["attribution_configuration"])


def attribute_run(ctx: ResampledContext, *, scorer: Any = None) -> dict:
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator
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



def verified_sample_matrices(ctx: ResampledContext, selection: dict, manifest: dict) -> tuple[list[dict], dict]:
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
            # Match the calibrator's event-cache contract, including early-stop,
            # tolerance and insertion settings; run/selection options live in
            # the separately verified manifest instead of individual events.
            non_event_settings = {
                "sample_glob", "sample_limit", "sample_scope", "node_scope", "event_limit",
                "event_ids", "min_incoming", "include_reviewer", "overwrite", "keep_going",
                "allow_legacy_traces", "policy_pruning_rate", "saliency_floor", "visualizations",
            }
            for key, expected_value in attribution_config(ctx).to_dict().items():
                if key in non_event_settings:
                    continue
                if key not in configuration or configuration[key] != expected_value:
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



def _aggregate_from_artifacts(ctx: ResampledContext, manifest: dict) -> dict:
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
                      "resampled_experiment": ctx.config["resampled_experiment"],
                      "selection_sha256": file_sha(ctx.paths.calibration / "selection_manifest.json"),
                      "calibration": {"source": "all new independently sampled dev40 rollouts, including final-incorrect",
                                      "run_dir": str(ctx.paths.run_dir), "selection": selection}})
    return aggregate



def aggregate_run(ctx: ResampledContext) -> dict:
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




def validate_ready_for_evaluation(ctx: ResampledContext) -> dict:
    _validate_frozen(ctx);_check_hashes(ctx.source_hashes)
    path = ctx.paths.attribution / "aggregate_edge_matrices.json"
    aggregate = load_json(path)
    manifest = load_json(ctx.paths.attribution / "manifest.json")
    if (manifest.get("complete") is not True or manifest.get("aggregation_complete") is not True
            or manifest.get("failures") or manifest.get("aggregate_sha256") != file_sha(path)):
        raise ValueError("Resampled attribution is incomplete or stale")
    if aggregate != _aggregate_from_artifacts(ctx,manifest):
        raise ValueError("Aggregate differs from its complete node attributions")
    if aggregate.get("run_fingerprint") != ctx.config["fingerprint"]:
        raise ValueError("Aggregate belongs to another resampled run")
    return aggregate


def _policy_payloads(ctx: ResampledContext, aggregate: dict) -> dict[int,dict]:
    from AgentAttrPrune.attribution.edge import build_pruned_masks
    from AgentAttrPrune.pruning import RoundPolicy
    result = {}
    for rate in ctx.percentages:
        payload = build_pruned_masks(aggregate,metric="token_ratio",statistic="conditional_mean",
            threshold=None,pruning_rate=rate/100,unobserved="keep",pruning_scope="global")
        payload["static_sweep"] = {"label":f"prune_{rate:02d}pct","requested_pruning_percent":rate,
            "aggregate_json":str(ctx.paths.attribution/"aggregate_edge_matrices.json"),
            "aggregate_sha256":file_sha(ctx.paths.attribution/"aggregate_edge_matrices.json")}
        payload["resampled_source"] = {"calibration_seed":ctx.calibration_seed,
            "calibration_plan_sha256":digest(ctx.plan),"run_fingerprint":ctx.config["fingerprint"]}
        result[rate] = RoundPolicy.from_dict(payload).to_dict()
    return result


def _policy_manifest(ctx: ResampledContext, policies: dict[int,dict]) -> dict:
    root = ctx.paths.run_dir / "policies"
    return {"schema_version":1,"complete":True,"run_fingerprint":ctx.config["fingerprint"],
        "calibration_seed":ctx.calibration_seed,"order_plan_sha256":digest(ctx.order_plan),
        "policies":[{"pruning_percent":rate,"policy_json":str(root/f"prune_{rate}pct.json"),
                     "policy_sha256":digest(payload),"file_sha256":file_sha(root/f"prune_{rate}pct.json")}
                    for rate,payload in policies.items()]}


def prepare_policies(ctx: ResampledContext) -> dict[int,dict]:
    policies = _policy_payloads(ctx,validate_ready_for_evaluation(ctx))
    for rate,payload in policies.items(): freeze_json(ctx.paths.run_dir/"policies"/f"prune_{rate}pct.json",payload)
    freeze_json(ctx.paths.run_dir/"policies/manifest.json",_policy_manifest(ctx,policies))
    return policies


def load_policies(ctx: ResampledContext) -> dict[int,dict]:
    policies = _policy_payloads(ctx,validate_ready_for_evaluation(ctx))
    for rate,payload in policies.items():
        if load_json(ctx.paths.run_dir/"policies"/f"prune_{rate}pct.json") != payload:
            raise ValueError("Frozen resampled pruning policy changed")
    if load_json(ctx.paths.run_dir/"policies/manifest.json") != _policy_manifest(ctx,policies):
        raise ValueError("Frozen resampled policy manifest changed")
    return policies


def summarize_run(ctx: ResampledContext) -> dict:
    from experiments_attrprune.resampled_evaluation import validate_completed_policy
    summaries = [validate_completed_policy(ctx,pruning_percent=rate,
        eval_batch_size=ctx.eval_batch_size,limit=ctx.limit) for rate in ctx.percentages]
    rows = [{"calibration_seed":ctx.calibration_seed,"pruning_percent":s["pruning_percent"],
        "total":s["total"],"correct":s["correct"],"accuracy":s["accuracy"],
        "llm_calls":s["llm_calls"],"prompt_tokens":s["prompt_tokens"],
        "generated_tokens":s["generated_tokens"],"elapsed_seconds":s["elapsed_seconds"],
        "retained_edges":s["retained_edges"],
        "reviewer_source_round":s["execution_contract"]["reviewer_source_round"]} for s in summaries]
    result = {"complete":True,"calibration_seed":ctx.calibration_seed,"run_fingerprint":ctx.config["fingerprint"],
              "percentages":list(ctx.percentages),"rows":rows,"evaluations":summaries}
    write_json(ctx.paths.run_dir/"sweep_summary.json",result)
    with (ctx.paths.run_dir/"sweep_summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-seed",type=int,choices=(1,2),required=True)
    parser.add_argument("--reference-run",type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--stage",choices=STAGES,default="all")
    parser.add_argument("--percentages",type=int,nargs="+",default=list(PERCENTAGES))
    parser.add_argument("--pruning-percent",type=int,choices=PERCENTAGES)
    parser.add_argument("--eval-batch-size",type=int,default=4)
    parser.add_argument("--limit",type=int,default=1531)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    if tuple(args.percentages)!=PERCENTAGES: parser.error("Keep the six frozen rates: 25 50 60 70 80 90")
    if min(args.eval_batch_size,args.limit)<1: parser.error("Batch size and limit must be positive")
    if args.pruning_percent is not None and args.stage!="eval": parser.error("--pruning-percent requires --stage eval")
    return args


def _child_args(args: argparse.Namespace, stage: str, rate: int | None = None) -> list[str]:
    cmd=[sys.executable,"-u","-m","experiments_attrprune.resampled_mmlu",
         "--calibration-seed",str(args.calibration_seed),"--reference-run",str(args.reference_run),
         "--stage",stage,"--eval-batch-size",str(args.eval_batch_size),"--limit",str(args.limit)]
    if args.run_dir is not None:cmd += ["--run-dir",str(args.run_dir)]
    if args.resume:cmd += ["--resume"]
    if rate is not None:cmd += ["--pruning-percent",str(rate)]
    return cmd


def main(argv: Sequence[str] | None = None) -> None:
    args=parse_args(argv);ctx=prepare_context(args)
    if args.dry_run:
        print(json.dumps({"dry_run":True,"run_dir":str(ctx.paths.run_dir),"calibration_seed":ctx.calibration_seed,
            "generation_seed":ctx.config["seed"],"selected_dataset_indices":ctx.plan["selected_dataset_indices"],
            "fingerprint":ctx.config["fingerprint"],"round_orders":ctx.round_orders,
            "percentages":list(ctx.percentages),"eval_batch_size":ctx.eval_batch_size,"limit":ctx.limit},indent=2))
        return
    if args.stage in {"collect","attribute","eval","all"} and not os.environ.get("CUDA_VISIBLE_DEVICES","").strip():
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for GPU stages")
    if args.stage=="all":
        for stage in ("prepare","collect","attribute","aggregate"):
            subprocess.run(_child_args(args,stage),check=True)
        for rate in ctx.percentages:subprocess.run(_child_args(args,"eval",rate),check=True)
        subprocess.run(_child_args(args,"summarize"),check=True)
        return
    if args.stage=="eval" and args.pruning_percent is None:
        for rate in ctx.percentages:subprocess.run(_child_args(args,"eval",rate),check=True)
        return
    if args.stage=="eval":
        from experiments_attrprune.resampled_evaluation import evaluate_policy
        result=asyncio.run(evaluate_policy(ctx,pruning_percent=args.pruning_percent,
            resume=args.resume,eval_batch_size=ctx.eval_batch_size,limit=ctx.limit))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        return
    with run_lock(ctx.paths.run_dir):
        if args.stage=="prepare":result=prepare_run(ctx)
        elif args.stage=="collect":result=asyncio.run(collect_run(ctx))
        elif args.stage=="attribute":result=attribute_run(ctx)
        elif args.stage=="aggregate":result=aggregate_run(ctx)
        elif args.stage=="policies":result=prepare_policies(ctx)
        else:result=summarize_run(ctx)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
