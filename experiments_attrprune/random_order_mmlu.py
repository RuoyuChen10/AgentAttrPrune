"""Isolated dev40 random-order dense calibration, without pruning.

The reference run supplies the exact questions, worker identities, prompts and
decoding settings. A frozen rollout plan separates execution order from matrix
coordinates. GPU stages run in separate processes under ``calibrate`` so the
generation model is released before the attribution model is loaded. The
default remains ten repeats per question; ``--repeats 1`` uses one rollout.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

from experiments_attrprune.common import (
    DEFAULT_RESULTS_ROOT, ExperimentPaths, config_fingerprint, load_json,
    reviewer_agent_ids_from_config, worker_specs_from_config, write_json,
)


NUM_QUESTIONS = 40
REPEATS = 10
NUM_ROUNDS = 2
NUM_WORKERS = 5
EXPERIMENT_VERSION = 1
DEFAULT_REFERENCE = DEFAULT_RESULTS_ROOT / "mmlu/qwen3_8b/dev40_correctness_aware"
DEFAULT_RUN = DEFAULT_RESULTS_ROOT / "mmlu/qwen3_8b/dev40_random_order10_correctness_aware"
STAGES = ("collect", "attribute", "aggregate", "visualize", "calibrate")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def make_rollout_plan(
    agents: list[str], sample_indices: list[int], *, repeats: int = REPEATS,
    num_rounds: int = NUM_ROUNDS, order_seed: int = 0,
) -> dict:
    if (not agents or len(set(agents)) != len(agents) or not sample_indices
            or len(set(sample_indices)) != len(sample_indices)
            or repeats < 1 or num_rounds < 1):
        raise ValueError("The rollout plan requires unique agents/questions and positive sizes")
    rollouts = []
    for sample_index in sample_indices:
        for repeat_index in range(repeats):
            orders = []
            for round_index in range(num_rounds):
                seed = int(digest([EXPERIMENT_VERSION, order_seed, sample_index,
                                   repeat_index, round_index]), 16)
                order = list(agents)
                random.Random(seed).shuffle(order)
                orders.append(order)
            rollouts.append({
                "rollout_id": f"sample_{sample_index:05d}_repeat_{repeat_index:02d}",
                "sample_index": sample_index, "repeat_index": repeat_index,
                "round_orders": orders,
            })
    return {
        "schema_version": 1, "experiment_version": EXPERIMENT_VERSION,
        "order_seed": order_seed, "agents": agents, "num_rounds": num_rounds,
        "num_questions": len(sample_indices), "repeats_per_question": repeats,
        "num_rollouts": len(rollouts), "rollouts": rollouts,
    }


@dataclass
class RunContext:
    paths: ExperimentPaths
    config: dict
    plan: dict
    samples: dict[int, dict]
    reference_aggregate: dict
    resume: bool


def read_reference(reference_run: Path) -> tuple[dict, dict[int, dict], dict, dict]:
    """Read the frozen original questions; do not resample or rewrite the run."""
    paths = ExperimentPaths(reference_run)
    config = load_json(paths.run_config)
    if config.get("fingerprint") != config_fingerprint(config):
        raise ValueError("Reference run fingerprint is invalid")
    if (config.get("benchmark") != "mmlu" or config.get("num_rounds") != NUM_ROUNDS
            or len(config.get("workers", [])) != NUM_WORKERS
            or config.get("splits", {}).get("calibration") != "dev"):
        raise ValueError("Reference must be a five-worker, two-round MMLU dev run")
    decoding = config["decoding"]
    if decoding.get("temperature") != 0 or decoding.get("enable_thinking") is not False:
        raise ValueError("Reference must use greedy generation with thinking disabled")
    reviewer_agent_ids_from_config(config)
    if [w.get("node_index") for w in config["workers"]] != list(range(NUM_WORKERS)):
        raise ValueError("Reference workers must have stable canonical node indices")
    selection_path = paths.calibration / "selection_manifest.json"
    selection = load_json(selection_path)
    records = selection["samples"]
    indices = [r["sample_index"] for r in records]
    if len(indices) != NUM_QUESTIONS or set(indices) != set(range(NUM_QUESTIONS)):
        raise ValueError("Reference selection must contain exactly dev indices 0..39")
    samples = {}
    for record in records:
        trace_path = Path(record["trace_json"])
        if file_sha(trace_path) != record["sha256"]:
            raise ValueError(f"Reference trace changed: {trace_path}")
        trace = load_json(trace_path)
        if (trace["config"].get("sample_index") != record["sample_index"]
                or trace["config"].get("fingerprint") != config["fingerprint"]):
            raise ValueError("Reference trace identity does not match its selection")
        samples[record["sample_index"]] = trace["sample"]
    manifest = load_json(paths.attribution / "manifest.json")
    aggregate_path = paths.attribution / "aggregate_edge_matrices.json"
    aggregate_sha = file_sha(aggregate_path)
    aggregate = load_json(aggregate_path)
    if (manifest.get("complete") is not True or manifest.get("aggregation_complete") is not True
            or manifest.get("failures") or manifest.get("aggregate_sha256") != aggregate_sha
            or aggregate.get("num_aggregated_samples") != NUM_QUESTIONS
            or aggregate.get("aggregation_scope") != "all"
            or aggregate.get("run_fingerprint") != config["fingerprint"]
            or aggregate.get("attribution_configuration", {}).get("attribution_mode") != "correctness-aware"):
        raise ValueError("Reference requires a complete correctness-aware all-dev40 aggregate")
    identity = {
        "run_dir": str(reference_run), "run_fingerprint": config["fingerprint"],
        "config_sha256": file_sha(paths.run_config),
        "selection_sha256": file_sha(selection_path), "aggregate_sha256": aggregate_sha,
    }
    return config, samples, aggregate, identity


def prepare_context(args: argparse.Namespace) -> RunContext:
    reference_run, run_dir = args.reference_run.resolve(), args.run_dir.resolve()
    if (run_dir == reference_run or reference_run in run_dir.parents
            or run_dir in reference_run.parents):
        raise ValueError("Use an independent output directory outside the reference run")
    reference, samples, aggregate, identity = read_reference(reference_run)
    agents = [w["agent_id"] for w in reference["workers"]]
    plan = make_rollout_plan(agents, sorted(samples), repeats=args.repeats,
                             num_rounds=NUM_ROUNDS, order_seed=args.order_seed)
    config = copy.deepcopy(reference)
    config["topology"]["worker_spatial"] = "independent uniform permutation per rollout/round; dense forward DAG"
    config["random_order_experiment"] = {
        "version": EXPERIMENT_VERSION, "num_questions": plan["num_questions"],
        "repeats_per_question": plan["repeats_per_question"], "order_seed": args.order_seed,
        "plan_sha256": digest(plan), "reference": identity,
        "generation_batch_size": 1, "attribution_batch_size": args.attr_batch_size,
        "spatial_input_order": "execution prefix", "temporal_input_order": "canonical agent identity",
        "reviewer_input_order": "canonical agent identity",
        "attribution_mode": "correctness-aware", "aggregation_scope": "all",
    }
    config["fingerprint"] = config_fingerprint(config)
    paths = ExperimentPaths(run_dir)
    if paths.run_config.exists():
        if load_json(paths.run_config) != config:
            raise ValueError("Run configuration/reference changed; choose a new run directory")
        if load_json(run_dir / "rollout_plan.json") != plan:
            raise ValueError("Frozen rollout plan changed; refusing to resample")
    elif args.stage not in {"collect", "calibrate"}:
        raise FileNotFoundError("Run collect before the requested stage")
    elif run_dir.exists() and any(p.name != ".random_order.lock" for p in run_dir.iterdir()):
        raise FileExistsError("Output directory is nonempty and has no random-order run configuration")
    return RunContext(paths, config, plan, samples, aggregate, args.resume)


@contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".random_order.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another stage is already using this random-order run") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def freeze_json(path: Path, value: dict) -> None:
    if path.exists():
        if load_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        write_json(path, value)


def trace_path_for(ctx: RunContext, rollout: dict) -> Path:
    return ctx.paths.traces / rollout["rollout_id"] / "trace.json"


def validate_trace(ctx: RunContext, rollout: dict, trace: dict) -> None:
    """Check identity, exact execution order and actual edge provenance on CPU."""
    config = trace.get("config", {})
    for key, value in {**ctx.config, **rollout}.items():
        if config.get(key) != value:
            raise ValueError(f"Trace configuration differs at {key}: {rollout['rollout_id']}")
    if (trace.get("schema_version") != 1 or trace.get("generation_complete") is not True
            or trace.get("token_ids_complete") is not True
            or trace.get("sample") != ctx.samples[rollout["sample_index"]]):
        raise ValueError("Trace is incomplete or its question changed")
    agents = ctx.plan["agents"]
    expected = [(r, a) for r, order in enumerate(rollout["round_orders"]) for a in order]
    expected.append((NUM_ROUNDS, "dense_reviewer"))
    events = trace.get("events", [])
    if [(e.get("round"), e.get("agent_id")) for e in events] != expected:
        raise ValueError("Trace node coverage/order differs from the frozen plan")
    prior = {}
    aliases = reviewer_agent_ids_from_config(ctx.config)
    for event_id, event in enumerate(events):
        r, target = expected[event_id]
        if (event.get("event_id") != event_id or event.get("input_ids") is None
                or event.get("generated_ids") is None or event.get("hit_max_new_tokens")):
            raise ValueError("Trace has invalid event IDs, missing token IDs or truncation")
        generated = np.asarray(event["generated_ids"])
        if generated.size >= int(ctx.config["decoding"]["max_new_tokens"]):
            raise ValueError("Truncated generation cannot enter formal attribution")
        if target == "dense_reviewer":
            wanted = [("spatial", NUM_ROUNDS - 1, a) for a in agents]
        else:
            order = rollout["round_orders"][r]
            wanted = [("spatial", r, a) for a in order[:order.index(target)]]
            if r:
                wanted += [("temporal", r - 1, a) for a in agents]
        incoming = event.get("incoming", [])
        if [(i.get("edge_type"), i.get("round"), i.get("agent_id")) for i in incoming] != wanted:
            raise ValueError("Trace incoming edges/order differ from the dense permutation")
        for item in incoming:
            source = prior[(item["round"], item["agent_id"])]
            if (item.get("event_id") != source["event_id"] or item.get("output") != source["output"]
                    or item.get("source_output") != source["output"]
                    or item.get("source_generated_ids") != source["generated_ids"]):
                raise ValueError("Incoming provenance does not match the original source")
            if target == "dense_reviewer" and item.get("reviewer_agent_id") != aliases[item["agent_id"]]:
                raise ValueError("Reviewer source aliases changed")
        prior[(r, target)] = event
    from AgentAttrPrune.datasets import evaluate_prediction
    outcome = evaluate_prediction("mmlu", events[-1]["output"], trace["sample"])
    if any(trace.get("final", {}).get(k) != outcome[k] for k in ("parsed_answer", "gold_answer", "correct")):
        raise ValueError("Trace final outcome does not match the recorded reviewer output")


def selection_snapshot(ctx: RunContext) -> dict:
    expected = {r["rollout_id"] for r in ctx.plan["rollouts"]}
    actual = {p.parent.name for p in ctx.paths.traces.glob("sample_*/trace.json")}
    if actual != expected:
        raise ValueError(f"Rollout traces incomplete/unexpected: missing={len(expected-actual)}, extra={sorted(actual-expected)}")
    records = []
    for rollout in ctx.plan["rollouts"]:
        path = trace_path_for(ctx, rollout)
        validate_trace(ctx, rollout, load_json(path))
        records.append({**rollout, "sample_id": rollout["rollout_id"],
                        "trace_json": str(path.resolve()), "sha256": file_sha(path)})
    return {"schema_version": 1, "run_fingerprint": ctx.config["fingerprint"],
            "plan_sha256": digest(ctx.plan), "num_questions": ctx.plan["num_questions"],
            "repeats_per_question": ctx.plan["repeats_per_question"],
            "num_rollouts": len(records), "samples": records}


def validate_selection(ctx: RunContext) -> dict:
    path = ctx.paths.calibration / "selection_manifest.json"
    saved = load_json(path)
    if saved != selection_snapshot(ctx):
        raise ValueError("Frozen selection/trace SHA changed; refusing partial or stale attribution")
    return saved


async def collect_run(ctx: RunContext, *, llm: Any = None) -> dict:
    from AgentAttrPrune.datasets import evaluate_prediction
    from AgentAttrPrune.graph.random_order import RandomOrderDenseGraph
    from AgentAttrPrune.tracing import GenerationTraceRecorder
    from experiments_attrprune.collect_correct import set_reproducible
    from experiments_attrprune.common import hf_model_reference

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
                graph = RandomOrderDenseGraph(
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
    return summary


def attribution_config(ctx: RunContext):
    from AgentAttrPrune.attribution.calibrator import AttributionConfig
    return AttributionConfig(attribution_mode="correctness-aware", aggregation_scope="all",
                             batch_size=ctx.config["random_order_experiment"]["attribution_batch_size"])


def attribute_run(ctx: RunContext, *, scorer: Any = None) -> dict:
    from AgentAttrPrune.attribution.calibrator import EdgeAttributionCalibrator
    validate_selection(ctx)
    if any(ctx.paths.attribution.glob("sample_*/event_*.json")) and not ctx.resume:
        raise FileExistsError("Attribution exists; use --resume or a new run directory")
    calibrator = EdgeAttributionCalibrator(
        ctx.paths.traces, ctx.paths.attribution, model_name=ctx.config["model_name"],
        config=attribution_config(ctx), scorer=scorer,
    )
    result = calibrator.run(attribute_only=True)
    validate_selection(ctx)
    return result["manifest"]


def event_digest(event: dict, trace: dict) -> str:
    # Match the existing correctness-aware cache identity without loading a scorer.
    payload = {k: event.get(k) for k in (
        "round", "agent_id", "incoming", "input_messages", "rendered_prompt", "input_ids",
        "attention_mask", "output", "generated_ids", "model_name", "llm_call",
    )}
    payload["gold_answer"] = str(trace["sample"]["answer"]).strip().upper()
    payload["run_fingerprint"] = trace["config"]["fingerprint"]
    return digest(payload)


def verified_sample_matrices(ctx: RunContext, selection: dict, manifest: dict) -> tuple[list[dict], dict]:
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


def aggregate_run(ctx: RunContext) -> dict:
    from AgentAttrPrune.attribution.edge import aggregate_sample_matrices
    manifest_path = ctx.paths.attribution / "manifest.json"
    manifest = load_json(manifest_path)
    # Invalidate publication before checks, so failed revalidation cannot leave a stale success flag.
    manifest["aggregation_complete"] = False
    manifest.pop("aggregate_sha256", None)
    write_json(manifest_path, manifest)
    selection = validate_selection(ctx)
    matrices, statuses = verified_sample_matrices(ctx, selection, manifest)
    aggregate = aggregate_sample_matrices(matrices, aggregation_scope="all")
    total = ctx.plan["num_rollouts"]
    counts = {edge: np.asarray(aggregate["metrics"]["token_ratio"][edge]["opportunity_count"])
              for edge in ("spatial", "temporal")}
    off_diagonal = ~np.eye(NUM_WORKERS, dtype=bool)
    for r in range(NUM_ROUNDS):
        spatial = counts["spatial"][r]
        if np.any(np.diag(spatial)) or not np.all((spatial + spatial.T)[off_diagonal] == total):
            raise ValueError("Spatial appearance counts violate the permutation invariant")
    if np.any(counts["temporal"][0]) or not np.all(counts["temporal"][1:] == total):
        raise ValueError("Temporal appearance counts violate the dense invariant")
    aggregate.update({
        "num_questions": ctx.plan["num_questions"],
        "repeats_per_question": ctx.plan["repeats_per_question"], "num_rollouts": total,
        "experiment": ctx.config["random_order_experiment"], "node_status_counts": statuses,
        "selection_sha256": file_sha(ctx.paths.calibration / "selection_manifest.json"),
        "spatial_off_diagonal_coverage_complete": bool(np.all(counts["spatial"][:, off_diagonal] > 0)),
        "unobserved_spatial_edges": [
            {"round": r, "source": ctx.plan["agents"][i], "target": ctx.plan["agents"][j]}
            for r in range(NUM_ROUNDS) for i in range(NUM_WORKERS) for j in range(NUM_WORKERS)
            if i != j and counts["spatial"][r, i, j] == 0
        ],
        "calibration": {"source": "all frozen random-order rollouts, including final-incorrect",
                        "run_dir": str(ctx.paths.run_dir), "selection": selection},
    })
    path = ctx.paths.attribution / "aggregate_edge_matrices.json"
    write_json(path, aggregate)
    manifest.update(aggregation_complete=True, aggregate_sha256=file_sha(path))
    write_json(manifest_path, manifest)
    return aggregate


def visualize_run(ctx: RunContext) -> dict:
    from experiments_attrprune.random_order_visualization import render_random_order_visualizations
    validate_selection(ctx)
    manifest = load_json(ctx.paths.attribution / "manifest.json")
    path = ctx.paths.attribution / "aggregate_edge_matrices.json"
    if (manifest.get("complete") is not True or manifest.get("failures")
            or manifest.get("aggregation_complete") is not True or manifest.get("aggregate_sha256") != file_sha(path)):
        raise ValueError("Complete, validated aggregation is required before visualization")
    aggregate = load_json(path)
    if aggregate.get("run_fingerprint") != ctx.config["fingerprint"]:
        raise ValueError("Aggregate belongs to a different run")
    return render_random_order_visualizations(aggregate, ctx.paths.attribution / "visualizations", ctx.reference_aggregate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="calibrate")
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Independent rollouts per question (default: 10); use a new run directory when changing it.")
    parser.add_argument("--attr-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.attr_batch_size < 1:
        parser.error("--attr-batch-size must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "stage": args.stage, "run_dir": str(ctx.paths.run_dir),
            "reference_run": str(args.reference_run.resolve()), "run_fingerprint": ctx.config["fingerprint"],
            "num_questions": ctx.plan["num_questions"],
            "repeats_per_question": ctx.plan["repeats_per_question"],
            "num_rollouts": ctx.plan["num_rollouts"], "order_seed": args.order_seed,
            "plan_sha256": digest(ctx.plan), "first_rollout": ctx.plan["rollouts"][0],
            "decoding": ctx.config["decoding"], "attribution_batch_size": args.attr_batch_size,
            "generates_pruning_policies": False,
        }, ensure_ascii=False, indent=2))
        return
    if args.stage in {"collect", "attribute", "calibrate"} and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly before a GPU stage")
    if args.stage == "calibrate":
        for stage in STAGES[:-1]:
            command = [sys.executable, "-m", "experiments_attrprune.random_order_mmlu", "--stage", stage,
                       "--reference-run", str(args.reference_run.resolve()), "--run-dir", str(args.run_dir.resolve()),
                       "--order-seed", str(args.order_seed), "--repeats", str(args.repeats),
                       "--attr-batch-size", str(args.attr_batch_size)]
            if args.resume:
                command.append("--resume")
            subprocess.run(command, check=True)
        return
    with run_lock(ctx.paths.run_dir):
        # Revalidate after acquiring the lock in case another invocation just finished.
        ctx = prepare_context(args)
        if args.stage == "collect":
            result = asyncio.run(collect_run(ctx))
        elif args.stage == "attribute":
            result = attribute_run(ctx)
        elif args.stage == "aggregate":
            result = aggregate_run(ctx)
        else:
            result = visualize_run(ctx)
        print(json.dumps({"stage": args.stage, "run_dir": str(ctx.paths.run_dir),
                          "complete": result.get("complete", True)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
