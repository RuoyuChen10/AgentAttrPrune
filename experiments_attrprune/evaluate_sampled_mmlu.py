"""Evaluate independently sampled DAGs from frozen, round-specific attribution.

Preparation is CPU-only. Each selected question gets one deterministic sampling
seed, shared across pruning rates/scopes, and fresh Bernoulli draws per round.
Completed traces are the durable checkpoint; runtime failures never become
incorrect predictions or permit publication of a complete evaluation summary.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from AgentAttrPrune.datasets import MMLUDataset, evaluate_prediction
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import random_order_mmlu as source_runner
from experiments_attrprune.common import (
    benchmark_data_identity, config_fingerprint, dataset_identities_match,
    hf_model_reference, load_json, reviewer_agent_ids_from_config,
    validate_prompt_contract, worker_specs_from_config, write_json,
)
from experiments_attrprune.evaluate_mmlu import (
    _mapping, _trace_final, _write_jsonl_atomic, set_reproducible,
)


DEFAULT_RUN = source_runner.DEFAULT_RUN.with_name("dev40_random_order1_correctness_aware")
EVALUATION_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def sampling_seed(seed: int, sample_index: int) -> int:
    """Share question seeds across rates; skipped draws may change edge pairing."""
    return int(digest({"namespace": "mmlu_sampled_dag_v1", "seed": int(seed),
                       "sample_index": int(sample_index)})[:15], 16)


@dataclass
class EvaluationContext:
    output_dir: Path
    source_config: dict
    config: dict
    policy: dict
    selection: dict
    samples: dict[int, dict]
    source_hashes: dict[str, str]


def _source(run_dir: Path) -> tuple[dict, dict, dict[str, str]]:
    """Use the existing read-only verifier, never its manifest-writing stage."""
    config = load_json(run_dir / "run_config.json")
    settings = config.get("random_order_experiment", {})
    if (settings.get("num_questions") != 40 or settings.get("repeats_per_question") != 1
            or settings.get("attribution_mode") != "correctness-aware"):
        raise ValueError("Sampling requires the complete random-order once40 source")
    args = source_runner.parse_args([
        "--run-dir", str(run_dir), "--reference-run", settings["reference"]["run_dir"],
        "--stage", "visualize", "--repeats", "1", "--resume",
        "--order-seed", str(settings["order_seed"]),
        "--attr-batch-size", str(settings["attribution_batch_size"]),
    ])
    ctx = source_runner.prepare_context(args)
    paths = [ctx.paths.run_config, run_dir / "rollout_plan.json",
             ctx.paths.calibration / "selection_manifest.json",
             ctx.paths.attribution / "manifest.json",
             ctx.paths.attribution / "aggregate_edge_matrices.json"]
    hashes = {str(path): source_runner.file_sha(path) for path in paths}
    manifest, aggregate = load_json(paths[-2]), load_json(paths[-1])
    if (manifest.get("complete") is not True or manifest.get("aggregation_complete") is not True
            or manifest.get("failures") or manifest.get("skipped_incorrect")
            or manifest.get("aggregate_sha256") != hashes[str(paths[-1])]
            or aggregate.get("num_aggregated_samples") != 40
            or aggregate.get("run_fingerprint") != config["fingerprint"]
            or aggregate.get("aggregation_scope") != "all"):
        raise ValueError("Source attribution/aggregate is incomplete, stale or failed")
    selection = source_runner.validate_selection(ctx)
    matrices, statuses = source_runner.verified_sample_matrices(ctx, selection, manifest)
    from AgentAttrPrune.attribution.edge import aggregate_sample_matrices
    rebuilt = aggregate_sample_matrices(matrices, aggregation_scope="all")
    if (len(matrices) != 40 or sum(statuses.values()) != 440
            or rebuilt["metrics"] != aggregate["metrics"]
            or statuses != aggregate.get("node_status_counts")):
        raise ValueError("Source aggregate differs from its complete 40-sample attribution")
    spatial = np.asarray(aggregate["metrics"]["token_ratio"]["spatial"]["opportunity_count"])
    temporal = np.asarray(aggregate["metrics"]["token_ratio"]["temporal"]["opportunity_count"])
    off = ~np.eye(len(ctx.plan["agents"]), dtype=bool)
    if (not np.all(spatial[:, off] > 0)
            or not np.all((spatial + spatial.transpose(0, 2, 1))[:, off] == 40)
            or np.any(spatial[:, np.eye(len(off), dtype=bool)])
            or np.any(temporal[0]) or not np.all(temporal[1:] == 40)):
        raise ValueError("Source opportunity counts violate random-order dense coverage")
    _check_hashes(hashes)
    return config, aggregate, hashes


def _check_hashes(hashes: Mapping[str, str]) -> None:
    for path, expected in hashes.items():
        if source_runner.file_sha(Path(path)) != expected:
            raise ValueError(f"Frozen source artifact changed: {path}")


def _reuse_frozen_policy(output: Path, rebuilt: dict) -> dict:
    """Keep the policy identity across CPU vector-math implementations.

    AVX2 and AVX512 can differ by one float32 ULP in the batched sigmoid
    arrays stored for inspection. Sampling itself uses the unchanged logits
    and scalar sigmoid. Validate the frozen hash and every operational field
    exactly, allowing only this rounding difference in the descriptive arrays.
    Never rewrite a checkpoint or relax the subsequent config/selection check.
    """
    policy_path = output / "sampling_policy.json"
    config_path = output / "evaluation_config.json"
    if not policy_path.exists() or not config_path.exists():
        return rebuilt
    frozen = load_json(policy_path)
    config = load_json(config_path)
    if digest(frozen) != config.get("sampling_policy_sha256"):
        raise ValueError("Evaluation checkpoint configuration changed: sampling_policy.json (hash mismatch)")
    if frozen == rebuilt:
        return frozen
    compatible = copy.deepcopy(rebuilt)
    try:
        for edge in ("spatial", "temporal"):
            expected = np.asarray(rebuilt["parameters"][edge]["probabilities"], dtype=np.float32)
            stored = np.asarray(frozen["parameters"][edge]["probabilities"], dtype=np.float64)
            effective = np.asarray(frozen["parameters"][edge]["effective_probabilities"], dtype=np.float64)
            lower = np.nextafter(expected, np.float32(-np.inf)).astype(np.float64)
            upper = np.nextafter(expected, np.float32(np.inf)).astype(np.float64)
            if (stored.shape != expected.shape or effective.shape != expected.shape
                    or not np.all(np.isfinite(stored))
                    or not np.array_equal(stored, stored.astype(np.float32).astype(np.float64))
                    or not np.all((lower <= stored) & (stored <= upper))
                    or not np.array_equal(effective, stored * np.asarray(rebuilt["masks"][edge]))):
                raise ValueError("probabilities differ beyond float32 rounding")
            for key in ("probabilities", "effective_probabilities"):
                compatible["parameters"][edge][key] = frozen["parameters"][edge][key]
        if compatible != frozen:
            raise ValueError("operational policy fields changed")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Evaluation checkpoint configuration changed: sampling_policy.json ({error})") from error
    return frozen


def prepare_context(args: argparse.Namespace, *, eval_batch_size: int = 1,
                    evaluation_version: int | str = EVALUATION_VERSION) -> EvaluationContext:
    # Defaults preserve every field/fingerprint of the original serial runner.
    # The additive batched runner freezes its actual batch/version before the
    # normal checkpoint compatibility checks, including on resume.
    if (isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int)
            or eval_batch_size < 1):
        raise ValueError("Evaluation batch size must be positive")
    from AgentAttrPrune.pruning.sampled import prepare_sampling_policy
    run_dir = args.run_dir.resolve()
    config, aggregate, hashes = _source(run_dir)
    validate_prompt_contract(config)
    if (config["splits"]["evaluation"] != args.split
            or config["decoding"]["enable_thinking"] is not False
            or config["decoding"]["temperature"] != 0):
        raise ValueError("Evaluation must preserve the source split and greedy decoding")
    expected = config.get("dataset")
    if not expected or args.split not in expected.get("splits", {}):
        raise ValueError("Source must fingerprint the evaluation dataset")
    data_dir = Path(expected.get("requested_data_dir") or expected["data_root"])
    actual = benchmark_data_identity("mmlu", data_dir, splits=expected["splits"],
                                     shuffle_seed=int(expected["shuffle_seed"]))
    if not dataset_identities_match(expected, actual):
        raise ValueError("MMLU dataset fingerprint changed")
    dataset = MMLUDataset(split=args.split, data_dir=data_dir, download=False,
                          shuffle_seed=int(expected["shuffle_seed"]))
    if args.limit > len(dataset):
        raise ValueError(f"Requested {args.limit} questions but dataset has {len(dataset)}")
    samples = {i: dataset.record_to_sample(dataset[i]) for i in range(args.limit)}
    selection = {"schema_version": 1, "benchmark": "mmlu", "split": args.split,
                 "dataset_size": len(dataset), "num_samples": len(samples),
                 "dataset": actual, "samples": [
                     {"sample_index": i, "sample_sha256": digest(sample),
                      "sampling_seed": sampling_seed(args.seed, i)}
                     for i, sample in samples.items()]}
    policy = prepare_sampling_policy(aggregate, pruning_rate=args.pruning_percent / 100,
                                     pruning_scope=args.pruning_scope, round_policy="per_round")
    output = (args.output_dir or run_dir / "evaluation" / "agentprune_sampling" /
              "per_round_sigmoid" / args.pruning_scope /
              f"prune_{args.pruning_percent:02d}pct").resolve()
    if (output == run_dir or output in run_dir.parents
            or (run_dir in output.parents and run_dir / "evaluation" not in output.parents)):
        raise ValueError("Use an independent evaluation directory")
    reference = Path(config["random_order_experiment"]["reference"]["run_dir"]).resolve()
    if output == reference or reference in output.parents or output in reference.parents:
        raise ValueError("Evaluation must not write into the reference run")
    policy = _reuse_frozen_policy(output, policy)
    state = {
        "schema_version": 1, "evaluation_version": evaluation_version,
        "source_run": str(run_dir), "source_run_fingerprint": config["fingerprint"],
        "source_artifact_sha256": hashes, "source_config": config,
        "output_dir": str(output), "benchmark": "mmlu", "split": args.split,
        "limit": args.limit, "seed": args.seed, "round_policy": "per_round",
        "pruning_scope": args.pruning_scope, "pruning_percent": args.pruning_percent,
        "sampling_policy_sha256": digest(policy), "selection_sha256": digest(selection),
        "generation_batch_size": eval_batch_size, "eval_batch_size": eval_batch_size, "save_traces": True,
        "truncation_policy": "evaluate recorded final answer; report all truncated events",
        "runtime_failure_policy": "retry on resume; never count as incorrect",
        "sampling_seed_rule": "sha256(mmlu_sampled_dag_v1,experiment_seed,sample_index); excludes pruning settings",
    }
    state["fingerprint"] = config_fingerprint(state)
    ctx = EvaluationContext(output, config, state, policy, selection, samples, hashes)
    _validate_prepared(ctx)
    return ctx


def _frozen_files(ctx: EvaluationContext) -> dict[str, dict]:
    return {"evaluation_config.json": ctx.config, "sampling_policy.json": ctx.policy,
            "selection_manifest.json": ctx.selection}


def _validate_prepared(ctx: EvaluationContext) -> None:
    output = ctx.output_dir
    if not output.exists():
        return
    entries = {path.name for path in output.iterdir()} - {".random_order.lock"}
    allowed = set(_frozen_files(ctx))
    config_path = output / "evaluation_config.json"
    if entries and not config_path.exists():
        raise FileExistsError("Nonempty evaluation directory has no frozen configuration")
    for name, expected in _frozen_files(ctx).items():
        path = output / name
        if path.exists() and load_json(path) != expected:
            raise ValueError(f"Evaluation checkpoint configuration changed: {name}")
        if not path.exists() and entries - allowed:
            raise ValueError(f"Incomplete frozen evaluation metadata: {name}")


def _prepare(ctx: EvaluationContext) -> None:
    _check_hashes(ctx.source_hashes)
    _validate_prepared(ctx)
    # Configuration is the ownership marker. A crash before the next two files
    # is safely resumable because only matching preparation metadata may exist.
    for name, payload in _frozen_files(ctx).items():
        source_runner.freeze_json(ctx.output_dir / name, payload)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace_path(ctx: EvaluationContext, index: int) -> Path:
    return ctx.output_dir / "traces" / f"sample_{index:05d}" / "trace.json"


def _mask_counts(realized: Mapping) -> dict:
    counts = {edge: np.asarray(realized["masks"][edge], dtype=int).sum(axis=(1, 2)).tolist()
              for edge in ("spatial", "temporal")}
    counts["worker_active"] = sum(sum(values) for values in counts.values())
    return counts


def _validate_trace(ctx: EvaluationContext, index: int, trace: dict, realized: dict) -> dict:
    """Validate successful execution even when its generation was truncated."""
    config = trace.get("config", {})
    if (trace.get("schema_version") != 1 or trace.get("token_ids_complete") is not True
            or trace.get("sample") != ctx.samples[index]
            or config.get("evaluation_fingerprint") != ctx.config["fingerprint"]
            or config.get("sample_index") != index
            or config.get("sampling_seed") != sampling_seed(ctx.config["seed"], index)
            or config.get("realized_graph") != realized
            or config.get("round_orders") != realized["round_orders"]):
        raise ValueError("Evaluation trace identity, sampled graph or token IDs changed")
    for key, value in ctx.source_config.items():
        if config.get(key) != value:
            raise ValueError(f"Evaluation trace source configuration changed: {key}")
    agents = [worker["agent_id"] for worker in ctx.source_config["workers"]]
    rounds = ctx.source_config["num_rounds"]
    expected = [(r, agent) for r, order in enumerate(realized["round_orders"]) for agent in order]
    expected.append((rounds, "dense_reviewer"))
    events = trace.get("events", [])
    if [(event.get("round"), event.get("agent_id")) for event in events] != expected:
        raise ValueError("Evaluation trace does not contain every scheduled node")
    aliases = reviewer_agent_ids_from_config(ctx.source_config)
    prior = {}
    for event_id, event in enumerate(events):
        r, agent = expected[event_id]
        if event.get("event_id") != event_id or event.get("sample_index") != index:
            raise ValueError("Evaluation event identity changed")
        if agent == "dense_reviewer":
            wanted = [("spatial", rounds - 1, a) for a in agents]
        else:
            j = agents.index(agent)
            wanted = [(edge, r if edge == "spatial" else r - 1, a)
                      for edge in ("spatial", "temporal") for i, a in enumerate(agents)
                      if realized["masks"][edge][r][i][j]]
        incoming = event.get("incoming", [])
        if [(item.get("edge_type"), item.get("round"), item.get("agent_id")) for item in incoming] != wanted:
            raise ValueError("Trace incoming edges differ from the sampled mask")
        for item in incoming:
            source = prior[(item["round"], item["agent_id"])]
            if (item.get("event_id") != source["event_id"]
                    or item.get("output") != source["output"]
                    or item.get("source_output") != source["output"]
                    or item.get("source_generated_ids") != source["generated_ids"]):
                raise ValueError("Evaluation trace incoming provenance changed")
            if agent == "dense_reviewer" and item.get("reviewer_agent_id") != aliases[item["agent_id"]]:
                raise ValueError("Reviewer input aliases changed")
        prior[(r, agent)] = event
    outcome = evaluate_prediction("mmlu", events[-1]["output"], ctx.samples[index])
    record = trace.get("evaluation_record", {})
    for key in ("parsed_answer", "gold_answer", "correct"):
        if trace.get("final", {}).get(key) != outcome[key] or record.get(key) != outcome[key]:
            raise ValueError("Recorded prediction differs from the final reviewer output")
    counts = _mask_counts(realized)
    truncated = [event["event_id"] for event in events if event.get("hit_max_new_tokens")]
    if (record.get("sample_index") != index or record.get("output") != events[-1]["output"]
            or record.get("evaluation_fingerprint") != ctx.config["fingerprint"]
            or record.get("sampled_edge_counts") != counts
            or record.get("sampling_seed") != config["sampling_seed"]
            or record.get("realized_graph") != realized
            or record.get("truncated_event_ids") != truncated
            or record.get("trace_json") != str(_trace_path(ctx, index))
            or record.get("metrics") != trace["final"].get("metrics")
            or record.get("edge_counts") != trace["final"].get("edge_counts")
            or record["edge_counts"].get("worker_active") != counts["worker_active"]):
        raise ValueError("Prediction metadata differs from its execution trace")
    for key in ("sample_elapsed_seconds", "evaluation_elapsed_seconds"):
        if not isinstance(record.get(key), (int, float)) or not np.isfinite(record[key]) or record[key] < 0:
            raise ValueError("Invalid evaluation duration")
    return record


def _resume_records(ctx: EvaluationContext, *, resume: bool) -> dict[int, dict]:
    from AgentAttrPrune.pruning.sampled import sample_graph
    path = ctx.output_dir / "predictions.jsonl"
    saved = _read_jsonl(path)
    prior = {int(record["sample_index"]): record for record in saved}
    if len(prior) != len(saved) or set(prior) - set(ctx.samples):
        raise ValueError("Duplicate or foreign prediction checkpoint")
    actual = set((ctx.output_dir / "traces").glob("sample_*/trace.json"))
    expected = {_trace_path(ctx, i) for i in ctx.samples}
    if actual - expected:
        raise ValueError("Unexpected trace outside the frozen selection")
    if (saved or actual or _read_jsonl(ctx.output_dir / "attempts.jsonl")) and not resume:
        raise FileExistsError("Evaluation has previous attempts; use --resume")
    records = {}
    for index in ctx.samples:
        trace_path = _trace_path(ctx, index)
        if not trace_path.exists():
            if index in prior:
                raise ValueError("Prediction checkpoint has no completed trace")
            continue
        realized = sample_graph(ctx.policy, seed=sampling_seed(ctx.config["seed"], index))
        record = _validate_trace(ctx, index, load_json(trace_path), realized)
        record = {**record, "trace_sha256": source_runner.file_sha(trace_path)}
        if index in prior and prior[index] != record:
            raise ValueError("Prediction checkpoint or trace SHA changed")
        records[index] = record
    return records


def _summary(ctx: EvaluationContext, records: dict[int, dict], attempts: list[dict]) -> dict:
    items = list(records.values())
    totals = {key: sum(float(item["metrics"].get(key, 0)) for item in items)
              for key in ("llm_calls", "prompt_tokens", "generated_tokens", "elapsed_seconds")}
    counts = np.asarray([[item["sampled_edge_counts"][edge] for edge in ("spatial", "temporal")]
                         for item in items], dtype=int)
    failed = [attempt for attempt in attempts if attempt["status"] == "failed"]
    correct = sum(bool(item["correct"]) for item in items)
    active = counts.sum(axis=(1, 2))
    return {
        "schema_version": 1, "complete": True, "evaluation_fingerprint": ctx.config["fingerprint"],
        "source_run_fingerprint": ctx.source_config["fingerprint"],
        "round_policy": "per_round", "pruning_scope": ctx.config["pruning_scope"],
        "pruning_percent": ctx.config["pruning_percent"], "benchmark": "mmlu",
        "split": ctx.config["split"], "total": len(items), "correct": correct,
        "accuracy": correct / len(items), **{key: int(totals[key]) for key in ("llm_calls", "prompt_tokens", "generated_tokens")},
        "graph_elapsed_seconds": totals["elapsed_seconds"],
        "elapsed_seconds": sum(item["evaluation_elapsed_seconds"] for item in items),
        "failed_attempts": len(failed),
        "failed_attempt_elapsed_seconds": sum(item.get("elapsed_seconds", 0) for item in failed),
        "failed_attempt_usage": {key: sum(item.get("usage", {}).get(key, 0) for item in failed)
                                 for key in ("llm_calls", "prompt_tokens", "generated_tokens")},
        "truncated_samples": sum(bool(item["truncated_event_ids"]) for item in items),
        "truncated_events": sum(len(item["truncated_event_ids"]) for item in items),
        "edge_counts": {"orientation": "[edge_type:spatial,temporal][round]",
                        "total": counts.sum(axis=0).tolist(), "mean": counts.mean(axis=0).tolist(),
                        "minimum": counts.min(axis=0).tolist(), "maximum": counts.max(axis=0).tolist(),
                        "worker_active_total": int(active.sum()), "worker_active_mean": float(active.mean()),
                        "worker_active_minimum": int(active.min()), "worker_active_maximum": int(active.max()),
                        "worker_candidate_per_sample": items[0]["edge_counts"]["worker_candidate"]},
        "sampling_policy_sha256": ctx.config["sampling_policy_sha256"],
        "selection_sha256": ctx.config["selection_sha256"],
        "predictions_sha256": source_runner.file_sha(ctx.output_dir / "predictions.jsonl"),
        "attempts_sha256": source_runner.file_sha(ctx.output_dir / "attempts.jsonl"),
        "predictions_jsonl": str(ctx.output_dir / "predictions.jsonl"),
        "attempts_jsonl": str(ctx.output_dir / "attempts.jsonl"),
        "traces_dir": str(ctx.output_dir / "traces"), "eval_batch_size": 1,
    }


async def evaluate(args: argparse.Namespace, *, llm: Any = None) -> dict:
    ctx = prepare_context(args)
    overview = {"output_dir": str(ctx.output_dir), "fingerprint": ctx.config["fingerprint"],
                "num_samples": len(ctx.samples), "round_policy": "per_round",
                "pruning_scope": args.pruning_scope, "pruning_percent": args.pruning_percent,
                "policy_sha256": digest(ctx.policy)}
    if args.dry_run:
        return {**overview, "dry_run": True}
    with source_runner.run_lock(ctx.output_dir):
        _prepare(ctx)
        if args.prepare_only:
            return {**overview, "prepared": True}
        records = _resume_records(ctx, resume=args.resume)
        attempts_path = ctx.output_dir / "attempts.jsonl"
        attempts = _read_jsonl(attempts_path)
        for item in attempts:
            if (item.get("evaluation_fingerprint") != ctx.config["fingerprint"]
                    or item.get("sample_index") not in ctx.samples
                    or item.get("status") not in {"started", "completed", "failed", "recovered"}):
                raise ValueError("Invalid attempt log identity or status")
        # A complete trace may have been committed just before the process
        # stopped. Preserve that successful work and explicitly close its audit
        # trail without replaying the graph.
        for index, record in records.items():
            if not any(item["sample_index"] == index
                       and item["status"] in {"completed", "recovered"}
                       and item.get("trace_sha256") == record["trace_sha256"] for item in attempts):
                attempts.append({"evaluation_fingerprint": ctx.config["fingerprint"],
                    "sample_index": index, "status": "recovered",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sampling_seed": record["sampling_seed"], "realized_graph": record["realized_graph"],
                    "trace_sha256": record["trace_sha256"], "correct": record["correct"],
                    "elapsed_seconds": record["evaluation_elapsed_seconds"]})
        if (ctx.output_dir / "evaluation_summary.json").exists() and len(records) != len(ctx.samples):
            raise ValueError("Published evaluation summary has missing checkpoints")
        _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", list(records.values()))
        _write_jsonl_atomic(attempts_path, attempts)
        pending = [i for i in ctx.samples if i not in records]
        if pending and llm is None:
            if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for evaluation")
            os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
            os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from AgentAttrPrune.llm import HFChat
            llm = HFChat(hf_model_reference(ctx.source_config["model_name"]))
        from AgentAttrPrune.graph.sampled_dag import SampledDAGGraph
        from AgentAttrPrune.pruning.sampled import sample_graph
        print(f"Validated {len(records)}/{len(ctx.samples)} completed evaluations; {len(pending)} remain.", flush=True)
        for index in pending:
            seed = sampling_seed(args.seed, index)
            realized = sample_graph(ctx.policy, seed=seed)
            set_reproducible(seed % (2**32))
            trace_path = _trace_path(ctx, index)
            recorder = GenerationTraceRecorder(trace_path, config={
                **ctx.source_config, "phase": "evaluation", "sample_index": index,
                "evaluation_mode": "agentprune_sampling", "evaluation_fingerprint": ctx.config["fingerprint"],
                "sampling_seed": seed, "realized_graph": realized,
                "round_orders": realized["round_orders"], "eval_batch_size": 1,
            }, sample=ctx.samples[index])
            attempt_number = 1 + sum(a["sample_index"] == index and a["status"] == "started" for a in attempts)
            base = {"evaluation_fingerprint": ctx.config["fingerprint"], "sample_index": index,
                    "attempt_number": attempt_number, "sampling_seed": seed,
                    "realized_graph": realized, "timestamp": datetime.now(timezone.utc).isoformat()}
            attempts.append({**base, "status": "started"})
            _write_jsonl_atomic(attempts_path, attempts)
            started = time.perf_counter()
            try:
                decoding = ctx.source_config["decoding"]
                graph = SampledDAGGraph(llm, worker_specs_from_config(ctx.source_config),
                    num_rounds=ctx.source_config["num_rounds"], realized_graph=realized,
                    temperature=decoding["temperature"], max_new_tokens=decoding["max_new_tokens"],
                    reviewer_agent_ids=reviewer_agent_ids_from_config(ctx.source_config))
                result = await graph.run_sample(ctx.samples[index], recorder=recorder,
                                                phase="evaluation", sample_index=index)
                outcome = evaluate_prediction("mmlu", result.final_answer, ctx.samples[index])
                elapsed = time.perf_counter() - started
                record = {"sample_index": index, "evaluation_fingerprint": ctx.config["fingerprint"],
                    "sampling_seed": seed, "realized_graph": realized,
                    "subject": ctx.samples[index].get("subject", ""),
                    "gold_answer": outcome["gold_answer"], "parsed_answer": outcome["parsed_answer"],
                    "correct": outcome["correct"], "output": result.final_answer,
                    "metrics": _mapping(result.metrics), "edge_counts": _mapping(result.edge_counts),
                    "sampled_edge_counts": _mask_counts(realized),
                    "sample_elapsed_seconds": elapsed, "evaluation_elapsed_seconds": elapsed,
                    "truncated_event_ids": [e["event_id"] for e in recorder.events if e["hit_max_new_tokens"]],
                    "trace_json": str(trace_path), "evaluation_details": outcome.get("details") or {}}
                trace = recorder.payload(_trace_final(result, gold=outcome["gold_answer"],
                    parsed=outcome["parsed_answer"], correct=outcome["correct"]))
                trace["evaluation_record"] = record
                _validate_trace(ctx, index, trace, realized)
                write_json(trace_path, trace)
                records[index] = {**record, "trace_sha256": source_runner.file_sha(trace_path)}
                _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", [records[i] for i in sorted(records)])
                attempts.append({**base, "status": "completed", "elapsed_seconds": elapsed,
                                 "correct": outcome["correct"], "trace_sha256": records[index]["trace_sha256"]})
                _write_jsonl_atomic(attempts_path, attempts)
            except BaseException as error:
                partial = ctx.output_dir / "failed_attempts" / f"sample_{index:05d}_attempt_{attempt_number:04d}.json"
                write_json(partial, recorder.payload({"runtime_error": f"{type(error).__name__}: {error}"}))
                attempts.append({**base, "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter() - started, "usage": recorder.summary,
                    "partial_trace_json": str(partial)})
                _write_jsonl_atomic(attempts_path, attempts)
                write_json(ctx.output_dir / "progress.json", {"complete": False,
                    "completed_samples": len(records), "total_samples": len(ctx.samples),
                    "failed_sample_index": index, "error": str(error)})
                raise
            done = len(records)
            elapsed_total = sum(item["evaluation_elapsed_seconds"] for item in records.values())
            eta = elapsed_total / done * (len(ctx.samples) - done)
            progress = {"complete": False, "completed_samples": done, "total_samples": len(ctx.samples),
                        "accuracy_so_far": sum(bool(item["correct"]) for item in records.values()) / done,
                        "successful_elapsed_seconds": elapsed_total, "eta_seconds": eta}
            write_json(ctx.output_dir / "progress.json", progress)
            print(f"[{done}/{len(ctx.samples)}] sample={index} correct={bool(record['correct'])} "
                  f"edges={record['sampled_edge_counts']['worker_active']} "
                  f"elapsed={record['sample_elapsed_seconds']:.1f}s ETA={eta / 60:.1f}min", flush=True)
        if len(records) != len(ctx.samples):
            raise RuntimeError("Evaluation incomplete; refusing to publish accuracy")
        _check_hashes(ctx.source_hashes)
        summary = _summary(ctx, records, attempts)
        write_json(ctx.output_dir / "evaluation_summary.json", summary)
        write_json(ctx.output_dir / "progress.json", {"complete": True,
                   "completed_samples": len(records), "total_samples": len(ctx.samples), "eta_seconds": 0})
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pruning-scope", choices=("global", "uniform"), required=True)
    parser.add_argument("--pruning-percent", type=int, choices=(50, 60), required=True)
    parser.add_argument("--round-policy", choices=("per_round",), default="per_round")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--limit", type=int, default=1531)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.dry_run and args.prepare_only:
        parser.error("Choose only one of --dry-run and --prepare-only")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    result = asyncio.run(evaluate(parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
