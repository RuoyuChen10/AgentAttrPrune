"""Independent 80% reevaluation that skips an empty final worker round.

The original reordered evaluator and its checkpoints remain unchanged. This
module uses separate output directories and fingerprints for the new rule.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import time
from typing import Any, Sequence

import numpy as np

from AgentAttrPrune.datasets import MMLUDataset, evaluate_prediction
from AgentAttrPrune.graph.fixed_order_skip_empty import execution_contract_for_policy
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune.common import (
    benchmark_data_identity, config_fingerprint, dataset_identities_match,
    hf_model_reference, load_json, reviewer_agent_ids_from_config,
    worker_specs_from_config, write_json,
)
from experiments_attrprune.evaluate_mmlu import _mapping, _trace_final, _write_jsonl_atomic, set_reproducible
from experiments_attrprune.evaluate_sampled_mmlu import digest, _read_jsonl
from experiments_attrprune.random_order_mmlu import file_sha, freeze_json, run_lock


@dataclass
class EvaluationState:
    run: Any
    output: Path
    config: dict
    policy: dict
    selection: dict
    samples: dict[int, dict]
    edge_counts: dict


def _edge_counts(policy: dict) -> dict:
    n, rounds = len(policy["agents"]), policy["num_rounds"]
    spatial = int(np.asarray(policy["masks"]["spatial"]).sum())
    temporal = int(np.asarray(policy["masks"]["temporal"]).sum())
    spatial_candidates = rounds * n * (n - 1) // 2
    temporal_candidates = (rounds - 1) * n * n
    candidates = spatial_candidates + temporal_candidates
    return {"matrix_orientation": "[round][source][target]",
            "spatial": {"active": spatial, "candidate": spatial_candidates},
            "temporal": {"active": temporal, "candidate": temporal_candidates},
            "worker_active": spatial + temporal, "worker_candidate": candidates,
            "worker_retention": (spatial + temporal) / candidates,
            "reviewer_dense_edges": n}


def prepare_evaluation(ctx: Any, *, pruning_percent: int, eval_batch_size: int, limit: int) -> EvaluationState:
    from experiments_attrprune.reordered_mmlu import load_policies
    if pruning_percent != 80:
        raise ValueError("This independent reevaluation is restricted to 80% pruning")
    if eval_batch_size < 1 or limit < 1:
        raise ValueError("Evaluation batch size and limit must be positive")
    if (ctx.config["decoding"]["temperature"] != 0
            or ctx.config["decoding"]["enable_thinking"] is not False):
        raise ValueError("Ordered evaluation requires greedy generation with thinking disabled")
    # The loader checks the complete calibration/projection before validating
    # the frozen policies. Avoid repeating the 440-node source audit here.
    policies = load_policies(ctx)
    if pruning_percent not in policies:
        raise ValueError("Requested pruning percentage has no frozen policy")
    policy = policies[pruning_percent]
    RoundPolicy.from_dict(policy)
    contract = execution_contract_for_policy(policy)
    expected = ctx.config["dataset"]
    data_dir = Path(expected.get("requested_data_dir") or expected["data_root"])
    actual = benchmark_data_identity("mmlu", data_dir, splits=expected["splits"],
                                     shuffle_seed=int(expected["shuffle_seed"]))
    if not dataset_identities_match(expected, actual):
        raise ValueError("MMLU dataset fingerprint changed")
    dataset = MMLUDataset(split="val", data_dir=data_dir, download=False,
                          shuffle_seed=int(expected["shuffle_seed"]))
    if limit > len(dataset):
        raise ValueError(f"Requested {limit} questions but validation contains {len(dataset)}")
    samples = {index: dataset.record_to_sample(dataset[index]) for index in range(limit)}
    selection = {"schema_version": 1, "benchmark": "mmlu", "split": "val",
                 "dataset": actual, "dataset_size": len(dataset), "num_samples": limit,
                 "samples": [{"sample_index": index, "sample_sha256": digest(sample)}
                             for index, sample in samples.items()]}
    output = (ctx.paths.evaluation / f"global_prune_{pruning_percent:02d}pct_val_skip_empty_round1").resolve()
    config = {"schema_version": 1, "evaluation_version": 1,
              "mode": "fixed_order_static_skip_empty_round1", "branch": ctx.branch,
              "execution_contract": contract,
              "baseline_evaluation_dir": str((ctx.paths.evaluation / "global_prune_80pct_val").resolve()),
              "run_fingerprint": ctx.config["fingerprint"], "run_config": ctx.config,
              "source_artifact_sha256": ctx.source_hashes,
              "order_plan_sha256": digest(ctx.order_plan), "round_orders": ctx.order_plan["round_orders"],
              "policy_sha256": digest(policy), "selection_sha256": digest(selection),
              "pruning_scope": "global", "pruning_percent": pruning_percent,
              "benchmark": "mmlu", "split": "val", "limit": limit,
              "seed": int(ctx.config["seed"]), "eval_batch_size": eval_batch_size,
              "save_traces": True, "output_dir": str(output),
              "edge_use": "all retained edges; no Bernoulli sampling",
              "truncation_policy": "score the recorded final answer; report truncated events",
              "runtime_failure_policy": "retry on resume; never count as incorrect"}
    config["fingerprint"] = config_fingerprint(config)
    state = EvaluationState(ctx, output, config, policy, selection, samples, _edge_counts(policy))
    _validate_frozen(state)
    return state


def _frozen_files(state: EvaluationState) -> dict[str, dict]:
    return {"evaluation_config.json": state.config, "resolved_policy.json": state.policy,
            "selection_manifest.json": state.selection}


def _validate_frozen(state: EvaluationState) -> None:
    if not state.output.exists():
        return
    entries = {path.name for path in state.output.iterdir()} - {".random_order.lock"}
    expected_files = _frozen_files(state)
    if entries and not (state.output / "evaluation_config.json").exists():
        raise FileExistsError("Nonempty evaluation directory has no frozen configuration")
    for name, expected in expected_files.items():
        path = state.output / name
        if path.exists() and load_json(path) != expected:
            raise ValueError(f"Evaluation checkpoint configuration changed: {name}")
        if not path.exists() and entries - set(expected_files):
            raise ValueError(f"Incomplete frozen evaluation metadata: {name}")


def _trace_path(state: EvaluationState, index: int) -> Path:
    return state.output / "traces" / f"sample_{index:05d}" / "trace.json"


def _validate_trace(state: EvaluationState, index: int, trace: dict) -> dict:
    config = trace.get("config", {})
    orders = state.config["round_orders"]
    if (trace.get("schema_version") != 1 or trace.get("token_ids_complete") is not True
            or trace.get("sample") != state.samples[index]
            or config.get("sample_index") != index
            or config.get("evaluation_fingerprint") != state.config["fingerprint"]
            or config.get("fixed_order_policy_sha256") != state.config["policy_sha256"]
            or config.get("fixed_order_masks") != state.policy["masks"]
            or config.get("eval_batch_size") != state.config["eval_batch_size"]
            or config.get("round_orders") != orders
            or config.get("evaluation_mode") != state.config["mode"]
            or config.get("execution_contract") != state.config["execution_contract"]):
        raise ValueError("Evaluation trace identity, policy, order or token IDs changed")
    for key, value in state.run.config.items():
        if config.get(key) != value:
            raise ValueError(f"Evaluation trace run configuration changed: {key}")
    agents = state.policy["agents"]
    rounds = state.policy["num_rounds"]
    contract = state.config["execution_contract"]
    schedule = [(r, agent) for r in contract["executed_worker_rounds"] for agent in orders[r]]
    schedule.append((rounds, "dense_reviewer"))
    events = trace.get("events", [])
    if [(event.get("round"), event.get("agent_id")) for event in events] != schedule:
        raise ValueError("Evaluation trace does not contain every scheduled node")
    aliases = reviewer_agent_ids_from_config(state.run.config)
    prior = {}
    masks = state.policy["masks"]
    for event_id, event in enumerate(events):
        r, agent = schedule[event_id]
        if event.get("event_id") != event_id or event.get("sample_index") != index:
            raise ValueError("Evaluation node identity changed")
        if agent == "dense_reviewer":
            wanted = [("spatial", contract["reviewer_source_round"], source) for source in agents]
        else:
            target = agents.index(agent)
            prefix = orders[r][:orders[r].index(agent)]
            wanted = [("spatial", r, source) for source in prefix
                      if masks["spatial"][r][agents.index(source)][target]]
            wanted += [("temporal", r - 1, source) for i, source in enumerate(agents)
                       if masks["temporal"][r][i][target]]
        incoming = event.get("incoming", [])
        if [(item.get("edge_type"), item.get("round"), item.get("agent_id")) for item in incoming] != wanted:
            raise ValueError("Evaluation incoming edges or message order differ from the frozen policy")
        for item in incoming:
            source = prior[(item["round"], item["agent_id"])]
            if (item.get("event_id") != source["event_id"]
                    or item.get("output") != source["output"]
                    or item.get("source_output") != source["output"]
                    or item.get("source_generated_ids") != source["generated_ids"]):
                raise ValueError("Evaluation incoming provenance changed")
            if agent == "dense_reviewer" and item.get("reviewer_agent_id") != aliases[item["agent_id"]]:
                raise ValueError("Reviewer input aliases changed")
        prior[r, agent] = event
    outcome = evaluate_prediction("mmlu", events[-1]["output"], state.samples[index])
    final, record = trace.get("final", {}), trace.get("evaluation_record", {})
    for key in ("parsed_answer", "gold_answer", "correct"):
        if final.get(key) != outcome[key] or record.get(key) != outcome[key]:
            raise ValueError("Recorded prediction differs from the reviewer output")
    if (record.get("sample_index") != index or record.get("output") != events[-1]["output"]
            or record.get("evaluation_fingerprint") != state.config["fingerprint"]
            or record.get("policy_sha256") != state.config["policy_sha256"]
            or record.get("trace_json") != str(_trace_path(state, index))
            or record.get("round_orders") != orders
            or record.get("execution_contract") != contract
            or record.get("metrics", {}).get("llm_calls") != len(schedule)
            or record.get("truncated_event_ids") != [e["event_id"] for e in events if e.get("hit_max_new_tokens")]
            or record.get("metrics") != final.get("metrics")
            or record.get("edge_counts") != final.get("edge_counts")
            or record.get("edge_counts") != state.edge_counts):
        raise ValueError("Prediction metadata differs from its execution trace")
    for key in ("sample_elapsed_seconds", "evaluation_elapsed_seconds"):
        if not isinstance(record.get(key), (int, float)) or not np.isfinite(record[key]) or record[key] < 0:
            raise ValueError("Invalid evaluation duration")
    return record


def _resume_records(state: EvaluationState, *, resume: bool) -> dict[int, dict]:
    saved = _read_jsonl(state.output / "predictions.jsonl")
    prior = {record["sample_index"]: record for record in saved}
    if len(prior) != len(saved) or set(prior) - set(state.samples):
        raise ValueError("Duplicate or foreign prediction checkpoint")
    actual = set((state.output / "traces").glob("sample_*/trace.json"))
    expected = {_trace_path(state, index) for index in state.samples}
    if actual - expected:
        raise ValueError("Unexpected trace outside the frozen selection")
    if (saved or actual or _read_jsonl(state.output / "attempts.jsonl")) and not resume:
        raise FileExistsError("Evaluation has previous attempts; use --resume")
    records = {}
    for index in state.samples:
        path = _trace_path(state, index)
        if not path.exists():
            if index in prior:
                raise ValueError("Prediction checkpoint has no completed trace")
            continue
        record = {**_validate_trace(state, index, load_json(path)), "trace_sha256": file_sha(path)}
        if index in prior and prior[index] != record:
            raise ValueError("Prediction checkpoint or trace SHA changed")
        records[index] = record
    return records


def _summary(state: EvaluationState, records: dict[int, dict], attempts: list[dict]) -> dict:
    items = list(records.values())
    metrics = [item["metrics"] for item in items]
    total = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    failed = [item for item in attempts if item["status"] == "failed"]
    return {
        "schema_version": 1, "complete": True, "mode": "fixed_order_static_skip_empty_round1",
        "branch": state.run.branch, "run_fingerprint": state.run.config["fingerprint"],
        "execution_contract": state.config["execution_contract"],
        "baseline_evaluation_dir": state.config["baseline_evaluation_dir"],
        "evaluation_fingerprint": state.config["fingerprint"],
        "pruning_scope": "global", "pruning_percent": state.config["pruning_percent"],
        "benchmark": "mmlu", "split": "val", "total": total, "correct": correct, "accuracy": correct / total,
        **{key: sum(int(metric.get(key, 0)) for metric in metrics)
           for key in ("llm_calls", "prompt_tokens", "generated_tokens")},
        "elapsed_seconds": sum(item["evaluation_elapsed_seconds"] for item in items),
        "elapsed_semantics": "nonoverlapping successful completion intervals; excludes model loading",
        "sample_elapsed_seconds_sum": sum(item["sample_elapsed_seconds"] for item in items),
        "graph_elapsed_seconds": sum(metric["elapsed_seconds"] for metric in metrics),
        "eval_batch_size": state.config["eval_batch_size"],
        "generation_batching": {
            "configured_max_batch_size": state.config["eval_batch_size"],
            "recorded_samples": sum("batch_metadata_calls" in metric for metric in metrics),
            "recorded_calls": sum(metric.get("batch_metadata_calls", 0) for metric in metrics),
            "oom_fallback_samples": sum(metric.get("oom_fallback_calls", 0) > 0 for metric in metrics),
            "oom_fallback_calls": sum(metric.get("oom_fallback_calls", 0) for metric in metrics),
            "max_oom_split_depth": max((metric.get("oom_max_split_depth", 0) for metric in metrics), default=0),
        },
        "edge_counts": state.edge_counts, "retained_edges": state.edge_counts["worker_active"],
        "candidate_edges": state.edge_counts["worker_candidate"],
        "round_orders": state.config["round_orders"], "order_plan_sha256": state.config["order_plan_sha256"],
        "policy_sha256": state.config["policy_sha256"], "selection_sha256": state.config["selection_sha256"],
        "truncated_samples": sum(bool(item["truncated_event_ids"]) for item in items),
        "truncated_events": sum(len(item["truncated_event_ids"]) for item in items),
        "failed_attempts": len(failed),
        "failed_attempt_usage": {key: sum(item.get("usage", {}).get(key, 0) for item in failed)
                                 for key in ("llm_calls", "prompt_tokens", "generated_tokens")},
        "predictions_sha256": file_sha(state.output / "predictions.jsonl"),
        "attempts_sha256": file_sha(state.output / "attempts.jsonl"),
        "predictions_jsonl": str(state.output / "predictions.jsonl"),
        "traces_dir": str(state.output / "traces"), "traces_saved": True,
    }


async def evaluate_policy(ctx: Any, *, pruning_percent: int, resume: bool = True,
                          eval_batch_size: int = 4, limit: int = 1531, llm: Any = None) -> dict:
    state = prepare_evaluation(ctx, pruning_percent=pruning_percent,
                               eval_batch_size=eval_batch_size, limit=limit)
    with run_lock(state.output):
        _validate_frozen(state)
        for name, payload in _frozen_files(state).items():
            freeze_json(state.output / name, payload)
        records = _resume_records(state, resume=resume)
        attempts_path = state.output / "attempts.jsonl"
        attempts = _read_jsonl(attempts_path)
        for item in attempts:
            if (item.get("evaluation_fingerprint") != state.config["fingerprint"]
                    or item.get("sample_index") not in state.samples
                    or item.get("status") not in {"started", "completed", "failed", "recovered"}):
                raise ValueError("Invalid attempt log identity or status")
        for index, record in records.items():
            if not any(item["sample_index"] == index and item["status"] in {"completed", "recovered"}
                       and item.get("trace_sha256") == record["trace_sha256"] for item in attempts):
                attempts.append({"evaluation_fingerprint": state.config["fingerprint"],
                    "sample_index": index, "status": "recovered", "timestamp": datetime.now(timezone.utc).isoformat(),
                    "trace_sha256": record["trace_sha256"], "correct": record["correct"]})
        summary_path = state.output / "evaluation_summary.json"
        if summary_path.exists() and len(records) != len(state.samples):
            raise ValueError("Published evaluation summary has missing checkpoints")
        _write_jsonl_atomic(state.output / "predictions.jsonl", [records[i] for i in sorted(records)])
        _write_jsonl_atomic(attempts_path, attempts)
        pending = [index for index in state.samples if index not in records]
        if pending:
            os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
            os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            set_reproducible(state.config["seed"])
            if llm is None:
                if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                    raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for evaluation")
                from AgentAttrPrune.llm import HFChat
                llm = HFChat(hf_model_reference(ctx.config["model_name"]))
            if eval_batch_size > 1 and not callable(getattr(llm, "async_batching", None)):
                raise ValueError("The language model must support async_batching for batch > 1")
        from AgentAttrPrune.graph.fixed_order_skip_empty import SkipEmptyFinalRoundGraph
        decoding = ctx.config["decoding"]
        graph = SkipEmptyFinalRoundGraph(llm, worker_specs_from_config(ctx.config), num_rounds=ctx.config["num_rounds"],
                                policy=state.policy, round_orders=state.config["round_orders"],
                                temperature=decoding["temperature"], max_new_tokens=decoding["max_new_tokens"],
                                reviewer_agent_ids=reviewer_agent_ids_from_config(ctx.config)) if pending else None
        print(f"{ctx.branch} global {pruning_percent}% skip-empty-round1: validated {len(records)}/{len(state.samples)}; "
              f"{len(pending)} pending; eval batch={eval_batch_size}.", flush=True)
        last_checkpoint = time.perf_counter()

        async def one(index: int) -> None:
            nonlocal last_checkpoint
            if eval_batch_size == 1:
                set_reproducible(state.config["seed"] + index)
            path = _trace_path(state, index)
            recorder = GenerationTraceRecorder(path, config={
                **ctx.config, "phase": "evaluation", "sample_index": index,
                "evaluation_mode": "fixed_order_static_skip_empty_round1", "evaluation_fingerprint": state.config["fingerprint"],
                "fixed_order_policy_sha256": state.config["policy_sha256"],
                "execution_contract": state.config["execution_contract"],
                "round_orders": state.config["round_orders"], "eval_batch_size": eval_batch_size,
            }, sample=state.samples[index])
            number = 1 + sum(a["sample_index"] == index and a["status"] == "started" for a in attempts)
            base = {"evaluation_fingerprint": state.config["fingerprint"], "sample_index": index,
                    "attempt_number": number, "timestamp": datetime.now(timezone.utc).isoformat()}
            attempts.append({**base, "status": "started"})
            _write_jsonl_atomic(attempts_path, attempts)
            started = time.perf_counter()
            try:
                result = await graph.run_sample(state.samples[index], recorder=recorder,
                                                phase="evaluation", sample_index=index)
                outcome = evaluate_prediction("mmlu", result.final_answer, state.samples[index])
                completed = time.perf_counter()
                elapsed = completed - started
                record = {"sample_index": index, "evaluation_fingerprint": state.config["fingerprint"],
                    "policy_sha256": state.config["policy_sha256"], "round_orders": state.config["round_orders"],
                    "execution_contract": state.config["execution_contract"],
                    "subject": state.samples[index].get("subject", ""),
                    "gold_answer": outcome["gold_answer"], "parsed_answer": outcome["parsed_answer"],
                    "correct": outcome["correct"], "output": result.final_answer,
                    "metrics": _mapping(result.metrics), "edge_counts": _mapping(result.edge_counts),
                    "sample_elapsed_seconds": elapsed,
                    "evaluation_elapsed_seconds": completed - last_checkpoint,
                    "truncated_event_ids": [e["event_id"] for e in recorder.events if e["hit_max_new_tokens"]],
                    "trace_json": str(path), "evaluation_details": outcome.get("details") or {}}
                trace = recorder.payload(_trace_final(result, gold=outcome["gold_answer"],
                    parsed=outcome["parsed_answer"], correct=outcome["correct"]))
                trace["evaluation_record"] = record
                _validate_trace(state, index, trace)
                write_json(path, trace)
                records[index] = {**record, "trace_sha256": file_sha(path)}
                _write_jsonl_atomic(state.output / "predictions.jsonl", [records[i] for i in sorted(records)])
                last_checkpoint = completed
                attempts.append({**base, "status": "completed", "elapsed_seconds": elapsed,
                                 "correct": outcome["correct"], "trace_sha256": records[index]["trace_sha256"]})
                _write_jsonl_atomic(attempts_path, attempts)
                write_json(state.output / "progress.json", {"complete": False,
                    "completed_samples": len(records), "total_samples": len(state.samples),
                    "accuracy_so_far": sum(bool(r["correct"]) for r in records.values()) / len(records)})
                print(f"[{len(records)}/{len(state.samples)}] sample={index} correct={bool(record['correct'])} "
                      f"edges={state.edge_counts['worker_active']} elapsed={elapsed:.1f}s", flush=True)
            except BaseException as error:
                partial = state.output / "failed_attempts" / f"sample_{index:05d}_attempt_{number:04d}.json"
                write_json(partial, recorder.payload({"runtime_error": f"{type(error).__name__}: {error}"}))
                attempts.append({**base, "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter() - started, "usage": recorder.summary,
                    "partial_trace_json": str(partial)})
                _write_jsonl_atomic(attempts_path, attempts)
                write_json(state.output / "progress.json", {"complete": False,
                    "completed_samples": len(records), "total_samples": len(state.samples),
                    "failed_sample_index": index, "error": str(error)})
                raise

        batching = llm.async_batching(eval_batch_size) if pending and eval_batch_size > 1 else nullcontext()
        async with batching:
            for offset in range(0, len(pending), eval_batch_size):
                tasks = [asyncio.create_task(one(index)) for index in pending[offset:offset + eval_batch_size]]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        if len(records) != len(state.samples):
            raise RuntimeError("Evaluation incomplete; refusing to publish accuracy")
        from experiments_attrprune.reordered_mmlu import load_policies
        if load_policies(ctx)[pruning_percent] != state.policy:
            raise ValueError("Frozen policy changed during evaluation")
        _validate_frozen(state)
        summary = _summary(state, records, attempts)
        if summary_path.exists() and load_json(summary_path) != summary:
            raise ValueError("Published evaluation summary differs from its checkpoints")
        write_json(summary_path, summary)
        write_json(state.output / "progress.json", {"complete": True,
                   "completed_samples": len(records), "total_samples": len(state.samples), "eta_seconds": 0})
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from experiments_attrprune.reordered_mmlu import DEFAULT_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("direct", "recalibrate"), required=True)
    parser.add_argument("--stage", choices=("prepare", "eval"), default="eval")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--pruning-percent", type=int, choices=(80,), default=80)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1531)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.eval_batch_size, args.limit) < 1:
        parser.error("Batch size and evaluation limit must be positive")
    return args


def prepare_context(args: argparse.Namespace) -> Any:
    """Read the existing branch without changing its frozen experiment settings."""
    from experiments_attrprune import reordered_mmlu as original
    suffix = "direct" if args.branch == "direct" else "recalibrated"
    run_dir = (args.run_dir or args.source_run.with_name(
        f"dev40_rowminuscol_{suffix}_correctness_aware")).resolve()
    existing = load_json(run_dir / "run_config.json")
    settings = existing["reordered_experiment"]
    if settings["branch"] != args.branch:
        raise ValueError("Existing reordered branch differs from --branch")
    if Path(settings["source_run"]).resolve() != args.source_run.resolve():
        raise ValueError("Existing source run differs from --source-run")
    inherited = original.parse_args([
        "--branch", args.branch, "--stage", "prepare", "--run-dir", str(run_dir),
        "--source-run", str(args.source_run), "--percentages", *map(str, settings["percentages"]),
        "--eval-batch-size", str(settings["eval_batch_size"]),
        "--limit", str(settings["evaluation_limit"]),
        "--attr-batch-size", str(settings["attribution_batch_size"]), "--resume",
    ])
    return original.prepare_context(inherited)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.stage == "prepare" or args.dry_run:
        state = prepare_evaluation(ctx, pruning_percent=80,
            eval_batch_size=args.eval_batch_size, limit=args.limit)
        if not args.dry_run:
            with run_lock(state.output):
                _validate_frozen(state)
                for name, payload in _frozen_files(state).items():
                    freeze_json(state.output / name, payload)
        result = {"dry_run": args.dry_run, "prepared": not args.dry_run,
                  "branch": ctx.branch, "output_dir": str(state.output),
                  "evaluation_fingerprint": state.config["fingerprint"],
                  "execution_contract": state.config["execution_contract"],
                  "edge_counts": state.edge_counts, "total_samples": len(state.samples),
                  "eval_batch_size": args.eval_batch_size,
                  "baseline_evaluation_dir": state.config["baseline_evaluation_dir"]}
    else:
        result = asyncio.run(evaluate_policy(ctx, pruning_percent=80, resume=args.resume,
            eval_batch_size=args.eval_batch_size, limit=args.limit))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
