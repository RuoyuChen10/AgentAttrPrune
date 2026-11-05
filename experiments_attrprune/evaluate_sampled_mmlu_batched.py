"""Fresh batch-4 evaluation of independently sampled MMLU communication graphs.

The source attribution, pruning rules and per-question graph seeds are shared
with the serial evaluator. A distinct frozen version/batch contract prevents
mixing its checkpoints with serial evaluations or batch-change continuations.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

from AgentAttrPrune.datasets import evaluate_prediction
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import evaluate_sampled_mmlu as legacy
from experiments_attrprune.common import (
    hf_model_reference, load_json, reviewer_agent_ids_from_config,
    worker_specs_from_config, write_json,
)
from experiments_attrprune.evaluate_mmlu import (
    _mapping, _trace_final, _write_jsonl_atomic, set_reproducible,
)

EVALUATION_VERSION = "sampled_batched_v1"
EVALUATION_MODE = "agentprune_sampling_batched"


def prepare_context(args: argparse.Namespace) -> legacy.EvaluationContext:
    return legacy.prepare_context(args, eval_batch_size=args.eval_batch_size,
                                  evaluation_version=EVALUATION_VERSION)


def _validate_trace(ctx: legacy.EvaluationContext, index: int,
                    trace: dict, realized: dict) -> dict:
    record = legacy._validate_trace(ctx, index, trace, realized)
    if (trace["config"].get("evaluation_mode") != EVALUATION_MODE
            or trace["config"].get("eval_batch_size") != ctx.config["eval_batch_size"]
            or record.get("eval_batch_size") != ctx.config["eval_batch_size"]
            or record.get("metrics", {}).get("llm_calls") != len(trace["events"])):
        raise ValueError("Batched evaluation trace mode, batch size or call count changed")
    return record


def _resume_records(ctx: legacy.EvaluationContext, *, resume: bool) -> dict[int, dict]:
    records = legacy._resume_records(ctx, resume=resume)
    for index, record in records.items():
        _validate_trace(ctx, index, load_json(Path(record["trace_json"])), record["realized_graph"])
    return records


def _validate_attempts(ctx: legacy.EvaluationContext, attempts: list[dict]) -> None:
    for row in attempts:
        if (row.get("evaluation_fingerprint") != ctx.config["fingerprint"]
                or row.get("sample_index") not in ctx.samples
                or row.get("status") not in {"started", "completed", "failed", "recovered"}):
            raise ValueError("Invalid batched evaluation attempt identity or status")


def _summary(ctx: legacy.EvaluationContext, records: dict[int, dict], attempts: list[dict]) -> dict:
    ordered = {i: records[i] for i in sorted(records)}
    result = legacy._summary(ctx, ordered, attempts)
    result.update({
        "evaluation_mode": EVALUATION_MODE,
        "eval_batch_size": ctx.config["eval_batch_size"],
        "generation_batch_size": ctx.config["generation_batch_size"],
        "elapsed_semantics": "nonoverlapping successful completion intervals; excludes model loading",
        "generation_batching": {
            "configured_max_batch_size": ctx.config["eval_batch_size"],
            "recorded_calls": sum(r["metrics"].get("batch_metadata_calls", 0) for r in ordered.values()),
            "oom_fallback_calls": sum(r["metrics"].get("oom_fallback_calls", 0) for r in ordered.values()),
            "max_oom_split_depth": max((r["metrics"].get("oom_max_split_depth", 0) for r in ordered.values()), default=0),
        },
    })
    return result


def validate_completed(args: argparse.Namespace) -> dict:
    """Read and verify the complete result without loading a model or rewriting it."""
    ctx = prepare_context(args)
    records = _resume_records(ctx, resume=True)
    if set(records) != set(ctx.samples):
        raise ValueError("Batched sampled evaluation is incomplete")
    attempts = legacy._read_jsonl(ctx.output_dir / "attempts.jsonl")
    _validate_attempts(ctx, attempts)
    successes = {(a["sample_index"], a.get("trace_sha256")) for a in attempts
                 if a["status"] in {"completed", "recovered"}}
    if any((i, r["trace_sha256"]) not in successes for i, r in records.items()):
        raise ValueError("Completed trace has no successful attempt provenance")
    legacy._check_hashes(ctx.source_hashes)
    summary = _summary(ctx, records, attempts)
    if load_json(ctx.output_dir / "evaluation_summary.json") != summary:
        raise ValueError("Published batched summary differs from its checkpoints")
    return summary


async def evaluate(args: argparse.Namespace, *, llm: Any = None) -> dict:
    ctx = prepare_context(args)
    batch_size = ctx.config["eval_batch_size"]
    overview = {"output_dir": str(ctx.output_dir), "fingerprint": ctx.config["fingerprint"],
                "num_samples": len(ctx.samples), "round_policy": "per_round",
                "pruning_scope": args.pruning_scope, "pruning_percent": args.pruning_percent,
                "eval_batch_size": batch_size, "policy_sha256": legacy.digest(ctx.policy)}
    if args.dry_run:
        return {**overview, "dry_run": True}
    with legacy.source_runner.run_lock(ctx.output_dir):
        legacy._prepare(ctx)
        if args.prepare_only:
            return {**overview, "prepared": True}
        records = _resume_records(ctx, resume=args.resume)
        attempts_path = ctx.output_dir / "attempts.jsonl"
        attempts = legacy._read_jsonl(attempts_path)
        _validate_attempts(ctx, attempts)
        for index, record in records.items():
            if not any(a["sample_index"] == index and a["status"] in {"completed", "recovered"}
                       and a.get("trace_sha256") == record["trace_sha256"] for a in attempts):
                attempts.append({"evaluation_fingerprint": ctx.config["fingerprint"],
                    "sample_index": index, "status": "recovered", "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sampling_seed": record["sampling_seed"], "realized_graph": record["realized_graph"],
                    "trace_sha256": record["trace_sha256"], "correct": record["correct"]})
        summary_path = ctx.output_dir / "evaluation_summary.json"
        if summary_path.exists() and len(records) != len(ctx.samples):
            raise ValueError("Published batched summary has missing checkpoints")
        _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", [records[i] for i in sorted(records)])
        _write_jsonl_atomic(attempts_path, attempts)
        pending = [i for i in ctx.samples if i not in records]
        if pending:
            os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
            os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            # Graph sampling has its own per-question torch.Generator; avoid
            # reseeding process-global state inside concurrent question tasks.
            set_reproducible(ctx.config["seed"])
            if llm is None:
                if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                    raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for evaluation")
                from AgentAttrPrune.llm import HFChat
                llm = HFChat(hf_model_reference(ctx.source_config["model_name"]))
            if not callable(getattr(llm, "async_batching", None)):
                raise ValueError("The language model must support async_batching for batch 4")
        from AgentAttrPrune.graph.sampled_dag import SampledDAGGraph
        from AgentAttrPrune.pruning.sampled import sample_graph
        print(f"Validated {len(records)}/{len(ctx.samples)} completed evaluations; {len(pending)} remain at batch={batch_size}.", flush=True)
        last_checkpoint = time.perf_counter()
        invocation_started = last_checkpoint
        initial_count = len(records)

        async def one(index: int) -> None:
            nonlocal last_checkpoint
            seed = legacy.sampling_seed(ctx.config["seed"], index)
            realized = sample_graph(ctx.policy, seed=seed)
            trace_path = legacy._trace_path(ctx, index)
            recorder = GenerationTraceRecorder(trace_path, config={
                **ctx.source_config, "phase": "evaluation", "sample_index": index,
                "evaluation_mode": EVALUATION_MODE,
                "evaluation_fingerprint": ctx.config["fingerprint"],
                "sampling_seed": seed, "realized_graph": realized,
                "round_orders": realized["round_orders"], "eval_batch_size": batch_size,
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
                completed = time.perf_counter()
                elapsed = completed - started
                record = {"sample_index": index, "evaluation_fingerprint": ctx.config["fingerprint"],
                    "eval_batch_size": batch_size,
                    "sampling_seed": seed, "realized_graph": realized,
                    "subject": ctx.samples[index].get("subject", ""),
                    "gold_answer": outcome["gold_answer"], "parsed_answer": outcome["parsed_answer"],
                    "correct": outcome["correct"], "output": result.final_answer,
                    "metrics": _mapping(result.metrics), "edge_counts": _mapping(result.edge_counts),
                    "sampled_edge_counts": legacy._mask_counts(realized),
                    "sample_elapsed_seconds": elapsed, "evaluation_elapsed_seconds": completed - last_checkpoint,
                    "truncated_event_ids": [e["event_id"] for e in recorder.events if e["hit_max_new_tokens"]],
                    "trace_json": str(trace_path), "evaluation_details": outcome.get("details") or {}}
                trace = recorder.payload(_trace_final(result, gold=outcome["gold_answer"],
                    parsed=outcome["parsed_answer"], correct=outcome["correct"]))
                trace["evaluation_record"] = record
                _validate_trace(ctx, index, trace, realized)
                write_json(trace_path, trace)
                records[index] = {**record, "trace_sha256": legacy.source_runner.file_sha(trace_path)}
                _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", [records[i] for i in sorted(records)])
                last_checkpoint = completed
                attempts.append({**base, "status": "completed", "elapsed_seconds": elapsed,
                                 "correct": outcome["correct"], "trace_sha256": records[index]["trace_sha256"]})
                _write_jsonl_atomic(attempts_path, attempts)
                done = len(records)
                eta = (completed - invocation_started) / (done - initial_count) * (len(ctx.samples) - done)
                write_json(ctx.output_dir / "progress.json", {"complete": False,
                    "completed_samples": done, "total_samples": len(ctx.samples),
                    "eval_batch_size": batch_size,
                    "accuracy_so_far": sum(bool(r["correct"]) for r in records.values()) / done,
                    "eta_seconds": eta})
                print(f"[{done}/{len(ctx.samples)}] sample={index} batch={batch_size} "
                      f"correct={bool(record['correct'])} edges={record['sampled_edge_counts']['worker_active']} "
                      f"elapsed={elapsed:.1f}s ETA={eta / 60:.1f}min", flush=True)
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

        batching = llm.async_batching(batch_size) if pending else nullcontext()
        async with batching:
            for offset in range(0, len(pending), batch_size):
                tasks = [asyncio.create_task(one(i)) for i in pending[offset:offset + batch_size]]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        if len(records) != len(ctx.samples):
            raise RuntimeError("Batched evaluation is incomplete; refusing to publish accuracy")
        legacy._check_hashes(ctx.source_hashes)
        summary = _summary(ctx, records, attempts)
        if summary_path.exists() and load_json(summary_path) != summary:
            raise ValueError("Published batched summary differs from checkpoints")
        write_json(summary_path, summary)
        write_json(ctx.output_dir / "progress.json", {"complete": True,
            "completed_samples": len(records), "total_samples": len(ctx.samples),
            "eval_batch_size": batch_size, "eta_seconds": 0})
        return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=legacy.DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pruning-scope", choices=("global", "uniform"), required=True)
    parser.add_argument("--pruning-percent", type=int, choices=(50, 60, 80), required=True)
    parser.add_argument("--round-policy", choices=("per_round",), default="per_round")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--limit", type=int, default=1531)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, choices=(4,), default=4)
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
    print(json.dumps(asyncio.run(evaluate(parse_args(argv))), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
