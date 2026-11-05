"""Continue a frozen serial sampled-DAG evaluation with batched remaining work.

Original predictions and trace bytes stay in their original directory. A new
checkpoint records their immutable references and labels the combined result as
mixed batch 1 -> batch 4, never as a uniformly batched evaluation.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

from AgentAttrPrune.datasets import evaluate_prediction
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune import evaluate_sampled_mmlu as legacy
from experiments_attrprune.common import (
    config_fingerprint, hf_model_reference, load_json, reviewer_agent_ids_from_config,
    worker_specs_from_config, write_json,
)
from experiments_attrprune.evaluate_mmlu import _mapping, _trace_final, _write_jsonl_atomic, set_reproducible

VERSION = 1
file_sha = legacy.source_runner.file_sha
freeze_json = legacy.source_runner.freeze_json
run_lock = legacy.source_runner.run_lock
digest = legacy.digest


@dataclass
class ContinuationContext:
    legacy: legacy.EvaluationContext
    current: legacy.EvaluationContext
    manifest: dict
    imported_records: dict[int, dict]
    imported_attempts: list[dict]

    @property
    def output_dir(self) -> Path:
        return self.current.output_dir

    @property
    def batch_size(self) -> int:
        return self.manifest["new_eval_batch_size"]


@contextmanager
def source_lock(source_dir: Path):
    """Use the existing serial writer's lock without creating/modifying files."""
    source_dir = Path(source_dir).resolve()
    path = source_dir / ".random_order.lock"
    if not path.is_file():
        raise FileNotFoundError(f"Missing serial writer lock: {path}")
    # NFS requires a writable descriptor for an exclusive flock. r+ leaves the
    # existing lock bytes intact, unlike the original writer's append mode.
    with path.open("r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Serial source evaluation is still running; stop its writer before continuing") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _source_context(source: Path) -> legacy.EvaluationContext:
    saved = load_json(source / "evaluation_config.json")
    if (saved.get("eval_batch_size") != 1 or saved.get("generation_batch_size") != 1
            or saved.get("pruning_percent") != 60 or saved.get("round_policy") != "per_round"
            or saved.get("pruning_scope") not in {"global", "uniform"}):
        raise ValueError("Continuation requires an original serial per-round 60% sampled evaluation")
    args = legacy.parse_args([
        "--run-dir", saved["source_run"], "--output-dir", str(source),
        "--pruning-scope", saved["pruning_scope"], "--pruning-percent", "60",
        "--split", saved["split"], "--limit", str(saved["limit"]),
        "--seed", str(saved["seed"]), "--resume",
    ])
    # This also preserves the original frozen policy's harmless CPU sigmoid ULPs.
    return legacy.prepare_context(args)


def _prepare_context_locked(args: argparse.Namespace) -> ContinuationContext:
    source = args.source_evaluation.resolve()
    old = _source_context(source)
    imported = legacy._resume_records(old, resume=True)
    if len(imported) == len(old.samples):
        raise ValueError("Serial evaluation is already complete; no remaining questions to continue")
    if (source / "evaluation_summary.json").exists():
        raise ValueError("Incomplete serial source unexpectedly has a published summary")
    attempts = legacy._read_jsonl(source / "attempts.jsonl")
    for item in attempts:
        if (item.get("evaluation_fingerprint") != old.config["fingerprint"]
                or item.get("sample_index") not in old.samples
                or item.get("status") not in {"started", "completed", "failed", "recovered"}):
            raise ValueError("Invalid original attempt log identity or status")
    metadata_names = (
        "evaluation_config.json", "sampling_policy.json", "selection_manifest.json",
        "predictions.jsonl", "attempts.jsonl", "progress.json", "evaluation_summary.json",
    )
    metadata = [source / name for name in metadata_names if (source / name).exists()]
    source_hashes = {str(p): file_sha(p) for p in metadata}
    manifest = {
        "schema_version": 1, "continuation_version": VERSION,
        "source_evaluation": str(source), "source_evaluation_fingerprint": old.config["fingerprint"],
        "source_file_sha256": source_hashes,
        "source_metadata_presence": {name: (source / name).exists() for name in metadata_names},
        "imported_samples": [
            {"sample_index": i, "trace_json": imported[i]["trace_json"],
             "trace_sha256": imported[i]["trace_sha256"], "eval_batch_size": 1}
            for i in sorted(imported)
        ],
        "imported_predictions_sha256": digest([imported[i] for i in sorted(imported)]),
        "imported_sample_count": len(imported), "total_samples": len(old.samples),
        "remaining_sample_indices": [i for i in old.samples if i not in imported],
        "new_eval_batch_size": args.eval_batch_size,
        "source_eval_batch_size": 1,
        "trace_storage": "immutable original trace references plus new continuation traces",
        "sampling_rule": "unchanged per-question sampling_seed and sampled graph; only generation batching changes",
    }
    output = (args.output_dir or source.with_name(source.name + f"_batch{args.eval_batch_size}_continuation")).resolve()
    run_dir = Path(old.config["source_run"]).resolve()
    reference = Path(old.source_config["random_order_experiment"]["reference"]["run_dir"]).resolve()
    if (output == source or output in source.parents or source in output.parents
            or output == run_dir or output in run_dir.parents
            or (run_dir in output.parents and run_dir / "evaluation" not in output.parents)
            or output == reference or reference in output.parents or output in reference.parents):
        raise ValueError("Use an independent continuation evaluation directory")
    config = {**old.config,
        "output_dir": str(output), "evaluation_version": "sampled_batch_continuation_v1",
        "protocol": "mixed_batch_continuation", "eval_batch_size": "mixed",
        "generation_batch_size": "mixed", "continuation_eval_batch_size": args.eval_batch_size,
        "source_evaluation": str(source), "source_evaluation_fingerprint": old.config["fingerprint"],
        "continuation_manifest_sha256": digest(manifest),
        "batch_size_by_origin": {"imported_serial": 1, "continuation": args.eval_batch_size},
    }
    config["fingerprint"] = config_fingerprint(config)
    current = legacy.EvaluationContext(output, old.source_config, config, old.policy,
                                       old.selection, old.samples, old.source_hashes)
    ctx = ContinuationContext(old, current, manifest, imported, attempts)
    _validate_prepared(ctx)
    _check_source(ctx)
    return ctx


def prepare_context(args: argparse.Namespace) -> ContinuationContext:
    """Read-only planning under the original evaluator's exclusive lock."""
    with source_lock(args.source_evaluation):
        return _prepare_context_locked(args)


def _frozen_files(ctx: ContinuationContext) -> dict[str, dict | list]:
    return {
        "evaluation_config.json": ctx.current.config,
        "sampling_policy.json": ctx.current.policy,
        "selection_manifest.json": ctx.current.selection,
        "continuation_manifest.json": ctx.manifest,
        "imported_predictions.json": [ctx.imported_records[i] for i in sorted(ctx.imported_records)],
    }


def _validate_prepared(ctx: ContinuationContext) -> None:
    if not ctx.output_dir.exists():
        return
    entries = {p.name for p in ctx.output_dir.iterdir()} - {".random_order.lock"}
    expected = _frozen_files(ctx)
    if entries and not (ctx.output_dir / "evaluation_config.json").exists():
        raise FileExistsError("Nonempty continuation directory has no frozen configuration")
    for name, payload in expected.items():
        p = ctx.output_dir / name
        if p.exists() and load_json(p) != payload:
            raise ValueError(f"Continuation source or configuration changed: {name}")
        if not p.exists() and entries - set(expected):
            raise ValueError(f"Incomplete continuation preparation: {name}")


def _check_source(ctx: ContinuationContext) -> None:
    legacy._check_hashes(ctx.legacy.source_hashes)
    legacy._check_hashes(ctx.manifest["source_file_sha256"])
    for name, present in ctx.manifest["source_metadata_presence"].items():
        if (ctx.legacy.output_dir / name).exists() != present:
            raise ValueError(f"Original serial metadata inventory changed: {name}")
    for row in ctx.manifest["imported_samples"]:
        p = Path(row["trace_json"])
        if file_sha(p) != row["trace_sha256"]:
            raise ValueError(f"Imported serial trace changed: {p}")
    actual = set((ctx.legacy.output_dir / "traces").glob("sample_*/trace.json"))
    expected = {Path(r["trace_json"]) for r in ctx.manifest["imported_samples"]}
    if actual != expected:
        raise ValueError("Original serial trace set changed after migration")


def _prepare_locked(ctx: ContinuationContext) -> dict:
    _validate_prepared(ctx)
    _check_source(ctx)
    for name, payload in _frozen_files(ctx).items():
        freeze_json(ctx.output_dir / name, payload)
    return {"prepared": True, "source_evaluation": str(ctx.legacy.output_dir),
            "output_dir": str(ctx.output_dir), "imported_samples": len(ctx.imported_records),
            "remaining_samples": len(ctx.current.samples) - len(ctx.imported_records),
            "eval_batch_size": "mixed", "continuation_eval_batch_size": ctx.batch_size}


def prepare_run(ctx: ContinuationContext) -> dict:
    with source_lock(ctx.legacy.output_dir), run_lock(ctx.output_dir):
        return _prepare_locked(ctx)


def _validate_new_trace(ctx: ContinuationContext, index: int, trace: dict, realized: dict) -> dict:
    record = legacy._validate_trace(ctx.current, index, trace, realized)
    if (index in ctx.imported_records
            or trace["config"].get("eval_batch_size") != ctx.batch_size
            or trace["config"].get("evaluation_mode") != "agentprune_sampling_batch_continuation"
            or record.get("eval_batch_size") != ctx.batch_size
            or record.get("origin") != "continuation"):
        raise ValueError("Continuation trace batching or provenance changed")
    if record.get("metrics", {}).get("llm_calls") != len(trace.get("events", [])):
        raise ValueError("Continuation trace calls do not match recorded nodes")
    return record


def _resume_records(ctx: ContinuationContext, *, resume: bool) -> dict[int, dict]:
    from AgentAttrPrune.pruning.sampled import sample_graph
    saved = legacy._read_jsonl(ctx.output_dir / "predictions.jsonl")
    prior = {int(row["sample_index"]): row for row in saved}
    if len(prior) != len(saved) or set(prior) - set(ctx.current.samples):
        raise ValueError("Duplicate or foreign continuation prediction checkpoint")
    actual = set((ctx.output_dir / "traces").glob("sample_*/trace.json"))
    expected = {legacy._trace_path(ctx.current, i) for i in ctx.current.samples if i not in ctx.imported_records}
    if actual - expected:
        raise ValueError("Unexpected continuation trace or duplicate imported question")
    if (saved or actual or legacy._read_jsonl(ctx.output_dir / "attempts.jsonl")) and not resume:
        raise FileExistsError("Continuation has previous attempts; use --resume")
    records = dict(ctx.imported_records)
    for index, record in ctx.imported_records.items():
        if saved and prior.get(index) != record:
            raise ValueError("Imported serial prediction was changed or removed")
    for index in ctx.current.samples:
        if index in ctx.imported_records:
            continue
        path = legacy._trace_path(ctx.current, index)
        if not path.exists():
            if index in prior:
                raise ValueError("Continuation prediction has no complete trace")
            continue
        realized = sample_graph(ctx.current.policy, seed=legacy.sampling_seed(ctx.current.config["seed"], index))
        record = _validate_new_trace(ctx, index, load_json(path), realized)
        record = {**record, "trace_sha256": file_sha(path)}
        if index in prior and prior[index] != record:
            raise ValueError("Continuation prediction or trace SHA changed")
        records[index] = record
    return records


def _validate_attempts(ctx: ContinuationContext, attempts: list[dict]) -> None:
    for item in attempts:
        if (item.get("evaluation_fingerprint") != ctx.current.config["fingerprint"]
                or item.get("sample_index") not in ctx.current.samples
                or item.get("status") not in {"started", "completed", "failed", "recovered"}
                or item.get("sample_index") in ctx.imported_records):
            raise ValueError("Invalid continuation attempt identity or status")


def _summary(ctx: ContinuationContext, records: dict[int, dict], attempts: list[dict]) -> dict:
    ordered = {i: records[i] for i in sorted(records)}
    result = legacy._summary(ctx.current, ordered, attempts)
    imported_count = len(ctx.imported_records)
    new_count = len(records) - imported_count
    source_failed = [a for a in ctx.imported_attempts if a["status"] == "failed"]
    result.update({
        "protocol": "mixed_batch_continuation", "eval_batch_size": "mixed",
        "continuation_eval_batch_size": ctx.batch_size,
        "batch_size_counts": {"1": imported_count, str(ctx.batch_size): new_count},
        "source_evaluation": str(ctx.legacy.output_dir),
        "source_evaluation_fingerprint": ctx.legacy.config["fingerprint"],
        "continuation_manifest_sha256": digest(ctx.manifest),
        "imported_samples": imported_count, "continued_samples": new_count,
        "imported_sample_indices": sorted(ctx.imported_records),
        "continued_sample_indices": [i for i in sorted(records) if i not in ctx.imported_records],
        "imported_elapsed_seconds": sum(r["evaluation_elapsed_seconds"] for r in ctx.imported_records.values()),
        "continuation_elapsed_seconds": sum(r["evaluation_elapsed_seconds"] for i, r in ordered.items() if i not in ctx.imported_records),
        "elapsed_semantics": "original serial successful intervals plus nonoverlapping continuation completion intervals; excludes model loading",
        "source_failed_attempts": len(source_failed),
        "source_failed_attempt_usage": {key: sum(a.get("usage", {}).get(key, 0) for a in source_failed)
                                        for key in ("llm_calls", "prompt_tokens", "generated_tokens")},
        "trace_storage": ctx.manifest["trace_storage"],
        "generation_batching": {
            "configured_continuation_max_batch_size": ctx.batch_size,
            "recorded_calls": sum(r["metrics"].get("batch_metadata_calls", 0) for i, r in ordered.items() if i not in ctx.imported_records),
            "oom_fallback_calls": sum(r["metrics"].get("oom_fallback_calls", 0) for i, r in ordered.items() if i not in ctx.imported_records),
            "max_oom_split_depth": max((r["metrics"].get("oom_max_split_depth", 0) for i, r in ordered.items() if i not in ctx.imported_records), default=0),
        },
    })
    return result


def validate_completed(args: argparse.Namespace) -> dict:
    """Read-only validation of imported and newly generated final results."""
    with source_lock(args.source_evaluation):
        ctx = _prepare_context_locked(args)
        records = _resume_records(ctx, resume=True)
        if set(records) != set(ctx.current.samples):
            raise ValueError("Continuation is incomplete")
        attempts = legacy._read_jsonl(ctx.output_dir / "attempts.jsonl")
        _validate_attempts(ctx, attempts)
        successful = {(a["sample_index"], a.get("trace_sha256")) for a in attempts if a["status"] in {"completed", "recovered"}}
        if any((i, row["trace_sha256"]) not in successful for i, row in records.items() if i not in ctx.imported_records):
            raise ValueError("Continuation has a complete trace without successful attempt provenance")
        summary = _summary(ctx, records, attempts)
        if load_json(ctx.output_dir / "evaluation_summary.json") != summary:
            raise ValueError("Published continuation summary differs from checkpoints")
        return summary


async def evaluate(args: argparse.Namespace, *, llm: Any = None) -> dict:
    # The serial writer cannot restart while the combined evaluation depends on
    # its original predictions. No source files are rewritten or copied.
    with source_lock(args.source_evaluation):
        ctx = _prepare_context_locked(args)
        overview = {"output_dir": str(ctx.output_dir), "source_evaluation": str(ctx.legacy.output_dir),
                    "imported_samples": len(ctx.imported_records),
                    "remaining_samples": len(ctx.current.samples) - len(ctx.imported_records),
                    "eval_batch_size": "mixed", "continuation_eval_batch_size": ctx.batch_size}
        if args.dry_run:
            return {**overview, "dry_run": True}
        with run_lock(ctx.output_dir):
            _prepare_locked(ctx)
            if args.prepare_only:
                return {**overview, "prepared": True}
            records = _resume_records(ctx, resume=args.resume)
            attempts_path = ctx.output_dir / "attempts.jsonl"
            attempts = legacy._read_jsonl(attempts_path)
            _validate_attempts(ctx, attempts)
            for index, record in records.items():
                if index in ctx.imported_records:
                    continue
                if not any(a["sample_index"] == index and a["status"] in {"completed", "recovered"}
                           and a.get("trace_sha256") == record["trace_sha256"] for a in attempts):
                    attempts.append({"evaluation_fingerprint": ctx.current.config["fingerprint"],
                        "sample_index": index, "status": "recovered", "timestamp": datetime.now(timezone.utc).isoformat(),
                        "sampling_seed": record["sampling_seed"], "realized_graph": record["realized_graph"],
                        "trace_sha256": record["trace_sha256"], "correct": record["correct"]})
            summary_path = ctx.output_dir / "evaluation_summary.json"
            if summary_path.exists() and len(records) != len(ctx.current.samples):
                raise ValueError("Published continuation summary has missing checkpoints")
            _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", [records[i] for i in sorted(records)])
            _write_jsonl_atomic(attempts_path, attempts)
            pending = [i for i in ctx.current.samples if i not in records]
            if pending:
                os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
                os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                set_reproducible(ctx.current.config["seed"])
                if llm is None:
                    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for evaluation")
                    from AgentAttrPrune.llm import HFChat
                    llm = HFChat(hf_model_reference(ctx.current.source_config["model_name"]))
                if not callable(getattr(llm, "async_batching", None)):
                    raise ValueError("The language model must support async_batching for batch 4")
            from AgentAttrPrune.graph.sampled_dag import SampledDAGGraph
            from AgentAttrPrune.pruning.sampled import sample_graph
            print(f"Imported {len(ctx.imported_records)} unchanged serial answers; validated {len(records)}/{len(ctx.current.samples)} total; "
                  f"{len(pending)} remain at batch={ctx.batch_size}.", flush=True)
            last_checkpoint = time.perf_counter()
            continuation_started = last_checkpoint
            initial_new = len(records) - len(ctx.imported_records)

            async def one(index: int) -> None:
                nonlocal last_checkpoint
                # sample_graph has a private CPU torch.Generator. No global RNG
                # reseed occurs inside concurrently executing question tasks.
                seed = legacy.sampling_seed(ctx.current.config["seed"], index)
                realized = sample_graph(ctx.current.policy, seed=seed)
                trace_path = legacy._trace_path(ctx.current, index)
                recorder = GenerationTraceRecorder(trace_path, config={
                    **ctx.current.source_config, "phase": "evaluation", "sample_index": index,
                    "evaluation_mode": "agentprune_sampling_batch_continuation",
                    "evaluation_fingerprint": ctx.current.config["fingerprint"],
                    "sampling_seed": seed, "realized_graph": realized,
                    "round_orders": realized["round_orders"], "eval_batch_size": ctx.batch_size,
                }, sample=ctx.current.samples[index])
                attempt_number = 1 + sum(a["sample_index"] == index and a["status"] == "started" for a in attempts)
                base = {"evaluation_fingerprint": ctx.current.config["fingerprint"], "sample_index": index,
                        "attempt_number": attempt_number, "sampling_seed": seed,
                        "realized_graph": realized, "timestamp": datetime.now(timezone.utc).isoformat()}
                attempts.append({**base, "status": "started"})
                _write_jsonl_atomic(attempts_path, attempts)
                started = time.perf_counter()
                try:
                    decoding = ctx.current.source_config["decoding"]
                    graph = SampledDAGGraph(llm, worker_specs_from_config(ctx.current.source_config),
                        num_rounds=ctx.current.source_config["num_rounds"], realized_graph=realized,
                        temperature=decoding["temperature"], max_new_tokens=decoding["max_new_tokens"],
                        reviewer_agent_ids=reviewer_agent_ids_from_config(ctx.current.source_config))
                    result = await graph.run_sample(ctx.current.samples[index], recorder=recorder,
                                                    phase="evaluation", sample_index=index)
                    outcome = evaluate_prediction("mmlu", result.final_answer, ctx.current.samples[index])
                    completed = time.perf_counter()
                    elapsed = completed - started
                    record = {"sample_index": index, "evaluation_fingerprint": ctx.current.config["fingerprint"],
                        "origin": "continuation", "eval_batch_size": ctx.batch_size,
                        "sampling_seed": seed, "realized_graph": realized,
                        "subject": ctx.current.samples[index].get("subject", ""),
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
                    _validate_new_trace(ctx, index, trace, realized)
                    write_json(trace_path, trace)
                    records[index] = {**record, "trace_sha256": file_sha(trace_path)}
                    _write_jsonl_atomic(ctx.output_dir / "predictions.jsonl", [records[i] for i in sorted(records)])
                    last_checkpoint = completed
                    attempts.append({**base, "status": "completed", "elapsed_seconds": elapsed,
                                     "correct": outcome["correct"], "trace_sha256": records[index]["trace_sha256"]})
                    _write_jsonl_atomic(attempts_path, attempts)
                    new_done = len(records) - len(ctx.imported_records) - initial_new
                    remaining = len(ctx.current.samples) - len(records)
                    eta = (completed - continuation_started) / max(1, new_done) * remaining
                    write_json(ctx.output_dir / "progress.json", {"complete": False,
                        "completed_samples": len(records), "total_samples": len(ctx.current.samples),
                        "imported_samples": len(ctx.imported_records),
                        "continued_samples": len(records) - len(ctx.imported_records),
                        "eval_batch_size": "mixed", "continuation_eval_batch_size": ctx.batch_size,
                        "accuracy_so_far": sum(bool(r["correct"]) for r in records.values()) / len(records),
                        "eta_seconds": eta})
                    print(f"[{len(records)}/{len(ctx.current.samples)}] sample={index} batch={ctx.batch_size} "
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
                        "completed_samples": len(records), "total_samples": len(ctx.current.samples),
                        "failed_sample_index": index, "error": str(error)})
                    raise

            batching = llm.async_batching(ctx.batch_size) if pending else nullcontext()
            async with batching:
                for offset in range(0, len(pending), ctx.batch_size):
                    tasks = [asyncio.create_task(one(i)) for i in pending[offset:offset + ctx.batch_size]]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
            if len(records) != len(ctx.current.samples):
                raise RuntimeError("Continuation is incomplete; refusing to publish accuracy")
            _check_source(ctx)
            summary = _summary(ctx, records, attempts)
            if summary_path.exists() and load_json(summary_path) != summary:
                raise ValueError("Published continuation summary differs from checkpoints")
            write_json(summary_path, summary)
            write_json(ctx.output_dir / "progress.json", {"complete": True,
                "completed_samples": len(records), "total_samples": len(ctx.current.samples),
                "imported_samples": len(ctx.imported_records),
                "continued_samples": len(records) - len(ctx.imported_records),
                "eval_batch_size": "mixed", "continuation_eval_batch_size": ctx.batch_size, "eta_seconds": 0})
            return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-batch-size", type=int, choices=(4,), default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.prepare_only:
        parser.error("Choose only one of --dry-run and --prepare-only")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    result = asyncio.run(evaluate(parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
