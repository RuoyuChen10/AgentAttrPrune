"""Evaluate dense or attribution-pruned graphs on a benchmark selection."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from AgentAttrPrune.datasets import (
    BENCHMARKS,
    GSM8KDataset,
    HumanEvalDataset,
    MMLUDataset,
    benchmark_definition,
    evaluate_prediction,
)
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import HFChat
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune.common import (
    DEFAULT_MODEL,
    ExperimentPaths,
    benchmark_data_identity,
    canonical_model_name,
    config_fingerprint,
    dataset_identities_match,
    default_data_dir,
    hf_model_reference,
    load_json,
    resolve_run_dir,
    reviewer_agent_ids_from_config,
    validate_prompt_contract,
    worker_specs_from_config,
    write_json,
)


EVALUATION_MODES = ("dense", "binary", "token_ratio", "word_ratio")
POLICY_FILENAMES = {
    "binary": "pruned_policy_binary.json",
    "token_ratio": "pruned_policy_token_ratio.json",
    "word_ratio": "pruned_policy_word_ratio.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the same frozen AgentAttrPrune graph configuration with "
            "a dense, binary-attribution, token-ratio-attribution, or "
            "word-ratio-attribution policy."
        )
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="mmlu")
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=EVALUATION_MODES, required=True)
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=None,
        help=(
            "Use this explicit policy instead of attribution/pruned_policy_<mode>.json. "
            "This is useful for evaluating multiple pruning rates of one metric."
        ),
    )
    parser.add_argument(
        "--evaluation-name",
        default=None,
        help=(
            "Optional output-directory prefix. The split is appended automatically; "
            "for example static_token_ratio_prune_05pct becomes "
            "static_token_ratio_prune_05pct_val."
        ),
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-index",
        action="append",
        type=int,
        default=[],
        help=(
            "Evaluate one exact dataset index. Repeat for a sparse selection. "
            "When provided, this replaces the contiguous --start-index/--limit "
            "selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help=(
            "Maximum concurrent questions and HF generation batch size. "
            "1 preserves serial inference; larger values require temperature=0. "
            "Batched GPU arithmetic may change greedy outputs."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from validated per-sample predictions.jsonl checkpoints. "
            "The run fingerprint, policy, dataset selection, seed, and trace mode "
            "must match the original evaluation exactly."
        ),
    )
    parser.add_argument(
        "--defer-index",
        action="append",
        type=int,
        default=[],
        help=(
            "Temporarily defer one sample index during this invocation while "
            "keeping it in the final evaluation selection. Repeat this option "
            "for multiple indices. No final summary is written until every "
            "deferred sample is completed by a later --resume run."
        ),
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Write one schema-v1 generation trace for every evaluated sample.",
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    definition = benchmark_definition(args.benchmark)
    args.split = args.split or definition.evaluation_split
    if args.limit is None:
        args.limit = definition.evaluation_limit
    return args


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Expected mapping-compatible metrics, got {type(value).__name__}")


def _load_run_config(paths: ExperimentPaths, args: argparse.Namespace) -> dict:
    if not paths.run_config.is_file():
        raise FileNotFoundError(
            f"Missing {paths.run_config}; run the traces stage before evaluation."
        )
    config = load_json(paths.run_config)
    declared_fingerprint = config.get("fingerprint")
    actual_fingerprint = config_fingerprint(config)
    if not declared_fingerprint or declared_fingerprint != actual_fingerprint:
        raise ValueError(
            "run_config.json has a missing or invalid fingerprint; refusing to "
            "evaluate a configuration whose identity cannot be verified."
        )

    configured_model = canonical_model_name(config["model_name"])
    requested_model = canonical_model_name(args.model_name)
    if configured_model != requested_model:
        raise ValueError(
            "Evaluation model must match run_config.json exactly: "
            f"configured={configured_model!r}, requested={requested_model!r}"
        )
    if int(config["seed"]) != int(args.seed):
        raise ValueError(
            "Evaluation seed must match run_config.json exactly: "
            f"configured={config['seed']!r}, requested={args.seed!r}"
        )
    requested_benchmark = str(getattr(args, "benchmark", "mmlu")).lower()
    configured_benchmark = str(config.get("benchmark", "mmlu")).lower()
    if configured_benchmark != requested_benchmark:
        raise ValueError(
            "Evaluation benchmark must match run_config.json exactly: "
            f"configured={configured_benchmark!r}, "
            f"requested={requested_benchmark!r}"
        )
    validate_prompt_contract(config, benchmark=configured_benchmark)
    configured_split = str((config.get("splits") or {}).get("evaluation", ""))
    if configured_split != args.split:
        raise ValueError(
            "Evaluation split must match run_config.json exactly: "
            f"configured={configured_split!r}, requested={args.split!r}"
        )
    return config


def _resolve_policy(
    mode: str,
    *,
    paths: ExperimentPaths,
    config: Mapping[str, Any],
    policy_json: Path | None = None,
) -> tuple[RoundPolicy, Path | None]:
    workers = list(config["workers"])
    agent_ids = [str(item["agent_id"]) for item in workers]
    agent_roles = [str(item["role"]) for item in workers]
    agent_types = [str(item.get("agent_type", "AnalyzeAgent")) for item in workers]
    num_rounds = int(config["num_rounds"])

    if mode == "dense":
        if policy_json is not None:
            raise ValueError("--policy-json cannot be used with --mode dense")
        policy = RoundPolicy.dense(
            agent_ids,
            num_rounds=num_rounds,
            agent_roles=agent_roles,
            agent_types=agent_types,
        )
        return policy, None
    if policy_json is not None:
        source_path = Path(policy_json)
    else:
        try:
            source_path = paths.attribution / POLICY_FILENAMES[mode]
        except KeyError as error:
            raise ValueError(f"Unsupported evaluation mode: {mode!r}") from error
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing attribution policy {source_path}; run the attribution stage first."
        )
    policy = RoundPolicy.from_json(source_path)
    policy.validate_agents(agent_ids)
    if policy.num_rounds != num_rounds:
        raise ValueError(
            f"Policy has {policy.num_rounds} rounds, run_config has {num_rounds}"
        )
    if policy.agent_roles and tuple(agent_roles) != policy.agent_roles:
        raise ValueError("Policy roles do not match run_config worker roles")
    if policy.agent_types and tuple(agent_types) != policy.agent_types:
        raise ValueError("Policy agent types do not match run_config worker types")
    if policy.metadata.get("run_fingerprint") != config["fingerprint"]:
        raise ValueError(
            "Attribution policy run fingerprint does not match run_config.json"
        )
    if policy.metadata.get("metric") != mode:
        raise ValueError(
            f"Attribution policy metric {policy.metadata.get('metric')!r} "
            f"does not match evaluation mode {mode!r}"
        )
    return policy, source_path


def _evaluation_directory_name(
    mode: str,
    split: str,
    evaluation_name: str | None,
) -> str:
    prefix = evaluation_name or mode
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prefix):
        raise ValueError(
            "--evaluation-name must contain only letters, numbers, '.', '_' or '-', "
            "and must start with a letter or number"
        )
    return f"{prefix}_{split}"


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_state(
    *,
    config: Mapping[str, Any],
    policy: RoundPolicy,
    mode: str,
    evaluation_name: str,
    evaluation_directory: str,
    benchmark: str,
    split: str,
    start: int,
    stop: int,
    dataset_size: int,
    seed: int,
    save_traces: bool,
    sample_indices: Sequence[int] | None = None,
    eval_batch_size: int = 1,
) -> dict[str, Any]:
    state = {
        "schema_version": 1,
        "run_fingerprint": str(config["fingerprint"]),
        "policy_fingerprint": _json_fingerprint(policy.to_dict()),
        "mode": mode,
        "evaluation_name": evaluation_name,
        "evaluation_directory": evaluation_directory,
        "benchmark": benchmark,
        "split": split,
        "start_index": start,
        "stop_index_exclusive": stop,
        "dataset_size": dataset_size,
        "seed": seed,
        "save_traces": save_traces,
        "eval_batch_size": eval_batch_size,
    }
    if sample_indices is not None:
        state["sample_indices"] = [int(index) for index in sample_indices]
    return state


def _ensure_evaluation_state(
    path: Path,
    expected: Mapping[str, Any],
    *,
    predictions_path: Path,
) -> Path:
    if path.is_file():
        current = load_json(path)
        # Evaluations written before batching was introduced were serial.
        current.setdefault("eval_batch_size", 1)
        if current != dict(expected):
            differing = sorted(
                key
                for key in set(current) | set(expected)
                if current.get(key) != expected.get(key)
            )
            raise ValueError(
                f"Evaluation checkpoint identity mismatch at {path}; changed "
                f"fields: {', '.join(differing)}. Choose another evaluation name "
                "or remove the incompatible output directory."
            )
        return path
    if predictions_path.is_file() and predictions_path.stat().st_size:
        raise ValueError(
            f"Cannot safely resume {predictions_path}: evaluation_state.json is "
            "missing, so the checkpoint identity cannot be verified."
        )
    return write_json(path, dict(expected))


def _load_prediction_checkpoints(
    path: Path,
    *,
    start: int,
    stop: int,
    selected_indices: Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    selected = (
        {int(index) for index in selected_indices}
        if selected_indices is not None
        else None
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON checkpoint at {path}:{line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"Checkpoint at {path}:{line_number} must be a JSON object"
                )
            if "sample_index" not in record:
                raise ValueError(
                    f"Checkpoint at {path}:{line_number} has no sample_index"
                )
            sample_index = int(record["sample_index"])
            if not start <= sample_index < stop:
                raise ValueError(
                    f"Checkpoint sample {sample_index} is outside [{start}, {stop})"
                )
            if selected is not None and sample_index not in selected:
                raise ValueError(
                    f"Checkpoint sample {sample_index} is outside the exact "
                    "sample-index selection"
                )
            if sample_index in records:
                raise ValueError(
                    f"Duplicate checkpoint for sample_index {sample_index} at {path}"
                )
            required = {
                "gold_answer",
                "parsed_answer",
                "correct",
                "output",
                "metrics",
                "edge_counts",
            }
            missing = sorted(required - set(record))
            if missing:
                raise ValueError(
                    f"Checkpoint sample {sample_index} is missing: {', '.join(missing)}"
                )
            records[sample_index] = dict(record)
    return records


def _common_edge_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    common: dict[str, Any] | None = None
    for record in records:
        edge_counts = _mapping(record.get("edge_counts"))
        if common is None:
            common = edge_counts
        elif edge_counts != common:
            raise ValueError("Evaluation checkpoints contain inconsistent edge counts")
    return common or {}


def _trace_final(result: Any, *, gold: str, parsed: str, correct: bool) -> dict:
    if hasattr(result, "to_trace_final"):
        final = dict(result.to_trace_final())
    else:  # Makes custom graph backends obey the same trace contract.
        final = {
            "output": str(result.final_answer),
            "parsed_answer": parsed,
            "metrics": _mapping(result.metrics),
            "edge_counts": _mapping(result.edge_counts),
        }
    final.update(
        {
            "parsed_answer": parsed,
            "gold_answer": gold,
            "correct": correct,
        }
    )
    return final


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    eval_batch_size = int(getattr(args, "eval_batch_size", 1))
    if eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    benchmark = str(getattr(args, "benchmark", "mmlu")).lower()
    definition = benchmark_definition(benchmark)
    requested_data_dir = getattr(args, "data_dir", None)
    split = getattr(args, "split", None) or definition.evaluation_split
    limit = getattr(args, "limit", None)
    if limit is None:
        limit = definition.evaluation_limit
    args.benchmark = benchmark
    args.split = split
    if args.start_index < 0 or limit < 1:
        raise ValueError("--start-index must be non-negative and --limit positive")

    os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
    os.environ["HF_CHAT_CAPTURE_SCORES"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_reproducible(args.seed)

    run_dir = args.run_dir or resolve_run_dir(
        model_name=args.model_name,
        seed=args.seed,
        results_root=args.results_root,
        run_name=args.run_name,
        benchmark=benchmark,
    )
    paths = ExperimentPaths(Path(run_dir))
    config = _load_run_config(paths, args)
    if eval_batch_size > 1 and float(config["decoding"]["temperature"]) != 0.0:
        raise ValueError(
            "--eval-batch-size > 1 requires temperature=0; concurrent sampling "
            "would change the per-sample random-number stream. Use batch size 1."
        )
    configured_dataset = config.get("dataset") or {}
    configured_data_dir = configured_dataset.get("data_root") or configured_dataset.get(
        "requested_data_dir"
    )
    data_dir = Path(
        requested_data_dir or configured_data_dir or default_data_dir(benchmark)
    )
    policy, source_policy = _resolve_policy(
        args.mode,
        paths=paths,
        config=config,
        policy_json=getattr(args, "policy_json", None),
    )

    dataset_classes = {
        "mmlu": MMLUDataset,
        "gsm8k": GSM8KDataset,
        "humaneval": HumanEvalDataset,
    }
    dataset = dataset_classes[benchmark](
        split=split,
        data_dir=data_dir,
        download=not args.no_download,
    )
    expected_data = config.get("dataset")
    if expected_data:
        actual_data = benchmark_data_identity(
            benchmark,
            data_dir,
            splits=(expected_data.get("splits") or {}).keys(),
            shuffle_seed=int(expected_data.get("shuffle_seed", 888)),
        )
        if not dataset_identities_match(expected_data, actual_data):
            raise ValueError(
                f"{benchmark} data identity differs from run_config.json; refusing to "
                "mix a different data directory or dataset snapshot into this run."
            )
    exact_sample_indices = list(getattr(args, "sample_index", None) or ())
    if exact_sample_indices:
        if len(exact_sample_indices) != len(set(exact_sample_indices)):
            raise ValueError("--sample-index values must be unique")
        invalid_indices = sorted(
            index for index in exact_sample_indices if not 0 <= index < len(dataset)
        )
        if invalid_indices:
            raise ValueError(
                f"--sample-index values are outside dataset size {len(dataset)}: "
                + ", ".join(str(index) for index in invalid_indices)
            )
        selected_indices = sorted(exact_sample_indices)
        start = selected_indices[0]
        stop = selected_indices[-1] + 1
    else:
        if args.start_index >= len(dataset):
            raise ValueError(
                f"--start-index {args.start_index} is outside dataset size {len(dataset)}"
            )
        start = args.start_index
        stop = min(len(dataset), start + limit)
        selected_indices = list(range(start, stop))

    evaluation_name = getattr(args, "evaluation_name", None)
    evaluation_dir_name = _evaluation_directory_name(
        args.mode, split, evaluation_name
    )
    evaluation_dir = paths.evaluation / evaluation_dir_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    resolved_policy_path = evaluation_dir / "resolved_policy.json"
    predictions_path = evaluation_dir / "predictions.jsonl"
    state_path = evaluation_dir / "evaluation_state.json"
    resolved_evaluation_name = evaluation_name or args.mode
    expected_state = _evaluation_state(
        config=config,
        policy=policy,
        mode=args.mode,
        evaluation_name=resolved_evaluation_name,
        evaluation_directory=evaluation_dir_name,
        benchmark=benchmark,
        split=split,
        start=start,
        stop=stop,
        dataset_size=len(dataset),
        seed=int(args.seed),
        save_traces=bool(args.save_traces),
        sample_indices=selected_indices if exact_sample_indices else None,
        eval_batch_size=eval_batch_size,
    )
    _ensure_evaluation_state(
        state_path,
        expected_state,
        predictions_path=predictions_path,
    )
    resolved_policy_path = policy.to_json(resolved_policy_path)
    checkpoint_records = _load_prediction_checkpoints(
        predictions_path,
        start=start,
        stop=stop,
        selected_indices=selected_indices if exact_sample_indices else None,
    )
    resume = bool(getattr(args, "resume", False))
    if checkpoint_records and not resume:
        raise FileExistsError(
            f"Evaluation checkpoints already exist at {predictions_path}. "
            "Use --resume to skip completed samples or choose another "
            "--evaluation-name."
        )
    resumed_samples = len(checkpoint_records)
    if resumed_samples:
        print(
            f"Validated {resumed_samples}/{len(selected_indices)} completed checkpoints "
            f"from {predictions_path}; resuming missing samples only."
        )
    deferred_indices = set(getattr(args, "defer_index", None) or ())
    invalid_deferred = sorted(
        sample_index
        for sample_index in deferred_indices
        if sample_index not in selected_indices
    )
    if invalid_deferred:
        raise ValueError(
            "--defer-index values are outside the evaluation selection: "
            f"{invalid_deferred}"
        )
    deferred_pending = sorted(deferred_indices - set(checkpoint_records))
    if deferred_pending:
        print(
            "Deferring sample indices for this invocation: "
            + ", ".join(str(index) for index in deferred_pending)
        )
    pending_indices = [
        sample_index
        for sample_index in selected_indices
        if sample_index not in checkpoint_records
        and sample_index not in deferred_indices
    ]

    decoding = config["decoding"]
    graph = None
    if pending_indices:
        llm = HFChat(hf_model_reference(config["model_name"]))
        graph_kwargs = {
            "num_rounds": int(config["num_rounds"]),
            "policy": policy,
            "temperature": float(decoding["temperature"]),
            "max_new_tokens": int(decoding["max_new_tokens"]),
        }
        if callable(getattr(definition.prompt_set, "prepare_reviewer_agent_ids", None)):
            graph_kwargs["reviewer_agent_ids"] = reviewer_agent_ids_from_config(
                config
            )
        if benchmark != "mmlu":
            graph_kwargs["prompt_set"] = definition.prompt_set
        graph = MultiAgentGraph(
            llm,
            worker_specs_from_config(config),
            **graph_kwargs,
        )

    progress = tqdm(
        total=len(selected_indices),
        initial=resumed_samples,
        desc=f"{args.mode} {benchmark} {split} evaluation",
    )
    invocation_started_at = time.perf_counter()
    last_checkpoint_at = invocation_started_at
    common_edge_counts = _common_edge_counts(list(checkpoint_records.values()))

    async def evaluate_sample(sample_index: int) -> None:
        nonlocal last_checkpoint_at, common_edge_counts
        if eval_batch_size == 1:
            set_reproducible(args.seed + sample_index)
        record = dataset[sample_index]
        sample = dataset.record_to_sample(record)
        trace_path = (
            evaluation_dir / "traces" / f"sample_{sample_index:05d}" / "trace.json"
        )
        recorder = None
        if args.save_traces:
            recorder = GenerationTraceRecorder(
                trace_path,
                config={
                    **config,
                    "phase": "evaluation",
                    "evaluation_mode": args.mode,
                    "evaluation_name": evaluation_name or args.mode,
                    "eval_batch_size": eval_batch_size,
                    "sample_index": sample_index,
                },
                sample=sample,
            )

        sample_started_at = time.perf_counter()
        assert graph is not None
        result = await graph.run_sample(
            sample,
            recorder=recorder,
            phase="evaluation",
            sample_index=sample_index,
        )
        outcome = evaluate_prediction(benchmark, result.final_answer, sample)
        parsed = str(outcome["parsed_answer"])
        gold = str(outcome["gold_answer"])
        is_correct = bool(outcome["correct"])
        metrics = _mapping(result.metrics)
        edge_counts = _mapping(result.edge_counts)
        if common_edge_counts and edge_counts != common_edge_counts:
            raise RuntimeError("Graph edge counts changed between evaluation samples")
        common_edge_counts = edge_counts

        if recorder is not None:
            recorder.write(
                _trace_final(result, gold=gold, parsed=parsed, correct=is_correct)
            )
        completed_at = time.perf_counter()
        sample_elapsed = completed_at - sample_started_at
        checkpoint_records[sample_index] = {
            "sample_index": sample_index,
            "subject": str(sample.get("subject", "")),
            "task_id": str(sample.get("task_id", "")),
            "gold_answer": gold,
            "parsed_answer": parsed,
            "correct": is_correct,
            "output": str(result.final_answer),
            "metrics": metrics,
            "edge_counts": edge_counts,
            "sample_elapsed_seconds": sample_elapsed,
            # Each completion owns the wall interval since the last checkpoint.
            # These intervals do not overlap, even when sample latencies do.
            "evaluation_elapsed_seconds": (
                completed_at - last_checkpoint_at
                if eval_batch_size > 1
                else sample_elapsed
            ),
            "evaluation_details": outcome.get("details") or {},
            "trace_json": str(trace_path) if recorder is not None else None,
        }
        _write_jsonl_atomic(
            predictions_path,
            [checkpoint_records[index] for index in sorted(checkpoint_records)],
        )
        last_checkpoint_at = completed_at
        progress.update(1)

    batching = (
        llm.async_batching(eval_batch_size)
        if pending_indices and eval_batch_size > 1
        else nullcontext()
    )
    try:
        async with batching:
            for offset in range(0, len(pending_indices), eval_batch_size):
                if eval_batch_size == 1:
                    await evaluate_sample(pending_indices[offset])
                    continue
                # Keep cohorts bounded, and preserve ordering inside each graph.
                # A completed sample checkpoints immediately, independently of
                # slower samples in its cohort.
                tasks = [
                    asyncio.create_task(evaluate_sample(index))
                    for index in pending_indices[offset : offset + eval_batch_size]
                ]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
    finally:
        progress.close()
    invocation_elapsed = time.perf_counter() - invocation_started_at

    missing_indices = [
        sample_index
        for sample_index in selected_indices
        if sample_index not in checkpoint_records
    ]
    if missing_indices:
        raise RuntimeError(
            "Evaluation checkpoint pass ended successfully, but the full selection "
            "is still incomplete. Missing sample indices: "
            + ", ".join(str(index) for index in missing_indices[:20])
            + ". Rerun with --resume after removing the corresponding "
            "--defer-index options; no final summary was written."
        )

    records = [checkpoint_records[index] for index in selected_indices]
    # Recompute every aggregate from the complete persisted record set. This
    # keeps resumed evaluations exact and independent of process boundaries.
    correct_count = sum(int(bool(record["correct"])) for record in records)
    llm_calls = sum(
        int(_mapping(record.get("metrics")).get("llm_calls", 0))
        for record in records
    )
    prompt_tokens = sum(
        int(_mapping(record.get("metrics")).get("prompt_tokens", 0))
        for record in records
    )
    generated_tokens = sum(
        int(_mapping(record.get("metrics")).get("generated_tokens", 0))
        for record in records
    )
    graph_elapsed = sum(
        float(_mapping(record.get("metrics")).get("elapsed_seconds", 0.0))
        for record in records
    )
    sample_elapsed_sum = sum(
        float(
            record.get(
                "sample_elapsed_seconds",
                _mapping(record.get("metrics")).get("elapsed_seconds", 0.0),
            )
        )
        for record in records
    )
    elapsed = sum(
        float(
            record.get(
                "evaluation_elapsed_seconds",
                record.get(
                    "sample_elapsed_seconds",
                    _mapping(record.get("metrics")).get("elapsed_seconds", 0.0),
                ),
            )
        )
        for record in records
    )
    edge_counts = _common_edge_counts(records)
    retained_edges = int(edge_counts.get("worker_active", 0))
    candidate_edges = int(edge_counts.get("worker_candidate", 0))
    edge_retention = float(
        edge_counts.get(
            "worker_retention",
            retained_edges / candidate_edges if candidate_edges else 1.0,
        )
    )
    predictions_path = _write_jsonl_atomic(predictions_path, records)
    batch_metrics = [_mapping(record.get("metrics")) for record in records]
    batch_recorded_samples = sum(
        "batch_metadata_calls" in metrics for metrics in batch_metrics
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": config["fingerprint"],
        "mode": args.mode,
        "evaluation_name": resolved_evaluation_name,
        "evaluation_directory": evaluation_dir_name,
        "benchmark": benchmark,
        "split": split,
        "start_index": start,
        "stop_index_exclusive": stop,
        "dataset_size": len(dataset),
        "total": len(records),
        "correct": correct_count,
        "accuracy": correct_count / len(records),
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "sample_elapsed_seconds_sum": sample_elapsed_sum,
        "invocation_elapsed_seconds": invocation_elapsed,
        "eval_batch_size": eval_batch_size,
        "generation_batching": {
            "configured_max_batch_size": eval_batch_size,
            "recorded_samples": batch_recorded_samples,
            "samples_without_batch_metadata": len(records) - batch_recorded_samples,
            "recorded_calls": sum(
                int(metrics.get("batch_metadata_calls", 0)) for metrics in batch_metrics
            ),
            "oom_fallback_samples": sum(
                int(metrics.get("oom_fallback_calls", 0)) > 0 for metrics in batch_metrics
            ),
            "oom_fallback_calls": sum(
                int(metrics.get("oom_fallback_calls", 0)) for metrics in batch_metrics
            ),
            "max_oom_split_depth": max(
                (int(metrics.get("oom_max_split_depth", 0)) for metrics in batch_metrics),
                default=0,
            ),
        },
        "graph_elapsed_seconds": graph_elapsed,
        "retained_edges": retained_edges,
        "candidate_edges": candidate_edges,
        "edge_retention": edge_retention,
        "edge_counts": edge_counts,
        "source_policy_json": str(source_policy) if source_policy else None,
        "resolved_policy_json": str(resolved_policy_path),
        "predictions_jsonl": str(predictions_path),
        "evaluation_state_json": str(state_path),
        "resumed": resume,
        "resumed_samples": resumed_samples,
        "checkpoint_interval_samples": 1,
        "traces_saved": bool(args.save_traces),
        "traces_dir": str(evaluation_dir / "traces") if args.save_traces else None,
    }
    if exact_sample_indices:
        summary["sample_indices"] = selected_indices
    summary_path = write_json(evaluation_dir / "evaluation_summary.json", summary)
    print(f"Saved {args.mode} evaluation summary to {summary_path}")
    return summary


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
