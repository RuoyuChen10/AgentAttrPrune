"""Evaluate dense or attribution-pruned graphs on a complete MMLU selection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from AgentAttrPrune.datasets import MMLUDataset
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import HFChat
from AgentAttrPrune.pruning import RoundPolicy
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    ExperimentPaths,
    canonical_model_name,
    config_fingerprint,
    hf_model_reference,
    load_json,
    mmlu_data_identity,
    normalize_answer,
    resolve_run_dir,
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
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mode", choices=EVALUATION_MODES, required=True)
    parser.add_argument("--split", choices=["dev", "val", "test"], default="val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1531)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Write one schema-v1 generation trace for every evaluated sample.",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


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
) -> tuple[RoundPolicy, Path | None]:
    workers = list(config["workers"])
    agent_ids = [str(item["agent_id"]) for item in workers]
    agent_roles = [str(item["role"]) for item in workers]
    agent_types = [str(item.get("agent_type", "AnalyzeAgent")) for item in workers]
    num_rounds = int(config["num_rounds"])

    if mode == "dense":
        policy = RoundPolicy.dense(
            agent_ids,
            num_rounds=num_rounds,
            agent_roles=agent_roles,
            agent_types=agent_types,
        )
        return policy, None
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
    if args.start_index < 0 or args.limit < 1:
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
    )
    paths = ExperimentPaths(Path(run_dir))
    config = _load_run_config(paths, args)
    policy, source_policy = _resolve_policy(args.mode, paths=paths, config=config)

    dataset = MMLUDataset(
        split=args.split,
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    expected_data = config.get("dataset")
    if expected_data:
        actual_data = mmlu_data_identity(
            args.data_dir,
            splits=(expected_data.get("splits") or {}).keys(),
            shuffle_seed=int(expected_data.get("shuffle_seed", 888)),
        )
        if actual_data != expected_data:
            raise ValueError(
                "MMLU data identity differs from run_config.json; refusing to "
                "mix a different data directory or CSV snapshot into this run."
            )
    if args.start_index >= len(dataset):
        raise ValueError(
            f"--start-index {args.start_index} is outside dataset size {len(dataset)}"
        )
    start = args.start_index
    stop = min(len(dataset), start + args.limit)

    evaluation_dir = paths.evaluation / f"{args.mode}_{args.split}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    resolved_policy_path = policy.to_json(evaluation_dir / "resolved_policy.json")

    decoding = config["decoding"]
    llm = HFChat(hf_model_reference(config["model_name"]))
    graph = MultiAgentGraph(
        llm,
        worker_specs_from_config(config),
        num_rounds=int(config["num_rounds"]),
        policy=policy,
        temperature=float(decoding["temperature"]),
        max_new_tokens=int(decoding["max_new_tokens"]),
    )

    records: list[dict[str, Any]] = []
    correct_count = 0
    llm_calls = 0
    prompt_tokens = 0
    generated_tokens = 0
    graph_elapsed = 0.0
    common_edge_counts: dict[str, Any] | None = None
    started_at = time.perf_counter()

    indices = tqdm(
        range(start, stop),
        desc=f"{args.mode} {args.split} evaluation",
    )
    for sample_index in indices:
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
                    "sample_index": sample_index,
                },
                sample=sample,
            )

        result = await graph.run_sample(
            sample,
            recorder=recorder,
            phase="evaluation",
            sample_index=sample_index,
        )
        parsed = normalize_answer(result.final_answer)
        gold = normalize_answer(sample["answer"])
        is_correct = parsed == gold
        metrics = _mapping(result.metrics)
        edge_counts = _mapping(result.edge_counts)
        if common_edge_counts is None:
            common_edge_counts = edge_counts
        elif edge_counts != common_edge_counts:
            raise RuntimeError("Graph edge counts changed between evaluation samples")

        if recorder is not None:
            recorder.write(
                _trace_final(result, gold=gold, parsed=parsed, correct=is_correct)
            )
        correct_count += int(is_correct)
        llm_calls += int(metrics.get("llm_calls", 0))
        prompt_tokens += int(metrics.get("prompt_tokens", 0))
        generated_tokens += int(metrics.get("generated_tokens", 0))
        graph_elapsed += float(metrics.get("elapsed_seconds", 0.0))
        records.append(
            {
                "sample_index": sample_index,
                "subject": str(sample.get("subject", "")),
                "gold_answer": gold,
                "parsed_answer": parsed,
                "correct": is_correct,
                "output": str(result.final_answer),
                "metrics": metrics,
                "trace_json": str(trace_path) if recorder is not None else None,
            }
        )

    elapsed = time.perf_counter() - started_at
    edge_counts = common_edge_counts or {}
    retained_edges = int(edge_counts.get("worker_active", 0))
    candidate_edges = int(edge_counts.get("worker_candidate", 0))
    edge_retention = float(
        edge_counts.get(
            "worker_retention",
            retained_edges / candidate_edges if candidate_edges else 1.0,
        )
    )
    predictions_path = _write_jsonl_atomic(
        evaluation_dir / "predictions.jsonl", records
    )
    summary = {
        "schema_version": 1,
        "run_fingerprint": config["fingerprint"],
        "mode": args.mode,
        "split": args.split,
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
        "graph_elapsed_seconds": graph_elapsed,
        "retained_edges": retained_edges,
        "candidate_edges": candidate_edges,
        "edge_retention": edge_retention,
        "edge_counts": edge_counts,
        "source_policy_json": str(source_policy) if source_policy else None,
        "resolved_policy_json": str(resolved_policy_path),
        "predictions_jsonl": str(predictions_path),
        "traces_saved": bool(args.save_traces),
        "traces_dir": str(evaluation_dir / "traces") if args.save_traces else None,
    }
    summary_path = write_json(evaluation_dir / "evaluation_summary.json", summary)
    print(f"Saved {args.mode} evaluation summary to {summary_path}")
    return summary


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
