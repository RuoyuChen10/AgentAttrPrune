"""Collect dense MMLU traces for every selected calibration sample."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from tqdm import tqdm

from AgentAttrPrune.datasets import MMLUDataset
from AgentAttrPrune.graph import MultiAgentGraph
from AgentAttrPrune.llm import HFChat
from AgentAttrPrune.tracing import GenerationTraceRecorder
from experiments_attrprune.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL,
    ExperimentPaths,
    append_jsonl,
    build_run_config,
    canonical_model_name,
    completed_attempt_indices,
    ensure_run_config,
    hf_model_reference,
    load_jsonl,
    mmlu_data_identity,
    normalize_answer,
    resolve_run_dir,
    worker_specs_from_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified dense MMLU graph, record every attempted sample, "
            "and save a full generation trace for every completed prediction."
        )
    )
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split", choices=["dev", "val", "test"], default="dev")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=285)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _metrics_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def latest_completed_attempts(path: Path) -> dict[int, dict]:
    completed = {}
    for item in load_jsonl(path):
        if item.get("status") == "completed":
            completed[int(item["sample_index"])] = item
    return completed


def write_summary(
    *,
    path: Path,
    attempts_path: Path,
    config: dict,
    split: str,
    start: int,
    stop: int,
) -> Path:
    completed = latest_completed_attempts(attempts_path)
    selected = [item for index, item in completed.items() if start <= index < stop]
    correct = sum(bool(item.get("correct")) for item in selected)
    return write_json(
        path,
        {
            "schema_version": 1,
            "run_fingerprint": config["fingerprint"],
            "split": split,
            "start_index": start,
            "stop_index_exclusive": stop,
            "attempted": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else 0.0,
            "prompt_tokens": sum(
                int((item.get("metrics") or {}).get("prompt_tokens", 0))
                for item in selected
            ),
            "generated_tokens": sum(
                int((item.get("metrics") or {}).get("generated_tokens", 0))
                for item in selected
            ),
            "attempts_jsonl": str(attempts_path),
        },
    )


async def collect(args: argparse.Namespace) -> dict:
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
    paths = ExperimentPaths(run_dir)
    dataset = MMLUDataset(
        split=args.split,
        data_dir=args.data_dir,
        download=not args.no_download,
    )
    if args.start_index >= len(dataset):
        raise ValueError(
            f"--start-index {args.start_index} is outside dataset size {len(dataset)}"
        )
    data_identity = mmlu_data_identity(
        args.data_dir,
        splits=(args.split, "val"),
    )
    config = build_run_config(
        model_name=args.model_name,
        num_workers=args.num_workers,
        num_rounds=args.num_rounds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        calibration_split=args.split,
        evaluation_split="val",
        dataset_identity=data_identity,
    )
    config = ensure_run_config(paths.run_config, config)
    attempts_path = paths.calibration / "attempts.jsonl"
    correct_path = paths.calibration / "correct_results.jsonl"
    existing_attempts = load_jsonl(attempts_path)
    if existing_attempts and not args.resume:
        raise FileExistsError(
            f"Calibration attempts already exist at {attempts_path}. "
            "Use --resume or choose another RUN_NAME."
        )
    completed = completed_attempt_indices(attempts_path) if args.resume else set()

    start = args.start_index
    stop = min(len(dataset), start + args.limit)
    llm = HFChat(hf_model_reference(args.model_name))
    graph = MultiAgentGraph(
        llm,
        worker_specs_from_config(config),
        num_rounds=args.num_rounds,
        policy=None,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    truncated_sample_indices: list[int] = []

    for sample_index in tqdm(range(start, stop), desc="dense dev calibration"):
        if sample_index in completed:
            continue
        set_reproducible(args.seed + sample_index)
        record = dataset[sample_index]
        sample = dataset.record_to_sample(record)
        trace_path = paths.traces / f"sample_{sample_index:05d}" / "trace.json"
        recorder = GenerationTraceRecorder(
            trace_path,
            config={
                **config,
                "phase": "calibration",
                "sample_index": sample_index,
            },
            sample=sample,
        )
        started = time.perf_counter()
        try:
            result = await graph.run_sample(
                sample,
                recorder=recorder,
                phase="calibration",
                sample_index=sample_index,
            )
            parsed = normalize_answer(result.final_answer)
            gold = normalize_answer(sample["answer"])
            correct = parsed == gold
            metrics = _metrics_dict(result.metrics)
            metrics["sample_elapsed_seconds"] = time.perf_counter() - started
            trace_final = result.to_trace_final()
            trace_final.update(
                {
                    "parsed_answer": parsed,
                    "gold_answer": gold,
                    "correct": correct,
                }
            )
            truncated_events = int(metrics.get("truncated_events", 0))
            if truncated_events:
                trace_final.update(
                    {
                        "generation_complete": False,
                        "truncated_event_count": truncated_events,
                    }
                )
                recorder.write(trace_final)
                truncated_sample_indices.append(sample_index)
                append_jsonl(
                    attempts_path,
                    {
                        "sample_index": sample_index,
                        "status": "truncated",
                        "correct": False,
                        "prediction_correct": correct,
                        "gold_answer": gold,
                        "parsed_answer": parsed,
                        "trace_json": str(trace_path),
                        "metrics": metrics,
                        "error": (
                            f"{truncated_events} node output(s) hit "
                            f"max_new_tokens={args.max_new_tokens}"
                        ),
                    },
                )
                continue
            recorder.write(trace_final)
            if correct:
                append_jsonl(
                    correct_path,
                    {
                        "sample_index": sample_index,
                        "trace_json": str(trace_path),
                        "gold_answer": gold,
                        "parsed_answer": parsed,
                        "metrics": metrics,
                    },
                )
            append_jsonl(
                attempts_path,
                {
                    "sample_index": sample_index,
                    "status": "completed",
                    "correct": correct,
                    "gold_answer": gold,
                    "parsed_answer": parsed,
                    "trace_json": str(trace_path),
                    "metrics": metrics,
                },
            )
        except Exception as error:
            append_jsonl(
                attempts_path,
                {
                    "sample_index": sample_index,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            raise

    summary_path = write_summary(
        path=paths.calibration / "summary.json",
        attempts_path=attempts_path,
        config=config,
        split=args.split,
        start=start,
        stop=stop,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Saved calibration summary to {summary_path}")
    if truncated_sample_indices:
        raise RuntimeError(
            "Calibration saved truncated traces for sample(s) "
            f"{truncated_sample_indices}; no attribution should run. Increase "
            "--max-new-tokens and use a new RUN_NAME."
        )
    return summary


def main() -> None:
    asyncio.run(collect(parse_args()))


if __name__ == "__main__":
    main()
