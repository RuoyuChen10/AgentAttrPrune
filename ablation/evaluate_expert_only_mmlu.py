"""Evaluate the first-round knowledgeable expert without the MAS or reviewer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from AgentAttrPrune.agents import AnalyzeAgent
from AgentAttrPrune.datasets import MMLUDataset
from AgentAttrPrune.llm import HFChat
from AgentAttrPrune.prompt import MMLUPromptSet
from experiments_attrprune.common import (
    config_fingerprint,
    hf_model_reference,
    load_json,
    load_jsonl,
    worker_specs_from_config,
    write_json,
)
from experiments_attrprune.evaluate_mmlu import set_reproducible


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the first configured MMLU worker (the knowledgeable "
            "expert), using its round-0 prompt with no incoming messages."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Existing AgentAttrPrune run containing run_config.json.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="MMLU split; defaults to the run's configured evaluation split.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of records to evaluate; defaults to the rest of the split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/evaluation/expert_only_SPLIT.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated per-sample checkpoints already in the output directory.",
    )
    return parser.parse_args(argv)


def _state_payload(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    expert = worker_specs_from_config(config)[0]
    decoding = dict(config["decoding"])
    dataset = dict(config["dataset"])
    return {
        "schema_version": 1,
        "kind": "expert_only_mmlu_evaluation",
        "run_fingerprint": str(config["fingerprint"]),
        "model_name": str(config["model_name"]),
        "split": split,
        "data_root": str(dataset["data_root"]),
        "shuffle_seed": dataset.get("shuffle_seed", 888),
        "seed": int(config["seed"]),
        "expert": {
            "agent_id": expert.id,
            "role": expert.role,
            "agent_type": expert.agent_type,
        },
        "decoding": {
            "temperature": float(decoding["temperature"]),
            "max_new_tokens": int(decoding["max_new_tokens"]),
            "enable_thinking": bool(decoding.get("enable_thinking", False)),
        },
    }


def _load_checkpoints(
    path: Path,
    *,
    dataset: MMLUDataset,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for record in load_jsonl(path):
        sample_index = int(record["sample_index"])
        if sample_index in records:
            raise ValueError(f"Duplicate checkpoint for sample {sample_index}: {path}")
        if sample_index < 0 or sample_index >= len(dataset):
            raise ValueError(
                f"Checkpoint sample index is outside the dataset: {sample_index}"
            )
        sample = dataset.record_to_sample(dataset[sample_index])
        gold = MMLUPromptSet.normalize_answer(sample["answer"])
        if str(record.get("gold_answer", "")) != gold:
            raise ValueError(
                "Checkpoint gold answer disagrees with the dataset at "
                f"sample {sample_index}"
            )
        records[sample_index] = dict(record)
    return records


async def evaluate(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration: {config_path}")
    config = load_json(config_path)
    if str(config.get("benchmark")) != "mmlu":
        raise ValueError("Expert-only ablation currently supports MMLU runs only")
    if config.get("fingerprint") != config_fingerprint(config):
        raise ValueError("run_config.json has a missing or invalid fingerprint")

    split = args.split or str(config["splits"]["evaluation"])
    dataset_config = dict(config["dataset"])
    dataset = MMLUDataset(
        split=split,
        data_dir=dataset_config["data_root"],
        shuffle_seed=dataset_config.get("shuffle_seed", 888),
        download=False,
    )
    start = int(args.start_index)
    if start < 0 or start >= len(dataset):
        raise ValueError(f"--start-index must be in [0, {len(dataset) - 1}]")
    stop = len(dataset) if args.limit is None else min(len(dataset), start + args.limit)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    selected_indices = list(range(start, stop))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "evaluation" / f"expert_only_{split}"
    )
    predictions_path = output_dir / "predictions.jsonl"
    state_path = output_dir / "evaluation_state.json"
    summary_path = output_dir / "evaluation_summary.json"
    expected_state = _state_payload(config, split)

    if state_path.exists():
        current_state = load_json(state_path)
        if current_state != expected_state:
            raise ValueError(
                f"Existing evaluation state does not match this run: {state_path}"
            )
    else:
        write_json(state_path, expected_state)

    if (
        predictions_path.exists()
        and predictions_path.stat().st_size
        and not args.resume
    ):
        raise FileExistsError(
            f"Checkpoints already exist: {predictions_path}; rerun with --resume"
        )
    checkpoint_records = _load_checkpoints(
        predictions_path,
        dataset=dataset,
    )
    resumed_samples = sum(index in checkpoint_records for index in selected_indices)
    pending_indices = [
        sample_index
        for sample_index in selected_indices
        if sample_index not in checkpoint_records
    ]

    if pending_indices:
        enable_thinking = bool(config["decoding"].get("enable_thinking", False))
        os.environ["HF_CHAT_ENABLE_THINKING"] = "1" if enable_thinking else "0"
        llm = HFChat(hf_model_reference(config["model_name"]))
        expert = AnalyzeAgent(
            worker_specs_from_config(config)[0],
            llm,
            MMLUPromptSet,
        )
        decoding = dict(config["decoding"])
        progress = tqdm(
            pending_indices,
            total=len(selected_indices),
            initial=resumed_samples,
            desc=f"expert-only mmlu {split}",
        )
        for sample_index in progress:
            set_reproducible(int(config["seed"]) + sample_index)
            sample = dataset.record_to_sample(dataset[sample_index])
            started_at = time.perf_counter()
            response = await expert.arun(
                task=str(sample["task"]),
                spatial_incoming=[],
                temporal_incoming=[],
                max_new_tokens=int(decoding["max_new_tokens"]),
                temperature=float(decoding["temperature"]),
            )
            parsed = MMLUPromptSet.normalize_answer(response.output)
            gold = MMLUPromptSet.normalize_answer(sample["answer"])
            record = {
                "sample_index": sample_index,
                "subject": str(sample.get("subject", "")),
                "gold_answer": gold,
                "parsed_answer": parsed,
                "correct": parsed == gold,
                "output": response.output,
                "elapsed_seconds": time.perf_counter() - started_at,
            }
            checkpoint_records[sample_index] = record
            ordered = [
                checkpoint_records[index]
                for index in sorted(checkpoint_records)
            ]
            temporary = predictions_path.with_suffix(".jsonl.tmp")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                for item in ordered:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(predictions_path)
            selected_so_far = [
                checkpoint_records[index]
                for index in selected_indices
                if index in checkpoint_records
            ]
            correct_so_far = sum(bool(item["correct"]) for item in selected_so_far)
            progress.set_postfix(
                accuracy=f"{correct_so_far / len(selected_so_far):.2%}"
            )

    records = [checkpoint_records[index] for index in selected_indices]
    correct = sum(bool(record["correct"]) for record in records)
    summary = {
        **expected_state,
        "start_index": start,
        "stop_index_exclusive": stop,
        "dataset_size": len(dataset),
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "resumed": bool(args.resume),
        "resumed_samples": resumed_samples,
        "predictions_jsonl": str(predictions_path),
        "evaluation_state_json": str(state_path),
    }
    write_json(summary_path, summary)
    print(f"Expert-only accuracy: {correct}/{len(records)} = {summary['accuracy']:.4%}")
    print(f"Summary: {summary_path}")
    print(f"Predictions: {predictions_path}")
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
