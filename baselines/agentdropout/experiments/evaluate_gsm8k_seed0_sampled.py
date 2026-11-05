"""Resumable evaluation through AgentDropout's original stochastic graph path.

Edges are drawn inside Graph.arun, with its original draw order and spatial cycle
checks. In particular, temporal edges use the optimized=True branch; they must
not be converted to fixed masks, whose legacy branch adds different filtering.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import time

import torch

from experiments.aligned_evaluate_mmlu import _flatten_answer
from experiments.run_gsm8k_aligned import _read_existing, _atomic_json
from experiments_attrprune.baseline_gsm8k_alignment import file_sha
from experiments.traced_original_graph import traced_graph_class


def realize_policy(graph, sample_seed):
    if not graph.optimized_spatial or not graph.optimized_temporal:
        raise ValueError("Original stochastic evaluation requires both optimized flags to remain True")
    realized = copy.deepcopy(graph)
    realized.generator = torch.Generator(device="cpu").manual_seed(sample_seed)
    # This existing wrapper records execution without replacing the scheduler.
    realized.__class__ = traced_graph_class(type(realized))
    return realized


async def evaluate(graph, dataset, *, num_rounds, output_dir: Path,
                   start_index, limit, batch_size, seed, resume, run_config):
    # Retain the original GSM8K evaluation's frozen parameter semantics.
    graph.optimized_spatial = graph.optimized_temporal = True
    for name in ("spatial_logits", "temporal_logits", "spatial_masks", "temporal_masks",
                 "spatial_logits_1", "temporal_logits_1"):
        values = getattr(graph, name, [])
        values = [values] if isinstance(values, torch.Tensor) else values
        for value in values:
            value.requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.jsonl"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    checkpoint = output_dir.parent / "pruning_artifacts/learned_dropout_policy.pt"
    policy_sha = file_sha(checkpoint)
    policy_path = output_dir / "policy.json"
    frozen_policy = {"checkpoint_sha256": policy_sha, "config_fingerprint": run_config["fingerprint"]}
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != frozen_policy:
            raise ValueError("Evaluation policy changed")
    elif path.exists():
        raise ValueError("Existing predictions are missing their frozen policy identity")
    else:
        _atomic_json(policy_path, frozen_policy)
    existing = _read_existing(path, dataset, seed) if resume else {}
    for record in existing.values():
        if (record.get("checkpoint_sha256") != policy_sha or
                record.get("config_fingerprint") != run_config["fingerprint"] or
                record.get("sampling_mode") != "original_graph_optimized_true"):
            raise ValueError("Existing prediction does not belong to this policy/configuration")
    stop = len(dataset) if limit is None else min(len(dataset), start_index + limit)
    indices = list(range(start_index, stop))
    pending = [index for index in indices if index not in existing]
    started = time.time()
    for offset in range(0, len(pending), batch_size):
        contexts = []
        tasks = []
        for index in pending[offset:offset + batch_size]:
            sample = dataset[index]
            realized = realize_policy(graph, seed + index)
            inputs = dataset.record_to_input(sample)
            contexts.append((index, inputs, dataset.record_to_target_answer(sample), realized))
            tasks.append(asyncio.create_task(realized.arun(inputs, num_rounds, case=True)))
        with torch.no_grad():
            results = await asyncio.gather(*tasks)
        records = []
        for (raw, _, outputs), (index, inputs, gold, realized) in zip(results, contexts):
            parsed = dataset.postprocess_answer(raw)
            records.append({
                "sample_index": index, "question": inputs["task"], "gold_answer": gold,
                "checkpoint_sha256": policy_sha, "config_fingerprint": run_config["fingerprint"],
                "raw_answer": _flatten_answer(raw), "parsed_answer": parsed,
                "correct": parsed == gold, "worker_outputs": outputs,
                "policy_seed": seed + index, "execution_trace": realized.execution_trace,
                "sampling_mode": "original_graph_optimized_true",
            })
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        existing.update({record["sample_index"]: record for record in records})
        completed = sum(index in existing for index in indices)
        correct = sum(bool(existing[index]["correct"]) for index in indices if index in existing)
        print(f"Evaluation {completed}/{len(indices)}; accuracy={correct/completed:.6f} ({correct}/{completed})", flush=True)
    selected = [existing[index] for index in indices]
    correct = sum(bool(record["correct"]) for record in selected)
    summary = {
        "benchmark": "gsm8k", "complete": len(selected) == len(indices), "split": dataset.split, "start_index": start_index,
        "limit": len(indices), "num_evaluated": len(selected), "num_correct": correct,
        "accuracy": correct / len(selected) if selected else 0.0,
        "seed": seed, "batch_size": batch_size,
        "elapsed_seconds_this_invocation": time.time() - started,
        "predictions": str(path), "run_config": run_config,
        "sampling_mode": "original_graph_optimized_true",
    }
    if file_sha(checkpoint) != policy_sha:
        raise ValueError("Learned policy changed during evaluation")
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
