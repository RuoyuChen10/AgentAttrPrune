"""Resumable evaluation through AgentDropout's original stochastic graph path.

Edges are drawn inside Graph.arun, with its original draw order and spatial cycle
checks. In particular, temporal edges use the optimized=True branch; they must
not be converted to fixed masks, whose legacy branch adds different filtering.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import time

import torch

from experiments.aligned_evaluate_mmlu import _append_batch, _flatten_answer, _read_existing
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
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.jsonl"
    if path.exists() and not resume:
        raise FileExistsError(f"{path} exists; use --resume")
    existing = _read_existing(path) if resume else {}
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
        results = await asyncio.gather(*tasks)
        records = []
        for (raw, _, outputs), (index, inputs, gold, realized) in zip(results, contexts):
            parsed = dataset.postprocess_answer(raw)
            records.append({
                "sample_index": index, "question": inputs["task"], "gold_answer": gold,
                "raw_answer": _flatten_answer(raw), "parsed_answer": parsed,
                "correct": parsed == gold, "worker_outputs": outputs,
                "policy_seed": seed + index, "execution_trace": realized.execution_trace,
                "sampling_mode": "original_graph_optimized_true",
            })
        _append_batch(path, records)
        existing.update({record["sample_index"]: record for record in records})
        completed = sum(index in existing for index in indices)
        correct = sum(bool(existing[index]["correct"]) for index in indices if index in existing)
        print(f"Evaluation {completed}/{len(indices)}; accuracy={correct/completed:.6f} ({correct}/{completed})", flush=True)
    selected = [existing[index] for index in indices]
    correct = sum(bool(record["correct"]) for record in selected)
    summary = {
        "benchmark": "mmlu", "split": dataset.split, "start_index": start_index,
        "limit": len(indices), "num_evaluated": len(selected), "num_correct": correct,
        "accuracy": correct / len(selected) if selected else 0.0,
        "seed": seed, "batch_size": batch_size,
        "elapsed_seconds_this_invocation": time.time() - started,
        "predictions": str(path), "run_config": run_config,
        "sampling_mode": "original_graph_optimized_true",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
