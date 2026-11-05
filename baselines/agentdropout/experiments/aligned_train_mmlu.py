"""AgentDropout training with explicit, reproducible experiment budgets."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import time
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from AgentDropout.utils.utils import frobenius_norm, nuclear_norm
from experiments.accuracy import Accuracy


def _parameters(values):
    if isinstance(values, torch.nn.ParameterList):
        return list(values.parameters())
    return [values]


def _matrices(values, size: int):
    if isinstance(values, torch.nn.ParameterList):
        return [value.reshape(size, size) for value in values]
    return [values.reshape(size, size)]


def _mean(values):
    return torch.mean(torch.stack(list(values)))


def _append_jsonl(path: Path | None, payload: Mapping) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        handle.flush()


def _infinite_loader(dataset, rng: np.random.Generator) -> Iterator:
    while True:
        for index in rng.permutation(len(dataset)):
            yield int(index), dataset[int(index)]


def _utility(dataset, raw_answer, correct_answer: str) -> tuple[str, float]:
    answer = dataset.postprocess_answer(raw_answer)
    accuracy = Accuracy()
    accuracy.update(answer, correct_answer)
    return answer, accuracy.get()


async def _node_dropout_stage(
    graph,
    dataset,
    *,
    num_iters: int,
    batch_size: int,
    num_rounds: int,
    lr: float,
    loader: Iterator,
    metrics_path: Path | None,
    seed: int,
) -> None:
    parameters = _parameters(graph.spatial_logits_1) + _parameters(graph.temporal_logits_1)
    optimizer = torch.optim.Adam(parameters, lr=lr)

    for iteration in range(num_iters):
        started = time.time()
        tasks = []
        contexts = []
        for batch_index in range(batch_size):
            sample_index, record = next(loader)
            realized = copy.deepcopy(graph)
            realized.spatial_logits_1 = graph.spatial_logits_1
            realized.temporal_logits_1 = graph.temporal_logits_1
            realized.generator = torch.Generator(device="cpu")
            realized.generator.manual_seed(
                seed + iteration * batch_size + batch_index
            )
            input_dict = dataset.record_to_input(record)
            correct = dataset.record_to_target_answer(record)
            tasks.append(asyncio.create_task(realized.arun(input_dict, num_rounds, skip=True)))
            contexts.append((sample_index, correct))

        results = await asyncio.gather(*tasks)
        losses = []
        answers = []
        utilities = []
        for (raw_answer, log_probability), (_, correct) in zip(results, contexts):
            answer, utility = _utility(dataset, raw_answer, correct)
            losses.append(-log_probability * utility)
            answers.append(answer)
            utilities.append(utility)
        total_loss = _mean(losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        payload = {
            "stage": "node_dropout",
            "iteration": iteration,
            "sample_indices": [item[0] for item in contexts],
            "answers": answers,
            "utilities": utilities,
            "mean_utility": float(np.mean(utilities)),
            "loss": float(total_loss.detach()),
            "elapsed_seconds": time.time() - started,
        }
        print(json.dumps(payload), flush=True)
        _append_jsonl(metrics_path, payload)

    graph.update_masks_dec()
    print(f"Selected node dropouts by round: {graph.skip_nodes}", flush=True)


def _regularization(graph, fixed_spatial, fixed_temporal, delta: float) -> torch.Tensor:
    spatial = _matrices(graph.spatial_logits, fixed_spatial.shape[0])
    temporal = _matrices(graph.temporal_logits, fixed_temporal.shape[0])
    spatial_nuclear = _mean(nuclear_norm(matrix) for matrix in spatial)
    temporal_nuclear = _mean(nuclear_norm(matrix) for matrix in temporal)
    spatial_distance = _mean(frobenius_norm(fixed_spatial, matrix) for matrix in spatial)
    temporal_distance = _mean(frobenius_norm(fixed_temporal, matrix) for matrix in temporal)
    return (
        spatial_nuclear
        + temporal_nuclear
        + F.relu(spatial_distance - delta)
        + F.relu(temporal_distance - delta)
    )


async def _edge_dropout_stage(
    graph,
    dataset,
    *,
    num_iters: int,
    batch_size: int,
    num_rounds: int,
    lr: float,
    delta: float,
    prune_every: int,
    pruning_rate: float,
    loader: Iterator,
    metrics_path: Path | None,
    seed: int,
) -> None:
    parameters = _parameters(graph.spatial_logits) + _parameters(graph.temporal_logits)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    size = len(graph.nodes)
    fixed_spatial = graph.fixed_spatial_masks.to(dtype=torch.float32).reshape(size, size)
    fixed_temporal = graph.fixed_temporal_masks.to(dtype=torch.float32).reshape(size, size)

    for iteration in range(num_iters):
        started = time.time()
        tasks = []
        contexts = []
        for batch_index in range(batch_size):
            sample_index, record = next(loader)
            realized = copy.deepcopy(graph)
            realized.spatial_logits = graph.spatial_logits
            realized.temporal_logits = graph.temporal_logits
            realized.generator = torch.Generator(device="cpu")
            realized.generator.manual_seed(
                seed + 10_000_000 + iteration * batch_size + batch_index
            )
            input_dict = dataset.record_to_input(record)
            correct = dataset.record_to_target_answer(record)
            tasks.append(asyncio.create_task(realized.arun(input_dict, num_rounds)))
            contexts.append((sample_index, correct))

        results = await asyncio.gather(*tasks)
        regularization = _regularization(graph, fixed_spatial, fixed_temporal, delta)
        losses = []
        answers = []
        utilities = []
        for (raw_answer, log_probability), (_, correct) in zip(results, contexts):
            answer, utility = _utility(dataset, raw_answer, correct)
            losses.append(-log_probability * utility + regularization)
            answers.append(answer)
            utilities.append(utility)
        total_loss = _mean(losses)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pruned = (iteration + 1) % prune_every == 0
        if pruned:
            if graph.diff:
                graph.update_masks_diff(pruning_rate)
            else:
                graph.update_masks(pruning_rate)
        payload = {
            "stage": "edge_dropout",
            "iteration": iteration,
            "sample_indices": [item[0] for item in contexts],
            "answers": answers,
            "utilities": utilities,
            "mean_utility": float(np.mean(utilities)),
            "loss": float(total_loss.detach()),
            "regularization": float(regularization.detach()),
            "pruned_this_iteration": pruned,
            "spatial_retained": [int(mask.sum()) for mask in _parameters(graph.spatial_masks)],
            "temporal_retained": [int(mask.sum()) for mask in _parameters(graph.temporal_masks)],
            "elapsed_seconds": time.time() - started,
        }
        print(json.dumps(payload), flush=True)
        _append_jsonl(metrics_path, payload)


async def train(
    graph,
    dataset,
    *,
    num_iters: int,
    num_rounds: int,
    lr: float,
    batch_size: int,
    prune_every: int,
    pruning_rate: float,
    delta: float,
    seed: int,
    metrics_path: Path | None = None,
) -> None:
    if num_iters < 1 or batch_size < 1 or prune_every < 1:
        raise ValueError("num_iters, batch_size, and prune_every must be positive")
    rng = np.random.default_rng(seed)
    loader = _infinite_loader(dataset, rng)
    if hasattr(graph, "spatial_logits_1"):
        await _node_dropout_stage(
            graph,
            dataset,
            num_iters=num_iters,
            batch_size=batch_size,
            num_rounds=num_rounds,
            lr=lr,
            loader=loader,
            metrics_path=metrics_path,
            seed=seed,
        )
    await _edge_dropout_stage(
        graph,
        dataset,
        num_iters=num_iters,
        batch_size=batch_size,
        num_rounds=num_rounds,
        lr=lr,
        delta=delta,
        prune_every=prune_every,
        pruning_rate=pruning_rate,
        loader=loader,
        metrics_path=metrics_path,
        seed=seed,
    )
