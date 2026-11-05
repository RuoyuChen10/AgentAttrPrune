"""SafeSieve with AgentAttrPrune's exact dev40, val1531, prompts and greedy backend.

The 40-question pool is cycled for 285 training draws. Statistics still update
in logical batches of 16; only LLM generation is microbatched in groups of four.
SafeSieve's graph, similarity outcome, threshold, pruning and node removal code
are reused. No remote similarity call or core source edit is performed.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import attrprune_mmlu_adapter as adapter
from AgentAttrPrune.llm import HFChat, generation_observer
from AgentAttrPrune.prompt.mmlu import ROLE_DESCRIPTIONS
from experiments_attrprune.baseline_mmlu_alignment import (
    DEFAULT_REFERENCE, digest, file_sha, freeze_alignment, load_alignment, verify_sources,
)
from SafeSieve.graph.graph import Graph
from SafeSieve.graph.edge_matrix import EdgeMatrixManager
from SafeSieve.graph.pruning_algorithm import dynamic_threshold
from SafeSieve.prompt.mmlu_prompt_set import ROLE_DESCRIPTION

TRAINING_DRAWS = 285
LOGICAL_BATCH_SIZE = 16
MICROBATCH_SIZE = 4
DEFAULT_RUN = ROOT / "result/mmlu/mmlu_attrprune_prompt_dev40_t0"
DEFAULT_SIMILARITY_LOG = ROOT / "assets/mmlu_similarity.log"
POLICY_FIELDS = ("spatial_logits", "temporal_logits", "spatial_masks", "temporal_masks")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen experiment file changed: {path.name}")
    else:
        write_json(path, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "summarize", "all"), default="all")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--similarity-log", type=Path, default=DEFAULT_SIMILARITY_LOG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def cached_similarity(path, alignment):
    """Recover a question-independent outcome of the original similarity engine."""
    text = Path(path).read_text()
    role_match = re.search(r"^MMLU roles: (\[.*\])$", text, re.M)
    matrix_match = re.search(r">>> Initial Similarity Matrix:\s*(\[\[.*?\]\])", text, re.S)
    model_match = re.search(r"Using LLM expert scoring for similarity calculation \(model: ([^)]+)\)", text)
    roles = [worker["role"] for worker in alignment["workers"]]
    if not role_match or ast.literal_eval(role_match.group(1)) != roles or not matrix_match or not model_match:
        raise ValueError("Original similarity log has no matching five-role LLM matrix")
    if any(ROLE_DESCRIPTION[role] != ROLE_DESCRIPTIONS[role] for role in roles):
        raise ValueError("Role descriptions changed; the original similarity outcome is no longer reusable")
    values = np.fromstring(matrix_match.group(1).replace("[", " ").replace("]", " "), sep=" ")
    if values.size != 25:
        raise ValueError("Expected a five-by-five cached similarity matrix")
    matrix = values.reshape(5, 5)
    if (not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T)
            or np.any(np.diag(matrix)) or (matrix < 0).any() or (matrix > 1).any()):
        raise ValueError("Invalid cached similarity matrix")
    return {"source_log": str(Path(path).resolve()), "source_sha256": file_sha(path),
            "engine": model_match.group(1), "task_type": "MMLU", "max_score": 10.0,
            "roles": roles, "role_descriptions": [ROLE_DESCRIPTION[role] for role in roles],
            "matrix": matrix.tolist(), "external_calls": 0,
            "precision": {"source": "NumPy-printed original-run log", "displayed_decimal_places": 2,
                          "full_precision_artifact_available": False,
                          "bitwise_original_internal_matrix_verified": False},
            "reuse_reason": "Original engine compares role descriptions only; all five descriptions are byte-identical"}


def prepare_context(args):
    alignment = load_alignment(args.reference_run)
    similarity = cached_similarity(args.similarity_log, alignment)
    model_name = alignment["reference_run_config"]["model_name"]
    backend_name = model_name if model_name.startswith(("local:", "hf:")) else f"local:{model_name}"
    HFChat.resolve_model_name(backend_name)
    source_paths = list((ROOT / "SafeSieve").rglob("*.py")) + [
        ROOT / "datasets/mmlu_dataset.py",
        Path(__file__).resolve(), Path(adapter.__file__).resolve(),
        adapter.ATTRPRUNE_ROOT / "experiments_attrprune/baseline_mmlu_alignment.py",
        adapter.ATTRPRUNE_ROOT / "AgentAttrPrune/llm/hf_chat.py",
    ]
    config = {
        "schema_version": 1, "experiment": "SafeSieve MMLU aligned to AgentAttrPrune seed0 dev40",
        "alignment_fingerprint": alignment["fingerprint"], "seed": 0,
        "model_name": model_name, "backend_name": backend_name,
        "decoding": alignment["reference_run_config"]["decoding"],
        "worker_ids": [worker["agent_id"] for worker in alignment["workers"]],
        "reviewer_agent_ids": alignment["reviewer_agent_ids"],
        "num_rounds": 2, "llm_microbatch_size": MICROBATCH_SIZE, "eval_batch_size": 4,
        "training": {"pool_size": 40, "draw_count": TRAINING_DRAWS,
                     "draw_plan": [index % 40 for index in range(TRAINING_DRAWS)],
                     "logical_batch_size": LOGICAL_BATCH_SIZE, "logical_batch_count": math.ceil(TRAINING_DRAWS / LOGICAL_BATCH_SIZE),
                     "possible_pruning_batches": [10, 15], "legacy_num_iterations_ignored": 10,
                     "method": "Unchanged statistics update; no optimizer.step in original enhanced_train"},
        "pruning": {"optimized_spatial": True, "optimized_temporal": False,
                    "initial_spatial_logits": 0.5, "pruning_rate": 0.25,
                    "start_pruning_batch": 10, "interval": 5, "max_pruning_times": 7,
                    "min_threshold": 0.65, "max_threshold": 0.9, "alpha_weight": 0.5,
                    "beta_growth_rate": 1.0, "min_agents": 2},
        "similarity": similarity, "evaluation_size": 1531,
        "source_hashes": {str(path.resolve()): file_sha(path) for path in source_paths},
        "semantics": {"graph": "Original SafeSieve Bernoulli spatial sampling and temporal cycle filtering",
                      "rounds": "Original graph executes both rounds; no additional empty-round shortcut",
                      "messages": "Exact AgentAttrPrune builders; graph-selected spatial order, canonical temporal/reviewer order",
                      "score": "Original SafeSieve MMLU parser; original invalid-answer random fallback in training only",
                      "resume": "Atomic logical-batch checkpoint with policy/statistics and RNG states"},
    }
    config["fingerprint"] = digest(config)
    directory = args.run_dir.expanduser().resolve()
    if directory == Path(alignment["reference_run"]) or Path(alignment["reference_run"]) in directory.parents:
        raise ValueError("Use a separate SafeSieve result directory")
    if directory.exists():
        files = {p.name for p in directory.iterdir()} - {".aligned.lock"}
        expected = directory / "run_config.json"
        if files and not expected.exists():
            raise FileExistsError("Existing output directory has no frozen aligned configuration")
        if expected.exists() and json.loads(expected.read_text()) != config:
            raise ValueError("Frozen aligned run configuration changed")
        if expected.exists() and not args.resume:
            raise FileExistsError("Aligned run exists; use --resume")
    return {"directory": directory, "alignment": alignment, "config": config}


def verify_context(ctx):
    verify_sources(ctx["alignment"])
    for path, expected in ctx["config"]["source_hashes"].items():
        if file_sha(path) != expected:
            raise ValueError(f"SafeSieve implementation changed: {path}")
    item = ctx["config"]["similarity"]
    if file_sha(item["source_log"]) != item["source_sha256"]:
        raise ValueError("Frozen original similarity log changed")


def prepare_run(ctx):
    verify_context(ctx)
    freeze_json(ctx["directory"] / "run_config.json", ctx["config"])
    freeze_alignment(ctx["directory"], ctx["alignment"])
    freeze_json(ctx["directory"] / "similarity.json", ctx["config"]["similarity"])


def build_graph(ctx, policy=None, *, llm_factory=None):
    adapter.configure(ctx["alignment"], llm_factory=llm_factory)
    workers = ctx["alignment"]["workers"]
    if policy:
        by_id = {worker["agent_id"]: worker for worker in workers}
        workers = [by_id[agent_id] for agent_id in policy["node_order"]]
    graph = Graph(domain="mmlu", llm_name=ctx["config"]["backend_name"],
        agent_names=[adapter.WORKER_NAME] * len(workers), decision_method=adapter.REVIEWER_NAME,
        optimized_spatial=True, optimized_temporal=False,
        node_kwargs=[{"id": worker["agent_id"], "role": worker["role"]} for worker in workers])
    graph.spatial_logits = torch.nn.Parameter(torch.full((len(workers) ** 2,), 0.5))
    if policy:
        for key in POLICY_FIELDS:
            dtype = getattr(torch, policy["dtypes"][key])
            values = torch.tensor(policy[key], dtype=dtype)
            setattr(graph, key, torch.nn.Parameter(values, requires_grad=key == "spatial_logits"))
        if policy_state(graph) != policy:
            raise ValueError("Restored graph differs from its saved policy")
    return graph


def policy_state(graph):
    return {"node_order": list(graph.node_order),
            "roles": [graph.nodes[agent_id].role for agent_id in graph.node_order],
            "optimized_spatial": graph.optimized_spatial, "optimized_temporal": graph.optimized_temporal,
            "dtypes": {name: str(getattr(graph, name).dtype).removeprefix("torch.") for name in POLICY_FIELDS},
            **{name: getattr(graph, name).detach().cpu().tolist() for name in POLICY_FIELDS}}


def manager_state(manager):
    return {"N": manager.N, "alpha": manager.alpha, "beta": manager.beta, "epsilon": manager.epsilon,
            "dtypes": {"similarity_matrix": str(manager.similarity_matrix.dtype),
                       "complementarity_count": str(manager.complementarity_count.dtype)},
            "similarity_matrix": manager.similarity_matrix.tolist(),
            "complementarity_count": manager.complementarity_count.tolist(),
            "complementarity_total": float(manager.complementarity_total)}


def restore_manager(state):
    manager = EdgeMatrixManager(state["N"], np.array(state["similarity_matrix"], dtype=state["dtypes"]["similarity_matrix"]),
                               alpha=state["alpha"], beta=state["beta"], epsilon=state["epsilon"])
    manager.complementarity_count = np.array(state["complementarity_count"], dtype=state["dtypes"]["complementarity_count"])
    manager.complementarity_total = state["complementarity_total"]
    return manager


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state().tolist()}


def restore_rng(state):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value
    random.setstate(tuples(state["python"]))
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))


def seed_run():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)


@lru_cache(maxsize=1)
def _legacy_dataset_class():
    # Import by file path to avoid the unrelated Hugging Face datasets package.
    path = ROOT / "datasets/mmlu_dataset.py"
    spec = importlib.util.spec_from_file_location("safesieve_aligned_mmlu_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MMLUDataset


def legacy_answer(response):
    return _legacy_dataset_class().postprocess_answer(None, response)


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


async def run_question(graph, sample, *, rounds=2):
    realized = copy.deepcopy(graph)
    realized.spatial_logits = graph.spatial_logits
    events, errors = [], []
    counts = {}

    def observe(event):
        identity = adapter.node_context.get()
        if identity is None:
            raise ValueError("Generation has no canonical node identity")
        node_id = identity["agent_id"]
        round_index = rounds if node_id == "dense_reviewer" else counts.get(node_id, 0)
        counts[node_id] = round_index + 1
        events.append(_plain({**event, **identity, "round": round_index}))

    token = generation_observer.set(observe)
    failure_token = adapter.node_failure.set(errors)
    started = time.perf_counter()
    try:
        answers, _ = await realized.arun({"task": sample["task"]}, num_rounds=rounds)
    finally:
        generation_observer.reset(token)
        adapter.node_failure.reset(failure_token)
    if errors:
        raise RuntimeError("A graph node failed; refusing to count missing advice: " + "; ".join(errors))
    if not answers or not isinstance(answers[0], str) or not answers[0].strip():
        raise RuntimeError("Reviewer produced no valid response")
    wanted = {agent_id: rounds for agent_id in graph.node_order}
    wanted["dense_reviewer"] = 1
    actual = {agent_id: sum(event["agent_id"] == agent_id for event in events) for agent_id in wanted}
    if actual != wanted or len(events) != sum(wanted.values()):
        raise ValueError("Generation audit is missing a scheduled node")
    return {"response": answers[0], "events": events,
            "spatial_adjacency": realized.spatial_adj_matrix.tolist(),
            "elapsed_seconds": time.perf_counter() - started}


async def run_batch(graph, samples):
    tasks = [asyncio.create_task(run_question(graph, sample)) for sample in samples]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def apply_statistics_and_pruning(graph, manager, results, samples, *, batch_index, pruning_count):
    """Successful-path operations mirror run_mmlu.enhanced_train in the same order."""
    if manager.N != graph.num_nodes:
        manager.update_matrix_size(graph.num_nodes)
    outcomes = []
    for result, sample in zip(results, samples):
        predicted = legacy_answer(result["response"])
        if predicted.upper() not in ("A", "B", "C", "D"):
            predicted = random.choice(["A", "B", "C", "D"])
        correct = predicted.upper() == sample["answer"].upper()
        if correct:
            manager.update_complementarity(np.asarray(result["spatial_adjacency"]), solved=True)
        outcomes.append({"parsed_answer": predicted, "correct": bool(correct)})
    manager.update_complementarity(graph.spatial_adj_matrix, any(row["correct"] for row in outcomes))
    beta = 1.0 + batch_index * LOGICAL_BATCH_SIZE / 100.0
    manager.update_beta_weight(beta)
    edge_score, _ = manager.get_edge_score_with_debug()
    threshold = dynamic_threshold(beta, base_threshold=0.65, max_threshold=0.9, growth_rate=1.0)
    pruned, removed = False, []
    if ((batch_index + 1) % 5 == 0 and batch_index + 1 >= 10 and pruning_count < 7
            and graph.optimized_spatial):
        size = graph.num_nodes
        active = (graph.spatial_masks.detach().cpu().numpy().reshape(size, size) > 0) & ~np.eye(size, dtype=bool)
        if np.any(edge_score[active] < threshold):
            graph.update_masks_with_zero_extension(0.25, similarity_matrix=edge_score)
            pruning_count += 1
            pruned = True
            old_order = list(graph.node_order)
            removed = graph.check_and_remove_isolated_nodes(2)
            if removed and len(graph.node_order) != len(old_order):
                removed_indices = [old_order.index(agent_id) for agent_id in removed]
                remaining = [i for i in range(len(old_order)) if i not in removed_indices]
                manager.remove_nodes_and_update_matrix(removed_indices, remaining)
    metric = {"batch_index": batch_index, "sample_count": len(samples), "beta": beta,
              "threshold": threshold, "pruned": pruned, "pruning_count": pruning_count,
              "removed_nodes": removed, "correct": sum(row["correct"] for row in outcomes),
              "spatial_edges": int(graph.spatial_masks.sum()), "node_order": list(graph.node_order)}
    return outcomes, metric, pruning_count


def _save_checkpoint(path, payload):
    value = _plain(payload)
    value["checkpoint_sha256"] = digest(value)
    write_json(path, value)


def _load_checkpoint(path, ctx, phase):
    state = json.loads(Path(path).read_text())
    checksum = state.pop("checkpoint_sha256")
    if digest(state) != checksum or state["run_fingerprint"] != ctx["config"]["fingerprint"] or state["phase"] != phase:
        raise ValueError("Checkpoint content or experiment identity changed")
    return state


def _save_trace(ctx, phase, index, sample, result):
    path = ctx["directory"] / phase / "traces" / f"sample_{index:05d}.json"
    trace = {"run_fingerprint": ctx["config"]["fingerprint"], "phase": phase,
             "sample_index": index, "sample": sample, **result}
    # Orphans from an interrupted logical batch are kept separately on retry.
    if path.exists():
        archive = path.with_name(f"sample_{index:05d}.previous_{time.time_ns()}.json")
        path.rename(archive)
    write_json(path, trace)
    return {"trace_json": str(path), "trace_sha256": file_sha(path)}


def validate_records(ctx, state, samples):
    if len(state["records"]) > len(samples):
        raise ValueError("Too many checkpoint records")
    for index, record in enumerate(state["records"]):
        if record["sample_index"] != index or record["sample_sha256"] != digest(samples[index]):
            raise ValueError("Checkpoint question identity changed")
        path = Path(record["trace_json"])
        expected_path = ctx["directory"] / state["phase"] / "traces" / f"sample_{index:05d}.json"
        if path != expected_path:
            raise ValueError("Trace path is outside the expected phase/sample location")
        if file_sha(path) != record["trace_sha256"]:
            raise ValueError("Saved generation trace was modified")
        trace = json.loads(path.read_text())
        if (trace["sample"] != samples[index] or trace["response"] != record["response"]
                or trace["run_fingerprint"] != ctx["config"]["fingerprint"]
                or trace["phase"] != state["phase"] or trace["sample_index"] != index):
            raise ValueError("Checkpoint response differs from the saved trace")


async def train(ctx, *, resume=False, llm_factory=None):
    path = ctx["directory"] / "training_checkpoint.json"
    plan = ctx["config"]["training"]["draw_plan"]
    samples = [ctx["alignment"]["calibration_samples"][index] for index in plan]
    state = _load_checkpoint(path, ctx, "training") if path.exists() else None
    if state and not resume:
        raise FileExistsError("Training checkpoint exists; use --resume")
    if state:
        validate_records(ctx, state, samples)
        expected_count = min(state["next_batch"] * LOGICAL_BATCH_SIZE, len(samples))
        if len(state["records"]) != expected_count or len(state["batches"]) != state["next_batch"]:
            raise ValueError("Training checkpoint is not a complete logical-batch prefix")
        if state["complete"]:
            if expected_count != len(samples):
                raise ValueError("Incomplete training was marked complete")
            freeze_json(ctx["directory"] / "policy.json", state["policy"])
            return state
    graph = build_graph(ctx, state["policy"] if state else None, llm_factory=llm_factory)
    if state:
        manager = restore_manager(state["manager"])
        restore_rng(state["rng"])
    else:
        seed_run()
        manager = EdgeMatrixManager(5, np.array(ctx["config"]["similarity"]["matrix"]), alpha=0.5, beta=1.0, epsilon=1e-6)
        state = {"phase": "training", "run_fingerprint": ctx["config"]["fingerprint"],
                 "complete": False, "next_batch": 0, "pruning_count": 0, "records": [], "batches": []}
    llm = graph.decision_node.llm
    async with llm.async_batching(MICROBATCH_SIZE):
        for batch_index in range(state["next_batch"], math.ceil(len(samples) / LOGICAL_BATCH_SIZE)):
            offset = batch_index * LOGICAL_BATCH_SIZE
            batch = samples[offset:offset + LOGICAL_BATCH_SIZE]
            results = await run_batch(graph, batch)
            outcomes, metric, count = apply_statistics_and_pruning(
                graph, manager, results, batch, batch_index=batch_index, pruning_count=state["pruning_count"])
            for index, (sample, result, outcome) in enumerate(zip(batch, results, outcomes), offset):
                saved = _save_trace(ctx, "training", index, sample, result)
                state["records"].append({"sample_index": index, "calibration_pool_index": plan[index],
                    "sample_sha256": digest(sample), "response": result["response"], **outcome, **saved})
            state.update(next_batch=batch_index + 1, pruning_count=count, policy=policy_state(graph),
                         manager=manager_state(manager), rng=rng_state(), complete=offset + len(batch) == len(samples))
            state["batches"].append(metric)
            verify_context(ctx)
            _save_checkpoint(path, state)
            print(f"Training batch {batch_index + 1}/18: {len(state['records'])}/285 draws; pruned={metric['pruned']}", flush=True)
    freeze_json(ctx["directory"] / "policy.json", state["policy"])
    return state


async def evaluate(ctx, *, resume=False, llm_factory=None):
    training = _load_checkpoint(ctx["directory"] / "training_checkpoint.json", ctx, "training")
    if not training["complete"] or len(training["records"]) != TRAINING_DRAWS:
        raise ValueError("A complete trained policy is required before evaluation")
    validate_records(ctx, training, [ctx["alignment"]["calibration_samples"][i] for i in ctx["config"]["training"]["draw_plan"]])
    policy_path = ctx["directory"] / "policy.json"
    policy = json.loads(policy_path.read_text())
    if policy != training["policy"]:
        raise ValueError("Published policy differs from the training checkpoint")
    path = ctx["directory"] / "evaluation_checkpoint.json"
    state = _load_checkpoint(path, ctx, "evaluation") if path.exists() else None
    samples = ctx["alignment"]["evaluation_samples"]
    if state and not resume:
        raise FileExistsError("Evaluation exists; use --resume")
    if state:
        validate_records(ctx, state, samples)
        if not state["complete"] and len(state["records"]) % 4:
            raise ValueError("Evaluation checkpoint is not a complete microbatch prefix")
        if state["policy_sha256"] != file_sha(policy_path):
            raise ValueError("Evaluation policy changed")
        if state["complete"]:
            if len(state["records"]) != len(samples):
                raise ValueError("Incomplete evaluation was marked complete")
            return summarize(ctx)
    graph = build_graph(ctx, policy, llm_factory=llm_factory)
    for name in ("spatial_logits", "temporal_logits"):
        getattr(graph, name).requires_grad_(False)
    if state:
        restore_rng(state["rng"])
    else:
        seed_run()
        state = {"phase": "evaluation", "run_fingerprint": ctx["config"]["fingerprint"],
                 "policy_sha256": file_sha(policy_path), "records": [], "complete": False}
    async with graph.decision_node.llm.async_batching(MICROBATCH_SIZE):
        for offset in range(len(state["records"]), len(samples), 4):
            batch = samples[offset:offset + 4]
            with torch.no_grad():
                results = await run_batch(graph, batch)
            for index, (sample, result) in enumerate(zip(batch, results), offset):
                predicted = legacy_answer(result["response"])
                saved = _save_trace(ctx, "evaluation", index, sample, result)
                state["records"].append({"sample_index": index, "sample_sha256": digest(sample),
                    "response": result["response"], "parsed_answer": predicted,
                    "correct": predicted == sample["answer"], **saved})
            if policy_state(graph) != policy:
                raise ValueError("Evaluation modified the frozen policy")
            state.update(rng=rng_state(), complete=len(state["records"]) == len(samples))
            verify_context(ctx)
            _save_checkpoint(path, state)
            print(f"Evaluation {len(state['records'])}/{len(samples)}", flush=True)
    return summarize(ctx)


def summarize(ctx):
    verify_context(ctx)
    state = _load_checkpoint(ctx["directory"] / "evaluation_checkpoint.json", ctx, "evaluation")
    samples = ctx["alignment"]["evaluation_samples"]
    validate_records(ctx, state, samples)
    if not state["complete"] or len(state["records"]) != len(samples):
        raise ValueError("Evaluation incomplete; refusing to publish final accuracy")
    if file_sha(ctx["directory"] / "policy.json") != state["policy_sha256"]:
        raise ValueError("Evaluation policy changed; refusing to publish final accuracy")
    totals = {"llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "elapsed_seconds": 0.0}
    batching = {"configured_max_batch_size": 4, "max_effective_batch_size": 0, "oom_fallback_calls": 0}
    for record in state["records"]:
        trace = json.loads(Path(record["trace_json"]).read_text())
        totals["elapsed_seconds"] += trace["elapsed_seconds"]
        for event in trace["events"]:
            totals["llm_calls"] += 1
            totals["prompt_tokens"] += np.asarray(event["input_ids"]).size
            totals["generated_tokens"] += np.asarray(event["generated_ids"]).size
            kwargs = event["generation_kwargs"]
            batching["max_effective_batch_size"] = max(batching["max_effective_batch_size"], kwargs.get("effective_batch_size", 1))
            batching["oom_fallback_calls"] += bool(kwargs.get("oom_fallback"))
    correct = sum(record["correct"] for record in state["records"])
    result = {"complete": True, "run_fingerprint": ctx["config"]["fingerprint"],
              "total": len(samples), "correct": correct, "accuracy": correct / len(samples),
              "training_unique_questions": 40, "training_draws": TRAINING_DRAWS,
              "logical_training_batch_size": LOGICAL_BATCH_SIZE, "eval_batch_size": 4,
              "generation_batching": batching, **totals,
              "elapsed_semantics": "Sum of per-question durations; concurrent durations overlap",
              "policy_sha256": state["policy_sha256"]}
    freeze_json(ctx["directory"] / "summary.json", result)
    return result


def main(argv=None):
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(ctx["directory"]),
                          "configuration": ctx["config"]}, indent=2))
        return
    if args.stage in ("all", "train", "eval") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly for GPU stages")
    ctx["directory"].mkdir(parents=True, exist_ok=True)
    with (ctx["directory"] / ".aligned.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_run(ctx)
        os.environ.update(HF_CHAT_ENABLE_THINKING="0", HF_CHAT_CAPTURE_SCORES="0", TOKENIZERS_PARALLELISM="false")
        result = {"prepared": True, "run_dir": str(ctx["directory"])}
        if args.stage in ("all", "train"):
            asyncio.run(train(ctx, resume=args.resume))
        if args.stage in ("all", "eval"):
            result = asyncio.run(evaluate(ctx, resume=args.resume))
        elif args.stage == "summarize":
            result = summarize(ctx)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
