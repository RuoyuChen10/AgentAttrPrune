"""SafeSieve on AgentAttrPrune's original seed0 train40 and complete test1319.

Only the question pool, MAS prompts, decoding and generation batching are aligned.
The original GSM8K statistics/pruning algorithm still consumes 40 questions once,
with logical batch4 and one conditional 25% spatial pruning check at batch10.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import attrprune_gsm8k_seed0_adapter as adapter
from experiments import run_gsm8k as original
from experiments.run_mmlu_attrprune_aligned import (
    write_json, freeze_json, policy_state, manager_state, restore_manager,
    rng_state, restore_rng, seed_run, _plain, _save_checkpoint, _load_checkpoint,
    _save_trace, validate_records,
)
from AgentAttrPrune.llm import HFChat, generation_observer
from AgentAttrPrune.prompt.gsm8k import ROLE_DESCRIPTIONS
from experiments_attrprune.baseline_gsm8k_alignment import (
    DEFAULT_REFERENCE, digest, file_sha, freeze_alignment, load_alignment, verify_sources,
)
from SafeSieve.graph.graph import Graph
from SafeSieve.graph.edge_matrix import EdgeMatrixManager
from SafeSieve.graph.pruning_algorithm import dynamic_threshold
from SafeSieve.prompt.gsm8k_prompt_set import ROLE_DESCRIPTION

TRAINING_DRAWS = 40
LOGICAL_BATCH_SIZE = MICROBATCH_SIZE = 4
DEFAULT_RUN = ROOT / "result/gsm8k/gsm8k_attrprune_seed0_train40_t0"
DEFAULT_ORIGINAL_RUN = ROOT / "assets/gsm8k_original"
POLICY_FIELDS = ("spatial_logits", "temporal_logits", "spatial_masks", "temporal_masks")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "summarize", "all"), default="all")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--original-run", type=Path, default=DEFAULT_ORIGINAL_RUN)
    parser.add_argument("--similarity-json", type=Path, default=ROOT / "assets/gsm8k_aligned_similarity.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def original_protocol(directory, alignment):
    directory = Path(directory).resolve()
    paths = [directory / "run_config.json", directory / "similarity.json"]
    old, similarity = [json.loads(p.read_text()) for p in paths]
    expected = {"train_questions": 40, "expected_questions": 1319, "batch_size": 4,
                "num_rounds": 2, "seed": 0, "optimized_spatial": True, "optimized_temporal": False,
                "initial_spatial_logits": .5, "pruning_rate": .25, "start_pruning_batch": 10,
                "imp_per_iterations": 5, "max_pruning_times": 7, "min_agents": 2,
                "alpha_weight": .5, "beta_growth_rate": 1., "min_threshold": .65,
                "max_threshold": .9, "use_llm_similarity": True, "similarity_model": "gpt-5.6-sol",
                "similarity_max_score": 10., "possible_pruning_batches": [10]}
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f"Original GSM8K protocol differs: {key}")
    roles = [worker["role"] for worker in alignment["workers"]]
    if old["worker_roles"] != roles or similarity["roles"] != roles or similarity["method"] != "llm":
        raise ValueError("Original GSM8K roles/similarity do not match the four canonical workers")
    matrix = np.asarray(similarity["matrix"], dtype=float)
    validate_similarity_matrix(matrix)
    # Confirm the role text that produced the cached prior remains recoverable.
    source_name = "SafeSieve/prompt/gsm8k_prompt_set.py"
    if file_sha(ROOT / source_name) != old["source_sha256"][source_name]:
        raise ValueError("Original role descriptions changed; cached similarity provenance is stale")
    return {"run": str(directory), "protocol": expected,
            "source_hashes": {str(p): file_sha(p) for p in paths},
            "original_role_descriptions": [ROLE_DESCRIPTION[role] for role in roles],
            "matrix": matrix.tolist()}


def validate_similarity_matrix(matrix):
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix))
            or (matrix < 0).any() or (matrix > 1).any()):
        raise ValueError("Expected a finite symmetric four-role similarity matrix in [0,1] with zero diagonal")


def prepare_context(args):
    alignment = load_alignment(args.reference_run)
    old = original_protocol(args.original_run, alignment)
    roles = [worker["role"] for worker in alignment["workers"]]
    model = alignment["reference_run_config"]["model_name"]
    backend = model if model.startswith(("local:", "hf:")) else f"local:{model}"
    HFChat.resolve_model_name(backend)
    source_paths = list((ROOT / "SafeSieve").rglob("*.py")) + [
        ROOT / "experiments/run_gsm8k.py", ROOT / "datasets/gsm8k_dataset.py",
        ROOT / "experiments/run_mmlu_attrprune_aligned.py", ROOT / "experiments/attrprune_mmlu_adapter.py",
        Path(__file__).resolve(), Path(adapter.__file__).resolve(),
        adapter.ATTRPRUNE_ROOT / "experiments_attrprune/baseline_gsm8k_alignment.py",
        adapter.ATTRPRUNE_ROOT / "experiments_attrprune/baseline_mmlu_alignment.py",
        adapter.ATTRPRUNE_ROOT / "AgentAttrPrune/llm/hf_chat.py",
    ]
    similarity = {"mode": "aligned_roles", "engine": "gpt-5.6-sol", "task_type": "GSM8K", "max_score": 10.,
                  "roles": roles, "role_descriptions": [ROLE_DESCRIPTIONS[role] for role in roles],
                  "old_role_descriptions_differ": any(ROLE_DESCRIPTIONS[r] != ROLE_DESCRIPTION[r] for r in roles)}
    similarity.update(interpretation="Recompute original expert-scoring algorithm on aligned role descriptions",
                      external_calls_if_uncached=6)
    config = {"schema_version": 1, "experiment": "SafeSieve GSM8K aligned to original AgentAttrPrune seed0 train40",
        "alignment_fingerprint": alignment["fingerprint"], "seed": 0,
        "model_name": model, "backend_name": backend, "decoding": alignment["reference_run_config"]["decoding"],
        "worker_ids": [w["agent_id"] for w in alignment["workers"]],
        "reviewer_agent_ids": alignment["reviewer_agent_ids"], "num_rounds": 2,
        "llm_microbatch_size": 4, "eval_batch_size": 4,
        "training": {"pool_size": 40, "draw_count": 40, "draw_plan": list(range(40)),
                     "logical_batch_size": 4, "logical_batch_count": 10, "possible_pruning_batches": [10],
                     "method": "Original GSM8K successful sampled-edge statistics; no optimizer/logit training"},
        "pruning": {"optimized_spatial": True, "optimized_temporal": False, "initial_spatial_logits": .5,
                    "pruning_rate": .25, "start_pruning_batch": 10, "interval": 5, "max_pruning_times": 7,
                    "min_threshold": .65, "max_threshold": .9, "alpha_weight": .5, "beta_growth_rate": 1., "min_agents": 2},
        "original_protocol": old, "similarity": similarity, "similarity_cache": str(args.similarity_json.resolve()), "evaluation_size": 1319,
        "source_hashes": {str(p.resolve()): file_sha(p) for p in source_paths},
        "semantics": {"graph": "Original SafeSieve spatial Bernoulli and temporal cycle filtering",
            "rounds": "Both original rounds; no added empty-round shortcut",
            "messages": "Exact AgentAttrPrune GSM8K builders; canonical IDs and reviewer aliases",
            "score": "Original GSM8K score_answer and gsm_get_predict; Decimal equality",
            "resume": "Atomic logical-batch checkpoint with policy/statistics and RNG states; first evaluation continues original post-training RNG"}}
    config["fingerprint"] = digest(config)
    directory = args.run_dir.expanduser().resolve()
    reference = Path(alignment["reference_run"])
    if directory == reference or reference in directory.parents:
        raise ValueError("Use a separate SafeSieve output directory")
    if directory.exists():
        files = {p.name for p in directory.iterdir()} - {".aligned.lock"}
        frozen = directory / "run_config.json"
        if files and not frozen.exists():
            raise FileExistsError("Output directory has no frozen aligned configuration")
        if frozen.exists() and json.loads(frozen.read_text()) != config:
            raise ValueError("Frozen aligned run configuration changed")
        if frozen.exists() and not args.resume:
            raise FileExistsError("Aligned experiment exists; use --resume")
    return {"directory": directory, "alignment": alignment, "config": config}


def verify_context(ctx):
    verify_sources(ctx["alignment"])
    hashes = {**ctx["config"]["source_hashes"], **ctx["config"]["original_protocol"]["source_hashes"]}
    for path, expected in hashes.items():
        if file_sha(path) != expected:
            raise ValueError(f"Frozen experiment source changed: {path}")


def prepare_run(ctx):
    verify_context(ctx)
    freeze_json(ctx["directory"] / "run_config.json", ctx["config"])
    freeze_alignment(ctx["directory"], ctx["alignment"])


def _validate_similarity_payload(ctx, item):
    if item.get("configuration") != ctx["config"]["similarity"]:
        raise ValueError("Similarity matrix belongs to another role-prompt configuration")
    matrix = np.asarray(item["matrix"], dtype=float)
    validate_similarity_matrix(matrix)
    pairs = item.get("pairs", [])
    expected = {(i,j) for i in range(4) for j in range(i+1,4)}
    if {(r["i"],r["j"]) for r in pairs} != expected or len(pairs) != 6:
        raise ValueError("Similarity must have exactly six audited expert responses")
    from SafeSieve.graph.prompt_similarity import _build_evaluation_prompt
    spec = ctx["config"]["similarity"]
    for row in pairs:
        i,j = row["i"],row["j"]
        prompt = _build_evaluation_prompt(spec["role_descriptions"][i], spec["role_descriptions"][j], "GSM8K", 10.)
        if row["messages"] != [{"role":"user","content":prompt}]:
            raise ValueError("Similarity request prompt differs from the original scoring algorithm")
        score = original.parse_similarity_score(row["response"], 10.)
        if score != row["score"] or matrix[i,j] != score/10.:
            raise ValueError("Similarity score differs from its audited response")
    return matrix


def load_similarity(ctx):
    return _validate_similarity_payload(ctx, json.loads((ctx["directory"] / "similarity.json").read_text()))


async def ensure_similarity(ctx):
    path = ctx["directory"] / "similarity.json"
    if path.exists():
        return load_similarity(ctx)
    cache = Path(ctx["config"]["similarity_cache"])
    spec = ctx["config"]["similarity"]
    if cache.exists():
        item = json.loads(cache.read_text())
    else:
        raise FileNotFoundError(f"Validated GSM8K role similarity cache is required: {cache}. "
                                "Use the bundled assets/gsm8k_aligned_similarity.json or supply --similarity-json.")
    _validate_similarity_payload(ctx,item)
    freeze_json(path,item)
    return load_similarity(ctx)


def build_graph(ctx, policy=None, *, llm_factory=None):
    adapter.configure(ctx["alignment"], llm_factory=llm_factory)
    workers = ctx["alignment"]["workers"]
    if policy:
        by_id = {w["agent_id"]: w for w in workers}
        workers = [by_id[i] for i in policy["node_order"]]
    graph = Graph(domain="gsm8k", llm_name=ctx["config"]["backend_name"],
        agent_names=[adapter.WORKER_NAME] * len(workers), decision_method=adapter.REVIEWER_NAME,
        optimized_spatial=True, optimized_temporal=False,
        node_kwargs=[{"id": w["agent_id"], "role": w["role"]} for w in workers])
    graph.spatial_logits = torch.nn.Parameter(torch.full((len(workers) ** 2,), .5), requires_grad=False)
    if policy:
        for key in POLICY_FIELDS:
            values = torch.tensor(policy[key], dtype=getattr(torch, policy["dtypes"][key]))
            setattr(graph, key, torch.nn.Parameter(values, requires_grad=False))
        if policy_state(graph) != policy:
            raise ValueError("Restored policy differs from its saved graph")
    return graph


@lru_cache(maxsize=1)
def predictor():
    path = ROOT / "datasets/gsm8k_dataset.py"
    spec = importlib.util.spec_from_file_location("safesieve_aligned_gsm_predictor", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.gsm_get_predict


def outcome(response, sample):
    predicted, correct = original.score_answer(response, sample["answer"], predictor())
    return {"parsed_answer": predicted, "correct": correct}


async def run_question(graph, sample):
    realized = copy.deepcopy(graph)
    original_update_memory = realized.update_memory
    def checked_update_memory():
        # Preserve the original GSM8K runner's per-round empty-output check.
        for node in realized.nodes.values():
            if not node.outputs or any(not isinstance(value, str) or not value.strip() for value in node.outputs):
                raise RuntimeError(f"Agent {node.id} produced no valid output")
        original_update_memory()
    realized.update_memory = checked_update_memory
    events, errors, counts = [], [], {}
    def observe(event):
        identity = adapter.node_context.get()
        if identity is None:
            raise ValueError("Generation has no canonical node identity")
        node = identity["agent_id"]
        round_index = 2 if node == "dense_reviewer" else counts.get(node, 0)
        counts[node] = round_index + 1
        events.append(_plain({**event, **identity, "round": round_index}))
    token = generation_observer.set(observe)
    failure_token = adapter.node_failure.set(errors)
    started = time.perf_counter()
    try:
        answers, _ = await realized.arun({"task": sample["task"]}, num_rounds=2)
    finally:
        generation_observer.reset(token); adapter.node_failure.reset(failure_token)
    if errors:
        raise RuntimeError("A graph node failed: " + "; ".join(errors))
    if not answers or not isinstance(answers[0], str) or not answers[0].strip():
        raise RuntimeError("Reviewer produced no valid output")
    expected = {agent: 2 for agent in graph.node_order}; expected["dense_reviewer"] = 1
    actual = {agent: sum(e["agent_id"] == agent for e in events) for agent in expected}
    if actual != expected or len(events) != sum(expected.values()):
        raise ValueError("Generation audit is missing a scheduled node")
    return {"response": answers[0], "events": events, "node_order": list(realized.node_order),
            "spatial_adjacency": realized.spatial_adj_matrix.tolist(), "elapsed_seconds": time.perf_counter() - started}


async def run_batch(graph, samples):
    tasks = [asyncio.create_task(run_question(graph, sample)) for sample in samples]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def apply_statistics_and_pruning(graph, manager, results, samples, *, batch_index, pruning_count):
    """Exact successful-path operations of original GSM8K run_phases calibration."""
    outcomes = [outcome(result["response"], sample) for result, sample in zip(results, samples)]
    for result, row in zip(results, outcomes):
        if row["correct"]:
            manager.update_complementarity(np.asarray(result["spatial_adjacency"]), solved=True)
    beta = 1.0 + batch_index * LOGICAL_BATCH_SIZE / 100.0
    manager.update_beta_weight(beta)
    scores = manager.get_edge_score()
    threshold = dynamic_threshold(beta=beta, base_threshold=.65, max_threshold=.9, growth_rate=1.)
    size = len(graph.node_order)
    active = (graph.spatial_masks.detach().cpu().numpy().reshape(size, size) > 0) & ~np.eye(size, dtype=bool)
    eligible = (batch_index + 1 >= 10 and (batch_index + 1) % 5 == 0 and pruning_count < 7)
    pruned, removed = False, []
    below = int(np.sum(scores[active] < threshold))
    if eligible and below:
        graph.update_masks_with_zero_extension(.25, similarity_matrix=scores)
        removed = original.remove_isolated_nodes(graph, manager, 2)
        pruning_count += 1; pruned = True
    metric = {"batch_index": batch_index, "sample_count": len(samples), "beta": beta, "threshold": threshold,
              "pruned": pruned, "pruning_count": pruning_count, "removed_nodes": removed,
              "correct": sum(r["correct"] for r in outcomes), "node_order": list(graph.node_order),
              "eligible_edges_below_threshold": below, "spatial_edges": int(graph.spatial_masks.sum())}
    return outcomes, metric, pruning_count


async def train(ctx, *, resume=False, llm_factory=None):
    path = ctx["directory"] / "training_checkpoint.json"
    samples = ctx["alignment"]["calibration_samples"]
    state = _load_checkpoint(path, ctx, "training") if path.exists() else None
    if state and not resume: raise FileExistsError("Training checkpoint exists; use --resume")
    if state:
        validate_records(ctx, state, samples)
        if (len(state["records"]) != min(state["next_batch"] * 4, 40)
                or len(state["batches"]) != state["next_batch"]):
            raise ValueError("Training checkpoint is not a complete logical-batch prefix")
        if file_sha(ctx["directory"] / "similarity.json") != state["similarity_sha256"]:
            raise ValueError("Training similarity changed")
        if state["complete"]:
            if len(state["records"]) != 40: raise ValueError("Incomplete training marked complete")
            freeze_json(ctx["directory"] / "policy.json", state["policy"])
            return state
    similarity = await ensure_similarity(ctx)
    if not state: seed_run()
    graph = build_graph(ctx, state["policy"] if state else None, llm_factory=llm_factory)
    if state:
        manager = restore_manager(state["manager"]); restore_rng(state["rng"])
    else:
        manager = EdgeMatrixManager(4, similarity, alpha=.5, beta=1., epsilon=1e-6)
        state = {"phase": "training", "run_fingerprint": ctx["config"]["fingerprint"], "complete": False,
                 "next_batch": 0, "pruning_count": 0, "records": [], "batches": [],
                 "similarity_sha256": file_sha(ctx["directory"] / "similarity.json")}
    async with graph.decision_node.llm.async_batching(4):
        for batch_index in range(state["next_batch"], 10):
            offset = batch_index * 4; batch = samples[offset:offset + 4]
            results = await run_batch(graph, batch)
            outcomes, metric, count = apply_statistics_and_pruning(graph, manager, results, batch,
                batch_index=batch_index, pruning_count=state["pruning_count"])
            for index, (sample, result, row) in enumerate(zip(batch, results, outcomes), offset):
                state["records"].append({"sample_index": index, "calibration_pool_index": index,
                    "sample_sha256": digest(sample), "response": result["response"], **row,
                    **_save_trace(ctx, "training", index, sample, result)})
            state.update(next_batch=batch_index + 1, pruning_count=count, policy=policy_state(graph),
                manager=manager_state(manager), rng=rng_state(), complete=len(state["records"]) == 40)
            state["batches"].append(metric); verify_context(ctx); _save_checkpoint(path, state)
            print(f"Training batch {batch_index+1}/10: {len(state['records'])}/40; pruned={metric['pruned']}", flush=True)
    freeze_json(ctx["directory"] / "policy.json", state["policy"])
    return state


async def evaluate(ctx, *, resume=False, llm_factory=None):
    training = _load_checkpoint(ctx["directory"] / "training_checkpoint.json", ctx, "training")
    if not training["complete"] or len(training["records"]) != 40:
        raise ValueError("Complete training is required")
    validate_records(ctx, training, ctx["alignment"]["calibration_samples"])
    if file_sha(ctx["directory"] / "similarity.json") != training["similarity_sha256"]:
        raise ValueError("Trained role similarity changed")
    policy_path = ctx["directory"] / "policy.json"; policy = json.loads(policy_path.read_text())
    if policy != training["policy"]: raise ValueError("Published policy differs from training")
    path = ctx["directory"] / "evaluation_checkpoint.json"
    state = _load_checkpoint(path, ctx, "evaluation") if path.exists() else None
    samples = ctx["alignment"]["evaluation_samples"]
    if state and not resume: raise FileExistsError("Evaluation exists; use --resume")
    if state:
        validate_records(ctx, state, samples)
        if not state["complete"] and len(state["records"]) % 4: raise ValueError("Incomplete microbatch checkpoint")
        if state["policy_sha256"] != file_sha(policy_path): raise ValueError("Evaluation policy changed")
        if state["complete"]:
            if len(state["records"]) != len(samples): raise ValueError("Incomplete evaluation marked complete")
            return summarize(ctx)
    graph = build_graph(ctx, policy, llm_factory=llm_factory)
    if state: restore_rng(state["rng"])
    else:
        restore_rng(training["rng"]); state = {"phase": "evaluation", "run_fingerprint": ctx["config"]["fingerprint"],
                             "policy_sha256": file_sha(policy_path), "records": [], "complete": False}
    async with graph.decision_node.llm.async_batching(4):
        for offset in range(len(state["records"]), len(samples), 4):
            batch = samples[offset:offset + 4]
            with torch.no_grad(): results = await run_batch(graph, batch)
            for index, (sample, result) in enumerate(zip(batch, results), offset):
                state["records"].append({"sample_index": index, "sample_sha256": digest(sample),
                    "response": result["response"], **outcome(result["response"], sample),
                    **_save_trace(ctx, "evaluation", index, sample, result)})
            if policy_state(graph) != policy: raise ValueError("Evaluation modified frozen policy")
            state.update(rng=rng_state(), complete=len(state["records"]) == len(samples))
            verify_context(ctx); _save_checkpoint(path, state)
            print(f"Evaluation {len(state['records'])}/{len(samples)}", flush=True)
    return summarize(ctx)


def summarize(ctx):
    verify_context(ctx)
    training = _load_checkpoint(ctx["directory"] / "training_checkpoint.json", ctx, "training")
    if not training["complete"] or len(training["records"]) != 40:
        raise ValueError("Complete training is required before publication")
    validate_records(ctx, training, ctx["alignment"]["calibration_samples"])
    if file_sha(ctx["directory"] / "similarity.json") != training["similarity_sha256"]:
        raise ValueError("Trained role similarity changed")
    if json.loads((ctx["directory"] / "policy.json").read_text()) != training["policy"]:
        raise ValueError("Published policy differs from training")
    state = _load_checkpoint(ctx["directory"] / "evaluation_checkpoint.json", ctx, "evaluation")
    samples = ctx["alignment"]["evaluation_samples"]; validate_records(ctx, state, samples)
    if not state["complete"] or len(state["records"]) != len(samples):
        raise ValueError("Evaluation incomplete; refusing final accuracy")
    if file_sha(ctx["directory"] / "policy.json") != state["policy_sha256"]:
        raise ValueError("Evaluation policy changed")
    totals = {"llm_calls": 0, "prompt_tokens": 0, "generated_tokens": 0, "elapsed_seconds": 0.}
    for record in state["records"]:
        trace = json.loads(Path(record["trace_json"]).read_text())
        totals["elapsed_seconds"] += trace["elapsed_seconds"]
        for event in trace["events"]:
            totals["llm_calls"] += 1; totals["prompt_tokens"] += np.asarray(event["input_ids"]).size
            totals["generated_tokens"] += np.asarray(event["generated_ids"]).size
    correct = sum(r["correct"] for r in state["records"])
    result = {"complete": True, "run_fingerprint": ctx["config"]["fingerprint"], "total": len(samples),
        "correct": correct, "accuracy": correct / len(samples), "training_unique_questions": 40,
        "training_draws": 40, "logical_training_batch_size": 4, "eval_batch_size": 4, **totals,
        "elapsed_semantics": "Sum of per-question durations; concurrent durations overlap",
        "policy_sha256": state["policy_sha256"]}
    freeze_json(ctx["directory"] / "summary.json", result); return result


def main(argv=None):
    args = parse_args(argv); ctx = prepare_context(args)
    cache = ctx["directory"] / "similarity.json"
    if not cache.exists():
        cache = args.similarity_json
    if not cache.is_file():
        raise FileNotFoundError(f"Validated GSM8K similarity cache is required: {cache}")
    _validate_similarity_payload(ctx, json.loads(cache.read_text()))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(ctx["directory"]), "configuration": ctx["config"]}, indent=2)); return
    if args.stage in ("all", "train", "eval") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for GPU stages")
    ctx["directory"].mkdir(parents=True, exist_ok=True)
    with (ctx["directory"] / ".aligned.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_run(ctx)
        if args.stage in ("prepare", "all", "train"):
            asyncio.run(ensure_similarity(ctx))
        os.environ.update(HF_CHAT_ENABLE_THINKING="0", HF_CHAT_CAPTURE_SCORES="0", TOKENIZERS_PARALLELISM="false")
        result = {"prepared": True, "run_dir": str(ctx["directory"])}
        if args.stage in ("all", "train"): asyncio.run(train(ctx, resume=args.resume))
        if args.stage in ("all", "eval"): result = asyncio.run(evaluate(ctx, resume=args.resume))
        elif args.stage == "summarize": result = summarize(ctx)
        print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
