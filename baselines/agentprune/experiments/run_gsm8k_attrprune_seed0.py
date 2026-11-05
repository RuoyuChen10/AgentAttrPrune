"""AgentPrune GSM8K with the exact AgentAttrPrune seed0 train40 and prompt contract.

The existing Graph performs sampling and scheduling. The original GSM8K policy
loss plus raw-logit regularization, 40 Adam updates,
ten rollouts per question and one-shot 50% pruning remain unchanged. The exact
seed0 pool, prompts, decoding and generation microbatching match AgentAttrPrune.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments import attrprune_gsm8k_seed0_adapter as adapter
from experiments_attrprune.baseline_gsm8k_alignment import load_alignment, verify_sources
from AgentPrune.graph.graph import Graph
from dataset.gsm8k_dataset import gsm_get_predict
from experiments.regularization import graph_regularization
from experiments.pruning_artifacts import build_pruning_artifact, save_pruning_artifacts
from experiments.run_mmlu_attrprune_aligned import (
    file_sha, jsonable, digest, write_json, freeze_json, policy_state, policy_digest, rng_state, restore_rng, commit, next_attempt, relative_records
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = adapter.ATTRPRUNE_ROOT / "results/AgentAttrPrune/gsm8k/qwen3_8b/train40_correctness_aware"
DEFAULT_PROTOCOL = ROOT / "assets/gsm8k_protocol.json"
DEFAULT_OUTPUT = ROOT / "result/gsm8k/gsm8k_attrprune_seed0_train40_t0"
POLICY_FIELDS = ("spatial_logits", "temporal_logits", "spatial_masks", "temporal_masks")


def protocol_from_artifact(path):
    saved = json.loads(Path(path).read_text())
    required = {"num_rounds": 2, "num_iterations": 40, "train_batch_size": 1,
                "samples_per_query": 10, "lr": 0.1, "prune_at_end": True,
                "pruning_rate": 0.5, "low_rank_regularization": True, "delta": 0.1,
                "seed": 0, "max_new_tokens": 2048,
                "optimized_spatial": True, "optimized_temporal": True}
    for name, expected in required.items():
        if saved.get(name) != expected:
            raise ValueError(f"Original GSM8K protocol differs: {name}")
    if sum(saved["agent_nums"]) != 4 or saved["mode"] != "FullConnected":
        raise ValueError("Expected the original four-worker GSM8K baseline")
    return {**required, "mode": "FullConnected", "temperature": 0.0,
            "eval_batch_size": 4, "llm_microbatch_size": 4, "pool_size": 40,
            "train_rollouts": 400, "eval_size": 1319,
            "loss": "mean(-log_probability * correctness) + original nuclear norm and Frobenius hinge on raw logits",
            "sampling": "Original Graph sigmoid/Bernoulli sampling and spatial cycle rejection; shared weights across rounds",
            "pruning_schedule": "One-shot 50% separately for spatial and temporal after update 40",
            "rounds": "Always execute both rounds, including disconnected workers",
            "selection": "Exact original AgentAttrPrune seed0 train40 order; ten rollouts per question",
            "answer_parser": "Original AgentPrune gsm_get_predict",
            "reference_protocol_sha256": file_sha(path)}


def training_plan(protocol):
    return [index for index in range(protocol["pool_size"])
            for _ in range(protocol["samples_per_query"])]


def make_manifest(args, alignment):
    protocol = protocol_from_artifact(args.protocol_reference)
    base = alignment["reference_run_config"]
    if (len(alignment["workers"]) != 4 or len(alignment["calibration_samples"]) != 40
            or len(alignment["evaluation_samples"]) != 1319
            or base["decoding"] != {"temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}):
        raise ValueError("Expected the complete original AttrPrune GSM8K configuration")
    model_name = base["model_name"]
    if not model_name.startswith(("local:", "hf:")):
        model_name = "local:" + model_name
    files = [Path(__file__), Path(adapter.__file__), Path(args.protocol_reference),
             ROOT / "experiments/run_mmlu_attrprune_aligned.py",
             ROOT / "experiments/attrprune_mmlu_prompt_adapter.py",
             ROOT / "AgentPrune/graph/graph.py", ROOT / "AgentPrune/graph/node.py",
             ROOT / "experiments/regularization.py", ROOT / "dataset/gsm8k_dataset.py",
             ROOT / "experiments/pruning_artifacts.py", adapter.ATTRPRUNE_ROOT / "AgentAttrPrune/llm/hf_chat.py"]
    manifest = {"schema_version": 1, "experiment": "gsm8k_attrprune_seed0_train40_t0",
                "alignment_fingerprint": alignment["fingerprint"], "model_name": model_name,
                "protocol": protocol, "training_plan": training_plan(protocol),
                "source_hashes": {**alignment["source_hashes"], **{str(p.resolve()): file_sha(p) for p in files}},
                "workers": alignment["workers"], "reviewer_agent_ids": alignment["reviewer_agent_ids"],
                "calibration_samples_sha256": digest(alignment["calibration_samples"]),
                "evaluation_samples_sha256": digest(alignment["evaluation_samples"])}
    manifest["fingerprint"] = digest(manifest)
    return manifest


def verify_manifest_sources(manifest):
    for path, expected in manifest["source_hashes"].items():
        if file_sha(path) != expected:
            raise ValueError(f"Source changed during the experiment: {path}")


def restore_policy(graph, saved):
    for name in POLICY_FIELDS:
        value = saved[name]
        if not torch.is_tensor(value) or value.shape != (16,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid saved policy tensor: {name}")
        if name.endswith("masks") and not torch.all((value == 0) | (value == 1)):
            raise ValueError(f"Nonbinary policy mask: {name}")
        with torch.no_grad():
            getattr(graph, name).copy_(value)
    if torch.any(graph.spatial_masks.reshape(4, 4).diag()):
        raise ValueError("Spatial self-connections are forbidden")


def load_checkpoint(run_dir, manifest, alignment):
    pointer_path = run_dir / "checkpoint.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(pointer_path.read_text())
    path = (run_dir / pointer["path"]).resolve()
    if run_dir.resolve() not in path.parents or file_sha(path) != pointer["sha256"]:
        raise ValueError("Checkpoint file is foreign or modified")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["fingerprint"] != manifest["fingerprint"] or pointer["fingerprint"] != saved["fingerprint"]:
        raise ValueError("Checkpoint configuration changed")
    if (pointer["stage"] != saved["stage"] or pointer["iteration"] != saved["iteration"]
            or pointer["evaluated"] != len(saved["evaluation"])
            or digest(saved["policy"]) != saved["policy_sha256"]):
        raise ValueError("Checkpoint metadata or policy hash mismatch")
    if not 0 <= saved["iteration"] <= 40 or len(saved["training"]) != saved["iteration"] * 10:
        raise ValueError("Training checkpoint has missing or duplicated rollouts")
    if saved["evaluation"] and saved["iteration"] != 40:
        raise ValueError("Evaluation cannot precede completed policy training")
    if len(saved["evaluation"]) > 1319:
        raise ValueError("Too many saved evaluation questions")
    for phase, records in (("training", saved["training"]), ("evaluation", saved["evaluation"])):
        for offset, row in enumerate(records):
            expected_index = manifest["training_plan"][offset] if phase == "training" else offset
            if row["sample_index"] != expected_index or row["phase"] != phase or row["ordinal"] != offset:
                raise ValueError("Saved question identity/order is inconsistent")
            trace_path = (run_dir / row["trace"]).resolve()
            if run_dir.resolve() not in trace_path.parents or file_sha(trace_path) != row["trace_sha256"]:
                raise ValueError("Saved trace is missing, modified or foreign")
            trace = json.loads(trace_path.read_text())
            sample = alignment["calibration_samples" if phase == "training" else "evaluation_samples"][expected_index]
            validate_trace(trace, sample, manifest)
            if trace["identity"] != {key: row[key] for key in ("phase", "sample_index", "ordinal")}:
                raise ValueError("Saved trace identity differs from checkpoint")
            if row["prediction"] != trace["final"]["prediction"] or row["correct"] != trace["final"]["correct"]:
                raise ValueError("Saved score differs from the trace")
        public = run_dir / ("training_results.json" if phase == "training" else "results.json")
        if public.exists():
            visible = json.loads(public.read_text())
            if visible != records[:len(visible)]:
                raise ValueError("Published results differ from the committed checkpoint")
    return saved


class Invocation:
    def __init__(self, recorder, node, spatial, temporal, messages):
        self.recorder = recorder
        round_index = recorder.calls[node.id]
        recorder.calls[node.id] += 1
        if node.agent_name == adapter.REVIEWER_NAME:
            round_index = 2
        self.metadata = {"agent_id": node.id, "role": node.role, "round": round_index,
                         "input_messages": messages, "incoming": []}
        for kind, info in (("spatial", spatial), ("temporal", temporal)):
            source_round = (1 if node.agent_name == adapter.REVIEWER_NAME else round_index) - int(kind == "temporal")
            self.metadata["incoming"].extend({"edge_type": kind, "agent_id": source, "role": row["role"],
                "source_round": source_round, "output": row["output"]} for source, row in info.items())

    def capture(self, event):
        if event["messages"] != self.metadata["input_messages"]:
            raise ValueError("Backend prompt differs from the recorded node prompt")
        output = event["output_text"]
        self.recorder.events.append(jsonable({**self.metadata, "generation": event,
            "output": output[0] if isinstance(output, list) else output}))

    def fail(self, error):
        self.recorder.errors.append(f"{self.metadata['agent_id']}: {type(error).__name__}: {error}")


class RolloutRecorder:
    def __init__(self):
        self.calls, self.events, self.errors = Counter(), [], []

    def begin(self, node, spatial, temporal, messages):
        return Invocation(self, node, spatial, temporal, messages)


def parse_answer(text):
    # This parser does not inspect instance state; retain the original baseline scorer.
    return gsm_get_predict(text)


def validate_trace(trace, sample, manifest):
    if (trace["fingerprint"] != manifest["fingerprint"] or trace["sample"] != sample
            or trace.get("errors") or len(trace["events"]) != 9):
        raise ValueError("Incomplete, failed or mismatched GSM8K rollout")
    ids = [row["agent_id"] for row in manifest["workers"]]
    expected = {(round_index, agent_id) for round_index in range(2) for agent_id in ids}
    expected.add((2, "aligned_gsm8k_reviewer"))
    events = {(row["round"], row["agent_id"]): row for row in trace["events"]}
    if set(events) != expected or len(events) != 9:
        raise ValueError("Missing or duplicated node invocation")
    for event in trace["events"]:
        for source in event["incoming"]:
            key = (source["source_round"], source["agent_id"])
            if key not in events or events[key]["output"] != source["output"]:
                raise ValueError("Incoming message has the wrong source output or round")
        if event["generation"]["generation_kwargs"].get("temperature") != 0:
            raise ValueError("Generation temperature changed")
    final = trace["final"]
    if (final["output"] != events[(2, "aligned_gsm8k_reviewer")]["output"]
            or final["prediction"] != parse_answer(final["output"])
            or final["correct"] != (final["prediction"] == sample["answer"])):
        raise ValueError("Inconsistent final answer or score")


def token_count(value):
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return len(value)


def new_graph(manifest):
    return Graph(domain="gsm8k", llm_name=manifest["model_name"],
        agent_names=[adapter.WORKER_NAME] * 4, decision_method=adapter.REVIEWER_NAME,
        optimized_spatial=True, optimized_temporal=True,
        node_kwargs=[{"id": row["agent_id"], "role": row["role"]} for row in manifest["workers"]])


async def run_batch(graph, samples, identities, output, manifest):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    async def one(sample, identity):
        rollout_started = time.perf_counter()
        realized = copy.deepcopy(graph)
        realized.spatial_logits, realized.temporal_logits = graph.spatial_logits, graph.temporal_logits
        recorder = RolloutRecorder()
        token = adapter.invocation_recorder.set(recorder)
        try:
            answers, log_probability = await realized.arun({"task": sample["task"]}, num_rounds=2)
        finally:
            adapter.invocation_recorder.reset(token)
        output_text = answers[0]
        prediction = parse_answer(output_text)
        trace = {"schema_version": 1, "fingerprint": manifest["fingerprint"], "identity": identity,
                 "sample": sample, "policy_sha256": policy_digest(graph), "events": recorder.events,
                 "errors": recorder.errors, "final": {"output": output_text, "prediction": prediction,
                   "gold_answer": sample["answer"], "correct": prediction == sample["answer"]}}
        ids = [row["agent_id"] for row in manifest["workers"]]
        positions = {agent_id: index for index, agent_id in enumerate(ids)}
        matrices = {kind: np.zeros((2, 4, 4), dtype=int) for kind in ("spatial", "temporal")}
        for event in recorder.events:
            if event["agent_id"] not in positions or event["round"] not in (0, 1):
                continue
            for incoming in event["incoming"]:
                matrices[incoming["edge_type"]][event["round"], positions[incoming["agent_id"]], positions[event["agent_id"]]] = 1
        trace["realized_graph"] = {"agents": ids, "matrix_orientation": "[round][source][target]",
                                  **{name: matrix.tolist() for name, matrix in matrices.items()}}
        path = output / f"rollout_{identity['ordinal']:05d}.json"
        write_json(path, trace)
        validate_trace(trace, sample, manifest)
        return {**identity, "prediction": prediction, "correct": prediction == sample["answer"],
                "llm_calls": len(recorder.events),
                "prompt_tokens": sum(token_count(event["generation"]["input_ids"]) for event in recorder.events),
                "generated_tokens": sum(token_count(event["generation"]["generated_ids"]) for event in recorder.events),
                "elapsed_seconds": time.perf_counter() - rollout_started,
                "trace": str(path), "trace_sha256": file_sha(path)}, log_probability

    async with adapter.backend().async_batching(batch_size=4):
        results = await asyncio.gather(*(one(sample, identity) for sample, identity in zip(samples, identities)))
    return results, time.perf_counter() - started


def export_policy(graph, run_dir, manifest):
    directory = run_dir / "pruning_artifacts"
    path = directory / "learned_edge_policy.json"
    artifact = build_pruning_artifact(graph, training_config=manifest)
    expected = {key: value for key, value in artifact.items() if key != "_checkpoint_tensors"}
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("Saved trained policy changed")
        checkpoint = directory / "learned_edge_policy.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved.get("workers") != expected["workers"] or saved.get("training_config") != manifest:
                raise ValueError("Exported policy checkpoint metadata changed")
            for name in POLICY_FIELDS:
                if not torch.equal(saved[name].reshape(-1), getattr(graph, name).detach().cpu()):
                    raise ValueError("Exported policy checkpoint differs from the committed policy")
        if checkpoint.exists() and (directory / "learned_edge_policy.png").exists():
            return
        # Recover an interrupted export only after validating every existing
        # policy value; this does not change the learned graph or evaluation.
    save_pruning_artifacts(graph, directory, training_config=manifest)


async def execute(args, alignment, manifest, saved):
    for name, value in {"HF_CHAT_ENABLE_THINKING": "0", "HF_CHAT_CAPTURE_SCORES": "0",
                        "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[name] = value
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    adapter.configure(alignment, model_name=manifest["model_name"], max_new_tokens=2048)
    graph = new_graph(manifest)
    fixed_masks = {kind: getattr(graph, f"{kind}_masks").detach().clone().to(
        dtype=getattr(graph, f"{kind}_logits").dtype).reshape(4, 4) for kind in ("spatial", "temporal")}
    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=0.1)
    state = copy.deepcopy(saved) if saved is not None else {
        "fingerprint": manifest["fingerprint"], "stage": "training", "iteration": 0,
        "training": [], "evaluation": []}
    if saved:
        restore_policy(graph, saved["policy"])
        if state["iteration"] < 40:
            if saved["optimizer"] is None:
                raise ValueError("Interrupted training requires the Adam checkpoint")
            optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    else:
        commit(args.run_dir, state, graph, optimizer)
    write_json(args.run_dir / "runtime.json", {"pid": os.getpid(), "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "llm_microbatch_size": 4,
        "model_device_map": {key: str(value) for key, value in getattr(getattr(adapter.backend(), "model", None), "hf_device_map", {}).items()}})
    for iteration in range(state["iteration"], 40):
        if args.stage == "eval":
            raise ValueError("Evaluation requires all 40 policy updates to be complete")
        verify_manifest_sources(manifest)
        indices = manifest["training_plan"][iteration * 10:(iteration + 1) * 10]
        identities = [{"phase": "training", "sample_index": index, "ordinal": iteration * 10 + offset}
                      for offset, index in enumerate(indices)]
        results, elapsed = await run_batch(graph, [alignment["calibration_samples"][index] for index in indices],
            identities, next_attempt(args.run_dir, "training", iteration), manifest)
        # Exactly the existing GSM8K reward loss and author regularizer.
        policy_loss = torch.stack([-log_probability * row["correct"] for row, log_probability in results]).mean()
        regularization = graph_regularization(graph, fixed_masks, delta=0.1)
        loss = policy_loss + regularization
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if iteration == 39:
            graph.update_masks(0.5)
        state["training"].extend(relative_records(results, args.run_dir))
        state["iteration"] = iteration + 1
        state["stage"] = "trained" if iteration == 39 else "training"
        commit(args.run_dir, state, graph, optimizer)
        print(f"Training {iteration + 1}/40: {len(state['training'])}/400 rollouts; "
              f"correct={sum(row['correct'] for row, _ in results)}/10; "
              f"policy_loss={policy_loss.item():.6f}; regularization={regularization.item():.6f}; masks={int(graph.spatial_masks.sum())}S/{int(graph.temporal_masks.sum())}T; seconds={elapsed:.2f}", flush=True)
    if (int(graph.spatial_masks.sum()), int(graph.temporal_masks.sum())) != (6, 8):
        raise ValueError("Final pruning masks differ from the original GSM8K schedule")
    export_policy(graph, args.run_dir, manifest)
    if args.stage == "train":
        return {"training_complete": True, "train_rollouts": 400, "evaluation_count": len(state["evaluation"])}
    frozen = policy_digest(graph)
    graph.spatial_logits.requires_grad_(False); graph.temporal_logits.requires_grad_(False)
    for offset in range(len(state["evaluation"]), 1319, 4):
        indices = list(range(offset, min(offset + 4, 1319)))
        identities = [{"phase": "evaluation", "sample_index": index, "ordinal": index} for index in indices]
        with torch.no_grad():
            results, elapsed = await run_batch(graph, [alignment["evaluation_samples"][index] for index in indices],
                identities, next_attempt(args.run_dir, "evaluation", offset), manifest)
        if policy_digest(graph) != frozen:
            raise ValueError("Policy changed during evaluation")
        state["evaluation"].extend(relative_records(results, args.run_dir))
        state["stage"] = "complete" if len(state["evaluation"]) == 1319 else "evaluation"
        commit(args.run_dir, state, graph)
        correct = sum(row["correct"] for row in state["evaluation"])
        print(f"Test {len(state['evaluation'])}/1319; correct={correct}; "
              f"accuracy={correct / len(state['evaluation']):.6f}; batch_seconds={elapsed:.2f}", flush=True)
    verify_manifest_sources(manifest)
    summary = {"completed": True, "fingerprint": manifest["fingerprint"], "total": 1319,
        "correct": sum(row["correct"] for row in state["evaluation"]), "training_pool_size": 40,
        "training_rollouts": 400, "training_updates": 40, "eval_batch_size": 4, "llm_microbatch_size": 4,
        "spatial_retained": 6, "temporal_retained": 8, "policy_unchanged_during_evaluation": True,
        "sampling": manifest["protocol"]["sampling"], "alignment_fingerprint": alignment["fingerprint"]}
    summary["accuracy"] = summary["correct"] / summary["total"]
    summary["token_usage"] = {key: sum(row[key] for row in state["evaluation"])
                             for key in ("llm_calls", "prompt_tokens", "generated_tokens")}
    write_json(args.run_dir / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--protocol-reference", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    alignment = load_alignment(args.reference_run.resolve())
    manifest = make_manifest(args, alignment)
    verify_sources(alignment)
    verify_manifest_sources(manifest)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(args.run_dir), "fingerprint": manifest["fingerprint"],
                          "alignment_fingerprint": alignment["fingerprint"], "protocol": manifest["protocol"],
                          "draws_per_calibration_question": dict(Counter(manifest["training_plan"])),
                          "reviewer_agent_ids": alignment["reviewer_agent_ids"]}, indent=2))
        return
    if args.stage != "prepare" and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Set CUDA_VISIBLE_DEVICES explicitly")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".aligned_gsm8k.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = {path.name for path in args.run_dir.iterdir()} - {".aligned_gsm8k.lock", "run.log", "launch.sh", "command.txt"}
        if entries and not args.resume:
            raise FileExistsError("Experiment already exists; use --resume or an independent directory")
        if entries and not (args.run_dir / "manifest.json").exists():
            raise ValueError("Nonempty experiment has no frozen manifest")
        freeze_json(args.run_dir / "manifest.json", manifest)
        freeze_json(args.run_dir / "alignment.json", alignment)
        if args.stage == "prepare":
            print(json.dumps({"prepared": True, "run_dir": str(args.run_dir), "fingerprint": manifest["fingerprint"], "protocol": manifest["protocol"]}, indent=2))
            return
        saved = load_checkpoint(args.run_dir, manifest, alignment)
        if args.stage == "eval" and (saved is None or saved["iteration"] != 40):
            raise ValueError("Evaluation requires all 40 policy updates to be complete")
        if saved and saved["stage"] == "complete" and (args.run_dir / "summary.json").exists():
            summary = json.loads((args.run_dir / "summary.json").read_text())
            correct = sum(row["correct"] for row in saved["evaluation"])
            if (len(saved["evaluation"]) != 1319 or summary.get("fingerprint") != manifest["fingerprint"]
                    or summary.get("total") != 1319 or summary.get("correct") != correct
                    or summary.get("accuracy") != correct / 1319 or summary.get("completed") is not True):
                raise ValueError("Completed summary differs from the verified checkpoint")
            print(json.dumps(summary, indent=2))
            return
        print(json.dumps(asyncio.run(execute(args, alignment, manifest, saved)), indent=2))


if __name__ == "__main__":
    main()
