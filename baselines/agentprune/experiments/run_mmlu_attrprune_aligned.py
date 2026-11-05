"""AgentPrune MMLU with the exact AgentAttrPrune seed0 dev40 and prompt contract.

The existing Graph performs sampling and scheduling. The original MMLU policy
loss, Adam updates, 16-rollout training batches, and two pruning updates remain.
Only the 40-question pool, prompts, and generation microbatching are aligned.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from experiments import attrprune_mmlu_prompt_adapter as adapter
from experiments_attrprune.baseline_mmlu_alignment import load_alignment, verify_sources
from AgentPrune.graph.graph import Graph
from dataset.mmlu_dataset import MMLUDataset
from experiments.pruning_artifacts import build_pruning_artifact, save_pruning_artifacts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = adapter.ATTRPRUNE_ROOT / "results/AgentAttrPrune/mmlu/qwen3_8b/dev40_correctness_aware"
DEFAULT_PROTOCOL = ROOT / "assets/mmlu_protocol.json"
DEFAULT_OUTPUT = ROOT / "result/mmlu/mmlu_attrprune_prompt_dev40_t0"
POLICY_FIELDS = ("spatial_logits", "temporal_logits", "spatial_masks", "temporal_masks")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen experiment metadata changed: {path}")
    else:
        write_json(path, value)


def protocol_from_artifact(path):
    saved = json.loads(Path(path).read_text())["training_config"]
    required = {"num_rounds": 2, "num_iterations": 10, "batch_size": 16,
                "learning_rate": 0.1, "prune_every_iterations": 5,
                "pruning_rate": 0.25, "seed": 0, "temperature": 0.0,
                "max_new_tokens": 2048, "optimized_spatial": True, "optimized_temporal": True}
    for name, expected in required.items():
        if saved.get(name) != expected:
            raise ValueError(f"Original MMLU protocol differs: {name}")
    if len(saved["agent_names"]) != 5 or saved["mode"] != "FullConnected":
        raise ValueError("Expected the existing five-worker FullConnected MMLU baseline")
    return {**required, "mode": "FullConnected", "train_batch_size": saved["batch_size"],
            "eval_batch_size": 4, "llm_microbatch_size": 4, "pool_size": 40,
            "train_rollouts": 160, "eval_size": 1531,
            "loss": "mean(-log_probability * correctness); original MMLU, no low-rank regularizer",
            "sampling": "Original Graph sigmoid/Bernoulli sampling and spatial cycle rejection; shared weights across rounds",
            "pruning_schedule": "25% separately for spatial and temporal after updates 5 and 10",
            "rounds": "Always execute both rounds, including disconnected workers",
            "selection": "Four independent seeded permutations of the same aligned dev40 pool",
            "answer_parser": "Original AgentPrune MMLUDataset.postprocess_answer",
            "reference_policy_sha256": file_sha(path)}


def training_plan(protocol):
    rng = np.random.RandomState(protocol["seed"])
    draws = []
    while len(draws) < protocol["train_rollouts"]:
        draws.extend(int(index) for index in rng.permutation(protocol["pool_size"]))
    return draws[:protocol["train_rollouts"]]


def make_manifest(args, alignment):
    protocol = protocol_from_artifact(args.protocol_reference)
    base = alignment["reference_run_config"]
    if (len(alignment["workers"]) != 5 or len(alignment["calibration_samples"]) != 40
            or len(alignment["evaluation_samples"]) != 1531
            or base["decoding"] != {"temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False}):
        raise ValueError("Expected the complete original AttrPrune MMLU configuration")
    model_name = base["model_name"]
    if not model_name.startswith(("local:", "hf:")):
        model_name = "local:" + model_name
    files = [Path(__file__), Path(adapter.__file__), Path(args.protocol_reference),
             ROOT / "AgentPrune/graph/graph.py", ROOT / "AgentPrune/graph/node.py",
             ROOT / "dataset/mmlu_dataset.py",
             ROOT / "experiments/pruning_artifacts.py", adapter.ATTRPRUNE_ROOT / "AgentAttrPrune/llm/hf_chat.py"]
    manifest = {"schema_version": 1, "experiment": "mmlu_attrprune_prompt_dev40_t0",
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


def policy_state(graph):
    return {name: getattr(graph, name).detach().cpu().clone() for name in POLICY_FIELDS}


def policy_digest(graph):
    return digest(policy_state(graph))


def restore_policy(graph, saved):
    for name in POLICY_FIELDS:
        value = saved[name]
        if not torch.is_tensor(value) or value.shape != (25,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid saved policy tensor: {name}")
        if name.endswith("masks") and not torch.all((value == 0) | (value == 1)):
            raise ValueError(f"Nonbinary policy mask: {name}")
        with torch.no_grad():
            getattr(graph, name).copy_(value)
    if torch.any(graph.spatial_masks.reshape(5, 5).diag()):
        raise ValueError("Spatial self-connections are forbidden")


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(saved["torch"])
    if saved["cuda"]:
        if not torch.cuda.is_available() or len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(saved["cuda"])


def commit(run_dir, state, graph, optimizer=None):
    payload = {**state, "policy": policy_state(graph), "policy_sha256": policy_digest(graph),
               "rng": rng_state(), "optimizer": optimizer.state_dict() if optimizer is not None else None}
    directory = run_dir / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"{state['stage']}_{state['iteration']:02d}_{len(state['evaluation']):05d}_{uuid.uuid4().hex}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    write_json(run_dir / "checkpoint.json", {"path": str(path.relative_to(run_dir)), "sha256": file_sha(path),
               "fingerprint": state["fingerprint"], "stage": state["stage"], "iteration": state["iteration"],
               "evaluated": len(state["evaluation"])})
    write_json(run_dir / "training_results.json", state["training"])
    write_json(run_dir / "results.json", state["evaluation"])


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
    if not 0 <= saved["iteration"] <= 10 or len(saved["training"]) != saved["iteration"] * 16:
        raise ValueError("Training checkpoint has missing or duplicated rollouts")
    if saved["evaluation"] and saved["iteration"] != 10:
        raise ValueError("Evaluation cannot precede completed policy training")
    if len(saved["evaluation"]) > 1531:
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
    return MMLUDataset.postprocess_answer(None, text)


def validate_trace(trace, sample, manifest):
    if (trace["fingerprint"] != manifest["fingerprint"] or trace["sample"] != sample
            or trace.get("errors") or len(trace["events"]) != 11):
        raise ValueError("Incomplete, failed or mismatched MMLU rollout")
    ids = [row["agent_id"] for row in manifest["workers"]]
    expected = {(round_index, agent_id) for round_index in range(2) for agent_id in ids}
    expected.add((2, "aligned_mmlu_reviewer"))
    events = {(row["round"], row["agent_id"]): row for row in trace["events"]}
    if set(events) != expected or len(events) != 11:
        raise ValueError("Missing or duplicated node invocation")
    for event in trace["events"]:
        for source in event["incoming"]:
            key = (source["source_round"], source["agent_id"])
            if key not in events or events[key]["output"] != source["output"]:
                raise ValueError("Incoming message has the wrong source output or round")
        if event["generation"]["generation_kwargs"].get("temperature") != 0:
            raise ValueError("Generation temperature changed")
    final = trace["final"]
    if (final["output"] != events[(2, "aligned_mmlu_reviewer")]["output"]
            or final["prediction"] != parse_answer(final["output"])
            or final["correct"] != (final["prediction"] == sample["answer"])):
        raise ValueError("Inconsistent final answer or score")


def token_count(value):
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return len(value)


def new_graph(manifest):
    return Graph(domain="mmlu", llm_name=manifest["model_name"],
        agent_names=[adapter.WORKER_NAME] * 5, decision_method=adapter.REVIEWER_NAME,
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
        matrices = {kind: np.zeros((2, 5, 5), dtype=int) for kind in ("spatial", "temporal")}
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


def next_attempt(run_dir, phase, index):
    return run_dir / "traces" / phase / f"batch_{index:05d}" / ("attempt_" + uuid.uuid4().hex)


def relative_records(results, run_dir):
    return [{**row, "trace": str(Path(row["trace"]).relative_to(run_dir))} for row, _ in results]


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
    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=0.1)
    state = copy.deepcopy(saved) if saved is not None else {
        "fingerprint": manifest["fingerprint"], "stage": "training", "iteration": 0,
        "training": [], "evaluation": []}
    if saved:
        restore_policy(graph, saved["policy"])
        if state["iteration"] < 10:
            if saved["optimizer"] is None:
                raise ValueError("Interrupted training requires the Adam checkpoint")
            optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    else:
        commit(args.run_dir, state, graph, optimizer)
    write_json(args.run_dir / "runtime.json", {"pid": os.getpid(), "hostname": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "llm_microbatch_size": 4,
        "model_device_map": {key: str(value) for key, value in getattr(getattr(adapter.backend(), "model", None), "hf_device_map", {}).items()}})
    for iteration in range(state["iteration"], 10):
        if args.stage == "eval":
            raise ValueError("Evaluation requires all 10 policy updates to be complete")
        verify_manifest_sources(manifest)
        indices = manifest["training_plan"][iteration * 16:(iteration + 1) * 16]
        identities = [{"phase": "training", "sample_index": index, "ordinal": iteration * 16 + offset}
                      for offset, index in enumerate(indices)]
        results, elapsed = await run_batch(graph, [alignment["calibration_samples"][index] for index in indices],
            identities, next_attempt(args.run_dir, "training", iteration), manifest)
        # Exactly the original MMLU objective and optimizer schedule.
        loss = torch.stack([-log_probability * row["correct"] for row, log_probability in results]).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (iteration + 1) % 5 == 0:
            graph.update_masks(0.25)
        state["training"].extend(relative_records(results, args.run_dir))
        state["iteration"] = iteration + 1
        state["stage"] = "trained" if iteration == 9 else "training"
        commit(args.run_dir, state, graph, optimizer)
        print(f"Training {iteration + 1}/10: {len(state['training'])}/160 rollouts; "
              f"correct={sum(row['correct'] for row, _ in results)}/16; "
              f"loss={loss.item():.6f}; masks={int(graph.spatial_masks.sum())}S/{int(graph.temporal_masks.sum())}T; seconds={elapsed:.2f}", flush=True)
    if (int(graph.spatial_masks.sum()), int(graph.temporal_masks.sum())) != (11, 14):
        raise ValueError("Final pruning masks differ from the original MMLU schedule")
    export_policy(graph, args.run_dir, manifest)
    if args.stage == "train":
        return {"training_complete": True, "train_rollouts": 160, "evaluation_count": len(state["evaluation"])}
    frozen = policy_digest(graph)
    graph.spatial_logits.requires_grad_(False); graph.temporal_logits.requires_grad_(False)
    for offset in range(len(state["evaluation"]), 1531, 4):
        indices = list(range(offset, min(offset + 4, 1531)))
        identities = [{"phase": "evaluation", "sample_index": index, "ordinal": index} for index in indices]
        with torch.no_grad():
            results, elapsed = await run_batch(graph, [alignment["evaluation_samples"][index] for index in indices],
                identities, next_attempt(args.run_dir, "evaluation", offset), manifest)
        if policy_digest(graph) != frozen:
            raise ValueError("Policy changed during evaluation")
        state["evaluation"].extend(relative_records(results, args.run_dir))
        state["stage"] = "complete" if len(state["evaluation"]) == 1531 else "evaluation"
        commit(args.run_dir, state, graph)
        correct = sum(row["correct"] for row in state["evaluation"])
        print(f"Test {len(state['evaluation'])}/1531; correct={correct}; "
              f"accuracy={correct / len(state['evaluation']):.6f}; batch_seconds={elapsed:.2f}", flush=True)
    verify_manifest_sources(manifest)
    summary = {"completed": True, "fingerprint": manifest["fingerprint"], "total": 1531,
        "correct": sum(row["correct"] for row in state["evaluation"]), "training_pool_size": 40,
        "training_rollouts": 160, "training_updates": 10, "eval_batch_size": 4, "llm_microbatch_size": 4,
        "spatial_retained": 11, "temporal_retained": 14, "policy_unchanged_during_evaluation": True,
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
    with (args.run_dir / ".aligned_mmlu.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = {path.name for path in args.run_dir.iterdir()} - {".aligned_mmlu.lock", "run.log", "launch.sh", "command.txt"}
        if entries and not args.resume:
            raise FileExistsError("Experiment already exists; use --resume or an independent directory")
        if entries and not (args.run_dir / "manifest.json").exists():
            raise ValueError("Nonempty experiment has no frozen manifest")
        freeze_json(args.run_dir / "manifest.json", manifest)
        freeze_json(args.run_dir / "alignment.json", alignment)
        if args.stage == "prepare":
            print(json.dumps({"prepared": True, "run_dir": str(args.run_dir),
                              "fingerprint": manifest["fingerprint"], "protocol": manifest["protocol"]}, indent=2))
            return
        saved = load_checkpoint(args.run_dir, manifest, alignment)
        if args.stage == "eval" and (saved is None or saved["iteration"] != 10):
            raise ValueError("Evaluation requires all 10 policy updates to be complete")
        if saved and saved["stage"] == "complete" and (args.run_dir / "summary.json").exists():
            summary = json.loads((args.run_dir / "summary.json").read_text())
            correct = sum(row["correct"] for row in saved["evaluation"])
            if (len(saved["evaluation"]) != 1531 or summary.get("fingerprint") != manifest["fingerprint"]
                    or summary.get("total") != 1531 or summary.get("correct") != correct
                    or summary.get("accuracy") != correct / 1531 or summary.get("completed") is not True):
                raise ValueError("Completed summary differs from the verified checkpoint")
            print(json.dumps(summary, indent=2))
            return
        print(json.dumps(asyncio.run(execute(args, alignment, manifest, saved)), indent=2))


if __name__ == "__main__":
    main()
