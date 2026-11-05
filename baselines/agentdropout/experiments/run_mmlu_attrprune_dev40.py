"""Train a new AgentDropout policy on AgentAttrPrune's exact seed0 MMLU dev40.

The released experiment's 160 node-stage + 160 edge-stage draws are retained.
Each stage visits four shuffled permutations of the same frozen 40 questions.
No policy from an earlier run is reused. All model operations are opt-in.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
ATTR_ROOT = Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT", ROOT.parent.parent)).expanduser().resolve()
for directory in (ROOT, ATTR_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np

from experiments_attrprune.baseline_mmlu_alignment import (
    DEFAULT_REFERENCE, digest, file_sha, freeze_alignment, load_alignment,
)

DEFAULT_RUN = ROOT / "results/AgentDropout/mmlu/qwen3_8b/mmlu_attrprune_prompt_dev40_t0"
DEFAULT_MODEL = None  # Resolve from the validated AgentAttrPrune reference configuration.
TRAINING = {
    "split": "dev", "dataset_size": 40, "node_iterations": 10,
    "edge_iterations": 10, "batch_size": 16, "learning_rate": 0.1,
    "prune_every_iterations": 5, "pruning_rate": 0.25, "delta": 0.1,
    "total_draws": 320, "node_draws": 160, "edge_draws": 160,
    "selection": "continuous seeded shuffled permutations of the exact dev40 pool",
}


def _local_dataset_namespace():
    import datasets
    local = str(ROOT / "datasets")
    if local not in datasets.__path__:
        datasets.__path__.insert(0, local)


class FrozenDataset:
    def __init__(self, samples, split):
        self.samples = samples
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    @staticmethod
    def record_to_input(record):
        return {"task": record["task"]}

    @staticmethod
    def record_to_target_answer(record):
        return record["answer"]

    def postprocess_answer(self, answer):
        # Preserve the original baseline's answer parser and reward definition.
        _local_dataset_namespace()
        from datasets.mmlu_dataset import MMLUDataset
        return MMLUDataset.postprocess_answer(self, answer)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default=None,
                        choices=("prepare", "train", "eval", "summarize", "all"))
    parser.add_argument("--stage", dest="stage_option", choices=("prepare", "train", "eval", "summarize", "all"))
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--llm-batch-size", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=int(os.environ.get("HF_CHAT_NUM_REPLICAS", "2")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage is not None and args.stage_option is not None and args.stage != args.stage_option:
        parser.error("Positional stage and --stage disagree")
    args.stage = args.stage_option or args.stage or "all"
    if min(args.eval_batch_size, args.llm_batch_size, args.replicas) < 1:
        parser.error("Batch sizes and replicas must be positive")
    return args


def training_plan(alignment):
    rng = np.random.default_rng(0)
    indices = [int(i) for _ in range(8) for i in rng.permutation(40)]
    plan = []
    for step in range(20):
        selected = indices[16 * step:16 * (step + 1)]
        plan.append({
            "stage": "node_dropout" if step < 10 else "edge_dropout",
            "iteration": step % 10, "sample_indices": selected,
            "sample_sha256": [digest(alignment["calibration_samples"][i]) for i in selected],
        })
    return plan


def prepare_context(args):
    alignment = load_alignment(args.reference_run)
    if len(alignment["calibration_samples"]) != 40 or len(alignment["evaluation_samples"]) != 1531:
        raise ValueError("Expected exactly 40 calibration and 1,531 evaluation samples")
    source_names = (
        "AgentDropout/graph/graph.py", "AgentDropout/graph/node.py",
        "AgentDropout/llm/hf_chat.py", "datasets/mmlu_dataset.py",
        "experiments/aligned_train_mmlu.py", "experiments/aligned_evaluate_mmlu.py",
        "experiments/evaluate_mmlu_dev40_sampled.py", "experiments/traced_original_graph.py",
        "experiments/dropout_artifacts.py", "experiments/mmlu_dev40_prompt_adapter.py",
        "experiments/run_mmlu_attrprune_dev40.py",
    )
    config = {
        "schema_version": 1, "benchmark": "mmlu", "method": "AgentDropout",
        "experiment": "mmlu_attrprune_prompt_dev40_t0", "seed": 0,
        "alignment_fingerprint": alignment["fingerprint"],
        "model": str((args.model_path or Path(alignment["reference_run_config"]["model_name"])).expanduser().resolve()),
        "temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False,
        "num_rounds": 2, "workers": alignment["workers"],
        "reviewer_agent_ids": alignment["reviewer_agent_ids"],
        "prompt_version": alignment["prompt_version"],
        "topology": "FullConnected candidate graph", "round_specific_policy": True,
        "node_dropout": True, "edge_dropout": True,
        "training": TRAINING,
        "evaluation": {"split": "val", "dataset_size": 1531, "start_index": 0,
                       "limit": 1531, "batch_size": args.eval_batch_size,
                       "edge_policy": "original Graph.arun optimized=True sigmoid(logits) sampling with per-example local RNG"},
        "llm_batch_size": args.llm_batch_size, "inference_replicas": args.replicas,
        "source_hashes": {str(ROOT / name): file_sha(ROOT / name) for name in source_names},
    }
    config["fingerprint"] = digest(config)
    plan = training_plan(alignment)
    run_dir = args.run_dir.expanduser().resolve()
    expected = {"run_config.json": config, "training_draw_plan.json": plan,
                "prompt_data_alignment.json": alignment}
    for filename, value in expected.items():
        path = run_dir / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    has_outputs = ((run_dir / "training_metrics.jsonl").exists()
                   or (run_dir / "pruning_artifacts").exists()
                   or (run_dir / "evaluation/predictions.jsonl").exists())
    if has_outputs and any(not (run_dir / name).is_file() for name in expected):
        raise ValueError("Existing results are missing their frozen configuration/cohort/plan")
    return {"alignment": alignment, "config": config, "plan": plan, "run_dir": run_dir}


def freeze_context(ctx):
    freeze_alignment(ctx["run_dir"], ctx["alignment"])
    from experiments_attrprune.random_order_mmlu import freeze_json
    freeze_json(ctx["run_dir"] / "run_config.json", ctx["config"])
    freeze_json(ctx["run_dir"] / "training_draw_plan.json", ctx["plan"])


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_training(ctx):
    records = [row for row in _jsonl(ctx["run_dir"] / "training_metrics.jsonl")
               if row.get("stage") in {"node_dropout", "edge_dropout"}]
    if len(records) != 20:
        raise ValueError("Training is incomplete: expected 10 node and 10 edge optimizer steps")
    for record, planned in zip(records, ctx["plan"]):
        for key in ("stage", "iteration", "sample_indices"):
            if record[key] != planned[key]:
                raise ValueError(f"Actual training draw differs from frozen plan: {key}")
        if not np.isfinite(record["loss"]):
            raise ValueError("Nonfinite optimizer loss")
        if record["stage"] == "edge_dropout" and record["pruned_this_iteration"] != ((record["iteration"] + 1) % 5 == 0):
            raise ValueError("Pruning schedule changed")


def _training_completion(ctx, save=False):
    validate_training(ctx)
    run = ctx["run_dir"]
    files = (run / "training_metrics.jsonl",
             run / "pruning_artifacts/learned_dropout_policy.pt",
             run / "pruning_artifacts/learned_dropout_policy.json")
    marker = {"config_fingerprint": ctx["config"]["fingerprint"],
              "sha256": {str(p): file_sha(p) for p in files}}
    path = run / "training_complete.json"
    if save:
        from experiments_attrprune.random_order_mmlu import freeze_json
        freeze_json(path, marker)
    elif not path.is_file() or json.loads(path.read_text()) != marker:
        raise ValueError("Training checkpoint is incomplete or has changed")


def validate_predictions(ctx):
    seen = set()
    dataset = FrozenDataset(ctx["alignment"]["evaluation_samples"], "val")
    for item in _jsonl(ctx["run_dir"] / "evaluation/predictions.jsonl"):
        index = item["sample_index"]
        if not isinstance(index, int) or index in seen or not 0 <= index < 1531:
            raise ValueError("Duplicate or invalid evaluation sample index")
        seen.add(index)
        sample = dataset[index]
        expected = {"question": sample["task"], "gold_answer": sample["answer"], "policy_seed": index}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("Evaluation sample differs from frozen reference")
        answer = dataset.postprocess_answer(item["raw_answer"])
        if item["parsed_answer"] != answer or item["correct"] != (answer == sample["answer"]):
            raise ValueError("Evaluation scoring changed")
        if item.get("sampling_mode") != "original_graph_optimized_true":
            raise ValueError("Evaluation graph sampling algorithm changed")
        traces = item.get("execution_trace", [])
        if [trace.get("round") for trace in traces] != [0, 1]:
            raise ValueError("Missing realized round graphs")
        for trace in traces:
            if trace["failed_indices"] or trace["executed_indices"] != [
                    i for i in trace["scheduled_indices"] if i not in trace["skipped_indices"]]:
                raise ValueError("Incomplete evaluation worker execution")
    return len(seen)


def status(ctx):
    count = validate_predictions(ctx)
    if (ctx["run_dir"] / "training_complete.json").is_file():
        _training_completion(ctx)
    return {"run_dir": str(ctx["run_dir"]), "calibration_questions": 40,
            "node_training_draws": 160, "edge_training_draws": 160,
            "training_complete": (ctx["run_dir"] / "training_complete.json").is_file(),
            "evaluation_completed": count, "evaluation_total": 1531,
            "alignment_fingerprint": ctx["alignment"]["fingerprint"],
            "training_batch_size": 16, "llm_batch_size": ctx["config"]["llm_batch_size"],
            "eval_batch_size": ctx["config"]["evaluation"]["batch_size"]}


@contextmanager
def _run_lock(path):
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already using this experiment directory") from error
        yield


async def run_model_stages(args, ctx):
    _local_dataset_namespace()
    import torch
    from AgentDropout.llm.hf_chat import HFChat
    from experiments.mmlu_dev40_prompt_adapter import build_graph
    from experiments.aligned_train_mmlu import train
    from experiments.evaluate_mmlu_dev40_sampled import evaluate
    from experiments.dropout_artifacts import restore_checkpoint, save_artifacts

    run = ctx["run_dir"]
    complete = (run / "training_complete.json").is_file()
    if args.stage == "eval" and not complete:
        raise ValueError("Train this experiment's new policy before evaluation")
    if complete:
        _training_completion(ctx)
        if not args.resume:
            raise FileExistsError("This run already has a learned policy; use --resume")
    elif (run / "training_metrics.jsonl").exists():
        raise ValueError("Partial training has no optimizer checkpoint; use a new run directory, do not mix draws")
    completed_predictions = validate_predictions(ctx)
    if complete and (args.stage == "train" or completed_predictions == 1531):
        return
    os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
    os.environ["HF_CHAT_NUM_REPLICAS"] = str(args.replicas)
    os.environ["HF_CHAT_BATCH_SIZE"] = str(args.llm_batch_size)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    HFChat.DEFAULT_MAX_TOKENS = 2048
    HFChat.DEFAULT_TEMPERATURE = 0.0
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    graph = build_graph(ctx["alignment"], ctx["config"]["model"])
    checkpoint = run / "pruning_artifacts/learned_dropout_policy.pt"
    if complete:
        saved = restore_checkpoint(graph, checkpoint)
        if [(w["id"], w["role"]) for w in saved["workers"]] != [
                (w["agent_id"], w["role"]) for w in ctx["alignment"]["workers"]]:
            raise ValueError("Saved policy worker identities differ from the frozen reference")
    else:
        await train(graph, FrozenDataset(ctx["alignment"]["calibration_samples"], "dev"),
                    num_iters=10, num_rounds=2, lr=0.1, batch_size=16, prune_every=5,
                    pruning_rate=0.25, delta=0.1, seed=0,
                    metrics_path=run / "training_metrics.jsonl")
        validate_training(ctx)
        save_artifacts(graph, run / "pruning_artifacts", ctx["config"])
        _training_completion(ctx, save=True)
    if args.stage in {"eval", "all"}:
        await evaluate(graph, FrozenDataset(ctx["alignment"]["evaluation_samples"], "val"),
                       num_rounds=2, output_dir=run / "evaluation", start_index=0, limit=1531,
                       batch_size=args.eval_batch_size, seed=0, resume=args.resume,
                       run_config=ctx["config"])
        if validate_predictions(ctx) != 1531:
            raise ValueError("Evaluation is incomplete")


def main(argv=None):
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run or args.stage == "summarize":
        print(json.dumps({**status(ctx), "dry_run": args.dry_run}, indent=2))
        return
    with _run_lock(ctx["run_dir"]):
        freeze_context(ctx)
        if args.stage != "prepare":
            asyncio.run(run_model_stages(args, ctx))
    print(json.dumps(status(ctx), indent=2))


if __name__ == "__main__":
    main()
