"""Train AgentDropout on AgentAttrPrune's exact original seed0 GSM8K train40.

Keep original GSM8K node 2x20 + edge 4x10 training, optimizer, 10% pruning
at edge updates 2 and 4, and original graph sampling. Align only the question
pool, prompt/identity contract, temperature and model microbatch size.
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

from experiments_attrprune.baseline_gsm8k_alignment import (
    DEFAULT_REFERENCE, digest, file_sha, freeze_alignment, load_alignment, verify_sources,
)

DEFAULT_RUN = ROOT / "results/AgentDropout/gsm8k/qwen3_8b/gsm8k_attrprune_seed0_train40_t0"
DEFAULT_MODEL = None  # Resolve from the validated AgentAttrPrune reference configuration.
TRAINING = {
    "split": "train", "dataset_size": 40, "node_iterations": 2, "node_batch_size": 20,
    "edge_iterations": 4, "edge_batch_size": 10, "learning_rate": 0.1,
    "prune_every_iterations": 2, "pruning_rate": 0.1, "delta": 0.1,
    "total_draws": 80, "node_draws": 40, "edge_draws": 40,
    "selection": "exact original AgentAttrPrune seed0 train40 in saved order, reused once per stage",
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
        value = float(record["answer"])
        return str(0.0 if value == 0 else value)

    def postprocess_answer(self, answer):
        # Retain the released GSM8K parser/reward, not the attribution parser.
        _local_dataset_namespace()
        from experiments.run_gsm8k_aligned import GSM8KDataset
        return GSM8KDataset.postprocess_answer(answer)


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
    plan = []
    for stage, updates, batch in (("node_dropout", 2, 20), ("edge_dropout", 4, 10)):
        for step in range(updates):
            selected = list(range(step * batch, (step + 1) * batch))
            plan.append({"stage": stage, "iteration": step, "sample_indices": selected,
                         "sample_sha256": [digest(alignment["calibration_samples"][i]) for i in selected]})
    return plan


def prepare_context(args):
    alignment = load_alignment(args.reference_run)
    if len(alignment["calibration_samples"]) != 40 or len(alignment["evaluation_samples"]) != 1319:
        raise ValueError("Expected exactly 40 calibration and 1,319 evaluation samples")
    source_names = (
        "AgentDropout/graph/graph.py", "AgentDropout/graph/node.py",
        "AgentDropout/llm/hf_chat.py", "datasets/gsm8k_dataset.py",
        "experiments/run_gsm8k_aligned.py", "experiments/aligned_train_mmlu.py",
        "experiments/aligned_evaluate_mmlu.py", "experiments/evaluate_mmlu_dev40_sampled.py",
        "experiments/evaluate_gsm8k_seed0_sampled.py", "experiments/traced_original_graph.py",
        "experiments/dropout_artifacts.py", "experiments/mmlu_dev40_prompt_adapter.py",
        "experiments/gsm8k_seed0_prompt_adapter.py", "experiments/run_gsm8k_attrprune_seed0.py",
    )
    config = {
        "schema_version": 1, "benchmark": "gsm8k", "method": "AgentDropout",
        "experiment": "gsm8k_attrprune_seed0_train40_t0", "seed": 0,
        "alignment_fingerprint": alignment["fingerprint"],
        "model": str((args.model_path or Path(alignment["reference_run_config"]["model_name"])).expanduser().resolve()),
        "temperature": 0.0, "max_new_tokens": 2048, "enable_thinking": False,
        "num_rounds": 2, "workers": alignment["workers"],
        "reviewer_agent_ids": alignment["reviewer_agent_ids"],
        "prompt_version": alignment["prompt_version"],
        "message_alignment": {
            "worker_and_reviewer_templates": "exact AgentAttrPrune GSM8KPromptSet builders",
            "worker_output": "unmodified model-generated text, matching AgentAttrPrune",
            "legacy_programming_expert_code_execution": False,
            "legacy_code_execution_change_reason": (
                "AgentAttrPrune's shared prompt requests a concise numeric answer rather than executable code; "
                "omit the original MathSolver Programming Expert code-execution postprocess instead of "
                "appending execution output to the aligned plain-text response"
            ),
            "dropout_sentinel": "preserve original GSM8K None. messages",
            "pruning_optimizer_and_graph_sampling": "unchanged original GSM8K protocol",
        },
        "topology": "FullConnected candidate graph", "round_specific_policy": True,
        "node_dropout": True, "edge_dropout": True,
        "training": TRAINING,
        "evaluation": {"split": "test", "dataset_size": 1319, "start_index": 0,
                       "limit": 1319, "batch_size": args.eval_batch_size,
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
    # Physical JSONL newlines only: Unicode U+0085/U+2028 are legal in strings.
    with path.open(encoding="utf-8", newline="\n") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_training(ctx):
    records = [row for row in _jsonl(ctx["run_dir"] / "training_metrics.jsonl")
               if row.get("stage") in {"node_dropout", "edge_dropout"}]
    if len(records) != 6:
        raise ValueError("Training is incomplete: expected 2 node and 4 edge optimizer steps")
    for record, planned in zip(records, ctx["plan"]):
        for key in ("stage", "iteration", "sample_indices"):
            if record[key] != planned[key]:
                raise ValueError(f"Actual training draw differs from frozen plan: {key}")
        if not np.isfinite(record["loss"]):
            raise ValueError("Nonfinite optimizer loss")
        if record["stage"] == "edge_dropout" and record["pruned_this_iteration"] != ((record["iteration"] + 1) % 2 == 0):
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
    records = _jsonl(ctx["run_dir"] / "evaluation/predictions.jsonl")
    checkpoint = ctx["run_dir"] / "pruning_artifacts/learned_dropout_policy.pt"
    expected_policy = ({"checkpoint_sha256": file_sha(checkpoint),
                        "config_fingerprint": ctx["config"]["fingerprint"]} if records else None)
    if records:
        policy_path = ctx["run_dir"] / "evaluation/policy.json"
        if not policy_path.is_file() or json.loads(policy_path.read_text()) != expected_policy:
            raise ValueError("Evaluation checkpoint identity changed or is missing")
    dataset = FrozenDataset(ctx["alignment"]["evaluation_samples"], "test")
    for item in records:
        index = item["sample_index"]
        if type(index) is not int or index in seen or not 0 <= index < 1319:
            raise ValueError("Duplicate or invalid evaluation sample index")
        seen.add(index)
        if any(item.get(key) != value for key, value in expected_policy.items()):
            raise ValueError("Prediction belongs to another checkpoint/configuration")
        sample = dataset[index]
        expected = {"question": sample["task"], "gold_answer": dataset.record_to_target_answer(sample), "policy_seed": index}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("Evaluation sample differs from frozen reference")
        answer = dataset.postprocess_answer(item["raw_answer"])
        if item["parsed_answer"] != answer or item["correct"] != (answer == dataset.record_to_target_answer(sample)):
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
    summary_path = ctx["run_dir"] / "evaluation/summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        correct = sum(bool(item["correct"]) for item in records)
        # Partial CPU test evaluations may publish a limited summary; only a
        # full 1319-question summary is a completed experiment.
        if (summary.get("run_config") != ctx["config"] or summary.get("benchmark") != "gsm8k"
                or summary.get("num_evaluated") != len(records)
                or summary.get("num_correct") != correct
                or summary.get("accuracy") != (correct / len(records) if records else 0.0)):
            raise ValueError("Published evaluation summary differs from saved predictions")
    return len(seen)


def status(ctx):
    count = validate_predictions(ctx)
    if (ctx["run_dir"] / "training_complete.json").is_file():
        _training_completion(ctx)
    return {"run_dir": str(ctx["run_dir"]), "calibration_questions": 40,
            "node_training_draws": 40, "edge_training_draws": 40,
            "training_complete": (ctx["run_dir"] / "training_complete.json").is_file(),
            "evaluation_completed": count, "evaluation_total": 1319,
            "alignment_fingerprint": ctx["alignment"]["fingerprint"],
            "node_training_batch_size": 20, "edge_training_batch_size": 10, "llm_batch_size": ctx["config"]["llm_batch_size"],
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
    from experiments.gsm8k_seed0_prompt_adapter import build_graph
    from experiments.run_gsm8k_aligned import train
    from experiments.evaluate_gsm8k_seed0_sampled import evaluate
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
    if complete and (args.stage == "train" or completed_predictions == 1319):
        return
    os.environ["HF_CHAT_ENABLE_THINKING"] = "0"
    os.environ["HF_CHAT_NUM_REPLICAS"] = str(args.replicas)
    os.environ["HF_CHAT_BATCH_SIZE"] = str(args.llm_batch_size)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_CHAT_TOP_P"] = "1"
    os.environ["HF_CHAT_TOP_K"] = "0"
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
        from types import SimpleNamespace
        original_args = SimpleNamespace(node_iters=2, node_batch_size=20, edge_iters=4,
            edge_batch_size=10, num_rounds=2, learning_rate=0.1, prune_every=2,
            pruning_rate=0.1, delta=0.1, seed=0)
        await train(graph, FrozenDataset(ctx["alignment"]["calibration_samples"], "train"),
                    original_args, run / "training_metrics.jsonl")
        validate_training(ctx)
        save_artifacts(graph, run / "pruning_artifacts", ctx["config"])
        _training_completion(ctx, save=True)
    if args.stage in {"eval", "all"}:
        await evaluate(graph, FrozenDataset(ctx["alignment"]["evaluation_samples"], "test"),
                       num_rounds=2, output_dir=run / "evaluation", start_index=0, limit=1319,
                       batch_size=args.eval_batch_size, seed=0, resume=args.resume,
                       run_config=ctx["config"])
        if validate_predictions(ctx) != 1319:
            raise ValueError("Evaluation is incomplete")


def main(argv=None):
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run or args.stage == "summarize":
        print(json.dumps({**status(ctx), "dry_run": args.dry_run}, indent=2))
        return
    if args.stage in {"all", "train", "eval"} and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly before model stages")
    with _run_lock(ctx["run_dir"]):
        freeze_context(ctx)
        if args.stage != "prepare":
            asyncio.run(run_model_stages(args, ctx))
    verify_sources(ctx["alignment"])
    prepare_context(args)
    print(json.dumps(status(ctx), indent=2))


if __name__ == "__main__":
    main()
