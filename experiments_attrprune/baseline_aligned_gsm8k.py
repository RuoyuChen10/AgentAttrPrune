"""Fixed-order GSM8K calibration on the exact train40 shared by three baselines.

Reuses the existing attribution and static evaluation implementation. Only the
calibration cohort changes; the original dataset/test ordering stays intact.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np

from AgentAttrPrune.datasets import GSM8KDataset
from AgentAttrPrune.prompt import GSM8KPromptSet
from experiments_attrprune import resampled_gsm8k as core
from experiments_attrprune.common import (
    ExperimentPaths, benchmark_data_identity, config_fingerprint,
    dataset_identities_match, load_json,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DATA = REPOSITORY.parent / "AgentPrune/dataset/gsm8k/train.jsonl"
DEFAULT_AUDIT = REPOSITORY / "results/AgentAttrPrune/analysis/seed0_calibration_cohort_audit_20261103T162335Z"
DEFAULT_RUN = core.DEFAULT_REFERENCE.with_name("train40_baseline_aligned_correctness_aware")
PERCENTAGES = (80, 70, 60)
COHORT_TYPE = "baseline_source_order_train40"
SELECTION_RULE = "original source-order first 40 train questions; exact AgentPrune/AgentDropout/SafeSieve cohort; no resampling"


def _pair(question: str, answer: str) -> tuple[str, str]:
    return str(question), GSM8KPromptSet.normalize_answer(str(answer))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_baseline_cohort(baseline_data: Path, audit_dir: Path) -> tuple[list[dict], dict, dict[str, str]]:
    """Cross-check saved baseline evidence, rather than infer a cohort from seed 0."""
    baseline_data, audit_dir = baseline_data.resolve(), audit_dir.resolve()
    records = _jsonl(baseline_data)[:core.NUM_QUESTIONS]
    if len(records) != core.NUM_QUESTIONS:
        raise ValueError("Baseline source has fewer than 40 training questions")
    samples = []
    for row in records:
        _, gold = GSM8KDataset._split_answer(row["answer"])
        samples.append({"task": row["question"], "benchmark": "gsm8k",
                        "question": row["question"], "answer": gold})
    expected = [_pair(s["question"], s["answer"]) for s in samples]
    if len({q for q, _ in expected}) != core.NUM_QUESTIONS:
        raise ValueError("Duplicate question in baseline calibration cohort")
    evidence_paths = [baseline_data, audit_dir / "audit.json"]
    audit = load_json(audit_dir / "audit.json")
    exports = {}
    for name in ("agentprune", "agentdropout", "safesieve"):
        path = audit_dir / f"{name}_cohorts.json"
        if core.file_sha(path) != audit["source_exports_sha256"][name]:
            raise ValueError(f"Baseline cohort audit changed: {name}")
        evidence_paths.append(path)
        exports[name] = load_json(path)
    ap = exports["agentprune"]["gsm8k_original"]
    ad = exports["agentdropout"]["runs"]["gsm8k_pruned"]
    ss = [r for r in exports["safesieve"]["runs"] if r["benchmark"] == "gsm8k"]
    if len(ss) != 1:
        raise ValueError("Expected one audited GSM8K SafeSieve run")
    ss = ss[0]
    for name, run in (("AgentPrune", ap), ("AgentDropout", ad), ("SafeSieve", ss)):
        actual = [_pair(s["question"], s["answer"]) for s in run["samples"]]
        if actual != expected:
            raise ValueError(f"Exact question/answer order differs from {name}")

    # Check the original saved evidence as well as the portable audit exports.
    ap_path = Path(ap["source"])
    ap_rows = load_json(ap_path)
    counts = Counter()
    seen = set()
    for row in ap_rows:
        index, repeat = int(row["query_index"]), int(row["sample_in_query"])
        if not 0 <= index < core.NUM_QUESTIONS or not 0 <= repeat < 10:
            raise ValueError("AgentPrune training cohort has an unexpected query/repeat")
        if (index, repeat) in seen or _pair(row["Question"]["task"], row["Question"]["answer"]) != expected[index]:
            raise ValueError("AgentPrune saved training questions are duplicated or mismatched")
        seen.add((index, repeat)); counts[index] += 1
    if counts != Counter({i: 10 for i in range(core.NUM_QUESTIONS)}):
        raise ValueError("AgentPrune evidence must contain 40 questions, each repeated 10 times")

    ad_config_path = Path(ad["run_dir"]) / "run_config.json"
    ad_metrics_path = Path(ad["run_dir"]) / "training_metrics.jsonl"
    ad_config = load_json(ad_config_path)
    question_hashes = [hashlib.sha256(q.encode()).hexdigest() for q, _ in expected]
    if (ad_config["training"]["selected_question_sha256"] != question_hashes
            or ad_config["datasets"]["train"]["sha256"] != core.file_sha(baseline_data)
            or core.file_sha(ad_metrics_path) != ad["metrics_sha256"]):
        raise ValueError("AgentDropout saved calibration evidence changed")
    ss_config_path = Path(ss["run_config"])
    ss_path = Path(ss["run_dir"]) / "calibration.jsonl"
    ss_rows = _jsonl(ss_path)
    if ([_pair(r["task"], r["target"]) for r in ss_rows] != expected
            or [r["index"] for r in ss_rows] != list(range(core.NUM_QUESTIONS))
            or ss["source_sha256"] != core.file_sha(baseline_data)):
        raise ValueError("SafeSieve saved calibration questions changed")
    evidence_paths += [ap_path, ad_config_path, ad_metrics_path, ss_config_path, ss_path]
    hashes = {str(p.resolve()): core.file_sha(p) for p in evidence_paths}
    provenance = {
        "cohort_type": COHORT_TYPE, "selection_rule": SELECTION_RULE,
        "source_jsonl": str(baseline_data), "source_sha256": core.file_sha(baseline_data),
        "source_indices": list(range(core.NUM_QUESTIONS)),
        "ordered_question_answer_sha256": core.digest(expected),
        "audit_dir": str(audit_dir), "baseline_artifact_sha256": hashes,
        "verified_baselines": {
            "AgentPrune": {"training_records": str(ap_path), "questions": core.NUM_QUESTIONS, "rollouts": len(ap_rows)},
            "AgentDropout": {"run_dir": ad["run_dir"], "questions": core.NUM_QUESTIONS},
            "SafeSieve": {"calibration_records": str(ss_path), "questions": core.NUM_QUESTIONS},
        },
    }
    return samples, provenance, hashes


def prepare_context(args: argparse.Namespace) -> core.ResampledContext:
    reference = args.reference_run.resolve()
    base, _, attr_settings, reference_hashes = core._reference(reference)
    identity = base["dataset"]
    data_dir = Path(identity.get("requested_data_dir") or identity["data_root"])
    actual = benchmark_data_identity("gsm8k", data_dir, splits=identity["splits"], shuffle_seed=identity["shuffle_seed"])
    if not dataset_identities_match(identity, actual):
        raise ValueError("GSM8K dataset fingerprint changed")
    dataset = GSM8KDataset(split="train", data_dir=data_dir, download=False, shuffle_seed=identity["shuffle_seed"])
    selected, provenance, evidence_hashes = load_baseline_cohort(args.baseline_data, args.cohort_audit_dir)
    if len(selected) != core.NUM_QUESTIONS or len({s["question"] for s in selected}) != core.NUM_QUESTIONS:
        raise ValueError("Expected exactly 40 distinct baseline questions")
    positions: dict[tuple[str, str], list[int]] = {}
    for j, row in enumerate(dataset):
        positions.setdefault(_pair(row["question"], row["correct_answer"]), []).append(j)
    indices = []
    for sample in selected:
        matches = positions.get(_pair(sample["question"], sample["answer"]), [])
        if len(matches) != 1:
            raise ValueError("Baseline question is missing or ambiguous in the AttrPrune dataset")
        indices.append(matches[0])
    samples = {i: dataset.record_to_sample(dataset[j]) for i, j in enumerate(indices)}
    if list(samples.values()) != selected:
        raise ValueError("Mapped calibration question content differs from the baseline cohort")
    hashes = {**reference_hashes, **evidence_hashes}
    agents = [w["agent_id"] for w in base["workers"]]
    orders = [list(agents) for _ in range(core.NUM_ROUNDS)]
    spatial = np.broadcast_to(np.triu(np.ones((len(agents), len(agents)), dtype=int), 1),
                              (core.NUM_ROUNDS, len(agents), len(agents))).copy()
    temporal = np.zeros_like(spatial); temporal[1:] = 1
    order_plan = {"schema_version": 1, "agents": agents, "round_orders": orders,
                  "order_rule": "original canonical fixed order",
                  "candidate_masks": {"spatial": spatial.tolist(), "temporal": temporal.tolist()}}
    plan = {
        "schema_version": 1, "cohort_type": COHORT_TYPE, "calibration_seed": 0,
        "selection_seed": None, "sampling_rule": SELECTION_RULE,
        "baseline_cohort": provenance, "dataset_size": len(dataset), "dataset_identity": identity,
        "selected_source_indices": list(range(core.NUM_QUESTIONS)), "selected_dataset_indices": indices,
        "agents": agents, "num_rounds": core.NUM_ROUNDS, "num_questions": core.NUM_QUESTIONS,
        "repeats_per_question": 1, "num_rollouts": core.NUM_QUESTIONS,
        "generation_rng_rule": "reference generation seed + mapped dataset_index; temperature=0",
        "rollouts": [{"rollout_id": f"sample_{i:05d}", "sample_index": i, "source_index": i,
                      "dataset_index": j, "sample_sha256": core.digest(samples[i]), "round_orders": orders}
                     for i, j in enumerate(indices)],
        "questions": [{"sample_index": i, "source_index": i, "dataset_index": j,
                       "sample": samples[i], "sample_sha256": core.digest(samples[i])}
                      for i, j in enumerate(indices)],
    }
    output = args.run_dir.resolve()
    if output == reference or reference in output.parents or output in reference.parents:
        raise ValueError("Use an independent output directory outside the reference")
    config = copy.deepcopy(base)
    # Compatibility key for the existing checkpoint/evaluation implementation;
    # cohort_type and selection_seed explicitly distinguish this deterministic cohort.
    config["resampled_experiment"] = {
        "version": 1, "cohort_type": COHORT_TYPE, "calibration_seed": 0, "selection_seed": None,
        "reference_run": str(reference), "reference_run_fingerprint": base["fingerprint"],
        "reference_artifact_sha256": reference_hashes, "baseline_cohort": provenance,
        "sampling_rule": SELECTION_RULE, "selected_dataset_indices": indices,
        "calibration_plan_sha256": core.digest(plan), "order_plan_sha256": core.digest(order_plan),
        "num_questions": core.NUM_QUESTIONS, "generation_seed": base["seed"], "generation_batch_size": 1,
        "attribution_configuration": copy.deepcopy(attr_settings), "attribution_batch_size": attr_settings["batch_size"],
        "eval_batch_size": args.eval_batch_size, "evaluation_limit": args.limit, "save_traces": True,
        "percentages": list(PERCENTAGES), "pruning_scope": "global", "sampling": "none; execute all retained edges",
        "empty_round_rule": "skip Round1 iff both spatial and temporal masks are empty; reviewer reads Round0",
        "calibration_source_description": "all new baseline-aligned source-order train40 rollouts, including final-incorrect",
    }
    config["fingerprint"] = config_fingerprint(config)
    ctx = core.ResampledContext(ExperimentPaths(output), config, plan, samples, orders, order_plan,
        base, hashes, 0, reference, args.resume, PERCENTAGES, args.eval_batch_size, args.limit)
    core._validate_frozen(ctx)
    core._check_hashes(hashes)
    return ctx


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=core.DEFAULT_REFERENCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-data", type=Path, default=DEFAULT_BASELINE_DATA)
    parser.add_argument("--cohort-audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--stage", choices=core.STAGES, default="all")
    parser.add_argument("--pruning-percent", type=int, choices=PERCENTAGES)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.eval_batch_size, args.limit) < 1:
        parser.error("Batch size and evaluation limit must be positive")
    if args.pruning_percent is not None and args.stage != "eval":
        parser.error("--pruning-percent requires --stage eval")
    return args


def _child_args(args: argparse.Namespace, stage: str, rate: int | None = None) -> list[str]:
    command = [sys.executable, "-u", "-m", "experiments_attrprune.baseline_aligned_gsm8k", "--stage", stage]
    for name in ("reference_run", "run_dir", "baseline_data", "cohort_audit_dir", "eval_batch_size", "limit"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.resume:
        command.append("--resume")
    if rate is not None:
        command.extend(["--pruning-percent", str(rate)])
    return command


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ctx = prepare_context(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(ctx.paths.run_dir), "cohort_type": COHORT_TYPE,
            "baseline_cohort": ctx.plan["baseline_cohort"], "selected_dataset_indices": ctx.plan["selected_dataset_indices"],
            "fingerprint": ctx.config["fingerprint"], "round_orders": ctx.round_orders,
            "percentages": list(PERCENTAGES), "eval_batch_size": ctx.eval_batch_size, "limit": ctx.limit,
            "attribution_batch_size": core.attribution_config(ctx).batch_size,
            "generation_batch_size": 1, "generation_seed": ctx.config["seed"], "decoding": ctx.config["decoding"]}, indent=2))
        return
    if args.stage in {"collect", "attribute", "eval", "all"} and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("Select CUDA_VISIBLE_DEVICES explicitly for GPU stages")
    if args.stage == "all":
        for stage in ("prepare", "collect", "attribute", "aggregate"):
            subprocess.run(_child_args(args, stage), check=True)
        for rate in PERCENTAGES:
            subprocess.run(_child_args(args, "eval", rate), check=True)
        subprocess.run(_child_args(args, "summarize"), check=True)
        return
    if args.stage == "eval" and args.pruning_percent is None:
        for rate in PERCENTAGES:
            subprocess.run(_child_args(args, "eval", rate), check=True)
        return
    if args.stage == "eval":
        from experiments_attrprune.resampled_gsm8k_evaluation import evaluate_policy
        result = asyncio.run(evaluate_policy(ctx, pruning_percent=args.pruning_percent, resume=args.resume,
            eval_batch_size=ctx.eval_batch_size, limit=ctx.limit))
    else:
        with core.run_lock(ctx.paths.run_dir):
            if args.stage == "prepare": result = core.prepare_run(ctx)
            elif args.stage == "collect": result = asyncio.run(core.collect_run(ctx))
            elif args.stage == "attribute": result = core.attribute_run(ctx)
            elif args.stage == "aggregate": result = core.aggregate_run(ctx)
            elif args.stage == "policies": result = core.prepare_policies(ctx)
            else: result = core.summarize_run(ctx)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
