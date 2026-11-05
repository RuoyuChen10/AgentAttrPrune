#!/usr/bin/env python3
"""Run the minimal, AttrPrune-aligned baselines in isolated Python processes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = {
    "agentprune": {
        "mmlu": "experiments.run_mmlu_attrprune_aligned",
        "gsm8k": "experiments.run_gsm8k_attrprune_seed0",
    },
    "agentdropout": {
        "mmlu": "experiments.run_mmlu_attrprune_dev40",
        "gsm8k": "experiments.run_gsm8k_attrprune_seed0",
    },
    "safesieve": {
        "mmlu": "experiments.run_mmlu_attrprune_aligned",
        "gsm8k": "experiments.run_gsm8k_attrprune_seed0",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=(*METHODS, "all"))
    parser.add_argument("benchmark", choices=("mmlu", "gsm8k"))
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="all")
    parser.add_argument("--attrprune-root", type=Path,
                        default=Path(os.environ.get("BASELINES_ATTRPRUNE_ROOT", ROOT.parent)))
    parser.add_argument("--reference-run", type=Path,
                        help="Frozen AgentAttrPrune reference; defaults to seed0 dev40/train40.")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-root", type=Path, default=ROOT / "results",
                             help="Results go under METHOD/BENCHMARK/RUN_NAME here.")
    destination.add_argument("--run-dir", type=Path, help="Exact output directory for a single method.")
    parser.add_argument("--run-name", default="aligned_seed0_t0_batch4")
    parser.add_argument("--replicas", type=int, default=2,
                        help="AgentDropout inference replicas (default: 2); ignored for other methods.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate actual inputs and protocols without loading models or writing results.")
    parser.add_argument("--print-command", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(argv)
    if args.method == "all" and args.run_dir is not None:
        parser.error("--run-dir needs a single method; use --output-root with all")
    if not args.run_name or args.run_name in {".", ".."} or Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a single directory name")
    if args.replicas < 1:
        parser.error("--replicas must be positive")
    args.attrprune_root = args.attrprune_root.expanduser().resolve()
    if not (args.attrprune_root / "experiments_attrprune/baseline_mmlu_alignment.py").is_file():
        parser.error(f"Missing shared alignment code in {args.attrprune_root}; set --attrprune-root")
    if not (args.attrprune_root / "experiments_attrprune/baseline_gsm8k_alignment.py").is_file():
        parser.error(f"Missing GSM8K alignment code in {args.attrprune_root}")
    if not args.dry_run and not args.print_command and args.stage != "prepare":
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            parser.error("Set CUDA_VISIBLE_DEVICES for train/eval/all, or use --dry-run / --stage prepare")
    return args


def command_for(args, method):
    run_dir = (args.run_dir if args.run_dir is not None else
               args.output_root / method / args.benchmark / args.run_name).expanduser().resolve()
    command = [sys.executable, "-u", "-m", METHODS[method][args.benchmark],
               "--stage", args.stage, "--run-dir", str(run_dir)]
    if args.reference_run is not None:
        command += ["--reference-run", str(args.reference_run.expanduser().resolve())]
    if method == "agentdropout":
        command += ["--eval-batch-size", "4", "--llm-batch-size", "4", "--replicas", str(args.replicas)]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    environment = dict(os.environ)
    environment.update(
        BASELINES_ATTRPRUNE_ROOT=str(args.attrprune_root),
        PYTHONPATH=os.pathsep.join((str(ROOT / method), str(args.attrprune_root))),
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false", HF_CHAT_ENABLE_THINKING="0", HF_CHAT_CAPTURE_SCORES="0",
    )
    if args.dry_run or args.stage == "prepare":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return command, environment


def main(argv=None):
    args = parse_args(argv)
    methods = tuple(METHODS) if args.method == "all" else (args.method,)
    for method in methods:
        command, environment = command_for(args, method)
        print(f"[{method}/{args.benchmark}] cwd={ROOT / method}", flush=True)
        print(shlex.join(command), flush=True)
        if args.print_command:
            continue
        result = subprocess.run(command, cwd=ROOT / method, env=environment)
        if result.returncode:
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
