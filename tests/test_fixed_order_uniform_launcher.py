"""The fixed-order uniform launcher changes allocation, not evaluation protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_mmlu_fixed_order_uniform.sh"


def launch(tmp_path, *arguments, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "PREPARE_ONLY", "RUN_DIR"):
        env.pop(key, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"),
               RUN_DIR=str(tmp_path / "original run"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, text=True, capture_output=True)


def test_default_commands_preserve_baseline_and_only_change_allocation(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 5
    prepare, *evaluations, summarize = commands
    assert prepare[1:5] == ["-u", "-m", "experiments_attrprune.static_pruning", "prepare"]
    assert prepare[prepare.index("--pruning-scope") + 1] == "per_round_edge_type"
    assert prepare[prepare.index("--percentages") + 1:prepare.index("--metric")] == ["50", "60", "80"]
    assert prepare[prepare.index("--metric") + 1] == "token_ratio"
    assert prepare[prepare.index("--statistic") + 1] == "conditional_mean"
    assert prepare[prepare.index("--unobserved") + 1] == "keep"
    for command, rate in zip(evaluations, (50, 60, 80)):
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.evaluate_mmlu"]
        assert command[command.index("--evaluation-name") + 1] == f"static_token_ratio_conditional_mean_prune_{rate}pct"
        assert command[command.index("--policy-json") + 1].endswith(
            f"/attribution/static_pruning/token_ratio_conditional_mean/policies/prune_{rate}pct.json")
        for key, expected in (("--benchmark", "mmlu"), ("--mode", "token_ratio"),
                              ("--eval-batch-size", "4"), ("--start-index", "0"),
                              ("--limit", "1531"), ("--seed", "0"), ("--split", "val")):
            assert command[command.index(key) + 1] == expected
        assert {"--resume", "--save-traces", "--no-download"}.issubset(command)
        assert "sampled" not in " ".join(command) and "skip_empty" not in " ".join(command)
    assert summarize[1:5] == ["-u", "-m", "experiments_attrprune.static_pruning", "summarize"]
    assert not (tmp_path / "logs").exists()


def test_cpu_prepare_needs_no_gpu_and_does_not_dispatch_evaluation(tmp_path):
    source = str(tmp_path / "source $literal `name`")
    result = launch(tmp_path, dry=False, PREPARE_ONLY="1", RUN_DIR=source)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert commands[0] == ["-", source]
    assert len(commands) == 2 and commands[1][2:4] == ["experiments_attrprune.static_pruning", "prepare"]
    assert commands[1][commands[1].index("--run-dir") + 1] == source
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


def test_existing_validated_sweep_is_reused_without_prepare_rewrite(tmp_path):
    source = tmp_path / "original run"
    manifest = source / "attribution/static_pruning/token_ratio_conditional_mean/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"preserve": "exact bytes"}\n')
    original = manifest.read_bytes()
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="6,7")
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert commands[0] == ["-", str(source)]
    assert len(commands) == 5
    assert not any("prepare" in command for command in commands)
    assert manifest.read_bytes() == original


def test_gpu_execution_requires_explicit_devices(tmp_path):
    result = launch(tmp_path, dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


@pytest.mark.parametrize("args", [("50",), ("global",), ("uniform",), ("prepare",)])
def test_unsupported_arguments_cannot_dispatch(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="6,7")
    assert result.returncode == 2 and "SHIM " not in result.stdout


def test_preflight_failure_blocks_all_policies_and_evaluations(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="6,7", PYTHON_BIN=str(failure))
    assert result.returncode == 7 and "exit_status=7" in result.stdout
    assert not (tmp_path / "original run").exists()


def test_first_evaluation_failure_blocks_later_rates(tmp_path):
    failure = tmp_path / "fail first evaluation"
    failure.write_text(f"#!{sys.executable}\nimport json,sys\n"
                      "print('SHIM '+json.dumps(sys.argv[1:]))\n"
                      "sys.exit(9 if 'experiments_attrprune.evaluate_mmlu' in sys.argv else 0)\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="6,7", PYTHON_BIN=str(failure))
    assert result.returncode == 9 and "exit_status=9" in result.stdout
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    evaluations = [c for c in commands if "experiments_attrprune.evaluate_mmlu" in c]
    assert len(evaluations) == 1
    assert evaluations[0][evaluations[0].index("--evaluation-name") + 1].endswith("50pct")
