from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_mmlu_reordered_pruning.sh"


def launch(tmp_path, *args, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "SOURCE_RUN", "RUN_DIR", "VAL_LIMIT", "EVAL_BATCH_SIZE", "ATTR_BATCH_SIZE"):
        env.pop(name, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env,
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("branch,stages", [
    ("direct", ["prepare", "visualize", "eval", "eval", "eval", "summarize"]),
    ("recalibrate", ["prepare", "collect", "attribute", "aggregate", "visualize", "eval", "eval", "eval", "summarize"]),
])
def test_separate_process_stages_preserve_full_sweep_and_requested_execution_order(tmp_path, branch, stages):
    result = launch(tmp_path, branch)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert [c[c.index("--stage") + 1] for c in commands] == stages
    assert [c[c.index("--pruning-percent") + 1] for c in commands if "--pruning-percent" in c] == ["80", "60", "50"]
    for command in commands:
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.reordered_mmlu"]
        assert command[command.index("--percentages") + 1:command.index("--percentages") + 4] == ["80", "60", "50"]
        assert command[command.index("--eval-batch-size") + 1] == "4"
        assert command[command.index("--limit") + 1] == "1531"
        assert "--resume" in command and "--dry-run" in command
    assert not (tmp_path / "logs").exists()


def test_prepare_works_without_gpu_and_preserves_literal_paths(tmp_path):
    output = str(tmp_path / "literal $value")
    result = launch(tmp_path, "direct", "prepare", dry=False, RUN_DIR=output)
    assert result.returncode == 0, result.stderr
    args = json.loads(next(line[5:] for line in result.stdout.splitlines() if line.startswith("SHIM ")))
    assert args[args.index("--run-dir") + 1] == output
    assert args[args.index("--stage") + 1] == "prepare"
    logs = list((tmp_path / "logs/direct").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


@pytest.mark.parametrize("args", [(), ("bad",), ("direct", "collect"), ("direct", "attribute"),
    ("direct", "aggregate"), ("direct", "bad"), ("recalibrate", "50", "50"), ("direct", "55")])
def test_invalid_arguments_cannot_start_inference(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="4,5")
    assert result.returncode == 2
    assert "SHIM " not in result.stdout


def test_gpu_stages_require_explicit_devices(tmp_path):
    result = launch(tmp_path, "recalibrate", dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


def test_first_failure_stops_following_stages(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, "recalibrate", dry=False, CUDA_VISIBLE_DEVICES="6,7", PYTHON_BIN=str(failure))
    assert result.returncode == 7
    assert "exit_status=7" in result.stdout
    assert "stage=collect" not in result.stdout


def test_eval_only_runs_each_rate_in_its_own_process(tmp_path):
    result = launch(tmp_path, "direct", "eval", "60", "50")
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    assert [c[c.index("--pruning-percent") + 1] for c in commands] == ["60", "50"]
