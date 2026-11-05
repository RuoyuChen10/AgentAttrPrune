"""The 80% launcher runs two independent scopes sequentially at batch 4."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_mmlu_sampled_80pct.sh"


def launch(tmp_path, *arguments, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "RUN_DIR", "OUTPUT_ROOT", "PREPARE_ONLY"):
        env.pop(key, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, text=True, capture_output=True)


def test_default_runs_only_global_then_uniform80_with_batch4(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    assert [c[c.index("--pruning-scope") + 1] for c in commands] == ["global", "uniform"]
    for command, scope in zip(commands, ("global", "uniform")):
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.evaluate_sampled_mmlu_batched"]
        assert command[command.index("--output-dir") + 1].endswith(f"/{scope}/prune_80pct")
        assert command[command.index("--pruning-percent") + 1] == "80"
        assert command[command.index("--eval-batch-size") + 1] == "4"
        assert command[command.index("--limit") + 1] == "1531"
        assert command[command.index("--seed") + 1] == "0"
        assert "--resume" in command and "--dry-run" in command
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_individual_scope_is_resumable(tmp_path, scope):
    result = launch(tmp_path, scope)
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip())
    assert command[command.index("--pruning-scope") + 1] == scope
    assert "--resume" in command


def test_prepare_keeps_literal_paths_and_separate_scope_logs(tmp_path):
    source = str(tmp_path / "source $literal `name`")
    output = str(tmp_path / "output $literal `name`")
    result = launch(tmp_path, dry=False, PREPARE_ONLY="1", RUN_DIR=source, OUTPUT_ROOT=output)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert len(commands) == 2
    for args, scope in zip(commands, ("global", "uniform")):
        assert args[args.index("--run-dir") + 1] == source
        assert args[args.index("--output-dir") + 1] == f"{output}/{scope}/prune_80pct"
        assert "--prepare-only" in args and "--dry-run" not in args
        logs = list((tmp_path / f"logs/{scope}").glob("*.log"))
        assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


def test_gpu_execution_requires_explicit_devices(tmp_path):
    result = launch(tmp_path, dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


@pytest.mark.parametrize("args", [("both",), ("unknown",), ("global", "50"), ("uniform", "60"), ("global", "global")])
def test_invalid_or_duplicate_scopes_cannot_dispatch(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="4,5")
    assert result.returncode == 2 and "SHIM " not in result.stdout


def test_failed_global_prevents_starting_uniform(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="4,5", PYTHON_BIN=str(failure))
    assert result.returncode == 7 and "exit_status=7" in result.stdout
    assert (tmp_path / "logs/global").is_dir()
    assert not (tmp_path / "logs/uniform").exists()
