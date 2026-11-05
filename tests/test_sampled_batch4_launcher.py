"""The continuation launcher never reruns completed 50% evaluations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_mmlu_sampled_batch4_resume.sh"


def launch(tmp_path, *arguments, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "SOURCE_EVALUATION", "OUTPUT_DIR", "PREPARE_ONLY"):
        env.pop(key, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, text=True, capture_output=True)


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_one_command_preserves_source_and_uses_new_batch4_output(tmp_path, scope):
    result = launch(tmp_path, scope)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    command = shlex.split(lines[0])
    assert command[1:4] == ["-u", "-m", "experiments_attrprune.continue_sampled_mmlu"]
    assert command[command.index("--source-evaluation") + 1].endswith(f"/{scope}/prune_60pct")
    assert command[command.index("--output-dir") + 1].endswith(f"/{scope}/prune_60pct_batch4_continuation")
    assert command[command.index("--eval-batch-size") + 1] == "4"
    assert "--resume" in command and "--dry-run" in command
    assert "50pct" not in result.stdout
    assert not (tmp_path / "logs").exists()


def test_prepare_uses_literal_source_and_separate_logs_without_gpu(tmp_path):
    source = str(tmp_path / "source $literal `name`" / "prune_60pct")
    output = str(tmp_path / "output $literal `name`")
    result = launch(tmp_path, "global", dry=False, PREPARE_ONLY="1",
                    SOURCE_EVALUATION=source, OUTPUT_DIR=output)
    assert result.returncode == 0, result.stderr
    args = json.loads(next(line[5:] for line in result.stdout.splitlines() if line.startswith("SHIM ")))
    assert args[args.index("--source-evaluation") + 1] == source
    assert args[args.index("--output-dir") + 1] == output
    assert "--prepare-only" in args and "--dry-run" not in args
    logs = list((tmp_path / "logs/global").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


def test_gpu_execution_requires_explicit_devices(tmp_path):
    result = launch(tmp_path, "uniform", dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


@pytest.mark.parametrize("args", [(), ("both",), ("unknown",), ("global", "50"), ("uniform", "60")])
def test_invalid_scope_or_rates_cannot_dispatch(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="0,1")
    assert result.returncode == 2 and "SHIM " not in result.stdout


def test_failed_migration_stops_and_reports_exit_status(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, "global", dry=False, CUDA_VISIBLE_DEVICES="0,1", PYTHON_BIN=str(failure))
    assert result.returncode == 7 and "exit_status=7" in result.stdout
