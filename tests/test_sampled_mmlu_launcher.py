from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_mmlu_random_order_sampling.sh"


def launch(tmp_path, *args, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "RUN_DIR", "OUTPUT_ROOT", "PREPARE_ONLY", "VAL_LIMIT", "SEED"):
        env.pop(key, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env,
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("scope", ["global", "uniform"])
def test_dry_run_dispatches_both_independent_rates_with_requested_semantics(tmp_path, scope):
    result = launch(tmp_path, scope)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    assert [c[c.index("--pruning-percent") + 1] for c in commands] == ["50", "60"]
    for command in commands:
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.evaluate_sampled_mmlu"]
        assert command[command.index("--round-policy") + 1] == "per_round"
        assert command[command.index("--pruning-scope") + 1] == scope
        assert command[command.index("--limit") + 1] == "1531"
        assert "--resume" in command and "--dry-run" in command
    assert not (tmp_path / "logs").exists()


def test_gpu_stage_never_dispatches_without_device_selection(tmp_path):
    result = launch(tmp_path, "global", dry=False)
    assert result.returncode == 2
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


def test_cpu_prepare_and_literal_output_paths_are_supported(tmp_path):
    output = str(tmp_path / "literal $value root")
    result = launch(tmp_path, "uniform", "60", dry=False, PREPARE_ONLY="1", OUTPUT_ROOT=output)
    assert result.returncode == 0, result.stderr
    args = json.loads(next(line[5:] for line in result.stdout.splitlines() if line.startswith("SHIM ")))
    assert "--prepare-only" in args
    assert args[args.index("--output-dir") + 1] == output + "/uniform/prune_60pct"
    logs = list((tmp_path / "logs/uniform").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


@pytest.mark.parametrize("args", [(), ("unknown",), ("global", "50", "50"), ("uniform", "0"), ("global", "55")])
def test_invalid_scope_or_rate_cannot_start_inference(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="0,1")
    assert result.returncode == 2
    assert "SHIM " not in result.stdout


def test_first_failure_stops_the_sweep_and_is_logged(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, "global", dry=False, CUDA_VISIBLE_DEVICES="0,1", PYTHON_BIN=str(failure))
    assert result.returncode == 7
    assert "exit_status=7" in result.stdout
    assert "pruning=60%" not in result.stdout
