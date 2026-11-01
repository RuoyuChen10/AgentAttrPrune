"""The smaller experiment must not silently start generation or overlap the old job."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_mmlu_dev40_random_order_once.sh"


def launch(tmp_path, stage="calibrate", *, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    code = sys.argv[2]\n"
        "    sys.argv = ['-c', *sys.argv[3:]]\n"
        "    exec(code)\n"
        "else:\n"
        "    print('SHIM ' + json.dumps(sys.argv[1:]))\n"
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "RUN_DIR", "RUN_NAME", "SOURCE_RUN", "REFERENCE_RUN",
                "CONDA_BIN", "ORDER_SEED", "ATTR_BATCH_SIZE", "LOG_DIR"):
        env.pop(key, None)
    env.update(CONDA_ENV="", PYTHON_BIN=str(shim), DRY_RUN="1" if dry else "0",
               RESULTS_ROOT=str(tmp_path / "results"), SOURCE_RUN=str(tmp_path / "source"),
               LOG_DIR=str(tmp_path / "logs"))
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER), stage], env=env, text=True,
                          capture_output=True, check=False)


def test_calibrate_prepares_then_attributes_exactly_once_without_generation(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 4
    assert commands[0][2] == "experiments_attrprune.random_order_subset"
    assert [c[c.index("--stage") + 1] for c in commands[1:]] == ["attribute", "aggregate", "visualize"]
    for c in commands[1:]:
        assert c[c.index("--repeats") + 1] == "1"
        assert c[c.index("--run-dir") + 1].endswith("dev40_random_order1_correctness_aware")
        assert "--resume" in c
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("stage", ["calibrate", "attribute"])
def test_gpu_stage_requires_explicit_devices(tmp_path, stage):
    result = launch(tmp_path, stage, dry=False)
    assert result.returncode == 2
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout


def test_old_run_lock_blocks_gpu_dispatch_but_allows_cpu_prepare(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with (source / ".random_order.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="2,3")
        assert result.returncode == 2
        assert "Ctrl-C" in result.stderr
        assert "SHIM " not in result.stdout
        prepared = launch(tmp_path, "prepare", dry=False)
        assert prepared.returncode == 0, prepared.stderr
        assert "experiments_attrprune.random_order_subset" in prepared.stdout


def test_actual_dispatch_logs_stage_times_and_preserves_literal_paths(tmp_path):
    source = str(tmp_path / "source with literal $value")
    result = launch(tmp_path, "prepare", dry=False, SOURCE_RUN=source)
    assert result.returncode == 0, result.stderr
    args = json.loads(next(line[5:] for line in result.stdout.splitlines() if line.startswith("SHIM ")))
    assert args[args.index("--source-run") + 1] == source
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert "stage=prepare started" in content
    assert "stage=prepare completed elapsed_seconds=" in content
    assert "exit_status=0" in content


def test_failed_stage_logs_nonzero_status_and_does_not_continue(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="2,3", PYTHON_BIN=str(failure))
    assert result.returncode == 7
    assert "exit_status=7" in result.stdout
    assert "stage=attribute started" not in result.stdout


def test_conflicting_run_dir_is_rejected(tmp_path):
    result = launch(tmp_path, RUN_DIR="/unrelated/run")
    assert result.returncode == 2
    assert "Set RUN_NAME" in result.stderr


def test_frozen_source_settings_are_used_for_every_attribution_stage(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    reference = str(tmp_path / "reference with literal $value")
    (source / "run_config.json").write_text(json.dumps({"random_order_experiment": {
        "reference": {"run_dir": reference}, "order_seed": 17, "attribution_batch_size": 2,
    }}))
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    for command in [shlex.split(line) for line in result.stdout.splitlines()][1:]:
        assert command[command.index("--reference-run") + 1] == reference
        assert command[command.index("--order-seed") + 1] == "17"
        assert command[command.index("--attr-batch-size") + 1] == "2"
    assert not (tmp_path / "results").exists()
    conflict = launch(tmp_path, "prepare", dry=False, ORDER_SEED="3")
    assert conflict.returncode == 2
    assert "remove conflicting overrides" in conflict.stderr
    assert "SHIM " not in conflict.stdout
