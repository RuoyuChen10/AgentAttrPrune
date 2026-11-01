"""The rerun launcher must isolate checkpoints and stop on execution failures."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "run_mmlu_reordered_skip_empty80.sh"


def invoke(tmp_path, *args, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint('SHIM '+json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    env = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "SOURCE_RUN", "DIRECT_RUN_DIR", "RECALIBRATE_RUN_DIR",
                 "VAL_LIMIT", "EVAL_BATCH_SIZE"):
        env.pop(name, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, text=True, capture_output=True)


def test_dry_queue_uses_only_new_evaluator_and_separate_processes(tmp_path):
    result = invoke(tmp_path, "both")
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert [c[c.index("--branch") + 1] for c in commands] == ["direct", "recalibrate"]
    for command in commands:
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.reordered_skip_empty_evaluation"]
        assert command[command.index("--stage") + 1] == "eval"
        assert command[command.index("--limit") + 1] == "1531"
        assert command[command.index("--eval-batch-size") + 1] == "4"
        assert "--resume" in command and "--dry-run" in command
    assert not (tmp_path / "logs").exists()


def test_prepare_without_gpu_preserves_branch_paths(tmp_path):
    direct = str(tmp_path / "direct $literal")
    recalibrate = str(tmp_path / "recalibrate `literal`")
    result = invoke(tmp_path, "both", "prepare", dry=False,
                    DIRECT_RUN_DIR=direct, RECALIBRATE_RUN_DIR=recalibrate)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert [c[c.index("--run-dir") + 1] for c in commands] == [direct, recalibrate]
    assert all(c[c.index("--stage") + 1] == "prepare" for c in commands)
    assert len(list((tmp_path / "logs").glob("*/*.log"))) == 2


def test_failed_first_evaluation_stops_queue(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = invoke(tmp_path, "both", dry=False, CUDA_VISIBLE_DEVICES="2,3", PYTHON_BIN=str(failure))
    assert result.returncode == 7
    assert "exit_status=7" in result.stdout
    assert not (tmp_path / "logs/recalibrate").exists()


@pytest.mark.parametrize("args", [("wrong",), ("direct", "all"), ("both", "eval", "60")])
def test_invalid_request_never_starts_python(tmp_path, args):
    result = invoke(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="2,3")
    assert result.returncode == 2 and "SHIM " not in result.stdout
    assert not (tmp_path / "logs").exists()


def test_eval_requires_explicit_idle_devices(tmp_path):
    result = invoke(tmp_path, "direct", dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout
