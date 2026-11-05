"""The source-order baseline cohort must use an isolated, resumable sweep."""

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_gsm8k_baseline_aligned_train40.sh"


def invoke(tmp_path, *arguments, dry=True, **overrides):
    shim = tmp_path / "python shim"
    shim.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "print('SHIM '+json.dumps({'args':sys.argv[1:],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),"
        "'thinking':os.environ.get('HF_CHAT_ENABLE_THINKING'),'scores':os.environ.get('HF_CHAT_CAPTURE_SCORES')}))\n"
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    for key in ("CUDA_VISIBLE_DEVICES", "STAGE", "REFERENCE_RUN", "BASELINE_DATA", "COHORT_AUDIT_DIR", "RUN_DIR"):
        env.pop(key, None)
    env.update(PYTHON_BIN=str(shim), LOG_ROOT=str(tmp_path / "logs"), DRY_RUN="1" if dry else "0")
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=env, text=True, capture_output=True)


def test_dry_run_dispatches_exact_stages_and_80_70_60_without_writes(tmp_path):
    result = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert [c[c.index("--stage") + 1] for c in commands] == [
        "prepare", "collect", "attribute", "aggregate", "policies", "eval", "eval", "eval", "summarize"
    ]
    assert [c[c.index("--pruning-percent") + 1] for c in commands if "--pruning-percent" in c] == ["80", "70", "60"]
    for command in commands:
        assert command[1:4] == ["-u", "-m", "experiments_attrprune.baseline_aligned_gsm8k"]
        assert command[command.index("--run-dir") + 1].endswith("/train40_baseline_aligned_correctness_aware")
        assert command[command.index("--reference-run") + 1].endswith("/train40_correctness_aware")
        assert command[command.index("--eval-batch-size") + 1] == "4"
        assert command[command.index("--limit") + 1] == "1319"
        assert "--resume" in command and "--dry-run" in command
        assert "--calibration-seed" not in command and "--seed" not in command
    assert not (tmp_path / "logs").exists()


def test_cpu_prepare_forwards_literal_cohort_paths_and_default_devices(tmp_path):
    paths = {"RUN_DIR": str(tmp_path / "new run $literal `name`"),
             "REFERENCE_RUN": str(tmp_path / "reference run"),
             "BASELINE_DATA": str(tmp_path / "baseline train.jsonl"),
             "COHORT_AUDIT_DIR": str(tmp_path / "frozen cohort audit")}
    result = invoke(tmp_path, "prepare", dry=False, **paths)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert len(calls) == 1
    assert calls[0]["gpu"] == "4,5"
    assert calls[0]["thinking"] == "0" and calls[0]["scores"] == "0"
    args = calls[0]["args"]
    for option, key in (("--run-dir", "RUN_DIR"), ("--reference-run", "REFERENCE_RUN"),
                        ("--baseline-data", "BASELINE_DATA"), ("--cohort-audit-dir", "COHORT_AUDIT_DIR")):
        assert args[args.index(option) + 1] == paths[key]
    assert args[args.index("--stage") + 1] == "prepare"
    assert "--dry-run" not in args
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


def test_eval_only_keeps_requested_order_and_device_override(tmp_path):
    result = invoke(tmp_path, dry=False, STAGE="eval", CUDA_VISIBLE_DEVICES="2,3")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert [c["args"][c["args"].index("--pruning-percent") + 1] for c in calls] == ["80", "70", "60"]
    assert all(c["gpu"] == "2,3" for c in calls)


def test_failed_stage_stops_before_collection_or_evaluation(tmp_path):
    failure = tmp_path / "failure"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = invoke(tmp_path, dry=False, PYTHON_BIN=str(failure))
    assert result.returncode == 7
    assert "exit_status=7" in result.stdout
    assert "stage=collect" not in result.stdout and "stage=eval" not in result.stdout


def test_launcher_lock_rejects_concurrent_dispatch(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    with (logs / ".launcher.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(tmp_path, "prepare", dry=False)
    assert result.returncode == 2
    assert "already active" in result.stderr
    assert "SHIM " not in result.stdout


@pytest.mark.parametrize("arguments", [("1",), ("8",), ("80",), ("prepare", "eval")])
def test_no_implicit_seed_or_pruning_rate_arguments(tmp_path, arguments):
    result = invoke(tmp_path, *arguments, dry=False)
    assert result.returncode == 2
    assert "SHIM " not in result.stdout
    assert not (tmp_path / "logs").exists()
