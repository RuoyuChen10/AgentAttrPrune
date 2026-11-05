"""The seed0 supplement must not replay or replace the original sweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "run_gsm8k_seed0_90pct.sh"


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


def test_exact_original_protocol_and_only_90pct_are_dispatched(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 3
    prepare, evaluation, verify = commands
    assert prepare[1:] == ["-", str(tmp_path / "original run"), "prepare"]
    assert verify[1:] == ["-", str(tmp_path / "original run"), "verify"]
    assert evaluation[1:4] == ["-u", "-m", "experiments_attrprune.evaluate_mmlu"]
    for flag, value in (("--benchmark", "gsm8k"), ("--mode", "token_ratio"),
                        ("--split", "test"), ("--start-index", "0"),
                        ("--limit", "1319"), ("--eval-batch-size", "4"),
                        ("--seed", "0"), ("--evaluation-name", "static_token_ratio_conditional_mean_global_prune_90pct")):
        assert evaluation[evaluation.index(flag) + 1] == value
    assert evaluation[evaluation.index("--policy-json") + 1].endswith(
        "/attribution/static_pruning/token_ratio_conditional_mean_global/policies/prune_90pct.json")
    assert {"--resume", "--save-traces", "--no-download"}.issubset(evaluation)
    assert "resampled" not in " ".join(evaluation) and "sampled" not in " ".join(evaluation)
    assert not (tmp_path / "logs").exists()


def test_cpu_prepare_only_and_literal_paths(tmp_path):
    source = str(tmp_path / "source $literal `name`")
    result = launch(tmp_path, dry=False, PREPARE_ONLY="1", RUN_DIR=source)
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert commands == [["-", source, "prepare"]]
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1 and "exit_status=0" in logs[0].read_text()


def test_gpu_required_before_any_writes(tmp_path):
    result = launch(tmp_path, dry=False)
    assert result.returncode == 2 and "CUDA_VISIBLE_DEVICES" in result.stderr
    assert "SHIM " not in result.stdout and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("args", [("50",), ("90",), ("1",), ("prepare",)])
def test_cannot_accidentally_run_a_different_rate_or_seed(tmp_path, args):
    result = launch(tmp_path, *args, dry=False, CUDA_VISIBLE_DEVICES="4,5")
    assert result.returncode == 2 and "SHIM " not in result.stdout


def test_failed_source_validation_prevents_model_dispatch(tmp_path):
    failure = tmp_path / "failed preparation"
    failure.write_text("#!/bin/sh\nexit 7\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="4,5", PYTHON_BIN=str(failure))
    assert result.returncode == 7 and "exit_status=7" in result.stdout


def test_evaluation_failure_does_not_claim_final_verification(tmp_path):
    failure = tmp_path / "failed evaluation"
    failure.write_text(f"#!{sys.executable}\nimport json,sys\n"
                      "print('SHIM '+json.dumps(sys.argv[1:]))\n"
                      "sys.exit(9 if 'experiments_attrprune.evaluate_mmlu' in sys.argv else 0)\n")
    failure.chmod(0o755)
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="4,5", PYTHON_BIN=str(failure))
    assert result.returncode == 9 and "exit_status=9" in result.stdout
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert len(commands) == 2 and commands[0][-1] == "prepare"


def test_success_checks_preservation_after_eval(tmp_path):
    result = launch(tmp_path, dry=False, CUDA_VISIBLE_DEVICES="4,5")
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line[5:]) for line in result.stdout.splitlines() if line.startswith("SHIM ")]
    assert len(commands) == 3 and commands[-1][-1] == "verify"
